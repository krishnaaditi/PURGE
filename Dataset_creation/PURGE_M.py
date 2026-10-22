#!/usr/bin/env python3
"""
PURGE-M class-by-class construction with SpurLens-style cue ranking.

For each COCO target class, sequentially:

1) Obtain exactly N candidate contextual cues (default: 9).
   - If --cue_json is provided, use cues from that JSON.
   - Otherwise generate/refill cues automatically with OpenAI.
   - Cues are cached in output_root/cue_candidates.json.

2) Scan the requested COCO image pool with OWLv2 for:
       target object + all N candidate cues
   Scores are cached in SQLite, so later steps reuse them.

3) SpurLens-style ranking:
   For each candidate cue:
       PA:
         object-present images -> rank by cue score
         compare TOP-K vs BOTTOM-K model YES rate
         PA_gap = PA_top - PA_bottom

       HR:
         object-absent images -> rank by cue score
         compare TOP-K vs BOTTOM-K model YES rate
         HR_gap = HR_top - HR_bottom

   By default PURGE-M chooses the cue with the largest HR gap
   (--select_by hr), because NY = object absent + cue present is the
   hallucination/shortcut partition.

4) Partition ALL scanned images using the selected target/cue pair:

       YY = target YES + cue YES
       YN = target YES + cue NO   -> retain
       NY = target NO  + cue YES  -> forget
       NN = target NO  + cue NO   -> evaluation

   Presence rule used for CLEAN final groups:
   - target object: COCO ground-truth annotation (target is always COCO80)
   - selected cue if it is a COCO80 class: COCO ground-truth annotation
   - selected cue if it is not a COCO80 class: OWLv2 score threshold

   OWLv2 is still used for SpurLens-style cue scoring/ranking.
   This prevents detector false positives (for example skateboard wheels
   being mistaken for a bicycle) from contaminating YY/NY.

5) Finish this target class completely, save results, then move to the
   next COCO class.

With NO --class_names argument, all 80 COCO classes are processed
automatically in standard COCO order.

The same physical image can appear under several target-class folders,
because grouping is target-relative.

Output:
PURGE_M/
├── yy/<class>/
├── yn/<class>/
├── ny/<class>/
├── nn/<class>/
├── manifests/<class>.jsonl
├── rankings/<class>.json
├── cue_candidates.json
├── selected_cues.json
├── summary.json
├── completed_classes.json
├── detector_cache.sqlite
└── vlm_cache.sqlite
"""

import argparse
import json
import os
import random
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    Owlv2ForObjectDetection,
    Owlv2Processor,
)


COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# ============================================================
# Basic utilities
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return name.replace(" ", "_")


def read_json(path: Path, default):
    if not path.is_file():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def append_jsonl(handle, obj) -> None:
    handle.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def normalize_cue(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^[\-\*\d\.\)\s]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'")


def target_overlap(cue: str, target: str) -> bool:
    cue_n = normalize_cue(cue)
    target_n = normalize_cue(target)

    if not cue_n or not target_n:
        return True

    if cue_n == target_n:
        return True

    # Reject obvious target phrase inclusion.
    if target_n in cue_n or cue_n in target_n:
        return True

    cue_tokens = set(cue_n.split())
    target_tokens = set(target_n.split())

    # Do not allow a cue like "person on street" for target "person".
    if cue_tokens & target_tokens:
        return True

    return False


def yes_rate(values: List[str]) -> float:
    usable = [x for x in values if x in ("yes", "no")]
    if not usable:
        return 0.0
    return sum(x == "yes" for x in usable) / len(usable)


def combo_name(target_present: bool, cue_present: bool) -> str:
    if target_present and cue_present:
        return "yy"
    if target_present and not cue_present:
        return "yn"
    if not target_present and cue_present:
        return "ny"
    return "nn"


def materialize(src: Path, dst: Path, mode: str) -> None:
    if mode == "none":
        return

    ensure_dir(dst.parent)

    if dst.exists() or dst.is_symlink():
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            # Cross-filesystem fallback.
            shutil.copy2(src, dst)
            return

    if mode == "symlink":
        os.symlink(src.resolve(), dst)
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    raise ValueError(mode)


# ============================================================
# Persistent detector score cache
# ============================================================

class DetectorCache:
    def __init__(self, db_path: Path):
        ensure_dir(db_path.parent)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scores (
                image_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY(image_id, query)
            )
            """
        )
        self.conn.commit()

    def get(self, image_id: int, query: str) -> Optional[float]:
        row = self.conn.execute(
            "SELECT score FROM scores WHERE image_id=? AND query=?",
            (int(image_id), query),
        ).fetchone()
        return None if row is None else float(row[0])

    def put_many(self, image_id: int, score_map: Dict[str, float]) -> None:
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO scores(image_id, query, score)
            VALUES (?, ?, ?)
            """,
            [
                (int(image_id), query, float(score))
                for query, score in score_map.items()
            ],
        )
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()


# ============================================================
# Persistent LVLM yes/no cache
# ============================================================

class VLMCache:
    def __init__(self, db_path: Path):
        ensure_dir(db_path.parent)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                image_id INTEGER NOT NULL,
                target TEXT NOT NULL,
                prediction TEXT NOT NULL,
                raw_answer TEXT,
                PRIMARY KEY(image_id, target)
            )
            """
        )
        self.conn.commit()

    def get(self, image_id: int, target: str):
        row = self.conn.execute(
            """
            SELECT prediction, raw_answer
            FROM predictions
            WHERE image_id=? AND target=?
            """,
            (int(image_id), target),
        ).fetchone()
        if row is None:
            return None
        return row[0], row[1]

    def put(self, image_id: int, target: str, prediction: str, raw: str):
        self.conn.execute(
            """
            INSERT OR REPLACE INTO predictions(
                image_id, target, prediction, raw_answer
            ) VALUES (?, ?, ?, ?)
            """,
            (int(image_id), target, prediction, raw),
        )
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()


# ============================================================
# OpenAI candidate cue generation
# ============================================================

def parse_cue_response(text: str) -> List[str]:
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    except Exception:
        pass

    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return quoted

    return [x.strip() for x in text.splitlines() if x.strip()]


def generate_missing_cues(
    *,
    target: str,
    accepted: List[str],
    num_cues: int,
    model_name: str,
    max_rounds: int = 8,
) -> List[str]:
    """
    Automatically refill until exactly num_cues valid cues are available.
    """
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "OpenAI package is required for automatic cue generation. "
            "Install with: python -m pip install openai"
        ) from exc

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Set it once with: export OPENAI_API_KEY='YOUR_KEY' "
            "or pass --cue_json."
        )

    client = OpenAI()
    clean = []

    for cue in accepted:
        cue = normalize_cue(cue)
        if not cue:
            continue
        if target_overlap(cue, target):
            continue
        if cue not in clean:
            clean.append(cue)

    clean = clean[:num_cues]

    for round_idx in range(1, max_rounds + 1):
        if len(clean) >= num_cues:
            break

        missing = num_cues - len(clean)
        ask_n = max(num_cues, missing + 5)

        prompt = f"""
We are discovering possible spurious/contextual visual cues for a multimodal model.

Target object: {target}

Return {ask_n} NEW candidate contextual cues.

We ultimately need exactly {num_cues} valid cues.
Already accepted cues: {clean if clean else "none"}

Rules:
- return contextual/background/co-occurring VISUAL entities or scene cues
- do not return the target itself
- do not include the target word in the cue
- do not return a physical part of the target
- do not duplicate any accepted cue
- prefer concrete visual nouns / short noun phrases
- prefer cues an open-vocabulary object detector such as OWLv2 can score
- keep cues diverse
- output JSON only, exactly as a JSON list of strings
""".strip()

        response = client.responses.create(
            model=model_name,
            input=prompt,
        )

        candidates = parse_cue_response(response.output_text)

        before = len(clean)

        for item in candidates:
            cue = normalize_cue(item)

            if not cue:
                continue
            if target_overlap(cue, target):
                continue
            if cue in clean:
                continue

            clean.append(cue)

            if len(clean) == num_cues:
                break

        print(
            f"[CUE REFILL][{target}] round={round_idx} "
            f"valid={len(clean)}/{num_cues}",
            flush=True,
        )

        if len(clean) == before:
            continue

    if len(clean) != num_cues:
        raise RuntimeError(
            f"Could not obtain exactly {num_cues} valid cues for '{target}'. "
            f"Got: {clean}"
        )

    return clean


def get_candidate_cues(
    *,
    args,
    target: str,
    cue_cache_path: Path,
    explicit_cues: Dict[str, List[str]],
) -> List[str]:

    cached = read_json(cue_cache_path, {})

    # Explicit JSON takes priority.
    source = explicit_cues.get(target, cached.get(target, []))

    clean = []
    for item in source:
        cue = normalize_cue(str(item))
        if not cue:
            continue
        if target_overlap(cue, target):
            continue
        if cue not in clean:
            clean.append(cue)

    clean = clean[:args.num_cues]

    if len(clean) < args.num_cues:
        clean = generate_missing_cues(
            target=target,
            accepted=clean,
            num_cues=args.num_cues,
            model_name=args.cue_model,
        )

    cached[target] = clean
    write_json(cue_cache_path, cached)

    print(
        f"[CUES][{target}] " + ", ".join(clean),
        flush=True,
    )

    return clean


# ============================================================
# OWLv2
# ============================================================

class OwlScorer:
    def __init__(self, model_id: str, device: str, dtype_name: str):
        self.device = device

        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype_name]

        print(f"Loading OWLv2: {model_id}", flush=True)

        self.processor = Owlv2Processor.from_pretrained(model_id)

        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device).eval()

    @torch.inference_mode()
    def score_queries(
        self,
        image: Image.Image,
        queries: List[str],
    ) -> Dict[str, float]:
        """
        Return maximum OWLv2 confidence for every query.
        """
        prompts = [f"a photo of {q}" for q in queries]

        inputs = self.processor(
            text=[prompts],
            images=image,
            return_tensors="pt",
        )

        inputs = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }

        outputs = self.model(**inputs)

        target_sizes = torch.tensor(
            [[image.height, image.width]],
            device=self.device,
        )

        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=0.0,
        )[0]

        out = {q: 0.0 for q in queries}

        labels = results["labels"].detach().cpu().tolist()
        scores = results["scores"].detach().float().cpu().tolist()

        for label, score in zip(labels, scores):
            label = int(label)
            if 0 <= label < len(queries):
                q = queries[label]
                out[q] = max(out[q], float(score))

        return out


def get_detector_scores(
    *,
    image_id: int,
    image_path: Path,
    queries: List[str],
    detector: OwlScorer,
    cache: DetectorCache,
) -> Dict[str, float]:

    result = {}
    missing = []

    for query in queries:
        score = cache.get(image_id, query)
        if score is None:
            missing.append(query)
        else:
            result[query] = score

    if missing:
        with load_rgb(image_path) as image:
            new_scores = detector.score_queries(image, missing)

        cache.put_many(image_id, new_scores)
        result.update(new_scores)

    return {q: float(result.get(q, 0.0)) for q in queries}


# ============================================================
# LLaVA binary scorer
# ============================================================

def parse_yes_no(text: str) -> str:
    m = re.search(r"\b(yes|no)\b", text.strip(), flags=re.IGNORECASE)
    return "unknown" if m is None else m.group(1).lower()


class LlavaScorer:
    def __init__(self, model_path: str, device: str, dtype_name: str):
        self.device = device

        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }[dtype_name]

        print(f"Loading LLaVA: {model_path}", flush=True)

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()

    @torch.inference_mode()
    def predict(self, image: Image.Image, target: str) -> Tuple[str, str]:
        question = f"Is there a {target} in the image? Answer only yes or no."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]

        try:
            prompt = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = f"<image>\n{question}"

        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
        )

        inputs = {
            k: (v.to(self.device) if torch.is_tensor(v) else v)
            for k, v in inputs.items()
        }

        generated = self.model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
        )

        prompt_len = inputs["input_ids"].shape[-1]
        new_ids = generated[:, prompt_len:]

        raw = self.processor.batch_decode(
            new_ids,
            skip_special_tokens=True,
        )[0].strip()

        return parse_yes_no(raw), raw


def get_vlm_prediction(
    *,
    image_id: int,
    image_path: Path,
    target: str,
    scorer: LlavaScorer,
    cache: VLMCache,
) -> Tuple[str, str]:

    cached = cache.get(image_id, target)
    if cached is not None:
        return cached

    with load_rgb(image_path) as image:
        pred, raw = scorer.predict(image, target)

    cache.put(image_id, target, pred, raw)
    return pred, raw


# ============================================================
# COCO helpers
# ============================================================

def image_path_from_id(coco: COCO, image_root: Path, image_id: int) -> Path:
    info = coco.loadImgs([image_id])[0]
    return image_root / info["file_name"]


# ============================================================
# Step 1 for a class:
# scan ALL requested images for target + 9 cues
# ============================================================

def scan_class_scores(
    *,
    args,
    coco: COCO,
    image_root: Path,
    image_ids: List[int],
    target: str,
    cues: List[str],
    detector: OwlScorer,
    detector_cache: DetectorCache,
) -> List[int]:

    queries = [target] + cues
    usable_ids = []

    for image_id in tqdm(
        image_ids,
        desc=f"{target}: OWLv2 target + {len(cues)} cues",
    ):
        image_path = image_path_from_id(
            coco,
            image_root,
            image_id,
        )

        if not image_path.is_file():
            continue

        get_detector_scores(
            image_id=image_id,
            image_path=image_path,
            queries=queries,
            detector=detector,
            cache=detector_cache,
        )

        usable_ids.append(image_id)

    return usable_ids


# ============================================================
# SpurLens-style PA / HR ranking
# ============================================================

def top_bottom_ids(
    ids: List[int],
    cue_scores: Dict[int, float],
    k_requested: int,
) -> Tuple[List[int], List[int]]:
    """
    Use non-overlapping top-K and bottom-K whenever possible.
    """
    ranked = sorted(
        ids,
        key=lambda image_id: cue_scores[image_id],
    )

    if len(ranked) < 2:
        return [], []

    k = min(k_requested, len(ranked) // 2)

    if k <= 0:
        return [], []

    bottom = ranked[:k]
    top = ranked[-k:]

    return top, bottom


def evaluate_ids_with_vlm(
    *,
    coco: COCO,
    image_root: Path,
    image_ids: List[int],
    target: str,
    vlm: LlavaScorer,
    vlm_cache: VLMCache,
) -> Tuple[float, int]:

    predictions = []

    for image_id in image_ids:
        path = image_path_from_id(
            coco,
            image_root,
            image_id,
        )

        if not path.is_file():
            continue

        pred, _ = get_vlm_prediction(
            image_id=image_id,
            image_path=path,
            target=target,
            scorer=vlm,
            cache=vlm_cache,
        )

        if pred in ("yes", "no"):
            predictions.append(pred)

    return yes_rate(predictions), len(predictions)


def rank_cues_spurlens(
    *,
    args,
    coco: COCO,
    image_root: Path,
    image_ids: List[int],
    target: str,
    target_positive_ids: set,
    cues: List[str],
    detector_cache: DetectorCache,
    vlm: LlavaScorer,
    vlm_cache: VLMCache,
) -> Dict:
    """
    SpurLens-style ranking.

    Target-object presence for PA/HR is taken from COCO ground truth.
    Candidate cue strength is the OWLv2 score.

    PA_gap:
        target present -> top-K cue-score model-YES rate
                        - bottom-K cue-score model-YES rate

    HR_gap:
        target absent -> top-K cue-score model-YES rate
                       - bottom-K cue-score model-YES rate
    """

    present_ids = [
        image_id
        for image_id in image_ids
        if image_id in target_positive_ids
    ]

    absent_ids = [
        image_id
        for image_id in image_ids
        if image_id not in target_positive_ids
    ]

    print(
        f"[{target}] COCO split: "
        f"object-present={len(present_ids)} "
        f"object-absent={len(absent_ids)}",
        flush=True,
    )

    metrics = {}

    for cue in cues:
        cue_scores = {
            image_id: float(detector_cache.get(image_id, cue) or 0.0)
            for image_id in image_ids
        }

        # PA: target object present; compare strong vs weak cue evidence.
        pa_top_ids, pa_bottom_ids = top_bottom_ids(
            present_ids,
            cue_scores,
            args.spur_k,
        )

        pa_top, pa_top_n = evaluate_ids_with_vlm(
            coco=coco,
            image_root=image_root,
            image_ids=pa_top_ids,
            target=target,
            vlm=vlm,
            vlm_cache=vlm_cache,
        )

        pa_bottom, pa_bottom_n = evaluate_ids_with_vlm(
            coco=coco,
            image_root=image_root,
            image_ids=pa_bottom_ids,
            target=target,
            vlm=vlm,
            vlm_cache=vlm_cache,
        )

        pa_gap = pa_top - pa_bottom

        # HR: target object absent; compare strong vs weak cue evidence.
        hr_top_ids, hr_bottom_ids = top_bottom_ids(
            absent_ids,
            cue_scores,
            args.spur_k,
        )

        hr_top, hr_top_n = evaluate_ids_with_vlm(
            coco=coco,
            image_root=image_root,
            image_ids=hr_top_ids,
            target=target,
            vlm=vlm,
            vlm_cache=vlm_cache,
        )

        hr_bottom, hr_bottom_n = evaluate_ids_with_vlm(
            coco=coco,
            image_root=image_root,
            image_ids=hr_bottom_ids,
            target=target,
            vlm=vlm,
            vlm_cache=vlm_cache,
        )

        hr_gap = hr_top - hr_bottom

        metrics[cue] = {
            "pa_top_yes_rate": float(pa_top),
            "pa_bottom_yes_rate": float(pa_bottom),
            "pa_gap": float(pa_gap),
            "pa_top_n": int(pa_top_n),
            "pa_bottom_n": int(pa_bottom_n),
            "hr_top_yes_rate": float(hr_top),
            "hr_bottom_yes_rate": float(hr_bottom),
            "hr_gap": float(hr_gap),
            "hr_top_n": int(hr_top_n),
            "hr_bottom_n": int(hr_bottom_n),
            "pa_top_cue_score_mean": (
                float(np.mean([cue_scores[x] for x in pa_top_ids]))
                if pa_top_ids else 0.0
            ),
            "pa_bottom_cue_score_mean": (
                float(np.mean([cue_scores[x] for x in pa_bottom_ids]))
                if pa_bottom_ids else 0.0
            ),
            "hr_top_cue_score_mean": (
                float(np.mean([cue_scores[x] for x in hr_top_ids]))
                if hr_top_ids else 0.0
            ),
            "hr_bottom_cue_score_mean": (
                float(np.mean([cue_scores[x] for x in hr_bottom_ids]))
                if hr_bottom_ids else 0.0
            ),
        }

    pa_ranking = sorted(
        cues,
        key=lambda cue: (
            metrics[cue]["pa_gap"],
            metrics[cue]["pa_top_cue_score_mean"]
            - metrics[cue]["pa_bottom_cue_score_mean"],
        ),
        reverse=True,
    )

    hr_ranking = sorted(
        cues,
        key=lambda cue: (
            metrics[cue]["hr_gap"],
            metrics[cue]["hr_top_cue_score_mean"]
            - metrics[cue]["hr_bottom_cue_score_mean"],
        ),
        reverse=True,
    )

    print()
    print(f"[SPURLENS PA RANKING][{target}]")
    for rank, cue in enumerate(pa_ranking, 1):
        m = metrics[cue]
        print(
            f"{rank:02d}. {cue:28s} "
            f"PA_top={m['pa_top_yes_rate']:.4f} "
            f"PA_bottom={m['pa_bottom_yes_rate']:.4f} "
            f"PA_gap={m['pa_gap']:.4f}"
        )

    print()
    print(f"[SPURLENS HR RANKING][{target}]")
    for rank, cue in enumerate(hr_ranking, 1):
        m = metrics[cue]
        print(
            f"{rank:02d}. {cue:28s} "
            f"HR_top={m['hr_top_yes_rate']:.4f} "
            f"HR_bottom={m['hr_bottom_yes_rate']:.4f} "
            f"HR_gap={m['hr_gap']:.4f}"
        )

    if args.select_by == "pa":
        selected = pa_ranking[0]
    elif args.select_by == "hr":
        selected = hr_ranking[0]
    elif args.select_by == "max":
        selected = max(
            cues,
            key=lambda cue: max(
                metrics[cue]["pa_gap"],
                metrics[cue]["hr_gap"],
            ),
        )
    else:
        raise ValueError(args.select_by)

    print()
    print(
        f"[SELECTED][{target}] cue='{selected}' "
        f"select_by={args.select_by} "
        f"PA_gap={metrics[selected]['pa_gap']:.4f} "
        f"HR_gap={metrics[selected]['hr_gap']:.4f}",
        flush=True,
    )

    return {
        "target_class": target,
        "candidate_cues": cues,
        "spur_k": args.spur_k,
        "target_presence_source": "COCO_annotation",
        "select_by": args.select_by,
        "selected_cue": selected,
        "pa_best_cue": pa_ranking[0],
        "hr_best_cue": hr_ranking[0],
        "pa_ranking": pa_ranking,
        "hr_ranking": hr_ranking,
        "metrics": metrics,
        "object_present_n": len(present_ids),
        "object_absent_n": len(absent_ids),
    }


# ============================================================
# Final YY / YN / NY / NN partition for one target class
# ============================================================

def partition_one_class(
    *,
    args,
    coco: COCO,
    name_to_id: Dict[str, int],
    image_root: Path,
    output_root: Path,
    image_ids: List[int],
    target: str,
    target_positive_ids: set,
    selected_cue: str,
    detector_cache: DetectorCache,
) -> Dict:
    """
    Build semantic YY / YN / NY / NN groups for ONE target class.

    Example target=person, selected_cue=bicycle:
        YY/person = person YES + bicycle YES
        YN/person = person YES + bicycle NO
        NY/person = person NO  + bicycle YES
        NN/person = person NO  + bicycle NO

    IMPORTANT CLEAN-PRESENCE RULE:
      - target is a COCO class -> COCO annotation decides target presence.
      - if selected cue is also a COCO class -> COCO annotation decides cue.
      - otherwise OWLv2 threshold decides cue presence.

    Therefore a skateboard image cannot become bicycle=YES merely because
    OWLv2 gives bicycle a false-positive score.
    """

    class_safe = safe_name(target)

    for combo in ("yy", "yn", "ny", "nn"):
        ensure_dir(output_root / combo / class_safe)

    ensure_dir(output_root / "manifests")
    manifest_path = output_root / "manifests" / f"{class_safe}.jsonl"

    cue_is_coco = selected_cue in name_to_id

    if cue_is_coco:
        cue_positive_ids = set(
            coco.getImgIds(catIds=[name_to_id[selected_cue]])
        )
        cue_presence_source = "COCO_annotation"
    else:
        cue_positive_ids = None
        cue_presence_source = "OWLv2"

    counts = {
        "yy": 0,
        "yn": 0,
        "ny": 0,
        "nn": 0,
        "missing_image": 0,
    }

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        for image_id in tqdm(
            image_ids,
            desc=f"{target}: YY/YN/NY/NN using '{selected_cue}'",
        ):
            info = coco.loadImgs([image_id])[0]
            src = image_root / info["file_name"]

            if not src.is_file():
                counts["missing_image"] += 1
                continue

            # Target presence is clean COCO ground truth.
            target_present = image_id in target_positive_ids

            # Detector score is still recorded for auditing.
            target_detector_score = detector_cache.get(image_id, target)
            cue_detector_score = detector_cache.get(image_id, selected_cue)

            if cue_is_coco:
                # For bicycle/backpack/car/dog/etc., COCO GT is authoritative.
                cue_present = image_id in cue_positive_ids
            else:
                # For building/water/road/runway/etc., use OWLv2.
                cue_score = float(cue_detector_score or 0.0)
                cue_present = cue_score >= args.cue_detector_threshold

            combo = combo_name(target_present, cue_present)
            counts[combo] += 1

            destination = (
                output_root
                / combo
                / class_safe
                / f"{image_id:012d}_{Path(info['file_name']).name}"
            )

            do_materialize = True
            if (
                args.max_materialized_per_combo > 0
                and counts[combo] > args.max_materialized_per_combo
            ):
                do_materialize = False

            if do_materialize:
                materialize(
                    src,
                    destination,
                    args.link_mode,
                )

            append_jsonl(
                manifest,
                {
                    "image_id": int(image_id),
                    "file_name": info["file_name"],
                    "target_class": target,
                    "selected_cue": selected_cue,
                    "target_present": bool(target_present),
                    "target_presence_source": "COCO_annotation",
                    "target_detector_score": (
                        None if target_detector_score is None
                        else float(target_detector_score)
                    ),
                    "cue_present": bool(cue_present),
                    "cue_presence_source": cue_presence_source,
                    "cue_detector_score": (
                        None if cue_detector_score is None
                        else float(cue_detector_score)
                    ),
                    "cue_detector_threshold": (
                        None if cue_is_coco
                        else float(args.cue_detector_threshold)
                    ),
                    "combo": combo,
                    "training_role": (
                        "retain" if combo == "yn"
                        else "forget" if combo == "ny"
                        else "evaluation"
                    ),
                    "materialized": bool(
                        do_materialize and args.link_mode != "none"
                    ),
                },
            )

    counts["selected_cue"] = selected_cue
    counts["cue_is_coco_class"] = bool(cue_is_coco)
    counts["cue_presence_source"] = cue_presence_source
    return counts


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "PURGE-M automatic class-by-class SpurLens-style "
            "cue ranking + YY/YN/NY/NN construction."
        )
    )

    parser.add_argument("--coco_image_root", required=True)
    parser.add_argument("--coco_ann_json", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--vlm_model_path", required=True)

    parser.add_argument(
        "--class_names",
        default="",
        help=(
            'Optional comma-separated classes, e.g. "person,bicycle". '
            "Omit for ALL 80 classes automatically."
        ),
    )

    # Candidate cues
    parser.add_argument("--num_cues", type=int, default=9)
    parser.add_argument("--cue_json", default="")
    parser.add_argument("--cue_model", default="gpt-4.1-mini")

    # Detector
    parser.add_argument(
        "--detector_model",
        default="google/owlv2-base-patch16-ensemble",
    )
    parser.add_argument("--detector_device", default="cuda:0")
    parser.add_argument(
        "--detector_dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )

    # Backward-compatible argument. Final target presence uses COCO GT.
    parser.add_argument(
        "--object_detector_threshold",
        type=float,
        default=0.20,
        help=(
            "Deprecated for final grouping: target presence uses COCO GT. "
            "Kept only so older run commands do not fail."
        ),
    )

    parser.add_argument(
        "--cue_detector_threshold",
        type=float,
        default=0.10,
        help="OWLv2 threshold for selected-cue YES/NO.",
    )

    # SpurLens top/bottom K
    parser.add_argument(
        "--spur_k",
        type=int,
        default=100,
        help="Top-K / bottom-K for PA and HR calculation.",
    )

    parser.add_argument(
        "--select_by",
        choices=["hr", "pa", "max"],
        default="hr",
        help=(
            "Which SpurLens gap selects the final cue. "
            "Default hr for hallucination-focused PURGE-M."
        ),
    )

    # LLaVA
    parser.add_argument("--vlm_device", default="cuda:0")
    parser.add_argument(
        "--vlm_dtype",
        choices=["bf16", "fp16", "fp32"],
        default="bf16",
    )

    # Image pool
    parser.add_argument(
        "--max_images_per_class",
        type=int,
        default=0,
        help=(
            "0 = ALL COCO images for every target class. "
            "Positive values are useful only for debugging."
        ),
    )

    # Materialization
    parser.add_argument(
        "--max_materialized_per_combo",
        type=int,
        default=0,
        help=(
            "0 = materialize every image assigned to every combo. "
            "Manifest always contains all processed samples."
        ),
    )

    parser.add_argument(
        "--link_mode",
        choices=["hardlink", "symlink", "copy", "none"],
        default="hardlink",
    )

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_root and start completely fresh.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    image_root = Path(args.coco_image_root)
    ann_json = Path(args.coco_ann_json)
    output_root = Path(args.output_root)

    if not image_root.is_dir():
        raise FileNotFoundError(image_root)

    if not ann_json.is_file():
        raise FileNotFoundError(ann_json)

    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)

    ensure_dir(output_root)
    ensure_dir(output_root / "rankings")
    ensure_dir(output_root / "manifests")

    print("Loading COCO...", flush=True)
    coco = COCO(str(ann_json))

    categories = coco.loadCats(coco.getCatIds())
    name_to_id = {
        item["name"]: int(item["id"])
        for item in categories
    }

    all_image_ids = sorted(coco.getImgIds())

    # Classes
    if args.class_names:
        requested = [
            x.strip()
            for x in args.class_names.split(",")
            if x.strip()
        ]

        invalid = [
            x for x in requested
            if x not in COCO80_CLASSES
        ]

        if invalid:
            raise ValueError(
                f"Unknown COCO classes: {invalid}"
            )

        classes = [
            x for x in COCO80_CLASSES
            if x in requested
        ]
    else:
        classes = COCO80_CLASSES.copy()

    print(
        f"Target classes ({len(classes)}): {classes}",
        flush=True,
    )

    print("FINAL GROUP RULE:")
    print("  YY = target YES + selected cue YES")
    print("  YN = target YES + selected cue NO  -> RETAIN")
    print("  NY = target NO  + selected cue YES -> FORGET")
    print("  NN = target NO  + selected cue NO")
    print("  Target/COCO-cue presence = COCO annotation")
    print("  Non-COCO cue presence    = OWLv2")

    # Optional fixed cue JSON
    explicit_cues = {}
    if args.cue_json:
        cue_json_path = Path(args.cue_json)
        if not cue_json_path.is_file():
            raise FileNotFoundError(cue_json_path)
        with open(cue_json_path, "r", encoding="utf-8") as f:
            explicit_cues = json.load(f)

    cue_cache_path = output_root / "cue_candidates.json"
    selected_path = output_root / "selected_cues.json"
    summary_path = output_root / "summary.json"
    completed_path = output_root / "completed_classes.json"

    selected_cues = read_json(selected_path, {})
    summary = read_json(summary_path, {})
    completed = read_json(completed_path, [])

    detector_cache = DetectorCache(
        output_root / "detector_cache.sqlite"
    )

    vlm_cache = VLMCache(
        output_root / "vlm_cache.sqlite"
    )

    # Load models once.
    detector = OwlScorer(
        args.detector_model,
        args.detector_device,
        args.detector_dtype,
    )

    vlm = LlavaScorer(
        args.vlm_model_path,
        args.vlm_device,
        args.vlm_dtype,
    )

    for class_idx, target in enumerate(classes, 1):

        print()
        print("=" * 80)
        print(
            f"[CLASS {class_idx}/{len(classes)}] {target.upper()}",
            flush=True,
        )
        print("=" * 80)

        if target in completed:
            print(
                f"[SKIP][{target}] already completed. "
                "Use --overwrite for a completely fresh run.",
                flush=True,
            )
            continue

        # Image pool for this target.
        image_ids = all_image_ids.copy()

        if args.max_images_per_class > 0:
            rng = random.Random(
                args.seed + class_idx
            )
            rng.shuffle(image_ids)
            image_ids = image_ids[
                :args.max_images_per_class
            ]

        # Ground-truth target presence for this target class.
        target_positive_ids = set(
            coco.getImgIds(catIds=[name_to_id[target]])
        )

        # ----------------------------------------------------
        # A. 9 cues for THIS class only.
        # ----------------------------------------------------

        cues = get_candidate_cues(
            args=args,
            target=target,
            cue_cache_path=cue_cache_path,
            explicit_cues=explicit_cues,
        )

        # ----------------------------------------------------
        # B. Scan ALL images for target + all 9 cues.
        # ----------------------------------------------------

        usable_ids = scan_class_scores(
            args=args,
            coco=coco,
            image_root=image_root,
            image_ids=image_ids,
            target=target,
            cues=cues,
            detector=detector,
            detector_cache=detector_cache,
        )

        # ----------------------------------------------------
        # C. SpurLens PA + HR ranking.
        # ----------------------------------------------------

        ranking_path = (
            output_root
            / "rankings"
            / f"{safe_name(target)}.json"
        )

        if ranking_path.is_file():
            ranking = read_json(ranking_path, {})
            selected_cue = ranking["selected_cue"]

            print(
                f"[RESUME][{target}] using saved ranking; "
                f"selected cue='{selected_cue}'",
                flush=True,
            )
        else:
            ranking = rank_cues_spurlens(
                args=args,
                coco=coco,
                image_root=image_root,
                image_ids=usable_ids,
                target=target,
                target_positive_ids=target_positive_ids,
                cues=cues,
                detector_cache=detector_cache,
                vlm=vlm,
                vlm_cache=vlm_cache,
            )

            selected_cue = ranking["selected_cue"]
            write_json(ranking_path, ranking)

        selected_cues[target] = selected_cue
        write_json(selected_path, selected_cues)

        # ----------------------------------------------------
        # D. ALL images -> YY/YN/NY/NN for THIS class.
        # Target=COCO GT. COCO cue=COCO GT. Non-COCO cue=OWLv2.
        # ----------------------------------------------------

        counts = partition_one_class(
            args=args,
            coco=coco,
            name_to_id=name_to_id,
            image_root=image_root,
            output_root=output_root,
            image_ids=usable_ids,
            target=target,
            target_positive_ids=target_positive_ids,
            selected_cue=selected_cue,
            detector_cache=detector_cache,
        )

        summary[target] = {
            "selected_cue": selected_cue,
            "pa_best_cue": ranking["pa_best_cue"],
            "hr_best_cue": ranking["hr_best_cue"],
            "pa_gap_selected":
                ranking["metrics"][selected_cue]["pa_gap"],
            "hr_gap_selected":
                ranking["metrics"][selected_cue]["hr_gap"],
            **counts,
        }

        write_json(summary_path, summary)

        completed.append(target)
        write_json(completed_path, completed)

        print()
        print(
            f"[DONE][{target}] cue='{selected_cue}' | "
            f"YY={counts['yy']} "
            f"YN={counts['yn']} "
            f"NY={counts['ny']} "
            f"NN={counts['nn']}",
            flush=True,
        )

        print(
            f"[NEXT] moving automatically to the next class.",
            flush=True,
        )

    detector_cache.close()
    vlm_cache.close()

    print()
    print("=" * 80)
    print("PURGE-M ALL REQUESTED CLASSES COMPLETE")
    print("=" * 80)
    print(f"Output: {output_root}")
    print("YN = retain, NY = forget, YY/NN = evaluation")


if __name__ == "__main__":
    main()