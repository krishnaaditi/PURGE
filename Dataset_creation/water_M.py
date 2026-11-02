#!/usr/bin/env python3
"""Create Waterbirds PURGE-M partitions with 9 contextual cues per class.

Waterbirds target ground truth comes from metadata:
    y=0 -> landbird
    y=1 -> waterbird

OWLv2 scores the nine candidate context cues. LLaVA answers whether the
target class is present when ranking cues. Final cue presence is determined
by the OWLv2 score threshold.

Partitions for each target class:
    YY = target present + cue present
    YN = target present + cue absent  -> retain
    NY = target absent  + cue present -> forget
    NN = target absent  + cue absent  -> evaluation

The same source image can appear once under each target-class folder.
"""

import argparse
import csv
import json
import random
import re
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    LlavaNextForConditionalGeneration,
    Owlv2ForObjectDetection,
    Owlv2Processor,
)


CLASS_NAMES = {0: "landbird", 1: "waterbird"}

# Exactly nine contextual cues for each Waterbirds target class.
DEFAULT_CUES = {
    "landbird": [
        "forest",
        "trees",
        "tree branches",
        "green foliage",
        "leaves",
        "grass",
        "woodland",
        "tree trunks",
        "bushes",
    ],
    "waterbird": [
        "water",
        "lake",
        "river",
        "ocean",
        "shoreline",
        "wetland",
        "reeds",
        "water reflection",
        "muddy bank",
    ],
}


def parse_split(value):
    mapping = {
        "train": 0, "0": 0,
        "val": 1, "valid": 1, "validation": 1, "1": 1,
        "test": 2, "2": 2,
    }
    key = str(value).strip().lower()
    if key not in mapping:
        raise ValueError(f"Unknown split {value!r}; expected train/val/test or 0/1/2")
    return mapping[key]


def normalize_cue(text):
    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\n\r.,;:!?\"'")


def load_rows(metadata_csv, image_root, limit=0, seed=42):
    rows = []
    with Path(metadata_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"img_filename", "y", "place", "split"}
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise ValueError(
                f"Metadata is missing columns {sorted(missing)}; found {sorted(columns)}"
            )

        for index, raw in enumerate(reader):
            if parse_split(raw["split"]) != 0:
                continue

            y = int(raw["y"])
            place = int(raw["place"])
            if y not in CLASS_NAMES or place not in (0, 1):
                raise ValueError(f"Unexpected Waterbirds labels y={y}, place={place}")

            relative_name = raw["img_filename"]
            image_path = Path(image_root) / relative_name
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in metadata is missing: {image_path}")

            # WILDS metadata normally has img_id. Use row index if not present.
            image_id = int(raw.get("img_id", index))

            rows.append({
                "image_id": image_id,
                "filename": relative_name,
                "path": image_path,
                "y": y,
                "class_name": CLASS_NAMES[y],
                "place": place,
                "place_name": "water" if place == 1 else "land",
            })

    random.Random(seed).shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise ValueError("No Waterbirds training rows found (expected split=0/train)")
    return rows


def load_cues(cue_json):
    if not cue_json:
        return DEFAULT_CUES

    with Path(cue_json).open("r", encoding="utf-8") as handle:
        supplied = json.load(handle)

    cues_by_class = {}
    for target in CLASS_NAMES.values():
        if target not in supplied:
            raise ValueError(f"Cue JSON is missing key {target!r}")

        cues = []
        for cue in supplied[target]:
            cue = normalize_cue(cue)
            if cue and cue not in cues:
                cues.append(cue)

        if len(cues) != 9:
            raise ValueError(
                f"{target} must have exactly 9 unique cues; got {len(cues)}: {cues}"
            )
        cues_by_class[target] = cues

    return cues_by_class


class ScoreCache:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cue_scores (
                image_id INTEGER NOT NULL,
                cue TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY(image_id, cue)
            )
            """
        )
        self.conn.commit()

    def get(self, image_id, cue):
        row = self.conn.execute(
            "SELECT score FROM cue_scores WHERE image_id=? AND cue=?",
            (int(image_id), cue),
        ).fetchone()
        return None if row is None else float(row[0])

    def put_many(self, image_id, score_map):
        self.conn.executemany(
            "INSERT OR REPLACE INTO cue_scores(image_id,cue,score) VALUES(?,?,?)",
            [
                (int(image_id), cue, float(score))
                for cue, score in score_map.items()
            ],
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


class VLMCache:
    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                image_id INTEGER NOT NULL,
                target TEXT NOT NULL,
                prediction TEXT NOT NULL,
                raw TEXT NOT NULL,
                PRIMARY KEY(image_id, target)
            )
            """
        )
        self.conn.commit()

    def get(self, image_id, target):
        row = self.conn.execute(
            "SELECT prediction, raw FROM predictions WHERE image_id=? AND target=?",
            (int(image_id), target),
        ).fetchone()
        return None if row is None else (row[0], row[1])

    def put(self, image_id, target, prediction, raw):
        self.conn.execute(
            "INSERT OR REPLACE INTO predictions VALUES(?,?,?,?)",
            (int(image_id), target, prediction, raw),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


class OwlScorer:
    def __init__(self, model_id, device, dtype):
        self.device = device
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        self.processor = Owlv2Processor.from_pretrained(model_id)
        self.model = Owlv2ForObjectDetection.from_pretrained(
            model_id,
            torch_dtype=dtype_map[dtype],
        ).to(device).eval()

    @torch.inference_mode()
    def score(self, image, cues):
        prompts = [f"a photo of {cue}" for cue in cues]
        inputs = self.processor(
            text=[prompts],
            images=image,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
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

        scores = {cue: 0.0 for cue in cues}
        labels = results["labels"].detach().cpu().tolist()
        values = results["scores"].detach().float().cpu().tolist()

        for label, value in zip(labels, values):
            label = int(label)
            if 0 <= label < len(cues):
                cue = cues[label]
                scores[cue] = max(scores[cue], float(value))

        return scores


class LlavaScorer:
    def __init__(self, model_path, device, dtype):
        self.device = device
        dtype_map = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }
        model_dtype = dtype_map[dtype]
        if torch.device(device).type == "cpu":
            model_dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=True,
        ).to(device).eval()

    @torch.inference_mode()
    def predict(self, image, target):
        question = (
            f"Is the bird in this image a {target}? "
            "Answer only yes or no."
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }]

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
            key: value.to(self.device) if torch.is_tensor(value) else value
            for key, value in inputs.items()
        }

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=8,
            do_sample=False,
            use_cache=True,
        )
        prompt_length = inputs["input_ids"].shape[-1]
        raw = self.processor.batch_decode(
            output_ids[:, prompt_length:],
            skip_special_tokens=True,
        )[0].strip()

        match = re.search(r"\b(yes|no)\b", raw, flags=re.IGNORECASE)
        prediction = match.group(1).lower() if match else "unknown"
        return prediction, raw


def cue_scores_for_row(row, cues, detector, cache):
    scores = {}
    missing = []

    for cue in cues:
        value = cache.get(row["image_id"], cue)
        if value is None:
            missing.append(cue)
        else:
            scores[cue] = value

    if missing:
        with Image.open(row["path"]) as handle:
            image = handle.convert("RGB")
            new_scores = detector.score(image, missing)
        cache.put_many(row["image_id"], new_scores)
        scores.update(new_scores)

    return {cue: float(scores.get(cue, 0.0)) for cue in cues}


def prediction_for_row(row, target, vlm, cache):
    cached = cache.get(row["image_id"], target)
    if cached is not None:
        return cached

    with Image.open(row["path"]) as handle:
        image = handle.convert("RGB")
        result = vlm.predict(image, target)

    cache.put(row["image_id"], target, result[0], result[1])
    return result


def yes_rate(predictions):
    usable = [item for item in predictions if item in ("yes", "no")]
    if not usable:
        return 0.0
    return sum(item == "yes" for item in usable) / len(usable)


def top_bottom(rows, score_by_id, k_requested):
    ranked = sorted(
        rows,
        key=lambda row: score_by_id[row["image_id"]],
    )
    if len(ranked) < 2:
        return [], []

    k = min(k_requested, len(ranked) // 2)
    if k <= 0:
        return [], []

    return ranked[-k:], ranked[:k]


def rank_target_cues(
    target,
    rows,
    cues,
    spur_k,
    detector,
    score_cache,
    vlm,
    vlm_cache,
):
    target_y = 0 if target == "landbird" else 1
    present = [row for row in rows if row["y"] == target_y]
    absent = [row for row in rows if row["y"] != target_y]

    print(
        f"[{target}] ground truth: present={len(present)}, absent={len(absent)}"
    )

    metrics = {}

    for cue in cues:
        all_scores = {}
        for row in rows:
            row_scores = cue_scores_for_row(
                row, [cue], detector, score_cache
            )
            all_scores[row["image_id"]] = row_scores[cue]

        pa_top, pa_bottom = top_bottom(present, all_scores, spur_k)
        hr_top, hr_bottom = top_bottom(absent, all_scores, spur_k)

        pa_top_preds = [
            prediction_for_row(row, target, vlm, vlm_cache)[0]
            for row in pa_top
        ]
        pa_bottom_preds = [
            prediction_for_row(row, target, vlm, vlm_cache)[0]
            for row in pa_bottom
        ]
        hr_top_preds = [
            prediction_for_row(row, target, vlm, vlm_cache)[0]
            for row in hr_top
        ]
        hr_bottom_preds = [
            prediction_for_row(row, target, vlm, vlm_cache)[0]
            for row in hr_bottom
        ]

        pa_top_rate = yes_rate(pa_top_preds)
        pa_bottom_rate = yes_rate(pa_bottom_preds)
        hr_top_rate = yes_rate(hr_top_preds)
        hr_bottom_rate = yes_rate(hr_bottom_preds)

        metrics[cue] = {
            "pa_top_yes_rate": pa_top_rate,
            "pa_bottom_yes_rate": pa_bottom_rate,
            "pa_gap": pa_top_rate - pa_bottom_rate,
            "hr_top_yes_rate": hr_top_rate,
            "hr_bottom_yes_rate": hr_bottom_rate,
            "hr_gap": hr_top_rate - hr_bottom_rate,
            "pa_top_n": len(pa_top),
            "pa_bottom_n": len(pa_bottom),
            "hr_top_n": len(hr_top),
            "hr_bottom_n": len(hr_bottom),
        }

        print(
            f"  {cue}: PA gap={metrics[cue]['pa_gap']:.4f}, "
            f"HR gap={metrics[cue]['hr_gap']:.4f}"
        )

    selected = max(cues, key=lambda cue: metrics[cue]["hr_gap"])
    print(f"[{target}] selected cue by HR gap: {selected}")

    return {
        "target": target,
        "candidate_cues": cues,
        "selected_cue": selected,
        "metrics": metrics,
    }


def materialize(src, dst, mode):
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if mode == "none":
        return False
    if dst.exists() or dst.is_symlink():
        return True

    if mode == "hardlink":
        try:
            shutil.copyfile(src, dst, follow_symlinks=True)
            # Replace copy with hardlink when supported.
            dst.unlink()
            dst.hardlink_to(src)
        except OSError:
            if not dst.exists():
                shutil.copy2(src, dst)
    elif mode == "symlink":
        dst.symlink_to(Path(src).resolve())
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(mode)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Create 9-cue PURGE-M partitions for Waterbirds."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_csv", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--vlm_model_path", required=True)
    parser.add_argument(
        "--cue_json",
        default="",
        help="Optional JSON with exactly 9 cues under landbird and waterbird.",
    )
    parser.add_argument(
        "--detector_model",
        default="google/owlv2-base-patch16-ensemble",
    )
    parser.add_argument("--detector_device", default="cuda:0")
    parser.add_argument("--vlm_device", default="cuda:0")
    parser.add_argument(
        "--detector_dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument(
        "--vlm_dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--cue_threshold", type=float, default=0.10)
    parser.add_argument("--spur_k", type=int, default=100)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--link_mode",
        choices=("hardlink", "symlink", "copy", "none"),
        default="hardlink",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    metadata_csv = (
        Path(args.metadata_csv)
        if args.metadata_csv
        else data_root / "metadata.csv"
    )
    output_root = Path(args.output_root)

    if not data_root.is_dir():
        raise FileNotFoundError(data_root)
    if not metadata_csv.is_file():
        raise FileNotFoundError(metadata_csv)
    if not 0.0 <= args.cue_threshold <= 1.0:
        raise ValueError("--cue_threshold must be between 0 and 1")

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"{output_root} is not empty; use --overwrite to replace it"
            )
        shutil.rmtree(output_root)

    rows = load_rows(
        metadata_csv,
        data_root,
        limit=args.limit,
        seed=args.seed,
    )
    cues_by_class = load_cues(args.cue_json)

    train_root = output_root / "train"
    for folder in (
        "yy", "yn", "ny", "nn",
        "retain", "forget",
        "manifests", "rankings",
    ):
        (train_root / folder).mkdir(parents=True, exist_ok=True)

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
    score_cache = ScoreCache(output_root / "cue_scores.sqlite")
    vlm_cache = VLMCache(output_root / "vlm_predictions.sqlite")

    summary = {}

    try:
        for target in ("landbird", "waterbird"):
            cues = cues_by_class[target]

            ranking = rank_target_cues(
                target=target,
                rows=rows,
                cues=cues,
                spur_k=args.spur_k,
                detector=detector,
                score_cache=score_cache,
                vlm=vlm,
                vlm_cache=vlm_cache,
            )

            selected_cue = ranking["selected_cue"]
            ranking_path = train_root / "rankings" / f"{target}.json"
            ranking_path.write_text(
                json.dumps(ranking, indent=2),
                encoding="utf-8",
            )

            counts = {"yy": 0, "yn": 0, "ny": 0, "nn": 0}
            manifest_path = train_root / "manifests" / f"{target}.jsonl"

            with manifest_path.open("w", encoding="utf-8") as manifest:
                for row in tqdm(rows, desc=f"{target}: partitioning"):
                    target_present = (
                        row["y"] == (0 if target == "landbird" else 1)
                    )

                    cue_score = cue_scores_for_row(
                        row,
                        [selected_cue],
                        detector,
                        score_cache,
                    )[selected_cue]
                    cue_present = cue_score >= args.cue_threshold

                    if target_present and cue_present:
                        combo = "yy"
                    elif target_present and not cue_present:
                        combo = "yn"
                    elif not target_present and cue_present:
                        combo = "ny"
                    else:
                        combo = "nn"

                    counts[combo] += 1
                    true_class = row["class_name"]
                    image_name = Path(row["filename"]).name
                    output_name = (
                        f"{row['image_id']:08d}_{target}_{image_name}"
                    )

                    combo_path = (
                        train_root / combo / target / true_class / output_name
                    )
                    did_materialize_combo = materialize(
                        row["path"], combo_path, args.link_mode
                    )

                    training_role = None
                    did_materialize_train = False

                    if combo == "yn":
                        training_role = "retain"
                        train_path = (
                            train_root / "retain" / true_class
                            / target / output_name
                        )
                        did_materialize_train = materialize(
                            row["path"], train_path, args.link_mode
                        )
                    elif combo == "ny":
                        training_role = "forget"
                        train_path = (
                            train_root / "forget" / true_class
                            / target / output_name
                        )
                        did_materialize_train = materialize(
                            row["path"], train_path, args.link_mode
                        )
                    else:
                        training_role = "evaluation"

                    manifest.write(json.dumps({
                        "image_id": row["image_id"],
                        "img_filename": row["filename"],
                        "y": row["y"],
                        "true_class": true_class,
                        "place": row["place"],
                        "place_name": row["place_name"],
                        "target_class": target,
                        "target_present": target_present,
                        "target_presence_source": "Waterbirds metadata y",
                        "selected_cue": selected_cue,
                        "cue_score": cue_score,
                        "cue_threshold": args.cue_threshold,
                        "cue_present": cue_present,
                        "cue_presence_source": "OWLv2",
                        "combo": combo,
                        "training_role": training_role,
                        "combo_materialized": did_materialize_combo,
                        "train_materialized": did_materialize_train,
                    }) + "\n")

            summary[target] = {
                "selected_cue": selected_cue,
                "counts": counts,
                "cue_threshold": args.cue_threshold,
                "candidate_cues": cues,
            }

            print(
                f"[DONE][{target}] cue={selected_cue!r} "
                f"YY={counts['yy']} YN={counts['yn']} "
                f"NY={counts['ny']} NN={counts['nn']}"
            )

    finally:
        score_cache.close()
        vlm_cache.close()

    (train_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Training root for clip.py: {train_root}")
    print(f"Summary: {train_root / 'summary.json'}")
    print("YN is retain; NY is forget; YY and NN are evaluation groups.")


if __name__ == "__main__":
    main()