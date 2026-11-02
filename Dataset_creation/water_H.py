#!/usr/bin/env python3
"""Create Waterbirds PURGE-H original/edited train partitions.

Waterbirds has no bird segmentation masks, so COCO Mask R-CNN estimates them.
SimpleLama receives the full image as context. The bird silhouette mask is
expanded slightly, and the generated fill is composited only in that region.

LLaVA answers whether a bird is present in each original and edited image.
Those answers determine the PURGE-H retain/forget routing.
"""

import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from simple_lama_inpainting import SimpleLama
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from transformers import AutoProcessor, LlavaNextForConditionalGeneration


SPLIT_MAP = {
    "train": 0,
    "0": 0,
    "val": 1,
    "valid": 1,
    "validation": 1,
    "1": 1,
    "test": 2,
    "2": 2,
}
CLASS_NAMES = {0: "landbird", 1: "waterbird"}


def parse_split(value):
    key = str(value).strip().lower()
    if key not in SPLIT_MAP:
        raise ValueError(
            f"Unknown split {value!r}; expected train/val/test or 0/1/2"
        )
    return SPLIT_MAP[key]


def parse_yes_no(text):
    match = re.search(r"\b(yes|no)\b", text.strip(), flags=re.IGNORECASE)
    return match.group(1).lower() if match else "unknown"


def route_original(prediction):
    # Original Waterbirds image: ground-truth bird presence is YES.
    if prediction == "yes":
        return "retain", "original_yes"
    return "forget", "original_no"


def route_edited(prediction):
    # Bird-removed image: ground-truth bird presence is NO.
    if prediction == "no":
        return "retain", "edited_no"
    return "forget", "edited_yes"


def load_train_rows(metadata_csv, image_root):
    rows = []

    with Path(metadata_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"img_filename", "y", "place", "split"}
        columns = set(reader.fieldnames or [])
        missing = required - columns
        if missing:
            raise ValueError(
                f"Metadata is missing columns {sorted(missing)}; "
                f"found {sorted(columns)}"
            )

        for row in reader:
            if parse_split(row["split"]) != 0:
                continue

            y = int(row["y"])
            place = int(row["place"])
            if y not in CLASS_NAMES or place not in (0, 1):
                raise ValueError(
                    f"Unexpected Waterbirds labels y={y}, place={place}"
                )

            image_path = Path(image_root) / row["img_filename"]
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Image listed in metadata is missing: {image_path}"
                )

            rows.append({
                "path": image_path,
                "filename": row["img_filename"],
                "y": y,
                "class_name": CLASS_NAMES[y],
                "place": place,
                "place_name": "water" if place == 1 else "land",
            })

    return rows


@torch.inference_mode()
def detect_bird_regions(image, detector, device, score_threshold):
    """Return COCO Mask R-CNN bird masks. COCO label 16 is bird."""
    rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    tensor = (
        torch.from_numpy(rgb)
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        .to(device)
    )

    prediction = detector([tensor])[0]
    keep = (
        (prediction["labels"] == 16)
        & (prediction["scores"] >= score_threshold)
    )

    if not bool(keep.any()):
        return [], (rgb.shape[1], rgb.shape[0])

    masks = prediction["masks"][keep, 0].detach().cpu().numpy()
    scores = prediction["scores"][keep].detach().cpu().tolist()

    detections = [
        {"mask": mask, "score": float(score)}
        for mask, score in zip(masks, scores)
    ]

    return detections, (rgb.shape[1], rgb.shape[0])


def build_bird_mask(image_size, detections, halo_pixels):
    """Use the strongest bird's segmentation silhouette, not its box."""
    if not detections:
        return None

    width, height = image_size
    best = max(detections, key=lambda item: item["score"])

    # Slightly inclusive threshold helps cover the bird's complete silhouette.
    core_mask = (best["mask"] >= 0.45).astype(np.uint8)
    if not core_mask.any():
        return None

    halo_pixels = max(2, min(int(halo_pixels), 24))
    kernel_size = 2 * halo_pixels + 1

    mask_image = Image.fromarray(core_mask * 255, mode="L")
    mask_image = mask_image.filter(ImageFilter.MaxFilter(kernel_size))
    mask_image = mask_image.filter(ImageFilter.GaussianBlur(radius=1.0))
    expanded_mask = (np.asarray(mask_image) >= 70).astype(np.uint8)

    if expanded_mask.shape != (height, width):
        raise RuntimeError("Bird mask dimensions do not match image dimensions")

    return expanded_mask


def inpaint_full_image(image, mask, lama):
    """Give LaMa the full image and keep original pixels outside the mask."""
    image = image.convert("RGB")
    hard_mask = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

    # LaMa receives the full image, so it can use the surrounding scene.
    fill = lama(image, hard_mask)

    if not isinstance(fill, Image.Image):
        fill = Image.fromarray(np.asarray(fill).astype(np.uint8))
    fill = fill.convert("RGB")

    if fill.size != image.size:
        fill = fill.resize(image.size, Image.Resampling.LANCZOS)

    # Feather outside the mask, but force full replacement everywhere inside
    # the expanded mask so original bird pixels cannot show through.
    blurred = np.asarray(
        hard_mask.filter(ImageFilter.GaussianBlur(radius=2.0)),
        dtype=np.uint8,
    )
    hard = np.asarray(hard_mask, dtype=np.uint8)
    alpha = Image.fromarray(np.maximum(blurred, hard), mode="L")

    return Image.composite(fill, image, alpha)


def residual_overlaps_target(detections, target_mask):
    """Return True if a post-inpainting bird detection overlaps the target."""
    if not detections:
        return False

    for detection in detections:
        residual = detection["mask"] >= 0.30
        residual_area = int(residual.sum())
        if residual_area == 0:
            continue

        intersection = int(
            np.logical_and(residual, target_mask > 0).sum()
        )
        if intersection >= 15 and intersection / residual_area >= 0.03:
            return True

    return False


def remove_bird_with_retries(
    image,
    detector,
    device,
    lama,
    score_threshold,
    max_retries=4,
):
    """Inpaint with progressively larger silhouette masks and verify removal."""
    source_detections, image_size = detect_bird_regions(
        image,
        detector,
        device,
        score_threshold,
    )
    if not source_detections:
        raise RuntimeError("Mask R-CNN did not detect a bird in the original")

    halos = [8, 12, 18, 24][:max_retries]
    verify_threshold = min(score_threshold, 0.08)
    attempts = []

    for attempt_number, halo in enumerate(halos, start=1):
        target_mask = build_bird_mask(
            image_size,
            source_detections,
            halo_pixels=halo,
        )
        if target_mask is None:
            continue

        edited = inpaint_full_image(image, target_mask, lama)

        remaining, _ = detect_bird_regions(
            edited,
            detector,
            device,
            score_threshold=verify_threshold,
        )
        still_detected = residual_overlaps_target(remaining, target_mask)

        attempts.append({
            "attempt": attempt_number,
            "halo_pixels": halo,
            "bird_detected_in_target_region": bool(still_detected),
            "residual_scores": [
                item["score"] for item in remaining
            ],
        })

        if not still_detected:
            return edited, target_mask, attempts

    raise RuntimeError(
        "Bird remained detectable after all inpainting attempts; "
        f"attempts={attempts}"
    )


@torch.inference_mode()
def ask_bird_presence(model, processor, image, device):
    question = "Is there a bird visible in this image? Answer only yes or no."
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }]

    try:
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = f"<image>\n{question}"

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    model_inputs = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            if key == "pixel_values":
                value = value.to(device=device, dtype=model.dtype)
            else:
                value = value.to(device)
        model_inputs[key] = value

    output_ids = model.generate(
        **model_inputs,
        max_new_tokens=8,
        do_sample=False,
        use_cache=True,
    )
    prompt_length = model_inputs["input_ids"].shape[-1]
    generated = output_ids[:, prompt_length:]
    raw_answer = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    return parse_yes_no(raw_answer), raw_answer


def save_image(image, path, max_side=0):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")

    if max_side > 0:
        image.thumbnail(
            (max_side, max_side),
            Image.Resampling.LANCZOS,
        )

    image.save(path, quality=95)
    return path


def save_preview(original, mask, edited, path):
    """Save original, mask overlay, and edited image side by side."""
    original = original.convert("RGB")
    edited = edited.convert("RGB")
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255, mode="L")

    overlay = original.copy()
    red = Image.new("RGB", original.size, (255, 0, 0))
    overlay.paste(
        red,
        mask=mask_image.point(lambda value: int(value * 0.40)),
    )

    panel_width, panel_height = 360, 270
    canvas = Image.new(
        "RGB",
        (panel_width * 3, panel_height + 30),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    panels = [
        (original, "Original"),
        (overlay, "Bird mask"),
        (edited, "Edited"),
    ]
    for index, (panel, label) in enumerate(panels):
        panel = panel.copy()
        panel.thumbnail(
            (panel_width, panel_height),
            Image.Resampling.LANCZOS,
        )
        x = index * panel_width
        canvas.paste(panel, (x, 30))
        draw.text((x + 8, 8), label, fill="black")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Create Waterbirds PURGE-H original/edited partitions."
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_csv", default="")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--lama_device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("bf16", "fp16", "fp32"),
        default="bf16",
    )
    parser.add_argument("--score_threshold", type=float, default=0.30)
    parser.add_argument(
        "--max_retries",
        type=int,
        choices=(1, 2, 3, 4),
        default=4,
    )
    parser.add_argument("--preview_count", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resize", type=int, default=0)
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
        raise FileNotFoundError(f"Missing Waterbirds directory: {data_root}")
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Missing metadata CSV: {metadata_csv}")
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score_threshold must be between 0 and 1")

    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory is not empty: {output_root}; "
                "use --overwrite to replace it"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = load_train_rows(metadata_csv, data_root)
    random.Random(args.seed).shuffle(rows)

    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No training images found; expected metadata split=0/train")

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    print("Loading COCO Mask R-CNN...")
    detector = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    ).to(device).eval()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]
    if device.type == "cpu":
        model_dtype = torch.float32

    print("Loading LLaVA...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    vlm = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    ).to(device).eval()

    print(f"Loading SimpleLama on {args.lama_device}...")
    lama = SimpleLama(device=torch.device(args.lama_device))

    train_root = output_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    manifest_path = train_root / "manifest.jsonl"
    rejection_path = train_root / "rejections.jsonl"

    counts = {
        "input_images": len(rows),
        "accepted_pairs": 0,
        "retain_images": 0,
        "forget_images": 0,
        "rejected_images": 0,
    }

    with (
        manifest_path.open("w", encoding="utf-8") as manifest,
        rejection_path.open("w", encoding="utf-8") as rejections,
    ):
        for index, row in enumerate(rows, start=1):
            try:
                with Image.open(row["path"]) as handle:
                    original = handle.convert("RGB")

                edited, final_mask, attempts = remove_bird_with_retries(
                    image=original,
                    detector=detector,
                    device=device,
                    lama=lama,
                    score_threshold=args.score_threshold,
                    max_retries=args.max_retries,
                )

                original_pred, original_raw = ask_bird_presence(
                    vlm, processor, original, device
                )
                edited_pred, edited_raw = ask_bird_presence(
                    vlm, processor, edited, device
                )

                if "unknown" in (original_pred, edited_pred):
                    raise RuntimeError(
                        "Could not parse LLaVA yes/no answer: "
                        f"original={original_raw!r}; edited={edited_raw!r}"
                    )

                original_partition, original_group = route_original(original_pred)
                edited_partition, edited_group = route_edited(edited_pred)

                relative_source = Path(row["filename"])
                stem = "__".join(relative_source.with_suffix("").parts)
                class_name = row["class_name"]

                original_out = save_image(
                    original,
                    train_root / original_partition / original_group
                    / class_name / f"{stem}_original.jpg",
                    args.resize,
                )
                edited_out = save_image(
                    edited,
                    train_root / edited_partition / edited_group
                    / class_name / f"{stem}_edited.jpg",
                    args.resize,
                )

                preview_out = None
                if counts["accepted_pairs"] < args.preview_count:
                    preview_out = save_preview(
                        original,
                        final_mask,
                        edited,
                        train_root / "previews" / f"{stem}_preview.jpg",
                    )

                record = {
                    "source_path": str(row["path"]),
                    "source_file": row["filename"],
                    "y": row["y"],
                    "class_name": class_name,
                    "place": row["place"],
                    "place_name": row["place_name"],
                    "object_query": "bird",
                    "original_gt_presence": "yes",
                    "original_prediction": original_pred,
                    "original_raw_answer": original_raw,
                    "original_partition": original_partition,
                    "original_group": original_group,
                    "edited_gt_presence": "no",
                    "edited_prediction": edited_pred,
                    "edited_raw_answer": edited_raw,
                    "edited_partition": edited_partition,
                    "inpainting_backend": "SimpleLama full-image context",
                    "inpainting_attempts": attempts,
                    "original_output": str(original_out.relative_to(output_root)),
                    "edited_output": str(edited_out.relative_to(output_root)),
                    "preview_output": (
                        str(preview_out.relative_to(output_root))
                        if preview_out
                        else None
                    ),
                }
                manifest.write(json.dumps(record) + "\n")
                manifest.flush()

                counts["accepted_pairs"] += 1
                counts[f"{original_partition}_images"] += 1
                counts[f"{edited_partition}_images"] += 1

            except Exception as error:
                rejections.write(
                    json.dumps({
                        "source_file": row["filename"],
                        "reason": str(error),
                    }) + "\n"
                )
                rejections.flush()
                counts["rejected_images"] += 1
                print(f"[rejected] {row['filename']}: {error}")

            if index % 20 == 0 or index == len(rows):
                print(
                    f"[{index}/{len(rows)}] "
                    f"pairs={counts['accepted_pairs']} "
                    f"retain={counts['retain_images']} "
                    f"forget={counts['forget_images']} "
                    f"rejected={counts['rejected_images']}"
                )

    summary = {
        **counts,
        "output_root": str(output_root),
        "routing": {
            "original_yes": "retain",
            "original_no": "forget",
            "edited_no": "retain",
            "edited_yes": "forget",
        },
    }
    (train_root / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Finished Waterbirds PURGE-H creation.")
    print(json.dumps(counts, indent=2))
    print(f"PAU root for clip.py: {train_root}")
    print(f"Previews: {train_root / 'previews'}")
    print(f"Manifest: {manifest_path}")
    print(f"Rejected images: {rejection_path}")


if __name__ == "__main__":
    main()