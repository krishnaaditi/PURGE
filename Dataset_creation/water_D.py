#!/usr/bin/env python3
"""Build rectangular-box PURGE-D partitions for prepared Waterbirds.

Expected input layout:
    data_root/{train,val,test}/
        landbird_land/       # if present
        landbird_water/      # if present
        waterbird_land/      # if present
        waterbird_water/     # if present

For each TRAIN image, a COCO-pretrained Faster R-CNN detects bird boxes.
RETAIN is a white canvas containing the rectangular bird boxes. FORGET is
the original image with those rectangles painted white. This uses rectangular
boxes, not polygon/shape masks. It is a practical Waterbirds adaptation; it
does not claim to reproduce undisclosed Waterbirds box annotations from the
paper.

Output layout (compatible with a trainer whose --pau_root points to train/):
    output_root/train/retain/{landbird,waterbird}/...
    output_root/train/forget/{landbird,waterbird}/...
    output_root/train/manifest.jsonl

Validation and test images are not transformed or used for training.
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
    fasterrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import to_tensor


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GROUP_RE = re.compile(r"^(landbird|waterbird)_(land|water)$", re.IGNORECASE)


def list_train_images(data_root: Path, metadata_csv: Path, split: str = "train"):
    split_values = {
        "train": {"train", "0"},
        "val": {"val", "valid", "validation", "1"},
        "test": {"test", "2"},
    }
    if split not in split_values:
        raise ValueError(f"Unsupported split: {split}")
    if not metadata_csv.is_file():
        raise FileNotFoundError(f"Missing Waterbirds metadata CSV: {metadata_csv}")

    samples = []
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"img_filename", "y", "split", "place"}
        columns = {name.strip() for name in (reader.fieldnames or [])}
        missing = required - columns
        if missing:
            raise ValueError(f"Metadata missing columns: {sorted(missing)}")

        for raw_row in reader:
            row = {key.strip(): value for key, value in raw_row.items() if key}
            if str(row["split"]).strip().lower() not in split_values[split]:
                continue

            y = int(row["y"])
            place = int(row["place"])
            if y not in (0, 1) or place not in (0, 1):
                raise ValueError(f"Expected binary y/place values; got y={y}, place={place}")

            image_path = data_root / row["img_filename"]
            if not image_path.is_file():
                raise FileNotFoundError(f"Image listed in metadata is missing: {image_path}")

            bird_class = "waterbird" if y == 1 else "landbird"
            background = "water" if place == 1 else "land"
            group_name = f"{bird_class}_{background}"
            samples.append((image_path, bird_class, background, group_name))

    if not samples:
        raise ValueError(f"No images found for split={split} in {metadata_csv}")
    return samples

def clamp_box(box, width: int, height: int):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2 + 0.999)))
    y2 = max(0, min(height, int(y2 + 0.999)))
    return x1, y1, x2, y2


def detect_bird_boxes(detector, bird_label: int, image: Image.Image,
                      device: torch.device, threshold: float):
    tensor = to_tensor(image).to(device)
    with torch.inference_mode():
        prediction = detector([tensor])[0]

    boxes = prediction["boxes"].detach().cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    keep = (labels == bird_label) & (scores >= threshold)
    return [(boxes[i].tolist(), float(scores[i])) for i in keep.nonzero().flatten()]


def create_box_views(image: Image.Image, detections):
    """Return object-box-only retain and box-removed context-only forget views."""
    image = image.convert("RGB")
    width, height = image.size
    source = image
    retain = Image.new("RGB", (width, height), color=(255, 255, 255))
    forget = source.copy()
    pixel = Image.new("RGB", (width, height), color=(255, 255, 255))

    boxes_out = []
    for box, score in detections:
        x1, y1, x2, y2 = clamp_box(box, width, height)
        if x2 <= x1 or y2 <= y1:
            continue
        # Rectangular box, deliberately not the bird's segmentation silhouette.
        retain.paste(source.crop((x1, y1, x2, y2)), (x1, y1))
        forget.paste(pixel.crop((x1, y1, x2, y2)), (x1, y1))
        boxes_out.append({"xyxy": [x1, y1, x2, y2], "score": score})

    return retain, forget, boxes_out


def save_image(image: Image.Image, path: Path, size: int = 0):
    path.parent.mkdir(parents=True, exist_ok=True)
    if size > 0:
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(
        description="Create bbox-based PURGE-D train partitions for Waterbirds."
    )
    parser.add_argument("--data_root", required=True,
                        help="Raw Waterbirds directory containing image folders")
    parser.add_argument("--metadata_csv", default="", help="Defaults to <data_root>/metadata.csv")
    parser.add_argument("--output_root", required=True,
                        help="Output root for generated retain/forget partitions")
    parser.add_argument("--split", default="train", choices=["train"],
                        help="Only train is accepted to prevent validation/test leakage")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score_threshold", type=float, default=0.30,
                        help="Minimum Faster R-CNN confidence for a bird box")
    parser.add_argument("--resize", type=int, default=0,
                        help="Resize outputs to NxN; 0 preserves source resolution")
    parser.add_argument("--skip_no_detection", action="store_true",
                        help="Skip images where the detector finds no bird; default is to stop")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional debug limit; 0 processes all train images")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    metadata_csv = Path(args.metadata_csv) if args.metadata_csv else data_root / "metadata.csv"
    samples = list_train_images(data_root, metadata_csv, args.split)
    if args.limit > 0:
        samples = samples[:args.limit]

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    categories = weights.meta["categories"]
    if "bird" not in categories:
        raise RuntimeError("The detector's COCO labels do not include 'bird'")
    bird_label = categories.index("bird")
    detector = fasterrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()

    manifest_path = output_root / args.split / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    totals = {"images": len(samples), "saved": 0, "no_detection": 0, "errors": 0}

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (image_path, bird_class, background, group_name) in enumerate(samples, 1):
            try:
                with Image.open(image_path) as handle:
                    image = handle.convert("RGB")
                rel = image_path.relative_to(data_root)
                stem = "__".join(rel.with_suffix("").parts)
                forget_neg_path = output_root / args.split / "forget_neg" / bird_class / f"{stem}_forget_neg.png"
                save_image(image, forget_neg_path, args.resize)
                detections = detect_bird_boxes(
                    detector, bird_label, image, device, args.score_threshold
                )
                retain, forget, boxes = create_box_views(image, detections)

                if not boxes:
                    totals["no_detection"] += 1
                    message = f"No bird box detected: {image_path}"
                    if args.skip_no_detection:
                        print(f"[skip] {message}")
                        continue
                    raise RuntimeError(message + "; lower --score_threshold or pass --skip_no_detection")

                # Include group and relative path in filenames to prevent collisions.
                rel = image_path.relative_to(data_root)
                stem = "__".join(rel.with_suffix("").parts)
                retain_path = output_root / args.split / "retain" / bird_class / f"{stem}_retain.png"
                forget_path = output_root / args.split / "forget" / bird_class / f"{stem}_forget.png"
                save_image(retain, retain_path, args.resize)
                save_image(forget, forget_path, args.resize)

                record = {
                    "split": args.split,
                    "label": bird_class,
                    "background": background,
                    "group": group_name,
                    "source_path": str(image_path),
                    "retain_path": str(retain_path),
                    "forget_path": str(forget_path),
                    "boxes": boxes,
                    "mask_type": "rectangular_bounding_boxes",
                }
                manifest.write(json.dumps(record) + "\n")
                totals["saved"] += 1

                if index % 100 == 0 or index == len(samples):
                    print(f"[{index}/{len(samples)}] saved={totals['saved']} "
                          f"no_detection={totals['no_detection']}")
            except Exception as error:
                totals["errors"] += 1
                print(f"[error] {image_path}: {error}")
                raise

    print("Finished PURGE-D Waterbirds box partition construction.")
    print(json.dumps(totals, indent=2))
    print(f"Manifest: {manifest_path}")
    print(f"Training PAU root: {output_root / args.split}")


if __name__ == "__main__":
    main()