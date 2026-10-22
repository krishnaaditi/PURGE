#!/usr/bin/env python3

"""
PURGE-D Dataset Construction from MS COCO
------------------------------------------

For each target class and each positive source image:

RETAIN:
    - Start with a completely white image.
    - Copy each target-object bounding-box region independently.
    - If an image contains multiple target objects, all target boxes
      are retained separately in ONE output image.

FORGET:
    - Start with the original image.
    - Replace each target-object bounding-box region independently
      with white.
    - If an image contains multiple target objects, all target boxes
      are masked separately in ONE output image.

FORGET_NEG:
    - Use a natural COCO image that does not contain the target class.

Example:
    Original image contains 4 apples.

    retain:
        apple box 1 visible
        apple box 2 visible
        apple box 3 visible
        apple box 4 visible
        everything else white

    forget:
        apple box 1 white
        apple box 2 white
        apple box 3 white
        apple box 4 white
        rest of original image preserved

IMPORTANT:
    Bounding boxes are NOT merged into one union bounding box.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from pycocotools.coco import COCO


# ============================================================
# COCO 80 classes
# ============================================================

COCO80_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


# ============================================================
# General utilities
# ============================================================

def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def safe_name(name: str) -> str:
    """Make a class name safe for use inside filenames."""
    return name.replace(" ", "_")


def load_rgb_image(path: str) -> np.ndarray:
    """Load an image as RGB uint8 NumPy array."""
    with Image.open(path) as img:
        return np.array(img.convert("RGB"), dtype=np.uint8)


def save_np_image(
    arr: np.ndarray,
    path: str,
    resize: Optional[int] = None,
) -> None:
    """
    Save RGB NumPy image.

    If resize > 0, output is resized to resize x resize.
    """

    ensure_dir(os.path.dirname(path))

    img = Image.fromarray(arr.astype(np.uint8))

    if resize is not None and resize > 0:
        img = img.resize(
            (resize, resize),
            Image.Resampling.BICUBIC,
        )

    img.save(path)


# ============================================================
# Bounding-box utilities
# ============================================================

def bbox_xyxy_from_coco(
    bbox: List[float],
) -> Tuple[int, int, int, int]:
    """
    Convert COCO bbox:

        [x, y, width, height]

    into:

        [x1, y1, x2, y2]
    """

    x, y, width, height = bbox

    x1 = int(round(x))
    y1 = int(round(y))
    x2 = int(round(x + width))
    y2 = int(round(y + height))

    return x1, y1, x2, y2


def clamp_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> Tuple[int, int, int, int]:
    """Clamp bounding box to valid image coordinates."""

    x1 = max(0, min(image_width - 1, x1))
    y1 = max(0, min(image_height - 1, y1))

    x2 = max(0, min(image_width, x2))
    y2 = max(0, min(image_height, y2))

    return x1, y1, x2, y2


# ============================================================
# PURGE-D transformations
# ============================================================

def build_retain_white(
    image: np.ndarray,
    annotations: List[Dict],
) -> np.ndarray:
    """
    Create RETAIN image.

    Each target object's bounding box is copied separately
    onto a white background.

    Multiple target objects remain separate.
    No union bounding box is used.
    """

    image_height, image_width = image.shape[:2]

    # Start from a completely white image
    output = np.full_like(image, 255)

    for annotation in annotations:

        x1, y1, x2, y2 = bbox_xyxy_from_coco(
            annotation["bbox"]
        )

        x1, y1, x2, y2 = clamp_bbox(
            x1,
            y1,
            x2,
            y2,
            image_width,
            image_height,
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # Copy this target object region separately
        output[y1:y2, x1:x2] = image[y1:y2, x1:x2]

    return output


def build_forget_whitebox(
    image: np.ndarray,
    annotations: List[Dict],
) -> np.ndarray:
    """
    Create FORGET image.

    Every target-object bounding box is independently
    replaced with white.

    Multiple objects receive multiple separate white boxes.

    No union bounding box is used.
    """

    image_height, image_width = image.shape[:2]

    # Start with the complete original image
    output = image.copy()

    for annotation in annotations:

        x1, y1, x2, y2 = bbox_xyxy_from_coco(
            annotation["bbox"]
        )

        x1, y1, x2, y2 = clamp_bbox(
            x1,
            y1,
            x2,
            y2,
            image_width,
            image_height,
        )

        if x2 <= x1 or y2 <= y1:
            continue

        # Mask only this object's bounding box
        output[y1:y2, x1:x2] = 255

    return output


# ============================================================
# Metadata
# ============================================================

def write_jsonl(
    records: List[Dict],
    path: str,
) -> None:
    """Write metadata records as JSONL."""

    ensure_dir(os.path.dirname(path))

    with open(path, "w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Create PURGE-D retain, forget, and natural-negative "
            "datasets from MS COCO."
        )
    )

    parser.add_argument(
        "--coco_image_root",
        required=True,
        type=str,
        help="Path to COCO image folder, e.g. train2017",
    )

    parser.add_argument(
        "--coco_ann_json",
        required=True,
        type=str,
        help="Path to instances_train2017.json",
    )

    parser.add_argument(
        "--output_root",
        required=True,
        type=str,
        help="Output dataset directory",
    )

    parser.add_argument(
        "--train_per_class",
        type=int,
        default=100,
        help="Number of positive source images per class for training",
    )

    parser.add_argument(
        "--test_per_class",
        type=int,
        default=20,
        help="Number of positive source images per class for testing",
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=384,
        help="Output image size. Use 0 to preserve original size.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--class_names",
        type=str,
        default="",
        help=(
            "Comma-separated subset of COCO classes. "
            "Example: apple,dog,car. "
            "Leave empty for all 80 classes."
        ),
    )

    parser.add_argument(
        "--min_bbox_w",
        type=float,
        default=12,
        help="Minimum bounding-box width",
    )

    parser.add_argument(
        "--min_bbox_h",
        type=float,
        default=12,
        help="Minimum bounding-box height",
    )

    args = parser.parse_args()

    # ========================================================
    # Validate paths
    # ========================================================

    if not os.path.isdir(args.coco_image_root):
        raise FileNotFoundError(
            f"COCO image directory does not exist:\n"
            f"{args.coco_image_root}"
        )

    if not os.path.isfile(args.coco_ann_json):
        raise FileNotFoundError(
            f"COCO annotation file does not exist:\n"
            f"{args.coco_ann_json}"
        )

    ensure_dir(args.output_root)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ========================================================
    # Load COCO
    # ========================================================

    print("=" * 70)
    print("Loading COCO annotations")
    print("=" * 70)

    coco = COCO(args.coco_ann_json)

    categories = coco.loadCats(
        coco.getCatIds()
    )

    name_to_id = {
        category["name"]: category["id"]
        for category in categories
    }

    all_image_ids = coco.getImgIds()

    # ========================================================
    # Select classes
    # ========================================================

    if args.class_names:

        requested = {
            name.strip()
            for name in args.class_names.split(",")
            if name.strip()
        }

        classes = [
            name
            for name in COCO80_CLASSES
            if name in requested
        ]

        unknown = requested - set(COCO80_CLASSES)

        if unknown:
            print(
                "[warning] Unknown classes ignored:",
                sorted(unknown)
            )

    else:
        classes = COCO80_CLASSES.copy()

    print(f"Number of classes: {len(classes)}")
    print(classes)
    print()

    metadata_records: List[Dict] = []

    # ========================================================
    # Process each class
    # ========================================================

    for class_index, class_name in enumerate(classes):

        print("=" * 70)
        print(f"Processing class: {class_name}")
        print("=" * 70)

        if class_name not in name_to_id:
            print(
                f"[skip] '{class_name}' is not present "
                f"in the COCO annotation file."
            )
            continue

        category_id = name_to_id[class_name]

        # ----------------------------------------------------
        # Get all non-crowd annotations for this class
        # ----------------------------------------------------

        annotation_ids = coco.getAnnIds(
            catIds=[category_id],
            iscrowd=False,
        )

        annotations = coco.loadAnns(
            annotation_ids
        )

        # ----------------------------------------------------
        # Filter small target objects
        # ----------------------------------------------------

        valid_annotations = [
            annotation
            for annotation in annotations
            if annotation["bbox"][2] >= args.min_bbox_w
            and annotation["bbox"][3] >= args.min_bbox_h
        ]

        # ----------------------------------------------------
        # Group target annotations by image
        # ----------------------------------------------------

        image_to_annotations: Dict[int, List[Dict]] = {}

        for annotation in valid_annotations:

            image_id = annotation["image_id"]

            if image_id not in image_to_annotations:
                image_to_annotations[image_id] = []

            image_to_annotations[image_id].append(
                annotation
            )

        # Images with at least one sufficiently large object
        positive_image_ids = list(
            image_to_annotations.keys()
        )

        # ----------------------------------------------------
        # Natural negatives
        #
        # Important:
        # Use ALL target-class image IDs here, including images
        # whose objects were too small and got filtered above.
        #
        # This guarantees that forget_neg truly does not contain
        # the target class according to COCO.
        # ----------------------------------------------------

        all_target_image_ids = set(
            coco.getImgIds(
                catIds=[category_id]
            )
        )

        negative_image_ids = [
            image_id
            for image_id in all_image_ids
            if image_id not in all_target_image_ids
        ]

        # ----------------------------------------------------
        # Deterministic randomization
        # ----------------------------------------------------

        positive_rng = random.Random(
            args.seed + class_index
        )

        negative_rng = random.Random(
            args.seed + 10000 + class_index
        )

        positive_rng.shuffle(
            positive_image_ids
        )

        negative_rng.shuffle(
            negative_image_ids
        )

        total_needed = (
            args.train_per_class
            + args.test_per_class
        )

        if len(positive_image_ids) < total_needed:
            print(
                f"[warning] {class_name}: requested "
                f"{total_needed} positive images but only "
                f"{len(positive_image_ids)} are available."
            )

        if len(negative_image_ids) < total_needed:
            print(
                f"[warning] {class_name}: requested "
                f"{total_needed} negative images but only "
                f"{len(negative_image_ids)} are available."
            )

        positive_image_ids = positive_image_ids[
            :total_needed
        ]

        negative_image_ids = negative_image_ids[
            :total_needed
        ]

        # ====================================================
        # Train / test split
        # ====================================================

        split_settings = [
            (
                "train",
                0,
                args.train_per_class,
            ),
            (
                "test",
                args.train_per_class,
                args.train_per_class + args.test_per_class,
            ),
        ]

        for split_name, start, end in split_settings:

            class_filename = safe_name(
                class_name
            )

            # Keep actual COCO class name for folder
            # Example:
            #   traffic light/
            # instead of traffic_light/
            class_folder = class_name

            retain_dir = os.path.join(
                args.output_root,
                "images",
                split_name,
                "retain",
                class_folder,
            )

            forget_dir = os.path.join(
                args.output_root,
                "images",
                split_name,
                "forget",
                class_folder,
            )

            forget_neg_dir = os.path.join(
                args.output_root,
                "images",
                split_name,
                "forget_neg",
                class_folder,
            )

            ensure_dir(retain_dir)
            ensure_dir(forget_dir)
            ensure_dir(forget_neg_dir)

            selected_positive_ids = (
                positive_image_ids[start:end]
            )

            selected_negative_ids = (
                negative_image_ids[start:end]
            )

            saved_retain = 0
            saved_forget = 0
            saved_negative = 0

            # =================================================
            # Positive source images
            # =================================================

            for image_id in selected_positive_ids:

                image_info = coco.loadImgs(
                    [image_id]
                )[0]

                source_path = os.path.join(
                    args.coco_image_root,
                    image_info["file_name"],
                )

                if not os.path.isfile(source_path):
                    print(
                        f"[missing] {source_path}"
                    )
                    continue

                try:
                    image = load_rgb_image(
                        source_path
                    )
                except Exception as error:
                    print(
                        f"[error] Failed to read "
                        f"{source_path}: {error}"
                    )
                    continue

                target_annotations = (
                    image_to_annotations[image_id]
                )

                # ---------------------------------------------
                # ONE retain image per source image
                # with ALL target objects retained separately
                # ---------------------------------------------

                retain_image = build_retain_white(
                    image,
                    target_annotations,
                )

                # ---------------------------------------------
                # ONE forget image per source image
                # with ALL target objects masked separately
                # ---------------------------------------------

                forget_image = build_forget_whitebox(
                    image,
                    target_annotations,
                )

                source_stem = Path(
                    image_info["file_name"]
                ).stem

                retain_filename = (
                    f"{source_stem}_"
                    f"{class_filename}_retain.png"
                )

                forget_filename = (
                    f"{source_stem}_"
                    f"{class_filename}_forget.png"
                )

                retain_path = os.path.join(
                    retain_dir,
                    retain_filename,
                )

                forget_path = os.path.join(
                    forget_dir,
                    forget_filename,
                )

                save_np_image(
                    retain_image,
                    retain_path,
                    args.resize,
                )

                save_np_image(
                    forget_image,
                    forget_path,
                    args.resize,
                )

                saved_retain += 1
                saved_forget += 1

                # ---------------------------------------------
                # Store every individual bounding box
                # ---------------------------------------------

                bbox_records = []

                image_height, image_width = (
                    image.shape[:2]
                )

                for annotation in target_annotations:

                    x1, y1, x2, y2 = (
                        bbox_xyxy_from_coco(
                            annotation["bbox"]
                        )
                    )

                    x1, y1, x2, y2 = clamp_bbox(
                        x1,
                        y1,
                        x2,
                        y2,
                        image_width,
                        image_height,
                    )

                    bbox_records.append({
                        "annotation_id":
                            int(annotation["id"]),

                        "bbox_xywh": [
                            float(value)
                            for value
                            in annotation["bbox"]
                        ],

                        "bbox_xyxy": [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2),
                        ],
                    })

                metadata_records.append({
                    "split":
                        split_name,

                    "class_name":
                        class_name,

                    "category_id":
                        int(category_id),

                    "source_image_id":
                        int(image_id),

                    "source_file_name":
                        image_info["file_name"],

                    "num_target_instances":
                        len(target_annotations),

                    "target_instances":
                        bbox_records,

                    "retain_path":
                        os.path.relpath(
                            retain_path,
                            args.output_root,
                        ),

                    "forget_path":
                        os.path.relpath(
                            forget_path,
                            args.output_root,
                        ),

                    "type":
                        "positive",
                })

            # =================================================
            # Natural negative images
            # =================================================

            for image_id in selected_negative_ids:

                image_info = coco.loadImgs(
                    [image_id]
                )[0]

                source_path = os.path.join(
                    args.coco_image_root,
                    image_info["file_name"],
                )

                if not os.path.isfile(source_path):
                    print(
                        f"[missing] {source_path}"
                    )
                    continue

                try:
                    image = load_rgb_image(
                        source_path
                    )
                except Exception as error:
                    print(
                        f"[error] Failed to read "
                        f"{source_path}: {error}"
                    )
                    continue

                source_stem = Path(
                    image_info["file_name"]
                ).stem

                negative_filename = (
                    f"{source_stem}_"
                    f"{class_filename}_negative.png"
                )

                negative_path = os.path.join(
                    forget_neg_dir,
                    negative_filename,
                )

                save_np_image(
                    image,
                    negative_path,
                    args.resize,
                )

                saved_negative += 1

                metadata_records.append({
                    "split":
                        split_name,

                    "class_name":
                        class_name,

                    "category_id":
                        int(category_id),

                    "source_image_id":
                        int(image_id),

                    "source_file_name":
                        image_info["file_name"],

                    "num_target_instances":
                        0,

                    "target_instances":
                        [],

                    "forget_neg_path":
                        os.path.relpath(
                            negative_path,
                            args.output_root,
                        ),

                    "type":
                        "natural_negative",
                })

            # =================================================
            # Split summary
            # =================================================

            print(
                f"[PURGE-D]"
                f"[{split_name}]"
                f"[{class_name}] "
                f"retain={saved_retain} | "
                f"forget={saved_forget} | "
                f"forget_neg={saved_negative}"
            )

    # ========================================================
    # Write metadata
    # ========================================================

    metadata_path = os.path.join(
        args.output_root,
        "metadata.jsonl",
    )

    write_jsonl(
        metadata_records,
        metadata_path,
    )

    print()
    print("=" * 70)
    print("PURGE-D generation complete")
    print("=" * 70)
    print(
        f"Output root : {args.output_root}"
    )
    print(
        f"Metadata    : {metadata_path}"
    )
    print(
        f"Records     : {len(metadata_records)}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()