#!/usr/bin/env python3

"""
PURGE-D generation for Waterbirds using bounding boxes.

The CUB segmentation mask is used ONLY to recover a tight bounding
box around the bird. The final PURGE-D transformations use rectangular
bounding boxes, matching the COCO PURGE-D construction.

For each Waterbirds image:

    RETAIN:
        Keep the complete bird bounding-box region.
        Replace everything outside the bounding box with white.

    FORGET:
        Replace the complete bird bounding-box region with white.
        Preserve everything outside the bounding box.

Waterbirds metadata:
    y:
        0 = landbird
        1 = waterbird

    place:
        0 = land background
        1 = water background

    split:
        0 = train
        1 = validation
        2 = test
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


# ============================================================
# Waterbirds metadata definitions
# ============================================================

CLASS_NAMES = {
    0: "landbird",
    1: "waterbird",
}

CLASS_IDS = {
    "landbird": 0,
    "waterbird": 1,
}

PLACE_NAMES = {
    0: "land",
    1: "water",
}

SPLIT_NAMES = {
    0: "train",
    1: "val",
    2: "test",
}


# ============================================================
# General utilities
# ============================================================

def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def load_rgb_image(path: Path) -> np.ndarray:
    """Load an RGB image as uint8 NumPy array."""

    with Image.open(path) as image:
        return np.asarray(
            image.convert("RGB"),
            dtype=np.uint8,
        )


def load_binary_mask(
    path: Path,
    target_width: int,
    target_height: int,
    threshold: int = 127,
) -> np.ndarray:
    """
    Load a CUB segmentation mask.

    If the mask dimensions differ from the Waterbirds image,
    resize the mask with nearest-neighbor interpolation.

    Returns:
        Boolean array of shape [H, W].
    """

    with Image.open(path) as mask_image:

        mask_image = mask_image.convert("L")

        expected_size = (
            target_width,
            target_height,
        )

        if mask_image.size != expected_size:

            print(
                f"[warning] Resizing mask "
                f"{mask_image.size} -> {expected_size}: "
                f"{path}"
            )

            mask_image = mask_image.resize(
                expected_size,
                Image.Resampling.NEAREST,
            )

        mask_np = np.asarray(
            mask_image,
            dtype=np.uint8,
        )

    return mask_np > threshold


# ============================================================
# Bounding-box utilities
# ============================================================

def mask_to_bbox(
    mask: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Compute tight bounding box around all foreground bird pixels.

    Returns:
        (x1, y1, x2, y2)

    where x2/y2 are exclusive, so the box can be used directly
    with NumPy slicing:

        image[y1:y2, x1:x2]
    """

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = int(xs.min())
    y1 = int(ys.min())

    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1

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

    x1 = max(
        0,
        min(image_width - 1, x1),
    )

    y1 = max(
        0,
        min(image_height - 1, y1),
    )

    x2 = max(
        0,
        min(image_width, x2),
    )

    y2 = max(
        0,
        min(image_height, y2),
    )

    return x1, y1, x2, y2


# ============================================================
# PURGE-D transformations
# ============================================================

def build_retain_white(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    PURGE-D RETAIN.

    Keep the entire target bounding-box rectangle.
    Everything outside the bounding box becomes white.
    """

    x1, y1, x2, y2 = bbox

    output = np.full_like(
        image,
        255,
    )

    output[y1:y2, x1:x2] = (
        image[y1:y2, x1:x2]
    )

    return output


def build_forget_whitebox(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    PURGE-D FORGET.

    Preserve the original image but replace the complete
    target bounding-box rectangle with white.
    """

    x1, y1, x2, y2 = bbox

    output = image.copy()

    output[y1:y2, x1:x2] = 255

    return output


# ============================================================
# Saving
# ============================================================

def save_np_image(
    image: np.ndarray,
    path: Path,
    resize: int,
) -> None:
    """
    Save a NumPy RGB image.

    If resize > 0, resize to resize x resize.
    """

    ensure_dir(
        str(path.parent)
    )

    pil_image = Image.fromarray(
        image.astype(np.uint8)
    )

    if resize > 0:

        pil_image = pil_image.resize(
            (resize, resize),
            Image.Resampling.BICUBIC,
        )

    pil_image.save(path)


def write_jsonl(
    records: List[Dict],
    path: Path,
) -> None:
    """Write metadata records to JSONL."""

    ensure_dir(
        str(path.parent)
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Sampling
# ============================================================

def select_rows(
    dataframe: pd.DataFrame,
    split_id: int,
    class_id: int,
    number_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """
    Select examples from one split/class.

    number_per_class:
        -1 -> use ALL available examples
         0 -> skip this split
        >0 -> randomly sample N examples
    """

    subset = dataframe[
        (dataframe["split"] == split_id)
        & (dataframe["y"] == class_id)
    ].copy()

    if number_per_class == 0:
        return subset.iloc[0:0].copy()

    if (
        number_per_class < 0
        or number_per_class >= len(subset)
    ):
        return subset.sort_values(
            "img_id"
        ).reset_index(drop=True)

    subset = subset.sample(
        n=number_per_class,
        random_state=seed,
        replace=False,
    )

    return subset.sort_values(
        "img_id"
    ).reset_index(drop=True)


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate bounding-box-based PURGE-D retain/forget "
            "partitions for Waterbirds."
        )
    )

    parser.add_argument(
        "--waterbirds_root",
        required=True,
        type=str,
        help=(
            "Path to waterbird_complete95_forest2water2"
        ),
    )

    parser.add_argument(
        "--segmentations_root",
        required=True,
        type=str,
        help=(
            "Path to the CUB segmentations directory"
        ),
    )

    parser.add_argument(
        "--output_root",
        required=True,
        type=str,
        help="Output PURGE-D directory",
    )

    parser.add_argument(
        "--train_per_class",
        type=int,
        default=-1,
        help=(
            "Number of training images per class. "
            "-1 uses all available images."
        ),
    )

    parser.add_argument(
        "--val_per_class",
        type=int,
        default=-1,
        help=(
            "Number of validation images per class. "
            "-1 uses all available images."
        ),
    )

    parser.add_argument(
        "--test_per_class",
        type=int,
        default=-1,
        help=(
            "Number of test images per class. "
            "-1 uses all available images."
        ),
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=384,
        help=(
            "Output image resolution. "
            "Use 0 to keep original resolution."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random sampling seed",
    )

    parser.add_argument(
        "--class_names",
        type=str,
        default="landbird,waterbird",
        help=(
            "Comma-separated classes. "
            "Options: landbird,waterbird"
        ),
    )

    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=127,
        help=(
            "Threshold used to convert CUB segmentation "
            "to a binary foreground mask."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # Reproducibility
    # ========================================================

    random.seed(args.seed)
    np.random.seed(args.seed)

    # ========================================================
    # Paths
    # ========================================================

    waterbirds_root = Path(
        args.waterbirds_root
    )

    segmentations_root = Path(
        args.segmentations_root
    )

    output_root = Path(
        args.output_root
    )

    metadata_path = (
        waterbirds_root
        / "metadata.csv"
    )

    # ========================================================
    # Validate paths
    # ========================================================

    if not waterbirds_root.is_dir():

        raise FileNotFoundError(
            f"Waterbirds directory does not exist:\n"
            f"{waterbirds_root}"
        )

    if not metadata_path.is_file():

        raise FileNotFoundError(
            f"metadata.csv does not exist:\n"
            f"{metadata_path}"
        )

    if not segmentations_root.is_dir():

        raise FileNotFoundError(
            f"Segmentation directory does not exist:\n"
            f"{segmentations_root}"
        )

    ensure_dir(
        str(output_root)
    )

    # ========================================================
    # Load metadata
    # ========================================================

    dataframe = pd.read_csv(
        metadata_path
    )

    required_columns = {
        "img_id",
        "img_filename",
        "y",
        "split",
        "place",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:

        raise ValueError(
            "metadata.csv is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    # ========================================================
    # Classes
    # ========================================================

    requested_classes = {
        name.strip()
        for name in args.class_names.split(",")
        if name.strip()
    }

    unknown_classes = (
        requested_classes
        - set(CLASS_IDS.keys())
    )

    if unknown_classes:

        raise ValueError(
            "Unknown class names: "
            f"{sorted(unknown_classes)}"
        )

    if not requested_classes:

        raise ValueError(
            "No classes were selected."
        )

    # ========================================================
    # Sampling settings
    # ========================================================

    split_counts = {
        0: args.train_per_class,
        1: args.val_per_class,
        2: args.test_per_class,
    }

    # ========================================================
    # Print configuration
    # ========================================================

    print()
    print("=" * 72)
    print("             PURGE-D Waterbirds Generation")
    print("=" * 72)

    print(
        f"Waterbirds       : {waterbirds_root}"
    )

    print(
        f"Segmentations    : {segmentations_root}"
    )

    print(
        f"Output           : {output_root}"
    )

    print(
        f"Classes          : "
        f"{', '.join(sorted(requested_classes))}"
    )

    print(
        f"Train / class    : {args.train_per_class}"
    )

    print(
        f"Val / class      : {args.val_per_class}"
    )

    print(
        f"Test / class     : {args.test_per_class}"
    )

    print(
        f"Resize           : {args.resize}"
    )

    print(
        f"Seed             : {args.seed}"
    )

    print(
        f"Mask threshold   : {args.mask_threshold}"
    )

    print("=" * 72)
    print()

    # ========================================================
    # Generate PURGE-D
    # ========================================================

    metadata_records: List[Dict] = []

    total_retain = 0
    total_forget = 0

    missing_images = 0
    missing_masks = 0
    empty_masks = 0

    for split_id, split_name in SPLIT_NAMES.items():

        count = split_counts[
            split_id
        ]

        if count == 0:
            continue

        for class_name in [
            "landbird",
            "waterbird",
        ]:

            if class_name not in requested_classes:
                continue

            class_id = CLASS_IDS[
                class_name
            ]

            sample_seed = (
                args.seed
                + split_id * 1000
                + class_id * 100
            )

            selected_rows = select_rows(
                dataframe=dataframe,
                split_id=split_id,
                class_id=class_id,
                number_per_class=count,
                seed=sample_seed,
            )

            available_count = len(
                dataframe[
                    (dataframe["split"] == split_id)
                    & (dataframe["y"] == class_id)
                ]
            )

            if (
                count > 0
                and count > available_count
            ):

                print(
                    f"[warning] "
                    f"{split_name}/{class_name}: "
                    f"requested {count}, "
                    f"but only {available_count} exist."
                )

            print(
                f"[{split_name}]"
                f"[{class_name}] "
                f"selected={len(selected_rows)}"
            )

            saved_retain = 0
            saved_forget = 0

            for _, row in tqdm(
                selected_rows.iterrows(),
                total=len(selected_rows),
                desc=(
                    f"{split_name}/{class_name}"
                ),
            ):

                # ============================================
                # Source image
                # ============================================

                relative_image_path = Path(
                    str(row["img_filename"])
                )

                image_path = (
                    waterbirds_root
                    / relative_image_path
                )

                # ============================================
                # Corresponding CUB segmentation
                # ============================================

                relative_mask_path = (
                    relative_image_path.with_suffix(
                        ".png"
                    )
                )

                mask_path = (
                    segmentations_root
                    / relative_mask_path
                )

                if not image_path.is_file():

                    print(
                        f"[missing image] "
                        f"{image_path}"
                    )

                    missing_images += 1
                    continue

                if not mask_path.is_file():

                    print(
                        f"[missing mask] "
                        f"{mask_path}"
                    )

                    missing_masks += 1
                    continue

                # ============================================
                # Load Waterbirds image
                # ============================================

                try:

                    image = load_rgb_image(
                        image_path
                    )

                except Exception as error:

                    print(
                        f"[image error] "
                        f"{image_path}: "
                        f"{error}"
                    )

                    missing_images += 1
                    continue

                image_height, image_width = (
                    image.shape[:2]
                )

                # ============================================
                # Load segmentation mask
                # ============================================

                try:

                    bird_mask = load_binary_mask(
                        path=mask_path,
                        target_width=image_width,
                        target_height=image_height,
                        threshold=args.mask_threshold,
                    )

                except Exception as error:

                    print(
                        f"[mask error] "
                        f"{mask_path}: "
                        f"{error}"
                    )

                    missing_masks += 1
                    continue

                # ============================================
                # Segmentation -> tight bounding box
                # ============================================

                bbox = mask_to_bbox(
                    bird_mask
                )

                if bbox is None:

                    print(
                        f"[empty mask] "
                        f"{mask_path}"
                    )

                    empty_masks += 1
                    continue

                x1, y1, x2, y2 = (
                    clamp_bbox(
                        *bbox,
                        image_width,
                        image_height,
                    )
                )

                if (
                    x2 <= x1
                    or y2 <= y1
                ):

                    print(
                        f"[invalid bbox] "
                        f"{mask_path}"
                    )

                    empty_masks += 1
                    continue

                bbox = (
                    x1,
                    y1,
                    x2,
                    y2,
                )

                # ============================================
                # PURGE-D retain
                # ============================================

                retain_image = (
                    build_retain_white(
                        image,
                        bbox,
                    )
                )

                # ============================================
                # PURGE-D forget
                # ============================================

                forget_image = (
                    build_forget_whitebox(
                        image,
                        bbox,
                    )
                )

                # ============================================
                # Metadata
                # ============================================

                img_id = int(
                    row["img_id"]
                )

                place_id = int(
                    row["place"]
                )

                place_name = PLACE_NAMES[
                    place_id
                ]

                group_name = (
                    f"{class_name}_"
                    f"{place_name}"
                )

                species_folder = (
                    relative_image_path.parent.name
                )

                source_stem = (
                    relative_image_path.stem
                )

                output_base = (
                    f"{img_id:06d}_"
                    f"{source_stem}"
                )

                # ============================================
                # Output directories
                # ============================================

                retain_path = (
                    output_root
                    / "images"
                    / split_name
                    / "retain"
                    / class_name
                    / f"{output_base}_retain.png"
                )

                forget_path = (
                    output_root
                    / "images"
                    / split_name
                    / "forget"
                    / class_name
                    / f"{output_base}_forget.png"
                )

                # ============================================
                # Save images
                # ============================================

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

                total_retain += 1
                total_forget += 1

                # ============================================
                # Metadata record
                # ============================================

                record = {
                    "img_id":
                        img_id,

                    "source_file":
                        str(
                            relative_image_path
                        ),

                    "segmentation_file":
                        str(
                            relative_mask_path
                        ),

                    "species_folder":
                        species_folder,

                    "split_id":
                        split_id,

                    "split":
                        split_name,

                    "y":
                        class_id,

                    "class_name":
                        class_name,

                    "place":
                        place_id,

                    "background":
                        place_name,

                    "group":
                        group_name,

                    "original_width":
                        int(image_width),

                    "original_height":
                        int(image_height),

                    "bbox_xyxy": [
                        int(x1),
                        int(y1),
                        int(x2),
                        int(y2),
                    ],

                    "bbox_xywh": [
                        int(x1),
                        int(y1),
                        int(x2 - x1),
                        int(y2 - y1),
                    ],

                    "retain_path":
                        str(
                            retain_path.relative_to(
                                output_root
                            )
                        ),

                    "forget_path":
                        str(
                            forget_path.relative_to(
                                output_root
                            )
                        ),
                }

                if (
                    "place_filename"
                    in dataframe.columns
                    and pd.notna(
                        row.get(
                            "place_filename"
                        )
                    )
                ):

                    record[
                        "place_filename"
                    ] = str(
                        row["place_filename"]
                    )

                metadata_records.append(
                    record
                )

            print(
                f"[PURGE-D]"
                f"[{split_name}]"
                f"[{class_name}] "
                f"retain={saved_retain} | "
                f"forget={saved_forget}"
            )

    # ========================================================
    # Save metadata
    # ========================================================

    metadata_output_path = (
        output_root
        / "metadata.jsonl"
    )

    write_jsonl(
        metadata_records,
        metadata_output_path,
    )

    # ========================================================
    # Final summary
    # ========================================================

    print()
    print("=" * 72)
    print("PURGE-D Waterbirds generation complete")
    print("=" * 72)

    print(
        f"Retain images  : {total_retain}"
    )

    print(
        f"Forget images  : {total_forget}"
    )

    print(
        f"Missing images : {missing_images}"
    )

    print(
        f"Missing masks  : {missing_masks}"
    )

    print(
        f"Empty masks    : {empty_masks}"
    )

    print(
        f"Metadata       : "
        f"{metadata_output_path}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()