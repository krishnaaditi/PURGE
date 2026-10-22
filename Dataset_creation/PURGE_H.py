#!/usr/bin/env python3
"""
PURGE-H: LaMa/SimpleLama-only counterfactual construction.

OUTPUT LAYOUT
-------------
OUTPUT/
├── retain/
│   ├── original_yes/
│   │   ├── apple/
│   │   ├── airplane/
│   │   └── ...
│   └── edited_no/
│       ├── apple/
│       ├── airplane/
│       └── ...
├── forget/
│   ├── original_no/
│   │   ├── apple/
│   │   ├── airplane/
│   │   └── ...
│   └── edited_yes/
│       ├── apple/
│       ├── airplane/
│       └── ...
├── preview/
│   ├── apple/
│   ├── airplane/
│   └── ...
├── metadata.jsonl
├── rejections.jsonl
└── summary.json

PURGE-H ROUTING
---------------
Original image has the object (GT = YES):
    model YES -> retain/original_yes
    model NO  -> forget/original_no

Edited image has the target removed (GT = NO):
    model NO  -> retain/edited_no
    model YES -> forget/edited_yes

INPAINTING
----------
The inpainting backend is ONLY LaMa through SimpleLama.

For each target instance:
  1. exact COCO segmentation mask
  2. paper-style 13x13 elliptical dilation
  3. 5x5 Gaussian smoothing + threshold
  4. 100% full-mask contextual crop (reflection padded if needed)
  5. LaMa removal
  6. try multiple context sizes / small adaptive halos
  7. compare several local blends of the same LaMa output
  8. optional second LaMa refinement if first-stage QC is not clean
  9. select the most natural local-background candidate
 10. reject only after retry/refinement if the result is still poor

No OmniPaint, PowerPaint, diffusion inpainting, Telea, or Navier-Stokes.

CLASS SELECTION
---------------
--class_names apple
--class_names "apple,airplane"
omit --class_names -> all COCO classes

SIZE MODES
----------
Small test:
    --samples_per_class 10

Global accepted pairs:
    --target_pairs 8000

At least 8000 retain AND 8000 forget images:
    --target_each_partition 8000
"""

import argparse
import json
import math
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from simple_lama_inpainting import SimpleLama
from tqdm import tqdm
from transformers import AutoProcessor, LlavaNextForConditionalGeneration


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
# Basic I/O
# ============================================================

def ensure_dir(path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    return name.replace(" ", "_")


def load_rgb(path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(image_np: np.ndarray, path, save_size: int = 0) -> None:
    path = Path(path)
    ensure_dir(path.parent)

    image = Image.fromarray(image_np.astype(np.uint8), mode="RGB")

    # 0 keeps native resolution.
    if save_size and save_size > 0:
        image = image.resize(
            (save_size, save_size),
            Image.Resampling.LANCZOS,
        )

    image.save(path, quality=96)


def write_jsonl(handle, record: Dict) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


# ============================================================
# Geometry
# ============================================================

def bbox_xyxy(bbox_xywh) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox_xywh
    return (
        int(round(x)),
        int(round(y)),
        int(round(x + w)),
        int(round(y + h)),
    )


def bbox_iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih
    union = aw * ah + bw * bh - inter

    return 0.0 if union <= 0 else float(inter / union)


def target_overlap_fraction(target_bbox, other_bbox) -> float:
    tx, ty, tw, th = target_bbox
    ox, oy, ow, oh = other_bbox

    ix1, iy1 = max(tx, ox), max(ty, oy)
    ix2 = min(tx + tw, ox + ow)
    iy2 = min(ty + th, oy + oh)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    target_area = tw * th

    return (
        0.0
        if target_area <= 0
        else float((iw * ih) / target_area)
    )


def mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def mask_solidity(mask: np.ndarray) -> float:
    binary = ((mask > 0).astype(np.uint8) * 255)
    area = float(np.count_nonzero(binary))

    if area <= 0:
        return 0.0

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return 0.0

    points = np.concatenate(contours, axis=0)

    if len(points) < 3:
        return 0.0

    hull_area = float(
        cv2.contourArea(
            cv2.convexHull(points)
        )
    )

    if hull_area <= 0:
        return 0.0

    return min(1.0, area / hull_area)


# ============================================================
# COCO target instances
# ============================================================

def collect_target_instances(
    coco: COCO,
    image_id: int,
    category_id: int,
    image_width: int,
    image_height: int,
    max_target_overlap: float,
):
    """
    Keep each target object as its own mask.

    Multiple apples / airplanes are removed sequentially. This is much
    more stable than giving LaMa one giant disconnected union mask.
    """

    all_annotations = coco.loadAnns(
        coco.getAnnIds(imgIds=[image_id])
    )

    if len(all_annotations) > 10:
        return None, "too_many_objects"

    target_annotations = coco.loadAnns(
        coco.getAnnIds(
            imgIds=[image_id],
            catIds=[category_id],
        )
    )

    if not target_annotations:
        return None, "no_target_annotation"

    if any(
        int(annotation.get("iscrowd", 0)) == 1
        for annotation in target_annotations
    ):
        return None, "target_contains_crowd"

    image_area = float(image_width * image_height)
    instances = []

    for annotation in target_annotations:

        raw_mask = coco.annToMask(annotation)

        if raw_mask is None:
            return None, "missing_mask"

        raw_mask = (raw_mask > 0).astype(np.uint8)

        area = float(raw_mask.sum())

        if area <= 0:
            return None, "empty_mask"

        area_ratio = area / image_area

        # Paper-style filter.
        if not (0.015 <= area_ratio <= 0.20):
            return None, f"bad_area_ratio:{area_ratio:.5f}"

        solidity = mask_solidity(raw_mask)

        if solidity < 0.40:
            return None, f"low_solidity:{solidity:.5f}"

        bbox = bbox_xyxy(annotation["bbox"])
        x1, y1, x2, y2 = bbox

        # Paper-style boundary filter.
        if (
            x1 < 0.035 * image_width
            or y1 < 0.035 * image_height
            or x2 > 0.965 * image_width
            or y2 > 0.965 * image_height
        ):
            return None, "near_image_boundary"

        max_iou = 0.0
        max_overlap = 0.0

        for other in all_annotations:

            if other["id"] == annotation["id"]:
                continue

            max_iou = max(
                max_iou,
                bbox_iou(
                    annotation["bbox"],
                    other["bbox"],
                ),
            )

            max_overlap = max(
                max_overlap,
                target_overlap_fraction(
                    annotation["bbox"],
                    other["bbox"],
                ),
            )

        if max_iou > 0.08:
            return None, f"high_bbox_iou:{max_iou:.5f}"

        # Extra conservative safeguard for held/heavily-occluded objects.
        if max_overlap > max_target_overlap:
            return None, f"high_target_overlap:{max_overlap:.5f}"

        instances.append(
            {
                "annotation_id":
                    int(annotation["id"]),

                "bbox_xyxy":
                    bbox,

                "bbox_xywh":
                    [float(v) for v in annotation["bbox"]],

                "raw_mask":
                    raw_mask,

                "raw_area":
                    int(raw_mask.sum()),

                "area_ratio":
                    float(area_ratio),

                "solidity":
                    float(solidity),

                "max_bbox_iou":
                    float(max_iou),

                "max_target_overlap":
                    float(max_overlap),
            }
        )

    # Larger target first.
    instances.sort(
        key=lambda item: item["raw_area"],
        reverse=True,
    )

    return instances, None


# ============================================================
# Mask processing
# ============================================================

def paper_mask(raw_mask: np.ndarray) -> np.ndarray:
    """
    Paper-style preprocessing:
      13x13 ellipse dilation
      5x5 Gaussian
      re-binarize
    """

    mask = ((raw_mask > 0).astype(np.uint8) * 255)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (13, 13),
    )

    mask = cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )

    mask = cv2.GaussianBlur(
        mask,
        (5, 5),
        0,
    )

    return ((mask >= 127).astype(np.uint8) * 255)


def adaptive_halo(raw_mask: np.ndarray, level: int) -> int:
    """
    level 0: no extra halo beyond paper mask
    level 1: small edge cleanup
    level 2: stronger rescue pass
    """

    box = mask_bbox(raw_mask)

    if box is None or level <= 0:
        return 0

    x1, y1, x2, y2 = box
    max_dim = max(x2 - x1, y2 - y1)

    if level == 1:
        return int(
            np.clip(
                round(0.025 * max_dim),
                4,
                14,
            )
        )

    return int(
        np.clip(
            round(0.045 * max_dim),
            7,
            24,
        )
    )


def add_halo(mask: np.ndarray, halo_pixels: int) -> np.ndarray:
    if halo_pixels <= 0:
        return mask.copy()

    size = 2 * halo_pixels + 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )

    return cv2.dilate(
        mask,
        kernel,
        iterations=1,
    )


# ============================================================
# Full object contextual crop
# ============================================================

def build_full_mask_crop(
    image: np.ndarray,
    full_mask: np.ndarray,
    context_ratio: float,
    min_side: int = 224,
    max_side: int = 640,
):
    """
    Guarantees 100% mask coverage.

    If the desired square goes outside the source image, reflection
    padding is used. The crop is NEVER shrunk so that part of the
    airplane/apple is cut away.
    """

    image_h, image_w = image.shape[:2]

    box = mask_bbox(full_mask)

    if box is None:
        return None, "empty_mask"

    x1, y1, x2, y2 = box

    mask_w = x2 - x1
    mask_h = y2 - y1

    if mask_w <= 0 or mask_h <= 0:
        return None, "invalid_mask_bbox"

    object_side = max(mask_w, mask_h)

    desired_side = int(
        math.ceil(
            object_side
            * (1.0 + 2.0 * context_ratio)
            / 8.0
        )
        * 8
    )

    desired_side = max(
        desired_side,
        int(math.ceil(object_side / 8.0) * 8),
        min_side,
    )

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    desired_x1 = int(math.floor(cx - desired_side / 2.0))
    desired_y1 = int(math.floor(cy - desired_side / 2.0))
    desired_x2 = desired_x1 + desired_side
    desired_y2 = desired_y1 + desired_side

    real_x1 = max(0, desired_x1)
    real_y1 = max(0, desired_y1)
    real_x2 = min(image_w, desired_x2)
    real_y2 = min(image_h, desired_y2)

    if real_x2 <= real_x1 or real_y2 <= real_y1:
        return None, "invalid_real_crop"

    real_image = image[
        real_y1:real_y2,
        real_x1:real_x2,
    ].copy()

    real_mask = full_mask[
        real_y1:real_y2,
        real_x1:real_x2,
    ].copy()

    full_pixels = int(np.count_nonzero(full_mask))
    crop_pixels = int(np.count_nonzero(real_mask))

    if full_pixels <= 0:
        return None, "empty_mask"

    coverage = crop_pixels / full_pixels

    if coverage < 0.9999:
        return None, f"partial_mask_coverage:{coverage:.6f}"

    pad_left = real_x1 - desired_x1
    pad_top = real_y1 - desired_y1
    pad_right = desired_x2 - real_x2
    pad_bottom = desired_y2 - real_y2

    padded_image = cv2.copyMakeBorder(
        real_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_REFLECT_101,
    )

    padded_mask = cv2.copyMakeBorder(
        real_mask,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    padded_h, padded_w = padded_image.shape[:2]
    square_side = max(padded_h, padded_w)

    if padded_h != square_side or padded_w != square_side:

        padded_image = cv2.copyMakeBorder(
            padded_image,
            0,
            square_side - padded_h,
            0,
            square_side - padded_w,
            borderType=cv2.BORDER_REFLECT_101,
        )

        padded_mask = cv2.copyMakeBorder(
            padded_mask,
            0,
            square_side - padded_h,
            0,
            square_side - padded_w,
            borderType=cv2.BORDER_CONSTANT,
            value=0,
        )

    original_padded_side = padded_image.shape[0]

    work_side = min(
        max_side,
        original_padded_side,
    )

    work_side = max(
        min_side,
        work_side,
    )

    work_side = int(
        math.ceil(
            work_side / 8.0
        )
        * 8
    )

    if work_side != original_padded_side:

        working_image = cv2.resize(
            padded_image,
            (work_side, work_side),
            interpolation=cv2.INTER_LANCZOS4,
        )

        working_mask = cv2.resize(
            padded_mask,
            (work_side, work_side),
            interpolation=cv2.INTER_NEAREST,
        )

    else:

        working_image = padded_image
        working_mask = padded_mask

    info = {
        "real_box": (
            int(real_x1),
            int(real_y1),
            int(real_x2),
            int(real_y2),
        ),

        "real_shape": (
            int(real_image.shape[0]),
            int(real_image.shape[1]),
        ),

        "pad": (
            int(pad_left),
            int(pad_top),
            int(pad_right),
            int(pad_bottom),
        ),

        "original_padded_side":
            int(original_padded_side),

        "work_side":
            int(work_side),

        "mask_coverage":
            float(coverage),
    }

    return (
        (
            real_image,
            real_mask,
            working_image,
            working_mask,
            info,
        ),
        None,
    )


def restore_generated_crop(
    generated_working: np.ndarray,
    crop_bundle,
) -> np.ndarray:

    (
        real_image,
        _,
        _,
        _,
        info,
    ) = crop_bundle

    padded_side = info["original_padded_side"]

    if generated_working.shape[0] != padded_side:

        generated_padded = cv2.resize(
            generated_working,
            (padded_side, padded_side),
            interpolation=cv2.INTER_LANCZOS4,
        )

    else:

        generated_padded = generated_working

    pad_left, pad_top, _, _ = info["pad"]
    real_h, real_w = info["real_shape"]

    generated_real = generated_padded[
        pad_top:pad_top + real_h,
        pad_left:pad_left + real_w,
    ].copy()

    if generated_real.shape != real_image.shape:
        raise RuntimeError(
            "Restored crop shape mismatch: "
            f"{generated_real.shape} vs {real_image.shape}"
        )

    return generated_real


# ============================================================
# LaMa
# ============================================================

def run_lama(
    lama,
    working_image: np.ndarray,
    working_mask: np.ndarray,
) -> np.ndarray:

    image_pil = Image.fromarray(
        working_image
    ).convert("RGB")

    mask_pil = Image.fromarray(
        working_mask
    ).convert("L")

    generated = lama(
        image_pil,
        mask_pil,
    )

    if generated.size != image_pil.size:

        generated = generated.resize(
            image_pil.size,
            Image.Resampling.LANCZOS,
        )

    return np.asarray(
        generated.convert("RGB"),
        dtype=np.uint8,
    )


# ============================================================
# Blending / harmonization
# ============================================================

def get_ring(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:

    binary = (
        (mask > 0)
        .astype(np.uint8)
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            2 * radius + 1,
            2 * radius + 1,
        ),
    )

    expanded = cv2.dilate(
        binary,
        kernel,
        iterations=1,
    )

    return (
        (expanded > 0)
        & (binary == 0)
    )


def feather_composite(
    original_crop: np.ndarray,
    generated_crop: np.ndarray,
    mask_crop: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Keep source pixels outside the local removal zone and softly blend
    the generated fill near the object boundary.
    """

    alpha = (
        mask_crop.astype(np.float32)
        / 255.0
    )

    alpha = cv2.GaussianBlur(
        alpha,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    alpha = np.clip(
        alpha,
        0.0,
        1.0,
    )[..., None]

    output = (
        original_crop.astype(np.float32)
        * (1.0 - alpha)
        +
        generated_crop.astype(np.float32)
        * alpha
    )

    return np.clip(
        output,
        0,
        255,
    ).astype(np.uint8)


def harmonize_generated_fill(
    generated_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> np.ndarray:
    """
    Mild chroma/brightness harmonization candidate.

    This is NOT another inpainting method. It only post-processes the LaMa
    output. It is useful for the kind of small blue/gray patch visible in
    otherwise-good examples.

    The unmodified LaMa output is also kept as a candidate, so this
    correction is used only when it scores better.
    """

    target = mask_crop > 0
    ring = get_ring(mask_crop, 24)

    if (
        not np.any(target)
        or not np.any(ring)
    ):
        return generated_crop.copy()

    lab = cv2.cvtColor(
        generated_crop,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)

    target_mean = lab[target].mean(axis=0)
    ring_median = np.median(
        lab[ring],
        axis=0,
    )

    # Limited shifts: do not force the fill to exactly equal the ring.
    shift = ring_median - target_mean

    shift[0] = np.clip(
        shift[0] * 0.35,
        -14.0,
        14.0,
    )

    shift[1] = np.clip(
        shift[1] * 0.65,
        -18.0,
        18.0,
    )

    shift[2] = np.clip(
        shift[2] * 0.65,
        -18.0,
        18.0,
    )

    corrected = lab.copy()

    for channel in range(3):
        channel_data = corrected[..., channel]
        channel_data[target] += shift[channel]
        corrected[..., channel] = channel_data

    corrected = np.clip(
        corrected,
        0,
        255,
    ).astype(np.uint8)

    return cv2.cvtColor(
        corrected,
        cv2.COLOR_LAB2RGB,
    )


def paste_crop(
    full_image: np.ndarray,
    crop: np.ndarray,
    info: Dict,
) -> np.ndarray:

    x1, y1, x2, y2 = info["real_box"]

    output = full_image.copy()

    output[
        y1:y2,
        x1:x2,
    ] = crop

    return output


# ============================================================
# Quality score
# ============================================================

def boundary_change(
    original_crop: np.ndarray,
    generated_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> float:
    ring = get_ring(
        mask_crop,
        radius=5,
    )

    if not np.any(ring):
        return 0.0

    difference = np.abs(
        original_crop.astype(np.float32)
        - generated_crop.astype(np.float32)
    ).mean(axis=2)

    return float(
        difference[ring].mean()
    )


def naturalness_score(
    candidate_crop: np.ndarray,
    mask_crop: np.ndarray,
) -> Optional[Dict]:
    """
    Lower = better.

    Main purpose:
      rank several LaMa-only candidates and reject catastrophic fills.

    Metrics:
      local color continuity
      brightness continuity
      local texture continuity
      gradient continuity
      seam continuity
      flat-blob penalty
    """

    target = mask_crop > 0
    ring6 = get_ring(mask_crop, 6)
    ring24 = get_ring(mask_crop, 24)

    if (
        not np.any(target)
        or not np.any(ring24)
    ):
        return None

    lab = cv2.cvtColor(
        candidate_crop,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)

    gray = cv2.cvtColor(
        candidate_crop,
        cv2.COLOR_RGB2GRAY,
    ).astype(np.float32)

    target_lab = lab[target].mean(axis=0)
    ring_lab = np.median(
        lab[ring24],
        axis=0,
    )

    color_delta = float(
        np.linalg.norm(
            target_lab
            - ring_lab
        )
    )

    brightness_delta = float(
        abs(
            gray[target].mean()
            - np.median(
                gray[ring24]
            )
        )
    )

    target_std = float(
        gray[target].std()
    )

    ring_std = float(
        gray[ring24].std()
    )

    texture_delta = float(
        abs(
            target_std
            - ring_std
        )
    )

    gx = cv2.Sobel(
        gray,
        cv2.CV_32F,
        1,
        0,
        ksize=3,
    )

    gy = cv2.Sobel(
        gray,
        cv2.CV_32F,
        0,
        1,
        ksize=3,
    )

    gradient = np.sqrt(
        gx * gx
        + gy * gy
    )

    gradient_delta = float(
        abs(
            gradient[target].mean()
            - gradient[ring24].mean()
        )
    )

    eroded = cv2.erode(
        target.astype(np.uint8),
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
    )

    inner_edge = target & (eroded == 0)

    seam_delta = 0.0

    if (
        np.any(inner_edge)
        and np.any(ring6)
    ):
        seam_delta = float(
            np.linalg.norm(
                lab[inner_edge].mean(axis=0)
                - lab[ring6].mean(axis=0)
            )
        )

    # Stronger flat patch penalty than previous version.
    flat_penalty = max(
        0.0,
        ring_std * 0.60
        - target_std,
    )

    total = (
        0.45 * color_delta
        + 0.30 * brightness_delta
        + 0.32 * texture_delta
        + 0.20 * gradient_delta
        + 0.45 * seam_delta
        + 1.20 * flat_penalty
    )

    return {
        "color_delta":
            float(color_delta),

        "brightness_delta":
            float(brightness_delta),

        "texture_delta":
            float(texture_delta),

        "gradient_delta":
            float(gradient_delta),

        "seam_delta":
            float(seam_delta),

        "flat_penalty":
            float(flat_penalty),

        "total_score":
            float(total),
    }


def strict_clean(
    quality: Dict,
    max_boundary_change: float,
    strict_score: float,
) -> bool:

    return (
        quality["boundary_change"]
        <= max_boundary_change

        and quality["total_score"]
        <= strict_score

        and quality["flat_penalty"]
        <= 18.0

        and quality["brightness_delta"]
        <= 58.0
    )


def fallback_clean(
    quality: Dict,
    max_boundary_change: float,
    fallback_score: float,
) -> bool:

    return (
        quality["boundary_change"]
        <= max_boundary_change

        and quality["total_score"]
        <= fallback_score

        and quality["flat_penalty"]
        <= 30.0

        and quality["brightness_delta"]
        <= 82.0

        and quality["color_delta"]
        <= 115.0
    )


# ============================================================
# Build candidates from one LaMa output
# ============================================================

def candidate_variants(
    original_crop: np.ndarray,
    generated_crop: np.ndarray,
    mask_crop: np.ndarray,
):
    """
    All variants come from the SAME LaMa output.

    raw_soft:
        original LaMa fill, modest feather

    raw_wide:
        original LaMa fill, slightly wider feather

    harmonized:
        mild local LAB correction to remove obvious blue/gray color cast

    The best local-background match wins.
    """

    variants = []

    raw_soft = feather_composite(
        original_crop,
        generated_crop,
        mask_crop,
        sigma=1.4,
    )

    variants.append(
        (
            "raw_soft",
            raw_soft,
        )
    )

    raw_wide = feather_composite(
        original_crop,
        generated_crop,
        mask_crop,
        sigma=2.6,
    )

    variants.append(
        (
            "raw_wide",
            raw_wide,
        )
    )

    harmonized_generated = harmonize_generated_fill(
        generated_crop,
        mask_crop,
    )

    harmonized = feather_composite(
        original_crop,
        harmonized_generated,
        mask_crop,
        sigma=2.0,
    )

    variants.append(
        (
            "harmonized",
            harmonized,
        )
    )

    return variants


# ============================================================
# One LaMa attempt
# ============================================================

def make_lama_candidates(
    image: np.ndarray,
    removal_mask: np.ndarray,
    lama,
    context_ratio: float,
):
    crop_bundle, error = build_full_mask_crop(
        image=image,
        full_mask=removal_mask,
        context_ratio=context_ratio,
        min_side=224,
        max_side=640,
    )

    if crop_bundle is None:
        return [], error

    (
        original_crop,
        mask_crop,
        working_image,
        working_mask,
        info,
    ) = crop_bundle

    if info["mask_coverage"] < 0.9999:
        return [], "mask_coverage_not_100_percent"

    try:

        generated_working = run_lama(
            lama,
            working_image,
            working_mask,
        )

    except Exception as error:

        return (
            [],
            (
                "lama_failure:"
                f"{type(error).__name__}:"
                f"{error}"
            ),
        )

    generated_crop = restore_generated_crop(
        generated_working,
        crop_bundle,
    )

    boundary = boundary_change(
        original_crop,
        generated_crop,
        mask_crop,
    )

    outputs = []

    for variant_name, final_crop in candidate_variants(
        original_crop,
        generated_crop,
        mask_crop,
    ):
        natural = naturalness_score(
            final_crop,
            mask_crop,
        )

        if natural is None:
            continue

        edited = paste_crop(
            image,
            final_crop,
            info,
        )

        quality = {
            **natural,

            "boundary_change":
                float(boundary),

            "context_ratio":
                float(context_ratio),

            "mask_coverage":
                float(
                    info[
                        "mask_coverage"
                    ]
                ),

            "work_side":
                int(
                    info[
                        "work_side"
                    ]
                ),

            "blend_variant":
                variant_name,
        }

        outputs.append(
            (
                edited,
                quality,
            )
        )

    if not outputs:
        return [], "no_scored_variants"

    return outputs, None


# ============================================================
# Remove one object: retry + refine
# ============================================================

def remove_one_instance(
    image: np.ndarray,
    raw_mask: np.ndarray,
    lama,
    max_inpaint_attempts: int,
    max_boundary_change: float,
    strict_score: float,
    fallback_score: float,
):
    max_inpaint_attempts = max(
        1,
        min(
            int(max_inpaint_attempts),
            3,
        ),
    )

    base_mask = paper_mask(
        raw_mask
    )

    configs = [
        (0, 0.40),
        (1, 0.60),
        (2, 0.80),
    ][
        :max_inpaint_attempts
    ]

    candidates = []
    attempt_log = []

    for halo_level, context_ratio in configs:

        halo_pixels = adaptive_halo(
            raw_mask,
            halo_level,
        )

        removal_mask = add_halo(
            base_mask,
            halo_pixels,
        )

        attempt_candidates, error = (
            make_lama_candidates(
                image=image,
                removal_mask=
                    removal_mask,
                lama=lama,
                context_ratio=
                    context_ratio,
            )
        )

        if not attempt_candidates:

            attempt_log.append(
                {
                    "stage":
                        "primary",
                    "halo_level":
                        halo_level,
                    "halo_pixels":
                        halo_pixels,
                    "context_ratio":
                        context_ratio,
                    "reason":
                        error,
                }
            )

            continue

        for candidate, quality in attempt_candidates:

            quality = {
                **quality,

                "halo_level":
                    int(halo_level),

                "halo_pixels":
                    int(halo_pixels),

                "refined":
                    False,

                "quality_tier":
                    "candidate",
            }

            candidates.append(
                (
                    quality[
                        "total_score"
                    ],
                    candidate,
                    quality,
                    removal_mask,
                )
            )

    if not candidates:

        return (
            None,
            None,
            "all_primary_lama_attempts_failed",
            attempt_log,
        )

    candidates.sort(
        key=lambda item:
            item[0]
    )

    # --------------------------------------------------------
    # Check 1: strict accepted candidate.
    # --------------------------------------------------------

    for (
        _,
        candidate,
        quality,
        removal_mask,
    ) in candidates:

        if strict_clean(
            quality,
            max_boundary_change,
            strict_score,
        ):

            quality[
                "quality_tier"
            ] = "strict_clean"

            return (
                candidate,
                quality,
                None,
                attempt_log,
            )

    # --------------------------------------------------------
    # Check 2: refine the best candidate with LaMa once more.
    # --------------------------------------------------------

    (
        _,
        best_candidate,
        best_quality,
        best_mask,
    ) = candidates[0]

    refine_mask = add_halo(
        best_mask,
        4,
    )

    refine_context = min(
        0.95,
        best_quality[
            "context_ratio"
        ]
        + 0.15,
    )

    refined_candidates, refine_error = (
        make_lama_candidates(
            image=best_candidate,
            removal_mask=
                refine_mask,
            lama=lama,
            context_ratio=
                refine_context,
        )
    )

    refined_scored = []

    for refined, refined_quality in refined_candidates:

        refined_quality = {
            **refined_quality,

            "halo_level":
                int(
                    best_quality[
                        "halo_level"
                    ]
                ),

            "halo_pixels":
                int(
                    best_quality[
                        "halo_pixels"
                    ]
                    + 4
                ),

            "refined":
                True,

            "quality_tier":
                "candidate",
        }

        refined_scored.append(
            (
                refined_quality[
                    "total_score"
                ],
                refined,
                refined_quality,
            )
        )

    if refined_scored:

        refined_scored.sort(
            key=lambda item:
                item[0]
        )

        best_refined_score, best_refined, best_refined_quality = (
            refined_scored[0]
        )

        if (
            best_refined_score
            < best_quality[
                "total_score"
            ]
        ):

            final_candidate = best_refined
            final_quality = best_refined_quality

        else:

            final_candidate = best_candidate
            final_quality = best_quality

    else:

        attempt_log.append(
            {
                "stage":
                    "refinement",
                "reason":
                    refine_error,
            }
        )

        final_candidate = best_candidate
        final_quality = best_quality

    if fallback_clean(
        final_quality,
        max_boundary_change,
        fallback_score,
    ):

        final_quality[
            "quality_tier"
        ] = "fallback_clean"

        return (
            final_candidate,
            final_quality,
            None,
            attempt_log,
        )

    return (
        None,
        final_quality,
        "failed_second_naturalness_check",
        attempt_log,
    )


# ============================================================
# Remove all target instances
# ============================================================

def remove_all_instances(
    original: np.ndarray,
    instances: List[Dict],
    lama,
    max_inpaint_attempts: int,
    max_boundary_change: float,
    strict_score: float,
    fallback_score: float,
):
    edited = original.copy()
    quality_records = []

    for index, instance in enumerate(
        instances,
        start=1,
    ):

        (
            next_image,
            quality,
            error,
            attempt_log,
        ) = remove_one_instance(
            image=edited,
            raw_mask=
                instance[
                    "raw_mask"
                ],
            lama=lama,
            max_inpaint_attempts=
                max_inpaint_attempts,
            max_boundary_change=
                max_boundary_change,
            strict_score=
                strict_score,
            fallback_score=
                fallback_score,
        )

        if next_image is None:

            return (
                None,
                quality_records,
                (
                    f"instance_{index}_"
                    f"inpainting_failed:"
                    f"{error}"
                ),
                {
                    "failed_instance":
                        index,

                    "last_quality":
                        quality,

                    "attempt_log":
                        attempt_log,
                },
            )

        quality_records.append(
            {
                "instance_index":
                    int(index),

                "annotation_id":
                    int(
                        instance[
                            "annotation_id"
                        ]
                    ),

                **quality,
            }
        )

        edited = next_image

    return (
        edited,
        quality_records,
        None,
        None,
    )


# ============================================================
# LLaVA
# ============================================================

def parse_yes_no(text: str) -> str:

    match = re.search(
        r"\b(yes|no)\b",
        text.strip(),
        flags=re.IGNORECASE,
    )

    return (
        "unknown"
        if match is None
        else match.group(1).lower()
    )


@torch.inference_mode()
def ask_llava(
    model,
    processor,
    image: Image.Image,
    class_name: str,
    device: str,
):
    question = (
        f"Is there a "
        f"{class_name} "
        f"in the image?"
    )

    messages = [
        {
            "role":
                "user",

            "content": [
                {
                    "type":
                        "image"
                },

                {
                    "type":
                        "text",

                    "text":
                        question
                },
            ],
        }
    ]

    try:

        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    except Exception:

        prompt = (
            f"<image>\n"
            f"{question}"
        )

    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt",
    )

    inputs = {
        key:
            (
                value.to(
                    device
                )
                if torch.is_tensor(
                    value
                )
                else value
            )

        for key, value
        in inputs.items()
    }

    outputs = model.generate(
        **inputs,
        max_new_tokens=8,
        do_sample=False,
        use_cache=True,
    )

    prompt_length = (
        inputs[
            "input_ids"
        ]
        .shape[-1]
    )

    generated = outputs[
        :,
        prompt_length:
    ]

    raw = processor.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()

    return (
        parse_yes_no(raw),
        raw,
    )


# ============================================================
# PURGE-H routes
# ============================================================

def route_original(prediction: str):
    if prediction == "yes":
        return (
            "retain",
            "original_yes",
        )

    return (
        "forget",
        "original_no",
    )


def route_edited(prediction: str):
    if prediction == "no":
        return (
            "retain",
            "edited_no",
        )

    return (
        "forget",
        "edited_yes",
    )


# ============================================================
# Output / preview
# ============================================================

def save_partition_image(
    image: np.ndarray,
    output_root: Path,
    partition: str,
    group: str,
    class_name: str,
    image_id: int,
    view: str,
    save_size: int,
):
    class_dir = safe_name(
        class_name
    )

    filename = (
        f"{image_id:012d}_"
        f"{class_dir}_"
        f"{view}.jpg"
    )

    path = (
        output_root
        / partition
        / group
        / class_dir
        / filename
    )

    save_rgb(
        image,
        path,
        save_size,
    )

    return path


def union_raw_masks(
    instances: List[Dict],
    height: int,
    width: int,
):
    union = np.zeros(
        (height, width),
        dtype=np.uint8,
    )

    for instance in instances:

        union = np.maximum(
            union,
            instance[
                "raw_mask"
            ],
        )

    return union


def make_mask_overlay(
    image: np.ndarray,
    union_mask: np.ndarray,
    class_name: str,
    instance_count: int,
):
    output = image.copy()
    mask = union_mask > 0

    output[
        mask
    ] = (
        0.30
        * output[
            mask
        ]
        +
        0.70
        * np.array(
            [
                255,
                0,
                0,
            ]
        )
    ).astype(np.uint8)

    pil = Image.fromarray(
        output
    )

    draw = ImageDraw.Draw(
        pil
    )

    box = mask_bbox(
        union_mask
    )

    if box is not None:

        x1, y1, x2, y2 = box

        draw.rectangle(
            (x1, y1, x2, y2),
            outline="red",
            width=4,
        )

        draw.text(
            (
                x1 + 4,
                max(2, y1 - 15),
            ),
            (
                f"{class_name} "
                f"x{instance_count}"
            ),
            fill="red",
        )

    return np.asarray(
        pil
    )


def fit_panel(
    image_np: np.ndarray,
    width: int = 600,
    height: int = 360,
):
    image = Image.fromarray(
        image_np.astype(np.uint8)
    )

    image.thumbnail(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "white",
    )

    canvas.paste(
        image,
        (
            (width - image.width) // 2,
            (height - image.height) // 2,
        ),
    )

    return canvas


def save_preview(
    original: np.ndarray,
    edited: np.ndarray,
    instances: List[Dict],
    class_name: str,
    original_prediction: str,
    edited_prediction: str,
    quality_records: List[Dict],
    output_path: Path,
):
    h, w = original.shape[:2]

    union_mask = union_raw_masks(
        instances,
        h,
        w,
    )

    overlay = make_mask_overlay(
        original,
        union_mask,
        class_name,
        len(instances),
    )

    preview = Image.new(
        "RGB",
        (1200, 860),
        "white",
    )

    preview.paste(
        fit_panel(original),
        (0, 65),
    )

    preview.paste(
        fit_panel(overlay),
        (600, 65),
    )

    preview.paste(
        fit_panel(edited),
        (0, 495),
    )

    draw = ImageDraw.Draw(
        preview
    )

    for box in [
        (0, 0, 599, 429),
        (600, 0, 1199, 429),
        (0, 430, 599, 859),
        (600, 430, 1199, 859),
    ]:

        draw.rectangle(
            box,
            outline="black",
            width=2,
        )

    draw.text(
        (12, 10),
        "1. ORIGINAL",
        fill="black",
    )

    draw.text(
        (12, 35),
        (
            "GT: YES | "
            f"LLaVA: "
            f"{original_prediction.upper()}"
        ),
        fill="black",
    )

    draw.text(
        (612, 10),
        "2. COMPLETE COCO TARGET",
        fill="black",
    )

    draw.text(
        (612, 35),
        (
            f"Class: {class_name} | "
            f"Instances: {len(instances)}"
        ),
        fill="black",
    )

    draw.text(
        (12, 440),
        "3. NATURAL LaMa REMOVAL",
        fill="black",
    )

    draw.text(
        (12, 465),
        (
            "GT: NO | "
            f"LLaVA: "
            f"{edited_prediction.upper()}"
        ),
        fill="black",
    )

    (
        original_partition,
        original_group,
    ) = route_original(
        original_prediction
    )

    (
        edited_partition,
        edited_group,
    ) = route_edited(
        edited_prediction
    )

    transition = (
        original_prediction[
            0
        ].upper()
        +
        edited_prediction[
            0
        ].upper()
    )

    lines = [
        "4. PURGE-H RESULT",
        "",
        f"Target: {class_name}",
        (
            "Original -> "
            f"{original_partition}/"
            f"{original_group}"
        ),
        (
            "Edited -> "
            f"{edited_partition}/"
            f"{edited_group}"
        ),
        f"Transition: {transition}",
        f"Removed objects: {len(instances)}",
        "",
    ]

    for quality in quality_records:

        lines.extend(
            [
                (
                    f"Object "
                    f"{quality['instance_index']}:"
                ),
                (
                    " tier="
                    f"{quality['quality_tier']}"
                ),
                (
                    " variant="
                    f"{quality['blend_variant']}"
                ),
                (
                    " coverage="
                    f"{quality['mask_coverage'] * 100:.2f}%"
                ),
                (
                    " natural="
                    f"{quality['total_score']:.2f}"
                ),
            ]
        )

    y = 450

    for line in lines[:17]:

        draw.text(
            (625, y),
            line,
            fill="black",
        )

        y += 23

    ensure_dir(
        output_path.parent
    )

    preview.save(
        output_path,
        quality=96,
    )


# ============================================================
# Process one source image
# ============================================================

def process_one_source(
    *,
    args,
    coco,
    model,
    processor,
    lama,
    output_root,
    metadata_file,
    rejection_file,
    class_name,
    category_id,
    image_id,
    preview_index,
):
    image_info = (
        coco.loadImgs(
            [image_id]
        )[0]
    )

    source_path = (
        Path(
            args.coco_image_root
        )
        / image_info[
            "file_name"
        ]
    )

    if not source_path.is_file():
        return None

    try:

        original = load_rgb(
            source_path
        )

    except Exception as error:

        write_jsonl(
            rejection_file,
            {
                "class_name":
                    class_name,

                "image_id":
                    int(image_id),

                "reason":
                    (
                        "image_read_error:"
                        f"{error}"
                    ),
            },
        )

        return None

    image_h, image_w = (
        original.shape[:2]
    )

    (
        instances,
        instance_error,
    ) = collect_target_instances(
        coco=coco,
        image_id=image_id,
        category_id=category_id,
        image_width=image_w,
        image_height=image_h,
        max_target_overlap=
            args.max_target_overlap,
    )

    if instances is None:

        write_jsonl(
            rejection_file,
            {
                "class_name":
                    class_name,

                "image_id":
                    int(image_id),

                "reason":
                    instance_error,
            },
        )

        return None

    # --------------------------------------------------------
    # ORIGINAL: GT YES.
    # --------------------------------------------------------

    (
        original_prediction,
        original_raw,
    ) = ask_llava(
        model=model,
        processor=processor,
        image=Image.fromarray(
            original
        ),
        class_name=
            class_name,
        device=
            args.device,
    )

    if original_prediction == "unknown":

        write_jsonl(
            rejection_file,
            {
                "class_name":
                    class_name,

                "image_id":
                    int(image_id),

                "reason":
                    "unknown_original_prediction",

                "raw":
                    original_raw,
            },
        )

        return None

    # --------------------------------------------------------
    # LaMa removal.
    # --------------------------------------------------------

    (
        edited,
        quality_records,
        removal_error,
        removal_details,
    ) = remove_all_instances(
        original=original,
        instances=instances,
        lama=lama,
        max_inpaint_attempts=
            args.max_inpaint_attempts,
        max_boundary_change=
            args.max_boundary_change,
        strict_score=
            args.strict_natural_score,
        fallback_score=
            args.fallback_natural_score,
    )

    if edited is None:

        write_jsonl(
            rejection_file,
            {
                "class_name":
                    class_name,

                "image_id":
                    int(image_id),

                "reason":
                    removal_error,

                "details":
                    removal_details,

                "accepted_instance_qc":
                    quality_records,
            },
        )

        return None

    # --------------------------------------------------------
    # EDITED: GT NO.
    # A YES is still valid: that is PURGE-H hallucination.
    # --------------------------------------------------------

    (
        edited_prediction,
        edited_raw,
    ) = ask_llava(
        model=model,
        processor=processor,
        image=Image.fromarray(
            edited
        ),
        class_name=
            class_name,
        device=
            args.device,
    )

    if edited_prediction == "unknown":

        write_jsonl(
            rejection_file,
            {
                "class_name":
                    class_name,

                "image_id":
                    int(image_id),

                "reason":
                    "unknown_edited_prediction",

                "raw":
                    edited_raw,
            },
        )

        return None

    (
        original_partition,
        original_group,
    ) = route_original(
        original_prediction
    )

    (
        edited_partition,
        edited_group,
    ) = route_edited(
        edited_prediction
    )

    original_path = save_partition_image(
        image=original,
        output_root=
            output_root,
        partition=
            original_partition,
        group=
            original_group,
        class_name=
            class_name,
        image_id=
            image_id,
        view=
            "original",
        save_size=
            args.save_size,
    )

    edited_path = save_partition_image(
        image=edited,
        output_root=
            output_root,
        partition=
            edited_partition,
        group=
            edited_group,
        class_name=
            class_name,
        image_id=
            image_id,
        view=
            "edited",
        save_size=
            args.save_size,
    )

    transition = (
        original_prediction[
            0
        ].upper()
        +
        edited_prediction[
            0
        ].upper()
    )

    preview_path = None

    if preview_index < args.preview_count:

        preview_path = (
            output_root
            / "preview"
            / safe_name(
                class_name
            )
            / (
                f"{image_id:012d}_"
                f"{safe_name(class_name)}_"
                f"{transition}.jpg"
            )
        )

        save_preview(
            original=original,
            edited=edited,
            instances=instances,
            class_name=
                class_name,
            original_prediction=
                original_prediction,
            edited_prediction=
                edited_prediction,
            quality_records=
                quality_records,
            output_path=
                preview_path,
        )

    clean_instances = [
        {
            key: value
            for key, value
            in instance.items()
            if key != "raw_mask"
        }
        for instance
        in instances
    ]

    write_jsonl(
        metadata_file,
        {
            "class_name":
                class_name,

            "category_id":
                int(category_id),

            "image_id":
                int(image_id),

            "source_file":
                image_info[
                    "file_name"
                ],

            "original_gt":
                "yes",

            "original_prediction":
                original_prediction,

            "original_raw":
                original_raw,

            "original_partition":
                original_partition,

            "original_group":
                original_group,

            "edited_gt":
                "no",

            "edited_prediction":
                edited_prediction,

            "edited_raw":
                edited_raw,

            "edited_partition":
                edited_partition,

            "edited_group":
                edited_group,

            "transition":
                transition,

            "num_removed_instances":
                len(
                    instances
                ),

            "inpainting_backend":
                "LaMa/SimpleLama",

            "instance_qc":
                quality_records,

            "instances":
                clean_instances,

            "original_path":
                str(
                    original_path
                    .relative_to(
                        output_root
                    )
                ),

            "edited_path":
                str(
                    edited_path
                    .relative_to(
                        output_root
                    )
                ),

            "preview_path":
                (
                    str(
                        preview_path
                        .relative_to(
                            output_root
                        )
                    )
                    if preview_path
                    else None
                ),
        },
    )

    retain_added = (
        int(
            original_partition
            == "retain"
        )
        +
        int(
            edited_partition
            == "retain"
        )
    )

    forget_added = (
        2
        - retain_added
    )

    print()
    print(
        f"[ACCEPT]"
        f"[{class_name}] "
        f"id={image_id} "
        f"transition={transition}"
    )

    print(
        f"  Original -> "
        f"{original_partition}/"
        f"{original_group}"
    )

    print(
        f"  Edited   -> "
        f"{edited_partition}/"
        f"{edited_group}"
    )

    for quality in quality_records:

        print(
            "  object "
            f"{quality['instance_index']}: "
            f"{quality['quality_tier']} | "
            f"variant="
            f"{quality['blend_variant']} | "
            f"coverage="
            f"{quality['mask_coverage'] * 100:.2f}% | "
            f"natural="
            f"{quality['total_score']:.2f}"
        )

    return {
        "retain_added":
            int(retain_added),

        "forget_added":
            int(forget_added),

        "transition":
            transition,

        "original_group":
            original_group,

        "edited_group":
            edited_group,
    }


# ============================================================
# Global goal
# ============================================================

def global_goal_met(
    args,
    accepted_pairs: int,
    retain_count: int,
    forget_count: int,
) -> bool:

    if args.target_each_partition > 0:

        return (
            retain_count
            >= args.target_each_partition
            and
            forget_count
            >= args.target_each_partition
        )

    if args.target_pairs > 0:

        return (
            accepted_pairs
            >= args.target_pairs
        )

    return False


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--coco_image_root",
        required=True,
    )

    parser.add_argument(
        "--coco_ann_json",
        required=True,
    )

    parser.add_argument(
        "--output_root",
        required=True,
    )

    parser.add_argument(
        "--model_path",
        required=True,
    )

    parser.add_argument(
        "--class_names",
        default="",
        help=(
            'One: "apple"; '
            'multiple: "apple,airplane"; '
            'omit for all 80.'
        ),
    )

    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--target_pairs",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--target_each_partition",
        type=int,
        default=0,
        help=(
            "Example: 8000 means continue until "
            "retain >=8000 AND forget >=8000."
        ),
    )

    parser.add_argument(
        "--max_per_class",
        type=int,
        default=0,
        help=(
            "Global mode only; 0 = unlimited."
        ),
    )

    parser.add_argument(
        "--max_inpaint_attempts",
        type=int,
        choices=[
            1,
            2,
            3,
        ],
        default=2,
    )

    parser.add_argument(
        "--device",
        default="cuda:0",
    )

    parser.add_argument(
        "--lama_device",
        default="cpu",
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "bf16",
            "fp16",
            "fp32",
        ],
        default="bf16",
    )

    parser.add_argument(
        "--save_size",
        type=int,
        default=0,
        help=(
            "0 = native resolution."
        ),
    )

    parser.add_argument(
        "--preview_count",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max_target_overlap",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--max_boundary_change",
        type=float,
        default=45.0,
    )

    parser.add_argument(
        "--strict_natural_score",
        type=float,
        default=84.0,
    )

    parser.add_argument(
        "--fallback_natural_score",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    if (
        args.target_pairs > 0
        and args.target_each_partition > 0
    ):

        raise ValueError(
            "Use either --target_pairs OR "
            "--target_each_partition, not both."
        )

    random.seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    output_root = Path(
        args.output_root
    )

    if (
        output_root.exists()
        and any(
            output_root.iterdir()
        )
    ):

        if args.overwrite:

            shutil.rmtree(
                output_root
            )

        else:

            raise RuntimeError(
                "Output directory is not empty. "
                "Use --overwrite."
            )

    ensure_dir(
        output_root
    )

    # ========================================================
    # COCO
    # ========================================================

    print(
        "Loading COCO annotations..."
    )

    coco = COCO(
        args.coco_ann_json
    )

    categories = coco.loadCats(
        coco.getCatIds()
    )

    name_to_id = {
        item["name"]:
            item["id"]
        for item
        in categories
    }

    # ========================================================
    # Classes
    # ========================================================

    if args.class_names:

        wanted = {
            item.strip()
            for item
            in args.class_names.split(",")
            if item.strip()
        }

        unknown = (
            wanted
            - set(
                COCO80_CLASSES
            )
        )

        if unknown:

            raise ValueError(
                f"Unknown COCO classes: "
                f"{sorted(unknown)}"
            )

        classes = [
            class_name
            for class_name
            in COCO80_CLASSES
            if class_name
            in wanted
        ]

    else:

        classes = (
            COCO80_CLASSES.copy()
        )

    print(
        f"Classes ({len(classes)}): "
        f"{classes}"
    )

    # Pre-create all required class folders, including original_no
    # INSIDE forget.
    for class_name in classes:

        cls = safe_name(
            class_name
        )

        for path in [
            output_root
            / "retain"
            / "original_yes"
            / cls,

            output_root
            / "retain"
            / "edited_no"
            / cls,

            output_root
            / "forget"
            / "original_no"
            / cls,

            output_root
            / "forget"
            / "edited_yes"
            / cls,

            output_root
            / "preview"
            / cls,
        ]:
            ensure_dir(
                path
            )

    # ========================================================
    # Per-class candidate queues
    # ========================================================

    class_states = {}

    for class_index, class_name in enumerate(
        classes
    ):

        category_id = (
            name_to_id[
                class_name
            ]
        )

        image_ids = list(
            set(
                coco.getImgIds(
                    catIds=[
                        category_id
                    ]
                )
            )
        )

        random.Random(
            args.seed
            + class_index
        ).shuffle(
            image_ids
        )

        class_states[
            class_name
        ] = {
            "category_id":
                category_id,

            "ids":
                image_ids,

            "cursor":
                0,

            "accepted":
                0,

            "preview":
                0,

            "exhausted":
                False,
        }

    # ========================================================
    # LLaVA
    # ========================================================

    if args.dtype == "bf16":
        dtype = torch.bfloat16

    elif args.dtype == "fp16":
        dtype = torch.float16

    else:
        dtype = torch.float32

    print(
        "Loading LLaVA..."
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            args.model_path,
            trust_remote_code=True,
        )
    )

    model = (
        LlavaNextForConditionalGeneration
        .from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        .to(
            args.device
        )
        .eval()
    )

    # ========================================================
    # SimpleLama
    # ========================================================

    print(
        f"Loading SimpleLama on "
        f"{args.lama_device}..."
    )

    lama = SimpleLama(
        device=torch.device(
            args.lama_device
        )
    )

    metadata_path = (
        output_root
        / "metadata.jsonl"
    )

    rejection_path = (
        output_root
        / "rejections.jsonl"
    )

    summary_path = (
        output_root
        / "summary.json"
    )

    metadata_file = open(
        metadata_path,
        "w",
        encoding="utf-8",
    )

    rejection_file = open(
        rejection_path,
        "w",
        encoding="utf-8",
    )

    accepted_pairs = 0
    retain_count = 0
    forget_count = 0

    transition_counts = {
        "YN": 0,
        "YY": 0,
        "NN": 0,
        "NY": 0,
    }

    try:

        global_mode = (
            args.target_pairs > 0
            or args.target_each_partition > 0
        )

        # ====================================================
        # GLOBAL MODE
        # ====================================================

        if global_mode:

            while not global_goal_met(
                args,
                accepted_pairs,
                retain_count,
                forget_count,
            ):

                accepted_this_round = 0
                any_available = False

                for class_name in classes:

                    if global_goal_met(
                        args,
                        accepted_pairs,
                        retain_count,
                        forget_count,
                    ):
                        break

                    state = (
                        class_states[
                            class_name
                        ]
                    )

                    if (
                        args.max_per_class > 0
                        and state["accepted"]
                        >= args.max_per_class
                    ):
                        continue

                    if state[
                        "exhausted"
                    ]:
                        continue

                    any_available = True

                    # Scan rejected source images until we find at most
                    # one accepted pair for this class in this round.
                    while (
                        state[
                            "cursor"
                        ]
                        < len(
                            state[
                                "ids"
                            ]
                        )
                    ):

                        image_id = (
                            state[
                                "ids"
                            ][
                                state[
                                    "cursor"
                                ]
                            ]
                        )

                        state[
                            "cursor"
                        ] += 1

                        result = process_one_source(
                            args=args,
                            coco=coco,
                            model=model,
                            processor=processor,
                            lama=lama,
                            output_root=
                                output_root,
                            metadata_file=
                                metadata_file,
                            rejection_file=
                                rejection_file,
                            class_name=
                                class_name,
                            category_id=
                                state[
                                    "category_id"
                                ],
                            image_id=
                                image_id,
                            preview_index=
                                state[
                                    "preview"
                                ],
                        )

                        if result is None:
                            continue

                        state[
                            "accepted"
                        ] += 1

                        if (
                            state[
                                "preview"
                            ]
                            < args.preview_count
                        ):
                            state[
                                "preview"
                            ] += 1

                        accepted_pairs += 1

                        retain_count += (
                            result[
                                "retain_added"
                            ]
                        )

                        forget_count += (
                            result[
                                "forget_added"
                            ]
                        )

                        transition_counts[
                            result[
                                "transition"
                            ]
                        ] += 1

                        accepted_this_round += 1

                        print(
                            f"\n[GLOBAL] "
                            f"pairs="
                            f"{accepted_pairs} | "
                            f"retain="
                            f"{retain_count} | "
                            f"forget="
                            f"{forget_count}"
                        )

                        break

                    if (
                        state[
                            "cursor"
                        ]
                        >= len(
                            state[
                                "ids"
                            ]
                        )
                    ):
                        state[
                            "exhausted"
                        ] = True

                if (
                    accepted_this_round == 0
                ):

                    print(
                        "\nNo more acceptable source images "
                        "were found in the available pool."
                    )

                    break

                if not any_available:
                    break

        # ====================================================
        # PER-CLASS TEST MODE
        # ====================================================

        else:

            for class_name in classes:

                state = (
                    class_states[
                        class_name
                    ]
                )

                print()
                print(
                    "=" * 72
                )

                print(
                    f"PURGE-H CLASS: "
                    f"{class_name}"
                )

                print(
                    "=" * 72
                )

                while (
                    state[
                        "accepted"
                    ]
                    < args.samples_per_class

                    and state[
                        "cursor"
                    ]
                    < len(
                        state[
                            "ids"
                        ]
                    )
                ):

                    image_id = (
                        state[
                            "ids"
                        ][
                            state[
                                "cursor"
                            ]
                        ]
                    )

                    state[
                        "cursor"
                    ] += 1

                    result = process_one_source(
                        args=args,
                        coco=coco,
                        model=model,
                        processor=processor,
                        lama=lama,
                        output_root=
                            output_root,
                        metadata_file=
                            metadata_file,
                        rejection_file=
                            rejection_file,
                        class_name=
                            class_name,
                        category_id=
                            state[
                                "category_id"
                            ],
                        image_id=
                            image_id,
                        preview_index=
                            state[
                                "preview"
                            ],
                    )

                    if result is None:
                        continue

                    state[
                        "accepted"
                    ] += 1

                    if (
                        state[
                            "preview"
                        ]
                        < args.preview_count
                    ):
                        state[
                            "preview"
                        ] += 1

                    accepted_pairs += 1

                    retain_count += (
                        result[
                            "retain_added"
                        ]
                    )

                    forget_count += (
                        result[
                            "forget_added"
                        ]
                    )

                    transition_counts[
                        result[
                            "transition"
                        ]
                    ] += 1

                print(
                    f"\n[SUMMARY]"
                    f"[{class_name}] "
                    f"accepted="
                    f"{state['accepted']}/"
                    f"{args.samples_per_class}"
                )

    finally:

        metadata_file.close()
        rejection_file.close()

    summary = {
        "accepted_pairs":
            int(
                accepted_pairs
            ),

        "saved_images":
            int(
                retain_count
                + forget_count
            ),

        "retain_images":
            int(
                retain_count
            ),

        "forget_images":
            int(
                forget_count
            ),

        "transition_counts":
            transition_counts,

        "per_class_accepted": {
            class_name:
                int(
                    state[
                        "accepted"
                    ]
                )

            for class_name, state
            in class_states.items()
        },
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            summary,
            handle,
            indent=2,
        )

    print()
    print(
        "=" * 72
    )

    print(
        "PURGE-H COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Accepted pairs : "
        f"{accepted_pairs}"
    )

    print(
        f"Retain images  : "
        f"{retain_count}"
    )

    print(
        f"Forget images  : "
        f"{forget_count}"
    )

    print(
        f"Output         : "
        f"{output_root}"
    )

    print(
        f"Summary        : "
        f"{summary_path}"
    )


if __name__ == "__main__":
    main()