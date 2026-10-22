#!/usr/bin/env bash
set -euo pipefail

# PURGE-M run configuration.
# Edit these values for your machine, or override them when calling the script.
PYTHON_BIN="${PYTHON_BIN:-python}"
COCO_IMAGE_ROOT="${COCO_IMAGE_ROOT:-/datasets/train2017}"
COCO_ANN_JSON="${COCO_ANN_JSON:-datasets/annotations/instances_train2017.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/PURGE_M}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-models/llava-v1.6-mistral-7b-hf}"

# Experiment settings. Change these values to configure your run.
NUM_CUES="${NUM_CUES:-9}"
SPUR_K="${SPUR_K:-100}"
SELECT_BY="${SELECT_BY:-hr}"
CUE_DETECTOR_THRESHOLD="${CUE_DETECTOR_THRESHOLD:-0.10}"
DETECTOR_DEVICE="${DETECTOR_DEVICE:-cuda:0}"
VLM_DEVICE="${VLM_DEVICE:-cuda:0}"
MAX_IMAGES_PER_CLASS="${MAX_IMAGES_PER_CLASS:-0}"
MAX_MATERIALIZED_PER_COMBO="${MAX_MATERIALIZED_PER_COMBO:-0}"
LINK_MODE="${LINK_MODE:-hardlink}"
SEED="${SEED:-42}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "${REPO_ROOT}"

"${PYTHON_BIN}" Dataset/PURGE_M.py \
  --coco_image_root "${COCO_IMAGE_ROOT}" \
  --coco_ann_json "${COCO_ANN_JSON}" \
  --output_root "${OUTPUT_ROOT}" \
  --vlm_model_path "${VLM_MODEL_PATH}" \
  --num_cues "${NUM_CUES}" \
  --spur_k "${SPUR_K}" \
  --select_by "${SELECT_BY}" \
  --cue_detector_threshold "${CUE_DETECTOR_THRESHOLD}" \
  --detector_device "${DETECTOR_DEVICE}" \
  --vlm_device "${VLM_DEVICE}" \
  --max_images_per_class "${MAX_IMAGES_PER_CLASS}" \
  --max_materialized_per_combo "${MAX_MATERIALIZED_PER_COMBO}" \
  --link_mode "${LINK_MODE}" \
  --seed "${SEED}"
