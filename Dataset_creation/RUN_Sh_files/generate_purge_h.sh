#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "${SCRIPT_DIR}/PURGE_H.py" \
  --coco_image_root "$1" \
  --coco_ann_json "$2" \
  --model_path "$3" \
  --output_root "$4" \
  --target_each_partition 8000 \
  --max_inpaint_attempts 3 \
  --device cuda:0 \
  --lama_device cpu \
  --dtype bf16 \
  --save_size 0 \
  --preview_count 10 \
  --max_boundary_change 45 \
  --strict_natural_score 84 \
  --fallback_natural_score 120 \
  --max_target_overlap 0.20 \
  --seed 42
