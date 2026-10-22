#!/usr/bin/env bash
# PURGE-D-IP generation wrapper
# Uses Dataset/PURGE_D_im.py
# Processes all 80 COCO classes unless --classes is provided.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/PURGE_D_in.py"

PYTHON_BIN="${PYTHON_BIN:-python}"

COCO_ROOT=""
ANN_JSON=""
OUTPUT_ROOT=""

CLASSES=""
SAMPLES_PER_CLASS=100
TARGET_PAIRS=0
MAX_PER_CLASS=0

QUALITY_MODE="natural"
MAX_INPAINT_ATTEMPTS=3
LAMA_DEVICE="cpu"
RETAIN_BACKGROUND="white"
SAVE_SIZE=0
PREVIEW_COUNT=10

MAX_TARGET_OVERLAP=0.20
MAX_BOUNDARY_CHANGE=45
STRICT_NATURAL_SCORE=84
FALLBACK_NATURAL_SCORE=120

SEED=42
OVERWRITE=0


usage() {
    cat <<'EOF'
Usage:

  bash Dataset/generate_purge_d_in.sh \
    --coco-root /path/to/coco/train2017 \
    --ann-json /path/to/coco/annotations/instances_train2017.json \
    --output-root ./outputs/PURGE_D_IP \
    [options]

Required:
  --coco-root PATH
  --ann-json PATH
  --output-root PATH

Class selection:
  --classes "apple,airplane"
      Optional. If omitted, all 80 COCO classes are processed.

Dataset size:
  --samples-per-class N
      Accepted retain/forget pairs per class.
      Default: 100

  --target-pairs N
      Global number of accepted source pairs.
      Example: 8000 -> about 8000 retain + 8000 forget images.
      Default: 0 (disabled)

  --max-per-class N
      Global mode only. 0 = unlimited.

Inpainting:
  --quality-mode {paper,natural}
      Default: natural

  --max-inpaint-attempts {1,2,3}
      Default: 3

  --lama-device DEVICE
      Default: cpu

Retain:
  --retain-background {white,black,gray}
      Default: white

Output:
  --save-size N
      0 preserves original resolution.
      Default: 0

  --preview-count N
      Default: 10

Quality control:
  --max-target-overlap FLOAT
      Default: 0.20

  --max-boundary-change FLOAT
      Default: 45

  --strict-natural-score FLOAT
      Default: 84

  --fallback-natural-score FLOAT
      Default: 120

Other:
  --seed N
      Default: 42

  --overwrite

  -h, --help
EOF
}


while [[ $# -gt 0 ]]; do
    case "$1" in

        --coco-root)
            COCO_ROOT="$2"
            shift 2
            ;;

        --ann-json)
            ANN_JSON="$2"
            shift 2
            ;;

        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;

        --classes)
            CLASSES="$2"
            shift 2
            ;;

        --samples-per-class)
            SAMPLES_PER_CLASS="$2"
            shift 2
            ;;

        --target-pairs)
            TARGET_PAIRS="$2"
            shift 2
            ;;

        --max-per-class)
            MAX_PER_CLASS="$2"
            shift 2
            ;;

        --quality-mode)
            QUALITY_MODE="$2"
            shift 2
            ;;

        --max-inpaint-attempts)
            MAX_INPAINT_ATTEMPTS="$2"
            shift 2
            ;;

        --lama-device)
            LAMA_DEVICE="$2"
            shift 2
            ;;

        --retain-background)
            RETAIN_BACKGROUND="$2"
            shift 2
            ;;

        --save-size)
            SAVE_SIZE="$2"
            shift 2
            ;;

        --preview-count)
            PREVIEW_COUNT="$2"
            shift 2
            ;;

        --max-target-overlap)
            MAX_TARGET_OVERLAP="$2"
            shift 2
            ;;

        --max-boundary-change)
            MAX_BOUNDARY_CHANGE="$2"
            shift 2
            ;;

        --strict-natural-score)
            STRICT_NATURAL_SCORE="$2"
            shift 2
            ;;

        --fallback-natural-score)
            FALLBACK_NATURAL_SCORE="$2"
            shift 2
            ;;

        --seed)
            SEED="$2"
            shift 2
            ;;

        --overwrite)
            OVERWRITE=1
            shift
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            echo "ERROR: unknown argument: $1"
            echo
            usage
            exit 1
            ;;
    esac
done


if [[ -z "$COCO_ROOT" ]]; then
    echo "ERROR: --coco-root is required."
    exit 1
fi

if [[ -z "$ANN_JSON" ]]; then
    echo "ERROR: --ann-json is required."
    exit 1
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    echo "ERROR: --output-root is required."
    exit 1
fi

if [[ ! -f "$PYTHON_SCRIPT" ]]; then
    echo "ERROR: PURGE_D_im.py not found:"
    echo "  $PYTHON_SCRIPT"
    exit 1
fi

if [[ ! -d "$COCO_ROOT" ]]; then
    echo "ERROR: COCO image directory not found:"
    echo "  $COCO_ROOT"
    exit 1
fi

if [[ ! -f "$ANN_JSON" ]]; then
    echo "ERROR: COCO annotation JSON not found:"
    echo "  $ANN_JSON"
    exit 1
fi


CMD=(
    "$PYTHON_BIN"
    "$PYTHON_SCRIPT"

    --coco_image_root "$COCO_ROOT"
    --coco_ann_json "$ANN_JSON"
    --output_root "$OUTPUT_ROOT"

    --samples_per_class "$SAMPLES_PER_CLASS"
    --target_pairs "$TARGET_PAIRS"
    --max_per_class "$MAX_PER_CLASS"

    --quality_mode "$QUALITY_MODE"
    --max_inpaint_attempts "$MAX_INPAINT_ATTEMPTS"
    --lama_device "$LAMA_DEVICE"

    --retain_background "$RETAIN_BACKGROUND"
    --save_size "$SAVE_SIZE"
    --preview_count "$PREVIEW_COUNT"

    --max_target_overlap "$MAX_TARGET_OVERLAP"
    --max_boundary_change "$MAX_BOUNDARY_CHANGE"
    --strict_natural_score "$STRICT_NATURAL_SCORE"
    --fallback_natural_score "$FALLBACK_NATURAL_SCORE"

    --seed "$SEED"
)


if [[ -n "$CLASSES" ]]; then
    CMD+=(
        --class_names "$CLASSES"
    )
fi


if [[ "$OVERWRITE" -eq 1 ]]; then
    CMD+=(
        --overwrite
    )
fi


echo
echo "============================================================"
echo "PURGE-D-IP generation"
echo "============================================================"
echo "Python script          : $PYTHON_SCRIPT"
echo "COCO images            : $COCO_ROOT"
echo "COCO annotations       : $ANN_JSON"
echo "Output                  : $OUTPUT_ROOT"

if [[ -n "$CLASSES" ]]; then
    echo "Classes                 : $CLASSES"
else
    echo "Classes                 : all 80 COCO classes"
fi

if [[ "$TARGET_PAIRS" -gt 0 ]]; then
    echo "Target pairs            : $TARGET_PAIRS"
    echo "Expected retain images  : ~$TARGET_PAIRS"
    echo "Expected forget images  : ~$TARGET_PAIRS"
else
    echo "Samples per class       : $SAMPLES_PER_CLASS"
fi

echo "Quality mode            : $QUALITY_MODE"
echo "LaMa attempts           : $MAX_INPAINT_ATTEMPTS"
echo "LaMa device             : $LAMA_DEVICE"
echo "Retain background       : $RETAIN_BACKGROUND"
echo "============================================================"
echo

"${CMD[@]}"
