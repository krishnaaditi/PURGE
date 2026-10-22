#!/usr/bin/env bash

set -euo pipefail


# ================================================================
# Default configuration
# ================================================================

COCO_ROOT=""
OUTPUT_ROOT="./PURGE_D"

TRAIN_PER_CLASS=100
TEST_PER_CLASS=20

RESIZE=384
SEED=42

MIN_BBOX_W=12
MIN_BBOX_H=12

# Empty = use all 80 COCO classes
CLASS_NAMES=""


# ================================================================
# Locate PURGE_D.py
# ================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/PURGE_D.py"




usage() {

    echo ""
    echo "PURGE-D Dataset Generation"
    echo ""
    echo "Usage:"
    echo ""
    echo "  bash generate_purge_d.sh \\"
    echo "      --coco-root PATH \\"
    echo "      [--output-root PATH] \\"
    echo "      [--train-per-class N] \\"
    echo "      [--test-per-class N] \\"
    echo "      [--resize N] \\"
    echo "      [--seed N] \\"
    echo "      [--min-bbox-w N] \\"
    echo "      [--min-bbox-h N] \\"
    echo "      [--classes \"apple,dog,car\"]"
    echo ""
    echo "Required:"
    echo ""
    echo "  --coco-root PATH"
    echo "      Root directory of MS COCO 2017."
    echo ""
    echo "      Expected structure:"
    echo ""
    echo "      PATH/"
    echo "      ├── train2017/"
    echo "      └── annotations/"
    echo "          └── instances_train2017.json"
    echo ""
    echo "Optional:"
    echo ""
    echo "  --output-root PATH"
    echo "      Output directory."
    echo "      Default: ./PURGE_D"
    echo ""
    echo "  --train-per-class N"
    echo "      Number of training source images per class."
    echo "      Default: 100"
    echo ""
    echo "  --test-per-class N"
    echo "      Number of test source images per class."
    echo "      Default: 20"
    echo ""
    echo "  --resize N"
    echo "      Generated image resolution N x N."
    echo "      Use 0 to preserve original resolution."
    echo "      Default: 384"
    echo ""
    echo "  --seed N"
    echo "      Random seed."
    echo "      Default: 42"
    echo ""
    echo "  --min-bbox-w N"
    echo "      Minimum object bounding-box width."
    echo "      Default: 12"
    echo ""
    echo "  --min-bbox-h N"
    echo "      Minimum object bounding-box height."
    echo "      Default: 12"
    echo ""
    echo "  --classes LIST"
    echo "      Optional comma-separated COCO class list."
    echo ""
    echo "      Example:"
    echo "          --classes \"apple,dog,car\""
    echo ""
    echo "      If omitted, ALL 80 COCO classes are generated."
    echo ""
    echo "  -h, --help"
    echo "      Show this help message."
    echo ""
}


# ================================================================
# Parse command-line arguments
# ================================================================

while [[ $# -gt 0 ]]; do

    case "$1" in

        --coco-root)
            COCO_ROOT="$2"
            shift 2
            ;;

        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;

        --train-per-class)
            TRAIN_PER_CLASS="$2"
            shift 2
            ;;

        --test-per-class)
            TEST_PER_CLASS="$2"
            shift 2
            ;;

        --resize)
            RESIZE="$2"
            shift 2
            ;;

        --seed)
            SEED="$2"
            shift 2
            ;;

        --min-bbox-w)
            MIN_BBOX_W="$2"
            shift 2
            ;;

        --min-bbox-h)
            MIN_BBOX_H="$2"
            shift 2
            ;;

        --classes)
            CLASS_NAMES="$2"
            shift 2
            ;;

        -h|--help)
            usage
            exit 0
            ;;

        *)
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;

    esac

done


================================================================

if [[ -z "${COCO_ROOT}" ]]; then

    echo ""
    echo "[ERROR] --coco-root is required."
    echo ""

    usage
    exit 1

fi


# ================================================================
# Construct COCO paths
# ================================================================

COCO_IMAGE_ROOT="${COCO_ROOT}/train2017"

COCO_ANN_JSON="${COCO_ROOT}/annotations/instances_train2017.json"


# ================================================================
# Validate files and directories
# ================================================================

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then

    echo ""
    echo "[ERROR] PURGE_D.py was not found:"
    echo ""
    echo "    ${PYTHON_SCRIPT}"
    echo ""
    echo "Place generate_purge_d.sh in the same directory as PURGE_D.py."
    echo ""

    exit 1

fi


if [[ ! -d "${COCO_IMAGE_ROOT}" ]]; then

    echo ""
    echo "[ERROR] COCO image directory was not found:"
    echo ""
    echo "    ${COCO_IMAGE_ROOT}"
    echo ""

    exit 1

fi


if [[ ! -f "${COCO_ANN_JSON}" ]]; then

    echo ""
    echo "[ERROR] COCO annotation file was not found:"
    echo ""
    echo "    ${COCO_ANN_JSON}"
    echo ""

    exit 1

fi


mkdir -p "${OUTPUT_ROOT}"


# ================================================================
# Print configuration
# ================================================================

echo ""
echo "================================================================"
echo "                  PURGE-D Dataset Generation"
echo "================================================================"
echo ""
echo "PURGE-D script:"
echo "  ${PYTHON_SCRIPT}"
echo ""
echo "COCO images:"
echo "  ${COCO_IMAGE_ROOT}"
echo ""
echo "COCO annotations:"
echo "  ${COCO_ANN_JSON}"
echo ""
echo "Output:"
echo "  ${OUTPUT_ROOT}"
echo ""
echo "Train images / class : ${TRAIN_PER_CLASS}"
echo "Test images / class  : ${TEST_PER_CLASS}"
echo "Resize               : ${RESIZE} x ${RESIZE}"
echo "Random seed          : ${SEED}"
echo "Minimum bbox width   : ${MIN_BBOX_W}"
echo "Minimum bbox height  : ${MIN_BBOX_H}"

if [[ -z "${CLASS_NAMES}" ]]; then

    echo "Classes              : ALL 80 COCO classes"

else

    echo "Classes              : ${CLASS_NAMES}"

fi

echo ""
echo "================================================================"
echo ""


# ================================================================
# Build Python command
# ================================================================

CMD=(
    python "${PYTHON_SCRIPT}"

    --coco_image_root
    "${COCO_IMAGE_ROOT}"

    --coco_ann_json
    "${COCO_ANN_JSON}"

    --output_root
    "${OUTPUT_ROOT}"

    --train_per_class
    "${TRAIN_PER_CLASS}"

    --test_per_class
    "${TEST_PER_CLASS}"

    --resize
    "${RESIZE}"

    --seed
    "${SEED}"

    --min_bbox_w
    "${MIN_BBOX_W}"

    --min_bbox_h
    "${MIN_BBOX_H}"
)


# ================================================================
# Optional class subset
# ================================================================

if [[ -n "${CLASS_NAMES}" ]]; then

    CMD+=(
        --class_names
        "${CLASS_NAMES}"
    )

fi


# ================================================================
# Run PURGE-D generation
# ================================================================

echo "[INFO] Starting PURGE-D generation..."
echo ""

"${CMD[@]}"


# ================================================================
# Finished
# ================================================================

echo ""
echo "================================================================"
echo "             PURGE-D Generation Completed"
echo "================================================================"
echo ""
echo "Dataset:"
echo "  ${OUTPUT_ROOT}"
echo ""
echo "Metadata:"
echo "  ${OUTPUT_ROOT}/metadata.jsonl"
echo ""
echo "================================================================"
