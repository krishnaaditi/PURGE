#!/usr/bin/env bash


set -euo pipefail



WATERBIRDS_ROOT=""
SEGMENTATIONS_ROOT=""

OUTPUT_ROOT="./Waterbirds_PURGE_D"


TRAIN_PER_CLASS=-1
VAL_PER_CLASS=-1
TEST_PER_CLASS=-1

RESIZE=384
SEED=42

MASK_THRESHOLD=127

# Default: both classes
CLASS_NAMES="landbird,waterbird"




SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_SCRIPT="${SCRIPT_DIR}/PURGE_D_Waterbirds.py"



usage() {

    echo ""
    echo "PURGE-D Waterbirds Generation"
    echo ""
    echo "Usage:"
    echo ""
    echo "bash generate_purge_d_waterbirds.sh \\"
    echo "  --waterbirds-root PATH \\"
    echo "  --segmentations-root PATH \\"
    echo "  --output-root PATH \\"
    echo "  [--train-per-class N] \\"
    echo "  [--val-per-class N] \\"
    echo "  [--test-per-class N] \\"
    echo "  [--resize N] \\"
    echo "  [--seed N] \\"
    echo "  [--mask-threshold N] \\"
    echo "  [--classes \"landbird,waterbird\"]"
    echo ""
    echo "Required:"
    echo ""
    echo "  --waterbirds-root PATH"
    echo "      Path to waterbird_complete95_forest2water2"
    echo ""
    echo "  --segmentations-root PATH"
    echo "      Path to extracted CUB segmentations directory"
    echo ""
    echo "Optional:"
    echo ""
    echo "  --output-root PATH"
    echo "      Output directory"
    echo "      Default: ./Waterbirds_PURGE_D"
    echo ""
    echo "  --train-per-class N"
    echo "      Number of training images per bird class"
    echo "      Default: -1"
    echo ""
    echo "  --val-per-class N"
    echo "      Number of validation images per bird class"
    echo "      Default: -1"
    echo ""
    echo "  --test-per-class N"
    echo "      Number of test images per bird class"
    echo "      Default: -1"
    echo ""
    echo "      Values:"
    echo "          -1 = use all available images"
    echo "           0 = skip this split"
    echo "          >0 = randomly sample N images per class"
    echo ""
    echo "  --resize N"
    echo "      Resize generated images to N x N"
    echo "      Use 0 to preserve original resolution"
    echo "      Default: 384"
    echo ""
    echo "  --seed N"
    echo "      Random seed"
    echo "      Default: 42"
    echo ""
    echo "  --mask-threshold N"
    echo "      Threshold for converting segmentation masks to binary masks"
    echo "      Default: 127"
    echo ""
    echo "  --classes LIST"
    echo "      Comma-separated class names"
    echo ""
    echo "      Examples:"
    echo "          --classes \"landbird,waterbird\""
    echo "          --classes \"landbird\""
    echo "          --classes \"waterbird\""
    echo ""
    echo "  -h, --help"
    echo "      Show this help message"
    echo ""
}


# ================================================================
# Parse arguments
# ================================================================

while [[ $# -gt 0 ]]; do

    case "$1" in

        --waterbirds-root)
            WATERBIRDS_ROOT="$2"
            shift 2
            ;;

        --segmentations-root)
            SEGMENTATIONS_ROOT="$2"
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

        --val-per-class)
            VAL_PER_CLASS="$2"
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

        --mask-threshold)
            MASK_THRESHOLD="$2"
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
            echo ""
            echo "[ERROR] Unknown argument: $1"
            usage
            exit 1
            ;;

    esac

done


# ================================================================
# Validate required arguments
# ================================================================

if [[ -z "${WATERBIRDS_ROOT}" ]]; then
    echo ""
    echo "[ERROR] --waterbirds-root is required."
    usage
    exit 1
fi


if [[ -z "${SEGMENTATIONS_ROOT}" ]]; then
    echo ""
    echo "[ERROR] --segmentations-root is required."
    usage
    exit 1
fi


# ================================================================
# Validate Python script
# ================================================================

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then

    echo ""
    echo "[ERROR] Cannot find:"
    echo ""
    echo "  ${PYTHON_SCRIPT}"
    echo ""
    echo "Place generate_purge_d_waterbirds.sh in the same"
    echo "directory as PURGE_D_Waterbirds.py."
    echo ""

    exit 1
fi


# ================================================================
# Validate Waterbirds dataset
# ================================================================

if [[ ! -d "${WATERBIRDS_ROOT}" ]]; then

    echo ""
    echo "[ERROR] Waterbirds directory does not exist:"
    echo ""
    echo "  ${WATERBIRDS_ROOT}"
    echo ""

    exit 1
fi


if [[ ! -f "${WATERBIRDS_ROOT}/metadata.csv" ]]; then

    echo ""
    echo "[ERROR] metadata.csv not found:"
    echo ""
    echo "  ${WATERBIRDS_ROOT}/metadata.csv"
    echo ""

    exit 1
fi


# ================================================================
# Validate segmentation directory
# ================================================================

if [[ ! -d "${SEGMENTATIONS_ROOT}" ]]; then

    echo ""
    echo "[ERROR] Segmentation directory does not exist:"
    echo ""
    echo "  ${SEGMENTATIONS_ROOT}"
    echo ""

    exit 1
fi


mkdir -p "${OUTPUT_ROOT}"


# ================================================================
# Show configuration
# ================================================================

echo ""
echo "================================================================"
echo "             PURGE-D Waterbirds Generation"
echo "================================================================"
echo ""
echo "Python script:"
echo "  ${PYTHON_SCRIPT}"
echo ""
echo "Waterbirds:"
echo "  ${WATERBIRDS_ROOT}"
echo ""
echo "CUB segmentations:"
echo "  ${SEGMENTATIONS_ROOT}"
echo ""
echo "Output:"
echo "  ${OUTPUT_ROOT}"
echo ""
echo "Train / class      : ${TRAIN_PER_CLASS}"
echo "Val / class        : ${VAL_PER_CLASS}"
echo "Test / class       : ${TEST_PER_CLASS}"
echo "Resize             : ${RESIZE}"
echo "Seed               : ${SEED}"
echo "Mask threshold     : ${MASK_THRESHOLD}"
echo "Classes            : ${CLASS_NAMES}"
echo ""
echo "================================================================"
echo ""


# ================================================================
# Run generation
# ================================================================

python "${PYTHON_SCRIPT}" \
    --waterbirds_root "${WATERBIRDS_ROOT}" \
    --segmentations_root "${SEGMENTATIONS_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --train_per_class "${TRAIN_PER_CLASS}" \
    --val_per_class "${VAL_PER_CLASS}" \
    --test_per_class "${TEST_PER_CLASS}" \
    --resize "${RESIZE}" \
    --seed "${SEED}" \
    --mask_threshold "${MASK_THRESHOLD}" \
    --class_names "${CLASS_NAMES}"


# ================================================================
# Finished
# ================================================================

echo ""
echo "================================================================"
echo "       PURGE-D Waterbirds Generation Completed"
echo "================================================================"
echo ""
echo "Output:"
echo "  ${OUTPUT_ROOT}"
echo ""
echo "Metadata:"
echo "  ${OUTPUT_ROOT}/metadata.jsonl"
echo ""
echo "================================================================"
