#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
TRAINER="${TRAINER:-${HERE}/Train_PURGE_H.py}"

PREPARED_ROOT="${PREPARED_ROOT:-/home/hq1351@ad.wayne.edu/home/hal/datasets/PURGE_M_prepared}"
DATA_ROOT="${DATA_ROOT:-${PREPARED_ROOT}/images}"
METADATA_JSONL="${METADATA_JSONL:-${PREPARED_ROOT}/metadata_train.jsonl}"

LLAVA_MODEL_PATH="${LLAVA_MODEL_PATH:-/home/hq1351@ad.wayne.edu/home/hal/models/llava-v1.6-mistral-7b-hf}"
QWEN_MODEL_PATH="${QWEN_MODEL_PATH:-/home/hq1351@ad.wayne.edu/home/hal/models/Qwen3-VL-8B-Instruct}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/home/hq1351@ad.wayne.edu/home/hal/test/PURGE_M_1epoch_400steps}"
MODEL="${MODEL:-both}"             # llava, qwen, or both
ALGO="${ALGO:-all}"                # ga, kl, npo, or all
EPOCHS="${EPOCHS:-1}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-400}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -f "$TRAINER" ]] || { echo "Trainer not found: $TRAINER" >&2; exit 1; }
[[ -f "$METADATA_JSONL" ]] || { echo "PURGE-M training metadata not found: $METADATA_JSONL" >&2; exit 1; }
[[ -d "$DATA_ROOT" ]] || { echo "PURGE-M image directory not found: $DATA_ROOT" >&2; exit 1; }

case "$MODEL" in
  llava) models=(llava) ;;
  qwen)  models=(qwen) ;;
  both)  models=(llava qwen) ;;
  *) echo "MODEL must be llava, qwen, or both." >&2; exit 2 ;;
esac

case "$ALGO" in
  ga)  algos=(ga) ;;
  kl)  algos=(kl) ;;
  npo) algos=(npo) ;;
  all) algos=(ga kl npo) ;;
  *) echo "ALGO must be ga, kl, npo, or all." >&2; exit 2 ;;
esac

"$PYTHON" -m py_compile "$TRAINER"

for model_name in "${models[@]}"; do
  if [[ "$model_name" == "llava" ]]; then
    model_path="$LLAVA_MODEL_PATH"
    model_args=(--llava_single_crop)
  else
    model_path="$QWEN_MODEL_PATH"
    model_args=()
  fi

  [[ -f "${model_path}/config.json" ]] || {
    echo "Model config not found: ${model_path}/config.json" >&2
    exit 1
  }

  for algo in "${algos[@]}"; do
    lr_lora="${LR_LORA:-2e-5}"
    lr_projector="${LR_PROJECTOR:-1e-5}"

    # Lower rates for Qwen KL, which produced NaNs at the previous rates.
    if [[ "$model_name" == "qwen" && "$algo" == "kl" ]]; then
      lr_lora="${QWEN_KL_LR_LORA:-2e-6}"
      lr_projector="${QWEN_KL_LR_PROJECTOR:-1e-6}"
    fi

    out_dir="${OUTPUT_ROOT}/${model_name}/${algo}"
    echo "Starting PURGE-M: model=${model_name}, algo=${algo}"
    echo "Epochs=${EPOCHS}; steps per epoch=${STEPS_PER_EPOCH}"
    echo "Output: ${out_dir}"

    "$PYTHON" -u "$TRAINER" \
      --model_path "$model_path" \
      --data_root "$DATA_ROOT" \
      --metadata_jsonl "$METADATA_JSONL" \
      --output_dir "$out_dir" \
      --algo "$algo" \
      --device_map balanced \
      --dtype bf16 \
      --epochs "$EPOCHS" \
      --steps_per_epoch "$STEPS_PER_EPOCH" \
      --lr_lora "$lr_lora" \
      --lr_projector "$lr_projector" \
      --lambda_forget 1 \
      --beta 0.1 \
      --seed 42 \
      "${model_args[@]}"
  done
done

echo "All requested PURGE-M training runs finished. Checkpoints: $OUTPUT_ROOT"