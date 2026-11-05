#!/usr/bin/env bash
set -Eeuo pipefail

HERE="/home/hq1351@ad.wayne.edu/home/hal/spu/Train"
TRAINER="${HERE}/Train_PURGE_H.py"

DATA_ROOT="/home/hq1351@ad.wayne.edu/home/hal/datasets/PURGE_H"
METADATA="${DATA_ROOT}/metadata.jsonl"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/hq1351@ad.wayne.edu/home/hal/test/PURGE_H}"

LLAVA_MODEL="/home/hq1351@ad.wayne.edu/home/hal/models/llava-v1.6-mistral-7b-hf"
QWEN_MODEL="/home/hq1351@ad.wayne.edu/home/hal/models/Qwen3-VL-8B-Instruct"

MODEL="${MODEL:-llava}"       # llava | qwen | both
ALGO="${ALGO:-npo}"           # ga | kl | npo | all
EPOCHS="${EPOCHS:-5}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

[[ -f "$TRAINER" ]] || { echo "Trainer not found: $TRAINER" >&2; exit 1; }
[[ -f "$METADATA" ]] || { echo "Metadata not found: $METADATA" >&2; exit 1; }

case "$MODEL" in
  llava) models=(llava) ;;
  qwen)  models=(qwen) ;;
  both)  models=(llava qwen) ;;
  *) echo "MODEL must be llava, qwen, or both." >&2; exit 2 ;;
esac

case "$ALGO" in
  ga|kl|npo) algos=("$ALGO") ;;
  all)       algos=(ga kl npo) ;;
  *) echo "ALGO must be ga, kl, npo, or all." >&2; exit 2 ;;
esac

for model_name in "${models[@]}"; do
  if [[ "$model_name" == "llava" ]]; then
    model_path="$LLAVA_MODEL"
  else
    model_path="$QWEN_MODEL"
  fi

  [[ -d "$model_path" ]] || {
    echo "Model directory not found: $model_path" >&2
    exit 1
  }

  for algo in "${algos[@]}"; do
    output_dir="${OUTPUT_ROOT}/${model_name}/${algo}"
    echo "Starting PURGE-H: model=${model_name}, algo=${algo}"
    echo "Output: $output_dir"

    extra_args=()
    if [[ "$model_name" == "llava" ]]; then
      extra_args+=(--llava_single_crop)
    fi

    python -u "$TRAINER" \
      --model_path "$model_path" \
      --data_root "$DATA_ROOT" \
      --metadata_jsonl "$METADATA" \
      --output_dir "$output_dir" \
      --algo "$algo" \
      --device_map balanced \
      --dtype bf16 \
      --epochs "$EPOCHS" \
      --steps_per_epoch "$STEPS_PER_EPOCH" \
      --lr_lora 2e-5 \
      --lr_projector 1e-5 \
      --lambda_forget 1 \
      --beta 0.1 \
      --seed 42 \
      "${extra_args[@]}"
  done
done

echo "Finished. Checkpoints are under $OUTPUT_ROOT"
