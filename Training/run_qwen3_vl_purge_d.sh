#!/usr/bin/env bash
set -euo pipefail

source "/home/hq1351@ad.wayne.edu/miniconda3/etc/profile.d/conda.sh"
conda activate spu

CUDA_VISIBLE_DEVICES=0,1 python -u "/home/hq1351@ad.wayne.edu/home/hal/spu/Train/Train_PURGE_D.py" \
  --model_family qwen3_vl \
  --model_path "/home/hq1351@ad.wayne.edu/home/hal/models/Qwen3-VL-8B-Instruct" \
  --data_root "/home/hq1351@ad.wayne.edu/home/hal/datasets/PURDE_D/images/train" \
  --output_dir "/home/hq1351@ad.wayne.edu/home/hal/spu/outputs/PURDE_D_Qwen3VL_paperNPO" \
  --algo npo \
  --device_map balanced \
  --dtype bf16 \
  --epochs 5 \
  --max_per_class 100 \
  --num_workers 0 \
  --lr_lora 2e-5 \
  --lr_projector 1e-5 \
  --lambda_forget 1.0 \
  --beta 0.1 \
  --gamma 1.0 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --train_projector \
  --grad_checkpointing
