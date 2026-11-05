# PURGE Training

This README describes how to prepare and train PURGE-D and PURGE-M with LLaVA and Qwen3-VL.

## PURGE-D Training

Expected dataset layout:

```text
PURGE_D/
└── images/
    ├── train/
    │   ├── retain/<class>/*      # Object-only images; answer: yes
    │   ├── forget/<class>/*      # Background-only images; answer: yes
    │   └── forget_neg/<class>/*  # Object-absent images; answer: no
    └── test/
        └── forget_neg/
```

### LLaVA

This example runs NPO using two visible GPUs. Replace `--algo npo` with another algorithm supported by the training script if needed.

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u Train/Train_PURGE_D.py \
  --model_family llava \
  --model_path "/path/to/llava-v1.6-mistral-7b-hf" \
  --data_root "/path/to/PURGE_D/images/train" \
  --output_dir "outputs/PURGE_D/llava_npo" \
  --algo npo \
  --device cuda:0 \
  --device_map llava_mistral_10_22 \
  --dtype bf16 \
  --epochs 5 \
  --steps_per_epoch 0 \
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
  --grad_checkpointing \
  --llava_single_crop
```

### Qwen3-VL

```bash
CUDA_VISIBLE_DEVICES=0,1 python -u Train/Train_PURGE_D.py \
  --model_family qwen3_vl \
  --model_path "/path/to/Qwen3-VL-8B-Instruct" \
  --data_root "/path/to/PURGE_D/images/train" \
  --output_dir "outputs/PURGE_D/qwen3_vl_npo" \
  --algo npo \
  --device_map balanced \
  --dtype bf16 \
  --epochs 5 \
  --steps_per_epoch 0 \
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
```

Set `--steps_per_epoch` to `0` to use the script’s full-dataset behavior for each epoch. Check the training script’s `--help` output for supported algorithm names and options.

## PURGE-M Training

PURGE-M partitions images according to target-object presence and cue presence:

```text
PURGE_M/
├── metadata_with_cues.csv
├── nn/<target-class>/*   # Target absent; cue absent — evaluation
├── ny/<target-class>/*   # Target absent; cue present — forget
├── yn/<target-class>/*   # Target present; cue absent — retain
└── yy/<target-class>/*   # Target present; cue present — evaluation
```

### Prepare the dataset

Use `Train/Train_PURGE_M.py` to create the prepared dataset:

```bash
python -u Train/Train_PURGE_M.py \
  --metadata_csv "/path/to/PURGE_M/metadata_with_cues.csv" \
  --image_root "/path/to/PURGE_M" \
  --output_root "/path/to/PURGE_M_prepared" \
  --link_mode hardlink
```

The prepared dataset contains training and evaluation metadata, plus the corresponding image partitions:

```text
PURGE_M_prepared/
├── metadata_train.jsonl
├── metadata_evaluation.jsonl
└── images/
    ├── train/
    │   ├── retain/<target-class>/*
    │   └── forget/<target-class>/*
    └── evaluation/
        └── <partition>/<target-class>/*
```

### Train PURGE-M

The launcher supports choosing a model (`llava`, `qwen`, or `both`) and an algorithm (`ga`, `kl`, `npo`, or `all`):

```bash
MODEL=both \
ALGO=all \
LLAVA_MODEL_PATH="/path/to/llava-v1.6-mistral-7b-hf" \
QWEN_MODEL_PATH="/path/to/Qwen3-VL-8B-Instruct" \
PREPARED_ROOT="/path/to/PURGE_M_prepared" \
OUTPUT_ROOT="outputs/PURGE_M" \
EPOCHS=5 \
STEPS_PER_EPOCH=0 \
CUDA_VISIBLE_DEVICES=0,1 \
bash Train/run_purge_m_train_all.sh
```

To run one model and one algorithm, change `MODEL` and `ALGO`. For example, use `MODEL=qwen ALGO=npo` to run NPO on Qwen3-VL only.
