## PURGE-D Training with LLaVA and Qwen3-VL
----------------------------------------------

```text
PURGE_D/
└── images/
    ├── train/
    │   ├── retain/<class>/*      # Object-only image; answer: yes
    │   ├── forget/<class>/*      # Background-only image; answer: yes
    │   └── forget_neg/<class>/*  # Object absent; answer: no
    └── test/
        └── forget_neg/*


Run with 2 gpu : LLaVA

CUDA_VISIBLE_DEVICES=0,1  python Train_PURGE_D.py" \
  --model_family llava \
  --model_path "/models/" \
  --data_root "/datasets/PURDE_D/images/train" \
  --output_dir "/outputs/" \
  --algo  \[choose]
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




Run with 2 gpu : Qwen

CUDA_VISIBLE_DEVICES=0,1 python -u "/Train_PURGE_D.py" \
  --model_family qwen3_vl \
  --model_path "/models/" \
  --data_root "/datasets/PURDE_D/images/train" \
  --output_dir "/outputs/" \
  --algo  \
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






## PURGE-M Training with LLaVA and Qwen3-VL
------------------------------------------

PURGE-M partitions images using target-object and cue presence:

```text
PURGE_M/
├── metadata_with_cues.csv
├── nn/<target-class>/*   # Target absent; cue absent — evaluation
├── ny/<target-class>/*   # Target absent; cue present — forget
├── yn/<target-class>/*   # Target present; cue absent — retain
└── yy/<target-class>/*   # Target present; cue present — evaluation
```

The preparation script uses valid cue assignments to create the training and evaluation metadata:

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

Use `Train/Train_PURGE_M.py` to prepare the dataset. Set the paths to your downloaded PURGE-M data:

```bash
python Train/Train_PURGE_M.py \
  --metadata_csv /path/to/PURGE_M/metadata_with_cues.csv \
  --image_root /path/to/PURGE_M \
  --output_root /path/to/PURGE_M_prepared \
  --link_mode hardlink
```

Use the PURGE-M training launcher to run GA, KL, and NPO. Choose `MODEL` as `llava`, `qwen`, or `both`; choose `ALGO` as `ga`, `kl`, `npo`, or `all`. Set the epoch and step counts to suit your run.

```bash
MODEL=both \
ALGO=all \
LLAVA_MODEL_PATH=/path/to/llava-v1.6-mistral-7b-hf \
QWEN_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct \
PREPARED_ROOT=/path/to/PURGE_M_prepared \
OUTPUT_ROOT=./outputs/PURGE_M \
EPOCHS=<number_of_epochs> \
STEPS_PER_EPOCH=<number_of_steps> \
CUDA_VISIBLE_DEVICES=0,1 \
bash Train/run_purge_m_train_all.sh
```

Replace the epoch and step placeholders with numbers before running. Use `STEPS_PER_EPOCH=0` to use all available training samples per epoch.

To run a single model or algorithm, change `MODEL` or `ALGO`. For example, use `MODEL=qwen ALGO=npo` to run NPO on Qwen3-VL only.
