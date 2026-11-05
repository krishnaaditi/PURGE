# PURGE: Training and Data Preparation

This repository contains scripts for three PURGE variants:

- **PURGE-D:** trains with retain, forget, and object-absent negative examples.
- **PURGE-H:** trains from JSONL metadata describing retain and forget examples.
- **PURGE-M:** prepares training/evaluation data from object and cue-presence metadata. The supplied PURGE-M script prepares data; model training uses the JSONL-based trainer.

Use a local model directory or a Hugging Face model ID for `--model_path`. Set the model path, dataset paths, and output path to match your machine. The commands below are examples; they do not assume a particular machine's directories.

## Repository files

```text
Train/
├── Train_PURGE_D.py       # PURGE-D trainer for LLaVA or Qwen3-VL
├── Train_PURGE_H.py       # JSONL-based PURGE-H trainer; also used for prepared PURGE-M data
├── Train_PURGE_M.py       # PURGE-M data preparation only
├── run_llava_purge_d.sh   # Example PURGE-D LLaVA command
├── run_qwen3_vl_purge_d.sh # Example PURGE-D Qwen3-VL command
├── run_purge_H.sh         # PURGE-H launcher
└── run_purge_m_prepare.sh # PURGE-M training launcher (name is misleading)
```

The shell examples may contain machine-specific paths. Edit those paths before running them, or use the commands below.

## Requirements

Activate an environment that has the required PyTorch, Transformers, PEFT, and image-processing dependencies installed. The VLM must be available locally or downloadable by Transformers. For two-GPU examples, make both GPUs visible with `CUDA_VISIBLE_DEVICES=0,1`.

## PURGE-D

### Data layout

`--data_root` must point to the `images/train` directory:

```text
PURGE_D/
└── images/
    ├── train/
    │   ├── retain/<class>/*      # Object-only images; answer target: yes
    │   ├── forget/<class>/*      # Object-removed/context-only images; suppress yes
    │   └── forget_neg/<class>/*  # Object absent; supervise no
    └── test/
        └── forget_neg/*
```

The supplied trainer reads the three `train` partitions. It does not evaluate the `test` directory during training. For NPO, the `forget` partition's `yes` response is the response being suppressed; `forget_neg` provides explicit `no` supervision.

### LLaVA example

Set `MODEL_PATH`, `DATA_ROOT`, and `OUTPUT_DIR` for your setup. `--model_family` identifies which implementation to run; select `llava` for LLaVA.

```bash
MODEL_PATH="/path/to/your/llava-model"
DATA_ROOT="/path/to/PURGE_D/images/train"
OUTPUT_DIR="outputs/PURGE_D/llava_npo"

CUDA_VISIBLE_DEVICES=0,1 python -u Train/Train_PURGE_D.py \
  --model_family llava \
  --model_path "$MODEL_PATH" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
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

### Qwen3-VL example

Change `MODEL_PATH` to your Qwen3-VL model directory and use `--model_family qwen3_vl` with `--device_map balanced`.

```bash
MODEL_PATH="/path/to/your/qwen3-vl-model"
DATA_ROOT="/path/to/PURGE_D/images/train"
OUTPUT_DIR="outputs/PURGE_D/qwen3_vl_npo"

CUDA_VISIBLE_DEVICES=0,1 python -u Train/Train_PURGE_D.py \
  --model_family qwen3_vl \
  --model_path "$MODEL_PATH" \
  --data_root "$DATA_ROOT" \
  --output_dir "$OUTPUT_DIR" \
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

The PURGE-D trainer accepts `ga`, `kl`, or `npo` for `--algo`. The `--train_projector` flag enables projector training alongside LoRA. `llava_mistral_10_22` is a two-GPU map for the supplied LLaVA Mistral model; choose a supported map matching your model and visible GPUs. Set `--steps_per_epoch 0` to use the full size of the retain/forget loaders; `--max_per_class` still limits examples sampled from each class. Set `--max_per_class 0` to disable that cap. Checkpoints, including per-epoch and final checkpoints, are written under `OUTPUT_DIR`.

## PURGE-H

PURGE-H is a separate behavior-based variant. It is **not** the same data method as PURGE-D. The supplied trainer expects precomputed metadata; it does not create the behavior-based partitions.

### Metadata format

Pass a JSONL file with one JSON object per image. Each record must include:

- `split`: starts with `retain` or `forget` (for example, `retain` or `forget/group1`).
- `file_name`: image path, absolute or relative to `--data_root`.
- `target_object`: object name used to form the yes/no question.
- `prediction`: the original model's `yes`/`no` prediction used as the training target.

For PURGE-H, retain examples should represent correct original predictions; forget examples should represent the original model's erroneous predictions, as determined by the partitioning procedure.

### Train one model and algorithm

Set `MODEL_PATH`, `DATA_ROOT`, `METADATA_JSONL`, and `OUTPUT_DIR` for your data and chosen model. Run the command once per model; change `MODEL_PATH` and `OUTPUT_DIR` for another model. The trainer accepts `ga`, `kl`, or `npo`.

```bash
MODEL_PATH="/path/to/your/model"
DATA_ROOT="/path/to/PURGE_H"
METADATA_JSONL="/path/to/PURGE_H/metadata.jsonl"
OUTPUT_DIR="outputs/PURGE_H/your_model/npo"

CUDA_VISIBLE_DEVICES=0,1 python -u Train/Train_PURGE_H.py \
  --model_path "$MODEL_PATH" \
  --data_root "$DATA_ROOT" \
  --metadata_jsonl "$METADATA_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --algo npo \
  --device_map balanced \
  --dtype bf16 \
  --epochs 5 \
  --steps_per_epoch 0 \
  --lr_lora 2e-5 \
  --lr_projector 1e-5 \
  --lambda_forget 1.0 \
  --beta 0.1 \
  --seed 42
```

Add `--llava_single_crop` when using a compatible LLaVA model and you want the trainer's single-crop image preprocessing. The PURGE-H trainer enables and optimizes its language-model LoRA and projector/fusion parameters. It saves `epoch_N` and `final` checkpoints under `OUTPUT_DIR`.

The supplied `Train/run_purge_H.sh` is an optional launcher. Edit its `HERE`, `DATA_ROOT`, `METADATA`, `OUTPUT_ROOT`, `LLAVA_MODEL`, and `QWEN_MODEL` values for your machine before using it. Set `MODEL` to `llava`, `qwen`, or `both`, and `ALGO` to `ga`, `kl`, `npo`, or `all`.

## PURGE-M

PURGE-M uses object presence and cue presence to define training and evaluation examples.

### Prepare PURGE-M data

The input CSV must contain `image`, `target_object`, `object_present`, `cue_object`, `cue_present`, `training_role`, and `partition` columns. Roles must agree with the following assignments:

| Role | Object present | Cue present | Use |
|---|---:|---:|---|
| `retain` | yes | no | Training retain set |
| `forget` | no | yes | Training forget set |
| `evaluation` | yes or no | yes or no | Held-out set; allowed pairs are object=yes/cue=yes or object=no/cue=no |

Run the preparation script:

```bash
python -u Train/Train_PURGE_M.py \
  --metadata_csv "/path/to/PURGE_M/metadata_with_cues.csv" \
  --image_root "/path/to/PURGE_M" \
  --output_root "/path/to/PURGE_M_prepared" \
  --link_mode hardlink
```

Use `--link_mode copy` if hard links are unavailable. The script refuses a non-empty output directory to avoid mixing results from an earlier preparation run.

The prepared output contains:

```text
PURGE_M_prepared/
├── metadata_train.jsonl
├── metadata_evaluation.jsonl
├── summary.json
└── images/
    ├── train/
    │   ├── retain/<target-class>/*
    │   └── forget/<target-class>/*
    └── evaluation/<partition>/<target-class>/*
```

### Train on prepared PURGE-M data

The attached PURGE-M file prepares the data; training uses the JSONL-based `Train/Train_PURGE_H.py` trainer. The generated `metadata_train.jsonl` is compatible with that trainer. Set `MODEL_PATH` to the model you want and run one command per model/algorithm.

```bash
MODEL_PATH="/path/to/your/model"
DATA_ROOT="/path/to/PURGE_M_prepared/images"
METADATA_JSONL="/path/to/PURGE_M_prepared/metadata_train.jsonl"
OUTPUT_DIR="outputs/PURGE_M/your_model/npo"

CUDA_VISIBLE_DEVICES=0,1 python -u Train/Train_PURGE_H.py \
  --model_path "$MODEL_PATH" \
  --data_root "$DATA_ROOT" \
  --metadata_jsonl "$METADATA_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --algo npo \
  --device_map balanced \
  --dtype bf16 \
  --epochs 5 \
  --steps_per_epoch 0 \
  --lr_lora 2e-5 \
  --lr_projector 1e-5 \
  --lambda_forget 1.0 \
  --beta 0.1 \
  --seed 42
```

The supplied `Train/run_purge_m_prepare.sh` name is misleading: it invokes `Train_PURGE_H.py` to train from prepared metadata; it does not prepare the CSV data. Edit its model and data path variables before running it. You can select `MODEL=llava`, `MODEL=qwen`, or `MODEL=both`, and `ALGO=ga`, `kl`, `npo`, or `all`.

## Choosing the model and algorithm

- For PURGE-D, use `--model_family llava` or `--model_family qwen3_vl` and provide that model's path with `--model_path`.
- For PURGE-H and prepared PURGE-M, provide the selected model directory or model ID with `--model_path`.
- The trainers support `ga`, `kl`, and `npo`. Run separate commands when you want to compare algorithms or models.
- `steps_per_epoch=0` selects the trainer's full-loader behavior. Use a positive value to cap the number of optimization steps per epoch.

## Checkpoints and outputs

Each training run writes its checkpoints and configuration files beneath the supplied `--output_dir`. Keep separate output directories for different models, algorithms, or experiments so that runs do not overwrite one another.
