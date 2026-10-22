# PURGE: Partition-Aware Unlearning for Removing Spurious Correlations in Large Vision-Language Models

This repository contains code and resources for **PURGE**, a framework for studying and reducing spurious object–context associations in large vision-language models (LVLMs). For example, a model may associate boats with water and answer “boat” based on the background even when the boat is absent.

![PURGE framework overview](Images/main_3.png)

*Overview of the PURGE framework.*

PURGE creates data partitions that distinguish object-grounded examples from examples influenced by contextual cues. The partitions support targeted unlearning while retaining useful visual knowledge. The paper studies Gradient Ascent (GA), KL-regularized unlearning (KL), and Negative Preference Optimization (NPO).

## PURGE dataset variants

![PURGE-D, PURGE-H, and PURGE-M dataset diagrams](Images/purge-dhm.png)

*Dataset construction strategies for PURGE-D, PURGE-H, and PURGE-M.*

| Variant | Construction | Purpose |
|---|---|---|
| **PURGE-D** | Data-level object/background disentanglement, including masking and counterfactual inpainting | Tests whether object and context can be separated in the training data. |
| **PURGE-H** | Behavior-driven partitioning using model responses and counterfactual examples | Identifies examples where a model’s prediction changes with contextual evidence. |
| **PURGE-M** | Model-informed semantic cue selection and object/cue presence | Builds partitions from target-object and contextual-cue evidence. |

## Repository structure

```text
PURGE/
├── Dataset_creation/
│   ├── RUN_Sh_files/                       # Shell wrappers for dataset-generation runs
│   ├── PURGE_D.py                          # PURGE-D for COCO
│   ├── PURGE_D_in.py                       # PURGE-D inpainting
│   ├── PURGE_D_Waterbirds.py               # PURGE-D for Waterbirds
│   ├── PURGE_H.py                          # PURGE-H
│   ├── PURGE_M.py                          # PURGE-M
│   ├── PURGE_croissant_final_NeurIPS2026.json
│   └── run_command.txt                     # Additional command examples
├── Training/                               # Training and unlearning code
├── Evaluation/                             # Evaluation code
├── Images/                                 # README diagrams (PNG files)
├── requirements.txt
└── README.md
```

The scripts may need local dataset, model, and output paths. Do not commit large datasets, model checkpoints, API keys, or private credentials to this repository.

## Data

### MS COCO 2017

PURGE generation scripts use COCO images and annotations. Download and extract the training images and annotations:

```bash
mkdir -p data/coco
cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip train2017.zip
unzip annotations_trainval2017.zip
cd ../..
```

This provides `data/coco/train2017/` and `data/coco/annotations/instances_train2017.json`.

### Waterbirds

The Waterbirds dataset is available from [grodino/waterbirds on Hugging Face](https://huggingface.co/datasets/grodino/waterbirds). Follow the dataset card for download and setup instructions. Some PURGE-D Waterbirds runs also require the segmentation files; see the relevant command and dataset documentation before running.

## Installation

From the repository root, create an environment and install the listed dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Some experiments require a CUDA-capable GPU and external model checkpoints. The required checkpoints are not included in this repository. Use the paths to datasets and model checkpoints available on your own machine.

## Dataset creation

Run commands from the repository root. The shell wrappers are in `Dataset_creation/RUN_Sh_files/`; edit their paths and experiment settings before running. The scripts also accept command-line options. Check the relevant script’s help for the exact options available in your checkout:

```bash
python Dataset_creation/PURGE_D.py --help
python Dataset_creation/PURGE_D_in.py --help
python Dataset_creation/PURGE_D_Waterbirds.py --help
python Dataset_creation/PURGE_H.py --help
python Dataset_creation/PURGE_M.py --help
```

### PURGE-D: COCO

Example using the COCO dataset and sampling 100 training and 20 test images per class:

```bash
bash Dataset_creation/RUN_Sh_files/generate_purge_d.sh \
  --coco-root /path/to/coco \
  --output-root ./outputs/PURGE_D \
  --train-per-class 100 \
  --test-per-class 20
```

### PURGE-D: Waterbirds

```bash
bash Dataset_creation/RUN_Sh_files/generate_purge_d_waterbirds.sh \
  --waterbirds-root /path/to/waterbird_complete95_forest2water2 \
  --segmentations-root /path/to/segmentations \
  --output-root ./outputs/Waterbirds_PURGE_D \
  --train-per-class -1 \
  --val-per-class -1 \
  --test-per-class -1
```

A value of `-1` follows the wrapper’s all-available-samples behavior. Check the wrapper help or source before changing sampling options.

### PURGE-D: inpainting

This example generates up to 8,000 target pairs and allows up to three inpainting attempts per pair:

```bash
bash Dataset_creation/RUN_Sh_files/generate_purge_d_in.sh \
  --coco-root /path/to/coco/train2017 \
  --ann-json /path/to/coco/annotations/instances_train2017.json \
  --output-root ./outputs/PURGE_D_IP \
  --target-pairs 8000 \
  --quality-mode natural \
  --max-inpaint-attempts 3
```

### PURGE-H

PURGE-H uses COCO annotations and a local LLaVA checkpoint. This positional-argument example follows the provided run instructions:

```bash
bash Dataset_creation/RUN_Sh_files/generate_purge_h.sh \
  /path/to/coco/train2017 \
  /path/to/coco/annotations/instances_train2017.json \
  /path/to/llava-v1.6-mistral-7b-hf \
  ./outputs/PURGE_H
```

### PURGE-M

PURGE-M uses COCO annotations, a local LLaVA checkpoint, and a cue detector. Update the paths for your machine. The example below keeps all settings visible and editable:

```bash
python Dataset_creation/PURGE_M.py \
  --coco_image_root /path/to/coco/train2017 \
  --coco_ann_json /path/to/coco/annotations/instances_train2017.json \
  --output_root ./outputs/PURGE_M \
  --vlm_model_path /path/to/llava-v1.6-mistral-7b-hf \
  --num_cues 9 \
  --spur_k 100 \
  --select_by hr \
  --cue_detector_threshold 0.10 \
  --detector_device cuda:0 \
  --vlm_device cuda:0 \
  --max_images_per_class 0 \
  --max_materialized_per_combo 0 \
  --link_mode hardlink \
  --seed 42
```

When `--max_images_per_class` or `--max_materialized_per_combo` is `0`, the script uses its unlimited/default behavior for that setting. Cue generation may require an API key or a cue JSON file, depending on the options used. Store API keys in environment variables; do not put them in the command, README, or Git history.

A shell wrapper for PURGE-M can be used if present in `RUN_Sh_files`. Edit the dataset, model, and output paths in the wrapper before running it. The `run_command.txt` file contains additional examples from the released scripts.

## Dataset output

The exact output layout depends on the selected method and options. PURGE-M organizes images by target/cue presence and writes manifests and summary metadata. Its four presence combinations are:

| Folder | Target object | Context cue |
|---|---|---|
| `yy` | Present | Present |
| `yn` | Present | Absent |
| `ny` | Absent | Present |
| `nn` | Absent | Absent |

Consult each script’s output messages and metadata files for the generated split names and labels. Use separate output directories when running different variants or settings.

## Training and evaluation

The `Training/` and `Evaluation/` directories contain the training, unlearning, and benchmark evaluation code. See the documentation and configuration files in those directories for experiment-specific commands. Record the model checkpoint, dataset variant, partition, unlearning objective, benchmark, and random seed when reproducing or comparing results.

The paper evaluates with LVLMs and vision-language models and reports results on benchmarks including CHAIR, POPE, Causal-HalBench, MM-SpuBench, AMBER, MMHal, and Waterbirds. Refer to the paper and the original benchmark documentation for exact protocols and metrics.

## Citation

If you use PURGE, please cite the paper:

```bibtex
@inproceedings{sarker2026purge,
  title     = {Partition-Aware Unlearning for Removing Spurious Correlations in Large Vision-Language Models},
  author    = {Sarker, Aditi and Shah, Nazreen and Ibn Sultan, Rafi and Jang, Rhongho and Zhu, Dongxiao and Khanduri, Prashant},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgments

This work uses the MS COCO and Waterbirds datasets and evaluates models with resources including CHAIR, POPE, Causal-HalBench, MM-SpuBench, AMBER, and MMHal. We thank the original authors and maintainers for making these datasets and benchmarks available. Please cite the original works when using these resources; their citations are listed in the paper and the corresponding dataset or benchmark documentation.
