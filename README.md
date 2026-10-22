# PURGE: Partition-Aware Unlearning for Removing Spurious Correlations in Large Vision-Language Models

This repository contains the code and dataset resources for **PURGE**, a framework for diagnosing and mitigating object–background shortcuts in large vision-language models (LVLMs).

![Overall PURGE framework](image/overall_diagram.png)

*Figure 1: Overall PURGE framework, from structured dataset construction to partition-aware unlearning.*

![PURGE dataset construction diagram](image/dataset_diagram.png)

*Figure 2: PURGE-D, PURGE-H, and PURGE-M dataset construction strategies.*

## Paper

**Partition-Aware Unlearning for Removing Spurious Correlations in Large Vision-Language Models**  
Aditi Sarker, Nazreen Shah, Rafi Ibn Sultan, Rhongho Jang, Dongxiao Zhu, and Prashant Khanduri  
NeurIPS 2026, Evaluations and Datasets Track.

## Overview

LVLMs can learn that an object is likely to appear with a particular background—for example, boats with water—and may rely on that context instead of object evidence. PURGE constructs controlled retain/forget partitions to measure and reduce this shortcut reliance while preserving object-grounded reasoning.

PURGE has two stages:

1. **Structured dataset construction:** create partitions that separate object evidence from contextual cues.
2. **Partition-aware unlearning:** use the partitions to suppress spurious associations while retaining useful object knowledge. The paper evaluates Gradient Ascent (GA), KL-regularized unlearning (KL), and Negative Preference Optimization (NPO).

### PURGE dataset construction strategies

| Partition | Strategy | Description |
|---|---|---|
| **PURGE-D** | Data-level disentanglement | Uses object masking and inpainting to separate object and background information. |
| **PURGE-H** | Hybrid, behavior-driven partitioning | Combines annotations, model predictions, and counterfactual inputs to identify shortcut-driven behavior. |
| **PURGE-M** | Model-informed semantic partitioning | Uses model behavior and semantic context cues to identify object- and cue-related examples. |

## Repository layout

```text
PURGE/
├── Dataset_creation/   # PURGE-D, PURGE-H, and PURGE-M data construction
├── Evaluation/         # Benchmark evaluation code
├── Training/           # Unlearning and model-training code
├── image/              # Overall framework and dataset diagrams
├── README.md
└── requirements.txt
```

## Dataset

The PURGE dataset is available on Hugging Face: [Krishnaaditi1/PURGE](https://huggingface.co/datasets/Krishnaaditi1/PURGE).

It contains the **PURGE-D**, **PURGE-H**, and **PURGE-M** partitions. The released metadata contains 47,838 PURGE-D records, 16,700 PURGE-H records, and 20,633 PURGE-M records. The corresponding archives are approximately 4.6 GB, 2.4 GB, and 3.2 GB.

For experiments that construct PURGE examples from COCO, first obtain the COCO images and annotations from the official source and set their local paths in the commands below. Follow the terms of use for COCO and any other source datasets or model checkpoints.

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The model-informed pipeline also requires a local vision-language model checkpoint and the detector/model dependencies listed in `requirements.txt`. Use paths to files available on your machine; do not commit credentials or private machine paths.

## Generate PURGE-M

The following command shows the configurable PURGE-M invocation. Replace the example paths with your local COCO and model paths:

```bash
python Dataset_creation/PURGE_M.py \
  --coco_image_root /path/to/coco/train2017 \
  --coco_ann_json /path/to/coco/annotations/instances_train2017.json \
  --output_root /path/to/output/PURGE_M \
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

To run it from a shell script, place the command in a `.sh` file, replace the paths with your own, and run:

```bash
chmod +x run_purge_m.sh
./run_purge_m.sh
```

See `Dataset_creation/`, `Training/`, and `Evaluation/` for the corresponding scripts and their available options.

## Evaluation

PURGE is evaluated in the paper with LLaVA-1.6-7B, Qwen3-VL-8B-Instruct, and Qwen3.5-9B, as well as CLIP for vision-language encoder experiments. The benchmark suite includes CHAIR, POPE, Causal-HalBench, MM-SpuBench, AMBER, MMHal, and Waterbirds. Evaluation scripts and benchmark-specific instructions are in `Evaluation/`.

## Citation

```bibtex
@inproceedings{sarker2026purge,
  title     = {Partition-Aware Unlearning for Removing Spurious Correlations in Large Vision-Language Models},
  author    = {Sarker, Aditi and Shah, Nazreen and Ibn Sultan, Rafi and Jang, Rhongho and Zhu, Dongxiao and Khanduri, Prashant},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgments

Please cite the original datasets, benchmarks, and pretrained models used in your experiments. The PURGE paper provides the full references and experimental details.
