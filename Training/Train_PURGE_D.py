#!/usr/bin/env python3
"""Unified launcher for the supplied PURGE-D LLaVA and Qwen3-VL trainers.

IMPORTANT: The supplied implementations are PURGE-D (object-only retain,
background-only forget, and object-absent forget_neg), not PURGE-H. PURGE-H
requires behavior-driven partitions constructed from model predictions and
counterfactual inputs. This launcher does not relabel PURGE-D as PURGE-H.

Usage:
  python Train_PURGE_D.py --model_family llava <LLaVA trainer arguments>
  python Train_PURGE_D.py --model_family qwen3_vl <Qwen trainer arguments>
"""
import argparse
import sys

_LLAVA_SOURCE = r'''#!/usr/bin/env python3
"""PURGE-D unlearning trainer for LLaVA-NeXT-Mistral-7B.

Expected data layout; --data_root points to images/train:

    images/train/
      retain/<class>/*       object-only images; answer: yes
      forget/<class>/*       object-removed context-only views; NPO suppresses yes
      forget_neg/<class>/*   object-absent hard negatives; supervise no
"""

import argparse
import contextlib
import glob
import inspect
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
PROJECTOR_KEYS = (
    "multi_modal_projector",
    "mm_projector",
    "vision_projector",
    "visual_projection",
    "vision_proj",
    "projector",
)
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


@dataclass
class Example:
    path: str
    class_name: str
    partition: str


class PurgeDTrainSet(Dataset):
    def __init__(self, data_root: str, partition: str,
                 max_per_class: int, seed: int):
        if partition not in {"retain", "forget", "forget_neg"}:
            raise ValueError(f"Unknown partition: {partition}")

        self.examples: List[Example] = []
        base = os.path.join(data_root, partition)
        if not os.path.isdir(base):
            return

        classes = sorted(
            name for name in os.listdir(base)
            if os.path.isdir(os.path.join(base, name))
        )
        offsets = {"retain": 0, "forget": 1, "forget_neg": 2}
        rng = random.Random(seed + offsets[partition])

        for class_name in classes:
            class_dir = os.path.join(base, class_name)
            paths = []
            for pattern in IMAGE_EXTENSIONS:
                paths.extend(
                    glob.glob(
                        os.path.join(class_dir, "**", pattern),
                        recursive=True,
                    )
                )
            paths = sorted(set(paths))

            if max_per_class > 0 and len(paths) > max_per_class:
                paths = sorted(rng.sample(paths, max_per_class))

            self.examples.extend(
                Example(path, class_name, partition) for path in paths
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict:
        example = self.examples[index]
        with Image.open(example.path) as image_file:
            image = image_file.convert("RGB")

        # For `forget`, yes is the spurious response NPO suppresses, not its ground-truth label.
        answer = "no" if example.partition == "forget_neg" else "yes"
        target = example.class_name.replace("_", " ")
        question = f"Is there a {target} in the image?"

        return {
            "image": image,
            "path": example.path,
            "class_name": example.class_name,
            "partition": example.partition,
            "question": question,
            "answer": answer,
        }


def one_item(batch):
    if len(batch) != 1:
        raise ValueError("This trainer uses micro-batch size 1.")
    return batch[0]


def cycle(loader):
    while True:
        for item in loader:
            yield item


def llava_mistral_device_map(first_gpu_layers: int = 10) -> Dict[str, int]:
    """Shard LLaVA-NeXT-Mistral-7B over two visible GPUs."""
    if not 0 < first_gpu_layers < 32:
        raise ValueError("first_gpu_layers must be between 1 and 31")

    device_map = {
        "model.vision_tower": 0,
        "model.multi_modal_projector": 0,
        "model.image_newline": 0,
        "model.language_model.embed_tokens": 0,
        "model.language_model.rotary_emb": 0,
        "model.language_model.norm": 1,
        "lm_head": 1,
    }
    device_map.update({
        f"model.language_model.layers.{i}": 0
        for i in range(first_gpu_layers)
    })
    device_map.update({
        f"model.language_model.layers.{i}": 1
        for i in range(first_gpu_layers, 32)
    })
    return device_map


def load_vlm(model_path: str, dtype: torch.dtype, device_map=None):
    errors = []

    try:
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        print("[model] loaded with AutoModelForImageTextToText", flush=True)
        return model
    except Exception as exc:
        errors.append(f"AutoModelForImageTextToText: {exc!r}")

    try:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        print("[model] loaded with AutoModelForVision2Seq", flush=True)
        return model
    except Exception as exc:
        errors.append(f"AutoModelForVision2Seq: {exc!r}")

    try:
        from transformers import LlavaNextForConditionalGeneration
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
        print("[model] loaded with LlavaNextForConditionalGeneration", flush=True)
        return model
    except Exception as exc:
        errors.append(f"LlavaNextForConditionalGeneration: {exc!r}")

    raise RuntimeError(
        "Could not load the model. Loader errors:\n" + "\n".join(errors)
    )


class PurgeDTrainer:
    def __init__(self, args):
        self.args = args

        if not torch.cuda.is_available() and args.device.startswith("cuda"):
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )

        if torch.cuda.device_count() < 2 and args.device_map != "balanced":
            raise RuntimeError(
                "This LLaVA run expects two visible GPUs. "
                "Set CUDA_VISIBLE_DEVICES=0,1."
            )

        self.device = torch.device(args.device)
        self.dtype = get_dtype(args.dtype)
        self.processor = AutoProcessor.from_pretrained(args.model_path)

        if args.llava_single_crop:
            image_processor = getattr(self.processor, "image_processor", None)
            if image_processor is None or not hasattr(
                image_processor, "image_grid_pinpoints"
            ):
                raise RuntimeError(
                    "--llava_single_crop requires a LLaVA-NeXT image processor."
                )
            image_processor.image_grid_pinpoints = [[336, 336]]

        if args.device_map == "llava_mistral_10_22":
            device_map = llava_mistral_device_map(first_gpu_layers=10)
        elif args.device_map == "llava_mistral_8_24":
            device_map = llava_mistral_device_map(first_gpu_layers=8)
        else:
            device_map = args.device_map

        self.model = load_vlm(
            args.model_path, self.dtype, device_map=device_map
        )

        if args.llava_single_crop:
            model_config = getattr(self.model, "config", None)
            if model_config is None or not hasattr(
                model_config, "image_grid_pinpoints"
            ):
                raise RuntimeError(
                    "Single-crop mode requires "
                    "model.config.image_grid_pinpoints."
                )
            # Keep processor and model patch counts consistent.
            model_config.image_grid_pinpoints = [[336, 336]]
            print(
                "[processor] image_grid_pinpoints=[[336, 336]]",
                flush=True,
            )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
            inference_mode=False,
        )
        self.model = get_peft_model(self.model, lora)

        if device_map is None:
            self.model.to(self.device)

        embedding = self.model.get_base_model().get_input_embeddings()
        self.input_device = next(embedding.parameters()).device

        base_model = self.model.get_base_model()
        if device_map:
            print(
                f"[model] device_map="
                f"{getattr(base_model, 'hf_device_map', device_map)}",
                flush=True,
            )

        self.projector_params = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if any(key in name.lower() for key in PROJECTOR_KEYS)
        }

        if args.train_projector:
            if not self.projector_params:
                raise RuntimeError(
                    "--train_projector was set, but no LLaVA projector parameters "
                    "were found; refusing a LoRA-only run."
                )
            projector_count = sum(
                parameter.numel()
                for parameter in self.projector_params.values()
            )
            print(
                f"[projector] tensors={len(self.projector_params)} "
                f"parameters={projector_count:,}",
                flush=True,
            )
            for parameter in self.projector_params.values():
                parameter.requires_grad_(True)

        self.projector_reference = (
            {
                name: parameter.detach().clone()
                for name, parameter in self.projector_params.items()
            }
            if args.train_projector else {}
        )

        if args.grad_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            if hasattr(base_model, "config"):
                base_model.config.use_cache = False

        self.trainable = [
            parameter for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        if not self.trainable:
            raise RuntimeError(
                "No trainable parameters found; check LoRA target module names."
            )

        print(
            f"[model] trainable="
            f"{sum(parameter.numel() for parameter in self.trainable):,}",
            flush=True,
        )

    def _messages(self, image, question: str,
                  answer: Optional[str] = None):
        user_content = [
            {"type": "image", "image": image},
            {"type": "text", "text": question},
        ]
        messages = [{"role": "user", "content": user_content}]
        if answer is not None:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            })
        return messages

    def _template(self, image, question: str,
                  answer: Optional[str], generation: bool):
        if not hasattr(self.processor, "apply_chat_template"):
            raise RuntimeError(
                "This processor does not provide apply_chat_template()."
            )

        messages = self._messages(image, question, answer)
        try:
            batch = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=generation,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not format the image/question using this model's "
                "chat template. Confirm that the model and processor paths "
                "refer to the same LLaVA checkpoint."
            ) from exc

        batch.pop("token_type_ids", None)

        if answer is not None:
            prompt = self.processor.apply_chat_template(
                self._messages(image, question, None),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt.pop("token_type_ids", None)
            prompt_length = int(prompt["input_ids"].shape[1])

            labels = batch["input_ids"].clone()
            if prompt_length > labels.shape[1]:
                raise RuntimeError(
                    "The prompt is longer than the prompt-plus-answer sequence."
                )
            labels[:, :prompt_length] = -100
            if not (labels != -100).any():
                raise RuntimeError("No answer tokens were selected for loss.")
            batch["labels"] = labels

        return self._to_device(batch)

    def _to_device(self, batch):
        return {
            key: value.to(self.input_device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _forward(self, batch):
        base = self.model.get_base_model()
        try:
            accepted = set(inspect.signature(base.forward).parameters)
            batch = {
                key: value for key, value in batch.items()
                if key in accepted or key == "labels"
            }
        except (TypeError, ValueError):
            pass
        return self.model(**batch, use_cache=False)

    def _ce(self, sample, answer: str):
        batch = self._template(
            sample["image"], sample["question"], answer, generation=False
        )
        return self._forward(batch).loss

    def _sequence_logprob(self, sample, answer: str):
        batch = self._template(
            sample["image"], sample["question"], answer, generation=False
        )
        labels = batch["labels"]
        model_inputs = {
            key: value for key, value in batch.items() if key != "labels"
        }
        logits = self._forward(model_inputs).logits[:, :-1, :].float()
        targets = labels[:, 1:]
        mask = targets.ne(-100)
        safe_targets = targets.masked_fill(~mask, 0)

        token_logprobs = F.log_softmax(logits, dim=-1).gather(
            -1, safe_targets.unsqueeze(-1)
        ).squeeze(-1)

        # Sum answer-token log probabilities for the NPO sequence ratio.
        return (token_logprobs * mask).sum(dim=-1)

    @contextlib.contextmanager
    def _reference_model_state(self):
        """Evaluate the original model without a second 7B model copy."""
        projector_params = (
            self.projector_params if self.args.train_projector else {}
        )
        tuned_projector = {
            name: parameter.detach().clone()
            for name, parameter in projector_params.items()
        }

        try:
            with torch.no_grad():
                for name, parameter in projector_params.items():
                    parameter.copy_(self.projector_reference[name])
            with self.model.disable_adapter():
                yield
        finally:
            with torch.no_grad():
                for name, parameter in projector_params.items():
                    parameter.copy_(tuned_projector[name])

    def _retain_kl(self, retain):
        """KL(P_updated || P_original) over the retain answer tokens."""
        batch = self._template(
            retain["image"],
            retain["question"],
            "yes",
            generation=False,
        )
        labels = batch["labels"]
        model_inputs = {
            key: value for key, value in batch.items() if key != "labels"
        }

        with self._reference_model_state(), torch.no_grad():
            reference_logits = (
                self._forward(model_inputs).logits[:, :-1, :].float()
            )
        current_logits = (
            self._forward(model_inputs).logits[:, :-1, :].float()
        )

        answer_mask = labels[:, 1:].ne(-100)
        log_p = F.log_softmax(current_logits, dim=-1)
        log_q = F.log_softmax(reference_logits, dim=-1)
        token_kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)

        return (
            (token_kl * answer_mask).sum()
            / answer_mask.sum().clamp_min(1)
        )

    @staticmethod
    def _clip_grad_norm_across_devices(parameters, max_norm: float) -> float:
        gradients = [
            parameter.grad for parameter in parameters
            if parameter.grad is not None
        ]
        if not gradients:
            return 0.0

        device_norms = []
        devices = sorted({gradient.device for gradient in gradients}, key=str)
        for device in devices:
            local_sq_norm = torch.zeros(
                (), device=device, dtype=torch.float32
            )
            for gradient in gradients:
                if gradient.device == device:
                    local_sq_norm += gradient.detach().float().pow(2).sum()
            device_norms.append(local_sq_norm.sqrt().cpu())

        total_norm = torch.stack(device_norms).norm(2).item()
        if max_norm > 0 and total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for gradient in gradients:
                gradient.mul_(scale)
        return total_norm

    def _npo_forget(self, forget, forget_answer: str):
        with self._reference_model_state(), torch.no_grad():
            logp_reference = self._sequence_logprob(
                forget, forget_answer
            )

        logp_current = self._sequence_logprob(forget, forget_answer)
        beta = self.args.beta

        # Paper Eq. (5):
        # (2*lambda/beta) * log(1 + (P_current/P_original)**beta)
        return (
            (2.0 * self.args.lambda_forget / beta)
            * F.softplus(beta * (logp_current - logp_reference))
        ).mean()

    def _loss(self, retain, forget, forget_neg=None):
        logs = {}
        total_loss = 0.0

        def backward_term(name, term, weight=1.0):
            nonlocal total_loss
            weighted_term = term * weight
            weighted_term.backward()
            total_loss += float(weighted_term.detach().float().item())
            logs[name] = term.detach()

        if self.args.algo == "ga":
            # L_retain = -log P(yes|retain)
            # L_forget = +lambda*log P(yes|forget)
            backward_term("retain_ce", self._ce(retain, "yes"))
            backward_term(
                "forget_yes_ce",
                self._ce(forget, "yes"),
                -self.args.lambda_forget,
            )

        elif self.args.algo == "kl":
            backward_term(
                "forget_yes_ce",
                self._ce(forget, "yes"),
                -self.args.lambda_forget,
            )
            backward_term(
                "retain_kl",
                self._retain_kl(retain),
                self.args.beta,
            )

        elif self.args.algo == "npo":
            backward_term("retain_ce", self._ce(retain, "yes"))
            backward_term(
                "npo_forget",
                self._npo_forget(forget, "yes"),
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.args.algo}")

        # PURGE-D's explicit object-absent negative supervision.
        if forget_neg is not None:
            backward_term(
                "forget_neg_no_ce",
                self._ce(forget_neg, "no"),
                self.args.gamma,
            )

        return total_loss, logs

    def save_checkpoint(self, output_dir: str,
                        checkpoint_epoch: Optional[object] = None):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)

        if self.args.train_projector and self.projector_params:
            projector_state = {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in self.projector_params.items()
            }
            torch.save(
                projector_state,
                os.path.join(output_dir, "projector_trainables.pt"),
            )

        self.processor.save_pretrained(output_dir)

        metadata = {
            "method": "PURGE-D",
            "algorithm": self.args.algo,
            "checkpoint_epoch": checkpoint_epoch,
            "model_path": self.args.model_path,
            "data_root": self.args.data_root,
            "train_projector": self.args.train_projector,
            "epochs": self.args.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "max_per_class": self.args.max_per_class,
            "seed": self.args.seed,
            "lambda_forget": self.args.lambda_forget,
            "beta": self.args.beta,
            "gamma": self.args.gamma,
            "lora_targets": LORA_TARGETS,
        }
        with open(
            os.path.join(output_dir, "purge_d_train_meta.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2)

    def train(self):
        retain_ds = PurgeDTrainSet(
            self.args.data_root, "retain",
            self.args.max_per_class, self.args.seed,
        )
        forget_ds = PurgeDTrainSet(
            self.args.data_root, "forget",
            self.args.max_per_class, self.args.seed,
        )
        neg_ds = PurgeDTrainSet(
            self.args.data_root, "forget_neg",
            self.args.max_per_class, self.args.seed,
        )

        if len(retain_ds) == 0 or len(forget_ds) == 0:
            raise RuntimeError(
                f"Expected non-empty {self.args.data_root}/retain and "
                f"{self.args.data_root}/forget directories."
            )
        if len(neg_ds) == 0 and not self.args.allow_missing_forget_neg:
            raise RuntimeError(
                f"Missing {self.args.data_root}/forget_neg. "
                "PURGE-D uses this partition for explicit 'no' supervision."
            )

        retain_loader = DataLoader(
            retain_ds, batch_size=1, shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=one_item, pin_memory=True,
        )
        forget_loader = DataLoader(
            forget_ds, batch_size=1, shuffle=True,
            num_workers=self.args.num_workers,
            collate_fn=one_item, pin_memory=True,
        )
        neg_loader = (
            DataLoader(
                neg_ds, batch_size=1, shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=one_item, pin_memory=True,
            )
            if len(neg_ds) else None
        )

        self.steps_per_epoch = max(len(retain_loader), len(forget_loader))
        if self.args.steps_per_epoch > 0:
            self.steps_per_epoch = self.args.steps_per_epoch

        retain_iter = cycle(retain_loader)
        forget_iter = cycle(forget_loader)
        neg_iter = cycle(neg_loader) if neg_loader is not None else None

        lora_params = [
            parameter for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        projector_params = [
            parameter for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]

        if not lora_params:
            raise RuntimeError("No trainable LLaVA LoRA parameters were found.")
        if self.args.train_projector and not projector_params:
            raise RuntimeError(
                "Projector training was requested, but no trainable projector "
                "parameters reached the optimizer."
            )
        print(
            f"[optimizer] LoRA parameters="
            f"{sum(parameter.numel() for parameter in lora_params):,}; "
            f"projector parameters="
            f"{sum(parameter.numel() for parameter in projector_params):,}",
            flush=True,
        )

        parameter_groups = []
        if lora_params:
            parameter_groups.append({
                "params": lora_params,
                "lr": self.args.lr_lora,
            })
        if projector_params:
            parameter_groups.append({
                "params": projector_params,
                "lr": self.args.lr_projector,
            })
        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=self.args.weight_decay,
        )

        os.makedirs(self.args.output_dir, exist_ok=True)
        print(
            f"[data] retain={len(retain_ds)} "
            f"forget={len(forget_ds)} forget_neg={len(neg_ds)}",
            flush=True,
        )
        print(
            f"[train] epochs={self.args.epochs}, "
            f"steps/epoch={self.steps_per_epoch}, algo={self.args.algo}",
            flush=True,
        )

        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()

        for epoch in range(1, self.args.epochs + 1):
            for step in range(1, self.steps_per_epoch + 1):
                global_step += 1
                retain = next(retain_iter)
                forget = next(forget_iter)
                forget_neg = next(neg_iter) if neg_iter is not None else None

                loss, logs = self._loss(retain, forget, forget_neg)

                if self.args.max_grad_norm > 0:
                    self._clip_grad_norm_across_devices(
                        self.trainable, self.args.max_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if global_step % self.args.log_every == 0 or global_step == 1:
                    log_text = " ".join(
                        f"{key}={value.item():.4f}"
                        for key, value in logs.items()
                    )
                    print(
                        f"[epoch {epoch}/{self.args.epochs} "
                        f"step {step}/{self.steps_per_epoch}] "
                        f"loss={loss:.5f} {log_text} "
                        f"elapsed={time.time() - start_time:.1f}s",
                        flush=True,
                    )

            epoch_dir = os.path.join(
                self.args.output_dir, f"epoch_{epoch}"
            )
            self.save_checkpoint(epoch_dir, checkpoint_epoch=epoch)

        self.save_checkpoint(self.args.output_dir, checkpoint_epoch="final")
        print(
            f"[done] final checkpoint saved to {self.args.output_dir}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="PURGE-D GA/KL/NPO LoRA training"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--data_root",
        required=True,
        help="images/train containing retain/, forget/, forget_neg/",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--algo", choices=["ga", "kl", "npo"], required=True
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--device_map",
        choices=[
            "balanced",
            "llava_mistral_8_24",
            "llava_mistral_10_22",
        ],
        default="llava_mistral_10_22",
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=0,
        help="0 uses the full retain/forget partition size",
    )
    parser.add_argument(
        "--max_per_class",
        type=int,
        default=100,
        help="100 per class approximates 8,000 samples over 80 classes; 0 disables the cap",
    )
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_lora", type=float, default=2e-5)
    parser.add_argument("--lr_projector", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_forget", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--train_projector", action="store_true")
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument(
        "--llava_single_crop",
        action="store_true",
        help=(
            "Use one 336x336 LLaVA crop to reduce memory. "
            "This changes the processor's default multi-crop preprocessing."
        ),
    )
    parser.add_argument(
        "--allow_missing_forget_neg",
        action="store_true",
        help="Diagnostic option; omit PURGE-D negative supervision.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.beta <= 0:
        raise ValueError("--beta must be positive")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.steps_per_epoch < 0:
        raise ValueError("--steps_per_epoch must be 0 or positive")
    if args.max_per_class < 0:
        raise ValueError("--max_per_class must be 0 or positive")

    set_seed(args.seed)
    trainer = PurgeDTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()'''

_QWEN3_VL_SOURCE = r'''
#!/usr/bin/env python3
"""PURGE-D training for Qwen3-VL using LoRA and optional visual merger training."""

import argparse
import contextlib
import glob
import json
import os
import random
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForImageTextToText, AutoProcessor


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")

PROJECTOR_KEYS = (
    "visual.merger",
    "deepstack_merger",
    "multi_modal_projector",
    "mm_projector",
    "vision_projector",
    "visual_projection",
    "vision_proj",
)

LORA_CANDIDATES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_dtype(name):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


@dataclass
class Example:
    path: str
    class_name: str
    partition: str


class PurgeDTrainSet(Dataset):
    def __init__(self, data_root, partition, max_per_class, seed):
        if partition not in {"retain", "forget", "forget_neg"}:
            raise ValueError(f"Unknown partition: {partition}")

        self.examples = []
        base = os.path.join(data_root, partition)
        if not os.path.isdir(base):
            return

        offset = {"retain": 0, "forget": 1, "forget_neg": 2}[partition]
        rng = random.Random(seed + offset)

        classes = sorted(
            name for name in os.listdir(base)
            if os.path.isdir(os.path.join(base, name))
        )

        for class_name in classes:
            class_dir = os.path.join(base, class_name)
            paths = []
            for pattern in IMAGE_EXTENSIONS:
                paths.extend(
                    glob.glob(
                        os.path.join(class_dir, "**", pattern),
                        recursive=True,
                    )
                )
            paths = sorted(set(paths))

            if max_per_class > 0 and len(paths) > max_per_class:
                paths = sorted(rng.sample(paths, max_per_class))

            self.examples.extend(
                Example(path, class_name, partition) for path in paths
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        with Image.open(example.path) as image_file:
            image = image_file.convert("RGB")

        # For `forget`, yes is the spurious response NPO suppresses, not its ground-truth label.
        answer = "no" if example.partition == "forget_neg" else "yes"
        class_text = example.class_name.replace("_", " ")
        question = f"Is there a {class_text} in the image?"

        return {
            "image": image,
            "path": example.path,
            "class_name": example.class_name,
            "partition": example.partition,
            "question": question,
            "answer": answer,
        }


def one_item(batch):
    if len(batch) != 1:
        raise ValueError("This trainer requires micro-batch size 1.")
    return batch[0]


def cycle(loader):
    while True:
        for item in loader:
            yield item


def find_lora_targets(model):
    available = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    targets = [name for name in LORA_CANDIDATES if name in available]
    if not targets:
        raise RuntimeError(
            "No supported Qwen linear modules were found for LoRA."
        )
    return targets


class PurgeDTrainer:
    def __init__(self, args):
        self.args = args

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required.")
        if torch.cuda.device_count() < 2:
            raise RuntimeError(
                "This script requires two visible GPUs for balanced loading. "
                "Run with CUDA_VISIBLE_DEVICES=0,1."
            )

        dtype = get_dtype(args.dtype)
        self.processor = AutoProcessor.from_pretrained(args.model_path)
        self.model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            dtype=dtype,
            low_cpu_mem_usage=True,
            device_map=args.device_map,
        )
        self.model.train()
        print("[model] loaded with AutoModelForImageTextToText", flush=True)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        lora_targets = find_lora_targets(self.model)
        print(f"[LoRA] target modules={lora_targets}", flush=True)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_targets,
            inference_mode=False,
        )
        self.model = get_peft_model(self.model, lora_config)

        base_model = self.model.get_base_model()
        self.input_device = next(
            base_model.get_input_embeddings().parameters()
        ).device

        self.projector_params = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if any(key in name.lower() for key in PROJECTOR_KEYS)
        }

        if args.train_projector:
            if not self.projector_params:
                raise RuntimeError(
                    "Projector training was requested, but no Qwen visual "
                    "merger parameters were found. Refusing a LoRA-only run."
                )

            for parameter in self.projector_params.values():
                parameter.requires_grad_(True)

            projector_count = sum(
                parameter.numel()
                for parameter in self.projector_params.values()
            )
            print(
                f"[projector] tensors={len(self.projector_params)} "
                f"parameters={projector_count:,}",
                flush=True,
            )
            for name in self.projector_params:
                print(f"[projector] trainable: {name}", flush=True)

        self.projector_reference = (
            {
                name: parameter.detach().clone()
                for name, parameter in self.projector_params.items()
            }
            if args.train_projector
            else {}
        )

        if args.grad_checkpointing:
            self.model.gradient_checkpointing_enable()
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            base_model.config.use_cache = False

        self.lora_params = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        self.projector_trainables = [
            parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and "lora_" not in name
        ]
        self.trainable = self.lora_params + self.projector_trainables

        if not self.lora_params:
            raise RuntimeError("No trainable LoRA parameters were found.")
        if args.train_projector and not self.projector_trainables:
            raise RuntimeError(
                "Projector flag is on but no projector is trainable."
            )

        print(
            f"[model] trainable LoRA="
            f"{sum(p.numel() for p in self.lora_params):,}; "
            f"projector="
            f"{sum(p.numel() for p in self.projector_trainables):,}",
            flush=True,
        )

    def _messages(self, image, question, answer=None):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }]
        if answer is not None:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            })
        return messages

    def _to_device(self, batch):
        return {
            key: value.to(self.input_device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    def _make_batch(self, sample, answer=None):
        batch = self.processor.apply_chat_template(
            self._messages(sample["image"], sample["question"], answer),
            tokenize=True,
            add_generation_prompt=(answer is None),
            return_dict=True,
            return_tensors="pt",
        )
        batch.pop("token_type_ids", None)

        if answer is not None:
            prompt = self.processor.apply_chat_template(
                self._messages(sample["image"], sample["question"]),
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            prompt.pop("token_type_ids", None)

            labels = batch["input_ids"].clone()
            prompt_length = int(prompt["input_ids"].shape[1])
            if prompt_length >= labels.shape[1]:
                raise RuntimeError("No assistant answer tokens were found.")
            labels[:, :prompt_length] = -100
            batch["labels"] = labels

        return self._to_device(batch)

    def _forward(self, batch):
        return self.model(**batch, use_cache=False)

    def _ce(self, sample, answer):
        return self._forward(self._make_batch(sample, answer)).loss

    def _sequence_logprob(self, sample, answer):
        batch = self._make_batch(sample, answer)
        labels = batch["labels"]
        inputs = {
            key: value for key, value in batch.items()
            if key != "labels"
        }

        logits = self._forward(inputs).logits[:, :-1, :].float()
        targets = labels[:, 1:]
        mask = targets.ne(-100)
        safe_targets = targets.masked_fill(~mask, 0)
        token_logp = F.log_softmax(logits, dim=-1).gather(
            -1, safe_targets.unsqueeze(-1)
        ).squeeze(-1)

        return (token_logp * mask).sum(dim=-1)

    @contextlib.contextmanager
    def _reference_state(self):
        tuned_projector = (
            {
                name: parameter.detach().clone()
                for name, parameter in self.projector_params.items()
            }
            if self.args.train_projector
            else {}
        )

        try:
            if self.args.train_projector:
                with torch.no_grad():
                    for name, parameter in self.projector_params.items():
                        parameter.copy_(self.projector_reference[name])

            with self.model.disable_adapter():
                yield
        finally:
            if self.args.train_projector:
                with torch.no_grad():
                    for name, parameter in self.projector_params.items():
                        parameter.copy_(tuned_projector[name])

    def _retain_kl(self, retain):
        batch = self._make_batch(retain, "yes")
        labels = batch["labels"]
        inputs = {
            key: value for key, value in batch.items()
            if key != "labels"
        }

        with self._reference_state(), torch.no_grad():
            reference_logits = self._forward(inputs).logits[:, :-1, :].float()

        current_logits = self._forward(inputs).logits[:, :-1, :].float()
        mask = labels[:, 1:].ne(-100)

        log_p = F.log_softmax(current_logits, dim=-1)
        log_q = F.log_softmax(reference_logits, dim=-1)
        token_kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1)

        return (token_kl * mask).sum() / mask.sum().clamp_min(1)

    def _npo_forget(self, forget):
        with self._reference_state(), torch.no_grad():
            logp_reference = self._sequence_logprob(forget, "yes")

        logp_current = self._sequence_logprob(forget, "yes")
        beta = self.args.beta

        return (
            2.0 * self.args.lambda_forget / beta
            * F.softplus(beta * (logp_current - logp_reference))
        ).mean()

    def _loss(self, retain, forget, forget_neg):
        logs = {}
        total_value = 0.0

        def backward_term(name, term, weight=1.0):
            nonlocal total_value
            weighted = term * weight
            weighted.backward()
            total_value += float(weighted.detach().float().item())
            logs[name] = term.detach()

        if self.args.algo == "ga":
            backward_term("retain_ce", self._ce(retain, "yes"))
            backward_term(
                "forget_yes_ce",
                self._ce(forget, "yes"),
                -self.args.lambda_forget,
            )
        elif self.args.algo == "kl":
            backward_term(
                "forget_yes_ce",
                self._ce(forget, "yes"),
                -self.args.lambda_forget,
            )
            backward_term(
                "retain_kl",
                self._retain_kl(retain),
                self.args.beta,
            )
        elif self.args.algo == "npo":
            backward_term("retain_ce", self._ce(retain, "yes"))
            backward_term("npo_forget", self._npo_forget(forget))
        else:
            raise ValueError(f"Unknown algorithm: {self.args.algo}")

        backward_term(
            "forget_neg_no_ce",
            self._ce(forget_neg, "no"),
            self.args.gamma,
        )
        return total_value, logs

    @staticmethod
    def _clip_grad_norm(parameters, max_norm):
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            return 0.0

        device_norms = []
        devices = sorted({grad.device for grad in grads}, key=str)
        for device in devices:
            norm_sq = torch.zeros((), device=device, dtype=torch.float32)
            for grad in grads:
                if grad.device == device:
                    norm_sq += grad.detach().float().pow(2).sum()
            device_norms.append(norm_sq.sqrt().cpu())

        total_norm = torch.stack(device_norms).norm(2).item()
        if max_norm > 0 and total_norm > max_norm:
            scale = max_norm / (total_norm + 1e-6)
            for grad in grads:
                grad.mul_(scale)
        return total_norm

    def save_checkpoint(self, output_dir, epoch):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.processor.save_pretrained(output_dir)

        if self.args.train_projector:
            state = {
                name: parameter.detach().cpu().contiguous()
                for name, parameter in self.projector_params.items()
            }
            torch.save(
                state,
                os.path.join(output_dir, "projector_trainables.pt"),
            )

        metadata = {
            "method": "PURGE-D",
            "algorithm": self.args.algo,
            "checkpoint_epoch": epoch,
            "model_path": self.args.model_path,
            "data_root": self.args.data_root,
            "train_projector": self.args.train_projector,
            "epochs": self.args.epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "max_per_class": self.args.max_per_class,
            "seed": self.args.seed,
            "lambda_forget": self.args.lambda_forget,
            "beta": self.args.beta,
            "gamma": self.args.gamma,
        }
        with open(
            os.path.join(output_dir, "purge_d_train_meta.json"),
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2)

    def train(self):
        retain_ds = PurgeDTrainSet(
            self.args.data_root, "retain",
            self.args.max_per_class, self.args.seed,
        )
        forget_ds = PurgeDTrainSet(
            self.args.data_root, "forget",
            self.args.max_per_class, self.args.seed,
        )
        neg_ds = PurgeDTrainSet(
            self.args.data_root, "forget_neg",
            self.args.max_per_class, self.args.seed,
        )

        if not retain_ds or not forget_ds or not neg_ds:
            raise RuntimeError(
                f"Need nonempty retain/, forget/, and forget_neg/ under "
                f"{self.args.data_root}"
            )

        def loader(dataset):
            return DataLoader(
                dataset,
                batch_size=1,
                shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=one_item,
                pin_memory=True,
            )

        retain_loader = loader(retain_ds)
        forget_loader = loader(forget_ds)
        neg_loader = loader(neg_ds)

        self.steps_per_epoch = max(len(retain_loader), len(forget_loader))
        if self.args.steps_per_epoch > 0:
            self.steps_per_epoch = self.args.steps_per_epoch

        retain_iter = cycle(retain_loader)
        forget_iter = cycle(forget_loader)
        neg_iter = cycle(neg_loader)

        parameter_groups = [{
            "params": self.lora_params,
            "lr": self.args.lr_lora,
        }]
        if self.projector_trainables:
            parameter_groups.append({
                "params": self.projector_trainables,
                "lr": self.args.lr_projector,
            })

        optimizer = torch.optim.AdamW(
            parameter_groups,
            weight_decay=self.args.weight_decay,
        )

        os.makedirs(self.args.output_dir, exist_ok=True)
        print(
            f"[data] retain={len(retain_ds)} forget={len(forget_ds)} "
            f"forget_neg={len(neg_ds)}",
            flush=True,
        )
        print(
            f"[train] epochs={self.args.epochs} "
            f"steps_per_epoch={self.steps_per_epoch} algo={self.args.algo}",
            flush=True,
        )

        optimizer.zero_grad(set_to_none=True)
        start_time = time.time()
        global_step = 0

        for epoch in range(1, self.args.epochs + 1):
            for step in range(1, self.steps_per_epoch + 1):
                global_step += 1
                retain = next(retain_iter)
                forget = next(forget_iter)
                forget_neg = next(neg_iter)

                loss, logs = self._loss(retain, forget, forget_neg)

                if self.args.max_grad_norm > 0:
                    self._clip_grad_norm(
                        self.trainable, self.args.max_grad_norm
                    )

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if global_step % self.args.log_every == 0 or global_step == 1:
                    log_text = " ".join(
                        f"{name}={value.item():.4f}"
                        for name, value in logs.items()
                    )
                    print(
                        f"[epoch {epoch}/{self.args.epochs} "
                        f"step {step}/{self.steps_per_epoch}] "
                        f"loss={loss:.5f} {log_text} "
                        f"elapsed={time.time() - start_time:.1f}s",
                        flush=True,
                    )

            self.save_checkpoint(
                os.path.join(self.args.output_dir, f"epoch_{epoch}"),
                epoch,
            )

        self.save_checkpoint(self.args.output_dir, "final")
        print(
            f"[done] checkpoint saved to {self.args.output_dir}",
            flush=True,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="PURGE-D Qwen3-VL LoRA plus projector training."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--algo", choices=["ga", "kl", "npo"], required=True)
    parser.add_argument(
        "--device_map", choices=["balanced"], default="balanced"
    )
    parser.add_argument(
        "--dtype", choices=["bf16", "fp16", "fp32"], default="bf16"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--max_per_class", type=int, default=100)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr_lora", type=float, default=2e-5)
    parser.add_argument("--lr_projector", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lambda_forget", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--train_projector", action="store_true")
    parser.add_argument("--grad_checkpointing", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.beta <= 0:
        raise ValueError("--beta must be positive.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.steps_per_epoch < 0 or args.max_per_class < 0:
        raise ValueError(
            "steps_per_epoch and max_per_class cannot be negative."
        )

    set_seed(args.seed)
    PurgeDTrainer(args).train()


if __name__ == "__main__":
    main()'''


def main():
    parser = argparse.ArgumentParser(
        description="Unified launcher for the supplied PURGE-D VLM trainers.",
        add_help=False,
    )
    parser.add_argument(
        "--model_family", choices=("llava", "qwen3_vl"), required=True,
        help="Choose which model-specific PURGE-D implementation to run.",
    )
    if ("--help" in sys.argv[1:] or "-h" in sys.argv[1:]) and "--model_family" not in sys.argv[1:]:
        print("Usage: Train_PURGE_D.py --model_family {llava,qwen3_vl} <trainer arguments>")
        print("Pass --help after --model_family to see that trainer's options.")
        return
    args, forwarded = parser.parse_known_args()
    source = _LLAVA_SOURCE if args.model_family == "llava" else _QWEN3_VL_SOURCE
    # Let the selected implementation parse its own architecture-specific flags.
    sys.argv = [sys.argv[0], *forwarded]
    namespace = {"__name__": "__main__", "__file__": __file__}
    exec(compile(source, f"<PURGE-D-{args.model_family}>", "exec"), namespace)


if __name__ == "__main__":
    main()
