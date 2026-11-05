
#!/usr/bin/env python3


import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor


PROJECTOR_KEYS = (
    "multi_modal_projector",
    "mm_projector",
    "vision_projector",
    "visual_projection",
    "visual.merger",
    "deepstack_merger",
)


def normalize_yes_no(value):
    value = str(value).strip().lower()
    if value in {"yes", "1", "true"}:
        return "yes"
    if value in {"no", "0", "false"}:
        return "no"
    raise ValueError(f"Expected yes/no prediction, got {value!r}")


def load_metadata(metadata_path, data_root):
    retain, forget = [], []
    data_root = Path(data_root)

    with open(metadata_path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)

            split = str(row.get("split", "")).strip()
            partition = split.split("/", 1)[0].lower()
            if partition not in {"retain", "forget"}:
                raise ValueError(
                    f"Metadata line {line_number}: invalid split {split!r}"
                )

            file_name = row.get("file_name")
            target = row.get("target_object")
            prediction = row.get("prediction")
            if not file_name or not target or prediction is None:
                raise ValueError(
                    f"Metadata line {line_number} must contain "
                    "file_name, target_object, and prediction."
                )

            image_path = Path(file_name)
            if not image_path.is_absolute():
                image_path = data_root / image_path
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Metadata line {line_number}: image missing: {image_path}"
                )

            item = {
                "image": str(image_path),
                "question": f"Is there a {target} in the image?",
                # In PURGE-H, retain rows use correct predictions and forget
                # rows use the original model's erroneous prediction.
                "label": normalize_yes_no(prediction),
            }
            (retain if partition == "retain" else forget).append(item)

    if not retain or not forget:
        raise ValueError(
            f"Need retain and forget records; got "
            f"retain={len(retain)}, forget={len(forget)}."
        )
    return retain, forget


def load_model(model_path, dtype, device_map, llava_single_crop):
    processor = AutoProcessor.from_pretrained(model_path)

    if llava_single_crop:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None and hasattr(
            image_processor, "image_grid_pinpoints"
        ):
            image_processor.image_grid_pinpoints = [[336, 336]]
            print(
                "[processor] LLaVA single-crop mode: [[336, 336]]",
                flush=True,
            )

    base = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False

    # Keep LLaVA's model patch-count calculation consistent with the processor.
    if llava_single_crop:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is not None and hasattr(
            image_processor, "image_grid_pinpoints"
        ):
            base.config.image_grid_pinpoints = (
                image_processor.image_grid_pinpoints
            )
            print(
                "[processor] synchronized model image grid:",
                base.config.image_grid_pinpoints,
                flush=True,
            )

    # Reduce activation memory during training.
    if hasattr(base, "gradient_checkpointing_enable"):
        try:
            base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            base.gradient_checkpointing_enable()
        if hasattr(base, "enable_input_require_grads"):
            base.enable_input_require_grads()
        print("[model] gradient checkpointing enabled", flush=True)

    print("[model] device map:", getattr(base, "hf_device_map", None), flush=True)
    base.requires_grad_(False)

    # Restrict LoRA to language-model projections. The vision encoder stays frozen.
    lora_regex = (
        r".*language_model.*\."
        r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    )
    model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=lora_regex,
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )

    # Train the multimodal projector/fusion modules for these two backbones.
    projector_params = []
    for name, param in model.named_parameters():
        if any(key in name for key in PROJECTOR_KEYS):
            param.requires_grad_(True)
            projector_params.append((name, param))

    lora_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" in name
    ]
    if not lora_params:
        raise RuntimeError("No language-model LoRA parameters found.")
    if not projector_params:
        raise RuntimeError(
            "No projector/fusion parameters found. Check PROJECTOR_KEYS "
            "against this model's module names."
        )

    print(f"[model] loaded {model_path}", flush=True)
    print(
        f"[model] trainable LoRA="
        f"{sum(p.numel() for _, p in lora_params):,}",
        flush=True,
    )
    print(
        f"[model] trainable projector/fusion="
        f"{sum(p.numel() for _, p in projector_params):,}",
        flush=True,
    )
    return model, processor, lora_params, projector_params


def model_input_device(model):
    return next(model.get_input_embeddings().parameters()).device


def make_inputs(processor, item, device):
    with Image.open(item["image"]) as image_file:
        image = image_file.convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": item["question"]},
        ],
    }]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    batch.pop("token_type_ids", None)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def answer_token(processor, label, device):
    ids = processor.tokenizer.encode(
        " " + label.capitalize(),
        add_special_tokens=False,
    )
    if not ids:
        raise RuntimeError(f"Could not tokenize answer {label!r}")
    return torch.tensor([ids], dtype=torch.long, device=device)


def answer_logprob(model, processor, item, device):
    batch = make_inputs(processor, item, device)
    prompt_length = batch["input_ids"].shape[1]
    target = answer_token(processor, item["label"], device)

    batch["input_ids"] = torch.cat((batch["input_ids"], target), dim=1)
    if "attention_mask" in batch:
        batch["attention_mask"] = torch.cat(
            (batch["attention_mask"], torch.ones_like(target)),
            dim=1,
        )

    # Qwen3-VL requires one multimodal token type per input token.
    # The appended answer tokens are text, so their type ID is 0.
    if "mm_token_type_ids" in batch:
        mm_types = batch["mm_token_type_ids"]
        answer_types = torch.zeros_like(target, dtype=mm_types.dtype)
        batch["mm_token_type_ids"] = torch.cat(
            (mm_types, answer_types.to(mm_types.device)),
            dim=1,
        )

    output = model(**batch, use_cache=False)
    answer_logits = output.logits[
        :, prompt_length - 1 : prompt_length + target.shape[1] - 1
    ]
    log_probs = F.log_softmax(answer_logits.float(), dim=-1)
    return log_probs.gather(-1, target.unsqueeze(-1)).sum()


def output_distribution_logprobs(model, processor, item, device):
    batch = make_inputs(processor, item, device)
    output = model(**batch, use_cache=False)
    return F.log_softmax(output.logits[:, -1, :].float(), dim=-1)


def reference_forward(
    model,
    processor,
    item,
    device,
    projector_params,
    initial_projector,
):
    """Get original-model probabilities with LoRA disabled."""
    current_projector = [
        (param, param.detach().cpu().clone())
        for _, param in projector_params
    ]
    was_training = model.training
    try:
        with torch.no_grad():
            for name, param in projector_params:
                param.copy_(initial_projector[name].to(param.device))

            model.eval()
            with model.disable_adapter():
                answer_lp = answer_logprob(model, processor, item, device)
                distribution_lp = output_distribution_logprobs(
                    model, processor, item, device
                )
        return answer_lp.detach(), distribution_lp.detach()
    finally:
        with torch.no_grad():
            for param, saved_value in current_projector:
                param.copy_(saved_value.to(param.device))
        model.train(was_training)


def save_checkpoint(model, projector_params, output_dir, name, args):
    checkpoint = Path(output_dir) / name
    checkpoint.mkdir(parents=True, exist_ok=True)

    # PEFT writes LoRA adapter files; write the separately trained projector too.
    model.save_pretrained(checkpoint)
    torch.save(
        {key: param.detach().cpu() for key, param in projector_params},
        checkpoint / "projector_fusion.pt",
    )
    with (checkpoint / "training_config.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(vars(args), f, indent=2)
    print(f"[checkpoint] saved {checkpoint}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--metadata_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--algo", choices=("ga", "kl", "npo"), required=True)
    parser.add_argument("--device_map", default="balanced")
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps_per_epoch", type=int, default=0)
    parser.add_argument("--lr_lora", type=float, default=2e-5)
    parser.add_argument("--lr_projector", type=float, default=1e-5)
    parser.add_argument("--lambda_forget", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--llava_single_crop",
        action="store_true",
        help="Use one 336x336 crop when supported; reduces LLaVA memory.",
    )
    args = parser.parse_args()

    if args.epochs < 1 or args.steps_per_epoch < 0:
        parser.error("epochs must be positive; steps_per_epoch cannot be negative")
    if args.algo in {"kl", "npo"} and args.beta <= 0:
        parser.error("beta must be positive for KL and NPO")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    retain, forget = load_metadata(args.metadata_jsonl, args.data_root)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    model, processor, lora_params, projector_params = load_model(
        args.model_path,
        dtype,
        args.device_map,
        args.llava_single_crop,
    )
    device = model_input_device(model)

    initial_projector = {
        name: param.detach().cpu().clone()
        for name, param in projector_params
    }
    optimizer = torch.optim.AdamW([
        {"params": [p for _, p in lora_params], "lr": args.lr_lora},
        {
            "params": [p for _, p in projector_params],
            "lr": args.lr_projector,
        },
    ])

    steps = args.steps_per_epoch or max(len(retain), len(forget))
    print(
        f"[data] retain={len(retain)} forget={len(forget)}",
        flush=True,
    )
    print(
        f"[train] algo={args.algo} epochs={args.epochs} "
        f"steps_per_epoch={steps}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        random.shuffle(retain)
        random.shuffle(forget)
        model.train()
        total_loss = 0.0

        for step in range(steps):
            r = retain[step % len(retain)]
            f = forget[step % len(forget)]

            # Compute the no-gradient reference before building current-model
            # gradient graphs to reduce peak GPU memory.
            ref_forget_logp = None
            ref_retain_distribution = None
            if args.algo == "npo":
                ref_forget_logp, _ = reference_forward(
                    model, processor, f, device,
                    projector_params, initial_projector,
                )
            elif args.algo == "kl":
                _, ref_retain_distribution = reference_forward(
                    model, processor, r, device,
                    projector_params, initial_projector,
                )

            optimizer.zero_grad(set_to_none=True)
            retain_logp = answer_logprob(model, processor, r, device)
            forget_logp = answer_logprob(model, processor, f, device)

            if args.algo == "ga":
                loss = -retain_logp + args.lambda_forget * forget_logp

            elif args.algo == "kl":
                current_distribution = output_distribution_logprobs(
                    model, processor, r, device
                )
                retain_kl = F.kl_div(
                    ref_retain_distribution,
                    current_distribution.exp(),
                    reduction="sum",
                )
                loss = (
                    args.lambda_forget * forget_logp
                    + args.beta * retain_kl
                )

            else:
                npo_forget = (
                    2.0 * args.lambda_forget / args.beta
                    * F.softplus(
                        args.beta * (forget_logp - ref_forget_logp)
                    )
                )
                loss = -retain_logp + npo_forget

            if not torch.isfinite(loss).all():
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch}, step {step + 1}; "
                    "stopping before saving an invalid checkpoint."
                )

            loss.backward()
            trainable_params = [
                p for group in optimizer.param_groups for p in group["params"]
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable_params, max_norm=1.0
            )
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                raise FloatingPointError(
                    f"Non-finite gradient at epoch {epoch}, step {step + 1}; "
                    "stopping before saving an invalid checkpoint."
                )

            optimizer.step()
            total_loss += float(loss.detach())

            if step == 0 or (step + 1) % 100 == 0 or step + 1 == steps:
                print(
                    f"[epoch {epoch}/{args.epochs} "
                    f"step {step + 1}/{steps}] "
                    f"loss={float(loss.detach()):.5f}",
                    flush=True,
                )

        print(
            f"[epoch {epoch}] mean_loss={total_loss / steps:.5f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        save_checkpoint(
            model, projector_params, args.output_dir, f"epoch_{epoch}", args
        )

    save_checkpoint(model, projector_params, args.output_dir, "final", args)
    print(f"[done] checkpoints saved under {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
