import argparse
import copy
import random
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import open_clip

HAL = Path("/home/hq1351@ad.wayne.edu/home/hal")


CLASS_TO_ID = {"landbird": 0, "waterbird": 1}
GROUP_NAMES = [
    "landbird_land",
    "landbird_water",
    "waterbird_land",
    "waterbird_water",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if isinstance(checkpoint.get(key), dict):
                checkpoint = checkpoint[key]
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint does not contain a state dictionary.")

    state = {}
    for key, value in checkpoint.items():
        if torch.is_tensor(value):
            state[key.removeprefix("module.")] = value
    return state


def create_classifier(pretrained_path, device):
    # The warning about pretrained=None is expected: weights are loaded below.
    clip_model = open_clip.create_model("RN50-quickgelu", pretrained=None)

    state = unwrap_state_dict(torch_load(pretrained_path))
    result = clip_model.load_state_dict(state, strict=False)
    loaded = len(state) - len(result.unexpected_keys)

    if loaded == 0:
        raise RuntimeError(f"No CLIP weights matched: {pretrained_path}")

    print(
        f"Loaded pretrained CLIP weights from {pretrained_path}; "
        f"missing={len(result.missing_keys)}, "
        f"unexpected={len(result.unexpected_keys)}"
    )

    # The text encoder is frozen. The visual encoder and new classifier head train.
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    for parameter in clip_model.visual.parameters():
        parameter.requires_grad = True

    classifier = nn.Linear(clip_model.visual.output_dim, 2)
    return clip_model.to(device), classifier.to(device)


def make_transform(training):
    mean = (0.48145466, 0.4578275, 0.40821073)
    std = (0.26862954, 0.26130258, 0.27577711)

    ops = []
    if training:
        ops.append(transforms.RandomHorizontalFlip())

    ops.extend(
        [
            transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return transforms.Compose(ops)


def split_id(value):
    value = str(value).strip().lower()
    names = {
        "train": 0,
        "val": 1,
        "valid": 1,
        "validation": 1,
        "test": 2,
    }
    return names[value] if value in names else int(float(value))


class WaterbirdsDataset(Dataset):
    def __init__(self, image_root, metadata_csv, split, training=False):
        self.root = Path(image_root)
        self.transform = make_transform(training)
        table = pd.read_csv(metadata_csv)

        required = {"y", "place", "split"}
        missing = required.difference(table.columns)
        if missing:
            raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")

        filename_col = next(
            (
                col for col in ("img_filename", "filename", "file_name", "path")
                if col in table.columns
            ),
            None,
        )
        if filename_col is None:
            raise ValueError(
                "metadata.csv needs one of: img_filename, filename, file_name, path"
            )

        table = table[table["split"].apply(split_id) == split]
        if table.empty:
            raise ValueError(f"No rows found for split={split}.")

        self.samples = []
        for _, row in table.iterrows():
            image_path = self.resolve_path(row[filename_col])
            self.samples.append(
                (image_path, int(row["y"]), int(row["place"]))
            )

    def resolve_path(self, filename):
        path = Path(str(filename))
        if path.is_absolute() and path.is_file():
            return path

        candidates = [self.root / path, self.root / "images" / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        # Fallback for a prepared/reorganized Waterbirds directory.
        matches = [
            p for p in self.root.rglob(path.name)
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if matches:
            return matches[0]

        raise FileNotFoundError(f"Cannot locate image: {filename}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label, place = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label, place


class PurgeDImages(Dataset):
    def __init__(self, root, partition):
        self.transform = make_transform(training=True)
        self.samples = []

        for class_name, label in CLASS_TO_ID.items():
            class_dir = Path(root) / partition / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected folder: {class_dir}")

            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, label))

        if not self.samples:
            raise ValueError(f"No images found in {Path(root) / partition}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), label


def locate_purge_d_root(root):
    root = Path(root)
    candidates = [root, root / "train", root / "images" / "train"]
    for candidate in candidates:
        if (candidate / "retain").is_dir() and (candidate / "forget").is_dir():
            return candidate
    raise FileNotFoundError(
        f"Could not find retain/ and forget/ under {root}."
    )


def balanced_indices(dataset, samples_per_class, seed, name):
    by_class = {0: [], 1: []}
    for index, (_path, label) in enumerate(dataset.samples):
        by_class[label].append(index)

    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        indices = by_class[label][:]
        rng.shuffle(indices)
        count = min(samples_per_class, len(indices))
        if count < samples_per_class:
            print(
                f"Warning: {name} has only {count} "
                f"{('landbird' if label == 0 else 'waterbird')} images."
            )
        selected.extend(indices[:count])

    rng.shuffle(selected)
    return selected


def logits_for(model, head, images):
    features = model.visual(images)
    return head(features.float())


@torch.no_grad()
def evaluate(model, head, loader, device, label):
    model.eval()
    head.eval()

    correct = 0
    count = 0
    group_correct = [0, 0, 0, 0]
    group_count = [0, 0, 0, 0]

    for images, labels, places in loader:
        images = images.to(device)
        labels = labels.to(device)
        places = places.to(device)

        predictions = logits_for(model, head, images).argmax(dim=1)
        matches = predictions.eq(labels)
        correct += matches.sum().item()
        count += labels.numel()

        group_ids = labels * 2 + places
        for group_id in range(4):
            mask = group_ids == group_id
            group_count[group_id] += mask.sum().item()
            group_correct[group_id] += matches[mask].sum().item()

    group_acc = [
        group_correct[i] / group_count[i] if group_count[i] else float("nan")
        for i in range(4)
    ]
    available = [value for value in group_acc if value == value]
    wga = min(available) if available else 0.0
    average_accuracy = correct / max(count, 1)

    print(
        f"[{label}] accuracy={average_accuracy:.4f} "
        f"WGA={wga:.4f} "
        + " ".join(
            f"{GROUP_NAMES[i]}={group_acc[i]:.4f}"
            for i in range(4)
        )
    )

    return {
        "average_accuracy": average_accuracy,
        "wga": wga,
        "group_accuracies": group_acc,
    }


def save_checkpoint(path, model, head, epoch, metrics):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "visual": model.visual.state_dict(),
            "head": head.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_classifier_checkpoint(path, model, head):
    checkpoint = torch_load(path)
    model.visual.load_state_dict(checkpoint["visual"])
    head.load_state_dict(checkpoint["head"])


def choose_accuracy_then_wga(candidates, output_path, tolerance, label):
    """Keep near-best-accuracy checkpoints, then choose the highest WGA."""
    if not candidates:
        raise RuntimeError(f"No checkpoints were recorded for {label}.")
    peak_accuracy = max(item["metrics"]["average_accuracy"] for item in candidates)
    floor = peak_accuracy - tolerance
    eligible = [item for item in candidates
                if item["metrics"]["average_accuracy"] >= floor]
    selected = max(eligible, key=lambda item: item["metrics"]["wga"])
    shutil.copy2(selected["path"], output_path)
    for item in candidates:
        Path(item["path"]).unlink(missing_ok=True)
    print(
        f"Selected {label} epoch={selected['epoch']} "
        f"accuracy={selected['metrics']['average_accuracy']:.4f} "
        f"WGA={selected['metrics']['wga']:.4f}; "
        f"best-accuracy floor={floor:.4f}"
    )
    return output_path


def train_baseline(model, head, train_loader, val_loader, args, device, output_dir):
    optimizer = torch.optim.AdamW(
        list(model.visual.parameters()) + list(head.parameters()),
        lr=args.baseline_lr,
        weight_decay=1e-4,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_finetune.pt"
    candidate_dir = output_dir / "baseline_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for old in candidate_dir.glob("epoch_*.pt"):
        old.unlink()
    candidates = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        total_loss = 0.0
        total = 0

        for images, labels, places in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            places = places.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = group_balanced_ce(logits_for(model, head, images), labels, places)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        val_metrics = evaluate(model, head, val_loader, device, "baseline-val")
        print(f"[baseline] epoch={epoch}/{args.epochs} "
              f"loss={total_loss / max(total, 1):.4f}")
        candidate_path = candidate_dir / f"epoch_{epoch}.pt"
        save_checkpoint(candidate_path, model, head, epoch, val_metrics)
        candidates.append({"epoch": epoch, "metrics": val_metrics, "path": candidate_path})

    choose_accuracy_then_wga(
        candidates, output_path, args.accuracy_tolerance, "baseline"
    )
    candidate_dir.rmdir()
    print(f"Saved baseline checkpoint: {output_path}")
    return output_path

def get_next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def group_balanced_ce(logits, labels, places):
    """Average classification loss across Waterbirds groups in this batch."""
    group_ids = labels * 2 + places
    losses = []
    for group_id in range(4):
        mask = group_ids == group_id
        if mask.any():
            losses.append(F.cross_entropy(logits[mask], labels[mask]))
    return torch.stack(losses).mean()


def make_group_balanced_loader(dataset, batch_size, num_workers, seed):
    group_ids = [label * 2 + place for _, label, place in dataset.samples]
    counts = [group_ids.count(group_id) for group_id in range(4)]
    weights = torch.tensor(
        [1.0 / max(counts[group_id], 1) for group_id in group_ids], dtype=torch.double
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=True, generator=generator
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True
    )
    return loader, counts


def train_purge_d_npo(
    model,
    head,
    reference_model,
    reference_head,
    retain_loader,
    forget_loader,
    group_loader,
    val_loader,
    args,
    device,
    output_dir,
):
    optimizer = torch.optim.AdamW(
        list(model.visual.parameters()) + list(head.parameters()),
        lr=args.purge_lr,
        weight_decay=1e-4,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_npo.pt"
    candidate_dir = output_dir / "purge_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    for old in candidate_dir.glob("epoch_*.pt"):
        old.unlink()
    candidates = []

    reference_model.eval()
    reference_head.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    for parameter in reference_head.parameters():
        parameter.requires_grad = False

    # Include the pre-unlearning checkpoint as a candidate in the trade-off.
    start_metrics = evaluate(model, head, val_loader, device, "PURGE-D-start")
    start_path = candidate_dir / "epoch_0.pt"
    save_checkpoint(start_path, model, head, 0, start_metrics)
    candidates.append({"epoch": 0, "metrics": start_metrics, "path": start_path})

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        reference_model.eval()
        reference_head.eval()
        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)
        group_iter = iter(group_loader)
        steps = max(len(retain_loader), len(forget_loader))
        total_loss = 0.0

        for _ in range(steps):
            (retain_images, retain_labels), retain_iter = get_next(retain_iter, retain_loader)
            (forget_images, forget_labels), forget_iter = get_next(forget_iter, forget_loader)
            (group_images, group_labels, group_places), group_iter = get_next(group_iter, group_loader)

            retain_images, retain_labels = retain_images.to(device), retain_labels.to(device)
            forget_images, forget_labels = forget_images.to(device), forget_labels.to(device)
            group_images = group_images.to(device)
            group_labels, group_places = group_labels.to(device), group_places.to(device)

            optimizer.zero_grad(set_to_none=True)
            retain_logits = logits_for(model, head, retain_images)
            retain_loss = F.cross_entropy(retain_logits, retain_labels)
            group_logits = logits_for(model, head, group_images)
            group_loss = group_balanced_ce(group_logits, group_labels, group_places)

            current_logits = logits_for(model, head, forget_images)
            current_target_logp = F.log_softmax(current_logits, dim=1).gather(
                1, forget_labels[:, None]
            ).squeeze(1)
            with torch.no_grad():
                reference_logits = logits_for(reference_model, reference_head, forget_images)
                reference_target_logp = F.log_softmax(reference_logits, dim=1).gather(
                    1, forget_labels[:, None]
                ).squeeze(1)

            log_ratio = current_target_logp - reference_target_logp
            forget_loss = (
                (2.0 * args.lambda_forget / args.beta)
                * F.softplus(args.beta * log_ratio).mean()
            )
            loss = retain_loss + forget_loss + args.group_loss_weight * group_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_metrics = evaluate(model, head, val_loader, device, "PURGE-D-val")
        print(f"[PURGE-D-NPO] epoch={epoch}/{args.epochs} "
              f"loss={total_loss / max(steps, 1):.4f}")
        candidate_path = candidate_dir / f"epoch_{epoch}.pt"
        save_checkpoint(candidate_path, model, head, epoch, val_metrics)
        candidates.append({"epoch": epoch, "metrics": val_metrics, "path": candidate_path})

    choose_accuracy_then_wga(
        candidates, output_path, args.accuracy_tolerance, "PURGE-D"
    )
    candidate_dir.rmdir()
    print(f"Saved PURGE-D-NPO checkpoint: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--waterbirds-root",
        default=str(
            HAL / "datasets/waterbirds/waterbird_complete95_forest2water2"
        ),
    )
    parser.add_argument(
        "--metadata-csv",
        default=str(
            HAL
            / "datasets/waterbirds/waterbird_complete95_forest2water2/metadata.csv"
        ),
    )
    parser.add_argument(
        "--purge-d-root",
        default=str(HAL / "datasets/Waterbirds_PURGE_D"),
    )
    parser.add_argument(
        "--clip-checkpoint",
        default=str(HAL / "models/RN50_openai.pt"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(HAL / "spu/outputs/Waterbirds_PURGE_D_corrected"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline-lr", type=float, default=1e-4)
    parser.add_argument("--purge-lr", type=float, default=1e-5)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--lambda-forget", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--group-loss-weight", type=float, default=0.5)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.02,
                        help="Allowed drop from the best validation average accuracy.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / "original"
    purge_dir = output_dir / "PURGE_D_NPO"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    purge_dir.mkdir(parents=True, exist_ok=True)

    purge_root = locate_purge_d_root(args.purge_d_root)
    print(f"Using PURGE-D root: {purge_root}")

    train_data = WaterbirdsDataset(
        args.waterbirds_root, args.metadata_csv, split=0, training=True
    )
    val_data = WaterbirdsDataset(
        args.waterbirds_root, args.metadata_csv, split=1, training=False
    )
    test_data = WaterbirdsDataset(
        args.waterbirds_root, args.metadata_csv, split=2, training=False
    )

    train_loader, group_counts = make_group_balanced_loader(
        train_data, args.batch_size, args.num_workers, args.seed
    )
    print("Group-balanced baseline counts:", dict(zip(GROUP_NAMES, group_counts)))
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    retain_data = PurgeDImages(purge_root, "retain")
    forget_data = PurgeDImages(purge_root, "forget")

    retain_indices = balanced_indices(
        retain_data, args.samples_per_class, args.seed, "retain"
    )
    forget_indices = balanced_indices(
        forget_data, args.samples_per_class, args.seed + 1, "forget"
    )

    group_loader, group_counts = make_group_balanced_loader(
        train_data, args.batch_size, args.num_workers, args.seed + 2
    )
    print("Group counts for WGA-focused loss:", dict(zip(GROUP_NAMES, group_counts)))

    retain_loader = DataLoader(
        Subset(retain_data, retain_indices),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    forget_loader = DataLoader(
        Subset(forget_data, forget_indices),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    print(
        f"Training samples: original={len(train_data)}, "
        f"retain={len(retain_indices)}, forget={len(forget_indices)}"
    )

    # 1. Fine-tune original CLIP-RN50 from pretrained OpenAI CLIP weights.
    model, head = create_classifier(args.clip_checkpoint, device)
    baseline_checkpoint = train_baseline(
        model, head, train_loader, val_loader,
        args, device, baseline_dir
    )

    # 2. Start PURGE-D from the selected baseline checkpoint.
    model, head = create_classifier(args.clip_checkpoint, device)
    load_classifier_checkpoint(baseline_checkpoint, model, head)

    reference_model = copy.deepcopy(model).to(device)
    reference_head = copy.deepcopy(head).to(device)
    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    for parameter in reference_head.parameters():
        parameter.requires_grad = False

    purge_checkpoint = train_purge_d_npo(
        model, head, reference_model, reference_head,
        retain_loader, forget_loader, group_loader, val_loader,
        args, device, purge_dir
    )

    # Report held-out test metrics for the selected checkpoints.
    model, head = create_classifier(args.clip_checkpoint, device)
    load_classifier_checkpoint(baseline_checkpoint, model, head)
    evaluate(model, head, test_loader, device, "baseline-test")

    model, head = create_classifier(args.clip_checkpoint, device)
    load_classifier_checkpoint(purge_checkpoint, model, head)
    evaluate(model, head, test_loader, device, "PURGE-D-NPO-test")

    print("\nFinished.")
    print(f"Baseline checkpoint: {baseline_checkpoint}")
    print(f"PURGE-D checkpoint:  {purge_checkpoint}")


if __name__ == "__main__":
    main()