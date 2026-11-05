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

    # Freeze the text encoder. Train the visual encoder and classifier head.
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    for parameter in clip_model.visual.parameters():
        parameter.requires_grad = True

    head = nn.Linear(clip_model.visual.output_dim, 2)
    return clip_model.to(device), head.to(device)


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
                col
                for col in ("img_filename", "filename", "file_name", "path")
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
            self.samples.append((image_path, int(row["y"]), int(row["place"])))

    def resolve_path(self, filename):
        path = Path(str(filename))
        if path.is_absolute() and path.is_file():
            return path

        candidates = [self.root / path, self.root / "images" / path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate

        matches = [
            p
            for p in self.root.rglob(path.name)
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


class PurgeMImages(Dataset):
    def __init__(self, root, partition):
        self.transform = make_transform(training=True)
        self.samples = []

        for true_class in CLASS_TO_ID:
            class_dir = Path(root) / partition / true_class
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected folder: {class_dir}")

            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                # Expected layout:
                # retain/<true_class>/<target_class>/<image>
                # forget/<true_class>/<target_class>/<image>
                relative_parts = path.relative_to(class_dir).parts[:-1]
                target_class = next(
                    (part for part in relative_parts if part in CLASS_TO_ID),
                    None,
                )

                # Retain examples are target-present. Forget examples are
                # target-absent, so their NPO label is the nested target class.
                if target_class is None:
                    if partition == "retain":
                        target_class = true_class
                    else:
                        raise ValueError(
                            f"Cannot infer forget target from {path}; expected "
                            "forget/<true_class>/<target_class>/<image>."
                        )

                self.samples.append((path, CLASS_TO_ID[target_class]))

        if not self.samples:
            raise ValueError(f"No images found in {Path(root) / partition}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target_label = self.samples[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
        return self.transform(image), target_label


def locate_purge_m_root(root):
    root = Path(root)
    candidates = [root, root / "train", root / "images" / "train"]
    for candidate in candidates:
        if (candidate / "retain").is_dir() and (candidate / "forget").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find retain/ and forget/ under {root}.")


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
            class_name = "landbird" if label == 0 else "waterbird"
            print(f"Warning: {name} has only {count} {class_name} images.")
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
            f"{GROUP_NAMES[i]}={group_acc[i]:.4f}" for i in range(4)
        )
    )
    return {"average_accuracy": average_accuracy, "wga": wga,
            "group_accuracies": group_acc}


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


def group_balanced_ce(logits, labels, places):
    group_ids = labels * 2 + places
    losses = []
    for group_id in range(4):
        mask = group_ids == group_id
        if mask.any():
            losses.append(F.cross_entropy(logits[mask], labels[mask]))
    return torch.stack(losses).mean()


def make_baseline_loader(dataset, batch_size, num_workers, seed):
    group_ids = [label * 2 + place for _, label, place in dataset.samples]
    counts = [group_ids.count(group_id) for group_id in range(4)]
    weights = torch.tensor(
        [1.0 / max(counts[group_id], 1) for group_id in group_ids],
        dtype=torch.double,
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=True, generator=generator
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    return loader, counts


def train_baseline(model, head, train_loader, val_loader, args, device, output_dir):
    optimizer = torch.optim.AdamW(
        list(model.visual.parameters()) + list(head.parameters()),
        lr=args.baseline_lr,
        weight_decay=1e-4,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "best_finetune.pt"

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
            loss = group_balanced_ce(
                logits_for(model, head, images), labels, places
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

        metrics = evaluate(model, head, val_loader, device, "baseline-val")
        print(
            f"[baseline] epoch={epoch}/{args.epochs} "
            f"loss={total_loss / max(total, 1):.4f}"
        )

    save_checkpoint(output_path, model, head, args.epochs, metrics)
    print(f"Saved group-balanced baseline: {output_path}")
    return output_path


def get_next(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator

def train_purge_m_npo(
    model, head, reference_model, reference_head,
    retain_loader, forget_loader, val_loader,
    args, device, output_dir,
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

    reference_model.eval()
    reference_head.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    for parameter in reference_head.parameters():
        parameter.requires_grad = False

    candidates = []
    metrics = evaluate(model, head, val_loader, device, "PURGE-M-start")
    candidate_path = candidate_dir / "epoch_0.pt"
    save_checkpoint(candidate_path, model, head, 0, metrics)
    candidates.append({"epoch": 0, "metrics": metrics, "path": candidate_path})

    for epoch in range(1, args.epochs + 1):
        model.train()
        head.train()
        retain_iter = iter(retain_loader)
        forget_iter = iter(forget_loader)
        steps = max(len(retain_loader), len(forget_loader))
        total_loss = 0.0

        for _ in range(steps):
            (rx, ry), retain_iter = get_next(retain_iter, retain_loader)
            (fx, fy), forget_iter = get_next(forget_iter, forget_loader)
            rx, ry = rx.to(device), ry.to(device)
            fx, fy = fx.to(device), fy.to(device)

            optimizer.zero_grad(set_to_none=True)
            retain_loss = F.cross_entropy(logits_for(model, head, rx), ry)

            current_logits = logits_for(model, head, fx)
            current_logp = F.log_softmax(current_logits, dim=1).gather(
                1, fy[:, None]
            ).squeeze(1)

            with torch.no_grad():
                reference_logits = logits_for(reference_model, reference_head, fx)
                reference_logp = F.log_softmax(
                    reference_logits, dim=1
                ).gather(1, fy[:, None]).squeeze(1)

            forget_loss = (
                (2.0 * args.lambda_forget / args.beta)
                * F.softplus(args.beta * (current_logp - reference_logp)).mean()
            )
            loss = retain_loss + forget_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate(model, head, val_loader, device, "PURGE-M-val")
        print(
            f"[PURGE-M-NPO] epoch={epoch}/{args.epochs} "
            f"loss={total_loss / max(steps, 1):.4f}"
        )
        candidate_path = candidate_dir / f"epoch_{epoch}.pt"
        save_checkpoint(candidate_path, model, head, epoch, metrics)
        candidates.append({
            "epoch": epoch, "metrics": metrics, "path": candidate_path
        })

    eligible = [
        c for c in candidates
        if c["metrics"]["wga"] >= args.min_val_wga
    ]
    if not eligible:
        raise RuntimeError(
            f"No checkpoint reached validation WGA >= {args.min_val_wga:.4f}."
        )
    selected = max(
        eligible, key=lambda c: c["metrics"]["average_accuracy"]
    )
    shutil.copy2(selected["path"], output_path)

    print(
        f"Selected epoch={selected['epoch']} "
        f"val_accuracy={selected['metrics']['average_accuracy']:.4f} "
        f"val_WGA={selected['metrics']['wga']:.4f}; "
        f"validation_WGA_floor={args.min_val_wga:.4f}"
    )
    for candidate in candidates:
        Path(candidate["path"]).unlink(missing_ok=True)
    candidate_dir.rmdir()
    print(f"Saved PURGE-M-NPO checkpoint: {output_path}")
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
        "--purge-m-root",
        default=str(HAL / "datasets/Waterbirds_PURGE_M"),
    )
    parser.add_argument(
        "--clip-checkpoint",
        default=str(HAL / "models/RN50_openai.pt"),
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default=None,
        help="Optional fine-tuned reference checkpoint; otherwise trained here.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(HAL / "spu/outputs/Waterbirds_PURGE_M_corrected"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--baseline-lr", type=float, default=1e-4)
    parser.add_argument("--purge-lr", type=float, default=1e-5)
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument("--lambda-forget", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--accuracy-tolerance", type=float, default=0.03)
    parser.add_argument("--min-val-wga", type=float, default=0.80)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    baseline_dir = output_dir / "original"
    purge_dir = output_dir / "PURGE_M_NPO"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    purge_dir.mkdir(parents=True, exist_ok=True)

    purge_root = locate_purge_m_root(args.purge_m_root)
    print(f"Using PURGE-M root: {purge_root}")

    train_data = WaterbirdsDataset(
        args.waterbirds_root, args.metadata_csv, split=0, training=True
    )
    val_data = WaterbirdsDataset(
        args.waterbirds_root, args.metadata_csv, split=1, training=False
    )
    test_data = WaterbirdsDataset(
        args.waterbirds_root, args.metadata_csv, split=2, training=False
    )

    train_loader, group_counts = make_baseline_loader(
        train_data, args.batch_size, args.num_workers, args.seed
    )
    print("Waterbirds train group counts:", dict(zip(GROUP_NAMES, group_counts)))

    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    retain_data = PurgeMImages(purge_root, "retain")
    forget_data = PurgeMImages(purge_root, "forget")

    for name, dataset in (("retain", retain_data), ("forget", forget_data)):
        target_counts = {
            class_name: sum(label == class_id for _, label in dataset.samples)
            for class_name, class_id in CLASS_TO_ID.items()
        }
        print(f"PURGE-M {name} target-class counts: {target_counts}")

    retain_indices = balanced_indices(
        retain_data, args.samples_per_class, args.seed, "retain"
    )
    forget_indices = balanced_indices(
        forget_data, args.samples_per_class, args.seed + 1, "forget"
    )

    retain_loader = DataLoader(
        Subset(retain_data, retain_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    forget_loader = DataLoader(
        Subset(forget_data, forget_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(
        f"Training samples: original={len(train_data)}, "
        f"retain={len(retain_indices)}, forget={len(forget_indices)}"
    )

    if args.baseline_checkpoint:
        baseline_checkpoint = Path(args.baseline_checkpoint)
        if not baseline_checkpoint.is_file():
            raise FileNotFoundError(
                f"Baseline checkpoint not found: {baseline_checkpoint}"
            )
        print(f"Using reference checkpoint: {baseline_checkpoint}")
    else:
        model, head = create_classifier(args.clip_checkpoint, device)
        baseline_checkpoint = train_baseline(
            model, head, train_loader, val_loader, args, device, baseline_dir
        )

    # Load the reference checkpoint as both the start model and fixed NPO model.
    model, head = create_classifier(args.clip_checkpoint, device)
    load_classifier_checkpoint(baseline_checkpoint, model, head)

    reference_model = copy.deepcopy(model).to(device)
    reference_head = copy.deepcopy(head).to(device)
    for parameter in reference_model.parameters():
        parameter.requires_grad = False
    for parameter in reference_head.parameters():
        parameter.requires_grad = False

    purge_checkpoint = train_purge_m_npo(
        model,
        head,
        reference_model,
        reference_head,
        retain_loader,
        forget_loader,
        val_loader,
        args,
        device,
        purge_dir,
    )

    model, head = create_classifier(args.clip_checkpoint, device)
    load_classifier_checkpoint(baseline_checkpoint, model, head)
    evaluate(model, head, test_loader, device, "baseline-test")

    model, head = create_classifier(args.clip_checkpoint, device)
    load_classifier_checkpoint(purge_checkpoint, model, head)
    evaluate(model, head, test_loader, device, "PURGE-M-NPO-test")

    print("\nFinished.")
    print(f"Baseline checkpoint: {baseline_checkpoint}")
    print(f"PURGE-M checkpoint: {purge_checkpoint}")


if __name__ == "__main__":
    main()