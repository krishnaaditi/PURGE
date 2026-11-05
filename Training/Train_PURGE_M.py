#!/usr/bin/env python3
"""Materialize PURGE-M train/evaluation folders from metadata_with_cues.csv.

Expected CSV columns:
    image, target_object, object_present, cue_object, cue_present,
    training_role, partition

PURGE-M training roles:
    retain      object present, cue absent; target answer is "yes"
    forget      object absent, cue present; suppress the spurious "yes"
    evaluation  remaining object/cue combinations; excluded from training
"""

import argparse
import csv
import json
import re
import shutil
from pathlib import Path


def yes_no(value, field, row_number):
    value = str(value).strip().lower()
    if value not in {"yes", "no"}:
        raise ValueError(
            f"CSV row {row_number}: {field} must be yes/no, got {value!r}"
        )
    return value


def safe_name(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return value.strip("_") or "unknown"


def materialize(source, destination, link_mode):
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        return

    if link_mode == "hardlink":
        try:
            destination.hardlink_to(source)
            return
        except OSError:
            # Cross-filesystem hardlinks are not possible; copy that image.
            shutil.copy2(source, destination)
            return

    shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True)
    parser.add_argument("--image_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument(
        "--link_mode",
        choices=("hardlink", "copy"),
        default="hardlink",
    )
    args = parser.parse_args()

    csv_path = Path(args.metadata_csv).resolve()
    image_root = Path(args.image_root).resolve()
    output_root = Path(args.output_root).resolve()

    if not csv_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root not found: {image_root}")

    # Avoid accidentally mixing results from an earlier run.
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}\n"
            "Choose a new --output_root or move the existing output first."
        )

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "image",
            "target_object",
            "object_present",
            "cue_object",
            "cue_present",
            "training_role",
            "partition",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {sorted(missing)}"
            )
        rows = list(reader)

    prepared = []
    role_counts = {"retain": 0, "forget": 0, "evaluation": 0}

    # Validate all rows and image paths before writing any output.
    for row_number, row in enumerate(rows, start=2):
        role = str(row["training_role"]).strip().lower()
        if role not in role_counts:
            raise ValueError(
                f"CSV row {row_number}: unknown training_role {role!r}"
            )

        object_present = yes_no(
            row["object_present"], "object_present", row_number
        )
        cue_present = yes_no(
            row["cue_present"], "cue_present", row_number
        )

        if role == "retain" and (object_present, cue_present) != ("yes", "no"):
            raise ValueError(
                f"CSV row {row_number}: retain must be object=yes, cue=no"
            )
        if role == "forget" and (object_present, cue_present) != ("no", "yes"):
            raise ValueError(
                f"CSV row {row_number}: forget must be object=no, cue=yes"
            )
        if role == "evaluation" and (object_present, cue_present) not in {
            ("yes", "yes"),
            ("no", "no"),
        }:
            raise ValueError(
                f"CSV row {row_number}: unexpected evaluation object/cue pair"
            )

        image_value = Path(str(row["image"]).strip())
        source = image_value if image_value.is_absolute() else image_root / image_value
        source = source.resolve()
        if not source.is_file():
            raise FileNotFoundError(
                f"CSV row {row_number}: image not found: {source}"
            )

        target = str(row["target_object"]).strip()
        cue = str(row["cue_object"]).strip()
        if not target or not cue:
            raise ValueError(
                f"CSV row {row_number}: target_object/cue_object is empty"
            )

        role_counts[role] += 1
        prepared.append({
            "role": role,
            "object_present": object_present,
            "cue_present": cue_present,
            "target": target,
            "cue": cue,
            "partition": safe_name(row["partition"]),
            "source": source,
            "coco_image_id": str(row.get("coco_image_id", "")).strip(),
            "cue_assignment_valid": str(
                row.get("cue_assignment_valid", "")
            ).strip(),
            "cue_assignment_source": str(
                row.get("cue_assignment_source", "")
            ).strip(),
        })

    images_root = output_root / "images"
    train_metadata = output_root / "metadata_train.jsonl"
    eval_metadata = output_root / "metadata_evaluation.jsonl"

    with train_metadata.open("w", encoding="utf-8") as train_file, \
         eval_metadata.open("w", encoding="utf-8") as eval_file:

        for item in prepared:
            role = item["role"]
            target_dir = safe_name(item["target"])
            cue_name = safe_name(item["cue"])
            image_id = safe_name(
                item["coco_image_id"] or item["source"].stem
            )
            extension = item["source"].suffix or ".jpg"
            filename = f"{image_id}_{cue_name}{extension}"

            if role in {"retain", "forget"}:
                relative_output = (
                    Path("train") / role / target_dir / filename
                )
                target_file = images_root / relative_output
                materialize(item["source"], target_file, args.link_mode)

                # Both PURGE-M training targets are "yes", as specified
                # by the PURGE-M objective in the paper.
                record = {
                    "purge_variant": "PURGE-M",
                    "file_name": relative_output.as_posix(),
                    "split": role,
                    "target_object": item["target"],
                    "cue_object": item["cue"],
                    "object_present": item["object_present"],
                    "cue_present": item["cue_present"],
                    "prediction": "yes",
                    "training_role": role,
                    "partition": item["partition"],
                    "coco_image_id": item["coco_image_id"],
                    "cue_assignment_valid": item["cue_assignment_valid"],
                    "cue_assignment_source": item["cue_assignment_source"],
                }
                train_file.write(json.dumps(record) + "\n")

            else:
                relative_output = (
                    Path("evaluation")
                    / item["partition"]
                    / target_dir
                    / filename
                )
                target_file = images_root / relative_output
                materialize(item["source"], target_file, args.link_mode)

                # Evaluation uses the object's ground-truth presence.
                record = {
                    "purge_variant": "PURGE-M",
                    "file_name": relative_output.as_posix(),
                    "split": f"evaluation/{item['partition']}",
                    "target_object": item["target"],
                    "cue_object": item["cue"],
                    "object_present": item["object_present"],
                    "cue_present": item["cue_present"],
                    "prediction": item["object_present"],
                    "training_role": "evaluation",
                    "partition": item["partition"],
                    "coco_image_id": item["coco_image_id"],
                    "cue_assignment_valid": item["cue_assignment_valid"],
                    "cue_assignment_source": item["cue_assignment_source"],
                }
                eval_file.write(json.dumps(record) + "\n")

    summary = {
        "method": "PURGE-M",
        "source_csv": str(csv_path),
        "image_root": str(image_root),
        "output_root": str(output_root),
        "role_counts": role_counts,
        "training_records": role_counts["retain"] + role_counts["forget"],
        "evaluation_records": role_counts["evaluation"],
        "link_mode": args.link_mode,
    }
    with (output_root / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"Training metadata:   {train_metadata}")
    print(f"Evaluation metadata: {eval_metadata}")


if __name__ == "__main__":
    main()