"""Prepare synthetic YOLO data and train the plate-type detector."""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml


CLASS_NAMES = ("type1", "type1a", "type1b", "other")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def prepare_dataset(source: Path, destination: Path, seed: int, val_fraction: float) -> int:
    """Copy a generated dataset into a standard YOLO detection layout."""
    meta_path = source / "meta.csv"
    labels_source = source / "labels"
    images_source = source / "images" / "synthetic"
    if not meta_path.is_file() or not labels_source.is_dir() or not images_source.is_dir():
        raise ValueError(f"{source} is not a generated dataset (meta.csv/images/labels missing).")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"Output directory is not empty; refusing to overwrite: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    images = sorted(
        path for path in images_source.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    records = [(image, labels_source / f"{image.stem}.txt") for image in images]
    records = [(image, label) for image, label in records if label.is_file()]
    if len(records) < 10:
        raise ValueError(f"At least 10 labeled images are required; found {len(records)}.")

    random.Random(seed).shuffle(records)
    val_count = max(1, round(len(records) * val_fraction))
    splits = {"train": records[val_count:], "val": records[:val_count]}
    for split, items in splits.items():
        image_dir = destination / "images" / split
        label_dir = destination / "labels" / split
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for image, source_label in items:
            shutil.copy2(image, image_dir / image.name)
            converted: list[str] = []
            for line_number, line in enumerate(source_label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) < 5:
                    raise ValueError(f"Malformed label {source_label}:{line_number}")
                class_id = int(fields[0])
                if class_id not in range(len(CLASS_NAMES)):
                    raise ValueError(f"Unknown class id in {source_label}:{line_number}")
                # Source labels add four quad points after the YOLO bbox. The
                # detector consumes only class + normalized bbox.
                converted.append(" ".join(fields[:5]))
            (label_dir / f"{image.stem}.txt").write_text(
                "\n".join(converted) + ("\n" if converted else ""), encoding="utf-8"
            )

    config = {
        "path": str(destination.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (destination / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return len(records)


def build_parser() -> argparse.ArgumentParser:
    """Create command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare synthetic labels and train a YOLOv8n plate-type detector."
    )
    parser.add_argument("--dataset", type=Path, default=Path("dataset/generated"))
    parser.add_argument("--out", type=Path, default=Path("training/run_plates"))
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from training/<out>/runs/plate_detector/weights/last.pt.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Prepare data and optionally start training."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.05 <= args.val_fraction <= 0.4:
        parser.error("--val-fraction must be between 0.05 and 0.4.")
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1:
        parser.error("--epochs, --imgsz and --batch must be positive.")
    prepared = args.out / "prepared"
    if args.resume:
        if not (prepared / "data.yaml").is_file():
            print(
                f"Prepared data not found: {prepared / 'data.yaml'}; "
                "keep the original training/run_plates directory.",
                file=sys.stderr,
            )
            return 1
        checkpoint = args.out / "runs" / "plate_detector" / "weights" / "last.pt"
        if not checkpoint.is_file():
            print(
                f"Resume checkpoint not found: {checkpoint}. "
                "No files were changed; use a fresh training run only if needed.",
                file=sys.stderr,
            )
            return 1
        print(f"Resuming from {checkpoint}; reusing {prepared / 'data.yaml'}")
    else:
        try:
            count = prepare_dataset(args.dataset, prepared, args.seed, args.val_fraction)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            print(f"Dataset preparation failed: {exc}", file=sys.stderr)
            return 1
        print(f"Prepared {count} synthetic images in {prepared}")
    if args.prepare_only:
        return 0

    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install training dependencies from training/requirements.txt.", file=sys.stderr)
        return 1

    model = YOLO(str(checkpoint)) if args.resume else YOLO(args.model)
    options = {
        "data": str(prepared / "data.yaml"),
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "cache": False,
        "plots": False,
        "project": str((args.out / "runs").resolve()),
        "name": "plate_detector",
        "exist_ok": args.resume,
    }
    if args.resume:
        options["resume"] = True
    else:
        options.update(
            epochs=args.epochs,
            seed=args.seed,
            deterministic=True,
        )
    result = model.train(**options)
    print(f"Best weights: {result.save_dir / 'weights' / 'best.pt'}")
    print("This detector is trained on synthetic data; validate on separate real images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
