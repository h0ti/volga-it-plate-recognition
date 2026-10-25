"""Build a detector fine-tuning set: synthetic scenes + manually annotated real frames.

The synthetic split from ``training/run_plates_4060/prepared`` is reused as-is and
linked (hard links, no duplicate bytes). Real frames come from the manual review
journal, where ``bbox`` is stored as ``x1,y1,x2,y2`` in pixels. Real frames are
repeated a few times so that a ~2% share of the training set still influences the
domain adaptation; validation frames are never repeated.

Example::

    training\\.venv\\Scripts\\python.exe training/build_det_v2_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("type1", "type1a", "type1b", "other")
CLASS_IDS = {name: index for index, name in enumerate(CLASS_NAMES)}


def read_review(path: Path) -> tuple[dict[str, list[dict[str, str]]], list[str]]:
    """Group annotated boxes per frame and collect negative frames."""
    per_frame: dict[str, list[dict[str, str]]] = defaultdict(list)
    negatives: list[str] = []
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if row.get("manual_review") == "annotated" and row.get("bbox"):
                per_frame[row["image"]].append(row)
            elif row.get("manual_review") == "negative" and row["image"] not in negatives:
                negatives.append(row["image"])
    return per_frame, negatives


def yolo_lines(rows: list[dict[str, str]], width: int, height: int) -> list[str]:
    """Convert xyxy pixel boxes into normalized YOLO rows."""
    lines: list[str] = []
    for row in rows:
        x1, y1, x2, y2 = (float(v) for v in row["bbox"].split(","))
        x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        cx = (x1 + x2) / 2 / width
        cy = (y1 + y2) / 2 / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height
        lines.append(f"{CLASS_IDS[row['plate_type']]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def link_or_copy(source: Path, destination: Path) -> None:
    """Hard-link a file when possible, otherwise copy it."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble detector fine-tuning dataset.")
    parser.add_argument("--prepared", type=Path, default=PROJECT_ROOT / "training/run_plates_4060/prepared")
    parser.add_argument("--review", type=Path, default=PROJECT_ROOT / "dataset/real_review_inbox/review.csv")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data/commons/images")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "training/run_det_v2/dataset")
    parser.add_argument("--real-val-fraction", type=float, default=0.2)
    parser.add_argument("--real-repeat", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args(argv)

    from PIL import Image

    if args.out.exists() and any(args.out.iterdir()):
        parser.error(f"Output directory is not empty: {args.out}")
    per_frame, negatives = read_review(args.review)
    frames = sorted(per_frame)
    rng = random.Random(args.seed)
    rng.shuffle(frames)
    val_count = max(1, round(len(frames) * args.real_val_fraction))
    val_frames, train_frames = frames[:val_count], frames[val_count:]

    for split in ("train", "val"):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Synthetic scenes keep their original split and labels.
    for split in ("train", "val"):
        for image in (args.prepared / "images" / split).iterdir():
            if not image.is_file():
                continue
            link_or_copy(image, args.out / "images" / split / image.name)
            label = args.prepared / "labels" / split / f"{image.stem}.txt"
            if label.is_file():
                link_or_copy(label, args.out / "labels" / split / label.name)

    def add_real(frame: str, split: str) -> None:
        source = args.images / frame
        if not source.is_file():
            print(f"warning: missing frame {frame}", file=sys.stderr)
            return
        with Image.open(source) as handle:
            width, height = handle.size
        lines = yolo_lines(per_frame[frame], width, height)
        if not lines:
            return
        for copy_index in range(args.real_repeat if split == "train" else 1):
            stem = f"real_{Path(frame).stem}_{copy_index}" if args.real_repeat > 1 else f"real_{Path(frame).stem}"
            destination = args.out / "images" / split / f"{stem}{source.suffix.lower()}"
            link_or_copy(source, destination)
            (args.out / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )

    for frame in train_frames:
        add_real(frame, "train")
    for frame in val_frames:
        add_real(frame, "val")
    for frame in negatives:
        if frame in per_frame:
            continue
        source = args.images / frame
        if source.is_file():
            link_or_copy(source, args.out / "images" / "train" / f"neg_{Path(frame).stem}{source.suffix.lower()}")
            (args.out / "labels" / "train" / f"neg_{Path(frame).stem}.txt").write_text("", encoding="utf-8")

    config = {
        "path": str(args.out.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (args.out / "data.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    counts = {split: len(list((args.out / "images" / split).iterdir())) for split in ("train", "val")}
    print(f"real frames: {len(frames)} annotated + {len(negatives)} negatives")
    print(f"train images: {counts['train']}, val images: {counts['val']}")
    print(f"data.yaml: {args.out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
