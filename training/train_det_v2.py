"""Fine-tune the plate detector on synthetic scenes plus real annotated frames.

The first detector was trained on synthetic scenes only and collapsed on real
photos (it labelled whole car bodies as ``other``). This run starts from those
weights and mixes 5.6k synthetic frames with the 131 manually annotated real
frames (repeated) and 10 real negatives.

Example::

    training\\.venv\\Scripts\\python.exe training/train_det_v2.py --epochs 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Create fine-tuning arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8n plate detector on real frames.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "training/run_det_v2/dataset/data.yaml")
    parser.add_argument("--init", type=Path, default=PROJECT_ROOT / "solution/weights/best.pt")
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "training/run_det_v2")
    parser.add_argument("--name", default="fine_tune")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr0", type=float, default=0.0008)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261025)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the fine-tuning and print the resulting checkpoint path."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.data.is_file():
        print(f"Dataset config not found: {args.data}; run build_det_v2_dataset.py first.", file=sys.stderr)
        return 1
    if not args.init.is_file():
        print(f"Initial weights not found: {args.init}", file=sys.stderr)
        return 1
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Install training dependencies from training/requirements.txt.", file=sys.stderr)
        return 1

    # Ultralytics scans labels with a process pool (multiprocessing.SimpleQueue).
    # Confined Windows sandboxes forbid named pipes, so the pool cannot start.
    # A serial shim is a drop-in for imap() and only costs scan time.
    import ultralytics.data.dataset as ultra_dataset

    class _SerialPool:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> bool:
            return False

        def imap(self, func, iterable):
            return map(func, iterable)

    ultra_dataset.ThreadPool = _SerialPool  # type: ignore[assignment]

    model = YOLO(str(args.init))
    result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.05,
        warmup_epochs=1,
        cos_lr=True,
        mosaic=1.0,
        close_mosaic=2,
        fliplr=0.0,
        flipud=0.0,
        degrees=4.0,
        translate=0.1,
        scale=0.35,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.35,
        plots=False,
        project=str(args.project.resolve()),
        name=args.name,
        exist_ok=True,
    )
    print(f"Best weights: {result.save_dir / 'weights' / 'best.pt'}")
    print("Validate on held-out real frames before claiming quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
