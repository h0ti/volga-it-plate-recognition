"""Sanitize a YOLO checkpoint before publishing: drop local absolute paths.

Ultralytics stores the full training arguments in the checkpoint, including the
absolute paths of the machine that trained it. This script keeps the model, EMA
and metrics but rewrites the paths to repository-relative placeholders.

Example::

    python training/strip_checkpoint.py solution/weights/best_v2.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Remove local paths from a YOLO checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, help="Write to another file instead of in place.")
    args = parser.parse_args(argv)
    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    train_args = state.get("train_args")
    if isinstance(train_args, dict):
        for key in ("model", "data", "project", "name", "save_dir"):
            if key in train_args and train_args[key]:
                train_args[key] = Path(str(train_args[key])).name
    state.pop("git", None)

    output = args.output or args.checkpoint
    torch.save(state, output)
    print(f"Sanitized checkpoint: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
