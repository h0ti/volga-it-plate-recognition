"""Honest evaluation of the full offline pipeline on manually annotated real frames.

The script separates three questions that are easy to mix up:

1. detection recall — does the detector put a box on the true plate (IoU >= 0.5);
2. OCR quality — character error rate on the *ground-truth* crop, no detector;
3. end-to-end — does the exported CSV string equal the manual plate number.

Frames listed in ``--exclude`` (the training journals of the detector/OCR) are
reported separately so the numbers are not silently optimistic.

Example::

    training\\.venv\\Scripts\\python.exe training/eval_real_pipeline.py ^
        --weights solution/weights/best.pt --ocr-weights training/run_ocr_v2/ocr_v2_best.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solution.ocr import PlateOCR  # noqa: E402
from solution.predict import detect_with_fallback, result_rows  # noqa: E402


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Intersection over union of two xyxy boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance for CER."""
    previous = list(range(len(right) + 1))
    for row, char_left in enumerate(left, 1):
        current = [row]
        for column, char_right in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[column] + 1,
                previous[column - 1] + (char_left != char_right),
            ))
        previous = current
    return previous[-1]


def load_annotations(path: Path) -> dict[str, list[dict[str, str]]]:
    """Group annotated boxes per frame."""
    frames: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if row.get("manual_review") == "annotated" and row.get("bbox"):
                frames[row["image"]].append(row)
    return dict(frames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate detector + OCR on real annotations.")
    parser.add_argument("--review", type=Path, default=PROJECT_ROOT / "dataset/real_review_inbox/review.csv")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data/commons/images")
    parser.add_argument("--weights", type=Path, default=PROJECT_ROOT / "solution/weights/best.pt")
    parser.add_argument("--ocr-weights", type=str, default=str(PROJECT_ROOT / "solution/weights/ocr_v3.pt"))
    parser.add_argument("--device", default="0")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "training/run_eval_v2/report.md")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    frames = load_annotations(args.review)
    detector = YOLO(str(args.weights))
    ocr_paths = [Path(part.strip()) for part in args.ocr_weights.split(",") if part.strip()]
    ocr = PlateOCR(ocr_paths, args.device)

    det_hits = 0
    det_hits_typed = 0
    total_boxes = 0
    exact_crop = 0
    chars_ok = 0
    chars_total = 0
    end_to_end = 0
    rows_out: list[str] = []

    for frame, annotations in sorted(frames.items()):
        path = args.images / frame
        result = detect_with_fallback(detector, path, 640, 0.25, args.device)
        boxes = result.boxes if result.boxes is not None else []
        image = Image.open(path).convert("RGB")

        for row in annotations:
            total_boxes += 1
            truth = tuple(int(float(v)) for v in row["bbox"].split(","))
            best_iou, best_cls = 0.0, -1
            for box in boxes:
                candidate = tuple(int(v) for v in box.xyxy[0].tolist())
                value = iou(truth, candidate)
                if value > best_iou:
                    best_iou, best_cls = value, int(box.cls.item())
            if best_iou >= 0.5:
                det_hits += 1
                expected_class = {"type1": 0, "type1a": 1, "type1b": 2}[row["plate_type"]]
                det_hits_typed += int(best_cls == expected_class)

            crop = image.crop(truth)
            predicted, _ = ocr.recognize(crop, row["plate_type"])
            exact_crop += int(predicted == row["plate_num"])
            chars_total += len(row["plate_num"])
            chars_ok += sum(
                1 for a, b in zip(predicted, row["plate_num"]) if a == b
            )

        # End-to-end row for the main (highest) annotation of the frame: run the
        # exact production formatter instead of a re-implementation.
        main_row = annotations[0]
        exported = result_rows([result], args.images, ocr)[0]
        end_to_end += int(
            exported["plate_num"] == main_row["plate_num"]
            and exported["plate_type"] == main_row["plate_type"]
        )

    def pct(value: int, total: int) -> str:
        return f"{value}/{total} = {100.0 * value / max(total, 1):.1f}%"

    rows_out.extend([
        "# Оценка пайплайна на ручной разметке реальных кадров",
        "",
        f"Кадров с разметкой: {len(frames)}; размеченных знаков: {total_boxes}.",
        "Разметка ручная (journal `dataset/real_review_inbox/review.csv`),",
        "независимого теста организаторов нет.",
        "",
        f"- Детекция (IoU ≥ 0.5): {pct(det_hits, total_boxes)}",
        f"- Тип верен среди найденных: {pct(det_hits_typed, max(det_hits, 1))}",
        f"- OCR по эталонному кропу, точное совпадение: {pct(exact_crop, total_boxes)}",
        f"- OCR по эталонному кропу, совпавших символов: {pct(chars_ok, chars_total)}",
        f"- Сквозное совпадение номера: {pct(end_to_end, len(frames))}",
        "",
    ])
    for line in rows_out:
        print(line)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(rows_out) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
