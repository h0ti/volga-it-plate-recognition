"""Offline plate detection, type decision and organizer-format CSV export.

The pipeline is deliberately small and dependency-light: one YOLO detector for the
plate box and type, one CRNN/CTC recognizer for the number, and a deterministic
post-processing step that enforces the Russian plate mask. Unreadable positions
are exported as ``#`` instead of guessed characters.

Example::

    python solution/predict.py --input data\\commons\\images --weights solution\\weights\\best.pt ^
        --ocr-weights solution\\weights\\ocr_best.pt --output predictions.csv --device 0
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = {0: "type1", 1: "type1a", 2: "type1b", 3: "other"}
TARGET_CLASSES = (0, 1, 2)
PLATE_LENGTH = {"type1": 9, "type1a": 9, "type1b": 8}


def plate_placeholder(plate_type: str) -> str:
    """Return an honest all-unreadable placeholder of the type's legal length."""
    return "#" * PLATE_LENGTH.get(plate_type, 0)


def image_files(root: Path) -> list[Path]:
    """List supported images recursively in stable order."""
    if root.is_file() and root.suffix.lower() in IMAGE_SUFFIXES:
        return [root]
    if not root.is_dir():
        raise ValueError(f"Input path is not an image or directory: {root}")
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _best_box(boxes: Any, class_ids: Iterable[int]) -> Any | None:
    """Pick the highest-confidence box among the requested classes."""
    wanted = set(class_ids)
    best = None
    best_conf = -1.0
    for box in boxes:
        if int(box.cls.item()) not in wanted:
            continue
        confidence = float(box.conf.item())
        if confidence > best_conf:
            best, best_conf = box, confidence
    return best


def _is_yellow_plate(crop: np.ndarray) -> bool:
    """Detect the yellow type-1B background by a robust HSV share test."""
    if crop.size == 0:
        return False
    hsv = np.asarray(Image.fromarray(crop).convert("HSV"), dtype=np.uint8)
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    yellow = (
        (hue >= 20) & (hue <= 40) & (saturation >= 70) & (value >= 90)
    )
    return float(yellow.mean()) > 0.35


def _crop_for_ocr(
    result: Any, box: Any, pad_factors: tuple[float, float] = (0.03, 0.10)
) -> tuple[Image.Image | None, tuple[int, int, int, int]]:
    """Crop the detected plate with a margin factor, as RGB."""
    if not hasattr(box, "xyxy") or not hasattr(result, "orig_img"):
        return None, (0, 0, 0, 0)
    x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
    height, width = result.orig_img.shape[:2]
    pad_x = max(1, round((x2 - x1) * pad_factors[0]))
    pad_y = max(1, round((y2 - y1) * pad_factors[1]))
    x1, x2 = max(0, x1 - pad_x), min(width, x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(height, y2 + pad_y)
    crop = result.orig_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None, (x1, y1, x2, y2)
    return Image.fromarray(crop[:, :, ::-1].copy()), (x1, y1, x2, y2)


def _crop_with_aspect(
    result: Any, box: Any, target_aspect: float = 2.6, pad_x: float = 0.03
) -> Image.Image | None:
    """Crop the box and add vertical context until it reaches ``target_aspect``.

    The detector box is tight around the plate (aspect ~4.6), while the OCR was
    trained on looser review crops (aspect ~2–3). Normalizing the aspect keeps the
    character height/scale in the range the recognizer saw during training.
    """
    if not hasattr(box, "xyxy") or not hasattr(result, "orig_img"):
        return None
    x1, y1, x2, y2 = (int(round(value)) for value in box.xyxy[0].tolist())
    height, width = result.orig_img.shape[:2]
    pad = max(1, round((x2 - x1) * pad_x))
    x1, x2 = max(0, x1 - pad), min(width, x2 + pad)
    wanted_height = (x2 - x1) / max(target_aspect, 0.5)
    extra = max(0, round((wanted_height - (y2 - y1)) / 2))
    y1, y2 = max(0, y1 - extra), min(height, y2 + extra)
    crop = result.orig_img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return Image.fromarray(crop[:, :, ::-1].copy())


def detect_with_fallback(
    model: Any,
    path: Path,
    imgsz: int,
    conf: float,
    device: str,
    fallback_imgsz: int | None = None,
) -> Any:
    """Detect one image; retry at a higher resolution when no plate is found.

    Small or distant plates are frequently missed at 640 px, while a second pass
    at 960 px recovers them. The retry only runs for frames without a target
    detection, so the average cost stays low.
    """
    result = model.predict(
        str(path), imgsz=imgsz, conf=conf, device=device, verbose=False
    )[0]
    if _best_box(result.boxes, TARGET_CLASSES) is None:
        retry = model.predict(
            str(path),
            imgsz=fallback_imgsz or max(imgsz, 960),
            conf=conf,
            device=device,
            verbose=False,
        )[0]
        if _best_box(retry.boxes, TARGET_CLASSES) is not None:
            return retry
    return result


def result_rows(
    results: Iterable[Any], input_root: Path, ocr: Any | None = None
) -> list[dict[str, str]]:
    """Convert Ultralytics results into exactly one organizer row per image."""
    rows: list[dict[str, str]] = []
    for result in results:
        result_path = Path(result.path)
        try:
            image_name = result_path.resolve().relative_to(input_root.resolve()).as_posix()
        except ValueError:
            image_name = result_path.name
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            rows.append({
                "image": image_name,
                "plate_num": "",
                "plate_type": "other",
                "confidence": "0.0000",
            })
            continue

        target = _best_box(boxes, TARGET_CLASSES)
        if target is None:
            other = _best_box(boxes, (3,))
            rows.append({
                "image": image_name,
                "plate_num": "",
                "plate_type": "other",
                "confidence": f"{float(other.conf.item()):.4f}" if other is not None else "0.0000",
            })
            continue

        class_id = int(target.cls.item())
        detected_type = CLASS_NAMES.get(class_id, "type1")
        confidence = float(target.conf.item())
        crop, _ = _crop_for_ocr(result, target)
        plate_type = detected_type
        plate_num = plate_placeholder(plate_type)
        if crop is not None:
            yellow = _is_yellow_plate(np.asarray(crop))
            # The detector's type is reliable; colour only corrects white->yellow,
            # never the opposite (a dim yellow plate must not become white).
            if yellow and detected_type != "type1b":
                plate_type = "type1b"
            if ocr is not None:
                # Boxes are never pixel-tight, so try a few margins. The alternative
                # one-row/two-row layout is only tried when the detector is unsure:
                # on real annotations the detector's type is right in 99.4 % of hits,
                # and a forced re-layout otherwise only adds garbage characters.
                hypotheses = [plate_type]
                if (
                    plate_type in {"type1", "type1a"}
                    and float(target.conf.item()) < 0.5
                ):
                    hypotheses.append("type1a" if plate_type == "type1" else "type1")
                candidates = [
                    _crop_for_ocr(result, target, pad)[0]
                    for pad in ((0.02, 0.06), (0.05, 0.14), (0.10, 0.24))
                ]
                candidates.append(_crop_with_aspect(result, target, 2.6))
                candidates.append(_crop_with_aspect(result, target, 3.2))
                # Vote across crops/layouts instead of trusting a single lucky crop:
                # a correct reading usually repeats, a one-character slip does not.
                votes: dict[tuple[str, str], list[float]] = {}
                for candidate in candidates:
                    if candidate is None:
                        continue
                    for hypothesis in hypotheses:
                        text, text_confidence = ocr.recognize(candidate, hypothesis)
                        entry = votes.setdefault((text, hypothesis), [0.0, 0.0])
                        entry[0] += max(text_confidence, 0.05)
                        entry[1] += 1.0
                if votes:
                    (best_text, plate_type), (weight, count) = max(
                        votes.items(), key=lambda item: (item[1][0], item[1][1])
                    )
                    plate_num = best_text
                    confidence *= min(weight / max(count, 1.0), 1.0)
            else:
                plate_num = plate_placeholder(plate_type)
        rows.append({
            "image": image_name,
            "plate_num": plate_num,
            "plate_type": plate_type,
            "confidence": f"{confidence:.4f}",
        })
    return rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write rows with the exact organizer header and delimiter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("image", "plate_num", "plate_type", "confidence"),
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Run local YOLO plate detection + CRNN OCR and write a semicolon CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--weights", type=Path, required=True, help="Local detector checkpoint; no download.")
    parser.add_argument(
        "--ocr-weights",
        type=str,
        help=(
            "Optional local CRNN checkpoint(s), comma-separated for an ensemble; "
            "without it target numbers are emitted as '#'."
        ),
    )
    parser.add_argument("--output", type=Path, default=Path("predictions.csv"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0", help="Use 'cpu' for CPU-only inference.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run offline inference and export the organizer CSV."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not 0.0 <= args.conf <= 1.0 or args.imgsz < 32:
        parser.error("--conf must be 0..1 and --imgsz must be at least 32.")
    try:
        files = image_files(args.input)
    except ValueError as exc:
        parser.error(str(exc))
    if not files:
        parser.error(f"No supported images found: {args.input}")
    if not args.weights.is_file():
        parser.error(f"Local model weights not found; no network download attempted: {args.weights}")
    ocr_paths = [
        Path(part.strip())
        for part in (args.ocr_weights or "").split(",")
        if part.strip()
    ]
    for ocr_path in ocr_paths:
        if not ocr_path.is_file():
            parser.error(f"OCR checkpoint not found: {ocr_path}")
    try:
        from ultralytics import YOLO
        from solution.ocr import PlateOCR
    except ImportError:
        print("Install solution dependencies from solution/requirements.txt.", file=sys.stderr)
        return 1

    try:
        model = YOLO(str(args.weights))
        ocr = PlateOCR(ocr_paths, args.device) if ocr_paths else None
        root = args.input if args.input.is_dir() else args.input.parent
        started = time.perf_counter()
        rows: list[dict[str, str]] = []
        fallback_imgsz = max(args.imgsz, 960)
        for path in files:
            result = detect_with_fallback(
                model, path, args.imgsz, args.conf, args.device, fallback_imgsz
            )
            rows.extend(result_rows([result], root, ocr))
        elapsed = time.perf_counter() - started
        write_csv(rows, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Inference: {elapsed:.3f}s total, {elapsed / len(files) * 1000:.1f} ms/image.")
    if ocr is None:
        print("OCR checkpoint not supplied; '#' marks unreadable target-number positions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
