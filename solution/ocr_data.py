"""Training data for the plate OCR: real review crops, synthetic scenes, clean renders.

The first OCR model was trained on clean digital renders only and collapsed on real
photographs. This module supplies three complementary sources plus a photometric /
geometric augmentation chain that mimics camera footage:

* real manually annotated crops from ``dataset/real_review_inbox/review.csv``;
* plates cropped from the synthetic scene generator (quad -> rectangle warp);
* clean GOST renders with random fonts and sizes.

Every source returns a PIL image, its canonical number and the plate type; the
caller normalizes them through :func:`solution.ocr.prepare_plate_crop`.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from dataset.generator.platekit import render, spec


TARGET_TYPES = ("type1", "type1a", "type1b")


@dataclass(frozen=True, eq=False)
class PlateSample:
    """One training crop with its ground-truth text, type and source frame."""

    image: Image.Image
    text: str
    plate_type: str
    origin: str
    source_id: str = ""


def _quad_to_pil(quad: Sequence[float]) -> list[float]:
    """Project quad TL->TR->BR->BL into PIL's QUAD order TL, BL, BR, TR."""
    q = list(quad)
    return [q[0], q[1], q[6], q[7], q[4], q[5], q[2], q[3]]


def load_real_samples(review_path: Path, images_dir: Path) -> list[PlateSample]:
    """Load manually annotated real crops (bbox) with validated numbers only."""
    samples: list[PlateSample] = []
    with review_path.open(encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle, delimiter=";")
        for row in rows:
            if row.get("manual_review") != "annotated":
                continue
            if not row.get("bbox") or not row.get("plate_num"):
                continue
            plate_type = row["plate_type"]
            if plate_type not in TARGET_TYPES:
                continue
            text = row["plate_num"].strip().upper()
            if not spec.is_valid_plate(text, plate_type):
                continue
            image_path = images_dir / row["image"]
            if not image_path.is_file():
                continue
            try:
                x1, y1, x2, y2 = (int(float(v)) for v in row["bbox"].split(","))
            except ValueError:
                continue
            width, height = x2 - x1, y2 - y1
            if width < 40 or height < 14:
                continue
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
            crop = image.crop(
                (max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2))
            )
            if crop.width < 40 or crop.height < 14:
                continue
            samples.append(PlateSample(crop, text, plate_type, "real_review", image_path.name))
    return samples


def load_scene_samples(
    meta_path: Path, root: Path, min_quad_width: float = 70.0
) -> list[PlateSample]:
    """Warp plate quads from the synthetic scene dataset into rectangles."""
    samples: list[PlateSample] = []
    with meta_path.open(encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle, delimiter=";")
        for row in rows:
            plate_type = row["plate_type"]
            if plate_type not in TARGET_TYPES:
                continue
            text = row["plate_num"].strip().upper()
            if not spec.is_valid_plate(text, plate_type):
                continue
            try:
                quad = [float(v) for v in row["quad"].split(",")]
            except (KeyError, ValueError):
                continue
            if len(quad) != 8:
                continue
            top = ((quad[2] - quad[0]) ** 2 + (quad[3] - quad[1]) ** 2) ** 0.5
            if top < min_quad_width:
                continue
            image_path = root / row["image"]
            if not image_path.is_file():
                continue
            width, height = (290, 170) if plate_type == "type1a" else (520, 112)
            with Image.open(image_path) as handle:
                image = handle.convert("RGB")
            crop = image.transform(
                (width, height), Image.QUAD, data=_quad_to_pil(quad), resample=Image.BICUBIC
            )
            samples.append(PlateSample(crop, text, plate_type, "synthetic_scene", image_path.name))
    return samples


def render_clean_samples(rng: random.Random, count: int) -> list[PlateSample]:
    """Render clean GOST plates at random widths for the sharp end of the domain."""
    samples: list[PlateSample] = []
    for _ in range(count):
        plate_type = rng.choice(TARGET_TYPES)
        text = spec.random_plate(rng, plate_type)
        width = rng.choice((280, 320, 360, 420, 520, 620))
        image = render.render_plate(
            text,
            plate_type,
            width,
            rng,
            draw_flag=(plate_type == "type1a" and rng.random() < 0.35),
        )
        samples.append(PlateSample(image, text, plate_type, "clean_render", f"clean_{len(samples)}"))
    return samples


def _random_quad(width: int, height: int, rng: random.Random, strength: float) -> list[float]:
    """Jittered corners of the full image, in PIL QUAD order TL, BL, BR, TR."""
    dx, dy = max(1, int(width * strength)), max(1, int(height * strength))
    jitter = lambda: rng.uniform(-1.0, 1.0)  # noqa: E731
    tl = (rng.uniform(0, dx) * jitter(), rng.uniform(0, dy) * jitter())
    tr = (width + rng.uniform(0, dx) * jitter(), rng.uniform(0, dy) * jitter())
    br = (width + rng.uniform(0, dx) * jitter(), height + rng.uniform(0, dy) * jitter())
    bl = (rng.uniform(0, dx) * jitter(), height + rng.uniform(0, dy) * jitter())
    return [tl[0], tl[1], bl[0], bl[1], br[0], br[1], tr[0], tr[1]]


def augment_plate(image: Image.Image, rng: random.Random, strength: float = 1.0) -> Image.Image:
    """Photometric and geometric augmentation that mimics camera footage.

    ``strength`` scales the blur, rotation, perspective and resolution-loss
    ranges; values above 1.0 produce harder samples for domain adaptation.
    """
    work = image.convert("RGB")
    width, height = work.size

    # Small random margin: detector boxes are never perfectly tight.
    if rng.random() < 0.6:
        pad_x = int(width * rng.uniform(0.01, 0.08))
        pad_y = int(height * rng.uniform(0.03, 0.18))
        canvas = Image.new("RGB", (width + 2 * pad_x, height + 2 * pad_y), (128, 128, 128))
        canvas.paste(work, (pad_x, pad_y))
        work = canvas

    # Perspective and slight rotation, then downscale to camera resolution.
    if rng.random() < 0.85:
        work = work.transform(
            work.size,
            Image.QUAD,
            data=_random_quad(
                work.width, work.height, rng, rng.uniform(0.01, 0.05) * strength
            ),
            resample=Image.BICUBIC,
        )
    if rng.random() < 0.5:
        work = work.rotate(
            rng.uniform(-5.0, 5.0) * strength,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=(110, 110, 110),
        )
    if rng.random() < 0.5:
        factor = rng.uniform(max(0.3, 0.45 / strength), 1.0)
        work = work.resize(
            (max(24, int(work.width * factor)), max(10, int(work.height * factor))),
            Image.BILINEAR,
        )

    if rng.random() < 0.75:
        work = ImageEnhance.Brightness(work).enhance(rng.uniform(0.5, 1.4))
    if rng.random() < 0.75:
        work = ImageEnhance.Contrast(work).enhance(rng.uniform(0.6, 1.6))
    if rng.random() < 0.4:
        work = ImageEnhance.Color(work).enhance(rng.uniform(0.0, 1.2))
    if rng.random() < 0.55:
        work = work.filter(
            ImageFilter.GaussianBlur(rng.uniform(0.2, min(1.4 * strength, 2.5)))
        )
    if rng.random() < 0.2:
        work = work.filter(ImageFilter.MedianFilter(3))

    array = np.asarray(work).astype(np.float32)
    if rng.random() < 0.5:
        array = array + np.random.default_rng(rng.randrange(2**31)).normal(
            0, rng.uniform(2, 12), array.shape
        )
    # Uneven illumination / glare: multiplicative horizontal or vertical gradient.
    if rng.random() < 0.5:
        size = array.shape[1] if rng.random() < 0.5 else array.shape[0]
        ramp = np.linspace(rng.uniform(0.6, 1.0), rng.uniform(0.9, 1.35), size, dtype=np.float32)
        shape = (1, size, 1) if array.shape[1] == size else (size, 1, 1)
        array = array * ramp.reshape(shape)
    array = np.clip(array, 0, 255).astype(np.uint8)
    work = Image.fromarray(array)

    # JPEG ringing and sensor noise.
    if rng.random() < 0.4:
        import io

        buffer = io.BytesIO()
        work.save(buffer, format="JPEG", quality=rng.randint(40, 88))
        buffer.seek(0)
        work = Image.open(buffer).convert("RGB")
    return work
