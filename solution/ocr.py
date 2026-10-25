"""Small CRNN/CTC OCR model for locally detected Russian plate crops."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch import nn

from dataset.generator.platekit import render, spec


ALPHABET = "0123456789ABEKMHOPCTYX"
BLANK_ID = 0
CHAR_TO_ID = {character: index + 1 for index, character in enumerate(ALPHABET)}
ID_TO_CHAR = {index + 1: character for index, character in enumerate(ALPHABET)}
INPUT_HEIGHT = 64
INPUT_WIDTH_LIMIT = 384


class PlateCRNN(nn.Module):
    """Compact convolutional recurrent network with a CTC output layer."""

    def __init__(self, class_count: int = len(ALPHABET) + 1) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(128, 192, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.sequence = nn.LSTM(
            input_size=192,
            hidden_size=128,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.1,
        )
        self.classifier = nn.Linear(256, class_count)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return per-time-step character logits in B×T×C layout."""
        features = self.features(images).squeeze(2).transpose(1, 2)
        sequence, _ = self.sequence(features)
        return self.classifier(sequence)


def _resize_height(image: Image.Image, height: int) -> Image.Image:
    width = max(1, round(image.width * height / max(image.height, 1)))
    return image.resize((width, height), Image.Resampling.BICUBIC)


def prepare_plate_crop(
    image: Image.Image, plate_type: str, height: int = INPUT_HEIGHT
) -> torch.Tensor:
    """Normalize plate pixels; serialize two-row plates in reading order."""
    gray = image.convert("L")
    row_height = max(8, round(height * 0.75))
    if plate_type == "type1a":
        middle = gray.height // 2
        rows = [
            _resize_height(gray.crop((0, 0, gray.width, middle)), row_height),
            _resize_height(gray.crop((0, middle, gray.width, gray.height)), row_height),
        ]
        width = rows[0].width + 8 + rows[1].width
        canvas = Image.new("L", (width, height), 255)
        canvas.paste(rows[0], (0, (height - rows[0].height) // 2))
        canvas.paste(rows[1], (rows[0].width + 8, (height - rows[1].height) // 2))
    else:
        plate = _resize_height(gray, row_height)
        canvas = Image.new("L", (plate.width, height), 255)
        canvas.paste(plate, (0, (height - plate.height) // 2))
    if canvas.width > INPUT_WIDTH_LIMIT:
        canvas = canvas.resize(
            (INPUT_WIDTH_LIMIT, height), Image.Resampling.BICUBIC
        )
    array = np.asarray(canvas, dtype=np.uint8).copy()
    return torch.from_numpy(array).unsqueeze(0)


def greedy_decode(logits: torch.Tensor, length: int) -> tuple[str, float]:
    """Greedy CTC decode with confidence averaged over emitted characters."""
    probabilities = logits[:length].softmax(dim=-1)
    scores, indices = probabilities.max(dim=-1)
    output: list[str] = []
    output_scores: list[float] = []
    previous = BLANK_ID
    for index, score in zip(indices.tolist(), scores.tolist()):
        if index != BLANK_ID and index != previous:
            output.append(ID_TO_CHAR.get(index, ""))
            output_scores.append(score)
        previous = index
    confidence = sum(output_scores) / len(output_scores) if output_scores else 0.0
    return "".join(output), confidence


def is_valid_plate(text: str, plate_type: str) -> bool:
    """Check OCR output against the project's Russian plate masks."""
    return spec.is_valid_plate(text, plate_type)


# Per-position character classes of the Russian masks (letters exclude Cyrillic
# look-alikes that are not printed on plates).
_LETTERS = set("ABEKMHOPCTYX")
_DIGITS = set("0123456789")
_POSITIONS = {
    "type1": (_LETTERS, _DIGITS, _DIGITS, _DIGITS, _LETTERS, _LETTERS, _DIGITS, _DIGITS, _DIGITS),
    "type1a": (_LETTERS, _DIGITS, _DIGITS, _DIGITS, _LETTERS, _LETTERS, _DIGITS, _DIGITS, _DIGITS),
    "type1b": (_LETTERS, _LETTERS, _DIGITS, _DIGITS, _DIGITS, _DIGITS, _DIGITS, _DIGITS),
}
PLATE_LENGTH = {"type1": 9, "type1a": 9, "type1b": 8}
MIN_PLATE_LENGTH = {"type1": 8, "type1a": 8, "type1b": 7}


def format_plate_text(text: str, plate_type: str) -> str:
    """Replace impossible characters with '#' and pad a short tail with '#'.

    The organizers allow '#' for unreadable positions, so a partial read is kept;
    characters are never guessed. Russian plates carry a 2- or 3-digit region code
    (8 or 9 characters for type 1/1A, 7 or 8 for type 1B), therefore the decoded
    length is preserved inside that range instead of being forced to the maximum.
    """
    positions = _POSITIONS.get(plate_type)
    if positions is None:
        return ""
    maximum = len(positions)
    minimum = MIN_PLATE_LENGTH[plate_type]
    cleaned: list[str] = []
    for index, character in enumerate(text.upper()[:maximum]):
        cleaned.append(character if character in positions[index] else "#")
    while len(cleaned) < minimum:
        cleaned.append("#")
    return "".join(cleaned)


class PlateOCR:
    """Load one or more local checkpoints and recognize a single plate crop.

    Several checkpoints are combined by averaging their per-character
    probabilities, which is cheap and makes the recognizer more robust on
    blurry crops than any single model.
    """

    def __init__(self, checkpoint: Path | Sequence[Path], device: str = "cpu") -> None:
        self.device = torch.device(f"cuda:{device}" if device.isdigit() else device)
        checkpoints = (
            [checkpoint] if isinstance(checkpoint, (str, Path)) else list(checkpoint)
        )
        if not checkpoints:
            raise ValueError("At least one OCR checkpoint is required.")
        self.models: list[PlateCRNN] = []
        for path in checkpoints:
            state = torch.load(path, map_location=self.device, weights_only=True)
            if state.get("alphabet") != ALPHABET:
                raise ValueError(f"OCR checkpoint alphabet mismatch: {path}")
            model = PlateCRNN()
            model.load_state_dict(state["state_dict"])
            model.to(self.device).eval()
            self.models.append(model)

    @torch.inference_mode()
    def recognize(
        self, image: Image.Image, plate_type: str, height: int = INPUT_HEIGHT
    ) -> tuple[str, float]:
        """Recognize one crop; unreadable positions become '#'."""
        normalized = prepare_plate_crop(image, plate_type, height=height)
        width = normalized.shape[-1]
        padded_width = ((width + 3) // 4) * 4
        batch = torch.full(
            (1, 1, height, padded_width), 255, dtype=torch.uint8
        )
        batch[:, :, :, :width] = normalized
        batch = batch.to(self.device).float().div_(255.0)
        probabilities = None
        for model in self.models:
            current = model(batch).softmax(dim=-1)
            probabilities = current if probabilities is None else probabilities + current
        assert probabilities is not None
        averaged = probabilities / len(self.models)
        text, confidence = greedy_decode(averaged[0].log(), padded_width // 4)
        formatted = format_plate_text(text, plate_type)
        if set(formatted) == {"#"}:
            return formatted, 0.0
        return formatted, confidence


def render_training_sample(rng: random.Random) -> tuple[torch.Tensor, str, str]:
    """Render one synthetic plate crop with mild, deterministic photo variation."""
    plate_type = rng.choices(
        ("type1", "type1a", "type1b"), weights=(0.36, 0.34, 0.30), k=1
    )[0]
    text = spec.random_plate(rng, plate_type)
    image = render.render_plate(
        text,
        plate_type,
        520,
        rng,
        draw_flag=(plate_type == "type1a" and rng.random() < 0.3),
    )
    if rng.random() < 0.7:
        image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.65, 1.25))
    if rng.random() < 0.7:
        image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.7, 1.35))
    if rng.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.8)))
    return prepare_plate_crop(image, plate_type), text, plate_type
