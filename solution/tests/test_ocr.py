"""Shape and decode tests for the local CTC recognizer."""

from __future__ import annotations

import unittest

import torch
from PIL import Image

from solution.ocr import ALPHABET, CHAR_TO_ID, PlateCRNN, greedy_decode, is_valid_plate, prepare_plate_crop


class OCRTests(unittest.TestCase):
    """Fast CPU-only OCR unit checks."""

    def test_two_row_plate_is_serialized_without_exceeding_model_width(self) -> None:
        image = Image.new("RGB", (290, 170), "white")
        tensor = prepare_plate_crop(image, "type1a")
        self.assertEqual(tensor.shape[0:2], (1, 64))
        self.assertLessEqual(tensor.shape[-1], 384)

    def test_network_emits_expected_ctc_sequence_length(self) -> None:
        model = PlateCRNN().eval()
        with torch.inference_mode():
            logits = model(torch.zeros((1, 1, 64, 128)))
        self.assertEqual(logits.shape, (1, 32, len(ALPHABET) + 1))

    def test_greedy_ctc_collapses_repeats_and_blanks(self) -> None:
        logits = torch.full((4, len(ALPHABET) + 1), -8.0)
        logits[0, CHAR_TO_ID["A"]] = 8.0
        logits[1, CHAR_TO_ID["A"]] = 8.0
        logits[2, 0] = 8.0
        logits[3, CHAR_TO_ID["B"]] = 8.0
        text, confidence = greedy_decode(logits, 4)
        self.assertEqual(text, "AB")
        self.assertGreater(confidence, 0.99)

    def test_plate_mask_accepts_unreadable_positions(self) -> None:
        self.assertTrue(is_valid_plate("A123BC777", "type1"))
        self.assertTrue(is_valid_plate("#" * 9, "type1a"))
        self.assertTrue(is_valid_plate("#" * 8, "type1b"))
        self.assertFalse(is_valid_plate("A12BC777", "type1"))


if __name__ == "__main__":
    unittest.main()
