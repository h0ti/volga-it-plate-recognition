"""Tests for offline CSV formatting helpers."""

from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

from solution.predict import image_files, plate_placeholder, result_rows, write_csv


class PredictionFormatTests(unittest.TestCase):
    """Exercise result formatting without loading model weights."""

    def test_type_placeholders_follow_plate_lengths(self) -> None:
        self.assertEqual(plate_placeholder("type1"), "#" * 9)
        self.assertEqual(plate_placeholder("type1a"), "#" * 9)
        self.assertEqual(plate_placeholder("type1b"), "#" * 8)
        self.assertEqual(plate_placeholder("other"), "")

    def test_empty_detection_produces_explicit_other_row(self) -> None:
        result = SimpleNamespace(
            path="frame.jpg",
            boxes=None,
        )
        rows = result_rows([result], Path("."))
        self.assertEqual(rows[0]["plate_type"], "other")
        self.assertEqual(rows[0]["plate_num"], "")
        self.assertEqual(rows[0]["confidence"], "0.0000")

    def test_detection_maps_to_type_and_unreadable_placeholder(self) -> None:
        box = SimpleNamespace(cls=SimpleNamespace(item=lambda: 2),
                              conf=SimpleNamespace(item=lambda: 0.875))

        class Boxes:
            def __len__(self) -> int:
                return 1

            def __iter__(self):
                return iter([box])

        rows = result_rows(
            [SimpleNamespace(path="frame.jpg", boxes=Boxes())], Path(".")
        )
        self.assertEqual(rows[0]["plate_type"], "type1b")
        self.assertEqual(rows[0]["plate_num"], "#" * 8)
        self.assertEqual(rows[0]["confidence"], "0.8750")

    def test_image_discovery_rejects_missing_input(self) -> None:
        with self.assertRaises(ValueError):
            image_files(Path("this-path-should-not-exist.jpg"))

    def test_csv_header_uses_required_semicolon_schema(self) -> None:
        output_path = Path(__file__).resolve().parent / "_tmp_predictions" / "predictions.csv"
        try:
            write_csv([], output_path)
            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                "image;plate_num;plate_type;confidence\n",
            )
        finally:
            if output_path.parent.exists():
                for child in output_path.parent.iterdir():
                    child.unlink()
                output_path.parent.rmdir()


if __name__ == "__main__":
    unittest.main()
