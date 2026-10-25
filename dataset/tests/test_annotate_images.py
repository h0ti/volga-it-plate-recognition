"""Тесты форматов номеров и безопасной записи ручной разметки."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from dataset.annotate_images import (
    CSV_FIELDS,
    UNKNOWN_LICENSE,
    Provenance,
    load_provenance,
    read_annotations,
    validate_plate_number,
    write_annotations,
)


class AnnotationFormatTests(unittest.TestCase):
    """Проверяет маски, provenance и сохранение без потери строк."""

    def test_type1_and_type1a_masks(self) -> None:
        self.assertIsNone(validate_plate_number("A123BC77", "type1"))
        self.assertIsNone(validate_plate_number("A123BC777", "type1a"))
        self.assertIsNotNone(validate_plate_number("D123BC77", "type1"))
        self.assertIsNotNone(validate_plate_number("A123BC7", "type1"))

    def test_yellow_plate_mask(self) -> None:
        self.assertIsNone(validate_plate_number("AB12377", "type1b"))
        self.assertIsNone(validate_plate_number("AB123777", "type1b"))
        self.assertIsNotNone(validate_plate_number("A123BC77", "type1b"))

    def test_other_allows_blank_or_unreadable(self) -> None:
        self.assertIsNone(validate_plate_number("", "other"))
        self.assertIsNone(validate_plate_number("ABC#123", "other"))
        self.assertIsNotNone(validate_plate_number("А123ВС77", "other"))

    def test_manifest_provenance_defaults_unknown_without_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_provenance(Path(directory)), {})
            image_dir = Path(directory) / "images"
            image_dir.mkdir()
            (Path(directory) / "manifest.json").write_text(
                json.dumps([{
                    "local": "one.jpg",
                    "descurl": "https://example.org/image",
                    "license": "CC0",
                    "author": "Example",
                }]),
                encoding="utf-8",
            )
            provenance = load_provenance(image_dir)["one.jpg"]
            self.assertEqual(provenance.license, "CC0")
            self.assertEqual(provenance.source_url, "https://example.org/image")
            self.assertEqual(provenance.attribution, "Example")
            self.assertEqual(Provenance().license, UNKNOWN_LICENSE)

    def test_write_and_read_preserves_all_image_annotations(self) -> None:
        row = {
            "annotation_id": "one.jpg#01",
            "image": "one.jpg",
            "bbox": "10,20,100,50",
            "plate_num": "A123BC77",
            "plate_type": "type1",
            "is_vehicle": "1",
            "source_url": "https://example.org/image",
            "license": "CC0",
            "license_verified": "0",
            "attribution": "Example",
            "manual_review": "annotated",
            "notes": "",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "review.csv"
            write_annotations(output, {"one.jpg": [row]})
            self.assertEqual(read_annotations(output), {"one.jpg": [row]})
            with output.open(encoding="utf-8-sig", newline="") as source:
                reader = csv.DictReader(source, delimiter=";")
                self.assertEqual(tuple(reader.fieldnames or ()), CSV_FIELDS)


if __name__ == "__main__":
    unittest.main()
