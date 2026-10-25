"""Минимальные тесты эвристик и безопасных проверок конфигурации."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import Image

from dataset.collect_open_datasets import (
    COLLECTION_FIELDS,
    Source,
    UNKNOWN_LICENSE,
    _download_requests,
    _safe_extract_zip,
    classify_plate_crop,
    collect,
    load_sources,
)


class CollectionTests(unittest.TestCase):
    """Проверки без сетевых запросов и API-ключей."""

    def test_requests_download_preserves_detected_image_format(self) -> None:
        image_data = io.BytesIO()
        Image.new("RGB", (8, 8), "white").save(image_data, format="JPEG")
        response = MagicMock()
        response.__enter__.return_value = response
        response.headers = {"Content-Length": str(len(image_data.getvalue()))}
        response.iter_content.return_value = [image_data.getvalue()]
        source = Source(
            name="checked-source",
            provider="requests",
            source_url="https://example.org/dataset",
            license="CC0",
            license_verified=True,
            attribution="",
            images_are_plate_crops=False,
            urls=("https://example.org/image",),
        )

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            with patch(
                "dataset.collect_open_datasets.requests.get",
                return_value=response,
            ) as get:
                _download_requests(source, destination)

            self.assertTrue((destination / "000001.jpg").is_file())
            self.assertEqual(
                get.call_args.kwargs["headers"]["User-Agent"],
                "VolgaITOpenDatasetCollector/1.0 (educational project)",
            )

    def test_type1a_hint_uses_crop_aspect_ratio(self) -> None:
        image = Image.new("RGB", (290, 170), "white")
        self.assertEqual(classify_plate_crop(image), ("type1a", "aspect_ratio_heuristic"))

    def test_type1b_hint_uses_yellow_hsv_pixels(self) -> None:
        image = Image.new("RGB", (520, 112), (255, 220, 0))
        self.assertEqual(classify_plate_crop(image), ("type1b", "HSV_color_heuristic"))

    def test_full_frame_is_not_classified_as_plate_crop(self) -> None:
        source_json = {
            "sources": [
                {
                    "name": "example",
                    "provider": "requests",
                    "source_url": "https://example.org/dataset",
                    "license": UNKNOWN_LICENSE,
                    "urls": ["https://example.org/image.jpg"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "sources.json"
            config.write_text(json.dumps(source_json), encoding="utf-8")
            source = load_sources(config)[0]
        self.assertFalse(source.license_verified)
        self.assertFalse(source.images_are_plate_crops)

    def test_verified_unknown_license_is_rejected(self) -> None:
        source_json = {
            "sources": [
                {
                    "name": "example",
                    "provider": "requests",
                    "source_url": "https://example.org/dataset",
                    "license": UNKNOWN_LICENSE,
                    "license_verified": True,
                    "urls": ["https://example.org/image.jpg"],
                }
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "sources.json"
            config.write_text(json.dumps(source_json), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "неизвестную лицензию"):
                load_sources(config)

    def test_zip_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with zipfile.ZipFile(archive, "w") as zipped:
                zipped.writestr("../outside.txt", "not allowed")
            with self.assertRaisesRegex(ValueError, "Небезопасный путь"):
                _safe_extract_zip(archive, root / "extract")
            self.assertFalse((root / "outside.txt").exists())

    def test_collection_keeps_suggestions_separate_and_appends_valid_csv(self) -> None:
        source = Source(
            name="checked-source",
            provider="requests",
            source_url="https://example.org/dataset",
            license="CC BY 4.0",
            license_verified=True,
            attribution="Example author",
            images_are_plate_crops=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset_dir = root / "dataset"
            dataset_dir.mkdir()
            (dataset_dir / "meta.csv").write_text("keep this file\n", encoding="utf-8")
            downloaded = root / "downloaded"
            downloaded.mkdir()
            Image.new("RGB", (290, 170), "white").save(downloaded / "plate.png")
            Image.new("RGB", (520, 112), (255, 220, 0)).save(downloaded / "yellow.jpg")
            with patch("dataset.collect_open_datasets._download_source", return_value=downloaded):
                copied, skipped = collect([source], dataset_dir, False, "all", 10)
            self.assertEqual((copied, skipped), (2, 0))
            self.assertEqual(
                (dataset_dir / "meta.csv").read_text(encoding="utf-8"),
                "keep this file\n",
            )
            with (dataset_dir / "collection.csv").open(
                encoding="utf-8-sig", newline=""
            ) as input_file:
                rows = list(csv.DictReader(input_file, delimiter=";"))
            self.assertEqual(tuple(rows[0]), COLLECTION_FIELDS)
            self.assertEqual({row["candidate_type"] for row in rows}, {"type1a", "type1b"})
            self.assertTrue(all(row["manual_review"] == "required" for row in rows))
            self.assertTrue(all(row["source_url"] == source.source_url for row in rows))
            self.assertEqual(len(list((dataset_dir / "images" / "real").rglob("*.*"))), 2)

    def test_unverified_source_is_quarantined(self) -> None:
        source = Source(
            name="unchecked-source",
            provider="requests",
            source_url="https://example.org/dataset",
            license=UNKNOWN_LICENSE,
            license_verified=False,
            attribution="",
            images_are_plate_crops=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloaded = root / "downloaded"
            downloaded.mkdir()
            Image.new("RGB", (640, 480), "gray").save(downloaded / "frame.png")
            with patch("dataset.collect_open_datasets._download_source", return_value=downloaded):
                copied, skipped = collect([source], root, True, "all", 10)
            self.assertEqual((copied, skipped), (1, 0))
            self.assertFalse((root / "images" / "real").exists())
            self.assertEqual(
                len(list((root / "quarantine" / "unverified").rglob("*.png"))),
                1,
            )
            with (root / "collection.csv").open(encoding="utf-8-sig", newline="") as input_file:
                row = next(csv.DictReader(input_file, delimiter=";"))
            self.assertEqual(row["status"], "quarantine_license_check")
            self.assertEqual(row["license"], UNKNOWN_LICENSE)


if __name__ == "__main__":
    unittest.main()
