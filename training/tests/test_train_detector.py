"""Tests for preparing YOLO data, including background-only negatives."""

from __future__ import annotations

import tempfile
import unittest
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from training.train_detector import main, prepare_dataset


class DetectorDataPreparationTests(unittest.TestCase):
    """Verify deterministic split layout and empty negative label handling."""

    def test_preparation_keeps_negative_frames_with_empty_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            images = source / "images" / "synthetic"
            labels = source / "labels"
            images.mkdir(parents=True)
            labels.mkdir()
            (source / "meta.csv").write_text(
                "image;plate_num;plate_type;bbox;quad;is_vehicle;is_synthetic;source;license;conditions\n",
                encoding="utf-8",
            )
            for index in range(10):
                Image.new("RGB", (32, 32), "white").save(images / f"frame_{index}.jpg")
                label = "0 0.5 0.5 0.25 0.25\n" if index < 5 else ""
                (labels / f"frame_{index}.txt").write_text(label, encoding="utf-8")

            destination = root / "prepared"
            count = prepare_dataset(source, destination, seed=42, val_fraction=0.2)

            self.assertEqual(count, 10)
            split_labels = list((destination / "labels").rglob("*.txt"))
            self.assertEqual(len(split_labels), 10)
            self.assertTrue(any(path.stat().st_size == 0 for path in split_labels))
            self.assertTrue((destination / "data.yaml").is_file())

    def test_resume_reuses_existing_checkpoint_and_reduces_loader_memory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "training"
            prepared = output / "prepared"
            prepared.mkdir(parents=True)
            (prepared / "data.yaml").write_text("path: .\n", encoding="utf-8")
            checkpoint = output / "runs" / "plate_detector" / "weights" / "last.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint placeholder")
            calls: dict[str, object] = {}

            class FakeYOLO:
                def __init__(self, model_path: str) -> None:
                    calls["model_path"] = model_path

                def train(self, **options: object) -> SimpleNamespace:
                    calls["options"] = options
                    return SimpleNamespace(save_dir=output / "runs" / "plate_detector")

            fake_ultralytics = ModuleType("ultralytics")
            fake_ultralytics.YOLO = FakeYOLO
            with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                result = main([
                    "--dataset", str(root / "dataset"),
                    "--out", str(output),
                    "--resume",
                    "--imgsz", "512",
                    "--batch", "4",
                    "--workers", "0",
                    "--device", "0",
                ])

            self.assertEqual(result, 0)
            self.assertEqual(calls["model_path"], str(checkpoint))
            options = calls["options"]
            self.assertTrue(options["resume"])
            self.assertEqual(options["workers"], 0)
            self.assertEqual(options["batch"], 4)
            self.assertEqual(options["imgsz"], 512)
            self.assertFalse(options["cache"])
            self.assertEqual(options["project"], str((output / "runs").resolve()))


if __name__ == "__main__":
    unittest.main()
