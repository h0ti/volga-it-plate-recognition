"""Train the CRNN/CTC plate OCR on real crops + synthetic scenes + clean renders.

Usage (PowerShell, project root)::

    training\.venv\Scripts\python.exe training\train_ocr_v2.py --epochs 12

The script reports character error rate on a held-out slice of real manually
annotated plates, which is the metric the previous synthetic-only model failed.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solution.ocr import (  # noqa: E402
    ALPHABET,
    CHAR_TO_ID,
    ID_TO_CHAR,
    INPUT_HEIGHT,
    PlateCRNN,
    greedy_decode,
    is_valid_plate,
    prepare_plate_crop,
)
from solution.ocr_data import (  # noqa: E402
    PlateSample,
    augment_plate,
    load_real_samples,
    load_scene_samples,
    render_clean_samples,
)


def encode(text: str) -> torch.Tensor:
    """Encode a plate string into CTC target ids."""
    return torch.tensor([CHAR_TO_ID[character] for character in text], dtype=torch.long)


class MixedPlateDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Random mixture of real crops, synthetic scenes and clean renders."""

    def __init__(
        self,
        real: list[PlateSample],
        scene: list[PlateSample],
        clean_count: int,
        epoch_len: int,
        seed: int,
        clean_seed: int,
    ) -> None:
        self.real = real
        self.scene = scene
        self.clean = render_clean_samples(random.Random(clean_seed), clean_count)
        self.epoch_len = epoch_len
        self.seed = seed
        self.epoch = 0
        self.order: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        """Rebuild the deterministic sampling order for an epoch."""
        self.epoch = epoch
        self.order = list(range(self.epoch_len))

    def __len__(self) -> int:
        return self.epoch_len

    def _pick(self, rng: random.Random) -> PlateSample:
        draw = rng.random()
        if self.real and draw < 0.45:
            return self.real[rng.randrange(len(self.real))]
        if self.scene and draw < 0.8:
            return self.scene[rng.randrange(len(self.scene))]
        if self.clean:
            return self.clean[rng.randrange(len(self.clean))]
        return self.scene[rng.randrange(len(self.scene))]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = random.Random(self.seed + self.epoch * 100_003 + index)
        sample = self._pick(rng)
        image = augment_plate(sample.image, rng)
        tensor = prepare_plate_crop(image, sample.plate_type)
        return tensor, encode(sample.text)


def collate_batch(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad images and concatenate targets for CTC loss."""
    max_width = max(item[0].shape[-1] for item in batch)
    max_width = ((max_width + 3) // 4) * 4
    images = torch.full((len(batch), 1, INPUT_HEIGHT, max_width), 255, dtype=torch.uint8)
    widths = torch.tensor([item[0].shape[-1] for item in batch], dtype=torch.long)
    targets = torch.cat([item[1] for item in batch])
    target_lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
    for index, (image, _) in enumerate(batch):
        images[index, :, :, : image.shape[-1]] = image
    return images.float().div_(255.0), targets, widths, target_lengths


def edit_distance(left: str, right: str) -> int:
    """Levenshtein distance between two plate strings."""
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


@torch.inference_mode()
def evaluate(
    model: PlateCRNN,
    samples: list[PlateSample],
    device: torch.device,
    limit: int | None = None,
) -> tuple[float, float, int, list[tuple[str, str, str]]]:
    """Return (CER, exact-match rate) on unmodified crops."""
    model.eval()
    errors = 0
    characters = 0
    exact = 0
    shown: list[tuple[str, str, str]] = []
    selected = samples[:limit] if limit else samples
    for sample in selected:
        tensor = prepare_plate_crop(sample.image, sample.plate_type)
        width = tensor.shape[-1]
        padded = ((width + 3) // 4) * 4
        batch = torch.full((1, 1, INPUT_HEIGHT, padded), 255, dtype=torch.uint8)
        batch[:, :, :, :width] = tensor
        logits = model(batch.to(device).float().div_(255.0))
        predicted, _ = greedy_decode(logits[0], padded // 4)
        errors += edit_distance(predicted, sample.text)
        characters += len(sample.text)
        exact += int(predicted == sample.text)
        if len(shown) < 15:
            shown.append((sample.text, predicted, sample.plate_type))
    cer = errors / max(characters, 1)
    return cer, exact / max(len(selected), 1), len(selected), shown


def build_parser() -> argparse.ArgumentParser:
    """Create OCR v2 training arguments."""
    parser = argparse.ArgumentParser(description="Train OCR on real + synthetic plate crops.")
    parser.add_argument("--review", type=Path, default=PROJECT_ROOT / "dataset/real_review_inbox/review.csv")
    parser.add_argument("--images", type=Path, default=PROJECT_ROOT / "data/commons/images")
    parser.add_argument("--synth-meta", type=Path, default=PROJECT_ROOT / "dataset/generated_4060/meta.csv")
    parser.add_argument("--synth-root", type=Path, default=PROJECT_ROOT / "dataset/generated_4060")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "training/run_ocr_v2/ocr_v2_best.pt")
    parser.add_argument("--init", type=Path, default=PROJECT_ROOT / "solution/weights/ocr_best.pt")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epoch-len", type=int, default=6400)
    parser.add_argument("--clean-count", type=int, default=2400)
    parser.add_argument("--real-holdout", type=int, default=40)
    parser.add_argument("--val-synth", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    return parser


def split_real(samples: list[PlateSample], holdout: int) -> tuple[list[PlateSample], list[PlateSample]]:
    """Split real crops by source frame so one car never leaks into validation."""
    frames = sorted({sample.source_id for sample in samples})
    step = max(1, len(frames) // max(holdout, 1))
    val_frames = {frame for index, frame in enumerate(frames) if index % step == 0}
    train = [sample for sample in samples if sample.source_id not in val_frames]
    validation = [sample for sample in samples if sample.source_id in val_frames]
    return train, validation[:holdout]


def main(argv: list[str] | None = None) -> int:
    """Train the OCR model and report real-crop CER every epoch."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch < 1 or args.epoch_len < args.batch:
        parser.error("epochs/batch/epoch-len must be positive and epoch-len >= batch.")
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            print("CUDA is not available; refusing to train OCR on CPU.", file=sys.stderr)
            return 1
        device = torch.device(f"cuda:{args.device}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    print("Loading real crops...", flush=True)
    real = load_real_samples(args.review, args.images)
    print(f"real annotated crops: {len(real)}", flush=True)
    print("Warping synthetic scene plates...", flush=True)
    scene = load_scene_samples(args.synth_meta, args.synth_root)
    print(f"synthetic scene crops: {len(scene)}", flush=True)
    if not real and not scene:
        print("No training samples found.", file=sys.stderr)
        return 1

    real_train, real_val = split_real(real, args.real_holdout)
    print(f"real train/val: {len(real_train)}/{len(real_val)}", flush=True)
    synth_val = scene[:: max(1, len(scene) // args.val_synth)][: args.val_synth]
    scene_train = scene

    dataset = MixedPlateDataset(
        real_train, scene_train, args.clean_count, args.epoch_len, args.seed, args.seed + 7
    )
    model = PlateCRNN()
    if args.init and args.init.is_file():
        state = torch.load(args.init, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"])
        print(f"initialized from {args.init}", flush=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    log_path = args.out.with_suffix(".log.txt")
    best_cer = float("inf")
    started = time.perf_counter()

    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        loader = DataLoader(
            dataset,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=collate_batch,
            persistent_workers=False,
            drop_last=True,
        )
        model.train()
        total_loss = 0.0
        for images, targets, widths, target_lengths in loader:
            logits = model(images.to(device))
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(log_probs, targets.to(device), widths.to(device) // 4, target_lengths.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
        scheduler.step()

        cer, exact, count, shown = evaluate(model, real_val, device)
        synth_cer, _, _, _ = evaluate(model, synth_val, device, limit=120)
        elapsed = time.perf_counter() - started
        line = (
            f"epoch {epoch + 1}/{args.epochs} loss={total_loss / max(len(loader), 1):.4f} "
            f"real_val_CER={cer:.4f} real_exact={exact:.3f} ({count}) "
            f"synth_val_CER={synth_cer:.4f} elapsed={elapsed / 60:.1f}min"
        )
        print(line, flush=True)
        if cer < best_cer:
            best_cer = cer
            torch.save(
                {
                    "state_dict": model.cpu().state_dict(),
                    "alphabet": ALPHABET,
                    "seed": args.seed,
                    "real_val_cer": cer,
                    "real_val_exact": exact,
                },
                args.out,
            )
            model.to(device)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                for target, predicted, plate_type in shown:
                    handle.write(f"  {plate_type:6s} GT={target} PRED={predicted}\n")

    print(f"Best real validation CER: {best_cer:.4f}")
    print(f"Checkpoint: {args.out}")
    print("Validation uses manually annotated real crops; it is not a hidden-test score.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
