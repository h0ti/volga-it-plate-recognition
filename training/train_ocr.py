"""Train the synthetic Russian plate OCR model with CTC loss."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from solution.ocr import (
    ALPHABET,
    CHAR_TO_ID,
    INPUT_HEIGHT,
    PlateCRNN,
    greedy_decode,
    render_training_sample,
)


class SyntheticOCRDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    """Fixed-seed synthetic plates held as compact uint8 arrays."""

    def __init__(self, count: int, seed: int) -> None:
        self.samples: list[tuple[torch.Tensor, torch.Tensor, str]] = []
        rng = random.Random(seed)
        for index in range(count):
            image, text, plate_type = render_training_sample(rng)
            self.samples.append(
                (image, torch.tensor([CHAR_TO_ID[c] for c in text], dtype=torch.long), plate_type)
            )
            if (index + 1) % 1000 == 0:
                print(f"Rendered OCR samples: {index + 1}/{count}", flush=True)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        return self.samples[index]


def collate_batch(
    batch: Sequence[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Pad images for a batch and concatenate CTC targets."""
    max_width = max(item[0].shape[-1] for item in batch)
    max_width = ((max_width + 3) // 4) * 4
    images = torch.full((len(batch), 1, INPUT_HEIGHT, max_width), 255, dtype=torch.uint8)
    target_lengths = torch.tensor([len(item[1]) for item in batch], dtype=torch.long)
    targets = torch.cat([item[1] for item in batch])
    widths = torch.tensor([item[0].shape[-1] for item in batch], dtype=torch.long)
    types = [item[2] for item in batch]
    for index, (image, _, _) in enumerate(batch):
        images[index, :, :, :image.shape[-1]] = image
    return images.float().div_(255.0), targets, widths, target_lengths, types


def edit_distance(left: str, right: str) -> int:
    """Compute Levenshtein distance for validation character error rate."""
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
def evaluate(model: PlateCRNN, loader: DataLoader, device: torch.device) -> float:
    """Return character error rate over a held-out synthetic split."""
    errors = 0
    characters = 0
    model.eval()
    for images, targets, widths, target_lengths, _ in loader:
        logits = model(images.to(device))
        offset = 0
        for index, target_length in enumerate(target_lengths.tolist()):
            target = "".join(
                {value: key for key, value in CHAR_TO_ID.items()}[token]
                for token in targets[offset:offset + target_length].tolist()
            )
            offset += target_length
            predicted, _ = greedy_decode(logits[index], int(widths[index].item()) // 4)
            errors += edit_distance(predicted, target)
            characters += len(target)
    return errors / max(characters, 1)


def build_parser() -> argparse.ArgumentParser:
    """Create OCR trainer arguments."""
    parser = argparse.ArgumentParser(
        description="Train a compact CRNN/CTC OCR model on synthetic plate crops."
    )
    parser.add_argument("--out", type=Path, default=Path("training/run_plates/ocr_best.pt"))
    parser.add_argument("--samples", type=int, default=12000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--device", default="0")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate samples, train the CRNN and save the best synthetic checkpoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.samples < 100 or args.epochs < 1 or args.batch < 1:
        parser.error("--samples must be >=100; epochs and batch must be positive.")
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        if not torch.cuda.is_available():
            print("CUDA is not available; refusing to run OCR training on CPU.", file=sys.stderr)
            return 1
        device = torch.device(f"cuda:{args.device}")
    if args.out.exists():
        parser.error(f"Refusing to overwrite existing checkpoint: {args.out}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True

    try:
        full_dataset = SyntheticOCRDataset(args.samples, args.seed)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not render OCR training data: {exc}", file=sys.stderr)
        return 1
    val_count = max(1, round(args.samples * 0.1))
    train_data, val_data = torch.utils.data.random_split(
        full_dataset,
        [args.samples - val_count, val_count],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_data, batch_size=args.batch, shuffle=True, num_workers=0, collate_fn=collate_batch
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch, shuffle=False, num_workers=0, collate_fn=collate_batch
    )

    model = PlateCRNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    best_cer = float("inf")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, targets, widths, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            input_lengths = widths.to(device) // 4
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            loss = criterion(
                log_probs,
                targets,
                input_lengths,
                target_lengths.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.item())
        cer = evaluate(model, val_loader, device)
        mean_loss = total_loss / max(len(train_loader), 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: loss={mean_loss:.4f}, "
            f"synthetic_val_CER={cer:.4f}",
            flush=True,
        )
        if cer < best_cer:
            best_cer = cer
            torch.save(
                {
                    "state_dict": model.cpu().state_dict(),
                    "alphabet": ALPHABET,
                    "seed": args.seed,
                    "samples": args.samples,
                    "synthetic_val_cer": cer,
                },
                args.out,
            )
            model.to(device)
    print(f"Best synthetic validation CER: {best_cer:.4f}")
    print(f"Checkpoint: {args.out}")
    print("Synthetic validation is not a measure of performance on real photographs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
