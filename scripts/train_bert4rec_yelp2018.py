from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from bert4rec.data import load_sequence_file
from bert4rec.trainer import TrainConfig, train_and_evaluate


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate BERT4Rec on Yelp2018")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/processed/yelp2018"),
        help="Directory containing train.txt, valid.txt, test.txt, metadata.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/bert4rec/yelp2018"),
        help="Directory to save checkpoint and metrics",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to txt log file. Default: <output-dir>/train_eval_log.txt",
    )

    parser.add_argument("--max-len", type=int, default=200)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--mask-prob", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    return parser.parse_args()


def main():
    args = parse_args()

    metadata_path = args.data_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing metadata file: {metadata_path}. Run preprocessing first."
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_sequences = load_sequence_file(args.data_dir / "train.txt")
    valid_sequences = load_sequence_file(args.data_dir / "valid.txt")
    test_sequences = load_sequence_file(args.data_dir / "test.txt")

    config = TrainConfig(
        num_items=metadata["num_items"],
        max_len=args.max_len,
        hidden_size=args.hidden_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        mask_prob=args.mask_prob,
        num_workers=args.num_workers,
    )

    results = train_and_evaluate(
        train_sequences=train_sequences,
        valid_sequences=valid_sequences,
        test_sequences=test_sequences,
        config=config,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        log_file_path=args.log_file,
    )

    _ = results


if __name__ == "__main__":
    main()
