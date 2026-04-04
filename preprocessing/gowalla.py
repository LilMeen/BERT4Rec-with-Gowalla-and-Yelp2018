from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class SequenceSplit:
    train_sequences: List[List[int]]
    valid_sequences: List[List[int]]
    test_sequences: List[List[int]]


def _iter_gowalla_rows(dataset_path: Path) -> Iterable[Tuple[str, str, str]]:
    """
    Yield rows as (raw_user_id, timestamp_iso, raw_item_id).
    Gowalla format: user_id\ttimestamp\tlat\tlon\tlocation_id
    """
    with dataset_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) != 5:
                continue
            raw_user, timestamp_iso, _lat, _lon, raw_item = row
            yield raw_user, timestamp_iso, raw_item


def _first_pass_count_user_interactions(dataset_path: Path) -> Counter:
    user_counts: Counter = Counter()
    for raw_user, _timestamp, _raw_item in _iter_gowalla_rows(dataset_path):
        user_counts[raw_user] += 1
    return user_counts


def _build_sequences(
    dataset_path: Path,
    min_interactions: int,
    drop_consecutive_duplicates: bool,
) -> Tuple[List[List[int]], Dict[str, int], Dict[str, int]]:
    user_counts = _first_pass_count_user_interactions(dataset_path)

    candidate_users = {u for u, c in user_counts.items() if c >= min_interactions}
    user_events: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for raw_user, timestamp_iso, raw_item in _iter_gowalla_rows(dataset_path):
        if raw_user not in candidate_users:
            continue
        user_events[raw_user].append((timestamp_iso, raw_item))

    user_id_map: Dict[str, int] = {}
    item_id_map: Dict[str, int] = {}

    next_user_id = 1
    next_item_id = 1
    sequences: List[List[int]] = []

    for raw_user in sorted(user_events.keys(), key=lambda x: int(x)):
        events = sorted(user_events[raw_user], key=lambda t: t[0])
        raw_items = [raw_item for _ts, raw_item in events]

        if drop_consecutive_duplicates and raw_items:
            dedup_items = [raw_items[0]]
            for item in raw_items[1:]:
                if item != dedup_items[-1]:
                    dedup_items.append(item)
            raw_items = dedup_items

        if len(raw_items) < max(min_interactions, 3):
            continue

        user_id_map[raw_user] = next_user_id
        next_user_id += 1

        mapped_seq: List[int] = []
        for raw_item in raw_items:
            if raw_item not in item_id_map:
                item_id_map[raw_item] = next_item_id
                next_item_id += 1
            mapped_seq.append(item_id_map[raw_item])

        sequences.append(mapped_seq)

    return sequences, user_id_map, item_id_map


def _split_sequences_for_bert4rec(sequences: List[List[int]]) -> SequenceSplit:
    train_sequences: List[List[int]] = []
    valid_sequences: List[List[int]] = []
    test_sequences: List[List[int]] = []

    for seq in sequences:
        # Need at least 3 items for train/valid/test style split.
        if len(seq) < 3:
            continue

        train_sequences.append(seq[:-2])
        valid_sequences.append(seq[:-1])
        test_sequences.append(seq)

    return SequenceSplit(
        train_sequences=train_sequences,
        valid_sequences=valid_sequences,
        test_sequences=test_sequences,
    )


def _write_sequence_file(sequences: List[List[int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(" ".join(str(item) for item in seq))
            f.write("\n")


def preprocess_gowalla(
    dataset_path: Path,
    output_dir: Path,
    min_interactions: int = 5,
    drop_consecutive_duplicates: bool = True,
) -> None:
    sequences, user_id_map, item_id_map = _build_sequences(
        dataset_path=dataset_path,
        min_interactions=min_interactions,
        drop_consecutive_duplicates=drop_consecutive_duplicates,
    )

    split = _split_sequences_for_bert4rec(sequences)

    _write_sequence_file(split.train_sequences, output_dir / "train.txt")
    _write_sequence_file(split.valid_sequences, output_dir / "valid.txt")
    _write_sequence_file(split.test_sequences, output_dir / "test.txt")

    metadata = {
        "dataset": "gowalla",
        "source_file": str(dataset_path),
        "num_users": len(split.train_sequences),
        "num_items": len(item_id_map),
        "min_interactions": min_interactions,
        "drop_consecutive_duplicates": drop_consecutive_duplicates,
        "user_id_starts_from": 1,
        "item_id_starts_from": 1,
        "padding_token_id": 0,
        "mask_token_id": len(item_id_map) + 1,
        "vocab_size": len(item_id_map) + 2,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with (output_dir / "user_id_map.json").open("w", encoding="utf-8") as f:
        json.dump(user_id_map, f)

    with (output_dir / "item_id_map.json").open("w", encoding="utf-8") as f:
        json.dump(item_id_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Gowalla for BERT4Rec")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/Gowalla_totalCheckins.txt"),
        help="Path to Gowalla_totalCheckins.txt",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/gowalla"),
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=5,
        help="Filter users with fewer interactions than this threshold",
    )
    parser.add_argument(
        "--keep-consecutive-duplicates",
        action="store_true",
        help="Keep consecutive duplicate check-ins; by default they are dropped",
    )

    args = parser.parse_args()

    preprocess_gowalla(
        dataset_path=args.input,
        output_dir=args.output_dir,
        min_interactions=args.min_interactions,
        drop_consecutive_duplicates=not args.keep_consecutive_duplicates,
    )
