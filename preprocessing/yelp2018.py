from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class SequenceSplit:
    train_sequences: List[List[int]]
    valid_sequences: List[List[int]]
    test_sequences: List[List[int]]


def _iter_yelp_reviews(input_path: Path, max_lines: int | None = None) -> Iterable[Tuple[str, str, str]]:
    """
    Yield tuples as (raw_user_id, date_iso, raw_business_id).
    Yelp review dataset format: one JSON object per line.
    """
    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if max_lines is not None and idx > max_lines:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            raw_user = row.get("user_id")
            raw_item = row.get("business_id")
            date_iso = row.get("date")
            if not raw_user or not raw_item or not date_iso:
                continue
            yield raw_user, date_iso, raw_item


def _count_entities(
    input_path: Path,
    max_lines: int | None,
) -> Tuple[Counter, Counter]:
    user_counts: Counter = Counter()
    item_counts: Counter = Counter()

    for raw_user, _date_iso, raw_item in _iter_yelp_reviews(input_path, max_lines=max_lines):
        user_counts[raw_user] += 1
        item_counts[raw_item] += 1

    return user_counts, item_counts


def _split_sequences_for_bert4rec(sequences: Sequence[Sequence[int]]) -> SequenceSplit:
    train_sequences: List[List[int]] = []
    valid_sequences: List[List[int]] = []
    test_sequences: List[List[int]] = []

    for seq in sequences:
        if len(seq) < 3:
            continue
        train_sequences.append(list(seq[:-2]))
        valid_sequences.append(list(seq[:-1]))
        test_sequences.append(list(seq))

    return SequenceSplit(
        train_sequences=train_sequences,
        valid_sequences=valid_sequences,
        test_sequences=test_sequences,
    )


def _write_sequence_file(sequences: Sequence[Sequence[int]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(" ".join(str(x) for x in seq))
            f.write("\n")


def preprocess_yelp2018(
    input_path: Path,
    output_dir: Path,
    min_user_interactions: int = 5,
    min_item_interactions: int = 5,
    drop_consecutive_duplicates: bool = True,
    max_lines: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    user_counts, item_counts = _count_entities(input_path=input_path, max_lines=max_lines)
    candidate_users = {u for u, c in user_counts.items() if c >= min_user_interactions}
    candidate_items = {i for i, c in item_counts.items() if c >= min_item_interactions}

    user_id_map: Dict[str, int] = {}
    item_id_map: Dict[str, int] = {}
    next_user_id = 1
    next_item_id = 1

    db_path = output_dir / "_tmp_interactions.sqlite"
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE interactions (
            user_int INTEGER NOT NULL,
            event_time TEXT NOT NULL,
            item_int INTEGER NOT NULL
        )
        """
    )

    insert_buffer: List[Tuple[int, str, int]] = []
    flush_size = 20000

    for raw_user, date_iso, raw_item in _iter_yelp_reviews(input_path, max_lines=max_lines):
        if raw_user not in candidate_users or raw_item not in candidate_items:
            continue

        if raw_user not in user_id_map:
            user_id_map[raw_user] = next_user_id
            next_user_id += 1
        if raw_item not in item_id_map:
            item_id_map[raw_item] = next_item_id
            next_item_id += 1

        insert_buffer.append((user_id_map[raw_user], date_iso, item_id_map[raw_item]))
        if len(insert_buffer) >= flush_size:
            conn.executemany(
                "INSERT INTO interactions (user_int, event_time, item_int) VALUES (?, ?, ?)",
                insert_buffer,
            )
            conn.commit()
            insert_buffer.clear()

    if insert_buffer:
        conn.executemany(
            "INSERT INTO interactions (user_int, event_time, item_int) VALUES (?, ?, ?)",
            insert_buffer,
        )
        conn.commit()
        insert_buffer.clear()

    conn.execute("CREATE INDEX idx_user_time ON interactions(user_int, event_time)")
    conn.commit()

    sequences: List[List[int]] = []
    current_user = None
    current_seq: List[int] = []

    cursor = conn.execute(
        "SELECT user_int, item_int FROM interactions ORDER BY user_int ASC, event_time ASC"
    )

    for user_int, item_int in cursor:
        if current_user is None:
            current_user = user_int

        if user_int != current_user:
            if len(current_seq) >= max(min_user_interactions, 3):
                sequences.append(current_seq)
            current_user = user_int
            current_seq = []

        if drop_consecutive_duplicates and current_seq and current_seq[-1] == item_int:
            continue
        current_seq.append(item_int)

    if current_seq and len(current_seq) >= max(min_user_interactions, 3):
        sequences.append(current_seq)

    conn.close()
    if db_path.exists():
        db_path.unlink()

    split = _split_sequences_for_bert4rec(sequences)
    _write_sequence_file(split.train_sequences, output_dir / "train.txt")
    _write_sequence_file(split.valid_sequences, output_dir / "valid.txt")
    _write_sequence_file(split.test_sequences, output_dir / "test.txt")

    metadata = {
        "dataset": "yelp2018",
        "source_file": str(input_path),
        "num_users": len(split.train_sequences),
        "num_items": len(item_id_map),
        "min_user_interactions": min_user_interactions,
        "min_item_interactions": min_item_interactions,
        "drop_consecutive_duplicates": drop_consecutive_duplicates,
        "max_lines": max_lines,
        "user_id_starts_from": 1,
        "item_id_starts_from": 1,
        "padding_token_id": 0,
        "mask_token_id": len(item_id_map) + 1,
        "vocab_size": len(item_id_map) + 2,
    }

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    with (output_dir / "user_id_map.json").open("w", encoding="utf-8") as f:
        json.dump(user_id_map, f)

    with (output_dir / "item_id_map.json").open("w", encoding="utf-8") as f:
        json.dump(item_id_map, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Yelp2018 for BERT4Rec")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("datasets/Yelp-2018/yelp_academic_dataset_review.json"),
        help="Path to yelp_academic_dataset_review.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/yelp2018"),
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=5,
        help="Filter users with interactions below this threshold",
    )
    parser.add_argument(
        "--min-item-interactions",
        type=int,
        default=5,
        help="Filter items with interactions below this threshold",
    )
    parser.add_argument(
        "--keep-consecutive-duplicates",
        action="store_true",
        help="Keep consecutive duplicate interactions; default drops them",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
        help="Limit number of input lines for quick experiments",
    )

    args = parser.parse_args()

    preprocess_yelp2018(
        input_path=args.input,
        output_dir=args.output_dir,
        min_user_interactions=args.min_user_interactions,
        min_item_interactions=args.min_item_interactions,
        drop_consecutive_duplicates=not args.keep_consecutive_duplicates,
        max_lines=args.max_lines,
    )
