from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import Dataset


class BERT4RecTrainDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        num_items: int,
        max_len: int,
        mask_prob: float,
    ) -> None:
        self.sequences = [list(seq) for seq in sequences if len(seq) > 0]
        self.num_items = num_items
        self.max_len = max_len
        self.mask_prob = mask_prob

        self.pad_token_id = 0
        self.mask_token_id = num_items + 1

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx][-self.max_len :]
        input_ids = list(seq)
        labels = [0] * len(input_ids)

        for i in range(len(input_ids)):
            token = input_ids[i]
            if random.random() < self.mask_prob:
                labels[i] = token
                p = random.random()
                if p < 0.8:
                    input_ids[i] = self.mask_token_id
                elif p < 0.9:
                    input_ids[i] = random.randint(1, self.num_items)
                else:
                    input_ids[i] = token

        pad_len = self.max_len - len(input_ids)
        input_ids = [self.pad_token_id] * pad_len + input_ids
        labels = [0] * pad_len + labels

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class BERT4RecEvalDataset(Dataset):
    def __init__(
        self,
        sequences: Sequence[Sequence[int]],
        num_items: int,
        max_len: int,
    ) -> None:
        self.sequences = [list(seq) for seq in sequences if len(seq) >= 2]
        self.num_items = num_items
        self.max_len = max_len

        self.pad_token_id = 0
        self.mask_token_id = num_items + 1

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int):
        seq = self.sequences[idx]
        target = seq[-1]
        prefix = seq[:-1]

        context = prefix[-(self.max_len - 1) :] + [self.mask_token_id]
        pad_len = self.max_len - len(context)
        input_ids = [self.pad_token_id] * pad_len + context

        # Used to filter already seen items during ranking.
        seen_items = set(prefix)

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(target, dtype=torch.long),
            torch.tensor(list(seen_items), dtype=torch.long),
        )


def collate_eval_batch(batch):
    input_ids = torch.stack([x[0] for x in batch], dim=0)
    targets = torch.stack([x[1] for x in batch], dim=0)
    seen_item_lists = [x[2] for x in batch]
    return input_ids, targets, seen_item_lists


def load_sequence_file(path: Path) -> List[List[int]]:
    sequences: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seq = [int(x) for x in line.split()]
            if seq:
                sequences.append(seq)
    return sequences
