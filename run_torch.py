#!/usr/bin/env python3
# coding=utf-8

import argparse
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


@dataclass
class SplitData:
    train: Dict[int, List[int]]
    valid: Dict[int, int]
    test: Dict[int, int]
    user_count: int
    item_count: int


class ClozeTrainDataset(Dataset):
    def __init__(self, windows: List[List[int]], max_len: int, mask_token: int, masked_lm_prob: float):
        self.windows = windows
        self.max_len = max_len
        self.mask_token = mask_token
        self.masked_lm_prob = masked_lm_prob

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        seq = self.windows[idx]
        input_ids = np.array(seq, dtype=np.int64)
        labels = np.zeros_like(input_ids)
        attention_mask = (input_ids > 0).astype(np.int64)

        candidate_positions = np.where(input_ids > 0)[0]
        if len(candidate_positions) == 0:
            return {
                "input_ids": torch.from_numpy(input_ids),
                "attention_mask": torch.from_numpy(attention_mask),
                "labels": torch.from_numpy(labels),
            }

        mask = np.random.rand(len(candidate_positions)) < self.masked_lm_prob
        masked_positions = candidate_positions[mask]
        if len(masked_positions) == 0:
            masked_positions = np.array([random.choice(candidate_positions.tolist())], dtype=np.int64)

        labels[masked_positions] = input_ids[masked_positions]
        input_ids[masked_positions] = self.mask_token

        return {
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(attention_mask),
            "labels": torch.from_numpy(labels),
        }


class BERT4RecTorch(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        hidden_dropout_prob: float,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()
        self.item_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.pos_embedding = nn.Embedding(max_len, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_attention_heads,
            dim_feedforward=intermediate_size,
            dropout=attention_probs_dropout_prob,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_hidden_layers)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.size()
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)

        x = self.item_embedding(input_ids) + self.pos_embedding(pos_ids)
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Transformer expects True at positions to ignore.
        key_padding_mask = attention_mask == 0
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        logits = self.out(x)
        return logits


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_interactions(path: str) -> Dict[int, List[int]]:
    user_items = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = int(parts[0])
            item = int(parts[1])
            user_items[user].append(item)
    return user_items


def build_remap(user_items: Dict[int, List[int]]) -> Tuple[Dict[int, int], Dict[int, int]]:
    item_set = set()
    for seq in user_items.values():
        item_set.update(seq)
    sorted_items = sorted(item_set)
    item2idx = {item: idx + 1 for idx, item in enumerate(sorted_items)}
    idx2item = {idx: item for item, idx in item2idx.items()}
    return item2idx, idx2item


def split_by_user(user_items: Dict[int, List[int]]) -> SplitData:
    train = {}
    valid = {}
    test = {}
    max_item = 0

    for u, seq in user_items.items():
        if not seq:
            continue
        max_item = max(max_item, max(seq))

        if len(seq) < 3:
            train[u] = seq[:]
            continue

        train[u] = seq[:-2]
        valid[u] = seq[-2]
        test[u] = seq[-1]

    return SplitData(
        train=train,
        valid=valid,
        test=test,
        user_count=len(user_items),
        item_count=max_item,
    )


def remap_split(split: SplitData, item2idx: Dict[int, int]) -> SplitData:
    train = {u: [item2idx[i] for i in seq if i in item2idx] for u, seq in split.train.items()}
    valid = {u: item2idx[i] for u, i in split.valid.items() if i in item2idx}
    test = {u: item2idx[i] for u, i in split.test.items() if i in item2idx}
    return SplitData(train=train, valid=valid, test=test, user_count=split.user_count, item_count=len(item2idx))


def left_pad_truncate(seq: List[int], max_len: int) -> List[int]:
    if len(seq) >= max_len:
        return seq[-max_len:]
    return [0] * (max_len - len(seq)) + seq


def build_train_windows(train_sequences: Dict[int, List[int]], max_len: int) -> List[List[int]]:
    windows = []
    for seq in train_sequences.values():
        if len(seq) == 0:
            continue
        if len(seq) <= max_len:
            windows.append(left_pad_truncate(seq, max_len))
        else:
            for end in range(max_len, len(seq) + 1):
                window = seq[end - max_len:end]
                windows.append(window)
    return windows


def get_last_masked_input(seq: List[int], mask_token: int, max_len: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    seq = seq + [mask_token]
    padded = left_pad_truncate(seq, max_len)
    attn = [1 if t > 0 else 0 for t in padded]
    target_pos = len(padded) - 1
    input_ids = torch.tensor(padded, dtype=torch.long)
    attention_mask = torch.tensor(attn, dtype=torch.long)
    return input_ids, attention_mask, target_pos


def sample_negatives(
    item_count: int,
    rated_set: set,
    true_item: int,
    n_neg: int,
    popularity_ids: List[int],
    popularity_probs: List[float],
) -> List[int]:
    negatives = []
    tried = set()
    max_trials = n_neg * 20
    trials = 0

    while len(negatives) < n_neg and trials < max_trials:
        trials += 1
        cand = int(np.random.choice(popularity_ids, p=popularity_probs))
        if cand == true_item or cand in rated_set or cand in tried:
            continue
        tried.add(cand)
        negatives.append(cand)

    if len(negatives) < n_neg:
        for cand in range(1, item_count + 1):
            if cand == true_item or cand in rated_set or cand in tried:
                continue
            tried.add(cand)
            negatives.append(cand)
            if len(negatives) == n_neg:
                break

    return negatives[:n_neg]


def rank_metrics(rank: int) -> Dict[str, float]:
    out = {
        "hit@1": 1.0 if rank < 1 else 0.0,
        "ndcg@1": 1.0 if rank < 1 else 0.0,
        "hit@5": 1.0 if rank < 5 else 0.0,
        "ndcg@5": 1.0 / math.log2(rank + 2) if rank < 5 else 0.0,
        "hit@10": 1.0 if rank < 10 else 0.0,
        "ndcg@10": 1.0 / math.log2(rank + 2) if rank < 10 else 0.0,
        "ap": 1.0 / (rank + 1),
    }
    return out


def evaluate_split(
    model: nn.Module,
    split: SplitData,
    split_name: str,
    device: torch.device,
    max_len: int,
    mask_token: int,
    item_pop_counter: Counter,
    eval_users_limit: int = 0,
):
    model.eval()
    pop_items = sorted(item_pop_counter.keys())
    pop_weights = np.array([item_pop_counter[i] for i in pop_items], dtype=np.float64)
    pop_weights = pop_weights / pop_weights.sum()

    users = list(split.test.keys() if split_name == "test" else split.valid.keys())
    if eval_users_limit > 0:
        users = users[:eval_users_limit]

    totals = Counter()
    valid_users = 0

    with torch.no_grad():
        for idx, u in enumerate(users, 1):
            train_seq = split.train.get(u, [])
            if split_name == "val":
                if u not in split.valid:
                    continue
                true_item = split.valid[u]
                context = train_seq
                rated = set(train_seq)
            else:
                if u not in split.test or u not in split.valid:
                    continue
                true_item = split.test[u]
                context = train_seq + [split.valid[u]]
                rated = set(context)

            input_ids, attention_mask, target_pos = get_last_masked_input(context, mask_token, max_len)
            input_ids = input_ids.unsqueeze(0).to(device)
            attention_mask = attention_mask.unsqueeze(0).to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)[0, target_pos]
            logits = logits.detach().cpu().numpy()

            negatives = sample_negatives(
                item_count=split.item_count,
                rated_set=rated,
                true_item=true_item,
                n_neg=100,
                popularity_ids=pop_items,
                popularity_probs=pop_weights,
            )
            candidates = [true_item] + negatives
            scores = logits[candidates]
            rank = int(scores.argsort().argsort()[0])

            m = rank_metrics(rank)
            totals.update(m)
            valid_users += 1

            if idx % 1000 == 0:
                print(f"[{split_name}] evaluated users: {idx}")

    if valid_users == 0:
        return {
            "valid_user": 0,
            "hit@1": 0.0,
            "ndcg@1": 0.0,
            "hit@5": 0.0,
            "ndcg@5": 0.0,
            "hit@10": 0.0,
            "ndcg@10": 0.0,
            "ap": 0.0,
        }

    return {
        "valid_user": valid_users,
        "hit@1": totals["hit@1"] / valid_users,
        "ndcg@1": totals["ndcg@1"] / valid_users,
        "hit@5": totals["hit@5"] / valid_users,
        "ndcg@5": totals["ndcg@5"] / valid_users,
        "hit@10": totals["hit@10"] / valid_users,
        "ndcg@10": totals["ndcg@10"] / valid_users,
        "ap": totals["ap"] / valid_users,
    }


def main():
    parser = argparse.ArgumentParser(description="PyTorch BERT4Rec with user-item txt data")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--bert_config_file", type=str, required=True)
    parser.add_argument("--checkpointDir", type=str, default="./checkpoints")
    parser.add_argument("--results_dir", type=str, default="./results/BERT4Rec")
    parser.add_argument("--signature", type=str, default="default")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_train_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--masked_lm_prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--eval_users_limit", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    if torch.cuda.is_available():
        print(f"[INFO] cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}")

    with open(args.bert_config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    config_max_len = int(cfg["max_position_embeddings"])
    max_len = int(args.max_seq_length) if args.max_seq_length and args.max_seq_length > 0 else config_max_len
    print(f"[INFO] max_seq_length={max_len} (config={config_max_len})")
    hidden_size = int(cfg["hidden_size"])
    num_hidden_layers = int(cfg["num_hidden_layers"])
    num_attention_heads = int(cfg["num_attention_heads"])
    intermediate_size = int(cfg["intermediate_size"])
    hidden_dropout_prob = float(cfg["hidden_dropout_prob"])
    attention_probs_dropout_prob = float(cfg["attention_probs_dropout_prob"])

    data_path = os.path.join(args.data_dir, f"{args.dataset_name}.txt")
    print(f"[INFO] loading interactions from {data_path}")
    raw_user_items = load_interactions(data_path)
    raw_split = split_by_user(raw_user_items)
    item2idx, _ = build_remap(raw_user_items)
    split = remap_split(raw_split, item2idx)

    mask_token = split.item_count + 1
    vocab_size = split.item_count + 2  # 0: pad, 1..N: items, N+1: mask

    print(
        "[INFO] users={}, items={}, train_users={}, valid_users={}, test_users={}".format(
            split.user_count,
            split.item_count,
            len(split.train),
            len(split.valid),
            len(split.test),
        )
    )

    windows = build_train_windows(split.train, max_len=max_len)
    print(f"[INFO] generated train windows={len(windows)}")

    train_dataset = ClozeTrainDataset(
        windows=windows,
        max_len=max_len,
        mask_token=mask_token,
        masked_lm_prob=args.masked_lm_prob,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = BERT4RecTorch(
        vocab_size=vocab_size,
        max_len=max_len,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        hidden_dropout_prob=hidden_dropout_prob,
        attention_probs_dropout_prob=attention_probs_dropout_prob,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    item_pop_counter = Counter()
    for seq in split.train.values():
        item_pop_counter.update(seq)

    os.makedirs(args.checkpointDir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    global_step = 0
    epoch_logs = []
    val_metrics_history = []
    best_val_ndcg10 = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(args.checkpointDir, f"{args.dataset_name}_{args.signature}_best.pt")

    print("[INFO] start training")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            global_step += 1

            if step % 100 == 0:
                avg_so_far = running_loss / step
                print(f"[TRAIN] epoch={epoch} step={step}/{len(train_loader)} loss={avg_so_far:.6f}")

            if args.num_train_steps > 0 and global_step >= args.num_train_steps:
                break

        avg_loss = running_loss / max(1, len(train_loader))
        epoch_time = time.time() - epoch_start

        epoch_log = {
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "epoch_time_seconds": round(epoch_time, 2),
            "global_step": global_step,
        }
        if torch.cuda.is_available():
            epoch_log["gpu_memory_peak_MB"] = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
            torch.cuda.reset_peak_memory_stats()

        epoch_logs.append(epoch_log)
        print(f"[EPOCH] {epoch_log}")

        if epoch % args.eval_every == 0:
            val_metrics = evaluate_split(
                model=model,
                split=split,
                split_name="val",
                device=device,
                max_len=max_len,
                mask_token=mask_token,
                item_pop_counter=item_pop_counter,
                eval_users_limit=args.eval_users_limit,
            )
            val_metrics["epoch"] = epoch
            val_metrics_history.append(val_metrics)
            print(f"[VAL] epoch={epoch} metrics={val_metrics}")

            if val_metrics["ndcg@10"] > best_val_ndcg10:
                best_val_ndcg10 = val_metrics["ndcg@10"]
                best_epoch = epoch
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "args": vars(args),
                        "bert_config": cfg,
                        "item_count": split.item_count,
                        "mask_token": mask_token,
                    },
                    best_ckpt_path,
                )
                print(f"[INFO] saved best checkpoint: {best_ckpt_path}")

        if args.num_train_steps > 0 and global_step >= args.num_train_steps:
            print(f"[INFO] reached num_train_steps={args.num_train_steps}, stopping training")
            break

    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[INFO] loaded best checkpoint from epoch={ckpt.get('epoch', -1)}")

    test_metrics = evaluate_split(
        model=model,
        split=split,
        split_name="test",
        device=device,
        max_len=max_len,
        mask_token=mask_token,
        item_pop_counter=item_pop_counter,
        eval_users_limit=args.eval_users_limit,
    )
    print(f"[TEST] metrics={test_metrics}")

    result = {
        "config": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "num_train_steps": args.num_train_steps,
            "masked_lm_prob": args.masked_lm_prob,
            "max_len": max_len,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "intermediate_size": intermediate_size,
            "dropout": hidden_dropout_prob,
            "attention_dropout": attention_probs_dropout_prob,
        },
        "dataset": args.dataset_name,
        "model": "BERT4RecTorch",
        "seed": args.seed,
        "topks": [5, 10, 20],
        "dataset_stats": {
            "n_users": split.user_count,
            "n_items": split.item_count,
            "train_interactions": int(sum(len(v) for v in split.train.values())),
            "valid_users": len(split.valid),
            "test_users": len(split.test),
        },
        "epoch_logs": epoch_logs,
        "val_metrics_history": val_metrics_history,
        "best_epoch": best_epoch,
        "best_val_ndcg@10": best_val_ndcg10,
        "test_metrics": test_metrics,
    }

    result_path = os.path.join(args.results_dir, f"{args.dataset_name}_seed{args.seed}_run0.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"[INFO] wrote results to {result_path}")


if __name__ == "__main__":
    main()
