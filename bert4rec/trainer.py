from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, TextIO

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from bert4rec.data import BERT4RecEvalDataset, BERT4RecTrainDataset, collate_eval_batch
from bert4rec.model import BERT4Rec


@dataclass
class TrainConfig:
    num_items: int
    max_len: int
    hidden_size: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1

    batch_size: int = 256
    eval_batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 20
    mask_prob: float = 0.2
    num_workers: int = 0

    k_list: Sequence[int] = (5, 10, 20)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _recall_and_ndcg_at_k(rank_list: List[int], target: int, k: int) -> tuple[float, float]:
    topk = rank_list[:k]
    hit = 1.0 if target in topk else 0.0
    if hit > 0:
        rank = topk.index(target) + 1
        ndcg = 1.0 / np.log2(rank + 1)
    else:
        ndcg = 0.0
    return hit, ndcg


def evaluate(
    model: BERT4Rec,
    eval_loader: DataLoader,
    num_items: int,
    k_list: Sequence[int],
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    metric_sums = {f"Recall@{k}": 0.0 for k in k_list}
    metric_sums.update({f"NDCG@{k}": 0.0 for k in k_list})
    effective_k = {k: min(k, num_items) for k in k_list}

    num_examples = 0

    with torch.no_grad():
        for input_ids, targets, seen_item_lists in eval_loader:
            input_ids = input_ids.to(device)
            targets = targets.to(device)

            logits = model(input_ids)
            # Take prediction at the final position (the [MASK] token position)
            scores = logits[:, -1, :]

            # Disallow PAD (0) and MASK token (num_items + 1) from recommendation candidates.
            scores[:, 0] = -1e9
            scores[:, num_items + 1] = -1e9

            for row_idx, seen_items in enumerate(seen_item_lists):
                if seen_items.numel() > 0:
                    seen = seen_items.tolist()
                    # Keep target item rankable even if it appears in seen history.
                    if targets[row_idx].item() in seen:
                        seen.remove(targets[row_idx].item())
                    if seen:
                        scores[row_idx, seen] = -1e9

            max_k = max(effective_k.values())
            topk_indices = torch.topk(scores, k=max_k, dim=1).indices.cpu().tolist()
            targets_list = targets.cpu().tolist()

            for rank_list, target in zip(topk_indices, targets_list):
                for k in k_list:
                    recall_k, ndcg_k = _recall_and_ndcg_at_k(rank_list, target, effective_k[k])
                    metric_sums[f"Recall@{k}"] += recall_k
                    metric_sums[f"NDCG@{k}"] += ndcg_k
                num_examples += 1

    if num_examples == 0:
        return {m: 0.0 for m in metric_sums}

    return {m: v / num_examples for m, v in metric_sums.items()}


def train_and_evaluate(
    train_sequences: Sequence[Sequence[int]],
    valid_sequences: Sequence[Sequence[int]],
    test_sequences: Sequence[Sequence[int]],
    config: TrainConfig,
    output_dir: Path,
    seed: int = 42,
    device: str = "cuda",
    log_file_path: Path | None = None,
) -> Dict[str, float]:
    set_seed(seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = output_dir / "best_model.pt"

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch_device = torch.device(device)

    model = BERT4Rec(
        num_items=config.num_items,
        max_len=config.max_len,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(torch_device)

    train_dataset = BERT4RecTrainDataset(
        sequences=train_sequences,
        num_items=config.num_items,
        max_len=config.max_len,
        mask_prob=config.mask_prob,
    )
    valid_dataset = BERT4RecEvalDataset(
        sequences=valid_sequences,
        num_items=config.num_items,
        max_len=config.max_len,
    )
    test_dataset = BERT4RecEvalDataset(
        sequences=test_sequences,
        num_items=config.num_items,
        max_len=config.max_len,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_eval_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_eval_batch,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_metric = -1.0
    best_epoch = -1

    def _log(message: str, file_handle: TextIO | None) -> None:
        if file_handle is None:
            return
        file_handle.write(message + "\n")
        file_handle.flush()

    if log_file_path is None:
        log_file_path = output_dir / "train_eval_log.txt"
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    with log_file_path.open("w", encoding="utf-8") as log_f:
        _log("=== BERT4Rec Training Log ===", log_f)
        _log(f"output_dir={output_dir}", log_f)
        _log(f"device={torch_device}", log_f)
        _log(f"num_train_sequences={len(train_sequences)}", log_f)
        _log(f"num_valid_sequences={len(valid_sequences)}", log_f)
        _log(f"num_test_sequences={len(test_sequences)}", log_f)
        _log("", log_f)

        for epoch in range(1, config.epochs + 1):
            model.train()
            total_loss = 0.0

            for input_ids, labels in train_loader:
                input_ids = input_ids.to(torch_device)
                labels = labels.to(torch_device)

                optimizer.zero_grad()
                logits = model(input_ids)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=0,
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / max(len(train_loader), 1)
            valid_metrics = evaluate(
                model=model,
                eval_loader=valid_loader,
                num_items=config.num_items,
                k_list=config.k_list,
                device=torch_device,
            )

            key_metric = valid_metrics.get("Recall@10", 0.0)
            _log(
                f"Epoch {epoch:03d} | train_loss={avg_train_loss:.4f} | "
                f"valid Recall@10={key_metric:.4f}",
                log_f,
            )

            if key_metric > best_metric:
                best_metric = key_metric
                best_epoch = epoch
                ckpt_payload = {
                    "model_state_dict": model.state_dict(),
                    "config": config.__dict__,
                    "best_epoch": best_epoch,
                    "best_valid_metrics": valid_metrics,
                }
                # Always overwrite same file => keep only one checkpoint.
                torch.save(ckpt_payload, best_ckpt_path)
                _log(
                    f"  -> New best checkpoint saved at epoch {epoch} (Recall@10={key_metric:.4f})",
                    log_f,
                )

        _log("", log_f)

    if not best_ckpt_path.exists():
        raise RuntimeError("Training finished but no checkpoint was saved.")

    best_data = torch.load(best_ckpt_path, map_location=torch_device, weights_only=False)
    model.load_state_dict(best_data["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        eval_loader=test_loader,
        num_items=config.num_items,
        k_list=config.k_list,
        device=torch_device,
    )

    results = {
        "best_epoch": best_epoch,
        "best_valid_metrics": best_data.get("best_valid_metrics", {}),
        "test_metrics": test_metrics,
        "checkpoint": str(best_ckpt_path),
    }

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with log_file_path.open("a", encoding="utf-8") as log_f:
        _log("=== Final Results ===", log_f)
        _log(json.dumps(results, indent=2), log_f)

    return results
