#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="./checkpoints/gowalla"
RESULTS_DIR="./results/BERT4Rec"
DATASET="gowalla"
DIM=64
MAX_SEQ_LENGTH=50

python -u run_torch.py \
  --dataset_name "${DATASET}" \
  --data_dir "./data" \
  --bert_config_file "./bert_train/bert_config_${DATASET}_${DIM}.json" \
  --checkpointDir "${CKPT_DIR}" \
  --results_dir "${RESULTS_DIR}" \
  --signature "torch-${DIM}" \
  --max_seq_length "${MAX_SEQ_LENGTH}" \
  --batch_size 256 \
  --epochs 50 \
  --learning_rate 1e-4 \
  --weight_decay 1e-4 \
  --masked_lm_prob 0.4 \
  --seed 2020
