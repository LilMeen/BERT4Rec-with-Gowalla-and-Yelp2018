# BERT4Rec on Gowalla

This workspace includes:
1. Gowalla preprocessing modules for BERT4Rec input format.
2. Yelp2018 preprocessing modules for BERT4Rec input format.
3. BERT4Rec model setup (adapted from FeiSun/BERT4Rec architecture) with train/eval scripts.
4. Single best-checkpoint saving (always overwrite one checkpoint file).

## Project Structure

- `datasets/Gowalla_totalCheckins.txt`: raw Gowalla file
- `datasets/Yelp-2018/yelp_academic_dataset_review.json`: raw Yelp2018 review file
- `preprocessing/gowalla.py`: convert raw Gowalla to train/valid/test sequences
- `preprocessing/yelp2018.py`: convert Yelp2018 review interactions to train/valid/test sequences
- `bert4rec/model.py`: BERT4Rec model
- `bert4rec/data.py`: train/eval datasets and sequence loaders
- `bert4rec/trainer.py`: training loop, evaluation metrics, checkpoint handling
- `scripts/train_bert4rec.py`: entrypoint for training and evaluation
- `scripts/train_bert4rec_yelp2018.py`: Yelp2018-focused train/eval entrypoint

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Preprocess Gowalla

```bash
python -m preprocessing.gowalla \
  --input datasets/Gowalla_totalCheckins.txt \
  --output-dir data/processed/gowalla \
  --min-interactions 5
```

Outputs:
- `data/processed/gowalla/train.txt`
- `data/processed/gowalla/valid.txt`
- `data/processed/gowalla/test.txt`
- `data/processed/gowalla/metadata.json`
- mapping files (`user_id_map.json`, `item_id_map.json`)

## 3) Train + Evaluate

```bash
python scripts/train_bert4rec.py \
  --data-dir data/processed/gowalla \
  --output-dir outputs/bert4rec/gowalla \
  --log-file outputs/bert4rec/gowalla/train_eval_log.txt \
  --epochs 20 \
  --batch-size 256 \
  --eval-batch-size 256 \
  --max-len 200 \
  --device cuda
```

If no CUDA is available, the code automatically falls back to CPU.

## Checkpoint Policy

- Best model is selected by `Recall@10` on validation set.
- Only one checkpoint is kept per run output folder, for example: `outputs/bert4rec/gowalla/best_model.pt`.
- New best checkpoints overwrite the previous one to save disk space.

## Metrics

After training:
- Training/evaluation logs are written to `<output-dir>/train_eval_log.txt`.
- `<output-dir>/metrics.json` is created with:
  - `best_epoch`
  - `best_valid_metrics`
  - `test_metrics`
  - checkpoint path

## Yelp2018 Workflow

### 1) Preprocess Yelp2018

```bash
python -m preprocessing.yelp2018 \
  --input datasets/Yelp-2018/yelp_academic_dataset_review.json \
  --output-dir data/processed/yelp2018 \
  --min-user-interactions 5 \
  --min-item-interactions 5
```

Optional quick run on partial data:

```bash
python -m preprocessing.yelp2018 \
  --input datasets/Yelp-2018/yelp_academic_dataset_review.json \
  --output-dir data/processed/yelp2018_small \
  --min-user-interactions 1 \
  --min-item-interactions 1 \
  --max-lines 100000
```

### 2) Train + Evaluate on Yelp2018

```bash
python scripts/train_bert4rec_yelp2018.py \
  --data-dir data/processed/yelp2018 \
  --output-dir outputs/bert4rec/yelp2018 \
  --log-file outputs/bert4rec/yelp2018/train_eval_log.txt \
  --epochs 20 \
  --batch-size 256 \
  --eval-batch-size 256 \
  --max-len 200 \
  --device cuda
```

### 3) One-command Run Script (Yelp2018)

```bash
bash run_yelp2018.sh
```
