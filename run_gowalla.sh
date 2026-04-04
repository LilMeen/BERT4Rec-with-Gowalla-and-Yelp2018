#!/usr/bin/env bash
set -euo pipefail

VENV_DIR=".venv"

if [ -n "${PYTHON_BIN:-}" ]; then
  :
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "Error: python/python3 not found in PATH"
  exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
if [ -f "$VENV_DIR/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
elif [ -f "$VENV_DIR/Scripts/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/Scripts/activate"
else
  echo "Error: cannot find virtualenv activate script in $VENV_DIR"
  exit 1
fi

cleanup() {
  if command -v deactivate >/dev/null 2>&1; then
    deactivate
  fi
}
trap cleanup EXIT

python -m pip install --upgrade pip
python -m pip install -r requirements.txt


python scripts/train_bert4rec.py \
  --data-dir data/processed/gowalla \
  --output-dir outputs/bert4rec/gowalla \
  --log-file outputs/bert4rec/gowalla/train_eval_log.txt \
  --epochs 20 \
  --batch-size 256 \
  --eval-batch-size 256 \
  --max-len 200 \
  --device cuda
