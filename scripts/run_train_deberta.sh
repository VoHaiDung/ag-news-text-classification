#!/usr/bin/env bash
set -euo pipefail

# Usage: ./scripts/run_train_deberta.sh [MODEL_DIR] [EPOCHS] [BATCH_SIZE] [LR]
MODEL_DIR=${1:-outputs/dapt_checkpoints/}
EPOCHS=${2:-7}
BATCH_SIZE=${3:-32}
LR=${4:-3e-5}

python -m src.train \
  --model_name "${MODEL_DIR}" \
  --train_split train_plus_pseudo \
  --val_split test \
  --output_dir outputs/checkpoints/deberta_lora \
  --num_train_epochs "${EPOCHS}" \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size 16 \
  --learning_rate "${LR}" \
  --stride 256 \
  --max_length 512
