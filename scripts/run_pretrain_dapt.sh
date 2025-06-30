#!/usr/bin/env bash
# Domain-Adaptive Pre-Training (DAPT) launcher for DeBERTa-v3
# Usage: bash scripts/run_pretrain_dapt.sh <UNLABELED_TEXT_FILE> [NUM_EPOCHS]

set -e  # Exit on error

UNLABELED_FILE=${1:-data/external/unlabeled.txt}
NUM_EPOCHS=${2:-5}

python -m src.pretrain_lm \
  --model_name microsoft/deberta-v3-large \
  --data_file "$UNLABELED_FILE" \
  --output_dir outputs/dapt_checkpoints/ \
  --num_train_epochs "$NUM_EPOCHS" \
  --batch_size 8 \
  --learning_rate 5e-5 \
  --logging_steps 100 \
  --save_steps 500 \
  --save_total_limit 2 \
  --use_fp16
