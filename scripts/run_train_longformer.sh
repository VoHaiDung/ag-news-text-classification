#!/usr/bin/env bash
# Fine‑tune Longformer‑LoRA on AG News
set -e

python src/train_longformer.py \
  --seed 42 \
  --model_name allenai/longformer-large-4096 \
  --max_length 4096 \
  --lr 2e-5 \
  --batch_size 4 \
  --epochs 5 \
  --weight_decay 0.01
