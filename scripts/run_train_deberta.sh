#!/usr/bin/env bash
# Fine‑tune DeBERTa‑LoRA on AG News
set -e

python src/train.py \
  --seed 42 \
  --model_name microsoft/deberta-v3-large \
  --longformer_name allenai/longformer-large-4096 \
  --max_length 512 \
  --stride 256 \
  --lr 2e-5 \
  --batch_size 16 \
  --epochs 5 \
  --weight_decay 0.01
