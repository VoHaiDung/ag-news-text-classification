#!/usr/bin/env bash
# K‑fold cross‑validation
set -e

python src/cv_run.py \
  --folds 5 \
  --seed 42 \
  --model_name microsoft/deberta-v3-large \
  --longformer_name allenai/longformer-large-4096 \
  --max_length 512 \
  --stride 256 \
  --batch_size 8 \
  --epochs 3 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --output_dir results/cv
