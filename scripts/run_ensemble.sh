#!/usr/bin/env bash
# Soft‑voting ensemble inference on test split
set -e

python src/ensemble.py \
  --deberta_dir results/deberta_lora \
  --longformer_dir results/longformer_lora \
  --data_dir data/interim \
  --batch_size 8 \
  --weight_deberta 0.5 \
  --weight_longformer 0.5 \
  --class_names World Sports Business Sci/Tech
