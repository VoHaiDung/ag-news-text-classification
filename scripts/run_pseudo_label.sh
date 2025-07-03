#!/usr/bin/env bash
# Generate pseudo‑labels with ensemble
set -e

python src/pseudo_label.py \
  --unlabeled_csv data/raw/unlabeled_news.csv \
  --deberta_dir results/deberta_lora \
  --longformer_dir results/longformer_lora \
  --output_csv data/processed/pseudo_labeled.csv \
  --prob_threshold 0.90
