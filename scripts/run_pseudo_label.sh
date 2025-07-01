#!/usr/bin/env bash
set -euo pipefail

# Generate pseudo-labels using ensemble of DeBERTa and Longformer
# Usage: bash scripts/run_pseudo_label.sh [THRESHOLD]
THRESHOLD=${1:-0.90}

python src/pseudo_label.py \
  --unlabeled_csv data/raw/unlabeled_news.csv \
  --deberta_dir results/deberta_lora \
  --longformer_dir results/longformer_lora \
  --output_csv data/processed/pseudo_labeled.csv \
  --prob_threshold "${THRESHOLD}"
