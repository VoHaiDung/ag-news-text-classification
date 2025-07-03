#!/usr/bin/env bash
# Prepare tokenized AG News dataset
set -e
python src/data_utils.py \
  --model_name microsoft/deberta-v3-large \
  --max_length 512 \
  --stride 256 \
  --output_dir data/interim/
