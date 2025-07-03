#!/usr/bin/env bash
# Train and save stacking classifier
set -e

python src/stacking.py \
  --logits results/logits_deberta.npy results/logits_longformer.npy \
  --labels results/labels.npy \
  --save_model outputs/checkpoints/stacking_model.joblib \
  --class_names World Sports Business Sci/Tech \
  --cv 5
