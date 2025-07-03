#!/usr/bin/env bash
# Run entire pipeline end‑to‑end
set -e

bash scripts/run_preprocess.sh
bash scripts/run_pretrain_dapt.sh data/external/unlabeled.txt 5
bash scripts/run_train_deberta.sh
bash scripts/run_train_longformer.sh
bash scripts/run_pseudo_label.sh
bash scripts/run_ensemble.sh
bash scripts/run_stack.sh
bash scripts/run_cv.sh
