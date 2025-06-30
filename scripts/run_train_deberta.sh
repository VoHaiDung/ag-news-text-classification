#!/usr/bin/env bash
# Fine-tune DeBERTa-v3-large (with optional LoRA + DAPT)
# Usage: bash scripts/run_train_deberta.sh <TRAIN_FILE> <VAL_FILE> [DAPT_MODEL_PATH]

set -e

TRAIN_FILE=${1:-data/processed/train.json}
VAL_FILE=${2:-data/processed/val.json}
DAPT_MODEL=${3:-microsoft/deberta-v3-large}

python -m src.train \
  --model_name_or_path "$DAPT_MODEL" \
  --train_file "$TRAIN_FILE" \
  --validation_file "$VAL_FILE" \
  --output_dir outputs/checkpoints/deberta_lora/ \
  --max_length 512 \
  --batch_size 8 \
  --num_train_epochs 4 \
  --learning_rate 2e-5 \
  --use_lora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.1 \
  --logging_steps 100 \
  --eval_steps 250 \
  --save_steps 500 \
  --save_total_limit 2 \
  --metric_for_best_model accuracy \
  --load_best_model_at_end \
  --use_fp16
