import os
import argparse
import logging
from typing import Dict, Tuple

import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import prepare_model_for_int8_training

from .deberta_lora import get_deberta_lora_model

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset_splits(data_dir: str, train_split: str, val_split: str) -> DatasetDict:
    ds = {
        "train": load_from_disk(os.path.join(data_dir, train_split)),
        "val": load_from_disk(os.path.join(data_dir, val_split)),
    }
    logger.info("Loaded splits: %s (%d) | %s (%d)",
                train_split, len(ds["train"]), val_split, len(ds["val"]))
    return ds


def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main():
    parser = argparse.ArgumentParser("LoRA fine‑tune DeBERTa‑v3 on AG News")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--data_dir", type=str, default="data/interim")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="outputs/checkpoints/deberta_lora")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    args = parser.parse_args()

    # Load data
    dataset = load_dataset_splits(args.data_dir, args.train_split, args.val_split)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Load LoRA‑wrapped model
    model = get_deberta_lora_model(base_model=args.model_name)
    model = prepare_model_for_int8_training(model)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_dir=os.path.join(args.output_dir, "logs"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved model and tokenizer to %s", args.output_dir)


if __name__ == "__main__":
    main()
