import os
import argparse
import logging
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification

from src.utils import configure_logger, prepare_dataloader, compute_metrics, print_classification_report

# Initialize logger
logger = configure_logger("results/logs/evaluate.log")

# Inference function for evaluation
# model_dir: path to pretrained model
def load_and_evaluate(
    model_dir: str,
    data_dir: str,
    split: str,
    batch_size: int,
    class_names: tuple
) -> None:
    logger.info(f"Loading dataset split '{split}' from {data_dir}...")
    ds = load_from_disk(os.path.join(data_dir, split))
    dataloader = prepare_dataloader(ds, batch_size)

    logger.info(f"Loading model from {model_dir}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    logger.info("Starting inference...")
    all_logits, all_labels = [], []
    use_amp = device.type == 'cuda'
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = outputs.logits.cpu().numpy()
            all_logits.append(logits)
            all_labels.append(labels)

    logits = np.vstack(all_logits)
    labels = np.concatenate(all_labels)

    metrics = compute_metrics(logits, labels)
    logger.info("Evaluation metrics: %s", metrics)

    print_classification_report(logits, labels, class_names)

# CLI entry point
def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned classification model on a dataset split")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved model directory")
    parser.add_argument("--data_dir", type=str, default="data/interim", help="Path to tokenized dataset directory")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (train/val/test)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--class_names", nargs="+", default=["World","Sports","Business","Sci/Tech"], help="List of class names for report")
    args = parser.parse_args()

    load_and_evaluate(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        class_names=tuple(args.class_names)
    )

if __name__ == '__main__':
    main()
