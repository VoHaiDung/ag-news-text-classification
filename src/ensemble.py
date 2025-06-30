import os
import argparse
import logging
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_from_disk, Dataset
from transformers import AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
from tqdm.auto import tqdm


def configure_logger(log_path: str = None) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = configure_logger("outputs/logs/ensemble.log")


def load_model(model_dir: str, device: torch.device) -> torch.nn.Module:
    logger.info(f"Loading model from {model_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    return model


def prepare_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    def collate_fn(batch):
        return {
            'input_ids': torch.tensor([ex['input_ids'] for ex in batch], dtype=torch.long),
            'attention_mask': torch.tensor([ex['attention_mask'] for ex in batch], dtype=torch.long),
            'labels': torch.tensor([ex['labels'] for ex in batch], dtype=torch.long),
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


def inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    all_logits, all_labels = [], []
    amp_ctx = torch.cuda.amp.autocast if device.type == 'cuda' else torch.no_grad
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].cpu().numpy()

            with amp_ctx():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.cpu().numpy()

            all_logits.append(logits)
            all_labels.append(labels)

    return np.vstack(all_logits), np.concatenate(all_labels)


def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='macro', zero_division=0
    )
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def print_classification_report(
    logits: np.ndarray,
    labels: np.ndarray,
    target_names: Tuple[str, ...]
) -> None:
    preds = logits.argmax(axis=1)
    report = classification_report(labels, preds, target_names=target_names, zero_division=0)
    logger.info("Detailed Classification Report:\n%s", report)


def main():
    parser = argparse.ArgumentParser(
        description="Ensemble DeBERTa & Longformer via weighted soft-voting"
    )
    parser.add_argument("--deberta_dir", type=str, required=True)
    parser.add_argument("--longformer_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/interim")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--weight_deberta", type=float, default=0.5)
    parser.add_argument("--weight_longformer", type=float, default=0.5)
    parser.add_argument("--save_logits", type=str, default="")
    parser.add_argument("--class_names", nargs="+",
                        default=["World", "Sports", "Business", "Sci/Tech"])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    # Load test dataset
    ds_test = load_from_disk(os.path.join(args.data_dir, "test"))
    logger.info("Loaded test dataset with %d samples", len(ds_test))
    dataloader = prepare_dataloader(ds_test, args.batch_size)

    # Load both fine-tuned models
    model_deberta = load_model(args.deberta_dir, device)
    model_longformer = load_model(args.longformer_dir, device)

    # Inference for both models
    logger.info("Running inference with DeBERTa...")
    logits_deberta, labels = inference(model_deberta, dataloader, device)

    logger.info("Running inference with Longformer...")
    logits_longformer, _ = inference(model_longformer, dataloader, device)

    # Soft-voting ensemble of logits
    logger.info("Ensembling outputs (weights: DeBERTa=%.2f, Longformer=%.2f)",
                args.weight_deberta, args.weight_longformer)
    logits_ensemble = (
        args.weight_deberta * logits_deberta +
        args.weight_longformer * logits_longformer
    )

    # Compute and log evaluation metrics
    metrics = compute_metrics(logits_ensemble, labels)
    logger.info("Ensemble Metrics:")
    for k, v in metrics.items():
        logger.info("  %s: %.4f", k, v)

    print_classification_report(logits_ensemble, labels, tuple(args.class_names))

    # Save logits and labels if required
    if args.save_logits:
        os.makedirs(args.save_logits, exist_ok=True)
        np.save(os.path.join(args.save_logits, "logits_deberta.npy"), logits_deberta)
        np.save(os.path.join(args.save_logits, "logits_longformer.npy"), logits_longformer)
        np.save(os.path.join(args.save_logits, "logits_ensemble.npy"), logits_ensemble)
        np.save(os.path.join(args.save_logits, "labels.npy"), labels)
        logger.info("Saved logits and labels to %s", args.save_logits)


if __name__ == "__main__":
    main()
