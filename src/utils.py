import os
import logging
import numpy as np
from typing import List, Tuple, Dict

import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)


# Create a logger that writes to console and optional file
def configure_logger(log_path: str = None) -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
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


# Compute accuracy, precision, recall, and F1 (macro)
def compute_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    preds = logits.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# Print detailed classification report per class
def print_classification_report(
    logits: np.ndarray,
    labels: np.ndarray,
    target_names: Tuple[str, ...]
) -> None:
    preds = logits.argmax(axis=1)
    report = classification_report(labels, preds, target_names=target_names, zero_division=0)
    print("Classification Report:\n", report)


# Create DataLoader from HuggingFace Dataset
def prepare_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:
    def collate_fn(batch):
        return {
            "input_ids": torch.tensor([ex["input_ids"] for ex in batch], dtype=torch.long),
            "attention_mask": torch.tensor([ex["attention_mask"] for ex in batch], dtype=torch.long),
            "labels": torch.tensor([ex["labels"] for ex in batch], dtype=torch.long),
        }

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


# Load logits from multiple models and labels
def load_logits_and_labels(logits_paths: List[str], labels_path: str) -> Tuple[np.ndarray, np.ndarray]:
    logits_list = [np.load(path) for path in logits_paths]
    X = np.concatenate(logits_list, axis=1)
    y = np.load(labels_path)
    return X, y