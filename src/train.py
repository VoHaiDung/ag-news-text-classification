import os
import logging
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk, DatasetDict
from peft import prepare_model_for_int8_training
from typing import Dict, Tuple

from src.deberta_lora import get_deberta_lora_model

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tokenized_dataset(data_dir: str = "data/interim/") -> DatasetDict:
    """
    Load tokenized AG News dataset from disk.

    Args:
        data_dir (str): Path to the directory containing tokenized splits.

    Returns:
        DatasetDict: Dictionary with 'train' and 'test' datasets.
    """
    dataset = {
        split: load_from_disk(os.path.join(data_dir, split))
        for split in ["train", "test"]
    }
    logger.info("Loaded tokenized datasets from %s", data_dir)
    return dataset


def compute_metrics(eval_pred: Tuple) -> Dict[str, float]:
    """
    Compute classification metrics.

    Args:
        eval_pred (Tuple): A tuple of (logits, labels)

    Returns:
        Dict[str, float]: Computed accuracy, precision, recall, and F1-score.
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    predictions = logits.argmax(axis=1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def train_model(
    model_name: str = "microsoft/deberta-v3-large",
    output_dir: str = "outputs/checkpoints/deberta/",
    data_dir: str = "data/interim/"
) -> None:
    """
    Fine-tune a DeBERTa-v3 model with LoRA on the AG News dataset.

    Args:
        model_name (str): Pretrained model identifier.
        output_dir (str): Output directory for saving model and tokenizer.
        data_dir (str): Input directory of tokenized dataset.
    """
    logger.info("Initializing training pipeline for %s", model_name)

    # Load data and tokenizer
    dataset = load_tokenized_dataset(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load and prepare model
    model = get_deberta_lora_model()
    model = prepare_model_for_int8_training(model)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=1,
        report_to="none",
        logging_dir=os.path.join(output_dir, "logs")
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved to %s", output_dir)


if __name__ == "__main__":
    train_model()
