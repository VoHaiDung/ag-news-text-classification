import os
from transformers import Trainer, TrainingArguments, AutoTokenizer
from datasets import load_from_disk
from peft import prepare_model_for_int8_training

from src.deberta_lora import get_deberta_lora_model


def load_tokenized_dataset(data_dir="data/interim/"):
    """
    Load tokenized AG News dataset from disk.

    Args:
        data_dir (str): Path to the directory containing tokenized splits.

    Returns:
        DatasetDict: Dictionary with train/test splits in Hugging Face format.
    """
    dataset = {
        split: load_from_disk(os.path.join(data_dir, split))
        for split in ["train", "test"]
    }
    return dataset


def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1 score.

    Args:
        eval_pred: Tuple of (logits, labels)

    Returns:
        Dict of metric names and values.
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
    model_name="microsoft/deberta-v3-large",
    output_dir="outputs/checkpoints/deberta/",
    data_dir="data/interim/"
):
    """
    Fine-tune DeBERTa-v3 model with LoRA on AG News dataset.

    Args:
        model_name (str): Name of the pretrained model.
        output_dir (str): Path to save trained model.
        data_dir (str): Path to tokenized dataset directory.
    """
    # Load tokenized dataset
    dataset = load_tokenized_dataset(data_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with LoRA
    model = get_deberta_lora_model()
    model = prepare_model_for_int8_training(model)  # Optional: memory-efficient for large models

    # Training arguments
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
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    # Save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model and tokenizer saved to {output_dir}")


# Optional command-line interface
if __name__ == "__main__":
    train_model()
