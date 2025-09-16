#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Minimal Example for AG News Text Classification
================================================

This script provides a minimal working example of the AG News classification
framework, demonstrating basic usage for quick experimentation.

Following pedagogical principles from:
- Goodfellow et al. (2016): "Deep Learning" - Chapter 1: Introduction
- Murphy (2012): "Machine Learning: A Probabilistic Perspective" - Quick Start Examples

Author: Võ Hải Dũng
License: MIT
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Simple configuration
CONFIG = {
    "model_name": "bert-base-uncased",  # Use smaller model for quick demo
    "max_length": 128,  # Shorter sequences for speed
    "batch_size": 32,
    "num_epochs": 2,
    "learning_rate": 2e-5,
    "output_dir": "./quickstart_output",
}

def load_sample_data(num_samples: int = 1000):
    """
    Load a sample of AG News data for quick testing.
    
    Args:
        num_samples: Number of samples to load
        
    Returns:
        DatasetDict with train and test splits
    """
    print(f"Loading {num_samples} samples from AG News...")
    
    # Load processed data if available
    processed_dir = PROJECT_ROOT / "data" / "processed"
    
    if not (processed_dir / "train.csv").exists():
        print("Processed data not found. Please run:")
        print("  python scripts/data_preparation/prepare_ag_news.py")
        sys.exit(1)
    
    # Load data
    train_df = pd.read_csv(processed_dir / "train.csv").head(num_samples)
    test_df = pd.read_csv(processed_dir / "test.csv").head(num_samples // 5)
    
    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
    
    return DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

def tokenize_data(dataset, tokenizer):
    """
    Tokenize dataset.
    
    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        
    Returns:
        Tokenized dataset
    """
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=CONFIG["max_length"]
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    return tokenized_dataset

def compute_metrics(eval_pred):
    """
    Compute evaluation metrics.
    
    Args:
        eval_pred: Evaluation predictions
        
    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro")
    }

def main():
    """Main execution function."""
    print("=" * 80)
    print("AG News Text Classification - Minimal Example")
    print("=" * 80)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    dataset = load_sample_data(num_samples=1000)
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Test samples: {len(dataset['test'])}")
    
    # Load model and tokenizer
    print(f"\nLoading model: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=4
    )
    
    # Tokenize data
    print("Tokenizing data...")
    tokenized_dataset = tokenize_data(dataset, tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_epochs"],
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        warmup_steps=100,
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard for minimal example
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\nStarting training...")
    print("-" * 40)
    trainer.train()
    
    # Evaluate
    print("\nEvaluating on test set...")
    results = trainer.evaluate()
    
    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Test Accuracy: {results['eval_accuracy']:.4f}")
    print(f"Test F1-Macro: {results['eval_f1_macro']:.4f}")
    print(f"Loss: {results['eval_loss']:.4f}")
    
    # Save model
    print(f"\nSaving model to {CONFIG['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(CONFIG["output_dir"])
    
    # Example prediction
    print("\n" + "=" * 80)
    print("EXAMPLE PREDICTIONS")
    print("=" * 80)
    
    sample_texts = [
        "The stock market reached record highs today as investors celebrated strong earnings.",
        "The Lakers defeated the Celtics in overtime with a stunning three-pointer.",
        "Scientists discover new exoplanet that could potentially harbor life.",
        "World leaders gather for climate summit to discuss carbon reduction goals."
    ]
    
    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    
    for text in sample_texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=CONFIG["max_length"])
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = predictions[0][predicted_class].item()
        
        print(f"\nText: {text[:80]}...")
        print(f"Predicted: {label_names[predicted_class]} (confidence: {confidence:.3f})")
    
    print("\n" + "=" * 80)
    print("Minimal example completed successfully!")
    print("For full training, see quickstart/train_simple.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
