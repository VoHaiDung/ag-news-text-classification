#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Training Script for AG News Classification
==================================================

This script provides a simplified training pipeline for AG News classification,
suitable for beginners and quick experiments.

Following educational approach from:
- Howard & Gugger (2020): "Deep Learning for Coders with fastai and PyTorch"
- Chollet (2021): "Deep Learning with Python" - Progressive Disclosure

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report

from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from configs.constants import AG_NEWS_CLASSES

# Setup logging
logger = setup_logging(name=__name__)

class SimpleConfig:
    """Simple configuration for training."""
    
    def __init__(self, **kwargs):
        # Model settings
        self.model_name = kwargs.get("model_name", "distilbert-base-uncased")
        self.num_labels = 4
        
        # Data settings
        self.max_length = kwargs.get("max_length", 256)
        self.train_batch_size = kwargs.get("batch_size", 32)
        self.eval_batch_size = kwargs.get("eval_batch_size", 64)
        
        # Training settings
        self.num_epochs = kwargs.get("num_epochs", 3)
        self.learning_rate = kwargs.get("learning_rate", 2e-5)
        self.warmup_ratio = kwargs.get("warmup_ratio", 0.1)
        self.weight_decay = kwargs.get("weight_decay", 0.01)
        
        # Other settings
        self.seed = kwargs.get("seed", 42)
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = kwargs.get("output_dir", f"./outputs/simple_{datetime.now():%Y%m%d_%H%M%S}")
        self.save_model = kwargs.get("save_model", True)
        self.use_fp16 = kwargs.get("fp16", False) and torch.cuda.is_available()

class AGNewsDataset(torch.utils.data.Dataset):
    """Simple PyTorch dataset for AG News."""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

class SimpleTrainer:
    """
    Simple trainer for AG News classification.
    
    Implements basic training loop following:
    - Goodfellow et al. (2016): "Deep Learning" - Training Deep Networks
    """
    
    def __init__(self, model, config, train_dataloader, eval_dataloader=None):
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Move model to device
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup mixed precision if requested
        self.scaler = torch.cuda.amp.GradScaler() if config.use_fp16 else None
        
        # Tracking
        self.train_losses = []
        self.eval_results = []
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
    
    def _setup_scheduler(self):
        """Setup linear scheduler with warmup."""
        total_steps = len(self.train_dataloader) * self.config.num_epochs
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"  Num epochs: {self.config.num_epochs}")
        logger.info(f"  Batch size: {self.config.train_batch_size}")
        logger.info(f"  Total optimization steps: {len(self.train_dataloader) * self.config.num_epochs}")
        
        for epoch in range(self.config.num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            logger.info(f"  Average training loss: {train_loss:.4f}")
            
            # Evaluation
            if self.eval_dataloader:
                eval_results = self.evaluate()
                self.eval_results.append(eval_results)
                logger.info(f"  Evaluation - Accuracy: {eval_results['accuracy']:.4f}, "
                           f"F1-Macro: {eval_results['f1_macro']:.4f}")
        
        # Save model
        if self.config.save_model:
            self.save_model()
        
        logger.info("Training completed!")
    
    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            if self.config.use_fp16:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
            
            # Backward pass
            if self.config.use_fp16:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})
        
        return total_loss / len(self.train_dataloader)
    
    def evaluate(self):
        """Evaluate model on validation set."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                total_loss += loss.item()
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_macro = f1_score(all_labels, all_predictions, average="macro")
        
        return {
            "loss": total_loss / len(self.eval_dataloader),
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "predictions": all_predictions,
            "labels": all_labels
        }
    
    def save_model(self):
        """Save trained model."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(output_dir)
        logger.info(f"Model saved to {output_dir}")
        
        # Save training history
        history = {
            "train_losses": self.train_losses,
            "eval_results": self.eval_results,
            "config": vars(self.config)
        }
        
        with open(output_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2, default=str)

def load_data(data_dir: Path):
    """Load processed AG News data."""
    logger.info(f"Loading data from {data_dir}")
    
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "validation.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    logger.info(f"  Train: {len(train_df)} samples")
    logger.info(f"  Validation: {len(val_df)} samples")
    logger.info(f"  Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Simple training script for AG News classification"
    )
    
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Model name from Hugging Face Hub"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory containing processed data"
    )
    
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for model"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "outputs" / "simple" / f"{datetime.now():%Y%m%d_%H%M%S}"
    
    # Ensure reproducibility
    ensure_reproducibility(seed=args.seed)
    
    # Create config
    config = SimpleConfig(
        model_name=args.model_name,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        fp16=args.fp16,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    # Load data
    train_df, val_df, test_df = load_data(args.data_dir)
    
    # Initialize tokenizer and model
    logger.info(f"Loading model: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels
    )
    
    # Create datasets
    train_dataset = AGNewsDataset(
        train_df["text"].values,
        train_df["label"].values,
        tokenizer,
        config.max_length
    )
    
    val_dataset = AGNewsDataset(
        val_df["text"].values,
        val_df["label"].values,
        tokenizer,
        config.max_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=4
    )
    
    eval_dataloader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Create trainer and train
    trainer = SimpleTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader
    )
    
    trainer.train()
    
    # Final evaluation on test set
    logger.info("\nFinal evaluation on test set:")
    
    test_dataset = AGNewsDataset(
        test_df["text"].values,
        test_df["label"].values,
        tokenizer,
        config.max_length
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=4
    )
    
    trainer.eval_dataloader = test_dataloader
    test_results = trainer.evaluate()
    
    logger.info(f"Test Accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Test F1-Macro: {test_results['f1_macro']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(
        test_results["labels"],
        test_results["predictions"],
        target_names=AG_NEWS_CLASSES
    ))
    
    # Save tokenizer
    tokenizer.save_pretrained(config.output_dir)
    
    logger.info(f"\nTraining complete! Model saved to {config.output_dir}")

if __name__ == "__main__":
    main()
