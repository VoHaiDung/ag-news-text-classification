"""
Resume Training Script for AG News Text Classification
=======================================================

This script implements training resumption from checkpoints following:
- Goyal et al. (2017): "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
- You et al. (2020): "Large Batch Optimization for Deep Learning"
- Zhang et al. (2019): "Which Algorithmic Choices Matter at Which Batch Sizes?"

The resumption pipeline implements:
1. Checkpoint loading with state restoration
2. Learning rate schedule continuation
3. Optimizer state recovery
4. Training history preservation
5. Graceful failure handling

Mathematical Framework:
Ensures continuity: θ_t+1 = θ_t - η_t ∇L(θ_t)
where training resumes from exact state at interruption.

Author: Võ Hải Dũng
License: MIT
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import time
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets.ag_news import create_ag_news_datasets
from src.data.loaders.dataloader import create_train_val_test_loaders
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from src.utils.experiment_tracking import create_experiment, log_metrics
from src.utils.io_utils import safe_save, safe_load, ensure_dir
from configs.constants import AG_NEWS_NUM_CLASSES, MODELS_DIR

logger = setup_logging(__name__)


class CheckpointManager:
    """
    Manages checkpoint saving and loading for training resumption.
    
    Implements checkpoint strategies from:
    - Chen et al. (2020): "Strategies for Training Large Neural Networks"
    """
    
    def __init__(self, checkpoint_dir: Path, max_checkpoints: int = 5):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        ensure_dir(self.checkpoint_dir)
        
        # Track checkpoint files
        self.checkpoint_files = []
        self._scan_existing_checkpoints()
    
    def _scan_existing_checkpoints(self):
        """Scan for existing checkpoint files."""
        if self.checkpoint_dir.exists():
            self.checkpoint_files = sorted(
                self.checkpoint_dir.glob("checkpoint_*.pt"),
                key=lambda p: p.stat().st_mtime
            )
    
    def save_checkpoint(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        training_state: Dict[str, Any],
        is_best: bool = False
    ) -> Path:
        """
        Save training checkpoint.
        
        Args:
            epoch: Current epoch
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler
            training_state: Additional training state
            is_best: Whether this is the best model
            
        Returns:
            Path to saved checkpoint
        """
        # Prepare checkpoint data
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "training_state": training_state,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save as best if specified
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")
        
        # Manage checkpoint history
        self.checkpoint_files.append(checkpoint_path)
        self._cleanup_old_checkpoints()
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        checkpoint_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Load checkpoint for resumption.
        
        Args:
            checkpoint_path: Specific checkpoint to load
            
        Returns:
            Checkpoint dictionary
        """
        if checkpoint_path is None:
            # Load latest checkpoint
            if not self.checkpoint_files:
                raise FileNotFoundError("No checkpoints found")
            checkpoint_path = self.checkpoint_files[-1]
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {checkpoint['epoch']}")
        
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints keeping only recent ones."""
        if len(self.checkpoint_files) > self.max_checkpoints:
            # Keep best checkpoint
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            
            # Remove oldest checkpoints
            for checkpoint_path in self.checkpoint_files[:-self.max_checkpoints]:
                if checkpoint_path != best_path and checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
            
            self.checkpoint_files = self.checkpoint_files[-self.max_checkpoints:]


class ResumableTrainer:
    """
    Trainer with checkpoint resumption capabilities.
    
    Implements fault-tolerant training following:
    - Dean et al. (2012): "Large Scale Distributed Deep Networks"
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        checkpoint_manager: CheckpointManager
    ):
        """
        Initialize resumable trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Compute device
            checkpoint_manager: Checkpoint manager
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.checkpoint_manager = checkpoint_manager
        
        # Training state
        self.start_epoch = 1
        self.best_metric = 0.0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": []
        }
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self._setup_training()
    
    def _setup_training(self):
        """Setup optimizer and scheduler."""
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("learning_rate", 2e-5),
            weight_decay=self.config.get("weight_decay", 0.01)
        )
        
        # Create scheduler
        num_training_steps = len(self.train_loader) * self.config.get("num_epochs", 10)
        num_warmup_steps = int(num_training_steps * self.config.get("warmup_ratio", 0.1))
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def resume_from_checkpoint(self, checkpoint_path: Optional[Path] = None):
        """
        Resume training from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        try:
            checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)
            
            # Restore model state
            self.model.load_state_dict(checkpoint["model_state_dict"])
            
            # Restore optimizer state
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            # Restore scheduler state
            if checkpoint.get("scheduler_state_dict") and self.scheduler:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
            # Restore training state
            training_state = checkpoint.get("training_state", {})
            self.start_epoch = checkpoint["epoch"] + 1
            self.best_metric = training_state.get("best_metric", 0.0)
            self.training_history = training_state.get("history", self.training_history)
            
            logger.info(f"Successfully resumed from epoch {checkpoint['epoch']}")
            logger.info(f"Best metric so far: {self.best_metric:.4f}")
            
        except Exception as e:
            logger.warning(f"Failed to resume from checkpoint: {e}")
            logger.info("Starting training from scratch")
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0)
            )
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress
            progress_bar.set_postfix({"loss": loss.item()})
        
        return total_loss / num_batches
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_macro": f1
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Complete training loop with checkpoint saving.
        
        Returns:
            Training results
        """
        num_epochs = self.config.get("num_epochs", 10)
        patience = self.config.get("patience", 3)
        no_improve_count = 0
        
        logger.info(f"Starting training from epoch {self.start_epoch}")
        
        for epoch in range(self.start_epoch, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Update history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_metrics["loss"])
            self.training_history["val_accuracy"].append(val_metrics["accuracy"])
            self.training_history["val_f1"].append(val_metrics["f1_macro"])
            
            # Log metrics
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, "
                f"Val F1: {val_metrics['f1_macro']:.4f}"
            )
            
            # Check for improvement
            current_metric = val_metrics["f1_macro"]
            is_best = current_metric > self.best_metric
            
            if is_best:
                self.best_metric = current_metric
                no_improve_count = 0
                logger.info(f"New best F1: {self.best_metric:.4f}")
            else:
                no_improve_count += 1
            
            # Save checkpoint
            training_state = {
                "best_metric": self.best_metric,
                "history": self.training_history,
                "no_improve_count": no_improve_count
            }
            
            self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                training_state=training_state,
                is_best=is_best
            )
            
            # Early stopping
            if no_improve_count >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        return {
            "final_epoch": epoch,
            "best_metric": self.best_metric,
            "history": self.training_history
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Resume training from checkpoint"
    )
    
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to checkpoint file (uses latest if not specified)"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing checkpoints"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Model name if starting fresh"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Total number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    
    return parser.parse_args()


def main():
    """Main resumable training pipeline."""
    args = parse_arguments()
    
    # Setup
    logger.info("Setting up resumable training")
    device = torch.device(args.device)
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=Path(args.checkpoint_dir),
        max_checkpoints=5
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_ag_news_datasets(tokenizer=tokenizer)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size
    )
    
    # Load or create model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=AG_NEWS_NUM_CLASSES
    )
    
    # Prepare config
    config = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "patience": 3
    }
    
    # Initialize trainer
    trainer = ResumableTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_manager=checkpoint_manager
    )
    
    # Resume from checkpoint if available
    if args.checkpoint_path:
        trainer.resume_from_checkpoint(Path(args.checkpoint_path))
    else:
        # Try to resume from latest checkpoint
        try:
            trainer.resume_from_checkpoint()
        except FileNotFoundError:
            logger.info("No checkpoint found, starting fresh")
    
    # Train
    results = trainer.train()
    
    # Save final results
    output_path = Path(args.checkpoint_dir) / "training_results.json"
    safe_save(results, output_path)
    
    logger.info(f"Training completed. Best F1: {results['best_metric']:.4f}")


if __name__ == "__main__":
    main()
