"""
Single Model Training Script for AG News Text Classification
=============================================================

This script implements comprehensive training pipeline for single models following
methodologies from:
- Goodfellow et al. (2016): "Deep Learning" - Training procedures
- Smith (2017): "A disciplined approach to neural network hyper-parameters"
- Dodge et al. (2019): "Show Your Work: Improved Reporting of Experimental Results"

The training pipeline implements:
1. Data loading with stratified sampling
2. Model initialization with configuration management
3. Training loop with gradient accumulation
4. Validation with early stopping
5. Experiment tracking and reproducibility

Mathematical Framework:
The optimization objective follows:
    θ* = argmin_θ L(θ) + λR(θ)
where L is the task loss, R is regularization, and λ is the regularization weight.

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.data.datasets.ag_news import AGNewsDataset, AGNewsConfig, create_ag_news_datasets
from src.data.loaders.dataloader import create_train_val_test_loaders
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility, create_reproducibility_report
from src.utils.experiment_tracking import create_experiment, log_hyperparameters, log_metrics, log_model
from src.utils.io_utils import safe_save, ensure_dir
from configs.config_loader import load_training_config, load_model_config
from configs.constants import (
    AG_NEWS_NUM_CLASSES,
    MAX_SEQUENCE_LENGTH,
    MODELS_DIR,
    LOGS_DIR,
    AG_NEWS_CLASSES
)

# Setup logging
logger = setup_logging(__name__)


class ModelTrainer:
    """
    Comprehensive model trainer implementing best practices from:
    - Howard & Ruder (2018): "Universal Language Model Fine-tuning"
    - Mosbach et al. (2021): "On the Stability of Fine-tuning BERT"
    
    The trainer implements:
    1. Mixed precision training (Micikevicius et al., 2018)
    2. Gradient accumulation for effective batch size scaling
    3. Learning rate scheduling with warmup
    4. Early stopping with patience
    5. Model checkpointing with best model selection
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        experiment_name: str
    ):
        """
        Initialize trainer with model and data.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Compute device
            experiment_name: Name for experiment tracking
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        
        # Move model to device
        self.model.to(device)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.get("fp16", False) else None
        
        # Tracking
        self.best_val_metric = 0.0
        self.best_epoch = 0
        self.patience_counter = 0
        self.training_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": [],
            "learning_rates": []
        }
        
        logger.info(f"Initialized trainer for {experiment_name}")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """
        Create optimizer with layer-wise learning rate decay.
        
        Implements differential learning rates following:
        - Howard & Ruder (2018): "Universal Language Model Fine-tuning"
        """
        learning_rate = self.config.get("learning_rate", 2e-5)
        weight_decay = self.config.get("weight_decay", 0.01)
        
        # Parameter groups with weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer_name = self.config.get("optimizer", "adamw").lower()
        
        if optimizer_name == "adamw":
            from transformers import AdamW
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(self.config.get("adam_beta1", 0.9), 
                       self.config.get("adam_beta2", 0.999)),
                eps=self.config.get("adam_epsilon", 1e-8)
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                optimizer_grouped_parameters,
                lr=learning_rate,
                momentum=self.config.get("momentum", 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self):
        """
        Create learning rate scheduler with warmup.
        
        Implements warmup scheduling following:
        - Vaswani et al. (2017): "Attention is All You Need"
        """
        num_training_steps = len(self.train_loader) * self.config.get("num_epochs", 10)
        num_warmup_steps = int(num_training_steps * self.config.get("warmup_ratio", 0.1))
        
        scheduler_type = self.config.get("scheduler", "linear").lower()
        
        if scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Implements training loop with gradient accumulation following:
        - Chen et al. (2016): "Training Deep Nets with Sublinear Memory Cost"
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        total_steps = 0
        
        gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 1)
        max_grad_norm = self.config.get("max_grad_norm", 1.0)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Mixed precision training
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
            else:
                # Standard training
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
            
            total_loss += loss.item() * gradient_accumulation_steps
            total_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
        
        return total_loss / total_steps
    
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation set.
        
        Computes comprehensive metrics following:
        - Sokolova & Lapalme (2009): "A systematic analysis of performance measures"
        
        Returns:
            Tuple of (validation loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
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
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="macro"
        )
        
        metrics = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1
        }
        
        return avg_loss, metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Complete training loop with early stopping.
        
        Implements training procedure following:
        - Prechelt (1998): "Early Stopping - But When?"
        
        Returns:
            Training results and history
        """
        num_epochs = self.config.get("num_epochs", 10)
        patience = self.config.get("early_stopping_patience", 3)
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.evaluate()
            
            # Log metrics
            current_lr = self.optimizer.param_groups[0]["lr"]
            log_metrics(
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    "learning_rate": current_lr
                },
                step=epoch,
                epoch=epoch
            )
            
            # Update history
            self.training_history["train_loss"].append(train_loss)
            self.training_history["val_loss"].append(val_loss)
            self.training_history["val_accuracy"].append(val_metrics["accuracy"])
            self.training_history["val_f1"].append(val_metrics["f1_macro"])
            self.training_history["learning_rates"].append(current_lr)
            
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}"
            )
            
            # Check for improvement
            metric_for_best = val_metrics.get(
                self.config.get("metric_for_best_model", "f1_macro"),
                val_metrics["f1_macro"]
            )
            
            if metric_for_best > self.best_val_metric:
                self.best_val_metric = metric_for_best
                self.best_epoch = epoch
                self.patience_counter = 0
                
                # Save best model
                self.save_model(is_best=True)
                logger.info(f"New best model saved with {self.config.get('metric_for_best_model', 'f1_macro')}: {metric_for_best:.4f}")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs")
                    break
        
        # Final save
        self.save_model(is_best=False, is_final=True)
        
        return {
            "best_epoch": self.best_epoch,
            "best_metric": self.best_val_metric,
            "final_epoch": epoch,
            "history": self.training_history
        }
    
    def save_model(self, is_best: bool = False, is_final: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model
            is_final: Whether this is the final model
        """
        save_dir = Path(self.config.get("output_dir", MODELS_DIR)) / self.experiment_name
        ensure_dir(save_dir)
        
        if is_best:
            save_path = save_dir / "best_model"
        elif is_final:
            save_path = save_dir / "final_model"
        else:
            save_path = save_dir / f"checkpoint_epoch_{self.best_epoch}"
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save training state
        state = {
            "config": self.config,
            "best_epoch": self.best_epoch,
            "best_metric": self.best_val_metric,
            "history": self.training_history
        }
        safe_save(state, save_path / "training_state.json")
        
        # Log model to experiment tracker
        log_model(
            self.model,
            model_name=self.experiment_name,
            epoch=self.best_epoch,
            metrics={"best_metric": self.best_val_metric}
        )


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train a single model on AG News dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="roberta-base",
        help="Name or path of pretrained model"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/models/single/roberta_large.yaml",
        help="Path to model configuration file"
    )
    
    # Training arguments
    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training/standard/base_training.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
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
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_SEQUENCE_LENGTH,
        help="Maximum sequence length"
    )
    
    # Experiment arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for experiment tracking"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Output directory for models"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    
    # Other arguments
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Use Weights & Biases for tracking"
    )
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Use MLflow for tracking"
    )
    
    return parser.parse_args()


def main():
    """
    Main training pipeline.
    
    Implements end-to-end training following:
    - Dodge et al. (2019): "Show Your Work: Improved Reporting"
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup reproducibility
    ensure_reproducibility(seed=args.seed, deterministic=True)
    
    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"{args.model_name.split('/')[-1]}_{timestamp}"
    
    logger.info(f"Starting experiment: {args.experiment_name}")
    
    # Create experiment tracker
    experiment = create_experiment(
        name=args.experiment_name,
        use_wandb=args.use_wandb,
        use_mlflow=args.use_mlflow
    )
    
    # Load configurations
    if Path(args.training_config).exists():
        training_config = load_training_config(
            Path(args.training_config).stem,
            Path(args.training_config).parent.parent.name
        )
    else:
        training_config = {}
    
    # Override with command line arguments
    training_config.update({
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "fp16": args.fp16,
        "output_dir": args.output_dir
    })
    
    # Log hyperparameters
    log_hyperparameters(training_config)
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    logger.info("Loading AG News dataset")
    data_config = AGNewsConfig(
        data_dir=Path(args.data_dir),
        max_length=args.max_length
    )
    
    train_dataset, val_dataset, test_dataset = create_ag_news_datasets(
        data_config,
        tokenizer=tokenizer
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size,
        num_workers=4
    )
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=AG_NEWS_NUM_CLASSES
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        experiment_name=args.experiment_name
    )
    
    # Train model
    logger.info("Starting training")
    start_time = time.time()
    results = trainer.train()
    training_time = time.time() - start_time
    
    logger.info(f"Training completed in {training_time:.2f} seconds")
    logger.info(f"Best epoch: {results['best_epoch']}")
    logger.info(f"Best metric: {results['best_metric']:.4f}")
    
    # Final evaluation on test set
    logger.info("Evaluating on test set")
    model.eval()
    test_predictions = []
    test_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            predictions = torch.argmax(outputs.logits, dim=-1)
            test_predictions.extend(predictions.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    # Calculate test metrics
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, test_predictions, average="macro"
    )
    
    test_metrics = {
        "test_accuracy": test_accuracy,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_f1_macro": test_f1
    }
    
    logger.info("Test Results:")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    
    # Log test metrics
    log_metrics(test_metrics, step=results["final_epoch"])
    
    # Generate classification report
    report = classification_report(
        test_labels,
        test_predictions,
        target_names=AG_NEWS_CLASSES,
        digits=4
    )
    logger.info(f"\nClassification Report:\n{report}")
    
    # Save final results
    final_results = {
        "experiment_name": args.experiment_name,
        "model_name": args.model_name,
        "training_config": training_config,
        "training_results": results,
        "test_metrics": test_metrics,
        "training_time_seconds": training_time,
        "classification_report": report
    }
    
    output_dir = Path(args.output_dir) / args.experiment_name
    ensure_dir(output_dir)
    safe_save(final_results, output_dir / "final_results.json")
    
    # Create reproducibility report
    create_reproducibility_report(
        experiment_config=training_config,
        results=final_results,
        save_path=output_dir / "reproducibility_report.json"
    )
    
    # Close experiment tracker
    if experiment:
        experiment.close()
    
    logger.info(f"All results saved to {output_dir}")
    logger.info("Training script completed successfully")


if __name__ == "__main__":
    main()
