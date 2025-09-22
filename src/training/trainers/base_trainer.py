"""
Base Trainer Implementation for AG News Text Classification
============================================================

This module implements the base trainer class following best practices from:
- Smith (2017): "A Disciplined Approach to Neural Network Hyper-Parameters"
- Goyal et al. (2017): "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
- You et al. (2020): "Large Batch Optimization for Deep Learning"
- Zhang et al. (2021): "Improved Regularization and Robustness for Fine-tuning"

Mathematical Foundation:
Optimization objective: min_θ L(θ) = E[ℓ(f_θ(x), y)] + λR(θ)
where ℓ is the loss function, R is regularization, λ is regularization weight.

Training dynamics follow:
θ_{t+1} = θ_t - η_t ∇L(θ_t)
where η_t is the learning rate at step t.

Author: Võ Hải Dũng
License: MIT
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict
import json
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler, autocast

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.types import TrainingConfig, EvaluationResults, TrainingState
from src.core.exceptions import TrainingError, CheckpointError
from src.utils.logging_config import get_logger
from src.utils.reproducibility import set_seed
from src.utils.memory_utils import optimize_memory
from src.utils.experiment_tracking import ExperimentTracker

logger = get_logger(__name__)


@dataclass
class TrainerConfig(TrainingConfig):
    """Extended configuration for trainer."""
    # Training strategy
    training_strategy: str = "standard"  # "standard", "adversarial", "curriculum"
    
    # Optimization
    use_swa: bool = False  # Stochastic Weight Averaging
    swa_start_epoch: int = 5
    swa_lr: float = 0.001
    
    # Gradient management
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    gradient_penalty: float = 0.0
    
    # Mixed precision
    use_mixed_precision: bool = True
    mixed_precision_opt_level: str = "O1"
    
    # Early stopping
    early_stopping_patience: int = 5
    early_stopping_delta: float = 0.0001
    early_stopping_metric: str = "f1_macro"
    
    # Checkpointing
    save_best_only: bool = True
    save_last: bool = True
    save_optimizer: bool = True
    save_scheduler: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Evaluation
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None
    
    # Logging
    logging_steps: int = 50
    log_level: str = "INFO"
    use_tensorboard: bool = True
    use_wandb: bool = False
    
    # Resource management
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    empty_cache_steps: int = 1000
    
    # Robustness
    use_ema: bool = False  # Exponential Moving Average
    ema_decay: float = 0.999
    noise_std: float = 0.0  # Gaussian noise for robustness


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting.
    
    Monitors validation metric and stops training when no improvement
    is observed for a specified number of epochs.
    """
    
    def __init__(
        self,
        patience: int = 5,
        delta: float = 0.0,
        mode: str = "max",
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait
            delta: Minimum change to qualify as improvement
            mode: "max" for metrics like accuracy, "min" for loss
            verbose: Print messages
        """
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if should stop training.
        
        Args:
            score: Current validation score
            epoch: Current epoch
            
        Returns:
            Whether to stop training
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == "max":
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(f"Validation score improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(
                    f"No improvement for {self.counter} epochs "
                    f"(best: {self.best_score:.4f} at epoch {self.best_epoch})"
                )
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info("Early stopping triggered")
        
        return self.early_stop


class ExponentialMovingAverage:
    """
    Exponential Moving Average of model parameters.
    
    Maintains shadow variables for better generalization.
    Based on Polyak & Juditsky (1992): "Acceleration of Stochastic Approximation"
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        """
        Initialize EMA.
        
        Args:
            model: Model to track
            decay: Decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self):
        """Apply shadow parameters to model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}


class BaseTrainer:
    """
    Base trainer class for AG News classification models.
    
    Provides comprehensive training functionality with:
    1. Mixed precision training
    2. Gradient accumulation
    3. Early stopping
    4. Model checkpointing
    5. Experiment tracking
    6. Memory optimization
    7. Robustness techniques
    
    The trainer follows best practices for stable and efficient training
    of deep learning models.
    """
    
    def __init__(
        self,
        model: AGNewsBaseModel,
        config: Optional[TrainerConfig] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer (will be created if None)
            scheduler: Learning rate scheduler
        """
        self.model = model
        self.config = config or TrainerConfig()
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config.device == "cuda"
            else "cpu"
        )
        self.model.to(self.device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optimizer or self._create_optimizer()
        self.scheduler = scheduler or self._create_scheduler()
        
        # Initialize training components
        self._init_training_components()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("-inf") if self._is_maximize_metric() else float("inf")
        self.training_history = defaultdict(list)
        
        # Create checkpoint directory
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized {self.__class__.__name__} "
            f"on {self.device} with {self.config.training_strategy} strategy"
        )
    
    def _init_training_components(self):
        """Initialize training components."""
        # Mixed precision
        if self.config.use_mixed_precision:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            delta=self.config.early_stopping_delta,
            mode="max" if self._is_maximize_metric() else "min"
        )
        
        # EMA
        if self.config.use_ema:
            self.ema = ExponentialMovingAverage(self.model, self.config.ema_decay)
        else:
            self.ema = None
        
        # Experiment tracking
        self.tracker = ExperimentTracker(
            use_tensorboard=self.config.use_tensorboard,
            use_wandb=self.config.use_wandb,
            config=self.config
        )
        
        # SWA
        if self.config.use_swa:
            from torch.optim.swa_utils import AveragedModel
            self.swa_model = AveragedModel(self.model)
        else:
            self.swa_model = None
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer based on configuration."""
        # Get parameters with weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        # Create optimizer
        if self.config.optimizer_name == "adamw":
            from torch.optim import AdamW
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_epsilon
            )
        elif self.config.optimizer_name == "sgd":
            from torch.optim import SGD
            optimizer = SGD(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                momentum=0.9,
                nesterov=True
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer_name}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler."""
        if not self.train_loader:
            return None
        
        num_training_steps = (
            len(self.train_loader) // self.config.gradient_accumulation_steps
            * self.config.num_epochs
        )
        
        if self.config.scheduler_name == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=num_training_steps // self.config.num_epochs,
                T_mult=1,
                eta_min=0
            )
        elif self.config.scheduler_name == "linear":
            from transformers import get_linear_schedule_with_warmup
            scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.num_warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            scheduler = None
        
        return scheduler
    
    def _is_maximize_metric(self) -> bool:
        """Check if metric should be maximized."""
        maximize_metrics = ["accuracy", "f1", "precision", "recall", "auc"]
        return any(m in self.config.early_stopping_metric for m in maximize_metrics)
    
    def train(
        self,
        num_epochs: Optional[int] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self.load_checkpoint(resume_from_checkpoint)
        
        # Set seed for reproducibility
        set_seed(self.config.seed)
        
        logger.info(
            f"Starting training for {num_epochs} epochs "
            f"from epoch {self.current_epoch}"
        )
        
        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch(epoch)
            
            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate()
                
                # Early stopping
                stop_metric = val_metrics.get(
                    self.config.early_stopping_metric,
                    val_metrics.get("loss", 0)
                )
                
                if self.early_stopping(stop_metric, epoch):
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
                
                # Save best model
                if self._is_best_model(stop_metric):
                    self.best_metric = stop_metric
                    self.save_checkpoint(
                        self.checkpoint_dir / "best_model.pt",
                        is_best=True
                    )
            
            # SWA update
            if self.config.use_swa and epoch >= self.config.swa_start_epoch:
                self.swa_model.update_parameters(self.model)
            
            # Logging
            self._log_epoch_summary(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_steps == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                )
        
        # Final cleanup
        self._finalize_training()
        
        # Return training history
        return {
            "history": dict(self.training_history),
            "best_metric": self.best_metric,
            "best_epoch": self.early_stopping.best_epoch,
            "total_steps": self.global_step
        }
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        epoch_predictions = []
        epoch_labels = []
        
        # Progress bar
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            disable=not logger.isEnabledFor(logging.INFO)
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if self.config.use_mixed_precision:
                with autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        labels=batch.get("labels")
                    )
                    loss = outputs.loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels")
                )
                loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # Add regularization
            if self.config.gradient_penalty > 0:
                loss = loss + self._compute_gradient_penalty()
            
            # Add noise for robustness
            if self.config.noise_std > 0 and self.training:
                loss = loss + torch.randn_like(loss) * self.config.noise_std
            
            # Backward pass
            if self.config.use_mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                # Optimizer step
                if self.config.use_mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                # Scheduler step
                if self.scheduler:
                    self.scheduler.step()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # EMA update
                if self.ema:
                    self.ema.update()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self._log_training_step(loss.item(), self.global_step)
                
                # Memory optimization
                if self.global_step % self.config.empty_cache_steps == 0:
                    optimize_memory()
            
            # Accumulate metrics
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)
            
            # Store predictions
            if batch.get("labels") is not None:
                preds = torch.argmax(outputs.logits, dim=-1)
                epoch_predictions.extend(preds.cpu().numpy())
                epoch_labels.extend(batch["labels"].cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Calculate epoch metrics
        epoch_metrics = {
            "loss": total_loss / total_samples,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        if epoch_labels:
            from sklearn.metrics import accuracy_score, f1_score
            epoch_metrics["accuracy"] = accuracy_score(epoch_labels, epoch_predictions)
            epoch_metrics["f1_macro"] = f1_score(
                epoch_labels, epoch_predictions, average='macro'
            )
        
        return epoch_metrics
    
    def _validate(self) -> Dict[str, float]:
        """
        Validate model on validation set.
        
        Returns:
            Validation metrics
        """
        self.model.eval()
        
        # Apply EMA if available
        if self.ema:
            self.ema.apply_shadow()
        
        total_loss = 0
        total_samples = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc="Validation",
                disable=not logger.isEnabledFor(logging.INFO)
            ):
                # Move batch to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels")
                )
                
                # Accumulate metrics
                if outputs.loss is not None:
                    total_loss += outputs.loss.item() * batch["input_ids"].size(0)
                    total_samples += batch["input_ids"].size(0)
                
                # Store predictions
                if batch.get("labels") is not None:
                    preds = torch.argmax(outputs.logits, dim=-1)
                    all_predictions.extend(preds.cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())
        
        # Restore original weights if EMA was applied
        if self.ema:
            self.ema.restore()
        
        # Calculate metrics
        val_metrics = {"loss": total_loss / total_samples if total_samples > 0 else 0}
        
        if all_labels:
            from sklearn.metrics import (
                accuracy_score, precision_recall_fscore_support,
                confusion_matrix
            )
            
            val_metrics["accuracy"] = accuracy_score(all_labels, all_predictions)
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='macro'
            )
            val_metrics["precision"] = precision
            val_metrics["recall"] = recall
            val_metrics["f1_macro"] = f1
            
            val_metrics["confusion_matrix"] = confusion_matrix(
                all_labels, all_predictions
            ).tolist()
        
        return val_metrics
    
    def _compute_gradient_penalty(self) -> torch.Tensor:
        """
        Compute gradient penalty for regularization.
        
        Returns:
            Gradient penalty loss
        """
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        
        if gradients:
            all_gradients = torch.cat(gradients)
            gradient_norm = torch.norm(all_gradients, p=2)
            penalty = self.config.gradient_penalty * gradient_norm
            return penalty
        
        return torch.tensor(0.0).to(self.device)
    
    def _is_best_model(self, metric: float) -> bool:
        """Check if current model is best so far."""
        if self._is_maximize_metric():
            return metric > self.best_metric
        else:
            return metric < self.best_metric
    
    def _log_training_step(self, loss: float, step: int):
        """Log training step metrics."""
        self.tracker.log({
            "train/loss": loss,
            "train/learning_rate": self.optimizer.param_groups[0]['lr'],
            "train/step": step
        }, step=step)
    
    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]]
    ):
        """Log epoch summary."""
        # Store in history
        for key, value in train_metrics.items():
            self.training_history[f"train_{key}"].append(value)
        
        if val_metrics:
            for key, value in val_metrics.items():
                if key != "confusion_matrix":  # Skip large structures
                    self.training_history[f"val_{key}"].append(value)
        
        # Log to tracker
        log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
        if val_metrics:
            log_dict.update({f"val/{k}": v for k, v in val_metrics.items() 
                           if k != "confusion_matrix"})
        
        self.tracker.log(log_dict, step=epoch)
        
        # Print summary
        logger.info(
            f"Epoch {epoch+1} Summary:\n"
            f"  Train Loss: {train_metrics['loss']:.4f}\n"
            f"  Train Acc: {train_metrics.get('accuracy', 0):.4f}\n"
            f"  Val Loss: {val_metrics.get('loss', 0):.4f}\n" if val_metrics else ""
            f"  Val F1: {val_metrics.get('f1_macro', 0):.4f}\n" if val_metrics else ""
        )
    
    def save_checkpoint(
        self,
        path: Union[str, Path],
        is_best: bool = False
    ):
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if self.config.save_optimizer else None,
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler and self.config.save_scheduler else None,
            "best_metric": self.best_metric,
            "training_history": dict(self.training_history),
            "config": self.config,
            "is_best": is_best
        }
        
        # Add EMA state
        if self.ema:
            checkpoint["ema_shadow"] = self.ema.shadow
        
        # Add SWA state
        if self.swa_model:
            checkpoint["swa_state_dict"] = self.swa_model.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Union[str, Path]):
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        
        if checkpoint.get("optimizer_state_dict") and self.config.save_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if checkpoint.get("scheduler_state_dict") and self.scheduler and self.config.save_scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.training_history = defaultdict(list, checkpoint.get("training_history", {}))
        
        # Load EMA state
        if checkpoint.get("ema_shadow") and self.ema:
            self.ema.shadow = checkpoint["ema_shadow"]
        
        # Load SWA state
        if checkpoint.get("swa_state_dict") and self.swa_model:
            self.swa_model.load_state_dict(checkpoint["swa_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path} (epoch {self.current_epoch})")
    
    def _finalize_training(self):
        """Finalize training and cleanup."""
        # Update SWA batch norm if needed
        if self.swa_model and self.train_loader:
            from torch.optim.swa_utils import update_bn
            update_bn(self.train_loader, self.swa_model, device=self.device)
            
            # Save SWA model
            torch.save(
                self.swa_model.state_dict(),
                self.checkpoint_dir / "swa_model.pt"
            )
        
        # Save final checkpoint
        if self.config.save_last:
            self.save_checkpoint(self.checkpoint_dir / "last_model.pt")
        
        # Close tracker
        self.tracker.close()
        
        logger.info("Training completed successfully!")
