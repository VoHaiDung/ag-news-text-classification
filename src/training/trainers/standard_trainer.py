"""
Standard Trainer Implementation for AG News Text Classification
================================================================

This module implements the standard training strategy with enhanced features
for transformer-based models, building upon the base trainer architecture.

Key Features:
- Label smoothing regularization (Müller et al., 2019)
- Dynamic loss scaling for mixed precision
- Adaptive learning rate scheduling
- Knowledge distillation support
- Multi-task learning capabilities

References:
- Müller et al. (2019): "When Does Label Smoothing Help?"
- Liu et al. (2019): "Multi-Task Deep Neural Networks for Natural Language Understanding"
- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.trainers.base_trainer import BaseTrainer, TrainerConfig
from src.models.base.base_model import AGNewsBaseModel
from src.training.objectives.losses.focal_loss import FocalLoss
from src.training.objectives.losses.label_smoothing import LabelSmoothingCrossEntropy
from src.core.exceptions import TrainingError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StandardTrainerConfig(TrainerConfig):
    """Configuration for standard trainer."""
    
    # Loss configuration
    loss_type: str = "cross_entropy"  # cross_entropy, focal, label_smoothing
    label_smoothing: float = 0.1
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # Multi-task learning
    use_multitask: bool = False
    auxiliary_task_weight: float = 0.1
    
    # Distillation
    use_distillation: bool = False
    distillation_temperature: float = 3.0
    distillation_alpha: float = 0.5
    
    # Advanced features
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_prob: float = 0.5
    
    # Dynamic loss scaling
    dynamic_loss_scale: bool = True
    initial_loss_scale: float = 2**15
    loss_scale_window: int = 2000


class StandardTrainer(BaseTrainer):
    """
    Standard trainer with enhanced features for transformer models.
    
    This trainer implements the standard training procedure with
    additional regularization and optimization techniques specifically
    designed for text classification tasks.
    """
    
    def __init__(
        self,
        model: AGNewsBaseModel,
        config: Optional[StandardTrainerConfig] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        teacher_model: Optional[AGNewsBaseModel] = None
    ):
        """
        Initialize standard trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            teacher_model: Teacher model for distillation
        """
        # Initialize configuration
        self.config = config or StandardTrainerConfig()
        
        # Initialize base trainer
        super().__init__(
            model=model,
            config=self.config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Teacher model for distillation
        if self.config.use_distillation and teacher_model:
            self.teacher_model = teacher_model.to(self.device)
            self.teacher_model.eval()
        else:
            self.teacher_model = None
        
        # Dynamic loss scaling
        if self.config.dynamic_loss_scale:
            self.loss_scale = self.config.initial_loss_scale
            self.loss_scale_window = 0
            self.min_loss_scale = 1.0
        
        logger.info(
            f"Initialized StandardTrainer with {self.config.loss_type} loss"
        )
    
    def _create_loss_function(self) -> nn.Module:
        """
        Create loss function based on configuration.
        
        Returns:
            Loss function module
        """
        if self.config.loss_type == "focal":
            return FocalLoss(
                alpha=self.config.focal_alpha,
                gamma=self.config.focal_gamma,
                reduction='mean'
            )
        elif self.config.loss_type == "label_smoothing":
            return LabelSmoothingCrossEntropy(
                smoothing=self.config.label_smoothing
            )
        else:
            return nn.CrossEntropyLoss()
    
    def _compute_loss(
        self,
        outputs: Any,
        labels: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute training loss with optional distillation and regularization.
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
            input_ids: Input token IDs for mixup
            
        Returns:
            Total loss
        """
        # Primary task loss
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs
        
        # Apply mixup if enabled
        if self.config.use_mixup and self.training:
            logits, labels = self._apply_mixup(logits, labels)
        
        # Classification loss
        classification_loss = self.criterion(logits, labels)
        
        total_loss = classification_loss
        
        # Distillation loss
        if self.config.use_distillation and self.teacher_model:
            distillation_loss = self._compute_distillation_loss(
                student_logits=logits,
                input_ids=input_ids
            )
            total_loss = (
                (1 - self.config.distillation_alpha) * classification_loss +
                self.config.distillation_alpha * distillation_loss
            )
        
        # Multi-task loss
        if self.config.use_multitask and hasattr(outputs, 'auxiliary_logits'):
            auxiliary_loss = self.criterion(outputs.auxiliary_logits, labels)
            total_loss += self.config.auxiliary_task_weight * auxiliary_loss
        
        return total_loss
    
    def _compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Args:
            student_logits: Student model predictions
            input_ids: Input token IDs
            
        Returns:
            Distillation loss
        """
        with torch.no_grad():
            teacher_outputs = self.teacher_model(input_ids=input_ids)
            teacher_logits = teacher_outputs.logits
        
        # KL divergence loss with temperature
        T = self.config.distillation_temperature
        distillation_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=-1),
            F.softmax(teacher_logits / T, dim=-1),
            reduction='batchmean'
        ) * (T * T)
        
        return distillation_loss
    
    def _apply_mixup(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply mixup augmentation.
        
        Args:
            logits: Model predictions
            labels: Ground truth labels
            
        Returns:
            Mixed logits and labels
        """
        batch_size = logits.size(0)
        
        # Sample lambda from Beta distribution
        lam = torch.distributions.Beta(
            self.config.mixup_alpha,
            self.config.mixup_alpha
        ).sample().to(self.device)
        
        # Random permutation for mixing
        index = torch.randperm(batch_size).to(self.device)
        
        # Mix predictions
        mixed_logits = lam * logits + (1 - lam) * logits[index]
        
        # Convert labels to one-hot for mixing
        labels_onehot = F.one_hot(labels, num_classes=logits.size(-1)).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels_onehot[index]
        
        return mixed_logits, mixed_labels
    
    def _train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute single training step.
        
        Args:
            batch: Input batch
            
        Returns:
            Loss and metrics
        """
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            token_type_ids=batch.get("token_type_ids")
        )
        
        # Compute loss
        loss = self._compute_loss(
            outputs=outputs,
            labels=batch["labels"],
            input_ids=batch["input_ids"]
        )
        
        # Dynamic loss scaling
        if self.config.dynamic_loss_scale and self.config.use_mixed_precision:
            loss = self._apply_dynamic_loss_scaling(loss)
        
        # Compute metrics
        with torch.no_grad():
            predictions = torch.argmax(outputs.logits, dim=-1)
            accuracy = (predictions == batch["labels"]).float().mean().item()
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy
        }
        
        return loss, metrics
    
    def _apply_dynamic_loss_scaling(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Apply dynamic loss scaling for mixed precision training.
        
        Args:
            loss: Original loss
            
        Returns:
            Scaled loss
        """
        # Check for overflow
        if torch.isnan(loss) or torch.isinf(loss):
            # Reduce loss scale
            self.loss_scale = max(self.loss_scale / 2, self.min_loss_scale)
            self.loss_scale_window = 0
            logger.warning(f"Loss overflow detected, reducing scale to {self.loss_scale}")
            return torch.tensor(0.0, requires_grad=True).to(self.device)
        
        # Increase loss scale if stable
        self.loss_scale_window += 1
        if self.loss_scale_window >= self.config.loss_scale_window:
            self.loss_scale = min(self.loss_scale * 2, self.config.initial_loss_scale)
            self.loss_scale_window = 0
        
        return loss * self.loss_scale
    
    def _optimize_step(self, loss: torch.Tensor):
        """
        Execute optimization step with gradient management.
        
        Args:
            loss: Training loss
        """
        # Scale gradients for mixed precision
        if self.config.dynamic_loss_scale and self.config.use_mixed_precision:
            loss = loss / self.loss_scale
        
        # Backward pass
        if self.config.use_mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if self.config.use_mixed_precision and self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm
            )
            
            # Log gradient norm
            if self.global_step % self.config.logging_steps == 0:
                self.tracker.log({"train/grad_norm": grad_norm}, step=self.global_step)
        
        # Optimizer step
        if self.config.use_mixed_precision and self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Learning rate scheduling
        if self.scheduler:
            self.scheduler.step()
        
        # Zero gradients
        self.optimizer.zero_grad()
    
    def train_with_validation(
        self,
        num_epochs: int,
        validation_interval: int = 1
    ) -> Dict[str, Any]:
        """
        Train with periodic validation.
        
        Args:
            num_epochs: Number of training epochs
            validation_interval: Epochs between validations
            
        Returns:
            Training results
        """
        logger.info(
            f"Starting standard training for {num_epochs} epochs "
            f"with validation every {validation_interval} epochs"
        )
        
        best_val_metric = float('-inf') if self._is_maximize_metric() else float('inf')
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            epoch_loss = 0
            epoch_steps = 0
            
            for batch in self.train_loader:
                # Move to device
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                # Training step
                loss, metrics = self._train_step(batch)
                
                # Optimization
                self._optimize_step(loss)
                
                epoch_loss += loss.item()
                epoch_steps += 1
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self.tracker.log(metrics, step=self.global_step)
            
            # Validation
            if (epoch + 1) % validation_interval == 0:
                val_metrics = self._validate()
                
                # Check for best model
                val_metric = val_metrics.get(
                    self.config.early_stopping_metric,
                    val_metrics.get("loss")
                )
                
                if self._is_better_metric(val_metric, best_val_metric):
                    best_val_metric = val_metric
                    self.save_checkpoint(
                        self.checkpoint_dir / "best_model.pt",
                        is_best=True
                    )
                
                logger.info(
                    f"Epoch {epoch + 1}: "
                    f"Train Loss: {epoch_loss / epoch_steps:.4f}, "
                    f"Val {self.config.early_stopping_metric}: {val_metric:.4f}"
                )
        
        return {
            "best_val_metric": best_val_metric,
            "final_epoch": num_epochs,
            "total_steps": self.global_step
        }
    
    def _is_better_metric(self, current: float, best: float) -> bool:
        """Check if current metric is better than best."""
        if self._is_maximize_metric():
            return current > best
        return current < best
