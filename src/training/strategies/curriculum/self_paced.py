"""
Self-Paced Learning Implementation
===================================

Implementation of self-paced learning for robust training with noisy data,
based on:
- Kumar et al. (2010): "Self-Paced Learning for Latent Variable Models"
- Jiang et al. (2015): "Self-Paced Learning with Diversity"
- Ma et al. (2017): "Self-Paced Co-training"

Mathematical Foundation:
Self-paced learning optimizes:
min_{w,v} E(w) + f(v; λ)
where v ∈ {0,1}^n are sample weights, λ controls learning pace

Key insight: Model automatically determines which samples to learn from,
starting with easy samples and gradually including harder ones.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

from src.training.trainers.base_trainer import BaseTrainer, TrainerConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SelfPacedConfig(TrainerConfig):
    """Configuration for self-paced learning."""
    
    # Self-paced strategy
    spl_type: str = "hard"  # "hard", "soft", "linear", "mixture"
    
    # Pace parameters
    initial_lambda: float = 0.1  # Initial threshold parameter
    lambda_growth_rate: float = 1.5  # Growth rate for lambda
    lambda_update_freq: int = 1  # Update lambda every N epochs
    max_lambda: float = 10.0  # Maximum lambda value
    
    # Sample weighting
    weight_scheme: str = "binary"  # "binary", "continuous", "probabilistic"
    min_weight: float = 0.0  # Minimum sample weight
    max_weight: float = 1.0  # Maximum sample weight
    
    # Diversity regularization
    use_diversity: bool = True  # Enable diversity-based selection
    diversity_weight: float = 0.1  # Weight for diversity term
    diversity_metric: str = "cosine"  # "cosine", "euclidean", "angular"
    
    # Adaptive pace
    adaptive_pace: bool = True  # Automatically adjust pace
    target_inclusion_rate: float = 0.8  # Target percentage of samples
    
    # Curriculum constraints
    min_samples_per_epoch: int = 100  # Minimum samples to include
    max_samples_per_epoch: Optional[int] = None  # Maximum samples
    
    # Loss thresholds
    loss_percentile: float = 70.0  # Percentile for loss threshold
    confidence_threshold: float = 0.5  # Confidence threshold for selection
    
    # Regularization
    regularization_type: str = "l2"  # "l1", "l2", "elastic"
    regularization_weight: float = 0.01


class SelfPacedWeightFunction:
    """
    Weight functions for self-paced learning.
    
    Different functions provide different selection strategies.
    """
    
    @staticmethod
    def hard_weighting(losses: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Hard weighting: binary selection based on threshold.
        
        v_i = 1 if l_i < λ else 0
        """
        return (losses < lambda_).float()
    
    @staticmethod
    def soft_weighting(losses: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Soft weighting: continuous weights based on loss.
        
        v_i = max(0, 1 - l_i/λ)
        """
        weights = 1.0 - losses / lambda_
        return torch.clamp(weights, min=0.0, max=1.0)
    
    @staticmethod
    def linear_weighting(losses: torch.Tensor, lambda_: float) -> torch.Tensor:
        """
        Linear weighting: linear decay from threshold.
        
        v_i = (λ - l_i) / λ if l_i < λ else 0
        """
        weights = (lambda_ - losses) / lambda_
        return torch.clamp(weights, min=0.0, max=1.0)
    
    @staticmethod
    def mixture_weighting(
        losses: torch.Tensor,
        lambda_: float,
        gamma: float = 0.5
    ) -> torch.Tensor:
        """
        Mixture weighting: combination of hard and soft.
        
        v_i = γ * hard(l_i, λ) + (1-γ) * soft(l_i, λ)
        """
        hard = SelfPacedWeightFunction.hard_weighting(losses, lambda_)
        soft = SelfPacedWeightFunction.soft_weighting(losses, lambda_)
        return gamma * hard + (1 - gamma) * soft


class DiversityRegularizer:
    """
    Diversity regularization for self-paced learning.
    
    Encourages selection of diverse samples to avoid overfitting
    to easy samples.
    """
    
    def __init__(self, metric: str = "cosine"):
        """
        Initialize diversity regularizer.
        
        Args:
            metric: Distance metric for diversity
        """
        self.metric = metric
    
    def compute_diversity(
        self,
        features: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute diversity score for weighted samples.
        
        Args:
            features: Sample features [batch_size, feature_dim]
            weights: Sample weights [batch_size]
            
        Returns:
            Diversity score
        """
        # Normalize features
        if self.metric == "cosine":
            features = F.normalize(features, p=2, dim=1)
        
        # Compute pairwise distances
        distances = torch.cdist(features, features, p=2)
        
        # Weight by sample selection
        weighted_distances = distances * weights.unsqueeze(1) * weights.unsqueeze(0)
        
        # Diversity is average weighted distance
        diversity = weighted_distances.sum() / (weights.sum() ** 2 + 1e-8)
        
        return diversity


class SelfPacedLearning(BaseTrainer):
    """
    Self-paced learning trainer.
    
    The model automatically determines curriculum by selecting samples
    based on current learning state. This provides robustness to noisy
    labels and improves convergence.
    
    Key features:
    1. Automatic sample selection based on loss
    2. Diversity regularization for balanced learning
    3. Adaptive pace adjustment
    4. Multiple weighting schemes
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[SelfPacedConfig] = None,
        **kwargs
    ):
        """
        Initialize self-paced learning trainer.
        
        Args:
            model: Model to train
            config: Self-paced learning configuration
            **kwargs: Additional trainer arguments
        """
        config = config or SelfPacedConfig()
        super().__init__(model, config, **kwargs)
        
        self.config: SelfPacedConfig = config
        
        # Initialize weight functions
        self.weight_functions = {
            "hard": SelfPacedWeightFunction.hard_weighting,
            "soft": SelfPacedWeightFunction.soft_weighting,
            "linear": SelfPacedWeightFunction.linear_weighting,
            "mixture": SelfPacedWeightFunction.mixture_weighting
        }
        self.weight_function = self.weight_functions[config.spl_type]
        
        # Initialize diversity regularizer
        if config.use_diversity:
            self.diversity_regularizer = DiversityRegularizer(config.diversity_metric)
        else:
            self.diversity_regularizer = None
        
        # Self-paced parameters
        self.lambda_ = config.initial_lambda
        self.sample_weights = None
        self.sample_losses = None
        
        # Statistics tracking
        self.inclusion_rates = []
        self.average_losses = []
        
        logger.info(
            f"Initialized SelfPacedLearning with {config.spl_type} weighting, "
            f"lambda={self.lambda_:.3f}"
        )
    
    def compute_sample_weights(
        self,
        losses: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute sample weights based on losses.
        
        Args:
            losses: Loss values for samples
            features: Optional features for diversity
            
        Returns:
            Sample weights
        """
        # Base weights from loss
        weights = self.weight_function(losses, self.lambda_)
        
        # Apply diversity regularization
        if self.diversity_regularizer is not None and features is not None:
            diversity = self.diversity_regularizer.compute_diversity(
                features, weights
            )
            # Adjust weights to increase diversity
            weights = weights + self.config.diversity_weight * diversity
            weights = torch.clamp(weights, 0.0, 1.0)
        
        # Apply constraints
        if self.config.min_samples_per_epoch > 0:
            num_selected = weights.sum().item()
            if num_selected < self.config.min_samples_per_epoch:
                # Select top-k samples by weight
                k = self.config.min_samples_per_epoch
                _, indices = torch.topk(weights, k, largest=True)
                min_weights = torch.zeros_like(weights)
                min_weights[indices] = 1.0
                weights = torch.maximum(weights, min_weights)
        
        if self.config.max_samples_per_epoch is not None:
            num_selected = weights.sum().item()
            if num_selected > self.config.max_samples_per_epoch:
                # Threshold to limit samples
                k = self.config.max_samples_per_epoch
                threshold = torch.topk(weights, k, largest=True)[0][-1]
                weights = (weights >= threshold).float() * weights
        
        return weights
    
    def update_pace(self, epoch: int, inclusion_rate: float):
        """
        Update learning pace (lambda parameter).
        
        Args:
            epoch: Current epoch
            inclusion_rate: Percentage of samples included
        """
        if epoch % self.config.lambda_update_freq != 0:
            return
        
        if self.config.adaptive_pace:
            # Adaptive adjustment based on inclusion rate
            if inclusion_rate < self.config.target_inclusion_rate:
                # Too few samples, increase lambda
                self.lambda_ *= self.config.lambda_growth_rate
            else:
                # Enough samples, moderate increase
                self.lambda_ *= (1.0 + (self.config.lambda_growth_rate - 1.0) * 0.5)
        else:
            # Fixed growth rate
            self.lambda_ *= self.config.lambda_growth_rate
        
        # Clamp lambda
        self.lambda_ = min(self.lambda_, self.config.max_lambda)
        
        logger.info(
            f"Updated lambda to {self.lambda_:.3f} "
            f"(inclusion rate: {inclusion_rate:.2%})"
        )
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Training step with self-paced weighting.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        # Forward pass
        inputs = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        outputs = self.model(
            inputs,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Compute per-sample loss
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        per_sample_loss = loss_fct(
            logits.view(-1, logits.size(-1)),
            labels.view(-1)
        ).view(labels.size())
        
        # Average over sequence length
        if attention_mask is not None:
            per_sample_loss = (per_sample_loss * attention_mask).sum(1) / attention_mask.sum(1)
        else:
            per_sample_loss = per_sample_loss.mean(1)
        
        # Get features for diversity (use hidden states)
        features = outputs.hidden_states[-1].mean(dim=1) if hasattr(outputs, 'hidden_states') else None
        
        # Compute sample weights
        with torch.no_grad():
            weights = self.compute_sample_weights(per_sample_loss, features)
        
        # Weighted loss
        weighted_loss = (per_sample_loss * weights).sum() / (weights.sum() + 1e-8)
        
        # Add regularization
        if self.config.regularization_weight > 0:
            reg_loss = self.compute_regularization_loss()
            total_loss = weighted_loss + self.config.regularization_weight * reg_loss
        else:
            total_loss = weighted_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.config.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_val
            )
        
        self.optimizer.step()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Metrics
        inclusion_rate = (weights > 0.5).float().mean().item()
        
        metrics = {
            "loss": total_loss.item(),
            "weighted_loss": weighted_loss.item(),
            "inclusion_rate": inclusion_rate,
            "lambda": self.lambda_,
            "avg_weight": weights.mean().item(),
            "num_selected": (weights > 0.5).sum().item()
        }
        
        return metrics
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """Compute regularization loss."""
        reg_loss = 0.0
        
        for param in self.model.parameters():
            if self.config.regularization_type == "l1":
                reg_loss += torch.abs(param).sum()
            elif self.config.regularization_type == "l2":
                reg_loss += torch.norm(param, p=2)
            elif self.config.regularization_type == "elastic":
                reg_loss += torch.abs(param).sum() + torch.norm(param, p=2)
        
        return reg_loss
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with self-paced learning.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        self.model.train()
        epoch_metrics = {
            "loss": 0.0,
            "weighted_loss": 0.0,
            "inclusion_rate": 0.0,
            "num_selected": 0
        }
        
        num_batches = 0
        
        for batch in self.train_loader:
            metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value
            
            num_batches += 1
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Update pace
        self.update_pace(epoch, epoch_metrics["inclusion_rate"])
        
        # Track statistics
        self.inclusion_rates.append(epoch_metrics["inclusion_rate"])
        self.average_losses.append(epoch_metrics["loss"])
        
        return epoch_metrics
