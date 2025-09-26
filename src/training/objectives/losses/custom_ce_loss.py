"""
Custom Cross-Entropy Loss Variants for AG News Text Classification
===================================================================

This module implements various custom cross-entropy loss functions with
enhanced capabilities for handling class imbalance, uncertainty, and
improved generalization.

Mathematical Foundation:
------------------------
Weighted CE: L = -Σ w_i * y_i * log(p_i)
Focal CE: L = -Σ α_i * (1-p_i)^γ * y_i * log(p_i)
Bi-Tempered: L = -Σ (1 - (1 - p_i^t1)^t2) / t2

References:
- Lin et al. (2017): "Focal Loss for Dense Object Detection"
- Amid et al. (2019): "Robust Bi-Tempered Logistic Loss"
- Cui et al. (2019): "Class-Balanced Loss Based on Effective Number of Samples"

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List
import numpy as np


class BalancedCrossEntropyLoss(nn.Module):
    """
    Balanced Cross-Entropy Loss for handling class imbalance.
    
    Automatically computes class weights based on effective number of samples
    or uses provided weights.
    """
    
    def __init__(
        self,
        class_counts: Optional[torch.Tensor] = None,
        beta: float = 0.999,
        weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        ignore_index: int = -100
    ):
        """
        Initialize Balanced Cross-Entropy Loss.
        
        Args:
            class_counts: Number of samples per class
            beta: Re-weighting strength (0 to 1)
            weight: Manual class weights (overrides automatic computation)
            reduction: Loss reduction method
            ignore_index: Index to ignore in loss
        """
        super().__init__()
        
        self.beta = beta
        self.reduction = reduction
        self.ignore_index = ignore_index
        
        # Compute effective weights
        if weight is not None:
            self.register_buffer('weight', weight)
        elif class_counts is not None:
            effective_num = 1.0 - torch.pow(beta, class_counts)
            weights = (1.0 - beta) / effective_num
            weights = weights / weights.sum() * len(weights)
            self.register_buffer('weight', weights)
        else:
            self.weight = None
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute balanced cross-entropy loss.
        
        Args:
            inputs: Predicted logits (N, C)
            targets: Ground truth labels (N,)
            
        Returns:
            Computed loss value
        """
        return F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction=self.reduction,
            ignore_index=self.ignore_index
        )


class AsymmetricCrossEntropyLoss(nn.Module):
    """
    Asymmetric Cross-Entropy Loss for handling noisy labels.
    
    Applies different penalties for false positives and false negatives.
    """
    
    def __init__(
        self,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = 'mean'
    ):
        """
        Initialize Asymmetric Cross-Entropy Loss.
        
        Args:
            gamma_pos: Focusing parameter for positive samples
            gamma_neg: Focusing parameter for negative samples
            clip: Probability clipping value
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute asymmetric cross-entropy loss.
        
        Args:
            inputs: Predicted logits (N, C)
            targets: Ground truth labels (N,) or one-hot (N, C)
            
        Returns:
            Computed loss value
        """
        num_classes = inputs.size(1)
        
        # Convert to one-hot if needed
        if targets.dim() == 1:
            targets_onehot = F.one_hot(targets, num_classes).float()
        else:
            targets_onehot = targets
        
        # Compute probabilities
        probs = torch.sigmoid(inputs)
        
        # Clip probabilities
        if self.clip > 0:
            probs = torch.clamp(probs, min=self.clip, max=1.0 - self.clip)
        
        # Compute asymmetric focusing
        probs_pos = probs[targets_onehot == 1]
        probs_neg = probs[targets_onehot == 0]
        
        # Positive loss
        loss_pos = -(1 - probs_pos).pow(self.gamma_pos) * probs_pos.log()
        
        # Negative loss
        loss_neg = -(probs_neg).pow(self.gamma_neg) * (1 - probs_neg).log()
        
        # Combine losses
        loss = torch.cat([loss_pos, loss_neg])
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class BiTemperedCrossEntropyLoss(nn.Module):
    """
    Bi-Tempered Cross-Entropy Loss for robust training.
    
    Uses two temperature parameters to handle outliers and
    improve generalization.
    """
    
    def __init__(
        self,
        t1: float = 0.5,
        t2: float = 1.0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Bi-Tempered Cross-Entropy Loss.
        
        Args:
            t1: Temperature for log probability
            t2: Temperature for normalization
            label_smoothing: Label smoothing factor
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute bi-tempered cross-entropy loss.
        
        Args:
            inputs: Predicted logits (N, C)
            targets: Ground truth labels (N,)
            
        Returns:
            Computed loss value
        """
        num_classes = inputs.size(1)
        
        # Compute tempered softmax
        if self.t2 == 1.0:
            probabilities = F.softmax(inputs / self.t1, dim=1)
        else:
            # Compute normalization constant
            exp_t1 = torch.exp(inputs / self.t1)
            Z = exp_t1.sum(dim=1, keepdim=True)
            probabilities = exp_t1 / Z
            
            # Apply second temperature
            probabilities = probabilities.pow(1.0 / self.t2)
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_onehot = F.one_hot(targets, num_classes).float()
            targets_smooth = targets_onehot * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes
        else:
            targets_smooth = F.one_hot(targets, num_classes).float()
        
        # Compute tempered log probabilities
        if self.t1 == 1.0:
            log_probabilities = torch.log(probabilities + 1e-12)
        else:
            # Tempered logarithm
            log_probabilities = (probabilities.pow(1 - self.t1) - 1) / (1 - self.t1)
        
        # Compute loss
        loss = -(targets_smooth * log_probabilities).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class PolyLoss(nn.Module):
    """
    Polynomial Cross-Entropy Loss for better optimization landscape.
    
    Adds polynomial terms to standard cross-entropy for smoother gradients.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        alpha: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Poly Loss.
        
        Args:
            epsilon: Weight of polynomial term
            alpha: Polynomial power
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.epsilon = epsilon
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute polynomial cross-entropy loss.
        
        Args:
            inputs: Predicted logits (N, C)
            targets: Ground truth labels (N,)
            
        Returns:
            Computed loss value
        """
        # Standard cross-entropy
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Get probability of correct class
        batch_size = targets.size(0)
        correct_probs = probs[torch.arange(batch_size), targets]
        
        # Polynomial term: (1 - p_correct)^alpha
        poly_term = (1 - correct_probs).pow(self.alpha)
        
        # Combined loss
        loss = ce_loss + self.epsilon * poly_term
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class SigmoidCrossEntropyLoss(nn.Module):
    """
    Sigmoid Cross-Entropy Loss for multi-label or imbalanced classification.
    
    Treats each class independently with sigmoid activation.
    """
    
    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        focal: bool = False,
        gamma: float = 2.0,
        alpha: float = 0.25
    ):
        """
        Initialize Sigmoid Cross-Entropy Loss.
        
        Args:
            weight: Manual class weights
            pos_weight: Weight for positive samples
            reduction: Loss reduction method
            focal: Whether to use focal variant
            gamma: Focal loss focusing parameter
            alpha: Focal loss balancing parameter
        """
        super().__init__()
        
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.focal = focal
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sigmoid cross-entropy loss.
        
        Args:
            inputs: Predicted logits (N, C)
            targets: Ground truth labels (N,) or (N, C) for multi-label
            
        Returns:
            Computed loss value
        """
        # Convert to multi-label format if needed
        if targets.dim() == 1:
            num_classes = inputs.size(1)
            targets = F.one_hot(targets, num_classes).float()
        
        # Basic BCE with logits
        loss = F.binary_cross_entropy_with_logits(
            inputs,
            targets,
            weight=self.weight,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        # Apply focal modulation if enabled
        if self.focal:
            probs = torch.sigmoid(inputs)
            p_t = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - p_t).pow(self.gamma)
            
            if self.alpha is not None:
                alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
                focal_weight = alpha_t * focal_weight
            
            loss = focal_weight * loss
        
        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
