"""
Focal Loss Implementation for Imbalanced Classification
========================================================

Implementation of Focal Loss for addressing class imbalance in AG News dataset,
based on:
- Lin et al. (2017): "Focal Loss for Dense Object Detection"
- Mukhoti et al. (2020): "Calibrating Deep Neural Networks using Focal Loss"

Focal Loss addresses class imbalance by down-weighting well-classified examples,
focusing training on hard negatives and misclassified examples.

Mathematical Foundation:
FL(p_t) = -α_t(1 - p_t)^γ log(p_t)
where p_t is the model's estimated probability for the ground-truth class,
α_t is the weighting factor, and γ is the focusing parameter.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    The focal loss applies a modulating term to the cross entropy loss,
    focusing learning on hard misclassified examples. It's particularly
    effective when dealing with extreme class imbalance.
    
    Properties:
    1. Down-weights well-classified examples
    2. Focuses on hard examples
    3. Tunable focusing parameter γ
    4. Optional class weighting α
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma: float = 2.0,
        reduction: str = 'mean',
        label_smoothing: float = 0.0,
        ignore_index: int = -100
    ):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples
                   or a list of weights for each class
            gamma: Exponent of the modulating factor (1 - p_t)^gamma
            reduction: Specifies the reduction to apply to the output
            label_smoothing: Label smoothing factor
            ignore_index: Specifies a target value that is ignored
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        
        # Validate parameters
        if gamma < 0:
            raise ValueError(f"Gamma should be >= 0, got {gamma}")
        
        if label_smoothing < 0 or label_smoothing > 1:
            raise ValueError(f"Label smoothing should be in [0, 1], got {label_smoothing}")
        
        # Process alpha
        if isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        elif isinstance(alpha, torch.Tensor):
            self.alpha = alpha
        elif alpha is not None:
            self.alpha = torch.tensor([alpha])
        
        logger.info(f"Initialized Focal Loss with gamma={gamma}, alpha={alpha}")
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            inputs: Predicted logits of shape (N, C) where N is batch size
                   and C is number of classes
            targets: Ground truth labels of shape (N,) with class indices
            
        Returns:
            Computed focal loss
        """
        # Get dimensions
        n = inputs.shape[0]
        c = inputs.shape[1]
        
        # Filter out ignore_index
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return torch.tensor(0.0, device=inputs.device)
        
        inputs = inputs[valid_mask]
        targets = targets[valid_mask]
        
        # Apply label smoothing if specified
        if self.label_smoothing > 0:
            targets = self._smooth_labels(targets, c)
            return self._focal_loss_with_smoothing(inputs, targets)
        
        # Standard focal loss computation
        # Get class probabilities
        p = F.softmax(inputs, dim=1)
        
        # Get class log probabilities
        ce_loss = F.log_softmax(inputs, dim=1)
        
        # Gather the probabilities at the target indices
        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        ce_loss = ce_loss.gather(1, targets.view(-1, 1)).squeeze(1)
        
        # Calculate focal term: (1 - p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma
        
        # Calculate focal loss
        loss = -focal_term * ce_loss
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_t = self._get_alpha(targets, c)
            loss = alpha_t * loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
    
    def _smooth_labels(
        self,
        targets: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        Apply label smoothing to targets.
        
        Args:
            targets: Original target labels
            num_classes: Number of classes
            
        Returns:
            Smoothed label distribution
        """
        confidence = 1.0 - self.label_smoothing
        smoothing_value = self.label_smoothing / (num_classes - 1)
        
        one_hot = torch.zeros(
            targets.size(0),
            num_classes,
            device=targets.device
        )
        one_hot.fill_(smoothing_value)
        one_hot.scatter_(1, targets.unsqueeze(1), confidence)
        
        return one_hot
    
    def _focal_loss_with_smoothing(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate focal loss with label smoothing.
        
        Args:
            inputs: Predicted logits
            targets: Smoothed target distribution
            
        Returns:
            Focal loss with label smoothing
        """
        # Get probabilities
        p = F.softmax(inputs, dim=1)
        
        # Calculate cross entropy with soft targets
        ce_loss = -(targets * F.log_softmax(inputs, dim=1)).sum(dim=1)
        
        # Calculate focal term using predicted probabilities
        # For soft targets, use the maximum probability as p_t approximation
        p_t = (p * targets).sum(dim=1)
        focal_term = (1 - p_t) ** self.gamma
        
        # Calculate focal loss
        loss = focal_term * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
    
    def _get_alpha(
        self,
        targets: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        Get alpha weights for each sample.
        
        Args:
            targets: Target labels
            num_classes: Number of classes
            
        Returns:
            Alpha weights for each sample
        """
        if self.alpha is None:
            return torch.ones_like(targets, dtype=torch.float32)
        
        if isinstance(self.alpha, torch.Tensor):
            if self.alpha.shape[0] == 1:
                # Single alpha value for binary classification
                alpha_t = torch.where(
                    targets == 1,
                    self.alpha,
                    1 - self.alpha
                )
            else:
                # Per-class alpha values
                alpha = self.alpha.to(targets.device)
                alpha_t = alpha.gather(0, targets)
        else:
            alpha_t = torch.ones_like(targets, dtype=torch.float32) * self.alpha
        
        return alpha_t


class AdaptiveFocalLoss(FocalLoss):
    """
    Adaptive Focal Loss that adjusts gamma based on training progress.
    
    Implements an adaptive version where the focusing parameter gamma
    changes during training, starting high for aggressive focusing
    and decreasing as the model improves.
    """
    
    def __init__(
        self,
        alpha: Optional[Union[float, List[float], torch.Tensor]] = None,
        gamma_init: float = 5.0,
        gamma_final: float = 2.0,
        total_steps: int = 10000,
        reduction: str = 'mean'
    ):
        """
        Initialize Adaptive Focal Loss.
        
        Args:
            alpha: Class weighting factor
            gamma_init: Initial gamma value
            gamma_final: Final gamma value
            total_steps: Total training steps for gamma scheduling
            reduction: Reduction method
        """
        super().__init__(alpha=alpha, gamma=gamma_init, reduction=reduction)
        
        self.gamma_init = gamma_init
        self.gamma_final = gamma_final
        self.total_steps = total_steps
        self.current_step = 0
        
        logger.info(
            f"Initialized Adaptive Focal Loss: "
            f"gamma {gamma_init} -> {gamma_final} over {total_steps} steps"
        )
    
    def update_gamma(self, step: Optional[int] = None):
        """
        Update gamma based on training progress.
        
        Args:
            step: Current training step
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1
        
        # Linear decay
        progress = min(self.current_step / self.total_steps, 1.0)
        self.gamma = self.gamma_init + (self.gamma_final - self.gamma_init) * progress
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with adaptive gamma.
        
        Args:
            inputs: Predicted logits
            targets: Target labels
            
        Returns:
            Adaptive focal loss
        """
        # Update gamma
        self.update_gamma()
        
        # Calculate focal loss with current gamma
        return super().forward(inputs, targets)


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for multi-label classification.
    
    Based on Ridnik et al. (2021): "Asymmetric Loss For Multi-Label Classification"
    
    Applies different focusing parameters for positive and negative samples,
    addressing the positive-negative imbalance in multi-label scenarios.
    """
    
    def __init__(
        self,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = 'mean'
    ):
        """
        Initialize Asymmetric Focal Loss.
        
        Args:
            gamma_pos: Focusing parameter for positive samples
            gamma_neg: Focusing parameter for negative samples
            clip: Probability clipping threshold
            reduction: Reduction method
        """
        super().__init__()
        
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction
        
        logger.info(
            f"Initialized Asymmetric Focal Loss: "
            f"gamma_pos={gamma_pos}, gamma_neg={gamma_neg}"
        )
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate asymmetric focal loss.
        
        Args:
            inputs: Predicted logits
            targets: Target labels (can be soft labels for multi-label)
            
        Returns:
            Asymmetric focal loss
        """
        # Get probabilities
        p = torch.sigmoid(inputs)
        
        # Calculate positive and negative losses separately
        # Positive loss
        p_pos = p[targets == 1]
        if len(p_pos) > 0:
            # Clip probabilities
            p_pos = torch.clamp(p_pos, min=self.clip, max=1.0)
            # Calculate focal term
            focal_weight_pos = (1 - p_pos) ** self.gamma_pos
            # Calculate loss
            loss_pos = -focal_weight_pos * torch.log(p_pos)
        else:
            loss_pos = torch.tensor(0.0, device=inputs.device)
        
        # Negative loss
        p_neg = p[targets == 0]
        if len(p_neg) > 0:
            # Clip probabilities
            p_neg = torch.clamp(p_neg, min=0.0, max=1.0 - self.clip)
            # Calculate focal term
            focal_weight_neg = p_neg ** self.gamma_neg
            # Calculate loss
            loss_neg = -focal_weight_neg * torch.log(1 - p_neg)
        else:
            loss_neg = torch.tensor(0.0, device=inputs.device)
        
        # Combine losses
        loss = torch.cat([loss_pos, loss_neg])
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# Export classes
__all__ = [
    'FocalLoss',
    'AdaptiveFocalLoss',
    'AsymmetricFocalLoss'
]
