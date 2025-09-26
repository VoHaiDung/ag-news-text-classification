"""
Label Smoothing Cross Entropy Loss Implementation
==================================================

This module implements label smoothing regularization for classification tasks,
which prevents the model from becoming over-confident in its predictions.

Mathematical Foundation:
------------------------
Standard cross-entropy loss:
L_ce = -∑_i y_i log(p_i)

Label smoothing loss:
L_ls = (1 - ε) * L_ce + ε * L_uniform
where ε is the smoothing parameter and L_uniform is uniform distribution loss.

References:
- Müller et al. (2019): "When Does Label Smoothing Help?"
- Szegedy et al. (2016): "Rethinking the Inception Architecture for Computer Vision"
- Pereyra et al. (2017): "Regularizing Neural Networks by Penalizing Confident Output"

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss.
    
    This loss function implements label smoothing by replacing the hard targets
    with a mixture of the original ground truth and uniform distribution.
    
    The smoothed label distribution is:
    y_smooth = (1 - ε) * y_onehot + ε / K
    where K is the number of classes.
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        reduction: str = 'mean',
        ignore_index: int = -100,
        dim: int = -1
    ):
        """
        Initialize Label Smoothing Cross Entropy Loss.
        
        Args:
            smoothing: Label smoothing factor (ε), typically 0.1
            reduction: Specifies reduction to apply to output ('none', 'mean', 'sum')
            ignore_index: Index to ignore in loss computation
            dim: Dimension along which to compute softmax
        """
        super().__init__()
        
        assert 0 <= smoothing < 1, f"smoothing value should be in [0, 1), got {smoothing}"
        
        self.smoothing = smoothing
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.dim = dim
        self.confidence = 1.0 - smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute label smoothing cross entropy loss.
        
        Args:
            inputs: Predicted logits of shape (N, C) or (N, C, H, W)
            targets: Ground truth labels of shape (N,) or (N, H, W)
            weight: Manual rescaling weight for each class
            
        Returns:
            Computed loss value
        """
        batch_size = inputs.size(0)
        num_classes = inputs.size(self.dim)
        
        # Handle different input dimensions
        if inputs.dim() > 2:
            # For segmentation tasks (N, C, H, W) -> (N*H*W, C)
            inputs = inputs.transpose(1, self.dim).contiguous()
            inputs = inputs.view(-1, num_classes)
            targets = targets.view(-1)
        
        # Create mask for valid targets
        mask = targets != self.ignore_index
        valid_targets = targets[mask]
        
        if valid_targets.numel() == 0:
            # No valid targets, return zero loss
            return torch.tensor(0.0, device=inputs.device, requires_grad=True)
        
        valid_inputs = inputs[mask]
        
        # Compute log probabilities
        log_probs = F.log_softmax(valid_inputs, dim=self.dim)
        
        # Create smoothed label distribution
        smooth_targets = self._smooth_targets(valid_targets, num_classes, valid_inputs.device)
        
        # Compute loss
        loss = -(smooth_targets * log_probs).sum(dim=self.dim)
        
        # Apply class weights if provided
        if weight is not None:
            weight_expanded = weight[valid_targets]
            loss = loss * weight_expanded
        
        # Apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        # else: reduction == 'none', return per-sample loss
        
        return loss
    
    def _smooth_targets(
        self,
        targets: torch.Tensor,
        num_classes: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create smoothed target distribution.
        
        Args:
            targets: Ground truth labels
            num_classes: Number of classes
            device: Device to create tensor on
            
        Returns:
            Smoothed target distribution
        """
        # Create one-hot encoding
        targets_onehot = torch.zeros(
            targets.size(0), num_classes, device=device
        )
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)
        
        # Apply label smoothing
        smooth_targets = targets_onehot * self.confidence + self.smoothing / num_classes
        
        return smooth_targets


class AdaptiveLabelSmoothing(nn.Module):
    """
    Adaptive Label Smoothing that adjusts smoothing based on model confidence.
    
    This variant dynamically adjusts the smoothing parameter based on the
    model's prediction confidence, applying more smoothing when the model
    is overconfident.
    """
    
    def __init__(
        self,
        base_smoothing: float = 0.1,
        confidence_threshold: float = 0.9,
        max_smoothing: float = 0.3,
        temperature: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Initialize Adaptive Label Smoothing.
        
        Args:
            base_smoothing: Base smoothing factor
            confidence_threshold: Threshold for adjusting smoothing
            max_smoothing: Maximum smoothing to apply
            temperature: Temperature for confidence scaling
            reduction: Reduction method
        """
        super().__init__()
        
        self.base_smoothing = base_smoothing
        self.confidence_threshold = confidence_threshold
        self.max_smoothing = max_smoothing
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive label smoothing loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
            
        Returns:
            Computed loss value
        """
        # Compute prediction confidence
        probs = F.softmax(inputs / self.temperature, dim=-1)
        max_probs, _ = probs.max(dim=-1)
        
        # Compute adaptive smoothing factor
        smoothing = self._compute_adaptive_smoothing(max_probs)
        
        # Apply label smoothing with adaptive factor
        num_classes = inputs.size(-1)
        targets_onehot = F.one_hot(targets, num_classes).float()
        
        smooth_targets = targets_onehot * (1 - smoothing).unsqueeze(-1) + \
                        smoothing.unsqueeze(-1) / num_classes
        
        # Compute loss
        log_probs = F.log_softmax(inputs, dim=-1)
        loss = -(smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    
    def _compute_adaptive_smoothing(self, confidence: torch.Tensor) -> torch.Tensor:
        """
        Compute adaptive smoothing factor based on confidence.
        
        Args:
            confidence: Model confidence scores
            
        Returns:
            Adaptive smoothing factors
        """
        # High confidence -> more smoothing
        over_confident = confidence > self.confidence_threshold
        
        smoothing = torch.full_like(confidence, self.base_smoothing)
        
        if over_confident.any():
            # Scale smoothing based on confidence
            scale = (confidence[over_confident] - self.confidence_threshold) / \
                   (1.0 - self.confidence_threshold)
            additional_smoothing = scale * (self.max_smoothing - self.base_smoothing)
            smoothing[over_confident] = self.base_smoothing + additional_smoothing
        
        return smoothing


class TemporalLabelSmoothing(nn.Module):
    """
    Temporal Label Smoothing that adjusts smoothing over training time.
    
    This variant gradually reduces label smoothing as training progresses,
    allowing the model to become more confident in later stages.
    """
    
    def __init__(
        self,
        initial_smoothing: float = 0.2,
        final_smoothing: float = 0.05,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        schedule: str = 'linear'  # linear, cosine, exponential
    ):
        """
        Initialize Temporal Label Smoothing.
        
        Args:
            initial_smoothing: Initial smoothing factor
            final_smoothing: Final smoothing factor
            warmup_epochs: Number of warmup epochs
            total_epochs: Total training epochs
            schedule: Smoothing schedule type
        """
        super().__init__()
        
        self.initial_smoothing = initial_smoothing
        self.final_smoothing = final_smoothing
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.schedule = schedule
        self.current_epoch = 0
    
    def set_epoch(self, epoch: int):
        """Update current epoch for temporal adjustment."""
        self.current_epoch = epoch
    
    def get_current_smoothing(self) -> float:
        """Get current smoothing factor based on epoch."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            return self.initial_smoothing * (self.current_epoch + 1) / self.warmup_epochs
        
        # Compute progress after warmup
        progress = (self.current_epoch - self.warmup_epochs) / \
                  (self.total_epochs - self.warmup_epochs)
        progress = min(1.0, progress)
        
        if self.schedule == 'linear':
            smoothing = self.initial_smoothing - \
                       (self.initial_smoothing - self.final_smoothing) * progress
        elif self.schedule == 'cosine':
            import math
            smoothing = self.final_smoothing + \
                       (self.initial_smoothing - self.final_smoothing) * \
                       (1 + math.cos(math.pi * progress)) / 2
        elif self.schedule == 'exponential':
            decay_rate = -math.log(self.final_smoothing / self.initial_smoothing)
            smoothing = self.initial_smoothing * math.exp(-decay_rate * progress)
        else:
            smoothing = self.initial_smoothing
        
        return smoothing
    
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal label smoothing loss.
        
        Args:
            inputs: Predicted logits
            targets: Ground truth labels
            
        Returns:
            Computed loss value
        """
        current_smoothing = self.get_current_smoothing()
        
        # Apply standard label smoothing with current factor
        loss_fn = LabelSmoothingCrossEntropy(smoothing=current_smoothing)
        return loss_fn(inputs, targets)
