"""
R-Drop Regularization Implementation
=====================================

This module implements R-Drop (Regularized Dropout) for consistent predictions
with different dropout masks during training.

Mathematical Foundation:
------------------------
R-Drop Loss: L = L_ce(y, p1) + L_ce(y, p2) + α * KL(p1 || p2)
where p1, p2 are predictions with different dropout masks.

References:
- Liang et al. (2021): "R-Drop: Regularized Dropout for Neural Networks"
- Wu et al. (2021): "R-Drop: Regularized Dropout for Neural Networks"

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any


class RDropLoss(nn.Module):
    """
    R-Drop Loss for regularizing model predictions with dropout.
    
    Enforces consistency between predictions from different dropout masks
    by minimizing KL divergence between them.
    """
    
    def __init__(
        self,
        alpha: float = 4.0,
        reduction: str = 'mean'
    ):
        """
        Initialize R-Drop Loss.
        
        Args:
            alpha: Weight for KL divergence term
            reduction: Loss reduction method
        """
        super().__init__()
        
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute R-Drop loss.
        
        Args:
            logits1: First forward pass logits (N, C)
            logits2: Second forward pass logits (N, C)
            targets: Ground truth labels (N,)
            
        Returns:
            Total loss and component losses
        """
        # Cross-entropy losses
        ce_loss1 = F.cross_entropy(logits1, targets, reduction=self.reduction)
        ce_loss2 = F.cross_entropy(logits2, targets, reduction=self.reduction)
        ce_loss = (ce_loss1 + ce_loss2) / 2
        
        # KL divergence between two predictions
        kl_loss = self.compute_kl_loss(logits1, logits2)
        
        # Total loss
        total_loss = ce_loss + self.alpha * kl_loss
        
        return total_loss, {
            'ce_loss': ce_loss,
            'kl_loss': kl_loss,
            'total_loss': total_loss
        }
    
    def compute_kl_loss(
        self,
        logits1: torch.Tensor,
        logits2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute symmetric KL divergence.
        
        Args:
            logits1: First logits
            logits2: Second logits
            
        Returns:
            KL divergence loss
        """
        # Convert to probabilities
        probs1 = F.softmax(logits1, dim=-1)
        probs2 = F.softmax(logits2, dim=-1)
        
        # Symmetric KL divergence
        kl_1_2 = F.kl_div(
            F.log_softmax(logits1, dim=-1),
            probs2,
            reduction='none'
        ).sum(dim=-1)
        
        kl_2_1 = F.kl_div(
            F.log_softmax(logits2, dim=-1),
            probs1,
            reduction='none'
        ).sum(dim=-1)
        
        kl_loss = (kl_1_2 + kl_2_1) / 2
        
        if self.reduction == 'mean':
            return kl_loss.mean()
        elif self.reduction == 'sum':
            return kl_loss.sum()
        return kl_loss


class RDropRegularizer:
    """
    R-Drop Regularizer for training with consistent dropout predictions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        alpha: float = 4.0,
        warmup_steps: int = 0
    ):
        """
        Initialize R-Drop Regularizer.
        
        Args:
            model: Model to regularize
            alpha: KL divergence weight
            warmup_steps: Steps before applying R-Drop
        """
        self.model = model
        self.alpha = alpha
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.loss_fn = RDropLoss(alpha=alpha)
    
    def forward_with_rdrop(
        self,
        inputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with R-Drop regularization.
        
        Args:
            inputs: Model inputs
            targets: Ground truth labels
            
        Returns:
            Loss and additional metrics
        """
        self.current_step += 1
        
        # Check if should apply R-Drop
        if self.current_step <= self.warmup_steps:
            outputs = self.model(**inputs)
            loss = F.cross_entropy(outputs.logits, targets)
            return loss, {'loss': loss}
        
        # First forward pass
        outputs1 = self.model(**inputs)
        logits1 = outputs1.logits
        
        # Second forward pass with different dropout
        outputs2 = self.model(**inputs)
        logits2 = outputs2.logits
        
        # Compute R-Drop loss
        loss, loss_dict = self.loss_fn(logits1, logits2, targets)
        
        return loss, loss_dict
