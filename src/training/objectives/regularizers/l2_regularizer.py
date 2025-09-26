"""
L2 Regularization Implementation for Neural Networks
=====================================================

This module implements various L2 regularization techniques for
preventing overfitting and improving generalization.

Mathematical Foundation:
------------------------
L2 Regularization: R(θ) = λ/2 * ||θ||²₂
Adaptive L2: R(θ) = Σᵢ λᵢ/2 * ||θᵢ||²₂

References:
- Krogh & Hertz (1992): "A Simple Weight Decay Can Improve Generalization"
- Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization"
- Zhang et al. (2018): "Three Mechanisms of Weight Decay Regularization"

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union


class L2Regularizer(nn.Module):
    """
    Standard L2 Regularization (Weight Decay).
    
    Adds penalty term proportional to squared L2 norm of parameters.
    """
    
    def __init__(
        self,
        lambda_reg: float = 0.01,
        exclude_bias: bool = True,
        exclude_norm: bool = True
    ):
        """
        Initialize L2 Regularizer.
        
        Args:
            lambda_reg: Regularization strength
            exclude_bias: Whether to exclude bias terms
            exclude_norm: Whether to exclude normalization layer parameters
        """
        super().__init__()
        
        self.lambda_reg = lambda_reg
        self.exclude_bias = exclude_bias
        self.exclude_norm = exclude_norm
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute L2 regularization term.
        
        Args:
            model: Model to regularize
            
        Returns:
            Regularization loss
        """
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Skip bias terms if configured
            if self.exclude_bias and 'bias' in name:
                continue
            
            # Skip normalization layers if configured
            if self.exclude_norm and any(
                norm in name for norm in ['LayerNorm', 'BatchNorm', 'GroupNorm']
            ):
                continue
            
            # Add L2 penalty
            reg_loss += torch.sum(param ** 2)
        
        return self.lambda_reg * 0.5 * reg_loss


class AdaptiveL2Regularizer(nn.Module):
    """
    Adaptive L2 Regularization with layer-specific penalties.
    
    Applies different regularization strengths to different layers
    based on their characteristics.
    """
    
    def __init__(
        self,
        base_lambda: float = 0.01,
        layer_decay: float = 0.75,
        min_lambda: float = 1e-5
    ):
        """
        Initialize Adaptive L2 Regularizer.
        
        Args:
            base_lambda: Base regularization strength
            layer_decay: Decay factor for deeper layers
            min_lambda: Minimum regularization strength
        """
        super().__init__()
        
        self.base_lambda = base_lambda
        self.layer_decay = layer_decay
        self.min_lambda = min_lambda
        self.layer_lambdas = {}
    
    def compute_layer_lambdas(self, model: nn.Module):
        """Compute layer-specific regularization strengths."""
        # Identify layers and their depths
        layer_groups = self._group_layers(model)
        
        for depth, layer_names in enumerate(layer_groups):
            lambda_val = max(
                self.base_lambda * (self.layer_decay ** depth),
                self.min_lambda
            )
            for name in layer_names:
                self.layer_lambdas[name] = lambda_val
    
    def _group_layers(self, model: nn.Module) -> List[List[str]]:
        """Group parameters by layer depth."""
        # Simplified grouping - in practice, use model-specific logic
        groups = []
        current_group = []
        
        for name, _ in model.named_parameters():
            if 'weight' in name:
                current_group.append(name)
            elif current_group:
                groups.append(current_group)
                current_group = []
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        """
        Compute adaptive L2 regularization.
        
        Args:
            model: Model to regularize
            
        Returns:
            Regularization loss
        """
        if not self.layer_lambdas:
            self.compute_layer_lambdas(model)
        
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, param in model.named_parameters():
            if not param.requires_grad or 'bias' in name:
                continue
            
            # Get layer-specific lambda
            lambda_val = self.layer_lambdas.get(name, self.base_lambda)
            
            # Add weighted L2 penalty
            reg_loss += lambda_val * 0.5 * torch.sum(param ** 2)
        
        return reg_loss
