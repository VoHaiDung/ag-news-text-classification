"""
Gradient Clipping Implementation for AG News Text Classification
=================================================================

This module implements various gradient clipping strategies to prevent
gradient explosion and ensure stable training.

Mathematical Foundation:
------------------------
Norm clipping: g' = g * min(1, threshold / ||g||)
Value clipping: g' = clip(g, -threshold, threshold)
Adaptive clipping: threshold = percentile(||g||_history, p)

References:
- Pascanu et al. (2013): "On the difficulty of training recurrent neural networks"
- Zhang et al. (2020): "Improved Gradient Clipping for Training Neural Networks"
- Menon et al. (2021): "Adaptive Gradient Clipping"

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple, Callable
import numpy as np
from collections import deque
import logging

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class GradientClipper:
    """
    Base gradient clipping manager with multiple strategies.
    """
    
    def __init__(
        self,
        clip_value: Optional[float] = None,
        clip_norm: Optional[float] = 1.0,
        norm_type: float = 2.0,
        adaptive: bool = False,
        history_size: int = 100
    ):
        """
        Initialize gradient clipper.
        
        Args:
            clip_value: Maximum absolute value for gradients
            clip_norm: Maximum norm for gradients
            norm_type: Type of norm (1, 2, or inf)
            adaptive: Whether to use adaptive clipping
            history_size: Size of gradient norm history for adaptive clipping
        """
        self.clip_value = clip_value
        self.clip_norm = clip_norm
        self.norm_type = norm_type
        self.adaptive = adaptive
        self.history_size = history_size
        
        # History for adaptive clipping
        self.norm_history = deque(maxlen=history_size)
        self.clip_history = deque(maxlen=history_size)
        self.step_count = 0
    
    def clip_gradients(
        self,
        parameters: Union[torch.Tensor, List[torch.Tensor]],
        aggregate_norm_fn: Optional[Callable] = None
    ) -> Tuple[float, float]:
        """
        Clip gradients using configured strategy.
        
        Args:
            parameters: Model parameters or gradients
            aggregate_norm_fn: Custom function to aggregate norms
            
        Returns:
            Tuple of (original_norm, clipped_norm)
        """
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()
        
        # Compute original norm
        original_norm = self._compute_norm(parameters, aggregate_norm_fn)
        self.norm_history.append(original_norm)
        
        # Apply clipping strategy
        if self.adaptive:
            clipped_norm = self._adaptive_clip(parameters, original_norm)
        elif self.clip_norm is not None:
            clipped_norm = self._norm_clip(parameters, original_norm)
        elif self.clip_value is not None:
            clipped_norm = self._value_clip(parameters)
        else:
            clipped_norm = original_norm
        
        self.clip_history.append(clipped_norm)
        self.step_count += 1
        
        return original_norm, clipped_norm
    
    def _compute_norm(
        self,
        parameters: List[torch.Tensor],
        aggregate_norm_fn: Optional[Callable] = None
    ) -> float:
        """
        Compute gradient norm.
        
        Args:
            parameters: Model parameters
            aggregate_norm_fn: Custom aggregation function
            
        Returns:
            Computed norm
        """
        if aggregate_norm_fn:
            return aggregate_norm_fn(parameters)
        
        # Standard norm computation
        if self.norm_type == float('inf'):
            total_norm = max(p.grad.data.abs().max() for p in parameters 
                           if p.grad is not None)
        else:
            total_norm = 0.0
            for p in parameters:
                if p.grad is not None:
                    param_norm = p.grad.data.norm(self.norm_type)
                    total_norm += param_norm.item() ** self.norm_type
            total_norm = total_norm ** (1. / self.norm_type)
        
        return total_norm
    
    def _norm_clip(
        self,
        parameters: List[torch.Tensor],
        original_norm: float
    ) -> float:
        """
        Apply norm-based gradient clipping.
        
        Args:
            parameters: Model parameters
            original_norm: Original gradient norm
            
        Returns:
            Clipped norm
        """
        clip_coef = self.clip_norm / (original_norm + 1e-6)
        
        if clip_coef < 1:
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
            return self.clip_norm
        
        return original_norm
    
    def _value_clip(
        self,
        parameters: List[torch.Tensor]
    ) -> float:
        """
        Apply value-based gradient clipping.
        
        Args:
            parameters: Model parameters
            
        Returns:
            Norm after clipping
        """
        for p in parameters:
            if p.grad is not None:
                p.grad.data.clamp_(-self.clip_value, self.clip_value)
        
        return self._compute_norm(parameters)
    
    def _adaptive_clip(
        self,
        parameters: List[torch.Tensor],
        original_norm: float
    ) -> float:
        """
        Apply adaptive gradient clipping.
        
        Args:
            parameters: Model parameters
            original_norm: Original gradient norm
            
        Returns:
            Clipped norm
        """
        if len(self.norm_history) < 10:
            # Use fixed clipping for initial steps
            return self._norm_clip(parameters, original_norm)
        
        # Compute adaptive threshold
        percentile = 90
        threshold = np.percentile(list(self.norm_history), percentile)
        
        # Apply adaptive clipping
        if original_norm > threshold:
            clip_coef = threshold / (original_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)
            return threshold
        
        return original_norm
    
    def get_statistics(self) -> dict:
        """Get clipping statistics."""
        if not self.norm_history:
            return {}
        
        return {
            "avg_gradient_norm": np.mean(list(self.norm_history)),
            "max_gradient_norm": np.max(list(self.norm_history)),
            "min_gradient_norm": np.min(list(self.norm_history)),
            "clip_rate": sum(c < o for o, c in 
                           zip(self.norm_history, self.clip_history)) / len(self.norm_history),
            "total_steps": self.step_count
        }


class PerLayerGradientClipper(GradientClipper):
    """
    Per-layer gradient clipping for fine-grained control.
    """
    
    def __init__(
        self,
        layer_clip_norms: Optional[dict] = None,
        default_clip_norm: float = 1.0,
        **kwargs
    ):
        """
        Initialize per-layer gradient clipper.
        
        Args:
            layer_clip_norms: Dictionary of layer names to clip norms
            default_clip_norm: Default clip norm for unnamed layers
            **kwargs: Additional arguments for base class
        """
        super().__init__(clip_norm=default_clip_norm, **kwargs)
        self.layer_clip_norms = layer_clip_norms or {}
        self.layer_statistics = {}
    
    def clip_gradients_per_layer(
        self,
        model: nn.Module
    ) -> dict:
        """
        Clip gradients per layer.
        
        Args:
            model: Model with gradients
            
        Returns:
            Dictionary of layer clipping statistics
        """
        stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            # Get layer-specific clip norm
            clip_norm = self.layer_clip_norms.get(name, self.clip_norm)
            
            # Compute and clip gradient norm
            grad_norm = param.grad.data.norm(self.norm_type)
            
            if grad_norm > clip_norm:
                param.grad.data.mul_(clip_norm / grad_norm)
                clipped_norm = clip_norm
            else:
                clipped_norm = grad_norm
            
            # Record statistics
            stats[name] = {
                "original_norm": grad_norm.item(),
                "clipped_norm": clipped_norm,
                "clipped": grad_norm > clip_norm
            }
            
            # Update layer statistics
            if name not in self.layer_statistics:
                self.layer_statistics[name] = []
            self.layer_statistics[name].append(grad_norm.item())
        
        return stats
