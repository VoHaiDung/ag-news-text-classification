"""
Gradient Accumulation Implementation
=====================================

This module implements gradient accumulation for simulating larger batch sizes
with limited memory.

Mathematical Foundation:
------------------------
Effective batch size = actual_batch_size * accumulation_steps
Gradient update: θ = θ - η * (1/N) * Σ∇L_i

References:
- Chen et al. (2016): "Training Deep Nets with Sublinear Memory Cost"
- Ott et al. (2018): "Scaling Neural Machine Translation"

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from contextlib import contextmanager


class GradientAccumulator:
    """
    Gradient accumulation manager for memory-efficient training.
    """
    
    def __init__(
        self,
        accumulation_steps: int = 1,
        max_grad_norm: Optional[float] = None,
        sync_gradients: bool = True
    ):
        """
        Initialize gradient accumulator.
        
        Args:
            accumulation_steps: Number of steps to accumulate
            max_grad_norm: Maximum gradient norm for clipping
            sync_gradients: Whether to sync gradients in distributed training
        """
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.sync_gradients = sync_gradients
        
        self.current_step = 0
        self.accumulated_loss = 0.0
    
    def accumulate(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None
    ) -> bool:
        """
        Accumulate gradients and optionally update model.
        
        Args:
            loss: Current batch loss
            model: Model to update
            optimizer: Optimizer
            scheduler: Optional learning rate scheduler
            
        Returns:
            Whether model was updated
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        scaled_loss.backward()
        
        self.accumulated_loss += loss.item()
        self.current_step += 1
        
        # Check if should update
        if self.current_step >= self.accumulation_steps:
            # Gradient clipping
            if self.max_grad_norm:
                
