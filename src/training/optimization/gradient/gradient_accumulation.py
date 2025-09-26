"""
Gradient Accumulation Implementation for AG News Text Classification
=====================================================================

This module implements gradient accumulation for simulating larger batch sizes
with limited memory, essential for training large transformer models.

Mathematical Foundation:
------------------------
Effective batch size = actual_batch_size * accumulation_steps
Gradient update: θ = θ - η * (1/N) * Σ∇L_i

where N = effective batch size, ∇L_i = gradient for mini-batch i

Memory Complexity:
O(M) where M is model size, independent of effective batch size

References:
- Chen et al. (2016): "Training Deep Nets with Sublinear Memory Cost"
- Ott et al. (2018): "Scaling Neural Machine Translation"
- Huang et al. (2019): "GPipe: Efficient Training of Giant Neural Networks"

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable
from contextlib import contextmanager
import logging
from dataclasses import dataclass

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AccumulationConfig:
    """Configuration for gradient accumulation."""
    
    accumulation_steps: int = 1
    max_grad_norm: Optional[float] = 1.0
    sync_gradients: bool = True
    normalize_gradients: bool = True
    gradient_checkpointing: bool = False
    mixed_precision: bool = False
    log_gradient_norm: bool = False
    clear_cache_steps: int = 100


class GradientAccumulator:
    """
    Gradient accumulation manager for memory-efficient training.
    
    Enables training with larger effective batch sizes by accumulating
    gradients over multiple forward-backward passes before updating weights.
    """
    
    def __init__(
        self,
        config: Optional[AccumulationConfig] = None
    ):
        """
        Initialize gradient accumulator.
        
        Args:
            config: Accumulation configuration
        """
        self.config = config or AccumulationConfig()
        
        # State tracking
        self.current_step = 0
        self.accumulated_loss = 0.0
        self.gradient_norms = []
        self.update_count = 0
        
        # Mixed precision scaler
        self.scaler = None
        if self.config.mixed_precision:
            from torch.cuda.amp import GradScaler
            self.scaler = GradScaler()
    
    def accumulate(
        self,
        loss: torch.Tensor,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        clip_grad_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Accumulate gradients and optionally update model.
        
        Args:
            loss: Current batch loss
            model: Model to update
            optimizer: Optimizer
            scheduler: Optional learning rate scheduler
            clip_grad_callback: Optional custom gradient clipping function
            
        Returns:
            Dictionary with update information
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.config.accumulation_steps
        
        # Backward pass with gradient accumulation
        if self.config.mixed_precision and self.scaler:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()
        
        # Track accumulated loss
        self.accumulated_loss += loss.item()
        self.current_step += 1
        
        # Initialize return info
        info = {
            "accumulated_loss": self.accumulated_loss / self.current_step,
            "current_step": self.current_step,
            "updated": False
        }
        
        # Check if should update
        if self.current_step >= self.config.accumulation_steps:
            # Perform gradient update
            info.update(self._update_model(
                model, optimizer, scheduler, clip_grad_callback
            ))
            info["updated"] = True
            
            # Reset accumulation
            self.reset_accumulation()
            self.update_count += 1
            
            # Clear cache periodically
            if self.update_count % self.config.clear_cache_steps == 0:
                self._clear_cache()
        
        return info
    
    def _update_model(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        clip_grad_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Perform model update with accumulated gradients.
        
        Args:
            model: Model to update
            optimizer: Optimizer
            scheduler: Optional scheduler
            clip_grad_callback: Optional gradient clipping function
            
        Returns:
            Update information
        """
        info = {}
        
        # Unscale gradients if using mixed precision
        if self.config.mixed_precision and self.scaler:
            self.scaler.unscale_(optimizer)
        
        # Normalize gradients if configured
        if self.config.normalize_gradients:
            self._normalize_gradients(model)
        
        # Compute gradient norm before clipping
        if self.config.log_gradient_norm:
            grad_norm_before = self._compute_gradient_norm(model)
            info["grad_norm_before_clip"] = grad_norm_before
        
        # Gradient clipping
        if clip_grad_callback:
            grad_norm = clip_grad_callback(model.parameters())
        elif self.config.max_grad_norm:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.max_grad_norm
            )
        else:
            grad_norm = self._compute_gradient_norm(model)
        
        info["grad_norm"] = grad_norm
        self.gradient_norms.append(grad_norm)
        
        # Optimizer step
        if self.config.mixed_precision and self.scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()
        
        # Learning rate scheduling
        if scheduler:
            scheduler.step()
            info["learning_rate"] = optimizer.param_groups[0]['lr']
        
        # Zero gradients
        optimizer.zero_grad()
        
        return info
    
    def _normalize_gradients(self, model: nn.Module):
        """
        Normalize gradients by effective batch size.
        
        Args:
            model: Model with gradients to normalize
        """
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.div_(self.config.accumulation_steps)
    
    def _compute_gradient_norm(self, model: nn.Module) -> float:
        """
        Compute total gradient norm.
        
        Args:
            model: Model with gradients
            
        Returns:
            Total gradient norm
        """
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm
    
    def _clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def reset_accumulation(self):
        """Reset accumulation state."""
        self.current_step = 0
        self.accumulated_loss = 0.0
    
    def get_effective_batch_size(self, batch_size: int) -> int:
        """
        Calculate effective batch size.
        
        Args:
            batch_size: Actual batch size
            
        Returns:
            Effective batch size after accumulation
        """
        return batch_size * self.config.accumulation_steps
    
    @contextmanager
    def no_sync_context(self, model: nn.Module):
        """
        Context manager for disabling gradient synchronization.
        
        Args:
            model: Model to disable sync for
        """
        if hasattr(model, 'no_sync') and self.config.sync_gradients:
            # For DDP models
            with model.no_sync():
                yield
        else:
            yield
    
    def should_update(self) -> bool:
        """Check if gradients should be updated."""
        return self.current_step >= self.config.accumulation_steps
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get accumulation statistics."""
        stats = {
            "update_count": self.update_count,
            "current_step": self.current_step,
            "accumulated_loss": self.accumulated_loss / max(1, self.current_step),
            "effective_updates": self.update_count,
            "accumulation_steps": self.config.accumulation_steps
        }
        
        if self.gradient_norms:
            stats["avg_gradient_norm"] = sum(self.gradient_norms) / len(self.gradient_norms)
            stats["max_gradient_norm"] = max(self.gradient_norms)
            stats["min_gradient_norm"] = min(self.gradient_norms)
        
        return stats


class DynamicGradientAccumulator(GradientAccumulator):
    """
    Dynamic gradient accumulation with adaptive step adjustment.
    
    Automatically adjusts accumulation steps based on memory usage
    and gradient variance.
    """
    
    def __init__(
        self,
        config: Optional[AccumulationConfig] = None,
        min_steps: int = 1,
        max_steps: int = 32,
        memory_threshold: float = 0.9,
        variance_threshold: float = 0.1
    ):
        """
        Initialize dynamic gradient accumulator.
        
        Args:
            config: Base configuration
            min_steps: Minimum accumulation steps
            max_steps: Maximum accumulation steps
            memory_threshold: GPU memory usage threshold
            variance_threshold: Gradient variance threshold
        """
        super().__init__(config)
        
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.memory_threshold = memory_threshold
        self.variance_threshold = variance_threshold
        
        # Dynamic adjustment state
        self.gradient_variances = []
        self.memory_usage_history = []
    
    def adjust_accumulation_steps(self):
        """Dynamically adjust accumulation steps based on metrics."""
        if not self.gradient_variances:
            return
        
        # Check memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.memory_usage_history.append(memory_used)
            
            # Increase steps if memory usage is high
            if memory_used > self.memory_threshold:
                self.config.accumulation_steps = min(
                    self.config.accumulation_steps * 2,
                    self.max_steps
                )
                logger.info(
                    f"Increased accumulation steps to {self.config.accumulation_steps} "
                    f"due to high memory usage ({memory_used:.2%})"
                )
        
        # Check gradient variance
        recent_variance = torch.var(torch.tensor(self.gradient_norms[-10:]))
        self.gradient_variances.append(recent_variance.item())
        
        # Decrease steps if gradient variance is low (stable training)
        if recent_variance < self.variance_threshold:
            self.config.accumulation_steps = max(
                self.config.accumulation_steps // 2,
                self.min_steps
            )
            logger.info(
                f"Decreased accumulation steps to {self.config.accumulation_steps} "
                f"due to stable gradients (variance: {recent_variance:.4f})"
            )
