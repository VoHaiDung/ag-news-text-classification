"""
Cyclic Learning Rate Schedulers
================================

Implementation of cyclic learning rate policies for neural network training,
based on:
- Smith (2015): "Cyclical Learning Rates for Training Neural Networks"
- Smith (2017): "Super-Convergence: Very Fast Training Using Large Learning Rates"
- Loshchilov & Hutter (2016): "SGDR: Stochastic Gradient Descent with Warm Restarts"

These schedulers implement various cyclic policies that help:
1. Escape local minima through periodic LR increases
2. Achieve super-convergence with appropriate ranges
3. Improve generalization through exploration

Author: Võ Hải Dũng
License: MIT
"""

import math
import logging
from typing import Optional, List, Union, Callable, Dict, Any
from enum import Enum

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import numpy as np

logger = logging.getLogger(__name__)


class CyclicMode(Enum):
    """Cyclic learning rate modes."""
    TRIANGULAR = "triangular"
    TRIANGULAR2 = "triangular2"
    EXP_RANGE = "exp_range"
    CUSTOM = "custom"


class CyclicLR(_LRScheduler):
    """
    Cyclic Learning Rate scheduler.
    
    Implements the cyclical learning rate policy from Leslie Smith's paper.
    The learning rate oscillates between base_lr and max_lr following
    different patterns.
    
    This helps training by:
    - Allowing the model to explore different regions of loss landscape
    - Preventing overfitting to specific minima
    - Potentially achieving super-convergence
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: Union[float, List[float]] = 1e-5,
        max_lr: Union[float, List[float]] = 1e-3,
        step_size_up: int = 2000,
        step_size_down: Optional[int] = None,
        mode: Union[str, CyclicMode] = CyclicMode.TRIANGULAR,
        gamma: float = 1.0,
        scale_fn: Optional[Callable] = None,
        scale_mode: str = 'cycle',
        cycle_momentum: bool = True,
        base_momentum: Union[float, List[float]] = 0.8,
        max_momentum: Union[float, List[float]] = 0.9,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize cyclic learning rate scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            base_lr: Lower learning rate boundaries
            max_lr: Upper learning rate boundaries
            step_size_up: Number of steps in the increasing half of cycle
            step_size_down: Number of steps in the decreasing half of cycle
            mode: One of {triangular, triangular2, exp_range}
            gamma: Constant for 'exp_range' mode
            scale_fn: Custom scaling function
            scale_mode: {'cycle', 'iterations'}
            cycle_momentum: Whether to cycle momentum inversely to LR
            base_momentum: Lower momentum boundaries
            max_momentum: Upper momentum boundaries
            last_epoch: Index of last epoch
            verbose: Whether to print updates
        """
        self.optimizer = optimizer
        
        # Handle single values
        if isinstance(base_lr, float):
            self.base_lrs = [base_lr] * len(optimizer.param_groups)
        else:
            self.base_lrs = list(base_lr)
            
        if isinstance(max_lr, float):
            self.max_lrs = [max_lr] * len(optimizer.param_groups)
        else:
            self.max_lrs = list(max_lr)
        
        # Step sizes
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.total_size = self.step_size_up + self.step_size_down
        self.step_ratio = self.step_size_up / self.total_size
        
        # Mode and scaling
        if isinstance(mode, str):
            mode = CyclicMode(mode)
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        
        # Momentum cycling
        self.cycle_momentum = cycle_momentum
        if isinstance(base_momentum, float):
            self.base_momentums = [base_momentum] * len(optimizer.param_groups)
        else:
            self.base_momentums = list(base_momentum)
            
        if isinstance(max_momentum, float):
            self.max_momentums = [max_momentum] * len(optimizer.param_groups)
        else:
            self.max_momentums = list(max_momentum)
        
        # Initialize
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized CyclicLR: mode={mode.value}, "
            f"base_lr={base_lr}, max_lr={max_lr}, "
            f"step_sizes=({step_size_up}, {step_size_down})"
        )
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step"""
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle
        
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)
        
        # Apply mode-specific scaling
        base_height = self._get_scale(cycle)
        
        lrs = []
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height_lr = (max_lr - base_lr) * base_height
            lr = base_lr + base_height_lr * scale_factor
            lrs.append(lr)
        
        # Update momentum if cycling
        if self.cycle_momentum:
            self._update_momentum(scale_factor)
        
        return lrs
    
    def _get_scale(self, cycle: int) -> float:
        """
        Get scaling factor based on mode.
        
        Args:
            cycle: Current cycle number
            
        Returns:
            Scale factor
        """
        if self.mode == CyclicMode.TRIANGULAR:
            return 1.0
        elif self.mode == CyclicMode.TRIANGULAR2:
            return 1 / (2 ** (cycle - 1))
        elif self.mode == CyclicMode.EXP_RANGE:
            return self.gamma ** self.last_epoch
        elif self.mode == CyclicMode.CUSTOM and self.scale_fn:
            if self.scale_mode == 'cycle':
                return self.scale_fn(cycle)
            else:
                return self.scale_fn(self.last_epoch)
        else:
            return 1.0
    
    def _update_momentum(self, scale_factor: float):
        """
        Update momentum inversely to learning rate.
        
        Args:
            scale_factor: Current scale factor
        """
        for param_group, base_mom, max_mom in zip(
            self.optimizer.param_groups,
            self.base_momentums,
            self.max_momentums
        ):
            if 'momentum' in param_group:
                # Inverse relationship: high LR -> low momentum
                momentum = max_mom - (max_mom - base_mom) * scale_factor
                param_group['momentum'] = momentum
            elif 'betas' in param_group:
                # For Adam-like optimizers
                beta1 = max_mom - (max_mom - base_mom) * scale_factor
                param_group['betas'] = (beta1, param_group['betas'][1])


class TriangularScheduler(_LRScheduler):
    """
    Triangular learning rate scheduler with warmup.
    
    Simplified version of CyclicLR with linear triangular waves
    and optional warmup period.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-5,
        max_lr: float = 1e-3,
        step_size: int = 1000,
        warmup_steps: int = 0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize triangular scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            step_size: Half cycle size
            warmup_steps: Linear warmup steps
            last_epoch: Index of last epoch
            verbose: Whether to print updates
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.warmup_steps = warmup_steps
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized TriangularScheduler: "
            f"[{base_lr}, {max_lr}], step_size={step_size}"
        )
    
    def get_lr(self) -> List[float]:
        """Calculate triangular learning rate"""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            lr = self.base_lr + (self.max_lr - self.base_lr) * warmup_factor
        else:
            # Triangular wave
            cycle_progress = (self.last_epoch - self.warmup_steps) / (2 * self.step_size)
            cycle = math.floor(cycle_progress)
            x = cycle_progress - cycle
            
            if x < 0.5:
                # Increasing phase
                lr = self.base_lr + (self.max_lr - self.base_lr) * (2 * x)
            else:
                # Decreasing phase
                lr = self.max_lr - (self.max_lr - self.base_lr) * (2 * (x - 0.5))
        
        return [lr for _ in self.base_lrs]


class ExponentialCyclicLR(_LRScheduler):
    """
    Exponential cyclic learning rate scheduler.
    
    Combines cyclic behavior with exponential decay for
    gradually diminishing cycles.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-5,
        max_lr: float = 1e-3,
        step_size: int = 1000,
        decay_rate: float = 0.99,
        warmup_steps: int = 0,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize exponential cyclic scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            base_lr: Initial minimum learning rate
            max_lr: Initial maximum learning rate
            step_size: Half cycle size
            decay_rate: Exponential decay per cycle
            warmup_steps: Linear warmup steps
            min_lr: Absolute minimum learning rate
            last_epoch: Index of last epoch
            verbose: Whether to print updates
        """
        self.initial_base_lr = base_lr
        self.initial_max_lr = max_lr
        self.step_size = step_size
        self.decay_rate = decay_rate
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized ExponentialCyclicLR: "
            f"decay_rate={decay_rate}, step_size={step_size}"
        )
    
    def get_lr(self) -> List[float]:
        """Calculate exponentially decaying cyclic learning rate"""
        if self.last_epoch < self.warmup_steps:
            # Warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            lr = self.initial_base_lr + (self.initial_max_lr - self.initial_base_lr) * warmup_factor
        else:
            # Cyclic with exponential decay
            adjusted_epoch = self.last_epoch - self.warmup_steps
            cycle = adjusted_epoch // (2 * self.step_size)
            cycle_progress = (adjusted_epoch % (2 * self.step_size)) / (2 * self.step_size)
            
            # Apply exponential decay to range
            decay_factor = self.decay_rate ** cycle
            current_base = max(self.min_lr, self.initial_base_lr * decay_factor)
            current_max = max(current_base, self.initial_max_lr * decay_factor)
            
            # Triangular wave within decayed range
            if cycle_progress < 0.5:
                lr = current_base + (current_max - current_base) * (2 * cycle_progress)
            else:
                lr = current_max - (current_max - current_base) * (2 * (cycle_progress - 0.5))
        
        return [lr for _ in self.base_lrs]


class SinusoidalLR(_LRScheduler):
    """
    Sinusoidal learning rate scheduler.
    
    Uses smooth sinusoidal waves instead of triangular waves
    for more gradual transitions.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: float = 1e-5,
        max_lr: float = 1e-3,
        period: int = 1000,
        warmup_steps: int = 0,
        phase_shift: float = 0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize sinusoidal scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            period: Period of sinusoidal wave
            warmup_steps: Linear warmup steps
            phase_shift: Phase shift in radians
            last_epoch: Index of last epoch
            verbose: Whether to print updates
        """
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.period = period
        self.warmup_steps = warmup_steps
        self.phase_shift = phase_shift
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized SinusoidalLR: "
            f"[{base_lr}, {max_lr}], period={period}"
        )
    
    def get_lr(self) -> List[float]:
        """Calculate sinusoidal learning rate"""
        if self.last_epoch < self.warmup_steps:
            # Warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            lr = self.base_lr + (self.max_lr - self.base_lr) * warmup_factor
        else:
            # Sinusoidal wave
            adjusted_epoch = self.last_epoch - self.warmup_steps
            angle = 2 * math.pi * adjusted_epoch / self.period + self.phase_shift
            amplitude = (self.max_lr - self.base_lr) / 2
            lr = self.base_lr + amplitude * (1 + math.sin(angle))
        
        return [lr for _ in self.base_lrs]


class AdaptiveCyclicLR(_LRScheduler):
    """
    Adaptive cyclic learning rate scheduler.
    
    Automatically adjusts cycle amplitude based on training progress
    and loss improvements.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        initial_base_lr: float = 1e-5,
        initial_max_lr: float = 1e-3,
        step_size: int = 1000,
        adaptation_rate: float = 0.1,
        min_amplitude: float = 1e-6,
        patience: int = 10,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize adaptive cyclic scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            initial_base_lr: Initial minimum learning rate
            initial_max_lr: Initial maximum learning rate
            step_size: Half cycle size
            adaptation_rate: Rate of amplitude adjustment
            min_amplitude: Minimum amplitude threshold
            patience: Cycles without improvement before reduction
            last_epoch: Index of last epoch
            verbose: Whether to print updates
        """
        self.base_lr = initial_base_lr
        self.max_lr = initial_max_lr
        self.step_size = step_size
        self.adaptation_rate = adaptation_rate
        self.min_amplitude = min_amplitude
        self.patience = patience
        
        # Tracking
        self.current_cycle = 0
        self.best_loss = float('inf')
        self.cycles_without_improvement = 0
        self.amplitude = initial_max_lr - initial_base_lr
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized AdaptiveCyclicLR: "
            f"adaptation_rate={adaptation_rate}"
        )
    
    def get_lr(self) -> List[float]:
        """Calculate adaptive cyclic learning rate"""
        # Determine current cycle
        cycle = self.last_epoch // (2 * self.step_size)
        if cycle > self.current_cycle:
            self._adapt_amplitude()
            self.current_cycle = cycle
        
        # Calculate position in cycle
        cycle_progress = (self.last_epoch % (2 * self.step_size)) / (2 * self.step_size)
        
        # Triangular wave with current amplitude
        if cycle_progress < 0.5:
            lr = self.base_lr + self.amplitude * (2 * cycle_progress)
        else:
            lr = self.base_lr + self.amplitude - self.amplitude * (2 * (cycle_progress - 0.5))
        
        return [lr for _ in self.base_lrs]
    
    def update_loss(self, loss: float):
        """
        Update scheduler with current loss.
        
        Args:
            loss: Current training loss
        """
        if loss < self.best_loss:
            self.best_loss = loss
            self.cycles_without_improvement = 0
        else:
            self.cycles_without_improvement += 1
    
    def _adapt_amplitude(self):
        """Adapt cycle amplitude based on training progress"""
        if self.cycles_without_improvement >= self.patience:
            # Reduce amplitude
            self.amplitude *= (1 - self.adaptation_rate)
            self.amplitude = max(self.min_amplitude, self.amplitude)
            self.cycles_without_improvement = 0
            
            logger.info(f"Reduced cycle amplitude to {self.amplitude:.2e}")
        
        # Update max_lr for tracking
        self.max_lr = self.base_lr + self.amplitude


def create_cyclic_scheduler(
    optimizer: Optimizer,
    scheduler_config: Dict[str, Any]
) -> _LRScheduler:
    """
    Factory function for creating cyclic schedulers.
    
    Args:
        optimizer: Optimizer to wrap
        scheduler_config: Configuration dictionary
        
    Returns:
        Configured cyclic scheduler
    """
    scheduler_type = scheduler_config.get('type', 'cyclic')
    
    if scheduler_type == 'cyclic':
        return CyclicLR(
            optimizer,
            base_lr=scheduler_config.get('base_lr', 1e-5),
            max_lr=scheduler_config.get('max_lr', 1e-3),
            step_size_up=scheduler_config.get('step_size_up', 2000),
            step_size_down=scheduler_config.get('step_size_down'),
            mode=scheduler_config.get('mode', 'triangular'),
            gamma=scheduler_config.get('gamma', 1.0),
            cycle_momentum=scheduler_config.get('cycle_momentum', True)
        )
    elif scheduler_type == 'triangular':
        return TriangularScheduler(
            optimizer,
            base_lr=scheduler_config.get('base_lr', 1e-5),
            max_lr=scheduler_config.get('max_lr', 1e-3),
            step_size=scheduler_config.get('step_size', 1000),
            warmup_steps=scheduler_config.get('warmup_steps', 0)
        )
    elif scheduler_type == 'exponential_cyclic':
        return ExponentialCyclicLR(
            optimizer,
            base_lr=scheduler_config.get('base_lr', 1e-5),
            max_lr=scheduler_config.get('max_lr', 1e-3),
            step_size=scheduler_config.get('step_size', 1000),
            decay_rate=scheduler_config.get('decay_rate', 0.99),
            warmup_steps=scheduler_config.get('warmup_steps', 0),
            min_lr=scheduler_config.get('min_lr', 1e-6)
        )
    elif scheduler_type == 'sinusoidal':
        return SinusoidalLR(
            optimizer,
            base_lr=scheduler_config.get('base_lr', 1e-5),
            max_lr=scheduler_config.get('max_lr', 1e-3),
            period=scheduler_config.get('period', 1000),
            warmup_steps=scheduler_config.get('warmup_steps', 0),
            phase_shift=scheduler_config.get('phase_shift', 0)
        )
    elif scheduler_type == 'adaptive_cyclic':
        return AdaptiveCyclicLR(
            optimizer,
            initial_base_lr=scheduler_config.get('initial_base_lr', 1e-5),
            initial_max_lr=scheduler_config.get('initial_max_lr', 1e-3),
            step_size=scheduler_config.get('step_size', 1000),
            adaptation_rate=scheduler_config.get('adaptation_rate', 0.1),
            min_amplitude=scheduler_config.get('min_amplitude', 1e-6),
            patience=scheduler_config.get('patience', 10)
        )
    else:
        raise ValueError(f"Unknown cyclic scheduler type: {scheduler_type}")


# Export public API
__all__ = [
    'CyclicMode',
    'CyclicLR',
    'TriangularScheduler',
    'ExponentialCyclicLR',
    'SinusoidalLR',
    'AdaptiveCyclicLR',
    'create_cyclic_scheduler'
]
