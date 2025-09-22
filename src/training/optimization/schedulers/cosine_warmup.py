"""
Cosine Learning Rate Scheduler with Warmup
===========================================

Implementation of cosine annealing with linear warmup for transformer training,
based on:
- Loshchilov & Hutter (2016): "SGDR: Stochastic Gradient Descent with Warm Restarts"
- Goyal et al. (2017): "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
- Liu et al. (2019): "On the Variance of the Adaptive Learning Rate and Beyond"

Mathematical Foundation:
During warmup: lr = lr_base * (step / warmup_steps)
After warmup: lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(π * T_cur / T_max))

Author: Võ Hải Dũng
License: MIT
"""

import logging
import math
from typing import Optional, List, Union, Callable
import warnings

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing scheduler with linear warmup.
    
    Implements a learning rate schedule that:
    1. Linearly increases from 0 to initial_lr during warmup
    2. Follows cosine annealing after warmup
    3. Optionally includes restarts (SGDR)
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize cosine warmup scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr: Minimum learning rate
            num_cycles: Number of cosine cycles
            last_epoch: The index of last epoch
            verbose: If True, prints a message to stdout for each update
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.num_cycles = num_cycles
        
        # Validate parameters
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})"
            )
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized CosineWarmupScheduler: "
            f"warmup={warmup_steps}, total={total_steps}, cycles={num_cycles}"
        )
    
    def get_lr(self) -> List[float]:
        """
        Compute learning rate for current step.
        
        Returns:
            List of learning rates for each param group
        """
        # During warmup
        if self.last_epoch < self.warmup_steps:
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        
        # After warmup: cosine annealing
        progress = (self.last_epoch - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        return [
            self.min_lr + (base_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * self.num_cycles * 2 * progress)
            )
            for base_lr in self.base_lrs
        ]


class LinearWarmupCosineDecay(_LRScheduler):
    """
    Linear warmup followed by cosine decay (used in BERT).
    
    This scheduler is specifically designed for transformer models,
    providing stable training dynamics.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            min_lr_ratio: Minimum lr as ratio of base lr
            last_epoch: The index of last epoch
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Get learning rate for current step"""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            decay_steps = self.total_steps - self.warmup_steps
            decay_progress = min(
                (self.last_epoch - self.warmup_steps) / decay_steps, 1.0
            )
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_progress))
            decay_factor = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_decay
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class CosineAnnealingWithRestarts(_LRScheduler):
    """
    Cosine annealing with warm restarts (SGDR).
    
    Implements the SGDR schedule where learning rate is reset
    periodically to escape local minima.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize SGDR scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            T_0: Number of iterations for the first restart
            T_mult: Factor to increase T_i after a restart
            eta_min: Minimum learning rate
            last_epoch: The index of last epoch
            verbose: If True, prints a message for each update
        """
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(f"Initialized SGDR: T_0={T_0}, T_mult={T_mult}")
    
    def get_lr(self) -> List[float]:
        """Get learning rate with restarts"""
        return [
            self.eta_min + (base_lr - self.eta_min) * 
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch: Optional[int] = None):
        """Update learning rate with restart logic"""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur += 1
            if self.T_cur >= self.T_i:
                self.T_cur = 0
                self.T_i *= self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Epoch must be non-negative")
            self.T_cur = epoch
            
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class OneCycleLR(_LRScheduler):
    """
    One Cycle Learning Rate Policy.
    
    Implements the 1cycle policy which combines:
    - Linear warmup to max_lr
    - Cosine annealing to min_lr
    - Optional super-convergence
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Union[float, List[float]],
        total_steps: int,
        pct_start: float = 0.3,
        anneal_strategy: str = 'cos',
        div_factor: float = 25.0,
        final_div_factor: float = 10000.0,
        last_epoch: int = -1
    ):
        """
        Initialize OneCycle scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            max_lr: Upper learning rate boundaries
            total_steps: Total number of training steps
            pct_start: Percentage of cycle spent increasing lr
            anneal_strategy: 'cos' or 'linear'
            div_factor: Initial lr = max_lr / div_factor
            final_div_factor: Final lr = max_lr / final_div_factor
            last_epoch: The index of last epoch
        """
        self.max_lr = max_lr if isinstance(max_lr, list) else [max_lr]
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        # Calculate phase boundaries
        self.step_up = int(total_steps * pct_start)
        self.step_down = total_steps - self.step_up
        
        # Set initial learning rates
        self.initial_lrs = [lr / div_factor for lr in self.max_lr]
        self.final_lrs = [lr / final_div_factor for lr in self.max_lr]
        
        # Set base_lrs to initial_lrs
        for group, lr in zip(optimizer.param_groups, self.initial_lrs):
            group['lr'] = lr
            
        super().__init__(optimizer, last_epoch)
        
        logger.info(
            f"Initialized OneCycleLR: max_lr={max_lr}, "
            f"steps={total_steps}, pct_start={pct_start}"
        )
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step"""
        if self.last_epoch < self.step_up:
            # Warmup phase
            pct = self.last_epoch / self.step_up
            return [
                initial_lr + pct * (max_lr - initial_lr)
                for initial_lr, max_lr in zip(self.initial_lrs, self.max_lr)
            ]
        else:
            # Annealing phase
            down_step = self.last_epoch - self.step_up
            pct = down_step / self.step_down
            
            if self.anneal_strategy == 'cos':
                # Cosine annealing
                anneal_factor = (1 + math.cos(math.pi * pct)) / 2
            else:
                # Linear annealing
                anneal_factor = 1 - pct
                
            return [
                final_lr + anneal_factor * (max_lr - final_lr)
                for max_lr, final_lr in zip(self.max_lr, self.final_lrs)
            ]


class PolynomialLR(_LRScheduler):
    """
    Polynomial learning rate decay.
    
    Used in BERT and other transformer models for smooth decay.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        power: float = 1.0,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        """
        Initialize polynomial scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            total_steps: Total training steps
            warmup_steps: Number of warmup steps
            power: Polynomial power
            min_lr: Minimum learning rate
            last_epoch: The index of last epoch
        """
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.power = power
        self.min_lr = min_lr
        
        super().__init__(optimizer, last_epoch)
        
    def get_lr(self) -> List[float]:
        """Get polynomial decay learning rate"""
        if self.last_epoch < self.warmup_steps:
            # Warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            remaining = self.total_steps - self.last_epoch
            decay_factor = (remaining / self.total_steps) ** self.power
            return [
                self.min_lr + (base_lr - self.min_lr) * decay_factor
                for base_lr in self.base_lrs
            ]


def create_scheduler(
    optimizer: Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    **kwargs
) -> _LRScheduler:
    """
    Factory function to create scheduler.
    
    Args:
        optimizer: Optimizer to wrap
        scheduler_type: Type of scheduler
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        **kwargs: Additional scheduler arguments
        
    Returns:
        Learning rate scheduler
    """
    schedulers = {
        'cosine_warmup': CosineWarmupScheduler,
        'linear_cosine': LinearWarmupCosineDecay,
        'sgdr': CosineAnnealingWithRestarts,
        'onecycle': OneCycleLR,
        'polynomial': PolynomialLR
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(
            f"Unknown scheduler: {scheduler_type}. "
            f"Available: {list(schedulers.keys())}"
        )
    
    scheduler_class = schedulers[scheduler_type]
    
    # Create scheduler with appropriate arguments
    if scheduler_type == 'cosine_warmup':
        scheduler = scheduler_class(
            optimizer, warmup_steps, total_steps, **kwargs
        )
    elif scheduler_type == 'linear_cosine':
        scheduler = scheduler_class(
            optimizer, warmup_steps, total_steps, **kwargs
        )
    elif scheduler_type == 'sgdr':
        T_0 = kwargs.get('T_0', warmup_steps)
        scheduler = scheduler_class(optimizer, T_0, **kwargs)
    elif scheduler_type == 'onecycle':
        max_lr = kwargs.get('max_lr', 1e-3)
        scheduler = scheduler_class(optimizer, max_lr, total_steps, **kwargs)
    elif scheduler_type == 'polynomial':
        scheduler = scheduler_class(
            optimizer, total_steps, warmup_steps, **kwargs
        )
    else:
        scheduler = scheduler_class(optimizer, **kwargs)
    
    logger.info(f"Created {scheduler_type} scheduler")
    return scheduler


# Export classes and functions
__all__ = [
    'CosineWarmupScheduler',
    'LinearWarmupCosineDecay',
    'CosineAnnealingWithRestarts',
    'OneCycleLR',
    'PolynomialLR',
    'create_scheduler'
]
