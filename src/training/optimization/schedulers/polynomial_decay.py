"""
Polynomial Learning Rate Decay Scheduler
=========================================

Implementation of polynomial learning rate decay with warmup, widely used in
transformer models, based on:
- Devlin et al. (2018): "BERT: Pre-training of Deep Bidirectional Transformers"
- Raffel et al. (2019): "Exploring the Limits of Transfer Learning with T5"
- He et al. (2020): "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"

Mathematical Foundation:
Warmup phase: lr = lr_base * (step / warmup_steps)
Decay phase: lr = lr_end + (lr_base - lr_end) * ((T_max - T_cur) / T_max) ^ power

This scheduler provides smooth, predictable decay crucial for stable convergence
in large-scale language model training.

Author: Võ Hải Dũng
License: MIT
"""

import math
import logging
from typing import Optional, List, Dict, Any, Union
import warnings

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

logger = logging.getLogger(__name__)


class PolynomialDecayScheduler(_LRScheduler):
    """
    Polynomial decay learning rate scheduler with linear warmup.
    
    This scheduler implements the polynomial decay strategy used in BERT and
    other transformer models, providing stable training dynamics through:
    1. Linear warmup from 0 to initial learning rate
    2. Polynomial decay to minimum learning rate
    3. Optional constant learning rate after decay
    
    The polynomial decay provides smoother transitions than step-based schedules
    while being more predictable than exponential decay.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 1.0,
        end_lr: float = 0.0,
        warmup_init_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize polynomial decay scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            total_steps: Total number of training steps
            power: Power of polynomial decay (1.0 = linear)
            end_lr: Final learning rate after decay
            warmup_init_lr: Initial learning rate for warmup
            last_epoch: Index of last epoch
            verbose: Whether to print updates
            
        Note:
            power=1.0 gives linear decay
            power>1.0 gives slower initial decay, faster final decay
            power<1.0 gives faster initial decay, slower final decay
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.end_lr = end_lr
        self.warmup_init_lr = warmup_init_lr
        
        # Validate parameters
        if warmup_steps > total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be <= total_steps ({total_steps})"
            )
        if power <= 0:
            raise ValueError(f"power ({power}) must be positive")
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized PolynomialDecayScheduler: "
            f"warmup={warmup_steps}, total={total_steps}, "
            f"power={power}, end_lr={end_lr}"
        )
    
    def get_lr(self) -> List[float]:
        """
        Calculate learning rate for current step.
        
        Returns:
            List of learning rates for each parameter group
        """
        # Handle case where we've exceeded total steps
        if self.last_epoch > self.total_steps:
            return [self.end_lr for _ in self.base_lrs]
        
        # Warmup phase
        if self.last_epoch < self.warmup_steps:
            if self.warmup_steps == 0:
                warmup_factor = 1.0
            else:
                warmup_factor = (self.warmup_init_lr + 
                    (1.0 - self.warmup_init_lr) * self.last_epoch / self.warmup_steps)
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Polynomial decay phase
        if self.last_epoch == self.total_steps:
            return [self.end_lr for _ in self.base_lrs]
        
        # Calculate decay factor
        decay_steps = self.total_steps - self.warmup_steps
        current_decay_step = self.last_epoch - self.warmup_steps
        
        if decay_steps == 0:
            decay_factor = 0.0
        else:
            decay_factor = (1.0 - current_decay_step / decay_steps) ** self.power
        
        return [
            self.end_lr + (base_lr - self.end_lr) * decay_factor
            for base_lr in self.base_lrs
        ]
    
    def _get_closed_form_lr(self) -> List[float]:
        """
        Get learning rate using closed form solution.
        
        This is used for resuming training from checkpoints.
        
        Returns:
            List of learning rates
        """
        return self.get_lr()


class InverseSquareRootScheduler(_LRScheduler):
    """
    Inverse square root learning rate scheduler.
    
    Used in Transformer models like the original "Attention is All You Need" paper.
    Provides aggressive initial decay that stabilizes over time.
    
    Formula: lr = scale_factor * min(step^(-0.5), step * warmup_steps^(-1.5))
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        scale_factor: float = 1.0,
        min_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize inverse square root scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            scale_factor: Scaling factor for learning rate
            min_lr: Minimum learning rate
            last_epoch: Index of last epoch
            verbose: Whether to print updates
        """
        self.warmup_steps = warmup_steps
        self.scale_factor = scale_factor
        self.min_lr = min_lr
        
        if warmup_steps == 0:
            raise ValueError("warmup_steps must be > 0 for InverseSquareRootScheduler")
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized InverseSquareRootScheduler: "
            f"warmup={warmup_steps}, scale={scale_factor}"
        )
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate using inverse square root decay"""
        step = max(1, self.last_epoch)
        
        if step < self.warmup_steps:
            # Linear warmup
            scale = step / self.warmup_steps
        else:
            # Inverse square root decay
            scale = (self.warmup_steps ** 0.5) / (step ** 0.5)
        
        scale *= self.scale_factor
        
        return [
            max(self.min_lr, base_lr * scale)
            for base_lr in self.base_lrs
        ]


class PolynomialDecayWithRestart(_LRScheduler):
    """
    Polynomial decay with periodic restarts.
    
    Combines polynomial decay with warm restarts to escape local minima
    while maintaining smooth learning rate transitions.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        restart_steps: List[int],
        restart_weights: List[float] = None,
        power: float = 1.0,
        end_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize polynomial decay with restarts.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Initial warmup steps
            restart_steps: Steps at which to restart
            restart_weights: Multipliers for learning rate at restarts
            power: Power of polynomial decay
            end_lr: Final learning rate
            last_epoch: Index of last epoch
            verbose: Whether to print updates
        """
        self.warmup_steps = warmup_steps
        self.restart_steps = sorted(restart_steps)
        self.restart_weights = restart_weights or [1.0] * len(restart_steps)
        self.power = power
        self.end_lr = end_lr
        
        # Track current segment
        self.current_segment = 0
        self.segment_start = 0
        self.segment_end = self.restart_steps[0] if restart_steps else float('inf')
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized PolynomialDecayWithRestart: "
            f"restarts at {restart_steps}"
        )
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate with restart logic"""
        # Check for restart
        if (self.current_segment < len(self.restart_steps) and 
            self.last_epoch >= self.restart_steps[self.current_segment]):
            self.current_segment += 1
            self.segment_start = self.restart_steps[self.current_segment - 1]
            
            if self.current_segment < len(self.restart_steps):
                self.segment_end = self.restart_steps[self.current_segment]
            else:
                self.segment_end = float('inf')
        
        # Calculate within-segment progress
        if self.last_epoch < self.warmup_steps:
            # Initial warmup
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Polynomial decay within segment
        segment_length = self.segment_end - self.segment_start
        segment_progress = (self.last_epoch - self.segment_start) / segment_length
        segment_progress = min(1.0, segment_progress)
        
        decay_factor = (1.0 - segment_progress) ** self.power
        
        # Apply restart weight if applicable
        if self.current_segment > 0:
            restart_weight = self.restart_weights[min(
                self.current_segment - 1, 
                len(self.restart_weights) - 1
            )]
        else:
            restart_weight = 1.0
        
        return [
            self.end_lr + (base_lr * restart_weight - self.end_lr) * decay_factor
            for base_lr in self.base_lrs
        ]


class LayerwiseDecayScheduler(_LRScheduler):
    """
    Layer-wise learning rate decay scheduler.
    
    Implements layer-wise adaptive learning rates where deeper layers
    have smaller learning rates. Used in fine-tuning large models.
    
    Based on:
    - Clark et al. (2020): "ELECTRA: Pre-training Text Encoders as Discriminators"
    - Bao et al. (2021): "BEiT: BERT Pre-Training of Image Transformers"
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        total_steps: int,
        layer_decay: float = 0.75,
        num_layers: int = 12,
        power: float = 1.0,
        end_lr: float = 0.0,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize layer-wise decay scheduler.
        
        Args:
            optimizer: Wrapped optimizer (must have layer-grouped parameters)
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            layer_decay: Decay factor per layer (deeper = smaller lr)
            num_layers: Number of layers in model
            power: Power for polynomial decay
            end_lr: Final learning rate
            last_epoch: Index of last epoch
            verbose: Whether to print updates
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.layer_decay = layer_decay
        self.num_layers = num_layers
        self.power = power
        self.end_lr = end_lr
        
        # Calculate layer-wise multipliers
        self.layer_multipliers = [
            layer_decay ** (num_layers - i) 
            for i in range(num_layers + 1)  # +1 for embeddings/classifier
        ]
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized LayerwiseDecayScheduler: "
            f"layers={num_layers}, decay={layer_decay}"
        )
    
    def get_lr(self) -> List[float]:
        """Calculate layer-wise learning rates"""
        # Base polynomial decay calculation
        if self.last_epoch < self.warmup_steps:
            base_scale = self.last_epoch / self.warmup_steps
        else:
            decay_steps = self.total_steps - self.warmup_steps
            current_decay = self.last_epoch - self.warmup_steps
            decay_factor = (1.0 - current_decay / decay_steps) ** self.power
            base_scale = decay_factor
        
        # Apply layer-wise scaling
        lrs = []
        for i, (base_lr, param_group) in enumerate(
            zip(self.base_lrs, self.optimizer.param_groups)
        ):
            # Determine layer index from parameter group
            # This assumes parameter groups are organized by layers
            layer_idx = param_group.get('layer_idx', min(i, len(self.layer_multipliers) - 1))
            layer_mult = self.layer_multipliers[layer_idx]
            
            lr = self.end_lr + (base_lr * layer_mult - self.end_lr) * base_scale
            lrs.append(lr)
        
        return lrs


class AdaptivePolynomialScheduler(_LRScheduler):
    """
    Adaptive polynomial scheduler with automatic adjustment.
    
    Monitors training metrics and adjusts decay rate dynamically
    to prevent premature convergence or instability.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int,
        min_steps: int,
        max_steps: int,
        initial_power: float = 1.0,
        end_lr: float = 0.0,
        patience: int = 10,
        threshold: float = 0.001,
        last_epoch: int = -1,
        verbose: bool = False
    ):
        """
        Initialize adaptive polynomial scheduler.
        
        Args:
            optimizer: Wrapped optimizer
            warmup_steps: Number of warmup steps
            min_steps: Minimum total steps
            max_steps: Maximum total steps
            initial_power: Initial polynomial power
            end_lr: Final learning rate
            patience: Steps to wait before adjustment
            threshold: Improvement threshold
            last_epoch: Index of last epoch
            verbose: Whether to print updates
        """
        self.warmup_steps = warmup_steps
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.current_total_steps = min_steps
        self.power = initial_power
        self.end_lr = end_lr
        self.patience = patience
        self.threshold = threshold
        
        # Tracking for adaptation
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.loss_history = []
        
        super().__init__(optimizer, last_epoch, verbose)
        
        logger.info(
            f"Initialized AdaptivePolynomialScheduler: "
            f"steps=[{min_steps}, {max_steps}], power={initial_power}"
        )
    
    def get_lr(self) -> List[float]:
        """Calculate adaptive learning rates"""
        if self.last_epoch < self.warmup_steps:
            warmup_factor = self.last_epoch / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        
        # Polynomial decay with current settings
        decay_steps = self.current_total_steps - self.warmup_steps
        current_decay = min(self.last_epoch - self.warmup_steps, decay_steps)
        
        if decay_steps == 0:
            decay_factor = 0.0
        else:
            decay_factor = max(0, 1.0 - current_decay / decay_steps) ** self.power
        
        return [
            self.end_lr + (base_lr - self.end_lr) * decay_factor
            for base_lr in self.base_lrs
        ]
    
    def update_metrics(self, loss: float):
        """
        Update scheduler based on training metrics.
        
        Args:
            loss: Current training loss
        """
        self.loss_history.append(loss)
        
        # Check for improvement
        if loss < self.best_loss - self.threshold:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Adjust if no improvement
        if self.patience_counter >= self.patience:
            self._adjust_schedule()
            self.patience_counter = 0
    
    def _adjust_schedule(self):
        """Adjust scheduling parameters based on training progress"""
        # Extend training if still improving
        if self.current_total_steps < self.max_steps:
            extension = min(
                self.current_total_steps * 0.2,
                self.max_steps - self.current_total_steps
            )
            self.current_total_steps += int(extension)
            logger.info(f"Extended training to {self.current_total_steps} steps")
        
        # Adjust power for smoother decay
        if len(self.loss_history) > 20:
            recent_std = torch.std(torch.tensor(self.loss_history[-20:])).item()
            if recent_std < 0.001:  # Plateau detected
                self.power = max(0.5, self.power * 0.9)
                logger.info(f"Adjusted decay power to {self.power}")


def create_polynomial_scheduler(
    optimizer: Optimizer,
    scheduler_config: Dict[str, Any]
) -> _LRScheduler:
    """
    Factory function to create polynomial-based schedulers.
    
    Args:
        optimizer: Optimizer to wrap
        scheduler_config: Configuration dictionary
        
    Returns:
        Configured scheduler instance
    """
    scheduler_type = scheduler_config.get('type', 'polynomial')
    
    if scheduler_type == 'polynomial':
        return PolynomialDecayScheduler(
            optimizer,
            warmup_steps=scheduler_config['warmup_steps'],
            total_steps=scheduler_config['total_steps'],
            power=scheduler_config.get('power', 1.0),
            end_lr=scheduler_config.get('end_lr', 0.0)
        )
    elif scheduler_type == 'inverse_sqrt':
        return InverseSquareRootScheduler(
            optimizer,
            warmup_steps=scheduler_config['warmup_steps'],
            scale_factor=scheduler_config.get('scale_factor', 1.0),
            min_lr=scheduler_config.get('min_lr', 0.0)
        )
    elif scheduler_type == 'polynomial_restart':
        return PolynomialDecayWithRestart(
            optimizer,
            warmup_steps=scheduler_config['warmup_steps'],
            restart_steps=scheduler_config['restart_steps'],
            restart_weights=scheduler_config.get('restart_weights'),
            power=scheduler_config.get('power', 1.0),
            end_lr=scheduler_config.get('end_lr', 0.0)
        )
    elif scheduler_type == 'layerwise':
        return LayerwiseDecayScheduler(
            optimizer,
            warmup_steps=scheduler_config['warmup_steps'],
            total_steps=scheduler_config['total_steps'],
            layer_decay=scheduler_config.get('layer_decay', 0.75),
            num_layers=scheduler_config.get('num_layers', 12),
            power=scheduler_config.get('power', 1.0),
            end_lr=scheduler_config.get('end_lr', 0.0)
        )
    elif scheduler_type == 'adaptive':
        return AdaptivePolynomialScheduler(
            optimizer,
            warmup_steps=scheduler_config['warmup_steps'],
            min_steps=scheduler_config['min_steps'],
            max_steps=scheduler_config['max_steps'],
            initial_power=scheduler_config.get('initial_power', 1.0),
            end_lr=scheduler_config.get('end_lr', 0.0),
            patience=scheduler_config.get('patience', 10),
            threshold=scheduler_config.get('threshold', 0.001)
        )
    else:
        raise ValueError(f"Unknown polynomial scheduler type: {scheduler_type}")


# Export public API
__all__ = [
    'PolynomialDecayScheduler',
    'InverseSquareRootScheduler',
    'PolynomialDecayWithRestart',
    'LayerwiseDecayScheduler',
    'AdaptivePolynomialScheduler',
    'create_polynomial_scheduler'
]
