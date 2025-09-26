"""
Lookahead Optimizer Implementation
===================================

This module implements the Lookahead optimizer following:
- Zhang et al. (2019): "Lookahead Optimizer: k steps forward, 1 step back"

Mathematical Foundation:
Lookahead maintains two sets of weights:
- Fast weights: θ_t updated by base optimizer
- Slow weights: φ_t updated every k steps

Update rule:
φ_{t+1} = φ_t + α(θ_t - φ_t) every k steps

This reduces variance in the training trajectory and improves convergence.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, Callable
from collections import defaultdict

import torch
from torch.optim import Optimizer

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class Lookahead(Optimizer):
    """
    Lookahead optimizer wrapper.
    
    Lookahead improves convergence by maintaining slow and fast weights,
    where slow weights are updated by interpolating with fast weights
    every k steps. This stabilizes training and reduces variance.
    
    Can wrap any base optimizer (SGD, Adam, etc.)
    """
    
    def __init__(
        self,
        base_optimizer: Optimizer,
        k: int = 5,
        alpha: float = 0.5,
        pullback_momentum: str = "none"
    ):
        """
        Initialize Lookahead optimizer.
        
        Args:
            base_optimizer: Base optimizer to wrap
            k: Number of fast weight updates before updating slow weights
            alpha: Slow weights update factor (interpolation coefficient)
            pullback_momentum: Momentum correction ("none", "pullback", "reset")
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid alpha: {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid k: {k}")
        
        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.k = k
        self.alpha = alpha
        self.pullback_momentum = pullback_momentum
        
        # Counter for fast weight updates
        self.step_count = 0
        
        # State for slow weights
        self.state = defaultdict(dict)
        
        # Initialize slow weights
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['slow_buffer'] = p.data.clone()
                if self.pullback_momentum == "pullback":
                    param_state['momentum_buffer'] = torch.zeros_like(p.data)
        
        logger.info(
            f"Initialized Lookahead with k={k}, alpha={alpha}, "
            f"base_optimizer={base_optimizer.__class__.__name__}"
        )
    
    def _backup_and_restore_state(self, backup: bool = True):
        """
        Backup or restore base optimizer state for momentum correction.
        
        Args:
            backup: If True, backup state; if False, restore state
        """
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                base_state = self.base_optimizer.state.get(p, {})
                
                if backup:
                    # Backup base optimizer state
                    param_state['backup_state'] = {}
                    for key, value in base_state.items():
                        if torch.is_tensor(value):
                            param_state['backup_state'][key] = value.clone()
                        else:
                            param_state['backup_state'][key] = value
                else:
                    # Restore base optimizer state
                    if 'backup_state' in param_state:
                        for key, value in param_state['backup_state'].items():
                            if torch.is_tensor(value):
                                base_state[key] = value.clone()
                            else:
                                base_state[key] = value
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: Closure that reevaluates the model and returns loss
            
        Returns:
            Loss value if closure is provided
        """
        # Perform base optimizer step
        loss = self.base_optimizer.step(closure)
        
        self.step_count += 1
        
        # Update slow weights every k steps
        if self.step_count % self.k == 0:
            # Backup base optimizer state if using pullback momentum
            if self.pullback_momentum == "pullback":
                self._backup_and_restore_state(backup=True)
            
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    
                    param_state = self.state[p]
                    slow_buffer = param_state['slow_buffer']
                    
                    # Update slow weights: φ = φ + α(θ - φ)
                    slow_buffer.add_(p.data - slow_buffer, alpha=self.alpha)
                    
                    # Copy slow weights to fast weights
                    p.data.copy_(slow_buffer)
                    
                    # Pullback momentum correction
                    if self.pullback_momentum == "pullback":
                        # Correct momentum using slow weights
                        base_state = self.base_optimizer.state.get(p, {})
                        if 'momentum_buffer' in base_state:
                            momentum = base_state['momentum_buffer']
                            momentum.mul_(self.alpha).add_(
                                param_state['momentum_buffer'],
                                alpha=1 - self.alpha
                            )
                            param_state['momentum_buffer'].copy_(momentum)
            
            # Reset base optimizer momentum if specified
            if self.pullback_momentum == "reset":
                for group in self.param_groups:
                    for p in group['params']:
                        base_state = self.base_optimizer.state.get(p, {})
                        if 'momentum_buffer' in base_state:
                            base_state['momentum_buffer'].zero_()
            
            # Restore base optimizer state if using pullback
            if self.pullback_momentum == "pullback":
                self._backup_and_restore_state(backup=False)
        
        return loss
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return state dictionary for checkpointing.
        
        Returns:
            State dictionary
        """
        return {
            'state': self.state,
            'base_optimizer_state': self.base_optimizer.state_dict(),
            'step_count': self.step_count,
            'k': self.k,
            'alpha': self.alpha,
            'pullback_momentum': self.pullback_momentum
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load optimizer state.
        
        Args:
            state_dict: State dictionary to load
        """
        self.state = state_dict['state']
        self.base_optimizer.load_state_dict(state_dict['base_optimizer_state'])
        self.step_count = state_dict['step_count']
        self.k = state_dict['k']
        self.alpha = state_dict['alpha']
        self.pullback_momentum = state_dict['pullback_momentum']
    
    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients of all optimized parameters.
        
        Args:
            set_to_none: Set gradients to None instead of zero
        """
        self.base_optimizer.zero_grad(set_to_none)
    
    def get_lr(self) -> list:
        """
        Get current learning rates.
        
        Returns:
            List of learning rates
        """
        if hasattr(self.base_optimizer, 'get_lr'):
            return self.base_optimizer.get_lr()
        else:
            return [group['lr'] for group in self.param_groups]
