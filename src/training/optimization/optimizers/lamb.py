"""
LAMB (Layer-wise Adaptive Moments) Optimizer Implementation
============================================================

This module implements the LAMB optimizer following:
- You et al. (2020): "Large Batch Optimization for Deep Learning: Training BERT in 76 minutes"

Mathematical Foundation:
LAMB combines the adaptivity of Adam with layer-wise adaptation:

Adam update: m_t = β₁m_{t-1} + (1-β₁)g_t
            v_t = β₂v_{t-1} + (1-β₂)g_t²
            r_t = m_t/(√v_t + ε)

Layer adaptation: ||θ_t||/||r_t + λθ_t|| if ||θ_t|| > 0 and ||r_t + λθ_t|| > 0
                 1 otherwise

Final update: θ_{t+1} = θ_t - η * layer_adaptation * (r_t + λθ_t)

LAMB enables large batch training without loss of accuracy.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Tuple, Dict, Any, Callable
import math

import torch
from torch.optim import Optimizer

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LAMB(Optimizer):
    """
    Layer-wise Adaptive Moments optimizer for large batch training.
    
    LAMB adapts the learning rate of each layer based on the ratio of
    weight norm to gradient norm, enabling stable training with very
    large batch sizes (up to 64K).
    
    Key features:
    - Layer-wise learning rate adaptation
    - Bias correction
    - Weight decay decoupling
    - Gradient clipping
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.01,
        grad_clipping: bool = True,
        max_grad_norm: float = 1.0,
        bias_correction: bool = True,
        adaptive: bool = True,
        debias: bool = True
    ):
        """
        Initialize LAMB optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added for numerical stability
            weight_decay: Weight decay coefficient
            grad_clipping: Whether to clip gradients
            max_grad_norm: Maximum gradient norm for clipping
            bias_correction: Whether to use bias correction
            adaptive: Whether to use layer-wise adaptation
            debias: Whether to debias second moment
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            grad_clipping=grad_clipping,
            max_grad_norm=max_grad_norm,
            bias_correction=bias_correction,
            adaptive=adaptive,
            debias=debias
        )
        
        super(LAMB, self).__init__(params, defaults)
        
        logger.info(
            f"Initialized LAMB optimizer with lr={lr}, "
            f"betas={betas}, weight_decay={weight_decay}"
        )
    
    def _get_layer_adaptation(
        self,
        param_norm: torch.Tensor,
        update_norm: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute layer-wise learning rate adaptation.
        
        Following Equation 6 from You et al. (2020):
        φ = ||θ|| / ||r + λθ|| if both norms > 0, else 1
        
        Args:
            param_norm: L2 norm of parameters
            update_norm: L2 norm of update
            
        Returns:
            Layer adaptation factor
        """
        # Compute trust ratio
        if param_norm > 0 and update_norm > 0:
            # Layer adaptation
            trust_ratio = param_norm / update_norm
        else:
            trust_ratio = 1.0
        
        return trust_ratio
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Perform a single optimization step.
        
        Args:
            closure: Closure that reevaluates the model and returns loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('LAMB does not support sparse gradients')
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Gradient clipping
                if group['grad_clipping']:
                    grad_norm = grad.norm(2.0)
                    if grad_norm > group['max_grad_norm']:
                        grad = grad * (group['max_grad_norm'] / grad_norm)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step'] if group['bias_correction'] else 1
                bias_correction2 = 1 - beta2 ** state['step'] if group['bias_correction'] else 1
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Compute adaptive learning rate
                if group['debias']:
                    # Debias second moment for better convergence
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Compute update
                update = (exp_avg / bias_correction1) / denom
                
                # Add weight decay
                if group['weight_decay'] != 0:
                    update.add_(p, alpha=group['weight_decay'])
                
                # Compute norms for layer adaptation
                if group['adaptive']:
                    param_norm = p.norm(2.0)
                    update_norm = update.norm(2.0)
                    
                    # Layer-wise adaptive learning rate
                    trust_ratio = self._get_layer_adaptation(param_norm, update_norm)
                    
                    # Clip trust ratio for stability
                    trust_ratio = min(trust_ratio, 10.0)
                else:
                    trust_ratio = 1.0
                
                # Apply update with layer adaptation
                p.add_(update, alpha=-group['lr'] * trust_ratio)
        
        return loss
    
    def get_lr(self) -> Dict[str, float]:
        """
        Get current learning rates for all parameter groups.
        
        Returns:
            Dictionary of learning rates
        """
        lrs = {}
        for i, group in enumerate(self.param_groups):
            lrs[f'group_{i}'] = group['lr']
        return lrs
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return state dictionary for checkpointing.
        
        Returns:
            State dictionary
        """
        state_dict = super().state_dict()
        # Add LAMB-specific state
        state_dict['lamb_version'] = '1.0'
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load optimizer state.
        
        Args:
            state_dict: State dictionary to load
        """
        # Check version compatibility
        version = state_dict.pop('lamb_version', '1.0')
        if version != '1.0':
            logger.warning(f"Loading LAMB state from version {version}")
        
        super().load_state_dict(state_dict)
