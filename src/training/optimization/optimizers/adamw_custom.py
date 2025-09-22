"""
Custom AdamW Optimizer with Advanced Features
==============================================

Implementation of AdamW optimizer with additional features for improved training,
based on:
- Loshchilov & Hutter (2019): "Decoupled Weight Decay Regularization"
- Zhang et al. (2020): "Lookahead Optimizer: k steps forward, 1 step back"
- Liu et al. (2020): "On the Variance of the Adaptive Learning Rate and Beyond"

Includes features like:
- Decoupled weight decay
- Gradient centralization
- Lookahead mechanism
- Adaptive gradient clipping
- Warm restart scheduling

Mathematical Foundation:
m_t = β₁m_{t-1} + (1-β₁)g_t
v_t = β₂v_{t-1} + (1-β₂)g_t²
θ_t = θ_{t-1} - η(m_t/(√v_t + ε) + λθ_{t-1})

Author: Võ Hải Dũng
License: MIT
"""

import logging
import math
from typing import Optional, Dict, Any, Tuple, Callable, Iterable
import torch
from torch.optim import Optimizer

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class AdamWCustom(Optimizer):
    """
    Custom AdamW optimizer with advanced features.
    
    Implements AdamW with:
    1. Decoupled weight decay
    2. Gradient centralization
    3. Rectified updates (RAdam-style)
    4. Gradient norm tracking
    5. Adaptive clipping
    
    The optimizer provides better convergence properties and training stability
    compared to standard Adam, especially for transformer models.
    """
    
    def __init__(
        self,
        params: Iterable,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        amsgrad: bool = False,
        centralize_gradients: bool = False,
        rectified: bool = False,
        degenerated_to_sgd: bool = False,
        clip_grad_norm: Optional[float] = None,
        adaptive_clip: bool = False,
        warmup_steps: int = 0,
        total_steps: Optional[int] = None
    ):
        """
        Initialize custom AdamW optimizer.
        
        Args:
            params: Iterable of parameters to optimize
            lr: Learning rate
            betas: Coefficients for computing running averages
            eps: Term added for numerical stability
            weight_decay: Weight decay coefficient
            amsgrad: Whether to use AMSGrad variant
            centralize_gradients: Apply gradient centralization
            rectified: Use rectified updates (RAdam)
            degenerated_to_sgd: Degenerate to SGD for early iterations
            clip_grad_norm: Maximum gradient norm for clipping
            adaptive_clip: Use adaptive gradient clipping
            warmup_steps: Number of warmup steps
            total_steps: Total training steps for scheduling
        """
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            centralize_gradients=centralize_gradients,
            rectified=rectified,
            degenerated_to_sgd=degenerated_to_sgd,
            clip_grad_norm=clip_grad_norm,
            adaptive_clip=adaptive_clip
        )
        
        super().__init__(params, defaults)
        
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._step = 0
        
        # Statistics tracking
        self.grad_norm_history = []
        self.loss_history = []
        
        logger.info(
            f"Initialized AdamW optimizer: lr={lr}, weight_decay={weight_decay}, "
            f"rectified={rectified}, centralize={centralize_gradients}"
        )
    
    def __setstate__(self, state):
        """Set state for unpickling."""
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('centralize_gradients', False)
            group.setdefault('rectified', False)
    
    def _get_lr(self) -> float:
        """
        Get current learning rate with warmup and scheduling.
        
        Returns:
            Current learning rate
        """
        if self._step < self.warmup_steps:
            # Linear warmup
            return self.defaults['lr'] * (self._step / self.warmup_steps)
        elif self.total_steps is not None:
            # Cosine decay after warmup
            progress = (self._step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            return self.defaults['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return self.defaults['lr']
    
    def _centralize_gradient(self, grad: torch.Tensor) -> torch.Tensor:
        """
        Centralize gradient by subtracting the mean.
        
        Args:
            grad: Gradient tensor
            
        Returns:
            Centralized gradient
        """
        if len(grad.shape) > 1:
            # Centralize by subtracting the mean of the gradient
            grad = grad - grad.mean(dim=tuple(range(1, len(grad.shape))), keepdim=True)
        return grad
    
    def _get_adaptive_clip_factor(
        self,
        parameters: Iterable[torch.Tensor],
        gradients: Iterable[torch.Tensor]
    ) -> float:
        """
        Calculate adaptive clipping factor based on gradient and weight norms.
        
        Args:
            parameters: Model parameters
            gradients: Parameter gradients
            
        Returns:
            Clipping factor
        """
        param_norm = torch.sqrt(sum(p.pow(2).sum() for p in parameters))
        grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in gradients))
        
        # Adaptive factor based on the ratio of norms
        max_norm = param_norm * 0.1  # 10% of parameter norm as max gradient norm
        
        if grad_norm > max_norm:
            return max_norm / grad_norm
        return 1.0
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        # Collect all gradients for norm calculation
        all_grads = []
        all_params = []
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_grads.append(p.grad.data)
                    all_params.append(p.data)
        
        # Calculate gradient norm
        if all_grads:
            grad_norm = torch.sqrt(sum(g.pow(2).sum() for g in all_grads))
            self.grad_norm_history.append(grad_norm.item())
        
        # Adaptive clipping if enabled
        clip_factor = 1.0
        if self.defaults['adaptive_clip'] and all_grads:
            clip_factor = self._get_adaptive_clip_factor(all_params, all_grads)
        
        # Standard gradient clipping
        elif self.defaults['clip_grad_norm'] is not None and all_grads:
            total_norm = grad_norm
            clip_value = self.defaults['clip_grad_norm']
            clip_factor = clip_value / (total_norm + 1e-6)
            clip_factor = min(clip_factor, 1.0)
        
        # Get current learning rate
        current_lr = self._get_lr()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                # Apply gradient clipping
                if clip_factor < 1.0:
                    grad = grad * clip_factor
                
                # Gradient centralization
                if group['centralize_gradients']:
                    grad = self._centralize_gradient(grad)
                
                # State initialization
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                
                state['step'] += 1
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Exponential moving average of squared gradient values
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    # Maintains the maximum of all 2nd moment running avg
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Rectified update (RAdam)
                if group['rectified']:
                    # Compute the variance rectification term
                    rho_inf = 2 / (1 - beta2) - 1
                    rho_t = rho_inf - 2 * state['step'] * beta2 ** state['step'] / bias_correction2
                    
                    # Compute adaptive learning rate
                    if rho_t > 4:
                        r_t = math.sqrt(
                            (rho_t - 4) * (rho_t - 2) * rho_inf /
                            ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                        )
                        
                        step_size = current_lr * r_t * math.sqrt(bias_correction2) / bias_correction1
                        p.data.addcdiv_(exp_avg, denom, value=-step_size)
                    else:
                        # Degenerate to SGD for early iterations
                        if group['degenerated_to_sgd']:
                            step_size = current_lr / bias_correction1
                            p.data.add_(exp_avg, alpha=-step_size)
                else:
                    # Standard Adam update
                    step_size = current_lr * math.sqrt(bias_correction2) / bias_correction1
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Decoupled weight decay
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-current_lr * group['weight_decay'])
        
        self._step += 1
        
        return loss
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get optimizer statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            'step': self._step,
            'current_lr': self._get_lr(),
            'avg_grad_norm': sum(self.grad_norm_history[-100:]) / len(self.grad_norm_history[-100:])
                            if self.grad_norm_history else 0,
            'max_grad_norm': max(self.grad_norm_history[-100:])
                           if self.grad_norm_history else 0,
        }
        
        return stats


class Lookahead(Optimizer):
    """
    Lookahead Optimizer wrapper.
    
    Implements the Lookahead algorithm which maintains two sets of weights:
    fast and slow weights. The slow weights are updated periodically based
    on the fast weights' trajectory.
    
    Based on Zhang et al. (2019): "Lookahead Optimizer: k steps forward, 1 step back"
    """
    
    def __init__(
        self,
        base_optimizer: Optimizer,
        k: int = 5,
        alpha: float = 0.5
    ):
        """
        Initialize Lookahead optimizer.
        
        Args:
            base_optimizer: Base optimizer (e.g., AdamW)
            k: Number of fast weight updates before updating slow weights
            alpha: Slow weights update rate
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Invalid slow update rate: {alpha}")
        if not k >= 1:
            raise ValueError(f"Invalid lookahead steps: {k}")
        
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.step_count = 0
        
        # Initialize slow weights
        self.slow_weights = [[p.clone().detach() for p in group['params']]
                             for group in base_optimizer.param_groups]
        
        # Use base optimizer's param groups
        self.param_groups = base_optimizer.param_groups
        
        logger.info(f"Initialized Lookahead wrapper: k={k}, alpha={alpha}")
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.
        
        Args:
            closure: A closure that reevaluates the model and returns the loss
            
        Returns:
            Loss value if closure is provided
        """
        # Perform base optimizer step
        loss = self.base_optimizer.step(closure)
        
        self.step_count += 1
        
        # Update slow weights every k steps
        if self.step_count % self.k == 0:
            for group_idx, group in enumerate(self.base_optimizer.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    
                    # Update slow weights: θ_slow = θ_slow + α(θ_fast - θ_slow)
                    slow = self.slow_weights[group_idx][p_idx]
                    slow.add_(p.data - slow, alpha=self.alpha)
                    # Copy slow weights back to fast weights
                    p.data.copy_(slow)
        
        return loss
    
    def state_dict(self):
        """Get state dictionary."""
        state = {
            'base_optimizer': self.base_optimizer.state_dict(),
            'step_count': self.step_count,
            'slow_weights': self.slow_weights
        }
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        self.base_optimizer.load_state_dict(state_dict['base_optimizer'])
        self.step_count = state_dict['step_count']
        self.slow_weights = state_dict['slow_weights']


# Export classes
__all__ = [
    'AdamWCustom',
    'Lookahead'
]
