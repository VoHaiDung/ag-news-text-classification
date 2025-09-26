"""
SAM (Sharpness Aware Minimization) Optimizer Implementation
===========================================================

This module implements the SAM optimizer following:
- Foret et al. (2021): "Sharpness-Aware Minimization for Efficiently Improving Generalization"
- Kwon et al. (2021): "ASAM: Adaptive Sharpness-Aware Minimization"

Mathematical Foundation:
SAM seeks parameters that lie in neighborhoods with uniformly low loss:

min_w L^SAM(w) = max_{||ε||_2 ≤ ρ} L(w + ε)

The gradient approximation:
∇L^SAM(w) ≈ ∇L(w)|_{w + ε(w)} where ε(w) = ρ * ∇L(w) / ||∇L(w)||_2

This requires two forward-backward passes per iteration.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, Callable

import torch
from torch.optim import Optimizer

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SAM(Optimizer):
    """
    Sharpness Aware Minimization optimizer.
    
    SAM improves model generalization by explicitly seeking parameters
    that lie in neighborhoods having uniformly low loss value, which
    corresponds to flat minima in the loss landscape.
    
    Requires two gradient computations per iteration.
    """
    
    def __init__(
        self,
        params,
        base_optimizer: Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        perturb_eps: float = 1e-12
    ):
        """
        Initialize SAM optimizer.
        
        Args:
            params: Parameters to optimize
            base_optimizer: Base optimizer (e.g., SGD, Adam)
            rho: Neighborhood size (perturbation radius)
            adaptive: Whether to use adaptive SAM (ASAM)
            perturb_eps: Small value for numerical stability
        """
        if rho <= 0:
            raise ValueError(f"Invalid rho: {rho}")
        
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps
        
        # Store parameter groups
        self.param_groups = self.base_optimizer.param_groups
        
        defaults = dict(rho=rho, adaptive=adaptive)
        super(SAM, self).__init__(params, defaults)
        
        logger.info(
            f"Initialized SAM with rho={rho}, adaptive={adaptive}, "
            f"base_optimizer={base_optimizer.__class__.__name__}"
        )
    
    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        First step: compute and apply perturbation ε(w).
        
        This moves parameters to w + ε(w) for computing SAM gradient.
        
        Args:
            zero_grad: Whether to zero gradients after step
        """
        # Compute gradient norm for each parameter
        grad_norm = self._grad_norm()
        
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + self.perturb_eps)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Store original parameter values
                param_state = self.state[p]
                param_state['old_p'] = p.data.clone()
                
                # Compute perturbation
                if self.adaptive:
                    # ASAM: scale perturbation by parameter norm
                    # ε = ρ * (w / ||w||) * (∇L / ||∇L||)
                    param_norm = p.data.norm(p=2)
                    adaptive_scale = scale * (param_norm + self.perturb_eps)
                    e_w = p.grad * adaptive_scale
                else:
                    # Standard SAM: ε = ρ * ∇L / ||∇L||
                    e_w = p.grad * scale
                
                # Apply perturbation: w = w + ε(w)
                p.add_(e_w)
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Second step: restore original parameters and apply SAM update.
        
        This restores w from w + ε(w) and applies the SAM gradient.
        
        Args:
            zero_grad: Whether to zero gradients after step
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                param_state = self.state[p]
                
                # Restore original parameters
                if 'old_p' in param_state:
                    p.data = param_state['old_p']
        
        # Apply base optimizer step with SAM gradient
        self.base_optimizer.step()
        
        if zero_grad:
            self.zero_grad()
    
    def step(self, closure: Optional[Callable] = None):
        """
        Perform complete SAM optimization step.
        
        Note: This requires the closure to be provided for two evaluations.
        
        Args:
            closure: Closure that reevaluates model and returns loss
            
        Returns:
            Loss value after SAM step
        """
        if closure is None:
            raise RuntimeError("SAM requires closure for two forward passes")
        
        # Enable gradient computation for closure
        with torch.enable_grad():
            # First forward-backward pass
            loss = closure()
        
        # Compute and apply perturbation
        self.first_step(zero_grad=True)
        
        # Second forward-backward pass at perturbed point
        with torch.enable_grad():
            loss = closure()
        
        # Apply SAM update
        self.second_step(zero_grad=True)
        
        return loss
    
    def _grad_norm(self) -> torch.Tensor:
        """
        Compute L2 norm of gradients.
        
        Returns:
            Gradient norm
        """
        shared_device = self.param_groups[0]['params'][0].device
        norm = torch.norm(
            torch.stack([
                # Handle different gradient shapes
                ((torch.abs(p) if group['adaptive'] else 1.0) * p.grad).norm(p=2).to(shared_device)
                for group in self.param_groups
                for p in group['params']
                if p.grad is not None
            ]),
            p=2
        )
        return norm
    
    def zero_grad(self, set_to_none: bool = False):
        """
        Clear gradients of all optimized parameters.
        
        Args:
            set_to_none: Set gradients to None instead of zero
        """
        self.base_optimizer.zero_grad(set_to_none)
    
    def state_dict(self) -> Dict[str, Any]:
        """
        Return state dictionary for checkpointing.
        
        Returns:
            State dictionary
        """
        return {
            'base_optimizer_state': self.base_optimizer.state_dict(),
            'rho': self.rho,
            'adaptive': self.adaptive,
            'state': self.state
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load optimizer state.
        
        Args:
            state_dict: State dictionary to load
        """
        self.base_optimizer.load_state_dict(state_dict['base_optimizer_state'])
        self.rho = state_dict['rho']
        self.adaptive = state_dict['adaptive']
        self.state = state_dict['state']


class ESAM(SAM):
    """
    Efficient SAM (ESAM) optimizer.
    
    ESAM reduces computational cost by using gradient approximation
    and avoiding the second forward pass in some iterations.
    
    Based on Du et al. (2022): "Efficient Sharpness-aware Minimization"
    """
    
    def __init__(
        self,
        params,
        base_optimizer: Optimizer,
        rho: float = 0.05,
        adaptive: bool = False,
        perturb_eps: float = 1e-12,
        grad_reduce: float = 0.5,
        random_perturbation: float = 0.0
    ):
        """
        Initialize ESAM optimizer.
        
        Args:
            params: Parameters to optimize
            base_optimizer: Base optimizer
            rho: Neighborhood size
            adaptive: Whether to use adaptive SAM
            perturb_eps: Small value for stability
            grad_reduce: Gradient reduction factor for efficiency
            random_perturbation: Random perturbation probability
        """
        super().__init__(params, base_optimizer, rho, adaptive, perturb_eps)
        
        self.grad_reduce = grad_reduce
        self.random_perturbation = random_perturbation
        self.step_count = 0
        
        logger.info(
            f"Initialized ESAM with grad_reduce={grad_reduce}, "
            f"random_perturbation={random_perturbation}"
        )
    
    def step(self, closure: Optional[Callable] = None):
        """
        Perform ESAM optimization step with reduced computation.
        
        Args:
            closure: Closure that reevaluates model
            
        Returns:
            Loss value
        """
        self.step_count += 1
        
        # Use random perturbation occasionally
        if torch.rand(1).item() < self.random_perturbation:
            # Skip SAM, use base optimizer only
            return self.base_optimizer.step(closure)
        
        # Use gradient approximation for efficiency
        if self.step_count % 2 == 0:
            # Approximate gradient using previous gradient
            with torch.enable_grad():
                loss = closure()
            
            # Apply reduced perturbation
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.grad.mul_(self.grad_reduce)
            
            # Single step update
            self.base_optimizer.step()
            self.zero_grad()
            
            return loss
        else:
            # Full SAM step
            return super().step(closure)
