"""
Fast Gradient Method (FGM) for Adversarial Training
====================================================

Implementation of FGM and its variants for adversarial training,
based on:
- Goodfellow et al. (2015): "Explaining and Harnessing Adversarial Examples"
- Miyato et al. (2017): "Adversarial Training Methods for Semi-Supervised Text Classification"
- Zhu et al. (2020): "FreeLB: Enhanced Adversarial Training for Natural Language Understanding"

Mathematical Foundation:
Adversarial perturbation: r_adv = ε * g/||g||₂
where g = ∇_x L(x, y; θ) is the gradient of loss w.r.t. input

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FGMConfig:
    """Configuration for FGM adversarial training"""
    
    # Perturbation parameters
    epsilon: float = 0.5  # Magnitude of perturbation
    epsilon_schedule: str = "constant"  # "constant", "linear", "cosine"
    epsilon_min: float = 0.1
    epsilon_max: float = 1.0
    
    # Attack parameters
    attack_iters: int = 1  # Number of attack iterations (1 for FGM, >1 for PGD)
    alpha: float = 0.3  # Step size for iterative attacks
    
    # Normalization
    norm_type: str = "l2"  # "l2", "linf"
    scale_by_grad_norm: bool = True
    
    # Target layers
    attack_embeddings: bool = True  # Attack embedding layer
    attack_word_embeddings: bool = True
    attack_position_embeddings: bool = False
    
    # Training strategy
    adv_loss_weight: float = 1.0  # Weight for adversarial loss
    use_accumulated_grad: bool = False  # Accumulate gradients over iterations
    
    # Defense mechanisms
    gradient_clipping: float = 1.0
    add_noise: bool = False  # Add random noise to perturbations
    noise_var: float = 0.1


class FGM:
    """
    Fast Gradient Method for adversarial training.
    
    Generates adversarial examples by perturbing embeddings in the direction
    that maximizes the loss, improving model robustness.
    """
    
    def __init__(self, model: nn.Module, config: Optional[FGMConfig] = None):
        """
        Initialize FGM.
        
        Args:
            model: Model to attack
            config: FGM configuration
        """
        self.model = model
        self.config = config or FGMConfig()
        
        # Backup for embeddings
        self.backup = {}
        self.embedding_backup = {}
        
        # Statistics
        self.attack_success_rate = 0.0
        self.perturbation_norms = []
        
        # Step counter for scheduling
        self.current_step = 0
        
        logger.info(f"Initialized FGM with epsilon={config.epsilon}")
    
    def attack(
        self,
        epsilon: Optional[float] = None,
        embedding_name: str = "word_embeddings"
    ):
        """
        Apply adversarial perturbation to embeddings.
        
        Args:
            epsilon: Perturbation magnitude (optional)
            embedding_name: Name of embedding layer to attack
        """
        epsilon = epsilon or self._get_epsilon()
        
        # Find embedding layers
        for name, param in self.model.named_parameters():
            if param.requires_grad and self._should_attack(name):
                # Backup original parameters
                self.backup[name] = param.data.clone()
                
                # Calculate perturbation
                if param.grad is not None:
                    # Normalize gradient
                    norm = self._compute_norm(param.grad)
                    
                    if norm != 0:
                        # Scale perturbation
                        if self.config.scale_by_grad_norm:
                            r_adv = epsilon * param.grad / norm
                        else:
                            r_adv = epsilon * torch.sign(param.grad)
                        
                        # Add noise if configured
                        if self.config.add_noise:
                            noise = torch.randn_like(r_adv) * self.config.noise_var
                            r_adv = r_adv + noise
                        
                        # Apply perturbation
                        param.data.add_(r_adv)
                        
                        # Track statistics
                        self.perturbation_norms.append(r_adv.norm().item())
    
    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
    
    def _should_attack(self, param_name: str) -> bool:
        """Check if parameter should be attacked"""
        if not self.config.attack_embeddings:
            return False
        
        if "word_embedding" in param_name and self.config.attack_word_embeddings:
            return True
        if "position_embedding" in param_name and self.config.attack_position_embeddings:
            return True
        if "embedding" in param_name and self.config.attack_embeddings:
            return True
        
        return False
    
    def _get_epsilon(self) -> float:
        """Get epsilon value with scheduling"""
        if self.config.epsilon_schedule == "constant":
            return self.config.epsilon
        elif self.config.epsilon_schedule == "linear":
            # Linear decay
            progress = min(self.current_step / 10000, 1.0)  # Assume 10k steps
            return self.config.epsilon_max - (self.config.epsilon_max - self.config.epsilon_min) * progress
        elif self.config.epsilon_schedule == "cosine":
            # Cosine annealing
            import math
            progress = min(self.current_step / 10000, 1.0)
            return self.config.epsilon_min + 0.5 * (self.config.epsilon_max - self.config.epsilon_min) * \
                   (1 + math.cos(math.pi * progress))
        else:
            return self.config.epsilon
    
    def _compute_norm(self, grad: torch.Tensor) -> torch.Tensor:
        """Compute gradient norm"""
        if self.config.norm_type == "l2":
            return torch.norm(grad)
        elif self.config.norm_type == "linf":
            return torch.max(torch.abs(grad))
        else:
            return torch.norm(grad)
    
    def step(self):
        """Update step counter"""
        self.current_step += 1


class PGD(FGM):
    """
    Projected Gradient Descent - iterative version of FGM.
    
    Applies multiple steps of FGM with projection to stay within epsilon-ball.
    """
    
    def __init__(self, model: nn.Module, config: Optional[FGMConfig] = None):
        """Initialize PGD"""
        super().__init__(model, config)
        
        if config and config.attack_iters <= 1:
            config.attack_iters = 3  # Default to 3 iterations for PGD
        
        self.original_backup = {}
        
        logger.info(f"Initialized PGD with {config.attack_iters} iterations")
    
    def attack_iterative(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Callable
    ) -> torch.Tensor:
        """
        Perform iterative attack.
        
        Args:
            inputs: Input embeddings
            labels: Target labels
            loss_fn: Loss function
            
        Returns:
            Final adversarial loss
        """
        # Backup original embeddings
        self._backup_embeddings()
        
        # Random initialization within epsilon-ball
        if self.config.attack_iters > 1:
            self._random_init()
        
        for i in range(self.config.attack_iters):
            # Zero gradients
            if self.model.training:
                self.model.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward(retain_graph=True)
            
            # Apply perturbation
            self.attack(epsilon=self.config.alpha)
            
            # Project back to epsilon-ball
            self._project()
        
        # Final forward pass
        outputs = self.model(inputs)
        adv_loss = loss_fn(outputs, labels)
        
        # Restore original embeddings
        self.restore()
        
        return adv_loss
    
    def _backup_embeddings(self):
        """Backup original embeddings"""
        for name, param in self.model.named_parameters():
            if self._should_attack(name):
                self.original_backup[name] = param.data.clone()
    
    def _random_init(self):
        """Random initialization within epsilon-ball"""
        for name, param in self.model.named_parameters():
            if self._should_attack(name) and param.grad is not None:
                # Random perturbation
                random_perturb = torch.empty_like(param.data).uniform_(
                    -self.config.epsilon, self.config.epsilon
                )
                param.data.add_(random_perturb)
    
    def _project(self):
        """Project perturbation back to epsilon-ball"""
        for name, param in self.model.named_parameters():
            if name in self.original_backup:
                # Calculate total perturbation
                perturbation = param.data - self.original_backup[name]
                
                # Project to epsilon-ball
                if self.config.norm_type == "l2":
                    norm = torch.norm(perturbation)
                    if norm > self.config.epsilon:
                        perturbation = perturbation * self.config.epsilon / norm
                elif self.config.norm_type == "linf":
                    perturbation = torch.clamp(
                        perturbation, -self.config.epsilon, self.config.epsilon
                    )
                
                # Apply projected perturbation
                param.data = self.original_backup[name] + perturbation


def adversarial_training_step(
    model: nn.Module,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    fgm: FGM,
    loss_fn: Optional[Callable] = None
) -> Dict[str, float]:
    """
    Perform one step of adversarial training.
    
    Args:
        model: Model to train
        inputs: Input tensors
        labels: Target labels
        optimizer: Optimizer
        fgm: FGM instance
        loss_fn: Loss function (default: CrossEntropyLoss)
        
    Returns:
        Dictionary of losses
    """
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    # Forward pass
    outputs = model(inputs)
    clean_loss = loss_fn(outputs, labels)
    
    # Backward pass
    clean_loss.backward()
    
    # Generate adversarial examples
    fgm.attack()
    
    # Forward pass on adversarial examples
    adv_outputs = model(inputs)
    adv_loss = loss_fn(adv_outputs, labels)
    
    # Backward pass on adversarial loss
    adv_loss.backward()
    
    # Restore original parameters
    fgm.restore()
    
    # Gradient clipping
    if fgm.config.gradient_clipping > 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), fgm.config.gradient_clipping
        )
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    # Update FGM step counter
    fgm.step()
    
    return {
        'clean_loss': clean_loss.item(),
        'adv_loss': adv_loss.item(),
        'total_loss': (clean_loss + fgm.config.adv_loss_weight * adv_loss).item()
    }


# Export classes and functions
__all__ = [
    'FGMConfig',
    'FGM',
    'PGD',
    'adversarial_training_step'
]
