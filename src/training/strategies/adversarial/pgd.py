"""
Projected Gradient Descent (PGD) for Adversarial Training
==========================================================

Implementation of PGD and its variants for robust model training,
based on:
- Madry et al. (2018): "Towards Deep Learning Models Resistant to Adversarial Attacks"
- Zhang et al. (2019): "Theoretically Principled Trade-off between Robustness and Accuracy"
- Shafahi et al. (2019): "Adversarial Training for Free!"

PGD is an iterative adversarial attack that provides stronger adversarial
examples compared to single-step methods like FGM.

Mathematical Foundation:
x_{t+1} = Proj_{B(x,ε)}(x_t + α * sign(∇_x L(x_t, y)))
where Proj is projection onto ε-ball B(x,ε) around original input x.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, Tuple, Callable, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.strategies.adversarial.fgm import FGM, FGMConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PGDConfig(FGMConfig):
    """Configuration for PGD adversarial training"""
    
    # PGD specific parameters
    num_steps: int = 10  # Number of PGD iterations
    step_size: float = 0.3  # Step size (alpha)
    random_start: bool = True  # Random initialization
    
    # Projection parameters
    norm_type: str = "linf"  # "linf", "l2"
    epsilon: float = 0.5  # Perturbation bound
    
    # Advanced PGD variants
    use_trades: bool = False  # TRADES loss
    trades_beta: float = 6.0  # TRADES parameter
    
    use_mart: bool = False  # MART loss
    mart_beta: float = 5.0  # MART parameter
    
    use_free: bool = False  # Free adversarial training
    free_replay: int = 4  # Number of replays for free AT
    
    # Optimization
    early_stop: bool = True  # Stop if loss doesn't increase
    early_stop_threshold: float = 0.01
    
    # Targeted attack
    targeted: bool = False  # Targeted vs untargeted attack
    target_class: Optional[int] = None


class PGD(FGM):
    """
    Projected Gradient Descent for adversarial training.
    
    Implements multi-step adversarial attack with projection
    to generate stronger adversarial examples.
    """
    
    def __init__(self, model: nn.Module, config: Optional[PGDConfig] = None):
        """
        Initialize PGD attacker.
        
        Args:
            model: Model to attack
            config: PGD configuration
        """
        # Initialize with FGM base
        super().__init__(model, config or PGDConfig())
        
        # Override config type
        self.config: PGDConfig = config or PGDConfig()
        
        # Store original parameters for projection
        self.original_params = {}
        
        # Attack statistics
        self.attack_stats = {
            'success_rate': 0,
            'avg_perturbation_norm': 0,
            'early_stops': 0
        }
        
        logger.info(
            f"Initialized PGD: steps={config.num_steps}, "
            f"epsilon={config.epsilon}, norm={config.norm_type}"
        )
    
    def generate_adversarial(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        loss_fn: Optional[Callable] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Generate adversarial examples using PGD.
        
        Args:
            inputs: Clean inputs
            labels: True labels
            loss_fn: Loss function to maximize
            
        Returns:
            Adversarial inputs and statistics
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        # Clone inputs
        adv_inputs = inputs.clone().detach()
        
        # Random initialization if configured
        if self.config.random_start:
            # Initialize with random noise
            noise = torch.empty_like(adv_inputs).uniform_(-self.config.epsilon, self.config.epsilon)
            adv_inputs = self._project(inputs + noise, inputs)
        
        # Store best adversarial examples (for early stopping)
        best_adv = adv_inputs.clone()
        best_loss = float('-inf')
        
        # PGD iterations
        for step in range(self.config.num_steps):
            adv_inputs.requires_grad = True
            
            # Forward pass
            outputs = self.model(adv_inputs)
            
            # Calculate loss
            if self.config.targeted:
                # Minimize loss for target class
                target = self._get_target_labels(labels)
                loss = -loss_fn(outputs, target)
            else:
                # Maximize loss for true class
                loss = loss_fn(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Get gradient
            grad = adv_inputs.grad.data
            
            # Update adversarial examples
            with torch.no_grad():
                # Take step
                if self.config.norm_type == "linf":
                    adv_inputs = adv_inputs + self.config.step_size * grad.sign()
                elif self.config.norm_type == "l2":
                    grad_norm = grad.view(grad.shape[0], -1).norm(2, dim=1).view(-1, 1, 1, 1)
                    grad_norm = torch.clamp(grad_norm, min=1e-8)
                    adv_inputs = adv_inputs + self.config.step_size * grad / grad_norm
                
                # Project back to epsilon ball
                adv_inputs = self._project(adv_inputs, inputs)
                
                # Early stopping check
                if self.config.early_stop:
                    current_loss = loss.item()
                    if current_loss > best_loss + self.config.early_stop_threshold:
                        best_loss = current_loss
                        best_adv = adv_inputs.clone()
                    elif step > self.config.num_steps // 2:
                        # Stop if no improvement in second half
                        self.attack_stats['early_stops'] += 1
                        break
            
            # Clear gradients
            if adv_inputs.grad is not None:
                adv_inputs.grad.zero_()
        
        # Use best adversarial examples
        if self.config.early_stop:
            adv_inputs = best_adv
        
        # Calculate statistics
        perturbation = adv_inputs - inputs
        stats = {
            'perturbation_norm': perturbation.norm().item(),
            'final_loss': best_loss,
            'num_steps': step + 1
        }
        
        return adv_inputs, stats
    
    def _project(self, perturbed: torch.Tensor, original: torch.Tensor) -> torch.Tensor:
        """
        Project perturbation back to epsilon ball.
        
        Args:
            perturbed: Perturbed inputs
            original: Original inputs
            
        Returns:
            Projected inputs
        """
        perturbation = perturbed - original
        
        if self.config.norm_type == "linf":
            # L-infinity projection
            perturbation = torch.clamp(perturbation, -self.config.epsilon, self.config.epsilon)
        elif self.config.norm_type == "l2":
            # L2 projection
            # Reshape to [batch_size, -1]
            batch_size = perturbation.shape[0]
            perturbation_flat = perturbation.view(batch_size, -1)
            
            # Calculate norms
            norms = perturbation_flat.norm(2, dim=1, keepdim=True)
            
            # Scale if exceeds epsilon
            scale = torch.clamp(norms / self.config.epsilon, min=1.0)
            perturbation_flat = perturbation_flat / scale
            
            # Reshape back
            perturbation = perturbation_flat.view_as(perturbation)
        
        # Return projected inputs
        return original + perturbation
    
    def _get_target_labels(self, true_labels: torch.Tensor) -> torch.Tensor:
        """
        Get target labels for targeted attack.
        
        Args:
            true_labels: True labels
            
        Returns:
            Target labels
        """
        if self.config.target_class is not None:
            # Fixed target class
            return torch.full_like(true_labels, self.config.target_class)
        else:
            # Random target class (different from true class)
            num_classes = 4  # AG News has 4 classes
            target_labels = torch.randint(0, num_classes, true_labels.shape, device=true_labels.device)
            
            # Ensure target != true
            mask = target_labels == true_labels
            target_labels[mask] = (target_labels[mask] + 1) % num_classes
            
            return target_labels
    
    def trades_loss(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        beta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Calculate TRADES loss.
        
        TRADES: TRadeoff-inspired Adversarial Defense via Surrogate-loss minimization
        Balances accuracy and robustness.
        
        Args:
            inputs: Clean inputs
            labels: True labels
            beta: Trade-off parameter
            
        Returns:
            TRADES loss
        """
        if beta is None:
            beta = self.config.trades_beta
        
        # Clean predictions
        clean_outputs = self.model(inputs)
        clean_loss = F.cross_entropy(clean_outputs, labels)
        
        # Generate adversarial examples
        adv_inputs, _ = self.generate_adversarial(inputs, labels)
        
        # Adversarial predictions
        adv_outputs = self.model(adv_inputs)
        
        # KL divergence between clean and adversarial
        clean_probs = F.softmax(clean_outputs, dim=1)
        adv_log_probs = F.log_softmax(adv_outputs, dim=1)
        robust_loss = F.kl_div(adv_log_probs, clean_probs, reduction='batchmean')
        
        # TRADES loss
        loss = clean_loss + beta * robust_loss
        
        return loss
    
    def mart_loss(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        beta: Optional[float] = None
    ) -> torch.Tensor:
        """
        Calculate MART loss.
        
        MART: Improving Adversarial Robustness Requires Revisiting Misclassified Examples
        
        Args:
            inputs: Clean inputs
            labels: True labels
            beta: MART parameter
            
        Returns:
            MART loss
        """
        if beta is None:
            beta = self.config.mart_beta
        
        # Generate adversarial examples
        adv_inputs, _ = self.generate_adversarial(inputs, labels)
        
        # Get predictions
        clean_outputs = self.model(inputs)
        adv_outputs = self.model(adv_inputs)
        
        # Get probabilities
        clean_probs = F.softmax(clean_outputs, dim=1)
        adv_probs = F.softmax(adv_outputs, dim=1)
        
        # Boosted CE loss
        true_probs = torch.gather(adv_probs, 1, labels.unsqueeze(1)).squeeze()
        boosted_ce = -torch.log(true_probs + 1e-8) * (1 - true_probs) ** beta
        
        # KL regularization
        kl_loss = F.kl_div(
            F.log_softmax(adv_outputs, dim=1),
            clean_probs,
            reduction='none'
        ).sum(1)
        
        # MART loss
        loss = boosted_ce.mean() + 0.1 * kl_loss.mean()
        
        return loss


class FreeAdversarialTraining:
    """
    Free Adversarial Training.
    
    Reuses gradient information across iterations to reduce
    computational cost of adversarial training.
    """
    
    def __init__(
        self,
        model: nn.Module,
        epsilon: float = 0.5,
        step_size: float = 0.3,
        replay: int = 4
    ):
        """
        Initialize Free AT.
        
        Args:
            model: Model to train
            epsilon: Perturbation bound
            step_size: Step size
            replay: Number of replays
        """
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.replay = replay
        
        # Store perturbations
        self.perturbations = {}
        
        logger.info(f"Initialized Free AT with replay={replay}")
    
    def train_step(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """
        Single training step with Free AT.
        
        Args:
            inputs: Input batch
            labels: Labels
            optimizer: Optimizer
            
        Returns:
            Average loss
        """
        batch_id = id(inputs)
        
        # Initialize perturbation if not exists
        if batch_id not in self.perturbations:
            self.perturbations[batch_id] = torch.zeros_like(inputs)
        
        total_loss = 0
        
        for _ in range(self.replay):
            # Add perturbation
            perturbed = inputs + self.perturbations[batch_id]
            perturbed.requires_grad = True
            
            # Forward pass
            outputs = self.model(perturbed)
            loss = F.cross_entropy(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update perturbation
            with torch.no_grad():
                # Get gradient w.r.t input
                grad = perturbed.grad.data
                
                # Update perturbation
                self.perturbations[batch_id] += self.step_size * grad.sign()
                
                # Project to epsilon ball
                self.perturbations[batch_id] = torch.clamp(
                    self.perturbations[batch_id],
                    -self.epsilon,
                    self.epsilon
                )
            
            # Update model
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        return total_loss / self.replay


# Export classes
__all__ = [
    'PGDConfig',
    'PGD',
    'FreeAdversarialTraining'
]
