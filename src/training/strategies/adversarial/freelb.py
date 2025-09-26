"""
FreeLB (Free Large-Batch) Adversarial Training Implementation
==============================================================

Implementation of FreeLB for adversarial training with free adversarial steps,
based on:
- Zhu et al. (2020): "FreeLB: Enhanced Adversarial Training for Natural Language Understanding"
- Shafahi et al. (2019): "Adversarial Training for Free!"

Mathematical Foundation:
FreeLB optimizes: min_θ E[max_||δ||≤ε L(θ, x+δ, y)]
using multiple PGD steps within single backward pass for efficiency.

Key Innovation: Accumulates gradients from multiple adversarial examples
without additional forward passes, achieving "free" adversarial training.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.training.trainers.base_trainer import BaseTrainer, TrainerConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FreeLBConfig(TrainerConfig):
    """Configuration for FreeLB adversarial training."""
    
    # Adversarial parameters
    adv_steps: int = 3  # Number of adversarial steps (K in paper)
    adv_lr: float = 1e-1  # Learning rate for adversarial perturbation
    adv_init_mag: float = 2e-2  # Initial perturbation magnitude
    adv_max_norm: float = 3e-1  # Maximum perturbation norm (epsilon)
    adv_norm_type: str = "l2"  # Norm type: "l2" or "linf"
    
    # Gradient accumulation for adversarial steps
    adv_grad_accumulation: bool = True  # Accumulate gradients across steps
    
    # Initialization strategy
    init_type: str = "uniform"  # "uniform", "normal", "zero"
    
    # Regularization
    use_kl_loss: bool = True  # Add KL divergence loss
    kl_weight: float = 1.0  # Weight for KL loss
    
    # Stability
    grad_clip_norm: float = 1.0  # Gradient clipping
    use_amp: bool = False  # Automatic mixed precision
    
    # Advanced options
    dynamic_epsilon: bool = False  # Dynamically adjust epsilon
    epsilon_schedule: str = "linear"  # "linear", "cosine", "constant"
    warmup_steps: int = 100  # Warmup steps for epsilon


class FreeLB(BaseTrainer):
    """
    FreeLB adversarial training implementation.
    
    Performs multiple adversarial steps within single backward pass,
    accumulating gradients for efficient adversarial training.
    
    Key features:
    1. Multiple PGD steps with gradient accumulation
    2. Free adversarial training (no extra forward passes)
    3. KL divergence regularization
    4. Dynamic epsilon scheduling
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[FreeLBConfig] = None,
        **kwargs
    ):
        """
        Initialize FreeLB trainer.
        
        Args:
            model: Model to train
            config: FreeLB configuration
            **kwargs: Additional trainer arguments
        """
        config = config or FreeLBConfig()
        super().__init__(model, config, **kwargs)
        
        self.config: FreeLBConfig = config
        self.step_count = 0
        
        # Initialize scaler for mixed precision
        if config.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        logger.info(
            f"Initialized FreeLB with {config.adv_steps} adversarial steps, "
            f"epsilon={config.adv_max_norm:.3f}, norm={config.adv_norm_type}"
        )
    
    def _initialize_delta(
        self,
        embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Initialize adversarial perturbation.
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask for valid tokens
            
        Returns:
            Initial perturbation delta
        """
        if self.config.init_type == "uniform":
            # Uniform initialization
            delta = torch.zeros_like(embeddings).uniform_(
                -self.config.adv_init_mag,
                self.config.adv_init_mag
            )
        elif self.config.init_type == "normal":
            # Normal initialization
            delta = torch.zeros_like(embeddings).normal_(
                0, self.config.adv_init_mag
            )
        else:
            # Zero initialization
            delta = torch.zeros_like(embeddings)
        
        # Mask out padding tokens
        if attention_mask is not None:
            delta = delta * attention_mask.unsqueeze(-1)
        
        # Project to epsilon ball
        delta = self._project_delta(delta, attention_mask)
        
        return delta
    
    def _project_delta(
        self,
        delta: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Project perturbation to epsilon ball.
        
        Args:
            delta: Perturbation to project
            attention_mask: Attention mask for valid tokens
            
        Returns:
            Projected perturbation
        """
        if self.config.adv_norm_type == "l2":
            # L2 norm projection
            if attention_mask is not None:
                # Compute norm only over valid tokens
                masked_delta = delta * attention_mask.unsqueeze(-1)
                norms = masked_delta.norm(p=2, dim=-1, keepdim=True)
                # Average over sequence length
                seq_lens = attention_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
                norms = norms / (seq_lens + 1e-8)
            else:
                norms = delta.norm(p=2, dim=-1, keepdim=True)
            
            # Scale if exceeds max norm
            scale = torch.clamp(norms / self.config.adv_max_norm, min=1.0)
            delta = delta / scale
            
        elif self.config.adv_norm_type == "linf":
            # L-infinity norm projection
            delta = torch.clamp(
                delta,
                -self.config.adv_max_norm,
                self.config.adv_max_norm
            )
        
        # Mask out padding tokens
        if attention_mask is not None:
            delta = delta * attention_mask.unsqueeze(-1)
        
        return delta
    
    def _get_current_epsilon(self) -> float:
        """
        Get current epsilon value with scheduling.
        
        Returns:
            Current epsilon value
        """
        if not self.config.dynamic_epsilon:
            return self.config.adv_max_norm
        
        if self.step_count < self.config.warmup_steps:
            # Warmup phase
            progress = self.step_count / self.config.warmup_steps
            if self.config.epsilon_schedule == "linear":
                epsilon = self.config.adv_max_norm * progress
            elif self.config.epsilon_schedule == "cosine":
                epsilon = self.config.adv_max_norm * (1 - np.cos(np.pi * progress)) / 2
            else:
                epsilon = self.config.adv_max_norm
        else:
            epsilon = self.config.adv_max_norm
        
        return epsilon
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        FreeLB training step with multiple adversarial iterations.
        
        Args:
            batch: Batch of training data
            
        Returns:
            Training metrics
        """
        self.model.train()
        self.step_count += 1
        
        # Move batch to device
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Get embeddings
        embeddings = self.model.get_input_embeddings()(input_ids)
        
        # Initialize delta
        delta = self._initialize_delta(embeddings, attention_mask)
        delta.requires_grad_(True)
        
        # Update epsilon if using scheduling
        if self.config.dynamic_epsilon:
            current_epsilon = self._get_current_epsilon()
            self.config.adv_max_norm = current_epsilon
        
        # Accumulate gradients over adversarial steps
        total_loss = 0.0
        total_kl_loss = 0.0
        
        for adv_step in range(self.config.adv_steps):
            # Add perturbation to embeddings
            if adv_step == 0:
                # First step: use original embeddings for clean loss
                perturbed_embeddings = embeddings
            else:
                perturbed_embeddings = embeddings + delta
            
            # Forward pass with perturbed embeddings
            outputs = self.model(
                inputs_embeds=perturbed_embeddings,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Compute loss
            loss = outputs.loss / self.config.adv_steps  # Average over steps
            
            # Add KL divergence loss if enabled
            if self.config.use_kl_loss and adv_step > 0:
                with torch.no_grad():
                    clean_outputs = self.model(
                        inputs_embeds=embeddings,
                        attention_mask=attention_mask
                    )
                
                # KL divergence between clean and adversarial predictions
                kl_loss = F.kl_div(
                    F.log_softmax(outputs.logits, dim=-1),
                    F.softmax(clean_outputs.logits, dim=-1),
                    reduction='batchmean'
                )
                loss = loss + self.config.kl_weight * kl_loss / (self.config.adv_steps - 1)
                total_kl_loss += kl_loss.item()
            
            total_loss += loss.item() * self.config.adv_steps
            
            # Backward pass
            if self.config.use_amp and self.scaler is not None:
                self.scaler.scale(loss).backward(retain_graph=True)
            else:
                loss.backward(retain_graph=True)
            
            # Update delta for next step (except last step)
            if adv_step < self.config.adv_steps - 1:
                # Get gradient w.r.t. delta
                delta_grad = delta.grad.clone().detach()
                
                # Normalize gradient
                if self.config.adv_norm_type == "l2":
                    denorm = torch.norm(
                        delta_grad.view(delta_grad.size(0), -1),
                        dim=1
                    ).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta_grad = delta_grad / denorm
                elif self.config.adv_norm_type == "linf":
                    delta_grad = delta_grad.sign()
                
                # Update delta with gradient ascent
                delta = delta + self.config.adv_lr * delta_grad
                
                # Project to epsilon ball
                delta = self._project_delta(delta, attention_mask)
                delta = delta.detach()
                delta.requires_grad_(True)
                
                # Clear gradients for delta
                if delta.grad is not None:
                    delta.grad.zero_()
        
        # Gradient clipping
        if self.config.grad_clip_norm > 0:
            if self.config.use_amp and self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip_norm
            )
        
        # Optimizer step
        if self.config.use_amp and self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Metrics
        metrics = {
            "loss": total_loss,
            "kl_loss": total_kl_loss / max(self.config.adv_steps - 1, 1),
            "epsilon": self.config.adv_max_norm,
            "adv_steps": self.config.adv_steps
        }
        
        return metrics
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with FreeLB.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        self.model.train()
        epoch_metrics = {
            "loss": 0.0,
            "kl_loss": 0.0
        }
        
        num_batches = 0
        
        for batch in self.train_loader:
            metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value
            
            num_batches += 1
            
            # Log progress
            if num_batches % 100 == 0:
                logger.info(
                    f"Epoch {epoch}, Batch {num_batches}: "
                    f"loss={metrics['loss']:.4f}, "
                    f"kl_loss={metrics.get('kl_loss', 0):.4f}"
                )
        
        # Average metrics
        for key in epoch_metrics:
            epoch_metrics[key] /= num_batches
        
        # Add epoch-specific metrics
        epoch_metrics["epoch"] = epoch
        epoch_metrics["epsilon"] = self.config.adv_max_norm
        
        return epoch_metrics
