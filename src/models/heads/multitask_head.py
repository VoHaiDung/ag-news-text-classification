"""
Multi-task Classification Head Implementation
==============================================

Implementation of multi-task learning heads for joint training on multiple
objectives, based on:
- Caruana (1997): "Multitask Learning"
- Ruder (2017): "An Overview of Multi-Task Learning in Deep Neural Networks"
- Liu et al. (2019): "Multi-Task Deep Neural Networks for Natural Language Understanding"

Mathematical Foundation:
Multi-task loss: L = Σ_i w_i * L_i(θ_shared, θ_i)
where w_i are task weights, θ_shared are shared parameters, θ_i are task-specific parameters.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TaskType(Enum):
    """Types of tasks supported."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    BINARY = "binary"
    MULTILABEL = "multilabel"
    RANKING = "ranking"


@dataclass
class TaskConfig:
    """Configuration for a single task."""
    name: str
    task_type: TaskType
    num_labels: int
    weight: float = 1.0
    hidden_size: int = 768
    dropout: float = 0.1
    label_smoothing: float = 0.0
    use_task_embedding: bool = False
    task_embedding_dim: int = 64


class UncertaintyWeighting(nn.Module):
    """
    Uncertainty-based task weighting.
    
    Based on Kendall et al. (2018): "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    
    def __init__(self, num_tasks: int):
        super().__init__()
        # Log variance parameters (learnable)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    
    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """
        Weight losses by uncertainty.
        
        L = Σ_i (1/2σ²_i) * L_i + log(σ_i)
        """
        weighted_losses = []
        
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)
        
        return sum(weighted_losses)


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal for adversarial training between tasks."""
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class TaskSpecificHead(nn.Module):
    """
    Task-specific classification head.
    
    Implements task-specific transformations and predictions.
    """
    
    def __init__(self, config: TaskConfig):
        super().__init__()
        self.config = config
        
        # Task embedding (optional)
        if config.use_task_embedding:
            self.task_embedding = nn.Parameter(
                torch.randn(1, config.task_embedding_dim)
            )
        
        # Task-specific layers
        layers = []
        
        # Hidden layer
        input_dim = config.hidden_size
        if config.use_task_embedding:
            input_dim += config.task_embedding_dim
        
        layers.extend([
            nn.Linear(input_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        ])
        
        self.transform = nn.Sequential(*layers)
        
        # Output layer based on task type
        if config.task_type == TaskType.CLASSIFICATION:
            self.output = nn.Linear(config.hidden_size, config.num_labels)
        elif config.task_type == TaskType.REGRESSION:
            self.output = nn.Linear(config.hidden_size, 1)
        elif config.task_type == TaskType.BINARY:
            self.output = nn.Linear(config.hidden_size, 1)
        elif config.task_type == TaskType.MULTILABEL:
            self.output = nn.Linear(config.hidden_size, config.num_labels)
        elif config.task_type == TaskType.RANKING:
            self.output = nn.Linear(config.hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for task-specific head.
        
        Args:
            hidden_states: Shared representations
            labels: Task labels
            
        Returns:
            Tuple of (logits, loss)
        """
        batch_size = hidden_states.size(0)
        
        # Add task embedding if configured
        if self.config.use_task_embedding:
            task_emb = self.task_embedding.expand(batch_size, -1)
            hidden_states = torch.cat([hidden_states, task_emb], dim=-1)
        
        # Transform
        transformed = self.transform(hidden_states)
        
        # Output
        logits = self.output(transformed)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)
        
        return logits, loss
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute task-specific loss."""
        if self.config.task_type == TaskType.CLASSIFICATION:
            if self.config.label_smoothing > 0:
                # Label smoothing
                num_classes = logits.size(-1)
                smooth_labels = torch.full_like(
                    logits,
                    self.config.label_smoothing / num_classes
                )
                smooth_labels.scatter_(
                    -1,
                    labels.unsqueeze(-1),
                    1.0 - self.config.label_smoothing
                )
                log_probs = F.log_softmax(logits, dim=-1)
                loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
            else:
                loss = F.cross_entropy(logits, labels)
                
        elif self.config.task_type == TaskType.REGRESSION:
            loss = F.mse_loss(logits.squeeze(-1), labels.float())
            
        elif self.config.task_type == TaskType.BINARY:
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(-1),
                labels.float()
            )
            
        elif self.config.task_type == TaskType.MULTILABEL:
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            
        elif self.config.task_type == TaskType.RANKING:
            # Pairwise ranking loss
            loss = F.margin_ranking_loss(
                logits[::2].squeeze(-1),
                logits[1::2].squeeze(-1),
                labels[::2]
            )
        
        return loss


class MultiTaskHead(nn.Module):
    """
    Multi-task learning head with shared and task-specific components.
    
    Implements various multi-task learning strategies:
    1. Hard parameter sharing
    2. Soft parameter sharing
    3. Cross-stitch networks
    4. Uncertainty weighting
    5. Gradient balancing
    """
    
    def __init__(
        self,
        task_configs: List[TaskConfig],
        shared_size: int = 768,
        use_uncertainty_weighting: bool = False,
        use_gradient_balancing: bool = False,
        use_cross_stitch: bool = False,
        adversarial_loss_weight: float = 0.0
    ):
        """
        Initialize multi-task head.
        
        Args:
            task_configs: Configuration for each task
            shared_size: Size of shared representation
            use_uncertainty_weighting: Use uncertainty-based weighting
            use_gradient_balancing: Balance gradients across tasks
            use_cross_stitch: Use cross-stitch networks
            adversarial_loss_weight: Weight for adversarial loss
        """
        super().__init__()
        
        self.task_configs = {config.name: config for config in task_configs}
        self.num_tasks = len(task_configs)
        self.shared_size = shared_size
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.use_gradient_balancing = use_gradient_balancing
        self.use_cross_stitch = use_cross_stitch
        self.adversarial_loss_weight = adversarial_loss_weight
        
        # Shared layers
        self.shared_transform = nn.Sequential(
            nn.Linear(shared_size, shared_size),
            nn.LayerNorm(shared_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            config.name: TaskSpecificHead(config)
            for config in task_configs
        })
        
        # Uncertainty weighting
        if use_uncertainty_weighting:
            self.uncertainty_weighter = UncertaintyWeighting(self.num_tasks)
        
        # Cross-stitch networks
        if use_cross_stitch:
            self.cross_stitch = nn.Parameter(
                torch.eye(self.num_tasks) + 0.1 * torch.randn(self.num_tasks, self.num_tasks)
            )
        
        # Task discriminator for adversarial training
        if adversarial_loss_weight > 0:
            self.task_discriminator = nn.Sequential(
                nn.Linear(shared_size, shared_size // 2),
                nn.ReLU(),
                nn.Linear(shared_size // 2, self.num_tasks)
            )
        
        logger.info(
            f"Initialized MultiTaskHead with {self.num_tasks} tasks: "
            f"{list(self.task_configs.keys())}"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_labels: Optional[Dict[str, torch.Tensor]] = None,
        task_name: Optional[str] = None,
        return_all_tasks: bool = False
    ) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass through multi-task head.
        
        Args:
            hidden_states: Input hidden states
            task_labels: Dictionary of labels per task
            task_name: Specific task to compute (None for all)
            return_all_tasks: Return predictions for all tasks
            
        Returns:
            Dictionary of (logits, loss) per task
        """
        # Shared transformation
        shared = self.shared_transform(hidden_states)
        
        # Cross-stitch if configured
        if self.use_cross_stitch and self.num_tasks > 1:
            # Split shared representation for each task
            task_reprs = shared.unsqueeze(1).repeat(1, self.num_tasks, 1)
            # Apply cross-stitch
            task_reprs = torch.einsum('btd,ts->bsd', task_reprs, self.cross_stitch)
        else:
            task_reprs = {task: shared for task in self.task_configs}
        
        # Task predictions
        outputs = {}
        losses = []
        
        tasks_to_compute = [task_name] if task_name else self.task_configs.keys()
        
        for i, task in enumerate(tasks_to_compute):
            if task in self.task_heads:
                # Get task representation
                if self.use_cross_stitch and self.num_tasks > 1:
                    task_repr = task_reprs[:, i]
                else:
                    task_repr = shared
                
                # Get task labels
                labels = task_labels.get(task) if task_labels else None
                
                # Task-specific forward
                logits, loss = self.task_heads[task](task_repr, labels)
                
                outputs[task] = (logits, loss)
                if loss is not None:
                    losses.append(loss)
        
        # Combine losses
        if losses:
            if self.use_uncertainty_weighting:
                total_loss = self.uncertainty_weighter(losses)
            else:
                # Weighted sum
                weights = [self.task_configs[task].weight for task in tasks_to_compute]
                total_loss = sum(w * l for w, l in zip(weights, losses))
            
            # Add adversarial loss if configured
            if self.adversarial_loss_weight > 0 and task_labels:
                adv_loss = self._compute_adversarial_loss(shared, task_labels)
                total_loss += self.adversarial_loss_weight * adv_loss
            
            # Store total loss
            outputs["total_loss"] = (None, total_loss)
        
        return outputs
    
    def _compute_adversarial_loss(
        self,
        shared: torch.Tensor,
        task_labels: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute adversarial loss to encourage task-invariant representations.
        
        Args:
            shared: Shared representations
            task_labels: Task labels
            
        Returns:
            Adversarial loss
        """
        # Apply gradient reversal
        reversed_shared = GradientReversalLayer.apply(shared, 1.0)
        
        # Predict task from reversed features
        task_logits = self.task_discriminator(reversed_shared)
        
        # Create task indices
        batch_size = shared.size(0)
        task_indices = []
        
        for i, task in enumerate(self.task_configs.keys()):
            if task in task_labels:
                task_indices.extend([i] * task_labels[task].size(0))
        
        task_indices = torch.tensor(task_indices, device=shared.device)
        
        # Compute cross-entropy loss
        adv_loss = F.cross_entropy(task_logits, task_indices)
        
        return adv_loss
    
    def get_task_gradients(self) -> Dict[str, float]:
        """
        Get gradient norms for each task.
        
        Returns:
            Dictionary of gradient norms per task
        """
        grad_norms = {}
        
        for task_name, task_head in self.task_heads.items():
            total_norm = 0
            for param in task_head.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms[task_name] = total_norm
        
        return grad_norms
