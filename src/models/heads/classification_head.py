"""
Classification Head Implementations
====================================

This module implements various classification heads for transformer models,
following research from:
- Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"
- Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- Chen et al. (2020): "Simple and Deep Graph Convolutional Networks"

Mathematical Foundation:
Classification head transforms hidden representations h ∈ ℝ^d to class logits:
y = σ(W₂·g(W₁·h + b₁) + b₂)
where g is activation function and σ is optional output activation.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.core.exceptions import ModelInitializationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ActivationType(Enum):
    """Supported activation functions."""
    RELU = "relu"
    GELU = "gelu"
    TANH = "tanh"
    SWISH = "swish"
    MISH = "mish"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"
    SELU = "selu"
    GLU = "glu"
    GEGLU = "geglu"


@dataclass
class ClassificationHeadConfig:
    """Configuration for classification head."""
    hidden_size: int = 768
    num_labels: int = 4
    dropout_rate: float = 0.1
    hidden_dropout_rate: float = 0.2
    use_layer_norm: bool = True
    use_batch_norm: bool = False
    num_hidden_layers: int = 2
    hidden_act: str = "gelu"
    intermediate_size: Optional[int] = None
    use_residual: bool = False
    use_weight_norm: bool = False
    init_std: float = 0.02
    label_smoothing: float = 0.0
    mixup_alpha: float = 0.0
    use_multi_sample_dropout: bool = False
    num_dropout_samples: int = 5


class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    """Mish activation function: x * tanh(softplus(x))"""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(F.softplus(x))


class GEGLU(nn.Module):
    """
    GEGLU activation function from GLU variants.
    
    Based on Shazeer (2020): "GLU Variants Improve Transformer"
    """
    
    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


def get_activation(activation: str) -> nn.Module:
    """
    Get activation function by name.
    
    Args:
        activation: Activation function name
        
    Returns:
        Activation module
    """
    activations = {
        ActivationType.RELU.value: nn.ReLU(),
        ActivationType.GELU.value: nn.GELU(),
        ActivationType.TANH.value: nn.Tanh(),
        ActivationType.SWISH.value: Swish(),
        ActivationType.MISH.value: Mish(),
        ActivationType.LEAKY_RELU.value: nn.LeakyReLU(0.1),
        ActivationType.ELU.value: nn.ELU(),
        ActivationType.SELU.value: nn.SELU(),
        ActivationType.GLU.value: nn.GLU(),
    }
    
    if activation not in activations:
        raise ValueError(f"Unknown activation: {activation}")
    
    return activations[activation]


class MultiSampleDropout(nn.Module):
    """
    Multi-sample dropout for improved regularization.
    
    Based on Inoue (2019): "Multi-Sample Dropout for Accelerated Training"
    """
    
    def __init__(self, dropout_rate: float, num_samples: int = 5):
        super().__init__()
        self.dropouts = nn.ModuleList([
            nn.Dropout(dropout_rate) for _ in range(num_samples)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x
        
        # Apply multiple dropout samples and average
        outputs = [dropout(x) for dropout in self.dropouts]
        return torch.stack(outputs).mean(dim=0)


class BaseClassificationHead(nn.Module):
    """
    Base classification head with configurable architecture.
    
    Provides a flexible classification head that can be customized
    with different activation functions, normalization strategies,
    and architectural choices.
    """
    
    def __init__(self, config: ClassificationHeadConfig):
        """
        Initialize classification head.
        
        Args:
            config: Classification head configuration
        """
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_labels = config.num_labels
        
        # Build layers
        self._build_layers()
        
        # Initialize weights
        self._init_weights()
        
        logger.info(
            f"Initialized {self.__class__.__name__} with "
            f"{config.num_hidden_layers} hidden layers"
        )
    
    def _build_layers(self):
        """Build classification head layers."""
        layers = []
        
        # Determine dimensions
        intermediate_size = self.config.intermediate_size or self.hidden_size
        
        # Input dropout
        if self.config.use_multi_sample_dropout:
            self.input_dropout = MultiSampleDropout(
                self.config.dropout_rate,
                self.config.num_dropout_samples
            )
        else:
            self.input_dropout = nn.Dropout(self.config.dropout_rate)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        
        for i in range(self.config.num_hidden_layers - 1):
            # Linear transformation
            if i == 0:
                in_dim = self.hidden_size
            else:
                in_dim = intermediate_size
            
            linear = nn.Linear(in_dim, intermediate_size)
            
            # Apply weight normalization if configured
            if self.config.use_weight_norm:
                linear = nn.utils.weight_norm(linear)
            
            self.hidden_layers.append(linear)
            
            # Normalization
            if self.config.use_layer_norm:
                self.hidden_layers.append(nn.LayerNorm(intermediate_size))
            elif self.config.use_batch_norm:
                self.hidden_layers.append(nn.BatchNorm1d(intermediate_size))
            
            # Activation
            if self.config.hidden_act == "geglu":
                self.hidden_layers.append(
                    GEGLU(intermediate_size, intermediate_size)
                )
            else:
                self.hidden_layers.append(get_activation(self.config.hidden_act))
            
            # Dropout
            self.hidden_layers.append(nn.Dropout(self.config.hidden_dropout_rate))
        
        # Output layer
        self.output_layer = nn.Linear(
            intermediate_size if self.config.num_hidden_layers > 1 else self.hidden_size,
            self.num_labels
        )
    
    def _init_weights(self):
        """Initialize weights using truncated normal distribution."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass through classification head.
        
        Args:
            hidden_states: Input hidden states [batch_size, hidden_size]
            labels: Target labels for loss computation
            return_hidden: Return intermediate hidden states
            
        Returns:
            Tuple of (logits, loss, hidden_states)
        """
        # Input dropout
        x = self.input_dropout(hidden_states)
        
        # Store for residual connection
        residual = x
        
        # Pass through hidden layers
        for layer in self.hidden_layers:
            x = layer(x)
        
        # Residual connection if configured
        if self.config.use_residual and self.config.num_hidden_layers > 1:
            # Project residual if dimensions don't match
            if residual.shape[-1] != x.shape[-1]:
                residual = nn.Linear(residual.shape[-1], x.shape[-1])(residual)
            x = x + residual
        
        # Store hidden representation
        hidden = x.clone() if return_hidden else None
        
        # Output projection
        logits = self.output_layer(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)
        
        return logits, loss, hidden
    
    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute classification loss with label smoothing.
        
        Args:
            logits: Model predictions
            labels: Target labels
            
        Returns:
            Computed loss
        """
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
            # Standard cross-entropy
            loss = F.cross_entropy(logits, labels)
        
        return loss


class HierarchicalClassificationHead(BaseClassificationHead):
    """
    Hierarchical classification head for multi-level classification.
    
    Implements hierarchical softmax for efficient large-scale classification
    following Morin & Bengio (2005): "Hierarchical Probabilistic Neural Network"
    """
    
    def __init__(
        self,
        config: ClassificationHeadConfig,
        hierarchy: Optional[Dict[int, List[int]]] = None
    ):
        """
        Initialize hierarchical classification head.
        
        Args:
            config: Classification head configuration
            hierarchy: Hierarchical structure of classes
        """
        super().__init__(config)
        
        self.hierarchy = hierarchy or self._build_default_hierarchy()
        
        # Build hierarchical classifiers
        self._build_hierarchical_layers()
    
    def _build_default_hierarchy(self) -> Dict[int, List[int]]:
        """Build default balanced hierarchy."""
        num_groups = int(np.sqrt(self.num_labels))
        hierarchy = {}
        
        for i in range(num_groups):
            start_idx = i * (self.num_labels // num_groups)
            end_idx = min(start_idx + (self.num_labels // num_groups), self.num_labels)
            hierarchy[i] = list(range(start_idx, end_idx))
        
        return hierarchy
    
    def _build_hierarchical_layers(self):
        """Build hierarchical classification layers."""
        self.group_classifier = nn.Linear(
            self.hidden_size,
            len(self.hierarchy)
        )
        
        self.class_classifiers = nn.ModuleDict({
            str(group_id): nn.Linear(self.hidden_size, len(classes))
            for group_id, classes in self.hierarchy.items()
        })
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        return_hidden: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Hierarchical forward pass.
        
        First predicts group, then predicts class within group.
        """
        # Get base hidden representation
        x = self.input_dropout(hidden_states)
        
        for layer in self.hidden_layers:
            x = layer(x)
        
        hidden = x.clone() if return_hidden else None
        
        # Group prediction
        group_logits = self.group_classifier(x)
        group_probs = F.softmax(group_logits, dim=-1)
        
        # Class prediction within each group
        batch_size = x.size(0)
        class_logits = torch.zeros(batch_size, self.num_labels, device=x.device)
        
        for group_id, classes in self.hierarchy.items():
            group_prob = group_probs[:, group_id:group_id+1]
            local_logits = self.class_classifiers[str(group_id)](x)
            local_probs = F.softmax(local_logits, dim=-1)
            
            # Combine group and class probabilities
            for i, class_idx in enumerate(classes):
                class_logits[:, class_idx] = (
                    torch.log(group_prob[:, 0] + 1e-10) +
                    torch.log(local_probs[:, i] + 1e-10)
                )
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(class_logits, labels)
        
        return class_logits, loss, hidden


class MultiTaskClassificationHead(nn.Module):
    """
    Multi-task classification head for joint learning.
    
    Based on Caruana (1997): "Multitask Learning" and
    Liu et al. (2019): "Multi-Task Deep Neural Networks"
    """
    
    def __init__(
        self,
        config: ClassificationHeadConfig,
        task_configs: Dict[str, Dict[str, Any]]
    ):
        """
        Initialize multi-task classification head.
        
        Args:
            config: Base configuration
            task_configs: Configuration for each task
        """
        super().__init__()
        
        self.config = config
        self.task_configs = task_configs
        
        # Shared layers
        self.shared_layers = self._build_shared_layers()
        
        # Task-specific heads
        self.task_heads = nn.ModuleDict({
            task_name: self._build_task_head(task_config)
            for task_name, task_config in task_configs.items()
        })
        
        logger.info(f"Initialized multi-task head with {len(task_configs)} tasks")
    
    def _build_shared_layers(self) -> nn.Module:
        """Build shared representation layers."""
        layers = []
        
        # Shared transformation
        layers.append(nn.Linear(self.config.hidden_size, self.config.hidden_size))
        layers.append(nn.LayerNorm(self.config.hidden_size))
        layers.append(get_activation(self.config.hidden_act))
        layers.append(nn.Dropout(self.config.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _build_task_head(self, task_config: Dict[str, Any]) -> nn.Module:
        """Build task-specific classification head."""
        return nn.Sequential(
            nn.Linear(self.config.hidden_size, task_config.get("hidden_size", 256)),
            get_activation(self.config.hidden_act),
            nn.Dropout(task_config.get("dropout", 0.1)),
            nn.Linear(task_config.get("hidden_size", 256), task_config["num_labels"])
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        task_name: Optional[str] = None,
        labels: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Multi-task forward pass.
        
        Args:
            hidden_states: Input hidden states
            task_name: Specific task to run (None for all tasks)
            labels: Dictionary of labels per task
            
        Returns:
            Dictionary of (logits, loss) per task
        """
        # Shared representation
        shared = self.shared_layers(hidden_states)
        
        outputs = {}
        
        # Process specified task or all tasks
        tasks_to_process = [task_name] if task_name else self.task_heads.keys()
        
        for task in tasks_to_process:
            if task in self.task_heads:
                logits = self.task_heads[task](shared)
                
                # Compute task loss
                loss = None
                if labels and task in labels:
                    loss = F.cross_entropy(logits, labels[task])
                
                outputs[task] = (logits, loss)
        
        return outputs
