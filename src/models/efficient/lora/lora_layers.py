"""
LoRA Layers Implementation
==========================

Low-level implementation of LoRA layers with advanced features.

Based on:
- Hu et al. (2021): "LoRA: Low-Rank Adaptation"
- Dettmers et al. (2023): "QLoRA: Efficient Finetuning of Quantized LLMs"
- Liu et al. (2024): "DoRA: Weight-Decomposed Low-Rank Adaptation"

Mathematical Foundation:
LoRA factorization: ΔW = BA where B ∈ R^(d×r), A ∈ R^(r×k)
Forward pass: h = xW₀ + x(BA)α/r = xW₀ + xΔW_scaled

Author: Võ Hải Dũng
License: MIT
"""

import math
from typing import Optional, Dict, Any, Tuple, Union, List
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.efficient.lora.lora_config import LoRAConfig, LoRAInitMethod
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoRALayer(nn.Module):
    """
    Base class for LoRA layers.
    
    Implements the core LoRA decomposition with support for:
    - Multiple initialization strategies
    - Rank adaptation
    - Weight merging
    - Gradient caching
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        layer_id: int = 0
    ):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension  
            config: LoRA configuration
            layer_id: Layer identifier for multi-layer setups
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        self.layer_id = layer_id
        
        # Get rank for this specific layer
        self.r = config.get_module_rank(f"layer_{layer_id}")
        self.scaling = config.lora_alpha / self.r
        
        # Initialize LoRA matrices
        self._init_lora_matrices()
        
        # Initialize dropout
        self.lora_dropout = nn.Dropout(p=config.lora_dropout) if config.lora_dropout > 0 else nn.Identity()
        
        # Weight merging flag
        self.merged = False
        
        # Gradient cache for efficient updates
        if config.enable_gradient_caching:
            self.register_buffer('cached_gradient_a', None)
            self.register_buffer('cached_gradient_b', None)
        
        # Statistics tracking
        self.forward_count = 0
        self.gradient_norm_history = []
        
        logger.debug(
            f"Initialized LoRA layer {layer_id}: "
            f"{in_features}->{out_features}, rank={self.r}"
        )
    
    def _init_lora_matrices(self):
        """Initialize LoRA A and B matrices"""
        # Matrix A: (r × in_features)
        self.lora_A = Parameter(torch.empty(self.r, self.in_features))
        
        # Matrix B: (out_features × r)
        self.lora_B = Parameter(torch.empty(self.out_features, self.r))
        
        # Apply initialization
        self._init_matrix(self.lora_A, self.config.init_method_a, fan_in=True)
        self._init_matrix(self.lora_B, self.config.init_method_b, fan_in=False)
    
    def _init_matrix(self, matrix: Parameter, method: LoRAInitMethod, fan_in: bool = True):
        """
        Initialize a matrix with specified method.
        
        Args:
            matrix: Matrix to initialize
            method: Initialization method
            fan_in: Whether this is the input matrix (A)
        """
        with torch.no_grad():
            if method == LoRAInitMethod.GAUSSIAN:
                nn.init.normal_(matrix, mean=0.0, std=self.config.init_std)
            elif method == LoRAInitMethod.UNIFORM:
                bound = 1 / math.sqrt(matrix.size(1))
                nn.init.uniform_(matrix, -bound, bound)
            elif method == LoRAInitMethod.XAVIER:
                nn.init.xavier_uniform_(matrix)
            elif method == LoRAInitMethod.KAIMING:
                # Kaiming initialization with proper mode
                if self.config.use_kaiming_scale:
                    nn.init.kaiming_uniform_(matrix, a=math.sqrt(5), mode='fan_in' if fan_in else 'fan_out')
                else:
                    nn.init.kaiming_uniform_(matrix, a=0, mode='fan_in' if fan_in else 'fan_out')
            elif method == LoRAInitMethod.ZEROS:
                nn.init.zeros_(matrix)
            elif method == LoRAInitMethod.ORTHOGONAL:
                nn.init.orthogonal_(matrix)
            else:
                raise ValueError(f"Unknown initialization method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LoRA layer.
        
        Computes: output = x @ (BA)ᵀ × (α/r)
        
        Args:
            x: Input tensor [batch_size, seq_len, in_features]
            
        Returns:
            LoRA output [batch_size, seq_len, out_features]
        """
        self.forward_count += 1
        
        if not self.merged:
            # Apply dropout to input
            x_dropped = self.lora_dropout(x)
            
            # Compute LoRA: (x @ Aᵀ) @ Bᵀ
            # Shape: [batch, seq, in] @ [in, r] @ [r, out] = [batch, seq, out]
            lora_output = x_dropped @ self.lora_A.t() @ self.lora_B.t()
            
            # Apply scaling
            return lora_output * self.scaling
        else:
            # If merged, the main linear layer already contains LoRA
            return torch.zeros_like(x[..., :self.out_features])
    
    def merge_weights(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Merge LoRA weights into main weights.
        
        Args:
            weight: Original weight matrix [out, in]
            
        Returns:
            Merged weight matrix
        """
        if not self.merged:
            # Compute ΔW = B @ A × (α/r)
            delta_w = self.lora_B @ self.lora_A * self.scaling
            
            # Add to original weights
            merged_weight = weight + delta_w
            self.merged = True
            
            logger.debug(f"Merged LoRA weights for layer {self.layer_id}")
            return merged_weight
        
        return weight
    
    def unmerge_weights(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Unmerge LoRA weights from main weights.
        
        Args:
            weight: Merged weight matrix
            
        Returns:
            Original weight matrix
        """
        if self.merged:
            # Subtract ΔW = B @ A × (α/r)
            delta_w = self.lora_B @ self.lora_A * self.scaling
            
            # Restore original weights
            original_weight = weight - delta_w
            self.merged = False
            
            logger.debug(f"Unmerged LoRA weights for layer {self.layer_id}")
            return original_weight
        
        return weight
    
    def get_delta_weight(self) -> torch.Tensor:
        """
        Get LoRA weight delta ΔW.
        
        Returns:
            Weight delta matrix [out_features, in_features]
        """
        return self.lora_B @ self.lora_A * self.scaling
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss for LoRA matrices.
        
        Returns:
            Regularization loss scalar
        """
        loss = 0.0
        
        # Orthogonal regularization
        if self.config.orthogonal_regularization:
            # Encourage A^T @ A ≈ I
            ata = self.lora_A @ self.lora_A.t()
            eye = torch.eye(self.r, device=ata.device)
            loss += torch.norm(ata - eye, p='fro') * 0.01
            
            # Encourage B^T @ B ≈ I  
            btb = self.lora_B.t() @ self.lora_B
            loss += torch.norm(btb - eye, p='fro') * 0.01
        
        return loss
    
    def compute_importance(self) -> float:
        """
        Compute importance score for this LoRA layer.
        
        Returns:
            Importance score
        """
        if self.config.importance_metric == "magnitude":
            # Magnitude-based importance
            importance = torch.norm(self.lora_B) * torch.norm(self.lora_A)
        elif self.config.importance_metric == "gradient":
            # Gradient-based importance
            if self.lora_A.grad is not None and self.lora_B.grad is not None:
                importance = torch.norm(self.lora_A.grad) * torch.norm(self.lora_B.grad)
            else:
                importance = torch.tensor(0.0)
        else:
            # Default to magnitude
            importance = torch.norm(self.lora_B) * torch.norm(self.lora_A)
        
        return importance.item()


class Linear(nn.Linear):
    """
    LoRA-enhanced Linear layer.
    
    Drop-in replacement for nn.Linear with LoRA support.
    Maintains full compatibility with standard Linear layer.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        bias: bool = True,
        device=None,
        dtype=None,
        layer_id: int = 0
    ):
        """
        Initialize LoRA Linear layer.
        
        Args:
            in_features: Input features
            out_features: Output features
            config: LoRA configuration
            bias: Whether to use bias
            device: Device placement
            dtype: Data type
            layer_id: Layer identifier
        """
        # Initialize parent Linear layer
        super().__init__(in_features, out_features, bias, device, dtype)
        
        self.config = config
        self.lora_layer = LoRALayer(in_features, out_features, config, layer_id)
        
        # Freeze pretrained weights by default
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = config.bias != "none"
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Standard linear transformation
        output = F.linear(x, self.weight, self.bias)
        
        # Add LoRA adaptation
        output = output + self.lora_layer(x)
        
        return output
    
    def merge_and_unload(self) -> None:
        """Merge LoRA weights and unload LoRA parameters"""
        with torch.no_grad():
            self.weight.data = self.lora_layer.merge_weights(self.weight.data)
            
        # Remove LoRA parameters to save memory
        del self.lora_layer
        
        logger.info("Merged and unloaded LoRA parameters")


class DoRALayer(LoRALayer):
    """
    DoRA (Weight-Decomposed LoRA) Layer.
    
    Decomposes pretrained weights into magnitude and direction:
    W = m * v where ||v|| = 1
    
    Then applies LoRA to the directional component.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        layer_id: int = 0
    ):
        """Initialize DoRA layer"""
        super().__init__(in_features, out_features, config, layer_id)
        
        # Magnitude vector for weight decomposition
        self.magnitude = Parameter(torch.ones(out_features))
    
    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        DoRA forward pass.
        
        Args:
            x: Input tensor
            weight: Original weight matrix
            
        Returns:
            DoRA output
        """
        # Get LoRA delta
        lora_output = super().forward(x)
        
        # Decompose weight into magnitude and direction
        weight_norm = torch.norm(weight, dim=1, keepdim=True)
        weight_direction = weight / (weight_norm + 1e-8)
        
        # Apply magnitude scaling
        scaled_direction = weight_direction * self.magnitude.unsqueeze(1)
        
        # Combine with LoRA
        output = x @ scaled_direction.t() + lora_output
        
        return output


class RankAdaptiveLoRALayer(LoRALayer):
    """
    Rank-Adaptive LoRA Layer.
    
    Dynamically adjusts rank during training based on importance scores.
    Implements AdaLoRA-style rank pruning.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig,
        layer_id: int = 0
    ):
        """Initialize rank-adaptive layer"""
        # Start with initial rank
        config_copy = config
        original_r = config.r
        if config.enable_rank_adaptation:
            config_copy.r = config.initial_rank
        
        super().__init__(in_features, out_features, config_copy, layer_id)
        
        # Restore original r for reference
        self.target_rank = original_r if not config.enable_rank_adaptation else config.target_rank
        
        # Importance scores for each rank
        self.register_buffer(
            'importance_scores',
            torch.ones(self.r)
        )
        
        # Current active rank
        self.active_rank = self.r
        
        # Pruning schedule
        self.pruning_steps = []
        self.current_step = 0
    
    def update_rank(self, target_rank: Optional[int] = None):
        """
        Update active rank based on importance.
        
        Args:
            target_rank: New target rank (optional)
        """
        if target_rank is None:
            target_rank = self.target_rank
            
        if target_rank >= self.active_rank:
            return
        
        # Compute importance scores
        importance = self._compute_rank_importance()
        
        # Sort by importance
        _, indices = torch.sort(importance, descending=True)
        keep_indices = indices[:target_rank]
        
        # Prune matrices
        with torch.no_grad():
            self.lora_A.data = self.lora_A.data[keep_indices]
            self.lora_B.data = self.lora_B.data[:, keep_indices]
        
        self.active_rank = target_rank
        self.r = target_rank
        self.scaling = self.config.lora_alpha / self.r
        
        logger.info(f"Updated rank to {target_rank} for layer {self.layer_id}")
    
    def _compute_rank_importance(self) -> torch.Tensor:
        """
        Compute importance scores for ranks.
        
        Returns:
            Importance scores
        """
        importance = torch.zeros(self.r)
        
        for i in range(self.r):
            # Importance = ||B[:, i]|| × ||A[i, :]||
            importance[i] = (
                torch.norm(self.lora_B[:, i]) * 
                torch.norm(self.lora_A[i, :])
            )
        
        self.importance_scores = importance
        return importance
    
    def step(self):
        """Update step counter and potentially prune"""
        self.current_step += 1
        
        # Check if we should prune
        if self.config.enable_rank_adaptation and self.current_step in self.pruning_steps:
            # Gradual rank reduction
            progress = self.current_step / max(self.pruning_steps)
            current_target = int(
                self.config.initial_rank - 
                (self.config.initial_rank - self.target_rank) * progress
            )
            self.update_rank(current_target)


def create_lora_layer(
    layer: nn.Module,
    config: LoRAConfig,
    layer_name: str = "",
    layer_id: int = 0
) -> nn.Module:
    """
    Factory function to create LoRA-enhanced layer.
    
    Args:
        layer: Original layer to enhance
        config: LoRA configuration
        layer_name: Name of the layer
        layer_id: Layer identifier
        
    Returns:
        LoRA-enhanced layer
    """
    if isinstance(layer, nn.Linear):
        # Check if this layer should be adapted
        if not config.should_adapt_module(layer_name):
            logger.debug(f"Skipping LoRA for layer: {layer_name}")
            return layer
        
        # Replace with LoRA Linear
        lora_layer = Linear(
            layer.in_features,
            layer.out_features,
            config,
            bias=layer.bias is not None,
            layer_id=layer_id
        )
        
        # Copy weights
        with torch.no_grad():
            lora_layer.weight.copy_(layer.weight)
            if layer.bias is not None:
                lora_layer.bias.copy_(layer.bias)
        
        logger.info(f"Created LoRA layer for: {layer_name}")
        return lora_layer
    
    # Return original if not supported
    logger.warning(f"Layer type {type(layer)} not supported for LoRA")
    return layer
