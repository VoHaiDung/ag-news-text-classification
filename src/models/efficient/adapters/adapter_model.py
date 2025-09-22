"""
Adapter-based Efficient Fine-tuning Implementation
===================================================

Implementation of adapter modules for parameter-efficient transfer learning,
based on:
- Houlsby et al. (2019): "Parameter-Efficient Transfer Learning for NLP"
- Pfeiffer et al. (2020): "AdapterHub: A Framework for Adapting Transformers"
- He et al. (2021): "Towards a Unified View of Parameter-Efficient Transfer Learning"

Mathematical Foundation:
Adapters add small bottleneck layers:
h <- h + f(hW_down)W_up
where W_down ∈ R^(d×r), W_up ∈ R^(r×d), and r << d

This adds only 2rd parameters per adapter instead of d² for full fine-tuning.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import MODELS
from src.core.exceptions import ModelInitializationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AdapterConfig:
    """Configuration for adapter modules."""
    adapter_size: int = 64  # Bottleneck dimension
    adapter_act: str = "relu"  # Activation function
    adapter_initializer_range: float = 0.0002
    non_linearity: str = "relu"
    dropout: float = 0.1
    residual_before_ln: bool = True
    adapter_scalar: float = 1.0  # Scaling factor for adapter output
    use_parallel_adapter: bool = False  # Parallel vs sequential
    use_multi_head_adapter: bool = False  # Multi-head adapters
    num_adapter_heads: int = 1
    share_adapter_weights: bool = False  # Share weights across layers
    trainable_layer_norm: bool = False
    use_gating: bool = False  # Gated adapter fusion
    mh_adapter: bool = False  # Multi-head adapter
    output_adapter: bool = True
    reduction_factor: int = 16  # Reduction factor for bottleneck


class AdapterLayer(nn.Module):
    """
    Single adapter layer implementation.
    
    Implements a bottleneck architecture that adds task-specific parameters
    while keeping the pretrained model frozen.
    """
    
    def __init__(self, config: AdapterConfig, input_size: int):
        """
        Initialize adapter layer.
        
        Args:
            config: Adapter configuration
            input_size: Input dimension size
        """
        super().__init__()
        self.config = config
        self.input_size = input_size
        
        # Calculate adapter size
        self.adapter_size = config.adapter_size or (input_size // config.reduction_factor)
        
        # Down projection
        self.adapter_down = nn.Linear(input_size, self.adapter_size)
        
        # Non-linearity
        self.activation = self._get_activation(config.non_linearity)
        
        # Up projection
        self.adapter_up = nn.Linear(self.adapter_size, input_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Optional layer norm
        if config.trainable_layer_norm:
            self.layer_norm = nn.LayerNorm(input_size)
        else:
            self.layer_norm = None
        
        # Optional gating mechanism
        if config.use_gating:
            self.gate = nn.Linear(input_size, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.ones_(self.gate.bias)
        else:
            self.gate = None
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "swish": nn.SiLU(),
            "leaky_relu": nn.LeakyReLU(0.1)
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """Initialize adapter weights."""
        # Initialize down projection
        nn.init.normal_(
            self.adapter_down.weight,
            std=self.config.adapter_initializer_range
        )
        nn.init.zeros_(self.adapter_down.bias)
        
        # Initialize up projection to near zero for residual connection
        nn.init.normal_(
            self.adapter_up.weight,
            std=self.config.adapter_initializer_range
        )
        nn.init.zeros_(self.adapter_up.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Adapted hidden states
        """
        # Store residual
        residual = hidden_states
        
        # Apply layer norm if configured
        if self.layer_norm and self.config.residual_before_ln:
            hidden_states = self.layer_norm(hidden_states)
        
        # Down projection
        down_projected = self.adapter_down(hidden_states)
        
        # Activation
        down_projected = self.activation(down_projected)
        
        # Dropout
        down_projected = self.dropout(down_projected)
        
        # Up projection
        up_projected = self.adapter_up(down_projected)
        
        # Apply gating if configured
        if self.gate:
            gate_value = torch.sigmoid(self.gate(hidden_states))
            up_projected = gate_value * up_projected
        
        # Scale adapter output
        up_projected = up_projected * self.config.adapter_scalar
        
        # Add residual connection
        output = residual + up_projected
        
        # Apply layer norm after if configured
        if self.layer_norm and not self.config.residual_before_ln:
            output = self.layer_norm(output)
        
        return output


class MultiHeadAdapter(nn.Module):
    """
    Multi-head adapter for capturing diverse adaptations.
    
    Based on multi-head attention principle but applied to adapters.
    """
    
    def __init__(self, config: AdapterConfig, input_size: int):
        """
        Initialize multi-head adapter.
        
        Args:
            config: Adapter configuration
            input_size: Input dimension
        """
        super().__init__()
        self.num_heads = config.num_adapter_heads
        self.head_size = input_size // self.num_heads
        
        # Create multiple adapter heads
        self.adapter_heads = nn.ModuleList([
            AdapterLayer(config, self.head_size)
            for _ in range(self.num_heads)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(input_size, input_size)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-head adapter.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            
        Returns:
            Adapted hidden states
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Reshape for multi-head processing
        hidden_states = hidden_states.view(
            batch_size, seq_len, self.num_heads, self.head_size
        )
        hidden_states = hidden_states.transpose(1, 2)  # [batch, heads, seq, head_size]
        
        # Process each head
        head_outputs = []
        for i, adapter_head in enumerate(self.adapter_heads):
            head_output = adapter_head(hidden_states[:, i])
            head_outputs.append(head_output)
        
        # Concatenate heads
        concatenated = torch.cat(head_outputs, dim=-1)
        concatenated = concatenated.view(batch_size, seq_len, hidden_size)
        
        # Output projection
        output = self.output_proj(concatenated)
        
        return output


class AdapterFusion(nn.Module):
    """
    Adapter fusion for combining multiple adapters.
    
    Based on Pfeiffer et al. (2021): "AdapterFusion: Non-Destructive Task Composition"
    """
    
    def __init__(
        self,
        num_adapters: int,
        hidden_size: int,
        dropout: float = 0.1
    ):
        """
        Initialize adapter fusion.
        
        Args:
            num_adapters: Number of adapters to fuse
            hidden_size: Hidden dimension size
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_adapters = num_adapters
        
        # Fusion attention
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(
        self,
        adapter_outputs: List[torch.Tensor],
        query_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse multiple adapter outputs.
        
        Args:
            adapter_outputs: List of adapter outputs
            query_states: Query hidden states
            
        Returns:
            Fused adapter output
        """
        # Stack adapter outputs
        stacked_outputs = torch.stack(adapter_outputs, dim=1)  # [batch, num_adapters, seq, hidden]
        
        # Compute attention scores
        Q = self.query(query_states)  # [batch, seq, hidden]
        K = self.key(stacked_outputs)  # [batch, num_adapters, seq, hidden]
        V = self.value(stacked_outputs)
        
        # Attention scores
        scores = torch.einsum('bsh,bash->bas', Q, K) / (self.temperature * math.sqrt(Q.size(-1)))
        attention_weights = F.softmax(scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum of adapter outputs
        fused_output = torch.einsum('bas,bash->bsh', attention_weights, V)
        
        return fused_output


@MODELS.register("adapter", aliases=["adapter_model", "peft_adapter"])
class AdapterModel(AGNewsBaseModel):
    """
    Model with adapter-based efficient fine-tuning.
    
    Provides parameter-efficient transfer learning by:
    1. Keeping pretrained weights frozen
    2. Adding small trainable adapter modules
    3. Achieving comparable performance with ~1% of parameters
    
    Supports various adapter configurations:
    - Standard sequential adapters
    - Parallel adapters
    - Multi-head adapters
    - Adapter fusion for multi-task learning
    """
    
    def __init__(
        self,
        base_model: AGNewsBaseModel,
        config: Optional[AdapterConfig] = None,
        adapter_names: Optional[List[str]] = None
    ):
        """
        Initialize adapter model.
        
        Args:
            base_model: Pretrained base model
            config: Adapter configuration
            adapter_names: Names of adapters (for multi-adapter setup)
        """
        super().__init__()
        
        self.base_model = base_model
        self.config = config or AdapterConfig()
        self.adapter_names = adapter_names or ["default"]
        
        # Freeze base model
        self._freeze_base_model()
        
        # Add adapters to model
        self._inject_adapters()
        
        # Initialize adapter fusion if multiple adapters
        if len(self.adapter_names) > 1:
            self._init_adapter_fusion()
        
        # Log parameter statistics
        self._log_parameter_stats()
    
    def _freeze_base_model(self):
        """Freeze all parameters in the base model."""
        for param in self.base_model.parameters():
            param.requires_grad = False
        logger.info("Froze all base model parameters")
    
    def _inject_adapters(self):
        """Inject adapter layers into the base model."""
        self.adapters = nn.ModuleDict()
        
        # Find transformer layers in base model
        if hasattr(self.base_model, 'transformer'):
            transformer = self.base_model.transformer
        elif hasattr(self.base_model, 'encoder'):
            transformer = self.base_model.encoder
        else:
            raise ModelInitializationError("Could not find transformer layers in base model")
        
        # Inject adapters into each transformer layer
        for name in self.adapter_names:
            layer_adapters = nn.ModuleList()
            
            # Iterate through transformer layers
            for layer_idx, layer in enumerate(transformer.layer):
                # Get hidden size from layer
                if hasattr(layer, 'attention'):
                    hidden_size = layer.attention.self.query.in_features
                else:
                    hidden_size = 768  # Default
                
                # Create adapter for this layer
                if self.config.use_multi_head_adapter:
                    adapter = MultiHeadAdapter(self.config, hidden_size)
                else:
                    adapter = AdapterLayer(self.config, hidden_size)
                
                layer_adapters.append(adapter)
                
                # Inject adapter into forward pass (monkey patching)
                self._inject_adapter_forward(layer, adapter)
            
            self.adapters[name] = layer_adapters
        
        logger.info(f"Injected {len(self.adapter_names)} adapters into {len(layer_adapters)} layers")
    
    def _inject_adapter_forward(self, layer: nn.Module, adapter: nn.Module):
        """
        Inject adapter into layer's forward pass.
        
        Args:
            layer: Transformer layer
            adapter: Adapter module
        """
        # Store original forward
        original_forward = layer.forward
        
        def forward_with_adapter(*args, **kwargs):
            # Get output from original forward
            outputs = original_forward(*args, **kwargs)
            
            # Apply adapter to hidden states
            if isinstance(outputs, tuple):
                hidden_states = outputs[0]
                adapted = adapter(hidden_states)
                return (adapted,) + outputs[1:]
            else:
                return adapter(outputs)
        
        # Replace forward method
        layer.forward = forward_with_adapter
    
    def _init_adapter_fusion(self):
        """Initialize adapter fusion for multiple adapters."""
        # Get hidden size
        if hasattr(self.base_model, 'hidden_size'):
            hidden_size = self.base_model.hidden_size
        else:
            hidden_size = 768
        
        self.adapter_fusion = AdapterFusion(
            num_adapters=len(self.adapter_names),
            hidden_size=hidden_size,
            dropout=self.config.dropout
        )
    
    def _log_parameter_stats(self):
        """Log parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        base_params = sum(p.numel() for p in self.base_model.parameters())
        
        reduction_ratio = 1 - (trainable_params / base_params)
        
        logger.info(
            f"Adapter Parameter Statistics:\n"
            f"  Base model parameters: {base_params:,}\n"
            f"  Adapter parameters: {trainable_params:,}\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Parameter reduction: {reduction_ratio:.1%}\n"
            f"  Compression ratio: {base_params/trainable_params:.1f}x"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        adapter_name: Optional[str] = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through adapter model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            adapter_name: Specific adapter to use (for multi-adapter)
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        # Select adapter
        if adapter_name and adapter_name in self.adapter_names:
            # Activate specific adapter
            self.set_active_adapter(adapter_name)
        
        # Forward through base model with adapters
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def set_active_adapter(self, adapter_name: str):
        """
        Set active adapter for inference.
        
        Args:
            adapter_name: Name of adapter to activate
        """
        if adapter_name not in self.adapter_names:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        # Implementation would activate specific adapter
        self.active_adapter = adapter_name
        logger.info(f"Set active adapter to '{adapter_name}'")
    
    def save_adapter(self, save_path: str, adapter_name: Optional[str] = None):
        """
        Save adapter weights.
        
        Args:
            save_path: Path to save adapter
            adapter_name: Specific adapter to save
        """
        adapter_name = adapter_name or "default"
        
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        # Save adapter state dict
        torch.save(
            self.adapters[adapter_name].state_dict(),
            save_path
        )
        logger.info(f"Saved adapter '{adapter_name}' to {save_path}")
    
    def load_adapter(self, load_path: str, adapter_name: Optional[str] = None):
        """
        Load adapter weights.
        
        Args:
            load_path: Path to load adapter from
            adapter_name: Name for loaded adapter
        """
        adapter_name = adapter_name or "default"
        
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not found")
        
        # Load adapter state dict
        state_dict = torch.load(load_path, map_location="cpu")
        self.adapters[adapter_name].load_state_dict(state_dict)
        logger.info(f"Loaded adapter '{adapter_name}' from {load_path}")
