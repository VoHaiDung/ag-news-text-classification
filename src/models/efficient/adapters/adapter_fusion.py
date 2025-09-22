"""
Adapter Fusion Implementation for Multi-Task Learning
======================================================

Implementation of adapter fusion for combining multiple task-specific adapters,
based on:
- Pfeiffer et al. (2021): "AdapterFusion: Non-Destructive Task Composition for Transfer Learning"
- Rücklé et al. (2021): "AdapterDrop: On the Efficiency of Adapters in Transformers"
- Wang et al. (2021): "Efficient Multi-Task Learning with Adaptive Feature Fusion"

Mathematical Foundation:
Adapter fusion combines N adapters using attention mechanism:
F(x) = Σᵢ αᵢ · Aᵢ(x)
where αᵢ = softmax(Wₖ·x)ᵀ(Wᵥ·Aᵢ(x))/√d

This enables non-destructive knowledge transfer across tasks.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.core.exceptions import ModelInitializationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AdapterFusionConfig:
    """Configuration for adapter fusion."""
    # Fusion architecture
    fusion_type: str = "attention"  # "attention", "gating", "mixture"
    num_adapters: int = 3
    hidden_size: int = 768
    adapter_size: int = 64
    
    # Attention fusion
    num_attention_heads: int = 1
    attention_dropout: float = 0.1
    use_query_projection: bool = True
    use_key_value_projection: bool = True
    
    # Gating fusion
    gating_type: str = "scalar"  # "scalar", "vector", "matrix"
    use_learnable_gates: bool = True
    
    # Mixture of experts
    num_experts: int = 4
    expert_capacity: float = 1.25
    use_load_balancing: bool = True
    load_balance_loss_weight: float = 0.01
    
    # Training
    adapter_dropout: float = 0.0
    regularization_weight: float = 0.01
    temperature: float = 1.0
    use_adapter_drop: bool = False
    adapter_drop_prob: float = 0.5
    
    # Efficiency
    share_adapters: bool = False
    dynamic_adapter_selection: bool = False
    sparsity_threshold: float = 0.1


class AdapterModule(nn.Module):
    """
    Single adapter module with bottleneck architecture.
    
    Implements down-projection → non-linearity → up-projection.
    """
    
    def __init__(
        self,
        input_size: int,
        adapter_size: int,
        dropout: float = 0.1
    ):
        """
        Initialize adapter module.
        
        Args:
            input_size: Input dimension
            adapter_size: Bottleneck dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Down projection
        self.down_project = nn.Linear(input_size, adapter_size)
        
        # Non-linearity
        self.activation = nn.ReLU()
        
        # Up projection
        self.up_project = nn.Linear(adapter_size, input_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(input_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights for residual connection."""
        # Initialize down projection
        nn.init.normal_(self.down_project.weight, std=0.02)
        nn.init.zeros_(self.down_project.bias)
        
        # Initialize up projection near zero for residual
        nn.init.normal_(self.up_project.weight, std=0.0002)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            Adapted output with residual connection
        """
        residual = x
        
        # Apply adapter
        x = self.down_project(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_project(x)
        
        # Residual connection
        x = x + residual
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class AttentionFusion(nn.Module):
    """
    Attention-based adapter fusion.
    
    Uses multi-head attention to combine adapter outputs based on
    input-dependent weights.
    """
    
    def __init__(self, config: AdapterFusionConfig):
        """
        Initialize attention fusion.
        
        Args:
            config: Fusion configuration
        """
        super().__init__()
        
        self.num_adapters = config.num_adapters
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.temperature = config.temperature
        
        # Query, key, value projections
        if config.use_query_projection:
            self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.query_proj = nn.Identity()
        
        if config.use_key_value_projection:
            self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
            self.value_proj = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.key_proj = nn.Identity()
            self.value_proj = nn.Identity()
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        for module in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        adapter_outputs: List[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse adapter outputs using attention.
        
        Args:
            query: Query tensor [batch_size, seq_len, hidden_size]
            adapter_outputs: List of adapter outputs
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (fused_output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Stack adapter outputs
        # [batch_size, num_adapters, seq_len, hidden_size]
        stacked_outputs = torch.stack(adapter_outputs, dim=1)
        
        # Compute query
        Q = self.query_proj(query)  # [batch_size, seq_len, hidden_size]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Compute keys and values for each adapter
        K_list = []
        V_list = []
        
        for i in range(self.num_adapters):
            K_i = self.key_proj(stacked_outputs[:, i])
            V_i = self.value_proj(stacked_outputs[:, i])
            
            K_i = K_i.view(batch_size, seq_len, self.num_heads, self.head_dim)
            V_i = V_i.view(batch_size, seq_len, self.num_heads, self.head_dim)
            
            K_list.append(K_i.transpose(1, 2))
            V_list.append(V_i.transpose(1, 2))
        
        # Stack keys and values
        K = torch.stack(K_list, dim=2)  # [batch_size, num_heads, num_adapters, seq_len, head_dim]
        V = torch.stack(V_list, dim=2)
        
        # Compute attention scores
        # [batch_size, num_heads, seq_len, num_adapters, seq_len]
        scores = torch.einsum('bhqd,bhasd->bhqas', Q, K) / (math.sqrt(self.head_dim) * self.temperature)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            scores = scores.masked_fill(~mask.bool(), float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=3)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        # [batch_size, num_heads, seq_len, head_dim]
        attended = torch.einsum('bhqas,bhasd->bhqd', attention_weights, V)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, seq_len, self.hidden_size)
        output = self.output_proj(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + query)
        
        # Average attention weights across heads
        avg_weights = attention_weights.mean(dim=1).mean(dim=2)  # [batch_size, num_adapters, seq_len]
        
        return output, avg_weights


class GatingFusion(nn.Module):
    """
    Gating-based adapter fusion.
    
    Uses learnable gates to control contribution of each adapter.
    """
    
    def __init__(self, config: AdapterFusionConfig):
        """
        Initialize gating fusion.
        
        Args:
            config: Fusion configuration
        """
        super().__init__()
        
        self.num_adapters = config.num_adapters
        self.hidden_size = config.hidden_size
        self.gating_type = config.gating_type
        
        # Gating network
        if config.gating_type == "scalar":
            # Scalar gates for each adapter
            self.gate_network = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_size // 2, config.num_adapters)
            )
        elif config.gating_type == "vector":
            # Vector gates (per-dimension)
            self.gate_network = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.num_adapters * config.hidden_size)
            )
        elif config.gating_type == "matrix":
            # Matrix gates (full transformation)
            self.gate_network = nn.ModuleList([
                nn.Linear(config.hidden_size, config.hidden_size)
                for _ in range(config.num_adapters)
            ])
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize gating weights."""
        if self.gating_type in ["scalar", "vector"]:
            for module in self.gate_network.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        else:
            for gate in self.gate_network:
                nn.init.xavier_uniform_(gate.weight)
                nn.init.zeros_(gate.bias)
    
    def forward(
        self,
        query: torch.Tensor,
        adapter_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse adapter outputs using gating.
        
        Args:
            query: Query tensor
            adapter_outputs: List of adapter outputs
            
        Returns:
            Tuple of (fused_output, gate_values)
        """
        batch_size, seq_len, hidden_size = query.shape
        
        if self.gating_type == "scalar":
            # Compute scalar gates
            gates = self.gate_network(query.mean(dim=1))  # [batch_size, num_adapters]
            gates = F.softmax(gates, dim=-1)
            
            # Apply gates
            output = torch.zeros_like(query)
            for i, adapter_output in enumerate(adapter_outputs):
                gate = gates[:, i:i+1].unsqueeze(1)  # [batch_size, 1, 1]
                output = output + gate * adapter_output
                
        elif self.gating_type == "vector":
            # Compute vector gates
            gates = self.gate_network(query)  # [batch_size, seq_len, num_adapters * hidden_size]
            gates = gates.view(batch_size, seq_len, self.num_adapters, hidden_size)
            gates = F.softmax(gates, dim=2)
            
            # Apply gates
            stacked_outputs = torch.stack(adapter_outputs, dim=2)
            output = (gates * stacked_outputs).sum(dim=2)
            
        elif self.gating_type == "matrix":
            # Compute matrix transformations
            transformed = []
            for i, (adapter_output, gate) in enumerate(zip(adapter_outputs, self.gate_network)):
                transformed.append(torch.sigmoid(gate(query)) * adapter_output)
            
            output = sum(transformed) / len(transformed)
            gates = torch.ones(batch_size, self.num_adapters) / self.num_adapters
        
        # Residual and layer norm
        output = self.layer_norm(output + query)
        
        return output, gates


class MixtureOfExpertsFusion(nn.Module):
    """
    Mixture of Experts fusion for adapters.
    
    Routes inputs to different adapter combinations using learned routing.
    Based on Shazeer et al. (2017): "Outrageously Large Neural Networks"
    """
    
    def __init__(self, config: AdapterFusionConfig):
        """
        Initialize MoE fusion.
        
        Args:
            config: Fusion configuration
        """
        super().__init__()
        
        self.num_experts = config.num_experts
        self.num_adapters = config.num_adapters
        self.expert_capacity = config.expert_capacity
        self.use_load_balancing = config.use_load_balancing
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.num_experts)
        )
        
        # Expert networks (combinations of adapters)
        self.expert_weights = nn.Parameter(
            torch.randn(config.num_experts, config.num_adapters)
        )
        
        # Load balancing loss
        self.load_balance_loss = 0.0
        
        # Initialize
        nn.init.xavier_uniform_(self.expert_weights)
    
    def forward(
        self,
        query: torch.Tensor,
        adapter_outputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route through mixture of experts.
        
        Args:
            query: Query tensor
            adapter_outputs: List of adapter outputs
            
        Returns:
            Tuple of (fused_output, routing_info)
        """
        batch_size, seq_len, hidden_size = query.shape
        
        # Compute routing probabilities
        routing_logits = self.router(query.mean(dim=1))  # [batch_size, num_experts]
        routing_probs = F.softmax(routing_logits, dim=-1)
        
        # Select top-k experts
        k = min(2, self.num_experts)  # Use top-2 experts
        top_k_probs, top_k_indices = torch.topk(routing_probs, k, dim=-1)
        
        # Normalize top-k probabilities
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(query)
        
        for i in range(k):
            expert_idx = top_k_indices[:, i]
            expert_prob = top_k_probs[:, i:i+1].unsqueeze(1)
            
            # Get expert weights for adapters
            expert_adapter_weights = F.softmax(self.expert_weights[expert_idx], dim=-1)
            
            # Combine adapters for this expert
            expert_output = torch.zeros_like(query)
            for j, adapter_output in enumerate(adapter_outputs):
                weight = expert_adapter_weights[:, j:j+1].unsqueeze(1)
                expert_output = expert_output + weight * adapter_output
            
            # Add weighted expert output
            output = output + expert_prob * expert_output
        
        # Compute load balancing loss if training
        if self.training and self.use_load_balancing:
            # Encourage uniform expert usage
            expert_usage = routing_probs.mean(dim=0)
            self.load_balance_loss = (expert_usage.var() * self.num_experts).mean()
        
        routing_info = {
            "routing_probs": routing_probs,
            "top_k_indices": top_k_indices,
            "expert_weights": F.softmax(self.expert_weights, dim=-1),
            "load_balance_loss": self.load_balance_loss
        }
        
        return output, routing_info


class AdapterFusion(nn.Module):
    """
    Main adapter fusion module.
    
    Combines multiple task-specific adapters for multi-task learning
    and transfer learning across tasks.
    """
    
    def __init__(self, config: AdapterFusionConfig):
        """
        Initialize adapter fusion.
        
        Args:
            config: Adapter fusion configuration
        """
        super().__init__()
        
        self.config = config
        
        # Create adapters
        self.adapters = nn.ModuleList([
            AdapterModule(
                config.hidden_size,
                config.adapter_size,
                config.adapter_dropout
            )
            for _ in range(config.num_adapters)
        ])
        
        # Create fusion mechanism
        if config.fusion_type == "attention":
            self.fusion = AttentionFusion(config)
        elif config.fusion_type == "gating":
            self.fusion = GatingFusion(config)
        elif config.fusion_type == "mixture":
            self.fusion = MixtureOfExpertsFusion(config)
        else:
            raise ValueError(f"Unknown fusion type: {config.fusion_type}")
        
        # Adapter dropout for regularization
        if config.use_adapter_drop:
            self.adapter_dropout = nn.Dropout(config.adapter_drop_prob)
        
        logger.info(
            f"Initialized AdapterFusion with {config.num_adapters} adapters "
            f"using {config.fusion_type} fusion"
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        adapter_mask: Optional[torch.Tensor] = None,
        return_fusion_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Forward pass through adapter fusion.
        
        Args:
            hidden_states: Input hidden states
            adapter_mask: Optional mask for adapters
            return_fusion_weights: Return fusion weights
            
        Returns:
            Fused output or tuple with fusion weights
        """
        # Apply adapters
        adapter_outputs = []
        
        for i, adapter in enumerate(self.adapters):
            # Skip if masked
            if adapter_mask is not None and not adapter_mask[i]:
                continue
            
            # Apply adapter
            adapter_output = adapter(hidden_states)
            
            # Apply adapter dropout if training
            if self.training and self.config.use_adapter_drop:
                if torch.rand(1).item() < self.config.adapter_drop_prob:
                    adapter_output = hidden_states  # Skip this adapter
            
            adapter_outputs.append(adapter_output)
        
        # Fuse adapter outputs
        if self.config.fusion_type == "mixture":
            fused_output, fusion_info = self.fusion(hidden_states, adapter_outputs)
            fusion_weights = fusion_info
        else:
            fused_output, fusion_weights = self.fusion(hidden_states, adapter_outputs)
        
        if return_fusion_weights:
            return fused_output, fusion_weights
        
        return fused_output
    
    def freeze_adapters(self, adapter_indices: Optional[List[int]] = None):
        """
        Freeze specific adapters.
        
        Args:
            adapter_indices: Indices of adapters to freeze (None for all)
        """
        if adapter_indices is None:
            adapter_indices = range(len(self.adapters))
        
        for i in adapter_indices:
            for param in self.adapters[i].parameters():
                param.requires_grad = False
        
        logger.info(f"Froze adapters: {adapter_indices}")
    
    def unfreeze_adapters(self, adapter_indices: Optional[List[int]] = None):
        """
        Unfreeze specific adapters.
        
        Args:
            adapter_indices: Indices of adapters to unfreeze (None for all)
        """
        if adapter_indices is None:
            adapter_indices = range(len(self.adapters))
        
        for i in adapter_indices:
            for param in self.adapters[i].parameters():
                param.requires_grad = True
        
        logger.info(f"Unfroze adapters: {adapter_indices}")
    
    def add_adapter(self, name: Optional[str] = None) -> int:
        """
        Add a new adapter.
        
        Args:
            name: Optional name for adapter
            
        Returns:
            Index of new adapter
        """
        new_adapter = AdapterModule(
            self.config.hidden_size,
            self.config.adapter_size,
            self.config.adapter_dropout
        )
        
        self.adapters.append(new_adapter)
        self.config.num_adapters += 1
        
        # Update fusion mechanism if needed
        if hasattr(self.fusion, 'num_adapters'):
            self.fusion.num_adapters = self.config.num_adapters
        
        logger.info(f"Added adapter {len(self.adapters) - 1}")
        
        return len(self.adapters) - 1
    
    def get_adapter_params(self, adapter_idx: int) -> Dict[str, torch.Tensor]:
        """
        Get parameters of specific adapter.
        
        Args:
            adapter_idx: Adapter index
            
        Returns:
            Dictionary of adapter parameters
        """
        return {
            name: param for name, param in 
            self.adapters[adapter_idx].named_parameters()
        }
