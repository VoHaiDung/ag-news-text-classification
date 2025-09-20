"""
Pooling Strategies for Sequence Representation
===============================================

This module implements various pooling strategies for aggregating token-level
representations into sequence-level representations, following research from:

- Conneau & Kiela (2018): "SentEval: An Evaluation Toolkit for Universal Sentence Representations"
- Reimers & Gurevych (2019): "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks"
- Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- Li et al. (2020): "On the Sentence Embeddings from Pre-trained Language Models"

Mathematical Foundation:
Given a sequence of token representations H = [h₁, h₂, ..., hₙ] ∈ ℝⁿˣᵈ,
pooling functions φ: ℝⁿˣᵈ → ℝᵈ aggregate them into a fixed-size representation.

Theoretical Principles:
1. Information Preservation: Minimize information loss during aggregation
2. Invariance Properties: Maintain desired invariances (e.g., permutation)
3. Computational Efficiency: O(n) complexity for sequence length n

Author: Võ Hải Dũng
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import math
from enum import Enum

from src.core.exceptions import ModelError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PoolingStrategy(Enum):
    """Enumeration of available pooling strategies."""
    CLS = "cls"
    MEAN = "mean"
    MAX = "max"
    MIN = "min"
    FIRST_LAST_AVG = "first_last_avg"
    WEIGHTED_MEAN = "weighted_mean"
    ATTENTION = "attention"
    HIERARCHICAL = "hierarchical"
    MULTI_LAYER = "multi_layer"
    SENTENCE = "sentence"


@dataclass
class PoolingConfig:
    """
    Configuration for pooling strategies.
    
    Attributes:
        strategy: Pooling strategy to use
        hidden_size: Dimension of hidden states
        num_layers: Number of layers for multi-layer pooling
        attention_heads: Number of attention heads for attention pooling
        dropout_rate: Dropout rate for regularization
        temperature: Temperature for attention weights
        use_layer_weights: Whether to use learnable layer weights
        normalize_output: Whether to normalize pooled output
    """
    strategy: str = "cls"
    hidden_size: int = 768
    num_layers: int = 12
    attention_heads: int = 1
    dropout_rate: float = 0.1
    temperature: float = 1.0
    use_layer_weights: bool = False
    normalize_output: bool = False


class BasePooler(nn.Module):
    """
    Abstract base class for pooling strategies.
    
    Defines the interface and common functionality for all pooling methods
    following the Strategy Pattern for interchangeable pooling algorithms.
    """
    
    def __init__(self, config: PoolingConfig):
        """
        Initialize base pooler.
        
        Args:
            config: Pooling configuration
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Pool hidden states into sequence representation.
        
        Args:
            hidden_states: Token representations [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            **kwargs: Additional arguments
            
        Returns:
            Pooled representation [batch_size, hidden_dim]
        """
        raise NotImplementedError("Subclasses must implement forward method")
    
    def _apply_mask(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        mask_value: float = -1e9
    ) -> torch.Tensor:
        """
        Apply attention mask to hidden states.
        
        Args:
            hidden_states: Hidden states to mask
            attention_mask: Binary attention mask
            mask_value: Value to use for masked positions
            
        Returns:
            Masked hidden states
        """
        if attention_mask is not None:
            # Expand mask to match hidden states dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            # Apply mask
            hidden_states = hidden_states.clone()
            hidden_states[~mask_expanded.bool()] = mask_value
        return hidden_states


class CLSPooler(BasePooler):
    """
    CLS token pooling strategy.
    
    Uses the representation of the [CLS] token (typically the first token)
    as the sequence representation, following BERT's pre-training objective.
    
    Mathematical Description:
    φ_cls(H) = h₁ where H = [h₁, h₂, ..., hₙ]
    
    References:
        Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"
    """
    
    def __init__(self, config: PoolingConfig):
        """Initialize CLS pooler with optional projection."""
        super().__init__(config)
        
        # Optional: Add a pooler layer like BERT
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Extract [CLS] token representation.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Not used for CLS pooling
            
        Returns:
            CLS representation [batch_size, hidden_dim]
        """
        # Extract first token (CLS token)
        cls_hidden_state = hidden_states[:, 0]
        
        # Apply pooler transformation
        pooled_output = self.dense(cls_hidden_state)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        
        return pooled_output


class MeanPooler(BasePooler):
    """
    Mean pooling strategy.
    
    Computes the average of all token representations, accounting for padding
    tokens through the attention mask.
    
    Mathematical Description:
    φ_mean(H, M) = (Σᵢ hᵢ * mᵢ) / Σᵢ mᵢ
    where M is the attention mask
    
    References:
        Reimers & Gurevych (2019): "Sentence-BERT"
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute mean pooling over sequence.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Mean pooled representation [batch_size, hidden_dim]
        """
        if attention_mask is not None:
            # Expand mask to match hidden states dimensions
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            
            # Sum only non-masked tokens
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            
            # Count non-masked tokens
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            
            # Compute mean
            pooled_output = sum_embeddings / sum_mask
        else:
            # Simple mean if no mask
            pooled_output = torch.mean(hidden_states, dim=1)
        
        return self.dropout(pooled_output)


class MaxPooler(BasePooler):
    """
    Max pooling strategy.
    
    Takes the maximum value across the sequence dimension for each feature,
    capturing the most salient features.
    
    Mathematical Description:
    φ_max(H) = max(h₁, h₂, ..., hₙ) element-wise
    
    Properties:
    - Invariant to sequence order
    - Captures most prominent features
    - May lose temporal information
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute max pooling over sequence.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Max pooled representation [batch_size, hidden_dim]
        """
        # Apply mask with very negative value for masked positions
        hidden_states = self._apply_mask(hidden_states, attention_mask, mask_value=-1e9)
        
        # Max pooling
        pooled_output, _ = torch.max(hidden_states, dim=1)
        
        return self.dropout(pooled_output)


class MinPooler(BasePooler):
    """
    Min pooling strategy.
    
    Takes the minimum value across the sequence dimension for each feature.
    Less common but useful for certain applications.
    
    Mathematical Description:
    φ_min(H) = min(h₁, h₂, ..., hₙ) element-wise
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute min pooling over sequence.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Min pooled representation [batch_size, hidden_dim]
        """
        # Apply mask with very positive value for masked positions
        hidden_states = self._apply_mask(hidden_states, attention_mask, mask_value=1e9)
        
        # Min pooling
        pooled_output, _ = torch.min(hidden_states, dim=1)
        
        return self.dropout(pooled_output)


class FirstLastAveragePooler(BasePooler):
    """
    First-Last Average pooling strategy.
    
    Averages the first and last token representations, combining
    global context (CLS) with final context information.
    
    Mathematical Description:
    φ_fl(H) = (h₁ + hₙ) / 2
    
    References:
        Li et al. (2020): "On the Sentence Embeddings from Pre-trained Language Models"
    """
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Average first and last token representations.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Averaged representation [batch_size, hidden_dim]
        """
        # Get first token
        first_token = hidden_states[:, 0]
        
        # Get last valid token for each sequence
        if attention_mask is not None:
            # Find last valid position
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = hidden_states.size(0)
            last_token = hidden_states[torch.arange(batch_size), seq_lengths]
        else:
            last_token = hidden_states[:, -1]
        
        # Average
        pooled_output = (first_token + last_token) / 2
        
        return self.dropout(pooled_output)


class AttentionPooler(BasePooler):
    """
    Attention-based pooling strategy.
    
    Learns attention weights to create a weighted average of token representations,
    allowing the model to focus on informative tokens.
    
    Mathematical Description:
    α = softmax(W_a · tanh(W_h · H^T))
    φ_att(H) = Σᵢ αᵢ · hᵢ
    
    References:
        Yang et al. (2016): "Hierarchical Attention Networks for Document Classification"
        Lin et al. (2017): "A Structured Self-attentive Sentence Embedding"
    """
    
    def __init__(self, config: PoolingConfig):
        """Initialize attention pooler with learnable parameters."""
        super().__init__(config)
        
        # Attention parameters
        self.attention_heads = config.attention_heads
        self.head_dim = config.hidden_size // config.attention_heads
        
        # Attention layers
        self.W_h = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_a = nn.Linear(config.hidden_size, config.attention_heads)
        
        # Temperature for attention softmax
        self.temperature = config.temperature
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute attention-weighted pooling.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Attention-pooled representation [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.size()
        
        # Compute attention scores
        # [batch_size, seq_len, hidden_dim]
        hidden_proj = torch.tanh(self.W_h(hidden_states))
        
        # [batch_size, seq_len, attention_heads]
        attention_scores = self.W_a(hidden_proj) / self.temperature
        
        # Apply mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(
                batch_size, seq_len, self.attention_heads
            )
            attention_scores = attention_scores.masked_fill(~mask_expanded.bool(), -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention to hidden states
        if self.attention_heads > 1:
            # Multi-head attention pooling
            # Reshape for multi-head
            hidden_states = hidden_states.view(
                batch_size, seq_len, self.attention_heads, self.head_dim
            )
            attention_weights = attention_weights.unsqueeze(-1)
            
            # Weighted sum
            pooled_output = torch.sum(hidden_states * attention_weights, dim=1)
            pooled_output = pooled_output.view(batch_size, hidden_dim)
        else:
            # Single-head attention
            attention_weights = attention_weights.squeeze(-1).unsqueeze(-1)
            pooled_output = torch.sum(hidden_states * attention_weights, dim=1)
        
        return self.dropout(pooled_output)


class WeightedMeanPooler(BasePooler):
    """
    Weighted mean pooling with learnable position weights.
    
    Learns importance weights for different positions in the sequence,
    useful when certain positions are consistently more informative.
    
    Mathematical Description:
    φ_wm(H) = Σᵢ wᵢ · hᵢ / Σᵢ wᵢ
    where w are learnable position weights
    """
    
    def __init__(self, config: PoolingConfig, max_seq_length: int = 512):
        """Initialize weighted mean pooler."""
        super().__init__(config)
        
        # Learnable position weights
        self.position_weights = nn.Parameter(torch.ones(max_seq_length))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute weighted mean pooling.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Weighted mean representation [batch_size, hidden_dim]
        """
        batch_size, seq_len, hidden_dim = hidden_states.size()
        
        # Get position weights for current sequence length
        weights = self.position_weights[:seq_len].unsqueeze(0).unsqueeze(-1)
        weights = weights.expand(batch_size, seq_len, hidden_dim)
        
        # Apply attention mask
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            weights = weights * mask_expanded
        
        # Weighted sum
        weighted_sum = torch.sum(hidden_states * weights, dim=1)
        weight_sum = torch.clamp(weights.sum(dim=1), min=1e-9)
        
        pooled_output = weighted_sum / weight_sum
        
        return self.dropout(pooled_output)


class MultiLayerPooler(BasePooler):
    """
    Multi-layer pooling strategy.
    
    Combines representations from multiple transformer layers to capture
    different levels of abstraction.
    
    Mathematical Description:
    φ_ml(H¹, H², ..., Hᴸ) = Σₗ γₗ · pool(Hˡ)
    where γₗ are layer weights and L is the number of layers
    
    References:
        Peters et al. (2018): "Deep contextualized word representations"
        Liu et al. (2019): "Linguistic Knowledge and Transferability of Contextual Representations"
    """
    
    def __init__(self, config: PoolingConfig):
        """Initialize multi-layer pooler."""
        super().__init__(config)
        
        self.num_layers = config.num_layers
        self.use_layer_weights = config.use_layer_weights
        
        if self.use_layer_weights:
            # Learnable layer weights
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers))
        else:
            # Fixed weights (average)
            self.register_buffer("layer_weights", torch.ones(self.num_layers) / self.num_layers)
        
        # Base pooler for each layer
        self.base_pooler = MeanPooler(config)
        
    def forward(
        self,
        hidden_states: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Pool representations from multiple layers.
        
        Args:
            hidden_states: Tuple of hidden states from each layer or 
                          tensor [batch_size, num_layers, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Multi-layer pooled representation [batch_size, hidden_dim]
        """
        # Handle different input formats
        if isinstance(hidden_states, tuple):
            # Tuple of layer outputs
            layer_outputs = hidden_states
        else:
            # Tensor with layer dimension
            if hidden_states.dim() == 4:
                layer_outputs = [hidden_states[:, i] for i in range(hidden_states.size(1))]
            else:
                raise ValueError("Invalid hidden_states format for multi-layer pooling")
        
        # Pool each layer
        pooled_layers = []
        for layer_output in layer_outputs[-self.num_layers:]:
            pooled = self.base_pooler(layer_output, attention_mask)
            pooled_layers.append(pooled)
        
        # Stack pooled representations
        pooled_stack = torch.stack(pooled_layers, dim=1)
        
        # Apply layer weights
        if self.use_layer_weights:
            weights = F.softmax(self.layer_weights, dim=0)
        else:
            weights = self.layer_weights
        
        weights = weights.view(1, -1, 1)
        pooled_output = torch.sum(pooled_stack * weights, dim=1)
        
        return self.dropout(pooled_output)


class HierarchicalPooler(BasePooler):
    """
    Hierarchical pooling strategy.
    
    Performs pooling at multiple granularities (e.g., words, phrases, sentences)
    and combines them hierarchically.
    
    Mathematical Description:
    φ_hier(H) = concat(pool_word(H), pool_phrase(H), pool_sent(H))
    
    References:
        Yang et al. (2016): "Hierarchical Attention Networks"
    """
    
    def __init__(self, config: PoolingConfig):
        """Initialize hierarchical pooler."""
        super().__init__(config)
        
        # Different pooling strategies for different levels
        self.word_pooler = MeanPooler(config)
        self.phrase_pooler = MaxPooler(config)
        self.sentence_pooler = AttentionPooler(config)
        
        # Projection to combine different levels
        self.projection = nn.Linear(config.hidden_size * 3, config.hidden_size)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        segment_boundaries: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute hierarchical pooling.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]
            segment_boundaries: Optional segment boundaries for phrase-level pooling
            
        Returns:
            Hierarchically pooled representation [batch_size, hidden_dim]
        """
        # Word-level pooling (mean)
        word_pooled = self.word_pooler(hidden_states, attention_mask)
        
        # Phrase-level pooling (max)
        phrase_pooled = self.phrase_pooler(hidden_states, attention_mask)
        
        # Sentence-level pooling (attention)
        sentence_pooled = self.sentence_pooler(hidden_states, attention_mask)
        
        # Concatenate all levels
        combined = torch.cat([word_pooled, phrase_pooled, sentence_pooled], dim=-1)
        
        # Project back to hidden size
        pooled_output = self.projection(combined)
        pooled_output = torch.tanh(pooled_output)
        
        return self.dropout(pooled_output)


class PoolingFactory:
    """
    Factory class for creating pooling strategies.
    
    Implements the Factory Pattern for centralized pooler instantiation
    with configuration management.
    """
    
    _poolers = {
        PoolingStrategy.CLS.value: CLSPooler,
        PoolingStrategy.MEAN.value: MeanPooler,
        PoolingStrategy.MAX.value: MaxPooler,
        PoolingStrategy.MIN.value: MinPooler,
        PoolingStrategy.FIRST_LAST_AVG.value: FirstLastAveragePooler,
        PoolingStrategy.WEIGHTED_MEAN.value: WeightedMeanPooler,
        PoolingStrategy.ATTENTION.value: AttentionPooler,
        PoolingStrategy.HIERARCHICAL.value: HierarchicalPooler,
        PoolingStrategy.MULTI_LAYER.value: MultiLayerPooler,
    }
    
    @classmethod
    def create_pooler(
        cls,
        strategy: Union[str, PoolingStrategy],
        config: Optional[PoolingConfig] = None
    ) -> BasePooler:
        """
        Create a pooler instance based on strategy.
        
        Args:
            strategy: Pooling strategy name or enum
            config: Pooling configuration
            
        Returns:
            Pooler instance
            
        Raises:
            ValueError: If strategy is not supported
        """
        if config is None:
            config = PoolingConfig()
        
        # Convert enum to string if necessary
        if isinstance(strategy, PoolingStrategy):
            strategy = strategy.value
        
        # Update config with strategy
        config.strategy = strategy
        
        if strategy not in cls._poolers:
            available = ", ".join(cls._poolers.keys())
            raise ValueError(
                f"Unknown pooling strategy: {strategy}. "
                f"Available strategies: {available}"
            )
        
        pooler_class = cls._poolers[strategy]
        
        # Special handling for weighted mean pooler
        if strategy == PoolingStrategy.WEIGHTED_MEAN.value:
            return pooler_class(config, max_seq_length=512)
        
        return pooler_class(config)
    
    @classmethod
    def register_pooler(cls, name: str, pooler_class: type):
        """
        Register a custom pooler class.
        
        Args:
            name: Pooler name
            pooler_class: Pooler class
        """
        cls._poolers[name] = pooler_class
        logger.info(f"Registered custom pooler: {name}")


def create_pooler(
    strategy: Union[str, PoolingStrategy],
    **kwargs
) -> BasePooler:
    """
    Convenience function to create a pooler.
    
    Args:
        strategy: Pooling strategy
        **kwargs: Configuration parameters
        
    Returns:
        Pooler instance
        
    Example:
        >>> pooler = create_pooler("attention", hidden_size=768, attention_heads=8)
    """
    config = PoolingConfig(**kwargs)
    return PoolingFactory.create_pooler(strategy, config)


# Export public API
__all__ = [
    # Enums
    "PoolingStrategy",
    
    # Configs
    "PoolingConfig",
    
    # Base classes
    "BasePooler",
    
    # Pooler implementations
    "CLSPooler",
    "MeanPooler",
    "MaxPooler",
    "MinPooler",
    "FirstLastAveragePooler",
    "WeightedMeanPooler",
    "AttentionPooler",
    "MultiLayerPooler",
    "HierarchicalPooler",
    
    # Factory
    "PoolingFactory",
    
    # Functions
    "create_pooler",
]
