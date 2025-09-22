"""
Attention-based Classification Head Implementation
===================================================

Implementation of attention-based classification heads for enhanced
feature aggregation and interpretability, based on:
- Bahdanau et al. (2015): "Neural Machine Translation by Jointly Learning to Align and Translate"
- Vaswani et al. (2017): "Attention is All You Need"
- Yang et al. (2016): "Hierarchical Attention Networks for Document Classification"
- Lin et al. (2017): "A Structured Self-attentive Sentence Embedding"

Mathematical Foundation:
Self-attention: Attention(Q,K,V) = softmax(QK^T/√d_k)V
Multi-head attention: MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class AttentionHeadConfig:
    """Configuration for attention-based classification head."""
    hidden_size: int = 768
    num_labels: int = 4
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.2
    
    # Attention type
    attention_type: str = "self"  # "self", "multi_head", "hierarchical", "structured"
    
    # Multi-head attention
    head_dim: Optional[int] = None
    use_relative_position: bool = False
    max_position_embeddings: int = 512
    
    # Structured self-attention
    num_attention_hops: int = 4  # For structured self-attention
    attention_penalty: float = 1.0  # Penalization for attention weights
    
    # Hierarchical attention
    use_word_attention: bool = True
    use_sentence_attention: bool = True
    sentence_encoder_layers: int = 2
    
    # Classification
    num_hidden_layers: int = 2
    intermediate_size: int = 3072
    use_pooler: bool = True
    pooling_strategy: str = "attention"  # "attention", "mean", "max", "cls"
    
    # Regularization
    use_layer_norm: bool = True
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02


class SelfAttentionPooling(nn.Module):
    """
    Self-attention pooling for sequence aggregation.
    
    Learns to weight different positions in the sequence based on
    their relevance for classification.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        """
        Initialize self-attention pooling.
        
        Args:
            hidden_size: Hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.attention_vector = nn.Parameter(torch.randn(hidden_size))
        self.attention_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.normal_(self.attention_vector, std=0.02)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention pooling.
        
        Args:
            hidden_states: Sequence hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (pooled_output, attention_weights)
        """
        # Project hidden states
        attention_scores = torch.tanh(self.attention_proj(hidden_states))
        
        # Compute attention weights
        attention_scores = torch.matmul(attention_scores, self.attention_vector)
        
        # Apply mask
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~attention_mask.bool(),
                float('-inf')
            )
        
        # Normalize
        attention_weights = F.softmax(attention_scores, dim=1)
        attention_weights = self.dropout(attention_weights)
        
        # Weighted sum
        pooled = torch.sum(
            hidden_states * attention_weights.unsqueeze(-1),
            dim=1
        )
        
        return pooled, attention_weights


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention for classification.
    
    Implements scaled dot-product attention with multiple heads
    for capturing different types of relationships.
    """
    
    def __init__(self, config: AttentionHeadConfig):
        """
        Initialize multi-head self-attention.
        
        Args:
            config: Attention head configuration
        """
        super().__init__()
        
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.head_dim
        
        # Query, Key, Value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Relative position embeddings
        if config.use_relative_position:
            self.relative_positions = nn.Embedding(
                2 * config.max_position_embeddings - 1,
                self.head_dim
            )
    
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Transpose for multi-head attention."""
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through multi-head attention.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            output_attentions: Return attention weights
            
        Returns:
            Tuple of (attended_values, attention_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply mask
        if attention_mask is not None:
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = (1.0 - extended_mask) * -10000.0
            attention_scores = attention_scores + extended_mask
        
        # Attention weights
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Attended values
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_shape)
        
        # Output projection
        attention_output = self.out_proj(context_layer)
        
        if output_attentions:
            return attention_output, attention_probs
        
        return attention_output, None


class StructuredSelfAttention(nn.Module):
    """
    Structured self-attention for learning multiple attention distributions.
    
    Based on Lin et al. (2017): "A Structured Self-attentive Sentence Embedding"
    """
    
    def __init__(self, config: AttentionHeadConfig):
        """
        Initialize structured self-attention.
        
        Args:
            config: Attention configuration
        """
        super().__init__()
        
        self.num_hops = config.num_attention_hops
        self.hidden_size = config.hidden_size
        self.penalty = config.attention_penalty
        
        # Attention parameters
        self.W_s1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_s2 = nn.Linear(config.hidden_size, config.num_attention_hops)
        
        # Output projection
        self.out_proj = nn.Linear(
            config.hidden_size * config.num_attention_hops,
            config.hidden_size
        )
        
        self.dropout = nn.Dropout(config.attention_dropout)
        
        # Initialize
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.xavier_uniform_(self.W_s1.weight)
        nn.init.xavier_uniform_(self.W_s2.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through structured self-attention.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
            
        Returns:
            Tuple of (output, attention_matrix, penalty_loss)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Compute attention matrix
        # [batch_size, num_hops, seq_len]
        attention = torch.tanh(self.W_s1(hidden_states))
        attention = self.W_s2(attention).transpose(1, 2)
        
        # Apply mask
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1).expand_as(attention)
            attention = attention.masked_fill(~mask.bool(), float('-inf'))
        
        # Normalize
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        # [batch_size, num_hops, hidden_size]
        attended = torch.bmm(attention, hidden_states)
        
        # Flatten and project
        attended_flat = attended.view(batch_size, -1)
        output = self.out_proj(attended_flat)
        output = self.dropout(output)
        
        # Compute diversity penalty
        if self.training:
            # Penalize redundancy in attention
            identity = torch.eye(self.num_hops).to(attention.device)
            identity = identity.unsqueeze(0).expand(batch_size, -1, -1)
            
            attention_t = attention.transpose(1, 2)
            diversity = torch.bmm(attention, attention_t)
            
            penalty = torch.norm(diversity - identity, p='fro', dim=(1, 2))
            penalty = penalty.mean() * self.penalty
        else:
            penalty = torch.tensor(0.0).to(attention.device)
        
        return output, attention, penalty


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention for document classification.
    
    Based on Yang et al. (2016): "Hierarchical Attention Networks"
    """
    
    def __init__(self, config: AttentionHeadConfig):
        """
        Initialize hierarchical attention.
        
        Args:
            config: Attention configuration
        """
        super().__init__()
        
        self.config = config
        
        # Word-level attention
        if config.use_word_attention:
            self.word_attention = SelfAttentionPooling(
                config.hidden_size,
                config.attention_dropout
            )
        
        # Sentence encoder
        if config.use_sentence_attention:
            self.sentence_encoder = nn.LSTM(
                config.hidden_size,
                config.hidden_size // 2,
                config.sentence_encoder_layers,
                bidirectional=True,
                batch_first=True,
                dropout=config.hidden_dropout
            )
            
            self.sentence_attention = SelfAttentionPooling(
                config.hidden_size,
                config.attention_dropout
            )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        sentence_boundaries: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through hierarchical attention.
        
        Args:
            hidden_states: Input hidden states
            sentence_boundaries: Sentence boundary indicators
            attention_mask: Attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = hidden_states.size(0)
        
        if sentence_boundaries is not None and self.config.use_sentence_attention:
            # Word-level attention within sentences
            sentence_reprs = []
            word_attentions = []
            
            for i in range(batch_size):
                # Find sentence boundaries
                boundaries = torch.where(sentence_boundaries[i])[0]
                if len(boundaries) == 0:
                    boundaries = torch.tensor([hidden_states.size(1) - 1])
                
                prev_boundary = 0
                sent_reprs = []
                
                for boundary in boundaries:
                    # Extract sentence
                    sentence = hidden_states[i, prev_boundary:boundary+1].unsqueeze(0)
                    
                    if self.config.use_word_attention:
                        # Apply word attention
                        sent_mask = attention_mask[i, prev_boundary:boundary+1].unsqueeze(0) if attention_mask is not None else None
                        sent_repr, word_attn = self.word_attention(sentence, sent_mask)
                        sent_reprs.append(sent_repr)
                        word_attentions.append(word_attn)
                    else:
                        # Average pooling
                        sent_repr = sentence.mean(dim=1)
                        sent_reprs.append(sent_repr)
                    
                    prev_boundary = boundary + 1
                
                if sent_reprs:
                    sentence_reprs.append(torch.stack(sent_reprs, dim=1))
            
            # Sentence-level processing
            if sentence_reprs:
                sentence_reprs = torch.cat(sentence_reprs, dim=0)
                
                # Encode sentences
                sentence_encoded, _ = self.sentence_encoder(sentence_reprs)
                
                # Sentence attention
                doc_repr, sent_attn = self.sentence_attention(sentence_encoded)
                
                attention_info = {
                    "word_attention": word_attentions if word_attentions else None,
                    "sentence_attention": sent_attn
                }
            else:
                # Fallback to simple attention
                doc_repr, attn = self.word_attention(hidden_states, attention_mask)
                attention_info = {"attention": attn}
        
        else:
            # Simple attention pooling
            if self.config.use_word_attention:
                doc_repr, attn = self.word_attention(hidden_states, attention_mask)
                attention_info = {"attention": attn}
            else:
                # Mean pooling
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    doc_repr = sum_embeddings / sum_mask
                else:
                    doc_repr = hidden_states.mean(dim=1)
                attention_info = {}
        
        return doc_repr, attention_info


class AttentionClassificationHead(nn.Module):
    """
    Main attention-based classification head.
    
    Combines various attention mechanisms for enhanced classification.
    """
    
    def __init__(self, config: AttentionHeadConfig):
        """
        Initialize attention classification head.
        
        Args:
            config: Configuration
        """
        super().__init__()
        
        self.config = config
        
        # Choose attention mechanism
        if config.attention_type == "self":
            self.attention = SelfAttentionPooling(
                config.hidden_size,
                config.attention_dropout
            )
        elif config.attention_type == "multi_head":
            self.attention = MultiHeadSelfAttention(config)
        elif config.attention_type == "structured":
            self.attention = StructuredSelfAttention(config)
        elif config.attention_type == "hierarchical":
            self.attention = HierarchicalAttention(config)
        else:
            raise ValueError(f"Unknown attention type: {config.attention_type}")
        
        # Classification layers
        self.classifier = self._build_classifier()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_classifier(self) -> nn.Module:
        """Build classification layers."""
        layers = []
        
        input_size = self.config.hidden_size
        
        for i in range(self.config.num_hidden_layers):
            # Linear layer
            layers.append(nn.Linear(input_size, self.config.intermediate_size))
            
            # Activation
            layers.append(nn.GELU())
            
            # Layer norm
            if self.config.use_layer_norm:
                layers.append(nn.LayerNorm(self.config.intermediate_size, eps=self.config.layer_norm_eps))
            
            # Dropout
            layers.append(nn.Dropout(self.config.hidden_dropout))
            
            input_size = self.config.intermediate_size
        
        # Output layer
        layers.append(nn.Linear(input_size, self.config.num_labels))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        sentence_boundaries: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Any]]:
        """
        Forward pass through attention classification head.
        
        Args:
            hidden_states: Input hidden states
            attention_mask: Attention mask
            labels: Target labels
            sentence_boundaries: Sentence boundaries for hierarchical attention
            return_attention: Return attention weights
            
        Returns:
            Tuple of (logits, loss, attention_info)
        """
        # Apply attention
        if self.config.attention_type == "self":
            pooled, attention_weights = self.attention(hidden_states, attention_mask)
            attention_info = attention_weights if return_attention else None
            
        elif self.config.attention_type == "multi_head":
            attended, attention_weights = self.attention(
                hidden_states,
                attention_mask,
                output_attentions=return_attention
            )
            pooled = attended.mean(dim=1)  # Average over sequence
            attention_info = attention_weights
            
        elif self.config.attention_type == "structured":
            pooled, attention_matrix, penalty = self.attention(hidden_states, attention_mask)
            attention_info = (attention_matrix, penalty) if return_attention else None
            
        elif self.config.attention_type == "hierarchical":
            pooled, attention_dict = self.attention(
                hidden_states,
                sentence_boundaries,
                attention_mask
            )
            attention_info = attention_dict if return_attention else None
        
        # Classification
        logits = self.classifier(pooled)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            
            # Add structured attention penalty if applicable
            if self.config.attention_type == "structured" and self.training:
                _, _, penalty = self.attention(hidden_states, attention_mask)
                loss = loss + penalty
        
        return logits, loss, attention_info
