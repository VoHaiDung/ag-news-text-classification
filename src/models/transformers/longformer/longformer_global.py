"""
Longformer with Global Attention for AG News Classification
============================================================

Implementation of Longformer with global attention patterns for efficient
processing of long documents, based on:
- Beltagy et al. (2020): "Longformer: The Long-Document Transformer"

Key Features:
1. Sliding window attention (local)
2. Global attention on specific tokens
3. Linear complexity O(n) instead of O(n²)
4. Efficient processing of documents up to 4096 tokens

Mathematical Foundation:
Attention pattern combines local and global attention:
Attention(Q,K,V) = LocalAttention(Q,K,V) + GlobalAttention(Q_g,K,V)

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    LongformerModel,
    LongformerConfig,
    LongformerTokenizer
)

from src.models.base.base_model import TransformerBaseModel, ModelOutputs
from src.models.base.pooling_strategies import PoolingFactory
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LongformerGlobalConfig:
    """Configuration for Longformer with global attention."""
    model_name: str = "allenai/longformer-large-4096"
    num_labels: int = 4
    max_length: int = 4096
    attention_window: int = 512  # Local attention window
    global_attention_positions: Optional[List[int]] = None
    use_cls_global: bool = True  # Use global attention on CLS
    use_sentence_global: bool = False  # Global attention on sentence boundaries
    num_global_tokens: int = 16  # Number of tokens with global attention
    dropout_prob: float = 0.1
    classifier_dropout: float = 0.2
    pooling_strategy: str = "cls"
    gradient_checkpointing: bool = False
    use_adaptive_global: bool = False  # Adaptive global attention selection


class GlobalAttentionSelector(nn.Module):
    """
    Module for selecting positions for global attention.
    
    Implements adaptive selection of important tokens for global attention
    based on content relevance.
    """
    
    def __init__(self, hidden_size: int, num_global: int = 16):
        """
        Initialize global attention selector.
        
        Args:
            hidden_size: Hidden dimension size
            num_global: Number of global attention positions
        """
        super().__init__()
        
        self.num_global = num_global
        
        # Importance scoring network
        self.importance_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Select global attention positions.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Global attention mask [batch_size, seq_len]
        """
        batch_size, seq_len = attention_mask.shape
        
        # Compute importance scores
        importance_scores = self.importance_scorer(hidden_states).squeeze(-1)
        
        # Mask padding positions
        importance_scores = importance_scores.masked_fill(
            ~attention_mask.bool(),
            float('-inf')
        )
        
        # Select top-k important positions
        _, top_indices = torch.topk(
            importance_scores,
            min(self.num_global, seq_len),
            dim=1
        )
        
        # Create global attention mask
        global_attention_mask = torch.zeros_like(attention_mask)
        global_attention_mask.scatter_(1, top_indices, 1)
        
        # Always include CLS token
        global_attention_mask[:, 0] = 1
        
        return global_attention_mask


@MODELS.register("longformer", aliases=["longformer-global"])
class LongformerGlobal(TransformerBaseModel):
    """
    Longformer model with configurable global attention patterns.
    
    Efficiently processes long documents by combining:
    1. Local sliding window attention for most tokens
    2. Global attention for selected important tokens
    3. Linear complexity scaling with sequence length
    
    Particularly effective for:
    - Long document classification
    - Documents with important global context
    - Hierarchical document understanding
    """
    
    __auto_register__ = True
    __model_name__ = "longformer"
    
    def __init__(self, config: Optional[LongformerGlobalConfig] = None):
        """
        Initialize Longformer with global attention.
        
        Args:
            config: Model configuration
        """
        self.config = config or LongformerGlobalConfig()
        
        # Initialize base
        super().__init__(pretrained_model_name=self.config.model_name)
        
        # Initialize Longformer
        self._init_longformer_model()
        self._init_global_attention()
        self._init_classifier()
        
        logger.info(
            f"Initialized Longformer with window size {self.config.attention_window} "
            f"and {self.config.num_global_tokens} global attention positions"
        )
    
    def _init_longformer_model(self):
        """Initialize Longformer model."""
        # Load configuration
        longformer_config = LongformerConfig.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            attention_window=self.config.attention_window,
            hidden_dropout_prob=self.config.dropout_prob,
            attention_probs_dropout_prob=self.config.dropout_prob
        )
        
        # Load model
        self.longformer = LongformerModel.from_pretrained(
            self.config.model_name,
            config=longformer_config
        )
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.longformer.gradient_checkpointing_enable()
        
        self.hidden_size = self.longformer.config.hidden_size
    
    def _init_global_attention(self):
        """Initialize global attention components."""
        if self.config.use_adaptive_global:
            self.global_selector = GlobalAttentionSelector(
                self.hidden_size,
                self.config.num_global_tokens
            )
        else:
            self.global_selector = None
    
    def _init_classifier(self):
        """Initialize classification head."""
        # Pooling strategy
        self.pooler = PoolingFactory.create_pooler(
            strategy=self.config.pooling_strategy,
            hidden_size=self.hidden_size
        )
        
        # Classification head
        self.pre_classifier = nn.Linear(self.hidden_size, self.hidden_size)
        self.classifier_dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.config.num_labels)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.pre_classifier.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
    
    def _prepare_global_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepare global attention mask.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Standard attention mask
            
        Returns:
            Global attention mask
        """
        batch_size, seq_len = input_ids.shape
        
        # Initialize global attention mask
        global_attention_mask = torch.zeros_like(attention_mask)
        
        # Set global attention based on configuration
        if self.config.use_cls_global:
            # CLS token gets global attention
            global_attention_mask[:, 0] = 1
        
        if self.config.global_attention_positions:
            # Specific positions get global attention
            for pos in self.config.global_attention_positions:
                if pos < seq_len:
                    global_attention_mask[:, pos] = 1
        
        if self.config.use_sentence_global:
            # Find sentence boundaries (periods, question marks, etc.)
            sentence_ends = (
                (input_ids == 1012) |  # period token
                (input_ids == 1029) |  # question mark
                (input_ids == 999)     # exclamation mark
            ).long()
            global_attention_mask = global_attention_mask | sentence_ends
        
        if self.config.use_adaptive_global and self.global_selector:
            # Get initial hidden states for selection
            with torch.no_grad():
                outputs = self.longformer(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    global_attention_mask=global_attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.last_hidden_state
            
            # Select additional global positions
            adaptive_global = self.global_selector(hidden_states, attention_mask)
            global_attention_mask = global_attention_mask | adaptive_global
        
        # Ensure we don't exceed maximum global positions
        if self.config.num_global_tokens > 0:
            # Limit number of global attention positions
            num_global = global_attention_mask.sum(dim=1)
            for i in range(batch_size):
                if num_global[i] > self.config.num_global_tokens:
                    # Keep only top positions
                    indices = global_attention_mask[i].nonzero().squeeze()
                    if len(indices) > self.config.num_global_tokens:
                        keep_indices = indices[:self.config.num_global_tokens]
                        global_attention_mask[i] = 0
                        global_attention_mask[i, keep_indices] = 1
        
        return global_attention_mask
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        global_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> ModelOutputs:
        """
        Forward pass through Longformer.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            global_attention_mask: Global attention mask
            labels: Target labels
            output_attentions: Return attention weights
            output_hidden_states: Return hidden states
            
        Returns:
            Model outputs with predictions
        """
        # Prepare global attention mask if not provided
        if global_attention_mask is None:
            global_attention_mask = self._prepare_global_attention_mask(
                input_ids,
                attention_mask
            )
        
        # Forward through Longformer
        outputs = self.longformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            global_attention_mask=global_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # Get sequence representation
        sequence_output = outputs.last_hidden_state
        
        # Apply pooling
        pooled_output = self.pooler(sequence_output, attention_mask)
        
        # Classification head
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.functional.relu(pooled_output)
        pooled_output = self.classifier_dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return ModelOutputs(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            embeddings=pooled_output,
            metadata={
                "global_attention_mask": global_attention_mask,
                "num_global_positions": global_attention_mask.sum(dim=1).tolist()
            }
        )
