"""
Hierarchical DeBERTa for Long Document Classification
==========================================================

Implementation of hierarchical attention mechanism for DeBERTa to handle
long documents, inspired by:
- Yang et al. (2016): "Hierarchical Attention Networks for Document Classification"
- Zhang et al. (2019): "HIBERT: Document Level Pre-training of Hierarchical BERT"
- Pappagari et al. (2019): "Hierarchical Transformers for Long Document Classification"

The model processes documents in a hierarchical manner:
1. Segment-level encoding with DeBERTa
2. Document-level attention over segments
3. Global classification based on document representation

Mathematical Foundation:
Document D = [S₁, S₂, ..., Sₙ] where Sᵢ are segments
h_i = DeBERTa(Sᵢ) for segment encoding
H = Attention([h₁, h₂, ..., hₙ]) for document encoding

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DebertaV2Model,
    DebertaV2Config,
    DebertaV2Tokenizer
)

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.models.base.pooling_strategies import PoolingStrategy, create_pooling_layer
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism for document-level encoding
    
    Computes attention over segment representations to create
    a unified document representation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        use_positional_encoding: bool = True,
        max_segments: int = 32
    ):
        """
        Initialize hierarchical attention.
        
        Args:
            hidden_size: Hidden dimension size
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout rate for attention
            use_positional_encoding: Whether to use positional encoding
            max_segments: Maximum number of segments
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        # Multi-head attention components
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(attention_dropout)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Positional encoding
        if use_positional_encoding:
            self.position_embeddings = nn.Embedding(max_segments, hidden_size)
        else:
            self.position_embeddings = None
        
        # Learnable CLS token for document representation
        self.doc_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        
        logger.info(f"Initialized hierarchical attention with {num_attention_heads} heads")
    
    def forward(
        self,
        segment_embeddings: torch.Tensor,
        segment_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply hierarchical attention over segments.
        
        Args:
            segment_embeddings: Segment representations [batch_size, num_segments, hidden_size]
            segment_mask: Mask for valid segments [batch_size, num_segments]
            
        Returns:
            Tuple of (document_embedding, attention_weights)
        """
        batch_size, num_segments, hidden_size = segment_embeddings.shape
        
        # Add positional encoding if available
        if self.position_embeddings is not None:
            position_ids = torch.arange(num_segments, device=segment_embeddings.device)
            position_embeds = self.position_embeddings(position_ids)
            segment_embeddings = segment_embeddings + position_embeds.unsqueeze(0)
        
        # Add document token
        doc_token = self.doc_token.expand(batch_size, -1, -1)
        segment_embeddings = torch.cat([doc_token, segment_embeddings], dim=1)
        
        # Update mask for document token
        if segment_mask is not None:
            doc_mask = torch.ones(batch_size, 1, device=segment_mask.device)
            segment_mask = torch.cat([doc_mask, segment_mask], dim=1)
        
        # Compute Q, K, V
        Q = self.query(segment_embeddings)
        K = self.key(segment_embeddings)
        V = self.value(segment_embeddings)
        
        # Reshape for multi-head attention
        Q = self._reshape_for_attention(Q)
        K = self._reshape_for_attention(K)
        V = self._reshape_for_attention(V)
        
        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # Apply mask if provided
        if segment_mask is not None:
            # Expand mask for attention heads
            extended_mask = segment_mask.unsqueeze(1).unsqueeze(2)
            extended_mask = extended_mask.expand(-1, self.num_attention_heads, -1, -1)
            attention_scores = attention_scores.masked_fill(
                extended_mask == 0,
                float('-inf')
            )
        
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # Apply attention to values
        context = torch.matmul(attention_probs, V)
        
        # Reshape back
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.hidden_size)
        
        # Output projection
        output = self.output_projection(context)
        
        # Add residual and normalize
        output = self.layer_norm(output + segment_embeddings)
        
        # Extract document representation (first token)
        doc_representation = output[:, 0, :]
        
        # Average attention weights for visualization
        attention_weights = attention_probs.mean(dim=1)[:, 0, 1:]  # Doc token attention to segments
        
        return doc_representation, attention_weights
    
    def _reshape_for_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape tensor for multi-head attention"""
        batch_size, seq_length, _ = x.shape
        
        x = x.view(batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        return x.permute(0, 2, 1, 3)


@dataclass
class HierarchicalDeBERTaConfig:
    """Configuration for Hierarchical DeBERTa"""
    
    # Model configuration
    model_name: str = "microsoft/deberta-v3-base"
    num_labels: int = 4
    
    # Hierarchical configuration
    max_segment_length: int = 256  # Maximum tokens per segment
    max_segments: int = 16  # Maximum number of segments
    segment_overlap: int = 50  # Token overlap between segments
    
    # Attention configuration
    num_attention_heads: int = 8
    attention_dropout: float = 0.1
    use_positional_encoding: bool = True
    
    # Pooling strategy for segments
    segment_pooling: str = "cls"  # "cls", "mean", "max", "attention"
    
    # Document-level configuration
    use_hierarchical_loss: bool = True  # Multi-level loss
    segment_loss_weight: float = 0.3  # Weight for segment-level loss
    
    # Training configuration
    freeze_embeddings: bool = False
    freeze_lower_layers: int = 0  # Number of lower layers to freeze
    gradient_checkpointing: bool = False
    
    # Advanced features
    use_sliding_window: bool = True
    use_segment_type_embeddings: bool = True
    cross_segment_attention: bool = False  # Allow attention across segments


@MODELS.register("deberta_hierarchical")
class HierarchicalDeBERTa(AGNewsBaseModel):
    """
    Hierarchical DeBERTa for Long Document Classification
    
    Processes long documents by:
    1. Splitting into overlapping segments
    2. Encoding each segment with DeBERTa
    3. Applying hierarchical attention over segments
    4. Generating document-level predictions
    
    This architecture enables processing of documents longer than
    the maximum sequence length of the base model.
    """
    
    def __init__(self, config: Optional[HierarchicalDeBERTaConfig] = None):
        """
        Initialize Hierarchical DeBERTa.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config or HierarchicalDeBERTaConfig()
        
        # Initialize tokenizer and base model
        self._init_base_components()
        
        # Initialize hierarchical components
        self._init_hierarchical_components()
        
        # Classification heads
        self._init_classification_heads()
        
        # Apply freezing if configured
        self._apply_freezing()
        
        logger.info(
            f"Initialized Hierarchical DeBERTa: "
            f"{self.config.max_segments} segments × {self.config.max_segment_length} tokens"
        )
    
    def _init_base_components(self):
        """Initialize DeBERTa model and tokenizer"""
        # Load tokenizer
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.config.model_name)
        
        # Load configuration
        deberta_config = DebertaV2Config.from_pretrained(self.config.model_name)
        
        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            deberta_config.gradient_checkpointing = True
        
        # Load model
        self.deberta = DebertaV2Model.from_pretrained(
            self.config.model_name,
            config=deberta_config
        )
        
        self.hidden_size = self.deberta.config.hidden_size
        
        # Segment type embeddings if configured
        if self.config.use_segment_type_embeddings:
            self.segment_type_embeddings = nn.Embedding(
                self.config.max_segments,
                self.hidden_size
            )
    
    def _init_hierarchical_components(self):
        """Initialize hierarchical attention components"""
        # Segment-level pooling
        self.segment_pooler = create_pooling_layer(
            PoolingStrategy(self.config.segment_pooling),
            self.hidden_size
        )
        
        # Hierarchical attention
        self.hierarchical_attention = HierarchicalAttention(
            hidden_size=self.hidden_size,
            num_attention_heads=self.config.num_attention_heads,
            attention_dropout=self.config.attention_dropout,
            use_positional_encoding=self.config.use_positional_encoding,
            max_segments=self.config.max_segments
        )
        
        # Optional: Cross-segment attention
        if self.config.cross_segment_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.config.num_attention_heads,
                dropout=self.config.attention_dropout,
                batch_first=True
            )
    
    def _init_classification_heads(self):
        """Initialize classification heads"""
        # Document-level classifier
        self.document_classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, self.config.num_labels)
        )
        
        # Optional: Segment-level classifier for auxiliary loss
        if self.config.use_hierarchical_loss:
            self.segment_classifier = nn.Linear(self.hidden_size, self.config.num_labels)
    
    def _apply_freezing(self):
        """Apply parameter freezing if configured"""
        if self.config.freeze_embeddings:
            for param in self.deberta.embeddings.parameters():
                param.requires_grad = False
            logger.info("Froze embedding parameters")
        
        if self.config.freeze_lower_layers > 0:
            layers_to_freeze = self.deberta.encoder.layer[:self.config.freeze_lower_layers]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
            logger.info(f"Froze {self.config.freeze_lower_layers} lower layers")
    
    def _segment_document(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Segment document into overlapping chunks.
        
        Args:
            input_ids: Full document token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            
        Returns:
            Tuple of (segmented_ids, segmented_mask)
        """
        batch_size, seq_length = input_ids.shape
        segment_length = self.config.max_segment_length
        overlap = self.config.segment_overlap
        stride = segment_length - overlap
        
        segments = []
        segment_masks = []
        
        for i in range(0, seq_length, stride):
            # Extract segment
            end_idx = min(i + segment_length, seq_length)
            segment = input_ids[:, i:end_idx]
            mask = attention_mask[:, i:end_idx]
            
            # Pad if necessary
            if segment.shape[1] < segment_length:
                pad_length = segment_length - segment.shape[1]
                segment = F.pad(segment, (0, pad_length), value=self.tokenizer.pad_token_id)
                mask = F.pad(mask, (0, pad_length), value=0)
            
            segments.append(segment)
            segment_masks.append(mask)
            
            # Stop if we've reached max segments
            if len(segments) >= self.config.max_segments:
                break
        
        # Pad to max_segments if necessary
        while len(segments) < self.config.max_segments:
            padding = torch.full(
                (batch_size, segment_length),
                self.tokenizer.pad_token_id,
                device=input_ids.device
            )
            segments.append(padding)
            segment_masks.append(torch.zeros_like(padding))
        
        # Stack segments
        segmented_ids = torch.stack(segments, dim=1)  # [batch, num_segments, segment_length]
        segmented_mask = torch.stack(segment_masks, dim=1)
        
        return segmented_ids, segmented_mask
    
    def _encode_segments(
        self,
        segmented_ids: torch.Tensor,
        segmented_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode each segment with DeBERTa.
        
        Args:
            segmented_ids: Segmented token IDs [batch, num_segments, segment_length]
            segmented_mask: Segmented attention mask
            
        Returns:
            Segment embeddings [batch, num_segments, hidden_size]
        """
        batch_size, num_segments, segment_length = segmented_ids.shape
        
        # Flatten for batch processing
        flat_ids = segmented_ids.view(-1, segment_length)
        flat_mask = segmented_mask.view(-1, segment_length)
        
        # Encode all segments
        outputs = self.deberta(
            input_ids=flat_ids,
            attention_mask=flat_mask
        )
        
        # Get segment representations
        segment_hidden = outputs.last_hidden_state  # [batch*segments, length, hidden]
        
        # Pool over tokens in each segment
        segment_embeddings = self.segment_pooler(segment_hidden, flat_mask)
        
        # Reshape back to [batch, num_segments, hidden]
        segment_embeddings = segment_embeddings.view(batch_size, num_segments, self.hidden_size)
        
        # Add segment type embeddings if configured
        if self.config.use_segment_type_embeddings:
            segment_positions = torch.arange(num_segments, device=segment_embeddings.device)
            segment_type_embeds = self.segment_type_embeddings(segment_positions)
            segment_embeddings = segment_embeddings + segment_type_embeds.unsqueeze(0)
        
        return segment_embeddings
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through Hierarchical DeBERTa.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask
            labels: Target labels
            return_attention_weights: Whether to return hierarchical attention weights
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with hierarchical processing
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Handle attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        # Segment the document
        segmented_ids, segmented_mask = self._segment_document(input_ids, attention_mask)
        
        # Encode segments
        segment_embeddings = self._encode_segments(segmented_ids, segmented_mask)
        
        # Create segment-level mask (which segments are valid)
        segment_validity = (segmented_mask.sum(dim=2) > 0).float()  # [batch, num_segments]
        
        # Apply hierarchical attention
        doc_embedding, attention_weights = self.hierarchical_attention(
            segment_embeddings,
            segment_validity
        )
        
        # Optional: Cross-segment attention
        if self.config.cross_segment_attention:
            cross_output, _ = self.cross_attention(
                segment_embeddings,
                segment_embeddings,
                segment_embeddings,
                key_padding_mask=(segment_validity == 0)
            )
            # Combine with hierarchical output
            doc_embedding = doc_embedding + cross_output.mean(dim=1)
        
        # Document-level classification
        doc_logits = self.document_classifier(doc_embedding)
        
        # Calculate losses
        loss = None
        if labels is not None:
            # Document-level loss
            doc_loss = F.cross_entropy(doc_logits, labels)
            loss = doc_loss
            
            # Optional: Segment-level auxiliary loss
            if self.config.use_hierarchical_loss and hasattr(self, 'segment_classifier'):
                # Segment-level predictions
                segment_logits = self.segment_classifier(segment_embeddings)
                
                # Expand labels for all segments
                expanded_labels = labels.unsqueeze(1).expand(-1, segment_embeddings.shape[1])
                
                # Flatten for loss calculation
                segment_logits_flat = segment_logits.view(-1, self.config.num_labels)
                expanded_labels_flat = expanded_labels.view(-1)
                segment_validity_flat = segment_validity.view(-1)
                
                # Calculate segment loss only for valid segments
                segment_loss = F.cross_entropy(
                    segment_logits_flat[segment_validity_flat > 0],
                    expanded_labels_flat[segment_validity_flat > 0]
                )
                
                # Combine losses
                loss = doc_loss + self.config.segment_loss_weight * segment_loss
        
        # Prepare outputs
        outputs = ModelOutputs(
            loss=loss,
            logits=doc_logits,
            hidden_states=doc_embedding,
            metadata={
                'num_segments': int(segment_validity.sum(dim=1).mean().item()),
                'max_segments': self.config.max_segments,
                'segment_length': self.config.max_segment_length
            }
        )
        
        if return_attention_weights:
            outputs.attentions = attention_weights
            outputs.metadata['hierarchical_attention'] = attention_weights.cpu().numpy()
        
        return outputs
    
    def get_segment_importance(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Get importance scores for each segment.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Segment importance scores
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask,
                return_attention_weights=True
            )
            
            # Extract attention weights (importance scores)
            importance = outputs.attentions.cpu().numpy()
            
        return importance
