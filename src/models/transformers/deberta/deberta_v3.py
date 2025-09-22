"""
DeBERTa-v3 Model Implementation for AG News Classification
===========================================================

This module implements DeBERTa-v3 (Decoding-enhanced BERT with Disentangled Attention)
following the architecture and improvements from:
- He et al. (2021): "DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training"
- He et al. (2020): "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"

Key innovations:
1. Disentangled attention mechanism
2. Enhanced mask decoder
3. ELECTRA-style pre-training (for v3)
4. Improved position embeddings

Mathematical Foundation:
The disentangled attention computes:
A_{i,j} = Q_i K_j^T + Q_i^r K_j^{r,T} + Q_i^c K_j^{c,T}
where r denotes relative position and c denotes content.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    DebertaV2Model,
    DebertaV2ForSequenceClassification,
    DebertaV2Config,
    AutoTokenizer
)

from src.models.base.base_model import TransformerBaseModel, ModelOutputs
from src.models.base.pooling_strategies import PoolingFactory, PoolingStrategy
from src.core.registry import MODELS
from src.core.exceptions import ModelInitializationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DeBERTaV3Config:
    """
    Configuration for DeBERTa-v3 model.
    
    Attributes:
        model_name: Pretrained model name or path
        num_labels: Number of classification labels
        max_length: Maximum sequence length
        hidden_dropout_prob: Dropout probability for hidden layers
        attention_probs_dropout_prob: Dropout probability for attention
        classifier_dropout: Dropout for classification head
        pooling_strategy: Strategy for pooling token representations
        use_differential_lr: Use different learning rates for layers
        layer_wise_lr_decay: Decay factor for layer-wise learning rates
        freeze_embeddings: Whether to freeze embedding layers
        freeze_n_layers: Number of encoder layers to freeze
        gradient_checkpointing: Enable gradient checkpointing
        position_embedding_type: Type of position embedding
        use_enhanced_decoder: Use enhanced mask decoder
        disentangled_attention: Use disentangled attention mechanism
    """
    model_name: str = "microsoft/deberta-v3-large"
    num_labels: int = 4
    max_length: int = 512
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    classifier_dropout: float = 0.15
    pooling_strategy: str = "mean"
    use_differential_lr: bool = True
    layer_wise_lr_decay: float = 0.95
    freeze_embeddings: bool = False
    freeze_n_layers: int = 0
    gradient_checkpointing: bool = False
    position_embedding_type: str = "relative_key_query"
    use_enhanced_decoder: bool = True
    disentangled_attention: bool = True


@MODELS.register("deberta-v3", aliases=["deberta", "debertav3"])
class DeBERTaV3Classifier(TransformerBaseModel):
    """
    DeBERTa-v3 model for AG News classification.
    
    This implementation provides state-of-the-art performance through:
    1. Disentangled attention mechanism for better position modeling
    2. Enhanced mask decoder for improved masked language modeling
    3. Optimized architecture from ELECTRA-style pre-training
    
    The model achieves superior performance on text classification tasks
    through improved attention patterns and position encoding.
    """
    
    __auto_register__ = True
    __model_name__ = "deberta-v3"
    
    def __init__(self, config: Optional[DeBERTaV3Config] = None):
        """
        Initialize DeBERTa-v3 classifier.
        
        Args:
            config: Model configuration
        """
        self.config = config or DeBERTaV3Config()
        
        # Initialize base transformer
        super().__init__(pretrained_model_name=self.config.model_name)
        
        # Override with DeBERTa specific configuration
        self._init_deberta_model()
        self._init_classification_head()
        self._init_pooling()
        
        # Apply initialization strategies
        if self.config.freeze_embeddings:
            self._freeze_embeddings()
        if self.config.freeze_n_layers > 0:
            self._freeze_encoder_layers(self.config.freeze_n_layers)
        
        logger.info(f"Initialized DeBERTa-v3 model with {self.config.num_labels} labels")
    
    def _init_deberta_model(self):
        """Initialize DeBERTa-v3 specific architecture."""
        try:
            # Load configuration
            model_config = DebertaV2Config.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels,
                hidden_dropout_prob=self.config.hidden_dropout_prob,
                attention_probs_dropout_prob=self.config.attention_probs_dropout_prob,
                position_embedding_type=self.config.position_embedding_type
            )
            
            # Initialize model
            self.deberta = DebertaV2Model.from_pretrained(
                self.config.model_name,
                config=model_config
            )
            
            # Enable gradient checkpointing if specified
            if self.config.gradient_checkpointing:
                self.deberta.gradient_checkpointing_enable()
            
            # Get hidden size for classification head
            self.hidden_size = self.deberta.config.hidden_size
            
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize DeBERTa-v3: {e}")
    
    def _init_classification_head(self):
        """
        Initialize enhanced classification head.
        
        Uses a multi-layer classification head with dropout and
        layer normalization for improved generalization.
        """
        self.pre_classifier = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.config.num_labels)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.pre_classifier.weight)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.pre_classifier.bias)
        nn.init.zeros_(self.classifier.bias)
    
    def _init_pooling(self):
        """Initialize pooling strategy."""
        self.pooler = PoolingFactory.create_pooler(
            strategy=self.config.pooling_strategy,
            hidden_size=self.hidden_size,
            dropout_rate=self.config.hidden_dropout_prob
        )
    
    def _freeze_embeddings(self):
        """Freeze embedding layers for fine-tuning."""
        for param in self.deberta.embeddings.parameters():
            param.requires_grad = False
        logger.info("Froze embedding layers")
    
    def _freeze_encoder_layers(self, n_layers: int):
        """
        Freeze first n encoder layers.
        
        Args:
            n_layers: Number of layers to freeze from bottom
        """
        for i in range(n_layers):
            for param in self.deberta.encoder.layer[i].parameters():
                param.requires_grad = False
        logger.info(f"Froze first {n_layers} encoder layers")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> ModelOutputs:
        """
        Forward pass through DeBERTa-v3 model.
        
        The forward pass implements:
        1. Token embeddings with disentangled position encoding
        2. Multiple transformer layers with disentangled attention
        3. Pooling strategy for sequence representation
        4. Classification head with regularization
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            token_type_ids: Token type IDs for segment embeddings
            position_ids: Position IDs for custom positioning
            labels: Target labels for computing loss
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            return_dict: Return ModelOutputs object
            
        Returns:
            Model outputs with logits and optional loss
        """
        # Pass through DeBERTa encoder
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # Get sequence representation
        sequence_output = outputs.last_hidden_state
        
        # Apply pooling strategy
        pooled_output = self.pooler(sequence_output, attention_mask)
        
        # Classification head
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.functional.gelu(pooled_output)
        pooled_output = self.layer_norm(pooled_output)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return ModelOutputs(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            embeddings=pooled_output
        )
    
    def get_layer_parameters(self):
        """
        Get parameters grouped by layers for differential learning rates.
        
        Returns:
            List of parameter groups with decay factors
        """
        if not self.config.use_differential_lr:
            return self.parameters()
        
        # Group parameters by layer
        parameter_groups = []
        
        # Embeddings - lowest learning rate
        parameter_groups.append({
            'params': self.deberta.embeddings.parameters(),
            'lr_scale': self.config.layer_wise_lr_decay ** (len(self.deberta.encoder.layer) + 1)
        })
        
        # Encoder layers - increasing learning rate
        for i, layer in enumerate(self.deberta.encoder.layer):
            lr_scale = self.config.layer_wise_lr_decay ** (len(self.deberta.encoder.layer) - i)
            parameter_groups.append({
                'params': layer.parameters(),
                'lr_scale': lr_scale
            })
        
        # Classification head - highest learning rate
        parameter_groups.append({
            'params': list(self.pre_classifier.parameters()) + 
                     list(self.classifier.parameters()) +
                     list(self.layer_norm.parameters()),
            'lr_scale': 1.0
        })
        
        return parameter_groups
    
    def resize_position_embeddings(self, new_max_position_embeddings: int):
        """
        Resize position embeddings for longer sequences.
        
        Args:
            new_max_position_embeddings: New maximum position
        """
        self.deberta.resize_position_embeddings(new_max_position_embeddings)
        logger.info(f"Resized position embeddings to {new_max_position_embeddings}")
