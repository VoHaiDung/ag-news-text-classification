"""
XLNet Classifier for AG News
=============================

Implementation of XLNet (Generalized Autoregressive Pretraining) for text classification,
based on:
- Yang et al. (2019): "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
- Dai et al. (2019): "Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context"

Key Features:
1. Permutation language modeling
2. Two-stream self-attention
3. Relative positional encoding
4. Segment recurrence mechanism

Mathematical Foundation:
XLNet maximizes the expected log likelihood:
max_θ E[Σ log p_θ(x_t | x_<t)]
over all possible permutations of the factorization order.

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
    XLNetModel,
    XLNetConfig,
    XLNetTokenizer
)

from src.models.base.base_model import TransformerBaseModel, ModelOutputs
from src.models.base.pooling_strategies import PoolingFactory
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class XLNetClassifierConfig:
    """Configuration for XLNet classifier."""
    model_name: str = "xlnet-large-cased"
    num_labels: int = 4
    max_length: int = 512
    dropout_prob: float = 0.1
    classifier_dropout: float = 0.2
    pooling_strategy: str = "last"  # XLNet typically uses last token
    use_cls_token: bool = False
    mem_len: int = 0  # Memory length for Transformer-XL
    reuse_len: int = 0  # Reuse length
    bi_data: bool = False  # Bidirectional data
    clamp_len: int = -1  # Clamp length
    same_length: bool = False  # Use same length attention
    summary_type: str = "last"  # Summary type for pooling
    gradient_checkpointing: bool = False


@MODELS.register("xlnet", aliases=["xlnet-large"])
class XLNetClassifier(TransformerBaseModel):
    """
    XLNet classifier for AG News.
    
    XLNet's autoregressive pretraining and permutation-based training
    provide strong performance on text classification through:
    1. Capturing bidirectional context
    2. Avoiding pretrain-finetune discrepancy
    3. Better handling of long sequences
    """
    
    __auto_register__ = True
    __model_name__ = "xlnet"
    
    def __init__(self, config: Optional[XLNetClassifierConfig] = None):
        """
        Initialize XLNet classifier.
        
        Args:
            config: Model configuration
        """
        self.config = config or XLNetClassifierConfig()
        
        # Initialize base
        super().__init__(pretrained_model_name=self.config.model_name)
        
        # Initialize XLNet
        self._init_xlnet_model()
        self._init_sequence_summary()
        self._init_classifier()
        
        logger.info(f"Initialized XLNet classifier with {self.config.num_labels} labels")
    
    def _init_xlnet_model(self):
        """Initialize XLNet model with specific configuration."""
        # Load configuration
        xlnet_config = XLNetConfig.from_pretrained(
            self.config.model_name,
            num_labels=self.config.num_labels,
            dropout=self.config.dropout_prob,
            mem_len=self.config.mem_len,
            reuse_len=self.config.reuse_len,
            bi_data=self.config.bi_data,
            clamp_len=self.config.clamp_len,
            same_length=self.config.same_length
        )
        
        # Load pretrained model
        self.xlnet = XLNetModel.from_pretrained(
            self.config.model_name,
            config=xlnet_config
        )
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            self.xlnet.gradient_checkpointing_enable()
        
        self.hidden_size = self.xlnet.config.d_model
    
    def _init_sequence_summary(self):
        """
        Initialize sequence summary for XLNet.
        
        XLNet uses a special summary mechanism different from BERT.
        """
        self.sequence_summary = nn.Identity()  # Will implement custom summary
        
        # Pooling strategy
        if self.config.pooling_strategy == "last":
            # XLNet typically uses the last token
            self.pool_pos = -1
        elif self.config.pooling_strategy == "first":
            self.pool_pos = 0
        else:
            # Use pooling factory for other strategies
            self.pooler = PoolingFactory.create_pooler(
                strategy=self.config.pooling_strategy,
                hidden_size=self.hidden_size
            )
            self.pool_pos = None
    
    def _init_classifier(self):
        """Initialize classification head for XLNet."""
        # Sequence summary projection
        self.summary_projection = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.classifier_dropout)
        )
        
        # Classifier
        self.classifier = nn.Linear(self.hidden_size, self.config.num_labels)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize classifier weights."""
        nn.init.normal_(self.summary_projection[0].weight, std=0.02)
        nn.init.zeros_(self.summary_projection[0].bias)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        input_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        mems: Optional[torch.Tensor] = None,
        perm_mask: Optional[torch.Tensor] = None,
        target_mapping: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> ModelOutputs:
        """
        Forward pass through XLNet.
        
        XLNet's forward pass includes:
        1. Permutation-based attention computation
        2. Two-stream self-attention
        3. Relative positional encoding
        4. Memory mechanism for long sequences
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask (0 for padding)
            token_type_ids: Segment token indices
            input_mask: Float mask for input
            head_mask: Mask for attention heads
            labels: Target labels
            mems: Memory from previous segment
            perm_mask: Permutation mask for training
            target_mapping: Target mapping for prediction
            output_attentions: Return attention weights
            output_hidden_states: Return hidden states
            
        Returns:
            Model outputs with predictions
        """
        # XLNet uses input_mask instead of attention_mask internally
        if input_mask is None and attention_mask is not None:
            input_mask = attention_mask.float()
        
        # Forward through XLNet
        outputs = self.xlnet(
            input_ids=input_ids,
            attention_mask=input_mask,
            mems=mems,
            perm_mask=perm_mask,
            target_mapping=target_mapping,
            token_type_ids=token_type_ids,
            input_mask=input_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        # Get sequence output
        sequence_output = outputs.last_hidden_state
        
        # Apply pooling
        if self.pool_pos is not None:
            # Use specific position
            pooled_output = sequence_output[:, self.pool_pos]
        else:
            # Use pooling strategy
            pooled_output = self.pooler(sequence_output, attention_mask)
        
        # Apply summary projection
        pooled_output = self.summary_projection(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return ModelOutputs(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            embeddings=pooled_output,
            metadata={"mems": outputs.mems} if hasattr(outputs, "mems") else {}
        )
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation with XLNet.
        
        Handles memory mechanism for generation.
        """
        # Remove token_type_ids if past is present
        if past:
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "mems": past,
            "use_mems": True if past else False
        }
