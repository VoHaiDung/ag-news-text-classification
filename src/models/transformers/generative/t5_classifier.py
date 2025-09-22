""" 
T5 Classifier for AG News Text Classification
==============================================

Implementation of T5 (Text-to-Text Transfer Transformer) adapted for classification,
based on:
- Raffel et al. (2020): "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- Xue et al. (2021): "mT5: A Massively Multilingual Pre-trained Text-to-Text Transformer"
- Nogueira et al. (2020): "Document Ranking with a Pretrained Sequence-to-Sequence Model"

Key adaptations:
1. Reformulation of classification as text generation
2. Verbalizer-based label mapping
3. Encoder-decoder architecture utilization
4. Prompt-based classification formulation

Mathematical Foundation:
T5 models all tasks as sequence-to-sequence:
P(y|x) = ∏ P(y_i | x, y_<i)
For classification: y is the textual label representation.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    T5Model,
    T5ForConditionalGeneration,
    T5Config,
    T5Tokenizer
)

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import MODELS
from src.core.exceptions import ModelInitializationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class T5ClassifierConfig:
    """Configuration for T5 classifier."""
    model_name: str = "t5-large"
    num_labels: int = 4
    max_source_length: int = 512
    max_target_length: int = 10
    dropout_rate: float = 0.1
    
    # Task prefix for T5
    task_prefix: str = "classify: "
    
    # Label verbalization
    label_map: Dict[int, str] = None  # Mapping from label ID to text
    
    # Classification strategy
    use_encoder_only: bool = False  # Use only encoder for classification
    use_generation: bool = True  # Use generation for classification
    
    # Training settings
    teacher_forcing: bool = True
    label_smoothing: float = 0.0
    beam_search: bool = False
    num_beams: int = 1
    
    # Efficient fine-tuning
    use_adapter: bool = False
    adapter_size: int = 64
    freeze_encoder: bool = False
    freeze_decoder: bool = False
    gradient_checkpointing: bool = False


class T5EncoderClassifier(nn.Module):
    """
    Classification head for T5 encoder-only classification.
    
    Uses T5 encoder representations for efficient classification
    without generation overhead.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1
    ):
        """
        Initialize T5 encoder classifier.
        
        Args:
            hidden_size: Hidden dimension
            num_labels: Number of classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.normal_(self.dense.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.zeros_(self.dense.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through classifier.
        
        Args:
            hidden_states: Encoder hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask
            
        Returns:
            Classification logits [batch_size, num_labels]
        """
        # Pool encoder outputs (mean pooling)
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = torch.mean(hidden_states, dim=1)
        
        # Classification layers
        pooled = self.dropout(pooled)
        pooled = self.dense(pooled)
        pooled = torch.tanh(pooled)
        pooled = self.dropout(pooled)
        logits = self.out_proj(pooled)
        
        return logits


class T5LabelVerbalizer:
    """
    Verbalizer for T5 label-to-text mapping.
    
    Handles bidirectional mapping between label IDs and text representations
    for T5's text-to-text formulation.
    """
    
    def __init__(
        self,
        tokenizer: T5Tokenizer,
        label_map: Dict[int, str]
    ):
        """
        Initialize verbalizer.
        
        Args:
            tokenizer: T5 tokenizer
            label_map: Mapping from label ID to text
        """
        self.tokenizer = tokenizer
        self.label_map = label_map
        self.reverse_map = {v: k for k, v in label_map.items()}
        
        # Tokenize labels
        self.label_tokens = {}
        for label_id, label_text in label_map.items():
            tokens = tokenizer.encode(label_text, add_special_tokens=False)
            self.label_tokens[label_id] = tokens
    
    def labels_to_tokens(
        self,
        labels: torch.Tensor,
        max_length: int = 10
    ) -> torch.Tensor:
        """
        Convert label IDs to token IDs.
        
        Args:
            labels: Label IDs [batch_size]
            max_length: Maximum token length
            
        Returns:
            Token IDs [batch_size, max_length]
        """
        batch_size = labels.size(0)
        token_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        
        for i, label in enumerate(labels):
            label_id = label.item()
            tokens = self.label_tokens[label_id]
            
            # Pad or truncate
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            
            token_ids[i, :len(tokens)] = torch.tensor(tokens)
            
            # Add EOS token
            if len(tokens) < max_length:
                token_ids[i, len(tokens)] = self.tokenizer.eos_token_id
        
        return token_ids
    
    def tokens_to_labels(
        self,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert generated tokens to label IDs.
        
        Args:
            token_ids: Generated token IDs [batch_size, seq_len]
            
        Returns:
            Label IDs [batch_size]
        """
        batch_size = token_ids.size(0)
        labels = torch.zeros(batch_size, dtype=torch.long)
        
        for i in range(batch_size):
            # Decode tokens to text
            tokens = token_ids[i].tolist()
            
            # Remove special tokens
            if self.tokenizer.eos_token_id in tokens:
                eos_idx = tokens.index(self.tokenizer.eos_token_id)
                tokens = tokens[:eos_idx]
            
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            text = text.strip().lower()
            
            # Map to label
            if text in self.reverse_map:
                labels[i] = self.reverse_map[text]
            else:
                # Default to first label if not found
                labels[i] = 0
        
        return labels


@MODELS.register("t5", aliases=["t5-classifier", "t5-large"])
class T5Classifier(AGNewsBaseModel):
    """
    T5 model adapted for text classification.
    
    Reformulates classification as a text-to-text task where:
    - Input: "classify: [text]"
    - Output: Label text (e.g., "sports", "business")
    
    Supports two modes:
    1. Generation mode: Generate label text
    2. Encoder-only mode: Use encoder for direct classification
    
    The model leverages T5's pre-training on diverse text-to-text tasks
    for improved transfer learning.
    """
    
    __auto_register__ = True
    __model_name__ = "t5"
    
    def __init__(self, config: Optional[T5ClassifierConfig] = None):
        """
        Initialize T5 classifier.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config or T5ClassifierConfig()
        
        # Default label map for AG News
        if self.config.label_map is None:
            self.config.label_map = {
                0: "world",
                1: "sports",
                2: "business",
                3: "technology"
            }
        
        # Initialize T5 components
        self._init_t5_model()
        self._init_classification_components()
        
        # Apply freezing if configured
        if self.config.freeze_encoder:
            self._freeze_encoder()
        if self.config.freeze_decoder:
            self._freeze_decoder()
        
        logger.info(
            f"Initialized T5 classifier with {self.config.num_labels} labels"
        )
    
    def _init_t5_model(self):
        """Initialize T5 model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
            
            # Load configuration
            t5_config = T5Config.from_pretrained(
                self.config.model_name,
                dropout_rate=self.config.dropout_rate,
                use_cache=False  # Disable caching for training
            )
            
            # Load model
            if self.config.use_generation:
                self.t5 = T5ForConditionalGeneration.from_pretrained(
                    self.config.model_name,
                    config=t5_config
                )
            else:
                self.t5 = T5Model.from_pretrained(
                    self.config.model_name,
                    config=t5_config
                )
            
            # Enable gradient checkpointing
            if self.config.gradient_checkpointing:
                self.t5.gradient_checkpointing_enable()
            
            self.hidden_size = self.t5.config.d_model
            
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize T5: {e}")
    
    def _init_classification_components(self):
        """Initialize classification-specific components."""
        # Label verbalizer
        self.verbalizer = T5LabelVerbalizer(
            self.tokenizer,
            self.config.label_map
        )
        
        # Encoder-only classifier if not using generation
        if self.config.use_encoder_only:
            self.encoder_classifier = T5EncoderClassifier(
                hidden_size=self.hidden_size,
                num_labels=self.config.num_labels,
                dropout=self.config.dropout_rate
            )
    
    def _freeze_encoder(self):
        """Freeze T5 encoder."""
        for param in self.t5.encoder.parameters():
            param.requires_grad = False
        logger.info("Froze T5 encoder")
    
    def _freeze_decoder(self):
        """Freeze T5 decoder."""
        if hasattr(self.t5, 'decoder'):
            for param in self.t5.decoder.parameters():
                param.requires_grad = False
            logger.info("Froze T5 decoder")
    
    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        prefix: str = None
    ) -> torch.Tensor:
        """
        Prepare inputs with task prefix.
        
        Args:
            input_ids: Original input IDs
            prefix: Task prefix
            
        Returns:
            Input IDs with prefix
        """
        prefix = prefix or self.config.task_prefix
        batch_size = input_ids.size(0)
        
        # Tokenize prefix
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        prefix_tensor = torch.tensor(prefix_ids).unsqueeze(0).repeat(batch_size, 1)
        prefix_tensor = prefix_tensor.to(input_ids.device)
        
        # Concatenate prefix with input
        prefixed_ids = torch.cat([prefix_tensor, input_ids], dim=1)
        
        # Truncate if necessary
        max_length = self.config.max_source_length
        if prefixed_ids.size(1) > max_length:
            prefixed_ids = prefixed_ids[:, :max_length]
        
        return prefixed_ids
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False
    ) -> ModelOutputs:
        """
        Forward pass through T5 classifier.
        
        T5's encoder-decoder architecture is utilized by:
        1. Encoding input text with task prefix
        2. Either generating label text or using encoder for classification
        3. Computing loss against target labels
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask for input
            labels: Target labels
            decoder_input_ids: Decoder input IDs (for training)
            decoder_attention_mask: Decoder attention mask
            output_attentions: Return attention weights
            output_hidden_states: Return hidden states
            
        Returns:
            Model outputs with logits and optional loss
        """
        # Prepare inputs with task prefix
        input_ids = self._prepare_inputs(input_ids)
        
        # Update attention mask if needed
        if attention_mask is not None:
            prefix_len = len(self.tokenizer.encode(self.config.task_prefix, add_special_tokens=False))
            prefix_mask = torch.ones(attention_mask.size(0), prefix_len, device=attention_mask.device)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
            
            # Truncate if necessary
            if attention_mask.size(1) > self.config.max_source_length:
                attention_mask = attention_mask[:, :self.config.max_source_length]
        
        if self.config.use_encoder_only:
            # Encoder-only classification
            encoder_outputs = self.t5.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states
            )
            
            # Classification
            logits = self.encoder_classifier(
                encoder_outputs.last_hidden_state,
                attention_mask
            )
            
            # Compute loss
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            
            return ModelOutputs(
                loss=loss,
                logits=logits,
                hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                attentions=encoder_outputs.attentions if output_attentions else None
            )
        
        else:
            # Generation-based classification
            if labels is not None:
                # Convert labels to target tokens
                target_ids = self.verbalizer.labels_to_tokens(
                    labels,
                    max_length=self.config.max_target_length
                ).to(input_ids.device)
                
                # Teacher forcing for training
                if self.training and self.config.teacher_forcing:
                    decoder_input_ids = target_ids
            
            # Forward through T5
            outputs = self.t5(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=target_ids if labels is not None else None,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True
            )
            
            # For inference, generate labels
            if not self.training and labels is None:
                generated_ids = self.t5.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=self.config.max_target_length,
                    num_beams=self.config.num_beams if self.config.beam_search else 1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Convert generated tokens to label logits
                predicted_labels = self.verbalizer.tokens_to_labels(generated_ids)
                
                # Create pseudo-logits for compatibility
                batch_size = input_ids.size(0)
                logits = torch.zeros(batch_size, self.config.num_labels, device=input_ids.device)
                logits[torch.arange(batch_size), predicted_labels] = 10.0  # High confidence
            else:
                # Use decoder logits
                logits = outputs.logits if hasattr(outputs, 'logits') else None
            
            return ModelOutputs(
                loss=outputs.loss if hasattr(outputs, 'loss') else None,
                logits=logits,
                hidden_states=outputs.encoder_hidden_states if output_hidden_states else None,
                attentions=outputs.encoder_attentions if output_attentions else None,
                metadata={
                    "decoder_hidden_states": outputs.decoder_hidden_states if output_hidden_states else None,
                    "decoder_attentions": outputs.decoder_attentions if output_attentions else None,
                    "generated_ids": generated_ids if not self.training and labels is None else None
                }
            )
