"""
GPT-2 Classifier for AG News Text Classification
=================================================

Implementation of GPT-2 adapted for text classification tasks, based on:
- Radford et al. (2019): "Language Models are Unsupervised Multitask Learners"
- Liu et al. (2021): "GPT Understands, Too"
- Radford et al. (2018): "Improving Language Understanding by Generative Pre-Training"

Key adaptations for classification:
1. Use of final token representation for classification
2. Optional prompt-based classification
3. Causal attention masking preservation
4. Classification head on top of generative model

Mathematical Foundation:
GPT-2 uses autoregressive modeling:
P(x) = ∏ P(x_i | x_<i)
For classification, we use: P(y|x) = softmax(W·h_n + b)
where h_n is the representation of the last token.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    GPT2Model,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

from src.models.base.base_model import TransformerBaseModel, ModelOutputs
from src.models.base.pooling_strategies import PoolingFactory
from src.core.registry import MODELS
from src.core.exceptions import ModelInitializationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class GPT2ClassifierConfig:
    """Configuration for GPT-2 classifier."""
    model_name: str = "gpt2-large"
    num_labels: int = 4
    max_length: int = 512
    dropout_prob: float = 0.1
    classifier_dropout: float = 0.2
    pooling_strategy: str = "last"  # GPT-2 typically uses last token
    use_lm_head: bool = False  # Use language modeling head for classification
    use_prompt: bool = False  # Use prompt-based classification
    prompt_template: str = "This news article is about"
    freeze_embeddings: bool = False
    freeze_n_layers: int = 0
    gradient_checkpointing: bool = False
    use_prefix_tuning: bool = False
    prefix_length: int = 10
    use_cls_token: bool = True  # Add CLS token for classification
    label_smoothing: float = 0.0
    temperature: float = 1.0


class PrefixTuning(nn.Module):
    """
    Prefix tuning for efficient fine-tuning of GPT-2.
    
    Based on Li & Liang (2021): "Prefix-Tuning: Optimizing Continuous Prompts"
    """
    
    def __init__(
        self,
        prefix_length: int,
        hidden_size: int,
        num_layers: int
    ):
        """
        Initialize prefix tuning.
        
        Args:
            prefix_length: Length of prefix
            hidden_size: Hidden dimension
            num_layers: Number of transformer layers
        """
        super().__init__()
        
        self.prefix_length = prefix_length
        self.num_layers = num_layers
        
        # Prefix embeddings for each layer
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_layers * 2, prefix_length, hidden_size)
        )
        
        # MLP to process prefix
        self.prefix_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.Tanh(),
            nn.Linear(hidden_size * 4, num_layers * 2 * hidden_size)
        )
        
        # Initialize
        nn.init.xavier_uniform_(self.prefix_embeddings)
    
    def forward(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Get prefix key-value pairs for all layers.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Dictionary with past key values
        """
        # Process prefix embeddings
        prefix = self.prefix_mlp(self.prefix_embeddings.mean(dim=0))
        prefix = prefix.view(self.prefix_length, self.num_layers * 2, -1)
        
        # Split into key and value for each layer
        past_key_values = []
        for i in range(self.num_layers):
            key = prefix[:, 2*i].unsqueeze(0).expand(batch_size, -1, -1)
            value = prefix[:, 2*i+1].unsqueeze(0).expand(batch_size, -1, -1)
            past_key_values.append((key, value))
        
        return tuple(past_key_values)


class GPT2ClassificationHead(nn.Module):
    """
    Classification head for GPT-2.
    
    Implements various strategies for extracting classification signals
    from autoregressive language models.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        dropout: float = 0.1,
        use_multi_layer: bool = True
    ):
        """
        Initialize classification head.
        
        Args:
            hidden_size: Hidden dimension
            num_labels: Number of classes
            dropout: Dropout rate
            use_multi_layer: Use multi-layer head
        """
        super().__init__()
        
        if use_multi_layer:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, num_labels)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_labels)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through classification head.
        
        Args:
            hidden_states: Hidden states from GPT-2
            
        Returns:
            Classification logits
        """
        return self.classifier(hidden_states)


@MODELS.register("gpt2", aliases=["gpt2-classifier", "gpt2-large"])
class GPT2Classifier(TransformerBaseModel):
    """
    GPT-2 model adapted for text classification.
    
    This implementation adapts the autoregressive GPT-2 model for
    classification tasks through:
    1. Using the final token representation for classification
    2. Optional prompt-based formulation
    3. Prefix tuning for efficient adaptation
    4. Preserving causal masking for consistency
    
    The model maintains GPT-2's strengths in understanding context
    while adapting it for discriminative tasks.
    """
    
    __auto_register__ = True
    __model_name__ = "gpt2"
    
    def __init__(self, config: Optional[GPT2ClassifierConfig] = None):
        """
        Initialize GPT-2 classifier.
        
        Args:
            config: Model configuration
        """
        self.config = config or GPT2ClassifierConfig()
        
        # Initialize base transformer
        super().__init__(pretrained_model_name=self.config.model_name)
        
        # Initialize GPT-2 components
        self._init_gpt2_model()
        self._init_classification_components()
        
        # Apply initialization strategies
        if self.config.freeze_embeddings:
            self._freeze_embeddings()
        if self.config.freeze_n_layers > 0:
            self._freeze_encoder_layers(self.config.freeze_n_layers)
        
        logger.info(
            f"Initialized GPT-2 classifier with {self.config.num_labels} labels"
        )
    
    def _init_gpt2_model(self):
        """Initialize GPT-2 model and tokenizer."""
        try:
            # Load configuration
            gpt2_config = GPT2Config.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels,
                resid_pdrop=self.config.dropout_prob,
                attn_pdrop=self.config.dropout_prob,
                use_cache=False  # Disable caching for training
            )
            
            # Load model
            if self.config.use_lm_head:
                self.gpt2 = GPT2LMHeadModel.from_pretrained(
                    self.config.model_name,
                    config=gpt2_config
                )
            else:
                self.gpt2 = GPT2Model.from_pretrained(
                    self.config.model_name,
                    config=gpt2_config
                )
            
            # Load tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained(self.config.model_name)
            
            # Add padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Add CLS token if needed
            if self.config.use_cls_token:
                self.tokenizer.add_special_tokens({'cls_token': '[CLS]'})
                self.gpt2.resize_token_embeddings(len(self.tokenizer))
            
            # Enable gradient checkpointing
            if self.config.gradient_checkpointing:
                self.gpt2.gradient_checkpointing_enable()
            
            self.hidden_size = self.gpt2.config.hidden_size
            self.num_layers = self.gpt2.config.n_layer
            
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize GPT-2: {e}")
    
    def _init_classification_components(self):
        """Initialize classification-specific components."""
        # Classification head
        self.classification_head = GPT2ClassificationHead(
            hidden_size=self.hidden_size,
            num_labels=self.config.num_labels,
            dropout=self.config.classifier_dropout,
            use_multi_layer=True
        )
        
        # Prefix tuning if enabled
        if self.config.use_prefix_tuning:
            self.prefix_tuning = PrefixTuning(
                prefix_length=self.config.prefix_length,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers
            )
        
        # Pooling strategy
        if self.config.pooling_strategy != "last":
            self.pooler = PoolingFactory.create_pooler(
                strategy=self.config.pooling_strategy,
                hidden_size=self.hidden_size
            )
    
    def _freeze_embeddings(self):
        """Freeze embedding layers."""
        for param in self.gpt2.wte.parameters():
            param.requires_grad = False
        for param in self.gpt2.wpe.parameters():
            param.requires_grad = False
        logger.info("Froze embedding layers")
    
    def _freeze_encoder_layers(self, n_layers: int):
        """
        Freeze first n transformer layers.
        
        Args:
            n_layers: Number of layers to freeze
        """
        for i in range(n_layers):
            for param in self.gpt2.h[i].parameters():
                param.requires_grad = False
        logger.info(f"Froze first {n_layers} transformer layers")
    
    def _prepare_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare inputs for GPT-2.
        
        Handles padding and attention mask creation for left-to-right models.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            Tuple of (prepared_input_ids, prepared_attention_mask)
        """
        batch_size, seq_len = input_ids.shape
        
        # Add CLS token if configured
        if self.config.use_cls_token:
            cls_token_id = self.tokenizer.cls_token_id
            cls_tokens = torch.full((batch_size, 1), cls_token_id, device=input_ids.device)
            input_ids = torch.cat([cls_tokens, input_ids], dim=1)
            
            if attention_mask is not None:
                cls_mask = torch.ones((batch_size, 1), device=attention_mask.device)
                attention_mask = torch.cat([cls_mask, attention_mask], dim=1)
        
        # Handle padding for GPT-2 (left-padding)
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        return input_ids, attention_mask
    
    def _add_prompt(
        self,
        input_ids: torch.Tensor,
        prompt_template: str = None
    ) -> torch.Tensor:
        """
        Add prompt to input for prompt-based classification.
        
        Args:
            input_ids: Original input IDs
            prompt_template: Prompt template
            
        Returns:
            Input IDs with prompt
        """
        prompt_template = prompt_template or self.config.prompt_template
        batch_size = input_ids.size(0)
        
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(
            prompt_template,
            add_special_tokens=False,
            return_tensors="pt"
        )
        prompt_ids = prompt_ids.repeat(batch_size, 1).to(input_ids.device)
        
        # Concatenate prompt with input
        prompted_ids = torch.cat([input_ids, prompt_ids], dim=1)
        
        # Truncate if necessary
        max_length = self.config.max_length
        if prompted_ids.size(1) > max_length:
            prompted_ids = prompted_ids[:, :max_length]
        
        return prompted_ids
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> ModelOutputs:
        """
        Forward pass through GPT-2 classifier.
        
        GPT-2's autoregressive nature is adapted for classification by:
        1. Processing input with causal attention
        2. Extracting representation from final/pooled positions
        3. Applying classification head
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask
            token_type_ids: Not used in GPT-2
            position_ids: Position IDs
            labels: Target labels for loss computation
            past_key_values: Cached key-value pairs for prefix tuning
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            return_dict: Return ModelOutputs object
            
        Returns:
            Model outputs with logits and optional loss
        """
        # Prepare inputs
        input_ids, attention_mask = self._prepare_inputs(input_ids, attention_mask)
        
        # Add prompt if configured
        if self.config.use_prompt:
            input_ids = self._add_prompt(input_ids)
            # Update attention mask
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        # Get prefix key-values if using prefix tuning
        if self.config.use_prefix_tuning and past_key_values is None:
            batch_size = input_ids.size(0)
            past_key_values = self.prefix_tuning(batch_size)
        
        # Forward through GPT-2
        if self.config.use_lm_head:
            outputs = self.gpt2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[-1] if output_hidden_states else None
            # Get hidden states before LM head
            sequence_output = self.gpt2.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask
            ).last_hidden_state
        else:
            outputs = self.gpt2(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True
            )
            sequence_output = outputs.last_hidden_state
            hidden_states = outputs.hidden_states
        
        # Get sequence representation
        if self.config.pooling_strategy == "last":
            # Get last non-padded token for each sequence
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = input_ids.size(0)
            pooled_output = sequence_output[
                torch.arange(batch_size, device=sequence_output.device),
                sequence_lengths
            ]
        else:
            # Use pooling strategy
            pooled_output = self.pooler(sequence_output, attention_mask)
        
        # Classification
        logits = self.classification_head(pooled_output)
        
        # Apply temperature scaling
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        
        # Compute loss
        loss = None
        if labels is not None:
            if self.config.label_smoothing > 0:
                # Label smoothing
                loss_fct = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return ModelOutputs(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=outputs.attentions if output_attentions else None,
            embeddings=pooled_output,
            metadata={
                "sequence_lengths": attention_mask.sum(dim=1).tolist() if attention_mask is not None else None,
                "uses_prefix": self.config.use_prefix_tuning,
                "uses_prompt": self.config.use_prompt
            }
        )
    
    def generate_with_classification(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate text continuation and classify.
        
        Combines generation and classification capabilities.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            
        Returns:
            Tuple of (generated_ids, classification_logits)
        """
        if not self.config.use_lm_head:
            raise ValueError("Generation requires LM head. Set use_lm_head=True")
        
        # Generate text
        with torch.no_grad():
            generated = self.gpt2.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Classify generated text
        outputs = self.forward(generated)
        
        return generated, outputs.logits
