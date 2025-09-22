"""
ELECTRA Discriminator for AG News Classification
=================================================

Implementation of ELECTRA (Efficiently Learning an Encoder that Classifies Token
Replacements Accurately) for text classification, based on:
- Clark et al. (2020): "ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators"

Key innovations:
1. Replaced token detection instead of masked language modeling
2. More sample-efficient pre-training
3. All tokens contribute to learning (not just masked ones)
4. Better performance with smaller compute budget

Mathematical Foundation:
ELECTRA uses a discriminator D that predicts whether each token is original or replaced:
L_disc = E[Σ_t -log D(x_t, t)]
where x_t is the token at position t.

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
    ElectraModel,
    ElectraConfig,
    ElectraTokenizer,
    ElectraForSequenceClassification
)

from src.models.base.base_model import TransformerBaseModel, ModelOutputs
from src.models.base.pooling_strategies import PoolingFactory
from src.core.registry import MODELS
from src.core.exceptions import ModelInitializationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ElectraDiscriminatorConfig:
    """Configuration for ELECTRA discriminator model."""
    model_name: str = "google/electra-large-discriminator"
    num_labels: int = 4
    max_length: int = 512
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    classifier_dropout: float = 0.15
    pooling_strategy: str = "cls"
    use_discriminator_head: bool = True
    discriminator_weight: float = 0.5
    use_gradient_reversal: bool = False
    reversal_lambda: float = 0.1
    freeze_embeddings: bool = False
    freeze_n_layers: int = 0
    gradient_checkpointing: bool = False
    use_differential_lr: bool = True
    layer_wise_lr_decay: float = 0.9
    reinit_n_layers: int = 0
    use_mixed_precision: bool = True


class GradientReversalLayer(nn.Module):
    """
    Gradient reversal layer for adversarial training.
    
    Based on Ganin et al. (2016): "Domain-Adversarial Training of Neural Networks"
    """
    
    def __init__(self, lambda_param: float = 1.0):
        super().__init__()
        self.lambda_param = lambda_param
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass (identity), backward pass (gradient reversal)."""
        return GradientReversalFunction.apply(x, self.lambda_param)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal function implementation."""
    
    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_param, None


class DiscriminatorHead(nn.Module):
    """
    Discriminator head for ELECTRA-style training.
    
    Predicts whether tokens are original or replaced,
    which can provide additional training signal.
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, 1)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict token authenticity.
        
        Args:
            hidden_states: Token representations [batch_size, seq_len, hidden_size]
            
        Returns:
            Discrimination scores [batch_size, seq_len]
        """
        x = self.dropout(hidden_states)
        x = self.dense(x)
        x = F.gelu(x)
        x = self.dropout(x)
        logits = self.out_proj(x).squeeze(-1)
        return logits


@MODELS.register("electra", aliases=["electra-large", "electra-discriminator"])
class ElectraDiscriminator(TransformerBaseModel):
    """
    ELECTRA discriminator model for AG News classification.
    
    ELECTRA's discriminative pre-training provides several advantages:
    1. More efficient learning from all input tokens
    2. Better sample efficiency than masked language modeling
    3. Strong performance with smaller model sizes
    4. Natural fit for classification tasks
    
    The model can optionally use the discriminator head as auxiliary task
    for improved regularization during fine-tuning.
    """
    
    __auto_register__ = True
    __model_name__ = "electra"
    
    def __init__(self, config: Optional[ElectraDiscriminatorConfig] = None):
        """
        Initialize ELECTRA discriminator.
        
        Args:
            config: Model configuration
        """
        self.config = config or ElectraDiscriminatorConfig()
        
        # Initialize base transformer
        super().__init__(pretrained_model_name=self.config.model_name)
        
        # Initialize ELECTRA components
        self._init_electra_model()
        self._init_classification_head()
        self._init_discriminator_components()
        self._init_pooling()
        
        # Apply initialization strategies
        if self.config.reinit_n_layers > 0:
            self._reinitialize_layers(self.config.reinit_n_layers)
        
        if self.config.freeze_embeddings:
            self._freeze_embeddings()
            
        if self.config.freeze_n_layers > 0:
            self._freeze_encoder_layers(self.config.freeze_n_layers)
        
        logger.info(
            f"Initialized ELECTRA discriminator with {self.config.num_labels} labels"
        )
    
    def _init_electra_model(self):
        """Initialize ELECTRA model and configuration."""
        try:
            # Load configuration
            electra_config = ElectraConfig.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_labels,
                hidden_dropout_prob=self.config.hidden_dropout_prob,
                attention_probs_dropout_prob=self.config.attention_dropout_prob
            )
            
            # Load pretrained model
            self.electra = ElectraModel.from_pretrained(
                self.config.model_name,
                config=electra_config
            )
            
            # Enable gradient checkpointing if specified
            if self.config.gradient_checkpointing:
                self.electra.gradient_checkpointing_enable()
            
            self.hidden_size = self.electra.config.hidden_size
            
        except Exception as e:
            raise ModelInitializationError(f"Failed to initialize ELECTRA: {e}")
    
    def _init_classification_head(self):
        """Initialize classification head with improved architecture."""
        # Multi-layer classification head
        self.classifier_layers = nn.ModuleList([
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.LayerNorm(self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(self.config.classifier_dropout)
        ])
        
        # Final classifier
        self.classifier = nn.Linear(self.hidden_size // 2, self.config.num_labels)
        
        # Initialize weights
        for layer in self.classifier_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def _init_discriminator_components(self):
        """Initialize discriminator-specific components."""
        if self.config.use_discriminator_head:
            # Discriminator head for auxiliary task
            self.discriminator_head = DiscriminatorHead(
                self.hidden_size,
                self.config.hidden_dropout_prob
            )
            
            # Gradient reversal for adversarial training
            if self.config.use_gradient_reversal:
                self.gradient_reversal = GradientReversalLayer(
                    self.config.reversal_lambda
                )
            else:
                self.gradient_reversal = None
    
    def _init_pooling(self):
        """Initialize pooling strategy."""
        self.pooler = PoolingFactory.create_pooler(
            strategy=self.config.pooling_strategy,
            hidden_size=self.hidden_size,
            dropout_rate=self.config.hidden_dropout_prob
        )
    
    def _freeze_embeddings(self):
        """Freeze embedding layers."""
        for param in self.electra.embeddings.parameters():
            param.requires_grad = False
        logger.info("Froze embedding layers")
    
    def _freeze_encoder_layers(self, n_layers: int):
        """
        Freeze first n encoder layers.
        
        Args:
            n_layers: Number of layers to freeze
        """
        for i in range(n_layers):
            for param in self.electra.encoder.layer[i].parameters():
                param.requires_grad = False
        logger.info(f"Froze first {n_layers} encoder layers")
    
    def _reinitialize_layers(self, n_layers: int):
        """
        Re-initialize top n layers for better task adaptation.
        
        Based on findings from Mosbach et al. (2021): "On the Stability of Fine-tuning"
        """
        encoder_layers = self.electra.encoder.layer
        num_layers = len(encoder_layers)
        
        for i in range(num_layers - n_layers, num_layers):
            layer = encoder_layers[i]
            
            # Re-initialize self-attention
            for module in [layer.attention.self.query,
                          layer.attention.self.key,
                          layer.attention.self.value,
                          layer.attention.output.dense]:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            
            # Re-initialize feed-forward
            for module in [layer.intermediate.dense,
                          layer.output.dense]:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        
        logger.info(f"Re-initialized top {n_layers} encoder layers")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        replaced_token_labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ) -> ModelOutputs:
        """
        Forward pass through ELECTRA discriminator.
        
        The forward pass includes:
        1. ELECTRA encoding with efficient attention
        2. Optional discriminator head for auxiliary task
        3. Classification with multi-layer head
        4. Optional adversarial gradient reversal
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask
            token_type_ids: Segment token indices
            position_ids: Position IDs
            labels: Target labels for classification
            replaced_token_labels: Labels for discriminator task
            output_attentions: Return attention weights
            output_hidden_states: Return all hidden states
            return_dict: Return ModelOutputs object
            
        Returns:
            Model outputs with logits and optional auxiliary losses
        """
        # Forward through ELECTRA encoder
        outputs = self.electra(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Discriminator auxiliary task
        discriminator_loss = None
        if self.config.use_discriminator_head and replaced_token_labels is not None:
            # Apply gradient reversal if configured
            if self.gradient_reversal:
                disc_hidden = self.gradient_reversal(sequence_output)
            else:
                disc_hidden = sequence_output
            
            # Discriminator predictions
            disc_logits = self.discriminator_head(disc_hidden)
            
            # Compute discriminator loss
            disc_loss_fct = nn.BCEWithLogitsLoss()
            discriminator_loss = disc_loss_fct(
                disc_logits.view(-1),
                replaced_token_labels.float().view(-1)
            )
        
        # Pool sequence representation
        pooled_output = self.pooler(sequence_output, attention_mask)
        
        # Pass through classification layers
        for layer in self.classifier_layers:
            pooled_output = layer(pooled_output)
        
        # Final classification
        logits = self.classifier(pooled_output)
        
        # Compute classification loss
        classification_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(
                logits.view(-1, self.config.num_labels),
                labels.view(-1)
            )
        
        # Combine losses
        total_loss = classification_loss
        if discriminator_loss is not None and self.config.use_discriminator_head:
            total_loss = (
                classification_loss + 
                self.config.discriminator_weight * discriminator_loss
            )
        
        return ModelOutputs(
            loss=total_loss,
            logits=logits,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions if output_attentions else None,
            embeddings=pooled_output,
            metadata={
                "classification_loss": classification_loss.item() if classification_loss else None,
                "discriminator_loss": discriminator_loss.item() if discriminator_loss else None
            }
        )
    
    def get_layer_parameters(self):
        """
        Get parameters grouped by layers for differential learning rates.
        
        Returns:
            List of parameter groups with decay factors
        """
        if not self.config.use_differential_lr:
            return self.parameters()
        
        parameter_groups = []
        
        # Embeddings - lowest learning rate
        parameter_groups.append({
            'params': self.electra.embeddings.parameters(),
            'lr_scale': self.config.layer_wise_lr_decay ** (len(self.electra.encoder.layer) + 1)
        })
        
        # Encoder layers - increasing learning rate
        for i, layer in enumerate(self.electra.encoder.layer):
            lr_scale = self.config.layer_wise_lr_decay ** (len(self.electra.encoder.layer) - i)
            parameter_groups.append({
                'params': layer.parameters(),
                'lr_scale': lr_scale
            })
        
        # Classification head - highest learning rate
        classifier_params = []
        for layer in self.classifier_layers:
            classifier_params.extend(layer.parameters())
        classifier_params.extend(self.classifier.parameters())
        
        parameter_groups.append({
            'params': classifier_params,
            'lr_scale': 1.0
        })
        
        # Discriminator head if used
        if self.config.use_discriminator_head:
            parameter_groups.append({
                'params': self.discriminator_head.parameters(),
                'lr_scale': 0.5  # Lower LR for auxiliary task
            })
        
        return parameter_groups
