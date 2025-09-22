"""
Domain-Adapted RoBERTa for AG News Classification
==================================================

Implementation of domain-adapted RoBERTa through continued pre-training on
news corpora, based on:
- Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
- Gururangan et al. (2020): "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks"
- Lee et al. (2020): "BioBERT: a pre-trained biomedical language representation model"

Domain adaptation improves performance by aligning the model's representations
with the target domain vocabulary and writing style.

Mathematical Foundation:
Domain adaptation loss: L = L_MLM + λ * L_task
where L_MLM is masked language modeling loss on domain data,
and L_task is the downstream task loss.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    RobertaModel,
    RobertaConfig,
    RobertaTokenizer,
    RobertaForMaskedLM
)

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DomainRoBERTaConfig:
    """Configuration for domain-adapted RoBERTa"""
    
    # Model configuration
    model_name: str = "roberta-base"
    num_labels: int = 4
    
    # Domain adaptation
    domain_pretrained_path: Optional[str] = None  # Path to domain-pretrained checkpoint
    use_domain_vocabulary: bool = False  # Use domain-specific vocabulary
    domain_vocab_size: int = 50265  # Size of domain vocabulary
    
    # Architecture modifications
    add_domain_adapter: bool = True  # Add domain-specific adapter layers
    adapter_size: int = 768  # Size of adapter bottleneck
    adapter_dropout: float = 0.1
    
    # Multi-task learning
    use_auxiliary_task: bool = True  # Use auxiliary task during training
    auxiliary_task: str = "topic_modeling"  # "mlm", "topic_modeling", "sentiment"
    auxiliary_weight: float = 0.1
    
    # Training configuration
    freeze_base_model: bool = False
    freeze_embeddings: bool = True
    freeze_lower_layers: int = 6  # Number of lower layers to freeze
    
    # Domain-specific features
    use_domain_embeddings: bool = True  # Add domain-specific embeddings
    domain_embedding_dim: int = 128
    num_domains: int = 5  # Number of domains (World, Sports, Business, Tech, General)
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    
    # Advanced features
    use_contrastive_learning: bool = True  # Contrastive learning between domains
    contrastive_temperature: float = 0.07
    gradient_checkpointing: bool = False


class DomainAdapter(nn.Module):
    """
    Domain-specific adapter module.
    
    Implements bottleneck adapter layers that capture domain-specific
    knowledge while keeping the base model parameters frozen.
    Based on Houlsby et al. (2019): "Parameter-Efficient Transfer Learning for NLP"
    """
    
    def __init__(
        self,
        hidden_size: int,
        adapter_size: int,
        dropout: float = 0.1
    ):
        """
        Initialize domain adapter.
        
        Args:
            hidden_size: Hidden dimension of the model
            adapter_size: Bottleneck dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        # Down-projection
        self.down_project = nn.Linear(hidden_size, adapter_size)
        
        # Non-linearity
        self.activation = nn.ReLU()
        
        # Up-projection
        self.up_project = nn.Linear(adapter_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize adapter weights"""
        # Initialize down-projection with small values
        nn.init.normal_(self.down_project.weight, std=0.01)
        nn.init.zeros_(self.down_project.bias)
        
        # Initialize up-projection to near-zero for residual connection
        nn.init.normal_(self.up_project.weight, std=0.01)
        nn.init.zeros_(self.up_project.bias)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            hidden_states: Input hidden states
            
        Returns:
            Adapted hidden states
        """
        # Store residual
        residual = hidden_states
        
        # Down-project
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Up-project
        hidden_states = self.up_project(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Add residual and normalize
        hidden_states = self.layer_norm(hidden_states + residual)
        
        return hidden_states


class DomainEmbedding(nn.Module):
    """
    Domain-specific embedding layer.
    
    Adds domain information to token embeddings to help the model
    distinguish between different news domains.
    """
    
    def __init__(
        self,
        num_domains: int,
        embedding_dim: int,
        hidden_size: int
    ):
        """
        Initialize domain embedding.
        
        Args:
            num_domains: Number of domains
            embedding_dim: Domain embedding dimension
            hidden_size: Model hidden size
        """
        super().__init__()
        
        # Domain embedding table
        self.domain_embeddings = nn.Embedding(num_domains, embedding_dim)
        
        # Projection to match hidden size
        self.projection = nn.Linear(embedding_dim, hidden_size)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Initialize embeddings
        nn.init.normal_(self.domain_embeddings.weight, std=0.02)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        domain_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add domain embeddings to hidden states.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            domain_ids: Domain IDs [batch_size]
            
        Returns:
            Hidden states with domain information
        """
        if domain_ids is None:
            # Default to general domain (id=4)
            batch_size = hidden_states.shape[0]
            domain_ids = torch.full((batch_size,), 4, device=hidden_states.device)
        
        # Get domain embeddings
        domain_embeds = self.domain_embeddings(domain_ids)  # [batch_size, embedding_dim]
        
        # Project to hidden size
        domain_embeds = self.projection(domain_embeds)  # [batch_size, hidden_size]
        
        # Add to all positions
        domain_embeds = domain_embeds.unsqueeze(1).expand_as(hidden_states)
        
        # Add and normalize
        hidden_states = self.layer_norm(hidden_states + domain_embeds)
        
        return hidden_states


class TopicModelingHead(nn.Module):
    """
    Auxiliary head for topic modeling.
    
    Predicts topic distributions as an auxiliary task to improve
    domain understanding.
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_topics: int = 50,
        dropout: float = 0.1
    ):
        """
        Initialize topic modeling head.
        
        Args:
            hidden_size: Model hidden size
            num_topics: Number of topics
            dropout: Dropout rate
        """
        super().__init__()
        
        self.transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.topic_projection = nn.Linear(hidden_size, num_topics)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict topic distribution.
        
        Args:
            hidden_states: Pooled hidden states
            
        Returns:
            Topic logits
        """
        hidden_states = self.transform(hidden_states)
        topic_logits = self.topic_projection(hidden_states)
        return topic_logits


@MODELS.register("roberta_domain")
class DomainAdaptedRoBERTa(AGNewsBaseModel):
    """
    Domain-adapted RoBERTa for news classification.
    
    Enhances RoBERTa with:
    1. Domain-specific adapters
    2. Domain embeddings
    3. Auxiliary tasks
    4. Contrastive learning between domains
    
    This model is particularly effective when pre-trained on
    large news corpora before fine-tuning on AG News.
    """
    
    def __init__(self, config: Optional[DomainRoBERTaConfig] = None):
        """
        Initialize domain-adapted RoBERTa.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config or DomainRoBERTaConfig()
        
        # Initialize tokenizer
        self.tokenizer = RobertaTokenizer.from_pretrained(self.config.model_name)
        
        # Initialize RoBERTa
        self._init_roberta()
        
        # Add domain-specific components
        if self.config.add_domain_adapter:
            self._add_domain_adapters()
        
        if self.config.use_domain_embeddings:
            self.domain_embedding = DomainEmbedding(
                self.config.num_domains,
                self.config.domain_embedding_dim,
                self.hidden_size
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.hidden_size, self.config.num_labels)
        )
        
        # Auxiliary task head
        if self.config.use_auxiliary_task:
            if self.config.auxiliary_task == "topic_modeling":
                self.auxiliary_head = TopicModelingHead(self.hidden_size)
            elif self.config.auxiliary_task == "mlm":
                self.auxiliary_head = RobertaForMaskedLM.from_pretrained(
                    self.config.model_name
                ).lm_head
        
        # Contrastive learning projection
        if self.config.use_contrastive_learning:
            self.contrastive_projection = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Linear(self.hidden_size, 128)
            )
        
        # Apply freezing
        self._apply_freezing()
        
        logger.info(
            f"Initialized DomainAdaptedRoBERTa with "
            f"adapters={config.add_domain_adapter}, "
            f"auxiliary={config.auxiliary_task}"
        )
    
    def _init_roberta(self):
        """Initialize RoBERTa model"""
        # Load configuration
        roberta_config = RobertaConfig.from_pretrained(self.config.model_name)
        
        # Update configuration
        roberta_config.attention_dropout = self.config.attention_dropout
        roberta_config.hidden_dropout = self.config.hidden_dropout
        
        if self.config.gradient_checkpointing:
            roberta_config.gradient_checkpointing = True
        
        # Load model
        if self.config.domain_pretrained_path:
            # Load from domain-pretrained checkpoint
            self.roberta = RobertaModel.from_pretrained(
                self.config.domain_pretrained_path,
                config=roberta_config
            )
            logger.info(f"Loaded domain-pretrained model from {self.config.domain_pretrained_path}")
        else:
            # Load standard pretrained model
            self.roberta = RobertaModel.from_pretrained(
                self.config.model_name,
                config=roberta_config
            )
        
        self.hidden_size = self.roberta.config.hidden_size
    
    def _add_domain_adapters(self):
        """Add domain adapters to transformer layers"""
        self.adapters = nn.ModuleList()
        
        for i, layer in enumerate(self.roberta.encoder.layer):
            # Add adapter after self-attention
            adapter = DomainAdapter(
                self.hidden_size,
                self.config.adapter_size,
                self.config.adapter_dropout
            )
            self.adapters.append(adapter)
            
            # Modify forward pass to include adapter
            original_forward = layer.forward
            
            def forward_with_adapter(hidden_states, *args, adapter=adapter, original_fn=original_forward, **kwargs):
                outputs = original_fn(hidden_states, *args, **kwargs)
                if isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                    hidden_states = adapter(hidden_states)
                    return (hidden_states,) + outputs[1:]
                else:
                    return adapter(outputs)
            
            layer.forward = forward_with_adapter
        
        logger.info(f"Added {len(self.adapters)} domain adapters")
    
    def _apply_freezing(self):
        """Apply parameter freezing based on configuration"""
        if self.config.freeze_base_model:
            for param in self.roberta.parameters():
                param.requires_grad = False
            logger.info("Froze entire base model")
        else:
            if self.config.freeze_embeddings:
                for param in self.roberta.embeddings.parameters():
                    param.requires_grad = False
                logger.info("Froze embeddings")
            
            if self.config.freeze_lower_layers > 0:
                layers_to_freeze = self.roberta.encoder.layer[:self.config.freeze_lower_layers]
                for layer in layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                logger.info(f"Froze {self.config.freeze_lower_layers} lower layers")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        domain_ids: Optional[torch.Tensor] = None,
        auxiliary_labels: Optional[torch.Tensor] = None,
        return_contrastive: bool = False,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through domain-adapted RoBERTa.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels for classification
            domain_ids: Domain IDs for domain embeddings
            auxiliary_labels: Labels for auxiliary task
            return_contrastive: Whether to return contrastive embeddings
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with predictions
        """
        # RoBERTa forward pass
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # Get pooled output (CLS token)
        sequence_output = outputs.last_hidden_state
        pooled_output = sequence_output[:, 0]  # CLS token
        
        # Add domain embeddings if configured
        if self.config.use_domain_embeddings and hasattr(self, 'domain_embedding'):
            sequence_output = self.domain_embedding(sequence_output, domain_ids)
            pooled_output = sequence_output[:, 0]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Calculate losses
        total_loss = 0
        losses = {}
        
        # Classification loss
        if labels is not None:
            classification_loss = F.cross_entropy(logits, labels)
            total_loss += classification_loss
            losses['classification_loss'] = classification_loss.item()
        
        # Auxiliary task loss
        if self.config.use_auxiliary_task and auxiliary_labels is not None:
            if self.config.auxiliary_task == "topic_modeling":
                aux_logits = self.auxiliary_head(pooled_output)
                aux_loss = F.cross_entropy(aux_logits, auxiliary_labels)
            elif self.config.auxiliary_task == "mlm":
                aux_logits = self.auxiliary_head(sequence_output)
                aux_loss = F.cross_entropy(
                    aux_logits.view(-1, self.config.domain_vocab_size),
                    auxiliary_labels.view(-1)
                )
            else:
                aux_loss = 0
            
            total_loss += self.config.auxiliary_weight * aux_loss
            losses['auxiliary_loss'] = aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0
        
        # Contrastive loss
        if self.config.use_contrastive_learning and labels is not None:
            contrastive_embeds = self.contrastive_projection(pooled_output)
            contrastive_loss = self._contrastive_loss(contrastive_embeds, labels)
            total_loss += 0.1 * contrastive_loss  # Fixed weight for contrastive loss
            losses['contrastive_loss'] = contrastive_loss.item()
        
        # Prepare outputs
        outputs = ModelOutputs(
            loss=total_loss if labels is not None else None,
            logits=logits,
            hidden_states=pooled_output,
            metadata={
                'model_type': 'domain_adapted_roberta',
                'has_adapters': self.config.add_domain_adapter,
                'auxiliary_task': self.config.auxiliary_task,
                **losses
            }
        )
        
        if return_contrastive and self.config.use_contrastive_learning:
            outputs.metadata['contrastive_embeddings'] = contrastive_embeds
        
        return outputs
    
    def _contrastive_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate contrastive loss for domain discrimination.
        
        Args:
            embeddings: Contrastive embeddings
            labels: Class labels (used as proxy for domains)
            
        Returns:
            Contrastive loss
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(embeddings, embeddings.T) / self.config.contrastive_temperature
        
        # Create mask for positive pairs (same class)
        labels_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        labels_mask.fill_diagonal_(False)
        
        # Compute loss
        exp_sim = torch.exp(similarity)
        
        # Sum of similarities with positive examples
        pos_sim = (exp_sim * labels_mask).sum(dim=1)
        
        # Sum of all similarities (excluding self)
        all_sim = exp_sim.sum(dim=1) - exp_sim.diag()
        
        # Contrastive loss
        loss = -torch.log(pos_sim / all_sim).mean()
        
        return loss
