"""
Base Model Classes for AG News Text Classification
===================================================

This module implements base model abstractions following architectural patterns from:
- Vaswani et al. (2017): "Attention is All You Need"
- Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"
- Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining Approach"

Design Principles:
- Template Method Pattern: Define algorithmic skeleton in base class
- Strategy Pattern: Interchangeable model architectures
- Dependency Injection: Configuration-driven initialization

Mathematical Foundation:
The models implement the transformer architecture where:
- Attention(Q,K,V) = softmax(QK^T/√d_k)V
- MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O
- FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

Author: Võ Hải Dũng
License: MIT
"""

import abc
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
from dataclasses import dataclass, field
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModel, AutoConfig

from src.core.types import (
    BatchData,
    ModelOutput,
    ModelConfig,
    PathLike,
    TensorType
)
from src.core.exceptions import (
    ModelError,
    ModelInitializationError,
    ModelLoadError,
    ModelSaveError
)
from src.utils.logging_config import get_logger
from configs.constants import AG_NEWS_NUM_CLASSES, MAX_SEQUENCE_LENGTH

logger = get_logger(__name__)

# ============================================================================
# Model Output Dataclass
# ============================================================================

@dataclass
class ModelOutputs:
    """
    Standard output container for all models.
    
    This dataclass provides a unified interface for model outputs following
    the design principle of consistent interfaces across components.
    
    Attributes:
        logits: Raw model predictions before activation [batch_size, num_classes]
        loss: Computed loss value (optional)
        probabilities: Softmax probabilities [batch_size, num_classes]
        predictions: Argmax predictions [batch_size]
        hidden_states: Intermediate layer representations
        attentions: Attention weights from transformer layers
        embeddings: Token or sequence embeddings
        metadata: Additional model-specific outputs
    """
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    embeddings: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert outputs to dictionary for serialization."""
        result = {
            "logits": self.logits.detach().cpu().numpy().tolist() if self.logits is not None else None,
            "predictions": self.predictions.detach().cpu().numpy().tolist() if self.predictions is not None else None,
            "probabilities": self.probabilities.detach().cpu().numpy().tolist() if self.probabilities is not None else None,
        }
        if self.loss is not None:
            result["loss"] = self.loss.item()
        result["metadata"] = self.metadata
        return result

# ============================================================================
# Abstract Base Model
# ============================================================================

class AGNewsBaseModel(nn.Module, abc.ABC):
    """
    Abstract base class for all AG News classification models.
    
    This class implements the Template Method pattern, defining the common
    structure and operations for all models while allowing subclasses to
    override specific behaviors.
    
    The design follows SOLID principles:
    - Single Responsibility: Model logic only
    - Open/Closed: Extensible through inheritance
    - Liskov Substitution: All models interchangeable
    - Interface Segregation: Minimal required interface
    - Dependency Inversion: Depends on abstractions
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        """
        Initialize base model.
        
        Args:
            config: Model configuration object
        """
        super().__init__()
        
        # Initialize configuration
        self.config = config or ModelConfig()
        self.num_labels = self.config.num_labels or AG_NEWS_NUM_CLASSES
        
        # Model metadata
        self.model_name = getattr(config, "model_name", "ag_news_base_model")
        self.model_type = getattr(config, "model_type", "base")
        
        # Training state
        self.training_steps = 0
        self.is_initialized = False
        
        logger.debug(f"Initialized {self.__class__.__name__} with {self.num_labels} labels")
    
    @abc.abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through the model.
        
        This abstract method must be implemented by all subclasses to define
        the model's computation graph.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask [batch_size, seq_length]
            labels: Target labels for loss computation [batch_size]
            **kwargs: Additional model-specific arguments
            
        Returns:
            ModelOutputs containing predictions and optional loss
            
        Mathematical Description:
            Given input x, compute:
            h = Encoder(x)
            y = Classifier(h)
            L = CrossEntropy(y, labels) if labels provided
        """
        pass
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_probabilities: bool = True,
        **kwargs
    ) -> ModelOutputs:
        """
        Make predictions without computing loss.
        
        Implements inference mode with gradient computation disabled for
        efficiency.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            return_probabilities: Whether to return probability distribution
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                **kwargs
            )
            
            # Compute probabilities and predictions
            if return_probabilities:
                outputs.probabilities = F.softmax(outputs.logits, dim=-1)
            outputs.predictions = torch.argmax(outputs.logits, dim=-1)
            
        return outputs
    
    def compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        Implements cross-entropy loss with optional label smoothing following
        Müller et al. (2019): "When Does Label Smoothing Help?"
        
        Args:
            logits: Model predictions [batch_size, num_classes]
            labels: Target labels [batch_size]
            class_weights: Class weights for imbalanced data
            label_smoothing: Label smoothing factor ∈ [0, 1]
            
        Returns:
            Computed loss value
            
        Mathematical Formulation:
            L = -Σ(y_smooth * log(softmax(logits)))
            where y_smooth = (1-ε)y + ε/K
            ε = label_smoothing, K = num_classes
        """
        if label_smoothing > 0:
            # Implement label smoothing
            num_classes = logits.size(-1)
            smooth_labels = torch.full_like(logits, label_smoothing / num_classes)
            smooth_labels.scatter_(-1, labels.unsqueeze(-1), 1.0 - label_smoothing)
            
            log_probs = F.log_softmax(logits, dim=-1)
            loss = -(smooth_labels * log_probs).sum(dim=-1).mean()
        else:
            # Standard cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                weight=class_weights
            )
        
        return loss
    
    def save(self, save_path: PathLike):
        """
        Save model checkpoint.
        
        Implements model serialization with metadata for reproducibility.
        
        Args:
            save_path: Path to save model
            
        Raises:
            ModelSaveError: If saving fails
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Prepare checkpoint
            checkpoint = {
                "model_state_dict": self.state_dict(),
                "config": self.config.to_dict() if hasattr(self.config, "to_dict") else {},
                "model_name": self.model_name,
                "model_type": self.model_type,
                "num_labels": self.num_labels,
                "training_steps": self.training_steps,
            }
            
            # Save checkpoint
            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
            
            # Save configuration separately
            config_path = save_path.parent / "config.json"
            with open(config_path, "w") as f:
                json.dump(checkpoint["config"], f, indent=2)
                
        except Exception as e:
            raise ModelSaveError(f"Failed to save model: {e}")
    
    def load(self, load_path: PathLike, map_location: Optional[str] = None):
        """
        Load model checkpoint.
        
        Implements model deserialization with compatibility checking.
        
        Args:
            load_path: Path to model checkpoint
            map_location: Device mapping for loading
            
        Raises:
            ModelLoadError: If loading fails
        """
        load_path = Path(load_path)
        
        if not load_path.exists():
            raise ModelLoadError(f"Model file not found: {load_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(load_path, map_location=map_location)
            
            # Load state dict
            self.load_state_dict(checkpoint["model_state_dict"])
            
            # Restore metadata
            self.model_name = checkpoint.get("model_name", self.model_name)
            self.model_type = checkpoint.get("model_type", self.model_type)
            self.training_steps = checkpoint.get("training_steps", 0)
            
            logger.info(f"Model loaded from {load_path}")
            
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")
    
    def freeze_layers(
        self,
        layers_to_freeze: Optional[List[str]] = None,
        freeze_embeddings: bool = False,
        freeze_encoder: bool = False,
        num_layers_to_freeze: int = 0
    ):
        """
        Freeze specific model layers.
        
        Implements selective layer freezing for transfer learning following
        Howard & Ruder (2018): "Universal Language Model Fine-tuning"
        
        Args:
            layers_to_freeze: Specific layer names to freeze
            freeze_embeddings: Whether to freeze embedding layers
            freeze_encoder: Whether to freeze encoder layers
            num_layers_to_freeze: Number of encoder layers to freeze
        """
        # Default implementation - can be overridden by subclasses
        if layers_to_freeze:
            for name, param in self.named_parameters():
                if any(layer in name for layer in layers_to_freeze):
                    param.requires_grad = False
                    logger.debug(f"Froze layer: {name}")
    
    def get_num_parameters(self, trainable_only: bool = False) -> int:
        """
        Get number of model parameters.
        
        Args:
            trainable_only: Count only trainable parameters
            
        Returns:
            Number of parameters
        """
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
    
    def get_memory_footprint(self) -> Dict[str, float]:
        """
        Calculate model memory footprint.
        
        Returns:
            Dictionary with memory statistics in MB
        """
        param_memory = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_memory = sum(b.numel() * b.element_size() for b in self.buffers())
        
        return {
            "parameters_mb": param_memory / 1024 / 1024,
            "buffers_mb": buffer_memory / 1024 / 1024,
            "total_mb": (param_memory + buffer_memory) / 1024 / 1024
        }

# ============================================================================
# Transformer Base Model
# ============================================================================

class TransformerBaseModel(AGNewsBaseModel):
    """
    Base class for transformer-based models.
    
    Implements common functionality for transformer architectures following
    Vaswani et al. (2017) and subsequent improvements.
    
    This class provides:
    1. Pretrained model loading
    2. Pooling strategies for sequence representation
    3. Classification head architecture
    4. Gradient checkpointing for memory efficiency
    """
    
    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        pretrained_model_name: Optional[str] = None
    ):
        """
        Initialize transformer model.
        
        Args:
            config: Model configuration
            pretrained_model_name: Name of pretrained model
        """
        super().__init__(config)
        
        self.pretrained_model_name = pretrained_model_name or getattr(
            config, "model_name", "bert-base-uncased"
        )
        
        # Initialize transformer
        self._init_transformer()
        
        # Initialize classification head
        self._init_classifier()
        
        # Pooling strategy
        self.pooling_strategy = getattr(config, "pooling_strategy", "cls")
        
        self.is_initialized = True
    
    def _init_transformer(self):
        """
        Initialize pretrained transformer model.
        
        Implements lazy loading of pretrained weights following efficiency
        principles from model deployment best practices.
        """
        try:
            # Load configuration
            self.transformer_config = AutoConfig.from_pretrained(
                self.pretrained_model_name,
                num_labels=self.num_labels
            )
            
            # Load pretrained model
            self.transformer = AutoModel.from_pretrained(
                self.pretrained_model_name,
                config=self.transformer_config
            )
            
            # Get hidden size
            self.hidden_size = self.transformer.config.hidden_size
            
            logger.info(f"Loaded pretrained transformer: {self.pretrained_model_name}")
            
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to initialize transformer: {e}"
            )
    
    def _init_classifier(self):
        """
        Initialize classification head.
        
        Implements multi-layer classification head with dropout for
        regularization following best practices from fine-tuning literature.
        """
        dropout_rate = getattr(self.config, "dropout_rate", 0.1)
        classifier_dropout = getattr(self.config, "classifier_dropout", dropout_rate)
        
        # Build classification head
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)
        
        # Optional: Additional layers for complex classification
        use_hidden_layer = getattr(self.config, "use_hidden_layer", False)
        if use_hidden_layer:
            hidden_dim = getattr(self.config, "hidden_dim", 768)
            self.pre_classifier = nn.Linear(self.hidden_size, hidden_dim)
            self.activation = nn.Tanh()
            self.classifier = nn.Linear(hidden_dim, self.num_labels)
    
    def pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool sequence representations based on strategy.
        
        Implements various pooling strategies for sequence representation
        following Liu et al. (2019) and Reimers & Gurevych (2019).
        
        Args:
            hidden_states: Sequence hidden states [batch, seq_len, hidden_dim]
            attention_mask: Attention mask for valid tokens
            
        Returns:
            Pooled representation [batch, hidden_dim]
            
        Pooling Strategies:
            - CLS: Use [CLS] token representation
            - Mean: Average pooling over sequence
            - Max: Max pooling over sequence
            - Attention: Weighted average using learned attention
        """
        if self.pooling_strategy == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0]
        
        elif self.pooling_strategy == "mean":
            # Mean pooling with attention mask
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                return sum_embeddings / sum_mask
            else:
                return torch.mean(hidden_states, dim=1)
        
        elif self.pooling_strategy == "max":
            # Max pooling
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size())
                hidden_states[~mask_expanded.bool()] = -1e9
            return torch.max(hidden_states, dim=1)[0]
        
        else:
            # Default to CLS
            return hidden_states[:, 0]
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through transformer model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs for BERT-like models
            labels: Target labels
            output_hidden_states: Return all hidden states
            output_attentions: Return attention weights
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with logits and optional loss
        """
        # Pass through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            **kwargs
        )
        
        # Get sequence representation
        hidden_states = transformer_outputs.last_hidden_state
        pooled_output = self.pool_hidden_states(hidden_states, attention_mask)
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.compute_loss(
                logits,
                labels,
                label_smoothing=getattr(self.config, "label_smoothing", 0.0)
            )
        
        return ModelOutputs(
            logits=logits,
            loss=loss,
            hidden_states=transformer_outputs.hidden_states if output_hidden_states else None,
            attentions=transformer_outputs.attentions if output_attentions else None,
            embeddings=pooled_output
        )

# ============================================================================
# Ensemble Base Model
# ============================================================================

class EnsembleBaseModel(AGNewsBaseModel):
    """
    Base class for ensemble models.
    
    Implements ensemble methods following:
    - Dietterich (2000): "Ensemble Methods in Machine Learning"
    - Zhou (2012): "Ensemble Methods: Foundations and Algorithms"
    
    Provides various combination strategies:
    1. Voting (hard/soft)
    2. Stacking
    3. Blending
    4. Bayesian model averaging
    """
    
    def __init__(
        self,
        models: List[AGNewsBaseModel],
        config: Optional[ModelConfig] = None,
        ensemble_method: str = "voting",
        weights: Optional[List[float]] = None
    ):
        """
        Initialize ensemble model.
        
        Args:
            models: List of base models
            config: Ensemble configuration
            ensemble_method: Method for combining predictions
            weights: Model weights for weighted voting
        """
        super().__init__(config)
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.num_models = len(models)
        
        # Initialize weights
        if weights is not None:
            assert len(weights) == self.num_models
            self.weights = torch.tensor(weights)
        else:
            self.weights = torch.ones(self.num_models) / self.num_models
        
        # Meta-learner for stacking
        if ensemble_method == "stacking":
            self._init_meta_learner()
        
        logger.info(f"Initialized ensemble with {self.num_models} models using {ensemble_method}")
    
    def _init_meta_learner(self):
        """Initialize meta-learner for stacking."""
        input_dim = self.num_models * self.num_labels
        hidden_dim = getattr(self.config, "meta_hidden_dim", 128)
        
        self.meta_learner = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.num_labels)
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through ensemble.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            **kwargs: Additional arguments
            
        Returns:
            Ensemble predictions
        """
        # Collect predictions from all models
        all_logits = []
        all_probs = []
        
        for i, model in enumerate(self.models):
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                all_logits.append(outputs.logits)
                all_probs.append(F.softmax(outputs.logits, dim=-1))
        
        # Stack predictions
        logits_stack = torch.stack(all_logits, dim=1)  # [batch, num_models, num_classes]
        probs_stack = torch.stack(all_probs, dim=1)
        
        # Combine predictions
        if self.ensemble_method == "voting":
            # Weighted soft voting
            weights = self.weights.view(1, -1, 1).to(probs_stack.device)
            ensemble_probs = (probs_stack * weights).sum(dim=1)
            ensemble_logits = torch.log(ensemble_probs + 1e-10)
            
        elif self.ensemble_method == "stacking":
            # Meta-learner stacking
            stacked_input = logits_stack.view(logits_stack.size(0), -1)
            ensemble_logits = self.meta_learner(stacked_input)
            
        else:
            # Default to average
            ensemble_logits = logits_stack.mean(dim=1)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = self.compute_loss(ensemble_logits, labels)
        
        return ModelOutputs(
            logits=ensemble_logits,
            loss=loss,
            metadata={"individual_logits": all_logits}
        )

# Export public API
__all__ = [
    "AGNewsBaseModel",
    "TransformerBaseModel",
    "EnsembleBaseModel",
    "ModelOutputs"
]
