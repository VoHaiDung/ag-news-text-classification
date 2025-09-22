"""
Soft Prompt Tuning for AG News Classification
==================================================

Implementation of soft prompt tuning (continuous prompts) for efficient adaptation,
based on:
- Lester et al. (2021): "The Power of Scale for Parameter-Efficient Prompt Tuning"
- Liu et al. (2021): "P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning"
- Li & Liang (2021): "Prefix-Tuning: Optimizing Continuous Prompts for Generation"

Soft prompts are learnable continuous embeddings prepended to the input,
allowing task adaptation without modifying model parameters.

Mathematical Foundation:
Instead of discrete tokens, we learn continuous vectors P ∈ R^(k×d):
h = f(concat([P; embed(x)]))
where P are learnable prompt embeddings and f is the frozen LM.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np

from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig
)

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PromptInitializer:
    """
    Initialize soft prompts with various strategies
    
    Different initialization strategies can significantly impact
    convergence and final performance.
    """
    
    @staticmethod
    def random_uniform(shape: Tuple[int, ...], scale: float = 0.5) -> torch.Tensor:
        """Uniform random initialization"""
        return torch.FloatTensor(*shape).uniform_(-scale, scale)
    
    @staticmethod
    def random_normal(shape: Tuple[int, ...], std: float = 0.02) -> torch.Tensor:
        """Normal random initialization"""
        return torch.randn(*shape) * std
    
    @staticmethod
    def from_vocab(
        shape: Tuple[int, ...],
        embeddings: torch.Tensor,
        vocab_size: int
    ) -> torch.Tensor:
        """Initialize from random vocabulary embeddings"""
        prompt_length = shape[0]
        embedding_dim = shape[1]
        
        # Sample random token indices
        random_indices = torch.randint(0, vocab_size, (prompt_length,))
        
        # Get embeddings for these tokens
        prompt_embeds = embeddings[random_indices].detach().clone()
        
        return prompt_embeds
    
    @staticmethod
    def from_words(
        words: List[str],
        tokenizer: Any,
        embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Initialize from specific words"""
        # Tokenize words
        token_ids = []
        for word in words:
            tokens = tokenizer.encode(word, add_special_tokens=False)
            if tokens:
                token_ids.append(tokens[0])  # Take first token
        
        if not token_ids:
            logger.warning("No valid tokens from words, using random init")
            return PromptInitializer.random_normal((len(words), embeddings.shape[1]))
        
        # Get embeddings
        token_ids = torch.tensor(token_ids)
        prompt_embeds = embeddings[token_ids].detach().clone()
        
        return prompt_embeds
    
    @staticmethod
    def class_aware(
        num_classes: int,
        prompt_length: int,
        embedding_dim: int,
        class_names: Optional[List[str]] = None
    ) -> torch.Tensor:
        """Initialize with class-aware structure"""
        # Allocate tokens per class
        tokens_per_class = prompt_length // num_classes
        remaining = prompt_length % num_classes
        
        prompt_embeds = []
        
        for i in range(num_classes):
            # Number of tokens for this class
            n_tokens = tokens_per_class + (1 if i < remaining else 0)
            
            # Initialize with slight variation for each class
            class_embed = torch.randn(n_tokens, embedding_dim) * 0.02
            class_embed += torch.randn(1, embedding_dim) * 0.1  # Class-specific bias
            
            prompt_embeds.append(class_embed)
        
        return torch.cat(prompt_embeds, dim=0)


@dataclass
class SoftPromptConfig:
    """Configuration for soft prompt tuning"""
    
    # Model configuration
    model_name: str = "bert-base-uncased"
    freeze_model: bool = True  # Freeze base model parameters
    
    # Prompt configuration
    prompt_length: int = 20  # Number of soft prompt tokens
    prompt_depth: int = 1  # Number of layers to insert prompts
    
    # Initialization
    init_strategy: str = "random_normal"  # Initialization strategy
    init_words: Optional[List[str]] = None  # Words for initialization
    init_scale: float = 0.02  # Scale for random initialization
    
    # Reparameterization
    reparameterize: bool = True  # Use reparameterization trick
    bottleneck_dim: Optional[int] = None  # Bottleneck dimension for reparam
    
    # Training
    prompt_lr: float = 3e-2  # Learning rate for prompts
    prompt_weight_decay: float = 0.0
    
    # Advanced features
    deep_prompt_tuning: bool = False  # Insert prompts at multiple layers
    prompt_dropout: float = 0.0
    learned_prompt_pooling: bool = False  # Learn pooling over prompts
    
    # Task-specific
    num_classes: int = 4
    class_specific_prompts: bool = False  # Different prompts per class
    
    # Regularization
    prompt_regularization: float = 0.01
    orthogonal_reg: bool = False  # Orthogonality constraint
    
    # Optimization
    gradient_accumulation: int = 1
    mixed_precision: bool = False


class PromptEncoder(nn.Module):
    """
    Encoder for soft prompts with optional reparameterization
    
    Reparameterization through an MLP can improve optimization
    and reduce the number of parameters.
    """
    
    def __init__(
        self,
        prompt_length: int,
        embedding_dim: int,
        bottleneck_dim: Optional[int] = None,
        dropout: float = 0.0
    ):
        """
        Initialize prompt encoder.
        
        Args:
            prompt_length: Number of prompt tokens
            embedding_dim: Dimension of embeddings
            bottleneck_dim: Bottleneck dimension for reparameterization
            dropout: Dropout rate
        """
        super().__init__()
        
        self.prompt_length = prompt_length
        self.embedding_dim = embedding_dim
        
        if bottleneck_dim is not None:
            # Reparameterization through bottleneck
            self.prompt_embeddings = Parameter(
                torch.randn(prompt_length, bottleneck_dim) * 0.02
            )
            
            self.mlp = nn.Sequential(
                nn.Linear(bottleneck_dim, bottleneck_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim * 2, embedding_dim)
            )
            
            logger.info(f"Using reparameterization: {bottleneck_dim} -> {embedding_dim}")
        else:
            # Direct soft prompts
            self.prompt_embeddings = Parameter(
                torch.randn(prompt_length, embedding_dim) * 0.02
            )
            self.mlp = None
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, batch_size: int = 1) -> torch.Tensor:
        """
        Generate prompt embeddings.
        
        Args:
            batch_size: Batch size for expansion
            
        Returns:
            Prompt embeddings [batch_size, prompt_length, embedding_dim]
        """
        if self.mlp is not None:
            # Apply reparameterization
            prompts = self.mlp(self.prompt_embeddings)
        else:
            prompts = self.prompt_embeddings
        
        # Apply dropout
        prompts = self.dropout(prompts)
        
        # Expand to batch size
        prompts = prompts.unsqueeze(0).expand(batch_size, -1, -1)
        
        return prompts


class DeepPromptEncoder(nn.Module):
    """
    Deep prompt encoder for multi-layer prompt insertion
    
    Inserts different prompts at multiple transformer layers for
    more expressive prompt tuning.
    """
    
    def __init__(
        self,
        num_layers: int,
        prompt_length: int,
        embedding_dim: int,
        share_prompts: bool = False
    ):
        """
        Initialize deep prompt encoder.
        
        Args:
            num_layers: Number of layers to insert prompts
            prompt_length: Length of prompts at each layer
            embedding_dim: Embedding dimension
            share_prompts: Share prompts across layers
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.prompt_length = prompt_length
        self.share_prompts = share_prompts
        
        if share_prompts:
            # Single set of prompts shared across layers
            self.prompt_encoder = PromptEncoder(prompt_length, embedding_dim)
        else:
            # Different prompts for each layer
            self.prompt_encoders = nn.ModuleList([
                PromptEncoder(prompt_length, embedding_dim)
                for _ in range(num_layers)
            ])
        
        logger.info(f"Deep prompts for {num_layers} layers (shared={share_prompts})")
    
    def forward(self, batch_size: int = 1, layer_id: Optional[int] = None) -> torch.Tensor:
        """
        Get prompts for specific layer or all layers.
        
        Args:
            batch_size: Batch size
            layer_id: Specific layer ID (None for all)
            
        Returns:
            Prompt embeddings
        """
        if layer_id is not None:
            # Get prompts for specific layer
            if self.share_prompts:
                return self.prompt_encoder(batch_size)
            else:
                return self.prompt_encoders[layer_id](batch_size)
        else:
            # Get all prompts
            all_prompts = []
            for i in range(self.num_layers):
                if self.share_prompts:
                    prompts = self.prompt_encoder(batch_size)
                else:
                    prompts = self.prompt_encoders[i](batch_size)
                all_prompts.append(prompts)
            
            return torch.stack(all_prompts)  # [num_layers, batch, length, dim]


@MODELS.register("soft_prompt")
class SoftPromptModel(AGNewsBaseModel):
    """
    Soft Prompt Tuning Model
    
    Implements parameter-efficient fine-tuning through continuous prompts.
    Only the prompt embeddings are trained while the base model remains frozen.
    
    This approach achieves competitive performance with full fine-tuning
    while requiring orders of magnitude fewer trainable parameters.
    """
    
    def __init__(self, config: Optional[SoftPromptConfig] = None):
        """
        Initialize soft prompt model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config or SoftPromptConfig()
        
        # Initialize base model and tokenizer
        self._init_base_model()
        
        # Initialize prompt components
        self._init_prompts()
        
        # Classification head
        self.classifier = nn.Linear(
            self.base_model.config.hidden_size,
            self.config.num_classes
        )
        
        # Freeze base model if configured
        if self.config.freeze_model:
            self._freeze_base_model()
        
        # Log parameter statistics
        self._log_parameters()
        
        logger.info(f"Initialized Soft Prompt Model with {self.config.prompt_length} tokens")
    
    def _init_base_model(self):
        """Initialize base language model and tokenizer"""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        self.base_model = AutoModel.from_pretrained(self.config.model_name)
        
        # Get embeddings
        self.embeddings = self.base_model.get_input_embeddings()
        self.embedding_dim = self.embeddings.embedding_dim
    
    def _init_prompts(self):
        """Initialize soft prompt components"""
        if self.config.deep_prompt_tuning:
            # Deep prompt tuning
            num_layers = self.base_model.config.num_hidden_layers
            self.prompt_encoder = DeepPromptEncoder(
                num_layers=num_layers,
                prompt_length=self.config.prompt_length,
                embedding_dim=self.embedding_dim,
                share_prompts=False
            )
        else:
            # Standard prompt tuning
            self.prompt_encoder = PromptEncoder(
                prompt_length=self.config.prompt_length,
                embedding_dim=self.embedding_dim,
                bottleneck_dim=self.config.bottleneck_dim,
                dropout=self.config.prompt_dropout
            )
        
        # Initialize prompts based on strategy
        self._initialize_prompts()
        
        # Optional: Learned prompt pooling
        if self.config.learned_prompt_pooling:
            self.prompt_pooler = nn.Sequential(
                nn.Linear(self.embedding_dim, self.embedding_dim),
                nn.Tanh()
            )
    
    def _initialize_prompts(self):
        """Initialize prompt embeddings with specified strategy"""
        initializer = PromptInitializer()
        
        if self.config.init_strategy == "random_uniform":
            init_embeds = initializer.random_uniform(
                (self.config.prompt_length, self.embedding_dim),
                scale=self.config.init_scale
            )
        elif self.config.init_strategy == "random_normal":
            init_embeds = initializer.random_normal(
                (self.config.prompt_length, self.embedding_dim),
                std=self.config.init_scale
            )
        elif self.config.init_strategy == "from_vocab":
            init_embeds = initializer.from_vocab(
                (self.config.prompt_length, self.embedding_dim),
                self.embeddings.weight,
                self.tokenizer.vocab_size
            )
        elif self.config.init_strategy == "from_words" and self.config.init_words:
            init_embeds = initializer.from_words(
                self.config.init_words,
                self.tokenizer,
                self.embeddings.weight
            )
        elif self.config.init_strategy == "class_aware":
            init_embeds = initializer.class_aware(
                self.config.num_classes,
                self.config.prompt_length,
                self.embedding_dim
            )
        else:
            init_embeds = initializer.random_normal(
                (self.config.prompt_length, self.embedding_dim)
            )
        
        # Set initial values
        if hasattr(self.prompt_encoder, 'prompt_embeddings'):
            with torch.no_grad():
                if self.prompt_encoder.mlp is not None:
                    # For reparameterized prompts, we need to adjust initialization
                    self.prompt_encoder.prompt_embeddings.data = init_embeds[:, :self.config.bottleneck_dim]
                else:
                    self.prompt_encoder.prompt_embeddings.data = init_embeds
    
    def _freeze_base_model(self):
        """Freeze base model parameters"""
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        logger.info("Froze base model parameters")
    
    def _log_parameters(self):
        """Log parameter statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        logger.info(f"Percentage trainable: {trainable_params/total_params*100:.2f}%")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass with soft prompts.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_length]
            attention_mask: Attention mask
            labels: Target labels
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with classification results
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Get input embeddings
        input_embeds = self.embeddings(input_ids)
        
        # Get soft prompt embeddings
        prompt_embeds = self.prompt_encoder(batch_size).to(device)
        
        # Concatenate prompts with input
        # [PROMPT TOKENS] [ORIGINAL INPUT]
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # Update attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size,
                self.config.prompt_length,
                dtype=attention_mask.dtype,
                device=device
            )
            combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        # Forward through base model
        outputs = self.base_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            **kwargs
        )
        
        # Get pooled output
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            # Mean pooling over sequence
            hidden_states = outputs.last_hidden_state
            
            if self.config.learned_prompt_pooling:
                # Pool only over prompt positions
                prompt_hidden = hidden_states[:, :self.config.prompt_length, :]
                pooled = self.prompt_pooler(prompt_hidden.mean(dim=1))
            else:
                # Standard pooling
                if combined_mask is not None:
                    mask_expanded = combined_mask.unsqueeze(-1).expand(hidden_states.size())
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                else:
                    pooled = hidden_states.mean(dim=1)
        
        # Classification
        logits = self.classifier(pooled)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            
            # Add regularization
            if self.config.prompt_regularization > 0:
                prompt_reg = self.config.prompt_regularization * torch.norm(prompt_embeds)
                loss = loss + prompt_reg
            
            if self.config.orthogonal_reg:
                # Orthogonality regularization
                prompt_flat = prompt_embeds.view(batch_size, -1)
                correlation = torch.matmul(prompt_flat, prompt_flat.t())
                I = torch.eye(batch_size, device=device)
                ortho_loss = torch.norm(correlation - I)
                loss = loss + 0.01 * ortho_loss
        
        return ModelOutputs(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            metadata={
                'prompt_length': self.config.prompt_length,
                'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad),
                'prompt_norm': torch.norm(prompt_embeds).item()
            }
        )
    
    def get_prompt_embeddings(self) -> torch.Tensor:
        """
        Get current prompt embeddings.
        
        Returns:
            Prompt embeddings tensor
        """
        with torch.no_grad():
            return self.prompt_encoder(1).squeeze(0)
    
    def save_prompts(self, path: Union[str, Path]):
        """
        Save only the prompt parameters.
        
        Args:
            path: Path to save prompts
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save prompt encoder state
        prompt_state = {
            'prompt_encoder': self.prompt_encoder.state_dict(),
            'classifier': self.classifier.state_dict(),
            'config': self.config
        }
        
        if hasattr(self, 'prompt_pooler'):
            prompt_state['prompt_pooler'] = self.prompt_pooler.state_dict()
        
        torch.save(prompt_state, path)
        logger.info(f"Saved prompts to {path}")
    
    def load_prompts(self, path: Union[str, Path]):
        """
        Load prompt parameters.
        
        Args:
            path: Path to load prompts from
        """
        path = Path(path)
        prompt_state = torch.load(path, map_location='cpu')
        
        self.prompt_encoder.load_state_dict(prompt_state['prompt_encoder'])
        self.classifier.load_state_dict(prompt_state['classifier'])
        
        if 'prompt_pooler' in prompt_state and hasattr(self, 'prompt_pooler'):
            self.prompt_pooler.load_state_dict(prompt_state['prompt_pooler'])
        
        logger.info(f"Loaded prompts from {path}")
