"""
Prompt-based Model Implementation
==================================

Implementation of prompt-based learning for few-shot text classification,
based on:
- Brown et al. (2020): "Language Models are Few-Shot Learners"
- Schick & Schütze (2021): "Exploiting Cloze Questions for Few Shot Text Classification"
- Liu et al. (2021): "Pre-train, Prompt, and Predict"
- Gao et al. (2021): "Making Pre-trained Language Models Better Few-shot Learners"

Mathematical Foundation:
Prompt-based learning reformulates classification as:
P(y|x) = P(v|[CLS] x [SEP] T(x) [MASK])
where T(x) is the prompt template and v is the verbalizer mapping.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AutoConfig
)

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PromptType(Enum):
    """Types of prompts supported."""
    MANUAL = "manual"  # Hand-crafted prompts
    SOFT = "soft"  # Learnable continuous prompts
    MIXED = "mixed"  # Combination of manual and soft
    PREFIX = "prefix"  # Prefix tuning
    PTUNING = "p-tuning"  # P-tuning


@dataclass
class PromptConfig:
    """Configuration for prompt-based models."""
    model_name: str = "roberta-large"
    prompt_type: PromptType = PromptType.MANUAL
    num_labels: int = 4
    max_length: int = 256
    
    # Prompt template
    template: str = "This news is about [MASK]."
    mask_token: str = "[MASK]"
    
    # Verbalizer mapping (label -> tokens)
    verbalizer: Dict[int, List[str]] = None
    
    # Soft prompt settings
    num_prompt_tokens: int = 10
    prompt_init: str = "random"  # "random", "vocab", "text"
    prompt_encoder_hidden_size: int = 512
    
    # Training settings
    use_calibration: bool = True
    use_ensemble_prompt: bool = False
    num_prompt_ensemble: int = 3
    dropout: float = 0.1
    
    # Few-shot settings
    num_demonstrations: int = 4
    demonstration_sampling: str = "random"  # "random", "similarity", "diverse"


class Verbalizer(nn.Module):
    """
    Verbalizer for mapping between labels and tokens.
    
    Handles the mapping between class labels and vocabulary tokens
    for masked language modeling based classification.
    """
    
    def __init__(
        self,
        tokenizer,
        label_words: Dict[int, List[str]],
        post_log_softmax: bool = True
    ):
        """
        Initialize verbalizer.
        
        Args:
            tokenizer: Tokenizer for converting words to IDs
            label_words: Mapping from labels to words
            post_log_softmax: Apply log-softmax to scores
        """
        super().__init__()
        
        self.tokenizer = tokenizer
        self.label_words = label_words
        self.post_log_softmax = post_log_softmax
        
        # Create label token ID mapping
        self.label_token_ids = {}
        for label, words in label_words.items():
            token_ids = []
            for word in words:
                # Handle multi-token words
                ids = tokenizer.encode(word, add_special_tokens=False)
                token_ids.extend(ids)
            self.label_token_ids[label] = token_ids
        
        # Create projection matrix
        self._create_projection_matrix()
    
    def _create_projection_matrix(self):
        """Create projection matrix from vocabulary to labels."""
        vocab_size = len(self.tokenizer)
        num_labels = len(self.label_words)
        
        # Initialize projection matrix
        projection = torch.zeros(num_labels, vocab_size)
        
        for label, token_ids in self.label_token_ids.items():
            for token_id in token_ids:
                projection[label, token_id] = 1.0 / len(token_ids)
        
        self.register_buffer('projection_matrix', projection)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Project vocabulary logits to label scores.
        
        Args:
            logits: Logits over vocabulary [batch_size, vocab_size]
            
        Returns:
            Label scores [batch_size, num_labels]
        """
        if self.post_log_softmax:
            logits = F.log_softmax(logits, dim=-1)
        
        # Project to label space
        label_scores = torch.matmul(logits, self.projection_matrix.t())
        
        return label_scores


class SoftPromptEncoder(nn.Module):
    """
    Soft prompt encoder for continuous prompts.
    
    Learns continuous prompt embeddings that are prepended to input.
    """
    
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        hidden_size: int = 512,
        init_method: str = "random"
    ):
        """
        Initialize soft prompt encoder.
        
        Args:
            num_tokens: Number of soft prompt tokens
            token_dim: Dimension of token embeddings
            hidden_size: Hidden size for prompt encoder
            init_method: Initialization method
        """
        super().__init__()
        
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        
        # Soft prompt embeddings
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_tokens, token_dim)
        )
        
        # Optional prompt encoder
        self.prompt_encoder = nn.Sequential(
            nn.Linear(token_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, token_dim),
            nn.LayerNorm(token_dim)
        )
        
        # Initialize
        self._init_prompts(init_method)
    
    def _init_prompts(self, method: str):
        """Initialize soft prompts."""
        if method == "random":
            nn.init.normal_(self.prompt_embeddings, mean=0, std=0.02)
        elif method == "uniform":
            nn.init.uniform_(self.prompt_embeddings, -0.5, 0.5)
        else:
            nn.init.xavier_uniform_(self.prompt_embeddings)
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Get soft prompt embeddings.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Prompt embeddings [batch_size, num_tokens, token_dim]
        """
        # Expand for batch
        prompts = self.prompt_embeddings.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        
        # Encode prompts
        prompts = self.prompt_encoder(prompts)
        
        return prompts


@MODELS.register("prompt", aliases=["prompt_based", "few_shot"])
class PromptModel(AGNewsBaseModel):
    """
    Prompt-based model for few-shot text classification.
    
    Reformulates classification as masked language modeling:
    1. Wraps input in prompt template
    2. Uses MLM to predict masked tokens
    3. Maps predictions to labels via verbalizer
    
    Supports various prompting strategies:
    - Manual prompts with hand-crafted templates
    - Soft prompts with learnable embeddings
    - Mixed prompts combining both
    - Ensemble of prompts for robustness
    """
    
    def __init__(self, config: Optional[PromptConfig] = None):
        """
        Initialize prompt-based model.
        
        Args:
            config: Prompt configuration
        """
        super().__init__()
        
        self.config = config or PromptConfig()
        
        # Initialize tokenizer and model
        self._init_backbone()
        
        # Initialize verbalizer
        self._init_verbalizer()
        
        # Initialize soft prompts if needed
        if self.config.prompt_type in [PromptType.SOFT, PromptType.MIXED]:
            self._init_soft_prompts()
        
        # Calibration parameters
        if self.config.use_calibration:
            self.calibration_bias = nn.Parameter(torch.zeros(self.config.num_labels))
        
        logger.info(
            f"Initialized PromptModel with {self.config.prompt_type.value} prompts"
        )
    
    def _init_backbone(self):
        """Initialize backbone MLM model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Load MLM model
        self.mlm_model = AutoModelForMaskedLM.from_pretrained(
            self.config.model_name
        )
        
        # Get model dimensions
        self.hidden_size = self.mlm_model.config.hidden_size
        self.vocab_size = self.mlm_model.config.vocab_size
        
        # Add special tokens if needed
        self.mask_token_id = self.tokenizer.mask_token_id
    
    def _init_verbalizer(self):
        """Initialize verbalizer with label-word mappings."""
        # Default AG News verbalizer
        if self.config.verbalizer is None:
            self.config.verbalizer = {
                0: ["world", "international", "global"],  # World
                1: ["sports", "game", "match"],           # Sports  
                2: ["business", "economy", "market"],     # Business
                3: ["technology", "science", "tech"]      # Sci/Tech
            }
        
        self.verbalizer = Verbalizer(
            self.tokenizer,
            self.config.verbalizer,
            post_log_softmax=True
        )
    
    def _init_soft_prompts(self):
        """Initialize soft prompt components."""
        # Get embedding dimension from model
        embeddings = self.mlm_model.get_input_embeddings()
        embedding_dim = embeddings.embedding_dim
        
        # Create soft prompt encoder
        self.soft_prompt_encoder = SoftPromptEncoder(
            num_tokens=self.config.num_prompt_tokens,
            token_dim=embedding_dim,
            hidden_size=self.config.prompt_encoder_hidden_size,
            init_method=self.config.prompt_init
        )
        
        # Ensemble of prompts
        if self.config.use_ensemble_prompt:
            self.prompt_ensemble = nn.ModuleList([
                SoftPromptEncoder(
                    num_tokens=self.config.num_prompt_tokens,
                    token_dim=embedding_dim,
                    hidden_size=self.config.prompt_encoder_hidden_size,
                    init_method=self.config.prompt_init
                )
                for _ in range(self.config.num_prompt_ensemble)
            ])
    
    def create_prompt_input(
        self,
        input_ids: torch.Tensor,
        template: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Create prompted input from original input.
        
        Args:
            input_ids: Original input token IDs
            template: Prompt template to use
            
        Returns:
            Tuple of (prompted_input_ids, attention_mask, mask_position)
        """
        template = template or self.config.template
        batch_size = input_ids.size(0)
        
        # Prepare template tokens
        template_text = template.replace("[MASK]", self.tokenizer.mask_token)
        template_ids = self.tokenizer.encode(template_text, add_special_tokens=False)
        
        # Find mask position in template
        mask_pos = template_ids.index(self.mask_token_id)
        
        # Combine input with template
        prompted_inputs = []
        attention_masks = []
        mask_positions = []
        
        for i in range(batch_size):
            # Get non-padded input
            valid_input = input_ids[i][input_ids[i] != self.tokenizer.pad_token_id]
            
            # Combine: [CLS] input [SEP] template [SEP]
            combined = torch.cat([
                torch.tensor([self.tokenizer.cls_token_id]),
                valid_input,
                torch.tensor([self.tokenizer.sep_token_id]),
                torch.tensor(template_ids),
                torch.tensor([self.tokenizer.sep_token_id])
            ])
            
            # Truncate if too long
            if len(combined) > self.config.max_length:
                combined = combined[:self.config.max_length]
            
            # Pad to max length
            padding_length = self.config.max_length - len(combined)
            if padding_length > 0:
                combined = torch.cat([
                    combined,
                    torch.full((padding_length,), self.tokenizer.pad_token_id)
                ])
            
            prompted_inputs.append(combined)
            
            # Create attention mask
            attention_mask = (combined != self.tokenizer.pad_token_id).long()
            attention_masks.append(attention_mask)
            
            # Calculate mask position in combined sequence
            # [CLS] + input + [SEP] + mask_pos
            actual_mask_pos = 1 + len(valid_input) + 1 + mask_pos
            mask_positions.append(actual_mask_pos)
        
        prompted_inputs = torch.stack(prompted_inputs)
        attention_masks = torch.stack(attention_masks)
        
        return prompted_inputs, attention_masks, mask_positions
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_soft_prompt: bool = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through prompt-based model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            use_soft_prompt: Whether to use soft prompts
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with predictions
        """
        batch_size = input_ids.size(0)
        use_soft = use_soft_prompt if use_soft_prompt is not None else (
            self.config.prompt_type in [PromptType.SOFT, PromptType.MIXED]
        )
        
        # Create prompted input
        prompted_ids, prompted_mask, mask_positions = self.create_prompt_input(
            input_ids
        )
        
        # Add soft prompts if configured
        if use_soft:
            # Get soft prompt embeddings
            soft_prompts = self.soft_prompt_encoder(batch_size)
            
            # Get input embeddings
            inputs_embeds = self.mlm_model.get_input_embeddings()(prompted_ids)
            
            # Prepend soft prompts
            inputs_embeds = torch.cat([soft_prompts, inputs_embeds], dim=1)
            
            # Adjust attention mask
            soft_mask = torch.ones(
                batch_size, 
                self.config.num_prompt_tokens,
                device=prompted_mask.device
            )
            prompted_mask = torch.cat([soft_mask, prompted_mask], dim=1)
            
            # Adjust mask positions
            mask_positions = [pos + self.config.num_prompt_tokens for pos in mask_positions]
            
            # Forward through MLM model with embeddings
            outputs = self.mlm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=prompted_mask,
                return_dict=True
            )
        else:
            # Forward through MLM model with input IDs
            outputs = self.mlm_model(
                input_ids=prompted_ids,
                attention_mask=prompted_mask,
                return_dict=True
            )
        
        # Extract predictions at mask positions
        logits = outputs.logits
        mask_logits = []
        
        for i, pos in enumerate(mask_positions):
            mask_logits.append(logits[i, pos])
        
        mask_logits = torch.stack(mask_logits)
        
        # Map to label scores via verbalizer
        label_scores = self.verbalizer(mask_logits)
        
        # Apply calibration
        if self.config.use_calibration:
            label_scores = label_scores + self.calibration_bias
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(label_scores, labels)
        
        return ModelOutputs(
            logits=label_scores,
            loss=loss,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
            metadata={
                "mask_logits": mask_logits,
                "mask_positions": mask_positions
            }
        )
    
    def generate_prompts(
        self,
        text: str,
        num_prompts: int = 5
    ) -> List[str]:
        """
        Generate multiple prompt templates for ensemble.
        
        Args:
            text: Input text
            num_prompts: Number of prompts to generate
            
        Returns:
            List of prompt templates
        """
        prompts = [
            f"This news is about [MASK].",
            f"The topic is [MASK].",
            f"Category: [MASK]. {text}",
            f"{text} This article discusses [MASK].",
            f"[MASK] news: {text}"
        ]
        
        return prompts[:num_prompts]
