"""
Prompt-Specific Classification Head
====================================

Implementation of specialized classification heads for prompt-based models,
based on:
- Schick & Schütze (2021): "It's Not Just Size That Matters"
- Gao et al. (2021): "Making Pre-trained Language Models Better Few-shot Learners"
- Han et al. (2021): "PTR: Prompt Tuning with Rules for Text Classification"

Prompt heads handle the unique requirements of prompt-based classification,
including verbalizer mapping and label word prediction.

Mathematical Foundation:
P(y|x) = P(v(y)|x, prompt(x))
where v(y) is the verbalizer mapping labels to words.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PromptHeadConfig:
    """Configuration for prompt classification head"""
    
    # Head type
    head_type: str = "mlm"  # "mlm", "generation", "direct"
    
    # Model dimensions
    hidden_size: int = 768
    vocab_size: int = 50265
    num_labels: int = 4
    
    # Verbalizer configuration
    label_words: Dict[int, List[int]] = None  # Label to token IDs mapping
    use_multiple_tokens: bool = True  # Use multiple tokens per label
    aggregate_method: str = "mean"  # "mean", "max", "sum" for multiple tokens
    
    # Calibration
    use_calibration: bool = True
    calibration_method: str = "contextual"  # "contextual", "affine", "none"
    
    # Advanced features
    use_label_smoothing: bool = False
    label_smoothing_epsilon: float = 0.1
    use_virtual_tokens: bool = False  # Virtual answer tokens
    num_virtual_tokens: int = 5
    
    # Training
    dropout_rate: float = 0.1
    use_bias: bool = True
    freeze_base_head: bool = False


class VerbalizerHead(nn.Module):
    """
    Verbalizer-based classification head.
    
    Maps model outputs to label words through a verbalizer,
    converting language modeling predictions to classification.
    """
    
    def __init__(self, config: PromptHeadConfig):
        """
        Initialize verbalizer head.
        
        Args:
            config: Head configuration
        """
        super().__init__()
        self.config = config
        
        # Validate label words
        if config.label_words is None:
            raise ValueError("label_words must be provided for verbalizer head")
        
        # Create label word embeddings
        self._create_label_embeddings()
        
        # Optional calibration layer
        if config.use_calibration:
            self.calibration = CalibrationLayer(
                config.vocab_size,
                config.calibration_method
            )
        
        # Optional projection for virtual tokens
        if config.use_virtual_tokens:
            self.virtual_projection = nn.Linear(
                config.hidden_size,
                config.num_virtual_tokens * config.hidden_size
            )
        
        logger.info(f"Initialized VerbalizerHead with {len(config.label_words)} labels")
    
    def _create_label_embeddings(self):
        """Create embeddings for label words"""
        # Store label word indices
        self.label_word_indices = {}
        max_words = 0
        
        for label, word_ids in self.config.label_words.items():
            self.label_word_indices[label] = torch.tensor(word_ids)
            max_words = max(max_words, len(word_ids))
        
        # Pad to same length
        for label in self.label_word_indices:
            current_len = len(self.label_word_indices[label])
            if current_len < max_words:
                # Pad with first word ID
                padding = self.label_word_indices[label][0].repeat(max_words - current_len)
                self.label_word_indices[label] = torch.cat([
                    self.label_word_indices[label],
                    padding
                ])
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        masked_positions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through verbalizer head.
        
        Args:
            hidden_states: Hidden states from model [batch_size, seq_len, hidden_size]
            lm_head: Language model head for vocabulary projection
            masked_positions: Positions of masked tokens [batch_size]
            
        Returns:
            Classification logits [batch_size, num_labels]
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        
        # Get hidden states at masked positions
        if masked_positions is not None:
            # Extract masked position representations
            masked_hidden = hidden_states[
                torch.arange(batch_size, device=device),
                masked_positions
            ]
        else:
            # Use first position (e.g., CLS token)
            masked_hidden = hidden_states[:, 0]
        
        # Project to vocabulary space
        vocab_logits = lm_head(masked_hidden)  # [batch_size, vocab_size]
        
        # Apply calibration if configured
        if self.config.use_calibration:
            vocab_logits = self.calibration(vocab_logits)
        
        # Extract label word logits
        label_logits = []
        
        for label in range(self.config.num_labels):
            word_indices = self.label_word_indices[label].to(device)
            
            # Get logits for label words
            word_logits = vocab_logits[:, word_indices]  # [batch_size, num_words]
            
            # Aggregate multiple words
            if self.config.aggregate_method == "mean":
                label_score = word_logits.mean(dim=1)
            elif self.config.aggregate_method == "max":
                label_score = word_logits.max(dim=1)[0]
            elif self.config.aggregate_method == "sum":
                label_score = word_logits.sum(dim=1)
            else:
                label_score = word_logits.mean(dim=1)
            
            label_logits.append(label_score)
        
        # Stack to create final logits
        logits = torch.stack(label_logits, dim=1)  # [batch_size, num_labels]
        
        return logits


class GenerationHead(nn.Module):
    """
    Generation-based classification head.
    
    For generative models that produce label text directly,
    maps generated text to classification labels.
    """
    
    def __init__(self, config: PromptHeadConfig):
        """
        Initialize generation head.
        
        Args:
            config: Head configuration
        """
        super().__init__()
        self.config = config
        
        # Projection layer
        self.projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
        # Label mappings
        self.label_tokens = {}  # Maps labels to token sequences
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        decoder_input_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through generation head.
        
        Args:
            hidden_states: Decoder hidden states
            decoder_input_ids: Decoder input token IDs
            
        Returns:
            Classification logits
        """
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Project to vocabulary
        vocab_logits = self.projection(hidden_states)
        
        # For classification, we typically use the first generated token
        # or aggregate over the generated sequence
        if len(vocab_logits.shape) == 3:
            # Take first position
            vocab_logits = vocab_logits[:, 0]
        
        # Map to label logits based on label tokens
        # This is simplified - actual implementation would be more sophisticated
        batch_size = vocab_logits.shape[0]
        label_logits = torch.zeros(batch_size, self.config.num_labels, device=vocab_logits.device)
        
        for label, tokens in self.label_tokens.items():
            if isinstance(tokens, list):
                # Average over multiple tokens
                for token in tokens:
                    label_logits[:, label] += vocab_logits[:, token]
                label_logits[:, label] /= len(tokens)
            else:
                label_logits[:, label] = vocab_logits[:, tokens]
        
        return label_logits


class DirectPromptHead(nn.Module):
    """
    Direct classification head for prompt models.
    
    Directly classifies based on prompt-enhanced representations
    without going through vocabulary space.
    """
    
    def __init__(self, config: PromptHeadConfig):
        """
        Initialize direct prompt head.
        
        Args:
            config: Head configuration
        """
        super().__init__()
        self.config = config
        
        # Multi-layer classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        
        # Optional attention over prompt positions
        self.prompt_attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=8,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Query vector for attention
        self.query = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through direct prompt head.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            prompt_mask: Mask for prompt positions
            
        Returns:
            Classification logits
        """
        batch_size = hidden_states.shape[0]
        
        # Apply attention over prompt positions
        query = self.query.expand(batch_size, -1, -1)
        
        if prompt_mask is not None:
            attended, _ = self.prompt_attention(
                query,
                hidden_states,
                hidden_states,
                key_padding_mask=~prompt_mask
            )
        else:
            attended, _ = self.prompt_attention(
                query,
                hidden_states,
                hidden_states
            )
        
        # Squeeze query dimension
        pooled = attended.squeeze(1)
        
        # Classify
        logits = self.classifier(pooled)
        
        return logits


class CalibrationLayer(nn.Module):
    """
    Calibration layer for adjusting model predictions.
    
    Improves probability calibration for prompt-based predictions.
    """
    
    def __init__(
        self,
        input_size: int,
        method: str = "contextual"
    ):
        """
        Initialize calibration layer.
        
        Args:
            input_size: Input dimension
            method: Calibration method
        """
        super().__init__()
        self.method = method
        
        if method == "affine":
            # Affine calibration
            self.weight = nn.Parameter(torch.ones(input_size))
            self.bias = nn.Parameter(torch.zeros(input_size))
            
        elif method == "contextual":
            # Contextual calibration with neural network
            self.calibrator = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, input_size)
            )
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply calibration to logits.
        
        Args:
            logits: Input logits
            
        Returns:
            Calibrated logits
        """
        if self.method == "affine":
            return logits * self.weight + self.bias
        elif self.method == "contextual":
            return logits + self.calibrator(logits)
        else:
            return logits


def create_prompt_head(
    head_type: str,
    config: PromptHeadConfig
) -> nn.Module:
    """
    Factory function to create prompt heads.
    
    Args:
        head_type: Type of head to create
        config: Head configuration
        
    Returns:
        Prompt head module
    """
    heads = {
        'verbalizer': VerbalizerHead,
        'generation': GenerationHead,
        'direct': DirectPromptHead
    }
    
    if head_type not in heads:
        raise ValueError(f"Unknown head type: {head_type}")
    
    return heads[head_type](config)


# Export classes
__all__ = [
    'PromptHeadConfig',
    'VerbalizerHead',
    'GenerationHead',
    'DirectPromptHead',
    'CalibrationLayer',
    'create_prompt_head'
]
