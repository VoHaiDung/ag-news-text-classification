"""
Tokenization Module
===================

Implements various tokenization strategies following:
- Kudo & Richardson (2018): "SentencePiece: A simple and language independent subword tokenizer"
- Sennrich et al. (2016): "Neural Machine Translation of Rare Words with Subword Units"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast
)
import torch

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.constants import MAX_SEQUENCE_LENGTH
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class TokenizationConfig:
    """Configuration for tokenization."""
    
    model_name: str = "bert-base-uncased"
    max_length: int = MAX_SEQUENCE_LENGTH
    padding: Union[bool, str] = "max_length"
    truncation: bool = True
    return_tensors: Optional[str] = "pt"
    
    # Special tokens
    add_special_tokens: bool = True
    
    # Additional options
    return_attention_mask: bool = True
    return_token_type_ids: bool = False
    
    # Sliding window for long texts
    stride: int = 0
    return_overflowing_tokens: bool = False

class Tokenizer:
    """
    Unified tokenizer interface.
    
    Supports multiple tokenization backends following:
    - Wolf et al. (2020): "Transformers: State-of-the-Art Natural Language Processing"
    """
    
    def __init__(
        self,
        config: Optional[TokenizationConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Initialize tokenizer.
        
        Args:
            config: Tokenization configuration
            tokenizer: Pre-initialized tokenizer
        """
        self.config = config or TokenizationConfig()
        
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self._load_tokenizer()
        
        logger.info(f"Initialized tokenizer: {self.tokenizer.__class__.__name__}")
    
    def _load_tokenizer(self):
        """Load tokenizer from model name."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True  # Use fast tokenizer when available
            )
            
            # Add special tokens if needed
            special_tokens = {
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
            }
            
            # Check and add missing special tokens
            tokens_added = 0
            for key, value in special_tokens.items():
                if getattr(self.tokenizer, key) is None:
                    setattr(self.tokenizer, key, value)
                    tokens_added += 1
            
            if tokens_added > 0:
                logger.info(f"Added {tokens_added} special tokens")
                
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def tokenize(
        self,
        text: Union[str, List[str]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize text(s).
        
        Args:
            text: Input text or list of texts
            **kwargs: Override default configuration
            
        Returns:
            Dictionary with tokenized outputs
        """
        # Merge with default config
        tokenization_kwargs = {
            "max_length": self.config.max_length,
            "padding": self.config.padding,
            "truncation": self.config.truncation,
            "return_tensors": self.config.return_tensors,
            "add_special_tokens": self.config.add_special_tokens,
            "return_attention_mask": self.config.return_attention_mask,
            "return_token_type_ids": self.config.return_token_type_ids,
        }
        tokenization_kwargs.update(kwargs)
        
        # Handle sliding window if configured
        if self.config.stride > 0:
            tokenization_kwargs["stride"] = self.config.stride
            tokenization_kwargs["return_overflowing_tokens"] = True
        
        return self.tokenizer(text, **tokenization_kwargs)
    
    def batch_tokenize(
        self,
        texts: List[str],
        batch_size: int = 32,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize texts in batches.
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            **kwargs: Tokenization arguments
            
        Returns:
            Batched tokenization outputs
        """
        all_outputs = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            outputs = self.tokenize(batch, **kwargs)
            all_outputs.append(outputs)
        
        # Concatenate outputs
        if all_outputs:
            concatenated = {}
            for key in all_outputs[0].keys():
                if isinstance(all_outputs[0][key], torch.Tensor):
                    concatenated[key] = torch.cat([o[key] for o in all_outputs])
                else:
                    concatenated[key] = sum([o[key] for o in all_outputs], [])
            
            return concatenated
        
        return {}
    
    def decode(
        self,
        token_ids: Union[List[int], torch.Tensor],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens
        )
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
    
    def save(self, save_directory: Union[str, Path]):
        """Save tokenizer."""
        self.tokenizer.save_pretrained(save_directory)
        logger.info(f"Tokenizer saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "Tokenizer":
        """Load tokenizer from pretrained."""
        config = TokenizationConfig(model_name=model_name_or_path, **kwargs)
        return cls(config)

# Factory functions for common tokenizers
def get_bert_tokenizer(**kwargs) -> Tokenizer:
    """Get BERT tokenizer."""
    config = TokenizationConfig(model_name="bert-base-uncased", **kwargs)
    return Tokenizer(config)

def get_roberta_tokenizer(**kwargs) -> Tokenizer:
    """Get RoBERTa tokenizer."""
    config = TokenizationConfig(model_name="roberta-base", **kwargs)
    return Tokenizer(config)

def get_deberta_tokenizer(**kwargs) -> Tokenizer:
    """Get DeBERTa tokenizer."""
    config = TokenizationConfig(model_name="microsoft/deberta-v3-base", **kwargs)
    return Tokenizer(config)
