"""
Sliding Window for Long Texts
==============================

Implements sliding window approach for handling long sequences following:
- Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"
- Beltagy et al. (2020): "Longformer: The Long-Document Transformer"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import torch

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from configs.constants import MAX_SEQUENCE_LENGTH

logger = setup_logging(__name__)

@dataclass
class SlidingWindowConfig:
    """Configuration for sliding window."""
    
    window_size: int = MAX_SEQUENCE_LENGTH
    stride: int = 128
    min_window_size: int = 50
    
    # Aggregation strategy
    aggregation: str = "mean"  # mean, max, first, vote
    
    # Padding
    pad_last_window: bool = True
    
    # Special tokens handling
    preserve_special_tokens: bool = True
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"

class SlidingWindow:
    """
    Sliding window processor for long texts.
    
    Following windowing strategies from:
    - Pappagari et al. (2019): "Hierarchical Transformers for Long Document Classification"
    """
    
    def __init__(self, config: Optional[SlidingWindowConfig] = None):
        """
        Initialize sliding window processor.
        
        Args:
            config: Sliding window configuration
        """
        self.config = config or SlidingWindowConfig()
        logger.info(f"Initialized sliding window: size={self.config.window_size}, stride={self.config.stride}")
    
    def create_windows(
        self,
        text: str,
        tokenizer: Any = None,
        return_offsets: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Create sliding windows from text.
        
        Args:
            text: Input text
            tokenizer: Optional tokenizer for token-level windowing
            return_offsets: Whether to return character offsets
            
        Returns:
            List of windows with metadata
        """
        if tokenizer:
            return self._create_token_windows(text, tokenizer, return_offsets)
        else:
            return self._create_char_windows(text, return_offsets)
    
    def _create_char_windows(
        self,
        text: str,
        return_offsets: bool = False
    ) -> List[Dict[str, Any]]:
        """Create character-level sliding windows."""
        windows = []
        text_length = len(text)
        
        start = 0
        window_id = 0
        
        while start < text_length:
            end = min(start + self.config.window_size, text_length)
            
            # Check if remaining text is too small
            if end - start < self.config.min_window_size and window_id > 0:
                # Merge with previous window if too small
                if windows:
                    windows[-1]['text'] += " " + text[start:end]
                    windows[-1]['end'] = end
                break
            
            window = {
                'window_id': window_id,
                'text': text[start:end],
                'start': start,
                'end': end,
                'is_last': end >= text_length
            }
            
            if return_offsets:
                window['char_offsets'] = (start, end)
            
            windows.append(window)
            
            start += self.config.stride
            window_id += 1
        
        return windows
    
    def _create_token_windows(
        self,
        text: str,
        tokenizer: Any,
        return_offsets: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Create token-level sliding windows.
        
        Following tokenization strategies from:
        - Kudo & Richardson (2018): "SentencePiece"
        """
        # Tokenize full text
        encoding = tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids']
        offset_mapping = encoding.get('offset_mapping', [])
        
        windows = []
        window_id = 0
        
        # Account for special tokens
        special_tokens_count = 2 if self.config.preserve_special_tokens else 0
        effective_window_size = self.config.window_size - special_tokens_count
        
        start_idx = 0
        
        while start_idx < len(input_ids):
            end_idx = min(start_idx + effective_window_size, len(input_ids))
            
            # Check minimum size
            if end_idx - start_idx < self.config.min_window_size and window_id > 0:
                # Extend previous window
                if windows:
                    windows[-1]['input_ids'].extend(input_ids[start_idx:end_idx])
                    windows[-1]['is_last'] = True
                break
            
            window_input_ids = input_ids[start_idx:end_idx]
            
            # Add special tokens if configured
            if self.config.preserve_special_tokens:
                # Add CLS at beginning
                if tokenizer.cls_token_id is not None:
                    window_input_ids = [tokenizer.cls_token_id] + window_input_ids
                
                # Add SEP at end
                if tokenizer.sep_token_id is not None:
                    window_input_ids = window_input_ids + [tokenizer.sep_token_id]
            
            window = {
                'window_id': window_id,
                'input_ids': window_input_ids,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'is_last': end_idx >= len(input_ids)
            }
            
            # Add text reconstruction if possible
            if offset_mapping:
                start_char = offset_mapping[start_idx][0] if start_idx < len(offset_mapping) else 0
                end_char = offset_mapping[min(end_idx-1, len(offset_mapping)-1)][1] if end_idx > 0 else len(text)
                window['text'] = text[start_char:end_char]
                
                if return_offsets:
                    window['char_offsets'] = (start_char, end_char)
                    window['token_offsets'] = (start_idx, end_idx)
            
            windows.append(window)
            
            start_idx += self.config.stride
            window_id += 1
        
        return windows
    
    def aggregate_predictions(
        self,
        predictions: List[torch.Tensor],
        strategy: Optional[str] = None
    ) -> torch.Tensor:
        """
        Aggregate predictions from multiple windows.
        
        Args:
            predictions: List of prediction tensors
            strategy: Aggregation strategy
            
        Returns:
            Aggregated prediction
        """
        if not predictions:
            raise ValueError("No predictions to aggregate")
        
        strategy = strategy or self.config.aggregation
        
        # Stack predictions
        stacked = torch.stack(predictions)
        
        if strategy == "mean":
            return stacked.mean(dim=0)
        elif strategy == "max":
            return stacked.max(dim=0)[0]
        elif strategy == "first":
            return predictions[0]
        elif strategy == "vote":
            # For classification - majority voting
            if len(stacked.shape) == 2:  # [windows, classes]
                votes = stacked.argmax(dim=1)
                # Return one-hot of most common class
                winner = torch.mode(votes)[0].item()
                result = torch.zeros_like(predictions[0])
                result[winner] = 1.0
                return result
            else:
                return stacked.mean(dim=0)
        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")
    
    def process_batch(
        self,
        texts: List[str],
        tokenizer: Any = None
    ) -> Dict[str, Any]:
        """
        Process batch of texts with sliding window.
        
        Args:
            texts: List of texts
            tokenizer: Optional tokenizer
            
        Returns:
            Batched windows with metadata
        """
        all_windows = []
        text_to_windows = {}
        
        for text_idx, text in enumerate(texts):
            windows = self.create_windows(text, tokenizer)
            
            # Track which windows belong to which text
            window_indices = []
            for window in windows:
                window['text_idx'] = text_idx
                window_indices.append(len(all_windows))
                all_windows.append(window)
            
            text_to_windows[text_idx] = window_indices
        
        return {
            'windows': all_windows,
            'text_to_windows': text_to_windows,
            'num_texts': len(texts),
            'num_windows': len(all_windows)
        }
