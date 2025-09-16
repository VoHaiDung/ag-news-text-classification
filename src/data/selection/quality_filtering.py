"""
Quality Filtering Module
========================

Filters low-quality data following:
- Raffel et al. (2020): "Exploring the Limits of Transfer Learning"
- Brown et al. (2020): "Language Models are Few-Shot Learners"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class QualityFilterConfig:
    """Configuration for quality filtering."""
    
    min_length: int = 10
    max_length: int = 1000
    min_unique_words: int = 5
    max_repetition_ratio: float = 0.3
    min_alpha_ratio: float = 0.7
    remove_duplicates: bool = True
    perplexity_threshold: Optional[float] = None

class QualityFilter:
    """
    Filter low-quality text data.
    
    Following filtering strategies from:
    - Gururangan et al. (2020): "Don't Stop Pretraining"
    """
    
    def __init__(self, config: QualityFilterConfig):
        """
        Initialize quality filter.
        
        Args:
            config: Filter configuration
        """
        self.config = config
        self.seen_hashes = set()
        
    def filter(self, texts: List[str]) -> List[bool]:
        """
        Filter texts based on quality criteria.
        
        Args:
            texts: List of texts
            
        Returns:
            List of boolean masks (True = keep)
        """
        masks = []
        
        for text in texts:
            # Check all criteria
            passed = True
            
            # Length check
            word_count = len(text.split())
            if word_count < self.config.min_length or word_count > self.config.max_length:
                passed = False
            
            # Unique words check
            unique_words = len(set(text.lower().split()))
            if unique_words < self.config.min_unique_words:
                passed = False
            
            # Repetition check
            if word_count > 0:
                repetition_ratio = 1 - (unique_words / word_count)
                if repetition_ratio > self.config.max_repetition_ratio:
                    passed = False
            
            # Alpha ratio check
            alpha_chars = sum(c.isalpha() for c in text)
            if len(text) > 0:
                alpha_ratio = alpha_chars / len(text)
                if alpha_ratio < self.config.min_alpha_ratio:
                    passed = False
            
            # Duplicate check
            if self.config.remove_duplicates:
                text_hash = hash(text)
                if text_hash in self.seen_hashes:
                    passed = False
                else:
                    self.seen_hashes.add(text_hash)
            
            masks.append(passed)
        
        # Log statistics
        kept = sum(masks)
        total = len(masks)
        logger.info(f"Quality filter: kept {kept}/{total} samples ({kept/total*100:.1f}%)")
        
        return masks
