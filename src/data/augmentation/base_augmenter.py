"""
Base Augmenter Module
=====================

Abstract base class for data augmentation following:
- Shorten et al. (2021): "Text Data Augmentation for Deep Learning"
- Wei & Zou (2019): "EDA: Easy Data Augmentation Techniques"
- Feng et al. (2021): "A Survey of Data Augmentation Approaches for NLP"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import random
import numpy as np
from pathlib import Path

import torch
from transformers import PreTrainedTokenizer

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.core.exceptions import AugmentationError
from configs.constants import AG_NEWS_CLASSES, MAX_SEQUENCE_LENGTH

logger = setup_logging(__name__)

@dataclass
class AugmentationConfig:
    """
    Base configuration for augmentation.
    
    Following configuration patterns from:
    - Bayer et al. (2022): "A Survey on Data Augmentation for Text Classification"
    """
    # Basic settings
    augmentation_rate: float = 0.5
    num_augmentations: int = 1
    
    # Quality control
    min_similarity: float = 0.8
    max_similarity: float = 0.99
    preserve_label: bool = True
    
    # Length constraints
    min_length: int = 10
    max_length: int = MAX_SEQUENCE_LENGTH
    
    # Randomization
    seed: int = 42
    temperature: float = 1.0
    
    # Performance
    batch_size: int = 32
    cache_augmented: bool = True
    
    # Validation
    validate_augmented: bool = True
    filter_invalid: bool = True

class BaseAugmenter(ABC):
    """
    Abstract base class for text augmentation.
    
    Implements common functionality following:
    - Coulombe (2018): "Text Data Augmentation Made Simple"
    - Marivate & Sefara (2020): "Improving Short Text Classification Through Global Augmentation Methods"
    """
    
    def __init__(
        self,
        config: Optional[AugmentationConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        name: str = "base_augmenter"
    ):
        """
        Initialize base augmenter.
        
        Args:
            config: Augmentation configuration
            tokenizer: Optional tokenizer
            name: Augmenter name
        """
        self.config = config or AugmentationConfig()
        self.tokenizer = tokenizer
        self.name = name
        
        # Set random seeds
        self.rng = random.Random(self.config.seed)
        self.np_rng = np.random.RandomState(self.config.seed)
        
        # Cache for augmented samples
        self.cache = {} if self.config.cache_augmented else None
        
        # Statistics tracking
        self.stats = {
            'total_augmented': 0,
            'successful': 0,
            'failed': 0,
            'filtered': 0,
            'cached': 0
        }
        
        logger.info(f"Initialized {self.name} augmenter")
    
    @abstractmethod
    def augment_single(
        self,
        text: str,
        label: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Augment single text sample.
        
        Args:
            text: Input text
            label: Optional label
            **kwargs: Additional arguments
            
        Returns:
            Augmented text(s)
        """
        pass
    
    def augment_batch(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        **kwargs
    ) -> List[List[str]]:
        """
        Augment batch of texts.
        
        Args:
            texts: List of texts
            labels: Optional labels
            **kwargs: Additional arguments
            
        Returns:
            List of augmented texts for each input
        """
        augmented_batch = []
        
        for i, text in enumerate(texts):
            label = labels[i] if labels else None
            
            # Check if should augment
            if self.rng.random() > self.config.augmentation_rate:
                augmented_batch.append([text])  # Return original
                continue
            
            # Get augmented versions
            augmented = self.augment_single(text, label, **kwargs)
            
            # Ensure list format
            if isinstance(augmented, str):
                augmented = [augmented]
            
            augmented_batch.append(augmented)
            
            # Update statistics
            self.stats['total_augmented'] += len(augmented)
            self.stats['successful'] += len(augmented)
        
        return augmented_batch
    
    def augment_dataset(
        self,
        dataset: Any,
        num_augmentations: Optional[int] = None,
        **kwargs
    ) -> Any:
        """
        Augment entire dataset.
        
        Args:
            dataset: Input dataset
            num_augmentations: Number of augmentations per sample
            **kwargs: Additional arguments
            
        Returns:
            Augmented dataset
        """
        num_augmentations = num_augmentations or self.config.num_augmentations
        
        augmented_data = []
        
        for item in dataset:
            # Extract text and label
            if isinstance(item, dict):
                text = item.get('text', '')
                label = item.get('label', None)
            elif isinstance(item, tuple):
                text, label = item[0], item[1] if len(item) > 1 else None
            else:
                text, label = str(item), None
            
            # Generate augmentations
            for _ in range(num_augmentations):
                aug_text = self.augment_single(text, label, **kwargs)
                
                if isinstance(aug_text, list):
                    for aug in aug_text:
                        augmented_data.append({
                            'text': aug,
                            'label': label,
                            'original': False,
                            'augmentation_method': self.name
                        })
                else:
                    augmented_data.append({
                        'text': aug_text,
                        'label': label,
                        'original': False,
                        'augmentation_method': self.name
                    })
        
        return augmented_data
    
    def validate_augmentation(
        self,
        original: str,
        augmented: str,
        label: Optional[int] = None
    ) -> bool:
        """
        Validate augmented text.
        
        Following validation practices from:
        - Ng et al. (2020): "SSMBA: Self-Supervised Manifold Based Data Augmentation"
        """
        # Length check
        aug_len = len(augmented.split())
        if aug_len < self.config.min_length or aug_len > self.config.max_length:
            return False
        
        # Similarity check
        if hasattr(self, 'compute_similarity'):
            similarity = self.compute_similarity(original, augmented)
            if similarity < self.config.min_similarity or similarity > self.config.max_similarity:
                return False
        
        # Label preservation check
        if self.config.preserve_label and label is not None:
            if hasattr(self, 'predict_label'):
                predicted = self.predict_label(augmented)
                if predicted != label:
                    return False
        
        return True
    
    def filter_augmentations(
        self,
        original: str,
        augmented: List[str],
        label: Optional[int] = None
    ) -> List[str]:
        """Filter invalid augmentations."""
        if not self.config.filter_invalid:
            return augmented
        
        filtered = []
        for aug in augmented:
            if self.validate_augmentation(original, aug, label):
                filtered.append(aug)
            else:
                self.stats['filtered'] += 1
        
        return filtered
    
    def get_cache_key(self, text: str, **kwargs) -> str:
        """Generate cache key for text."""
        # Simple hash-based key
        key_parts = [text[:50], str(len(text)), self.name]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return "|".join(key_parts)
    
    def get_from_cache(self, text: str, **kwargs) -> Optional[List[str]]:
        """Get augmented text from cache."""
        if not self.cache:
            return None
        
        key = self.get_cache_key(text, **kwargs)
        if key in self.cache:
            self.stats['cached'] += 1
            return self.cache[key]
        
        return None
    
    def add_to_cache(self, text: str, augmented: List[str], **kwargs):
        """Add augmented text to cache."""
        if self.cache is not None:
            key = self.get_cache_key(text, **kwargs)
            self.cache[key] = augmented
    
    def reset_stats(self):
        """Reset augmentation statistics."""
        self.stats = {
            'total_augmented': 0,
            'successful': 0,
            'failed': 0,
            'filtered': 0,
            'cached': 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get augmentation statistics."""
        return {
            **self.stats,
            'success_rate': self.stats['successful'] / max(self.stats['total_augmented'], 1),
            'filter_rate': self.stats['filtered'] / max(self.stats['total_augmented'], 1),
            'cache_hit_rate': self.stats['cached'] / max(self.stats['total_augmented'], 1)
        }
    
    def save_config(self, path: Union[str, Path]):
        """Save configuration to file."""
        import json
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
    
    def load_config(self, path: Union[str, Path]):
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        self.config = AugmentationConfig(**config_dict)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"{self.__class__.__name__}("
                f"name='{self.name}', "
                f"rate={self.config.augmentation_rate}, "
                f"num={self.config.num_augmentations})")

class CompositeAugmenter(BaseAugmenter):
    """
    Composite augmenter that combines multiple augmentation techniques.
    
    Following composition patterns from:
    - Karimi et al. (2021): "AEDA: An Easier Data Augmentation Technique for Text Classification"
    """
    
    def __init__(
        self,
        augmenters: List[BaseAugmenter],
        config: Optional[AugmentationConfig] = None,
        strategy: str = "sequential"  # sequential, random, all
    ):
        """
        Initialize composite augmenter.
        
        Args:
            augmenters: List of augmenters to combine
            config: Configuration
            strategy: Composition strategy
        """
        super().__init__(config, name="composite")
        self.augmenters = augmenters
        self.strategy = strategy
    
    def augment_single(
        self,
        text: str,
        label: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """Apply multiple augmentation techniques."""
        augmented = []
        
        if self.strategy == "sequential":
            # Apply augmenters in sequence
            current = text
            for augmenter in self.augmenters:
                result = augmenter.augment_single(current, label, **kwargs)
                if isinstance(result, list):
                    current = result[0] if result else current
                else:
                    current = result
            augmented.append(current)
            
        elif self.strategy == "random":
            # Randomly select an augmenter
            augmenter = self.rng.choice(self.augmenters)
            result = augmenter.augment_single(text, label, **kwargs)
            augmented = result if isinstance(result, list) else [result]
            
        elif self.strategy == "all":
            # Apply all augmenters independently
            for augmenter in self.augmenters:
                result = augmenter.augment_single(text, label, **kwargs)
                if isinstance(result, list):
                    augmented.extend(result)
                else:
                    augmented.append(result)
        
        return augmented
