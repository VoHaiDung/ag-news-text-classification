"""
Balanced Sampling Module
========================

Implements balanced sampling for imbalanced datasets following:
- Chawla et al. (2002): "SMOTE: Synthetic Minority Over-sampling Technique"
- He & Garcia (2009): "Learning from Imbalanced Data"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional
import torch
from torch.utils.data import Sampler
import numpy as np
from collections import Counter

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

class BalancedSampler(Sampler):
    """
    Balanced sampler for handling class imbalance.
    
    Following sampling strategies from:
    - Shen et al. (2016): "Relay Backpropagation for Effective Learning"
    """
    
    def __init__(
        self,
        dataset,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        seed: int = 42
    ):
        """
        Initialize balanced sampler.
        
        Args:
            dataset: Dataset with labels
            num_samples: Number of samples per epoch
            replacement: Sample with replacement
            seed: Random seed
        """
        self.dataset = dataset
        self.replacement = replacement
        self.seed = seed
        
        # Get labels
        self.labels = self._get_labels()
        
        # Calculate weights
        self.weights = self._calculate_weights()
        
        # Set number of samples
        self.num_samples = num_samples or len(self.dataset)
        
        logger.info(f"Created balanced sampler for {len(set(self.labels))} classes")
    
    def _get_labels(self) -> List[int]:
        """Extract labels from dataset."""
        labels = []
        
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            
            if isinstance(item, dict):
                label = item.get('label', item.get('labels', 0))
            elif isinstance(item, tuple):
                label = item[1] if len(item) > 1 else 0
            else:
                label = 0
            
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            labels.append(label)
        
        return labels
    
    def _calculate_weights(self) -> torch.Tensor:
        """Calculate sampling weights for balance."""
        class_counts = Counter(self.labels)
        num_classes = len(class_counts)
        
        # Calculate inverse frequency weights
        weights = []
        for label in self.labels:
            weight = 1.0 / (class_counts[label] * num_classes)
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def __iter__(self):
        """Generate indices."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement,
            generator=g
        ).tolist()
        
        return iter(indices)
    
    def __len__(self):
        """Get number of samples."""
        return self.num_samples
