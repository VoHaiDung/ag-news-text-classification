"""
Curriculum Learning Sampler
===========================

Implements curriculum learning sampling following:
- Bengio et al. (2009): "Curriculum Learning"
- Platanios et al. (2019): "Competence-based Curriculum Learning"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Callable
import torch
from torch.utils.data import Sampler
import numpy as np

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

class CurriculumSampler(Sampler):
    """
    Curriculum learning sampler that orders samples by difficulty.
    
    Following curriculum strategies from:
    - Kumar et al. (2010): "Self-Paced Learning for Latent Variable Models"
    """
    
    def __init__(
        self,
        dataset,
        difficulty_scores: Optional[List[float]] = None,
        curriculum_fn: Optional[Callable] = None,
        pacing_fn: str = "linear",
        num_epochs: int = 10,
        seed: int = 42
    ):
        """
        Initialize curriculum sampler.
        
        Args:
            dataset: Dataset to sample from
            difficulty_scores: Pre-computed difficulty scores
            curriculum_fn: Function to compute difficulty
            pacing_fn: Pacing function (linear, exponential, step)
            num_epochs: Total number of epochs
            seed: Random seed
        """
        self.dataset = dataset
        self.num_epochs = num_epochs
        self.current_epoch = 0
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        # Get difficulty scores
        if difficulty_scores is not None:
            self.difficulty_scores = difficulty_scores
        else:
            self.difficulty_scores = self._compute_difficulty(curriculum_fn)
        
        # Sort indices by difficulty
        self.sorted_indices = np.argsort(self.difficulty_scores)
        
        # Set pacing function
        self.pacing_fn = self._get_pacing_fn(pacing_fn)
        
        logger.info(f"Created curriculum sampler with {pacing_fn} pacing")
    
    def _compute_difficulty(self, curriculum_fn: Optional[Callable]) -> List[float]:
        """Compute difficulty scores for samples."""
        if curriculum_fn is None:
            # Default: use text length as difficulty
            scores = []
            for i in range(len(self.dataset)):
                item = self.dataset[i]
                if isinstance(item, dict) and 'text' in item:
                    score = len(item['text'].split())
                else:
                    score = i  # Fallback to index
                scores.append(score)
        else:
            scores = [curriculum_fn(self.dataset[i]) for i in range(len(self.dataset))]
        
        return scores
    
    def _get_pacing_fn(self, pacing_type: str) -> Callable:
        """Get pacing function for curriculum."""
        if pacing_type == "linear":
            return lambda t: min(1.0, 0.2 + 0.8 * t)
        elif pacing_type == "exponential":
            return lambda t: 1.0 - np.exp(-5 * t)
        elif pacing_type == "step":
            return lambda t: 0.3 if t < 0.3 else 0.6 if t < 0.6 else 1.0
        else:
            return lambda t: 1.0
    
    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum."""
        self.current_epoch = epoch
        self.rng = np.random.RandomState(self.seed + epoch)
    
    def __iter__(self):
        """Generate indices based on curriculum."""
        # Calculate curriculum progress
        progress = self.current_epoch / max(1, self.num_epochs - 1)
        
        # Get percentage of data to use
        data_percentage = self.pacing_fn(progress)
        num_samples = int(len(self.dataset) * data_percentage)
        num_samples = max(1, num_samples)
        
        # Select easiest samples based on curriculum
        selected_indices = self.sorted_indices[:num_samples].copy()
        
        # Shuffle selected samples
        self.rng.shuffle(selected_indices)
        
        return iter(selected_indices)
    
    def __len__(self):
        """Get number of samples for current epoch."""
        progress = self.current_epoch / max(1, self.num_epochs - 1)
        data_percentage = self.pacing_fn(progress)
        return int(len(self.dataset) * data_percentage)
