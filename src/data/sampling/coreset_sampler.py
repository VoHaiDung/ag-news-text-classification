"""
Coreset Sampling Module
=======================

Implements coreset selection for representative sampling following:
- Sener & Savarese (2018): "Active Learning for Convolutional Neural Networks: A Core-Set Approach"
- Campbell & Broderick (2018): "Bayesian Coreset Construction via Greedy Iterative Geodesic Ascent"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Union
import numpy as np
import torch
from torch.utils.data import Sampler
from sklearn.metrics.pairwise import euclidean_distances

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

class CoresetSampler(Sampler):
    """
    Coreset sampler for selecting representative subset.
    
    Implements k-center greedy algorithm from:
    - Wolf (2011): "Facility Location: Concepts, Models, Algorithms and Case Studies"
    """
    
    def __init__(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        n_samples: int,
        initial_indices: Optional[List[int]] = None,
        metric: str = "euclidean",
        seed: int = 42
    ):
        """
        Initialize coreset sampler.
        
        Args:
            embeddings: Feature embeddings of samples
            n_samples: Number of samples to select
            initial_indices: Initial selected indices
            metric: Distance metric
            seed: Random seed
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        self.embeddings = embeddings
        self.n_samples = min(n_samples, len(embeddings))
        self.metric = metric
        self.rng = np.random.RandomState(seed)
        
        # Initialize selected indices
        if initial_indices:
            self.selected_indices = list(initial_indices)
        else:
            # Random initialization
            first_idx = self.rng.randint(len(embeddings))
            self.selected_indices = [first_idx]
        
        # Compute coreset
        self._compute_coreset()
        
        logger.info(f"Created coreset with {len(self.selected_indices)} samples")
    
    def _compute_coreset(self):
        """
        Compute coreset using greedy k-center algorithm.
        
        Following algorithmic approach from:
        - Har-Peled & Kushal (2007): "Smaller Coresets for k-Median and k-Means Clustering"
        """
        n_total = len(self.embeddings)
        
        # Track minimum distances to selected points
        min_distances = np.full(n_total, np.inf)
        
        # Update distances for initial points
        for idx in self.selected_indices:
            distances = self._compute_distances(self.embeddings[idx])
            min_distances = np.minimum(min_distances, distances)
        
        # Greedily select remaining points
        while len(self.selected_indices) < self.n_samples:
            # Select point with maximum minimum distance
            new_idx = np.argmax(min_distances)
            
            # Add to selected set
            self.selected_indices.append(new_idx)
            
            # Update minimum distances
            new_distances = self._compute_distances(self.embeddings[new_idx])
            min_distances = np.minimum(min_distances, new_distances)
            
            # Set distance to self as -inf to avoid reselection
            min_distances[new_idx] = -np.inf
    
    def _compute_distances(self, embedding: np.ndarray) -> np.ndarray:
        """
        Compute distances from one embedding to all others.
        
        Args:
            embedding: Query embedding
            
        Returns:
            Distance array
        """
        if self.metric == "euclidean":
            distances = np.linalg.norm(self.embeddings - embedding, axis=1)
        elif self.metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            dot_product = np.dot(self.embeddings, embedding)
            norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(embedding)
            similarities = dot_product / (norms + 1e-8)
            distances = 1 - similarities
        else:
            # Default to euclidean
            distances = np.linalg.norm(self.embeddings - embedding, axis=1)
        
        return distances
    
    def get_subset_indices(self) -> List[int]:
        """Get selected coreset indices."""
        return self.selected_indices
    
    def compute_coverage(self) -> float:
        """
        Compute coverage radius of coreset.
        
        Coverage = max_{x in X} min_{c in C} d(x, c)
        """
        max_min_distance = 0
        
        for i in range(len(self.embeddings)):
            if i in self.selected_indices:
                continue
            
            min_distance = np.inf
            for j in self.selected_indices:
                distance = np.linalg.norm(self.embeddings[i] - self.embeddings[j])
                min_distance = min(min_distance, distance)
            
            max_min_distance = max(max_min_distance, min_distance)
        
        return max_min_distance
    
    def __iter__(self):
        """Iterate through selected indices."""
        return iter(self.selected_indices)
    
    def __len__(self):
        """Get number of selected samples."""
        return len(self.selected_indices)
