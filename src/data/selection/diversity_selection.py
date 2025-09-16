"""
Diversity-based Selection
=========================

Selects diverse samples for robust training following:
- Sener & Savarese (2018): "Active Learning for Convolutional Neural Networks"
- Geifman & El-Yaniv (2017): "Deep Active Learning over the Long Tail"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Union
import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

class DiversitySelector:
    """
    Select diverse samples using various strategies.
    
    Implements diversity maximization from:
    - Kulesza & Taskar (2012): "Determinantal Point Processes for Machine Learning"
    """
    
    def __init__(
        self,
        method: str = "clustering",
        metric: str = "euclidean",
        n_clusters: Optional[int] = None
    ):
        """
        Initialize diversity selector.
        
        Args:
            method: Selection method (clustering, dpp, maxmin)
            metric: Distance metric
            n_clusters: Number of clusters for clustering method
        """
        self.method = method
        self.metric = metric
        self.n_clusters = n_clusters
        
        logger.info(f"Initialized diversity selector with {method} method")
    
    def select_diverse_subset(
        self,
        embeddings: Union[np.ndarray, torch.Tensor],
        n_samples: int,
        initial_indices: Optional[List[int]] = None
    ) -> List[int]:
        """
        Select diverse subset of samples.
        
        Args:
            embeddings: Feature embeddings
            n_samples: Number of samples to select
            initial_indices: Initially selected indices
            
        Returns:
            Selected indices
        """
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        if self.method == "clustering":
            indices = self._clustering_selection(embeddings, n_samples)
        elif self.method == "maxmin":
            indices = self._maxmin_selection(embeddings, n_samples, initial_indices)
        elif self.method == "dpp":
            indices = self._dpp_selection(embeddings, n_samples)
        else:
            # Random fallback
            indices = np.random.choice(len(embeddings), n_samples, replace=False).tolist()
        
        logger.info(f"Selected {len(indices)} diverse samples")
        
        return indices
    
    def _clustering_selection(
        self,
        embeddings: np.ndarray,
        n_samples: int
    ) -> List[int]:
        """
        Select samples using clustering.
        
        Following cluster-based sampling from:
        - Zhdanov (2019): "Diverse Mini-Batch Active Learning"
        """
        n_clusters = self.n_clusters or min(n_samples, int(np.sqrt(len(embeddings))))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        selected_indices = []
        
        # Select samples from each cluster
        samples_per_cluster = n_samples // n_clusters
        remaining = n_samples % n_clusters
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Select samples closest to cluster center
            center = kmeans.cluster_centers_[cluster_id]
            distances = np.linalg.norm(embeddings[cluster_indices] - center, axis=1)
            
            n_select = samples_per_cluster
            if cluster_id < remaining:
                n_select += 1
            
            n_select = min(n_select, len(cluster_indices))
            closest_indices = cluster_indices[np.argsort(distances)[:n_select]]
            
            selected_indices.extend(closest_indices.tolist())
        
        return selected_indices[:n_samples]
    
    def _maxmin_selection(
        self,
        embeddings: np.ndarray,
        n_samples: int,
        initial_indices: Optional[List[int]] = None
    ) -> List[int]:
        """
        MaxMin diversity selection.
        
        Implements greedy max-min diversity from:
        - Ravi & Larochelle (2018): "Optimization as a Model for Few-Shot Learning"
        """
        if initial_indices:
            selected = list(initial_indices)
        else:
            # Random first point
            selected = [np.random.randint(len(embeddings))]
        
        # Compute distance matrix
        if self.metric == "cosine":
            distances = 1 - cosine_similarity(embeddings)
        else:
            distances = euclidean_distances(embeddings)
        
        while len(selected) < n_samples:
            # Find point with maximum minimum distance to selected points
            min_distances = distances[selected].min(axis=0)
            min_distances[selected] = -np.inf  # Exclude already selected
            
            next_idx = np.argmax(min_distances)
            selected.append(next_idx)
        
        return selected
    
    def _dpp_selection(
        self,
        embeddings: np.ndarray,
        n_samples: int
    ) -> List[int]:
        """
        Determinantal Point Process selection.
        
        Simplified DPP following:
        - Kulesza & Taskar (2012): "Determinantal Point Processes"
        """
        # Compute similarity kernel
        if self.metric == "cosine":
            kernel = cosine_similarity(embeddings)
        else:
            # Convert distances to similarities
            distances = euclidean_distances(embeddings)
            kernel = np.exp(-distances / distances.mean())
        
        selected = []
        
        # Greedy MAP inference for DPP
        for _ in range(n_samples):
            if not selected:
                # First point: highest quality (diagonal of kernel)
                scores = np.diag(kernel)
            else:
                # Subsequent points: quality-diversity tradeoff
                subkernel = kernel[np.ix_(selected, selected)]
                
                try:
                    inv_subkernel = np.linalg.inv(subkernel + 1e-6 * np.eye(len(selected)))
                except:
                    inv_subkernel = np.eye(len(selected))
                
                scores = np.diag(kernel).copy()
                
                for i in range(len(embeddings)):
                    if i not in selected:
                        k_i = kernel[selected, i]
                        scores[i] -= k_i.T @ inv_subkernel @ k_i
            
            scores[selected] = -np.inf
            next_idx = np.argmax(scores)
            selected.append(next_idx)
        
        return selected
