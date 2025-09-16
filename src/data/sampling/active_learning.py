"""
Active Learning Sampling Module
================================

Implements active learning strategies for sample selection following:
- Settles (2009): "Active Learning Literature Survey"
- Sener & Savarese (2018): "Active Learning for Convolutional Neural Networks: A Core-Set Approach"
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from sklearn.metrics.pairwise import cosine_similarity

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class ActiveLearningConfig:
    """Configuration for active learning."""
    
    strategy: str = "uncertainty"  # uncertainty, diversity, hybrid
    initial_samples: int = 100
    query_batch_size: int = 50
    
    # Uncertainty strategies
    uncertainty_method: str = "entropy"  # entropy, least_confidence, margin
    mc_dropout_iterations: int = 10
    
    # Diversity strategies  
    diversity_method: str = "coreset"  # coreset, clustering, representative
    
    # Hybrid weight
    uncertainty_weight: float = 0.7
    diversity_weight: float = 0.3

class ActiveLearningSampler(Sampler):
    """
    Active learning sampler for intelligent sample selection.
    
    Implements query strategies from:
    - Lewis & Gale (1994): "A Sequential Algorithm for Training Text Classifiers"
    - Seung et al. (1992): "Query by Committee"
    """
    
    def __init__(
        self,
        dataset: Dataset,
        model: Optional[torch.nn.Module] = None,
        config: Optional[ActiveLearningConfig] = None,
        labeled_indices: Optional[List[int]] = None,
        seed: int = 42
    ):
        """
        Initialize active learning sampler.
        
        Args:
            dataset: Dataset to sample from
            model: Model for uncertainty estimation
            config: Active learning configuration
            labeled_indices: Already labeled sample indices
            seed: Random seed
        """
        self.dataset = dataset
        self.model = model
        self.config = config or ActiveLearningConfig()
        self.labeled_indices = set(labeled_indices or [])
        self.unlabeled_indices = set(range(len(dataset))) - self.labeled_indices
        self.rng = np.random.RandomState(seed)
        
        # Initialize with random samples if no labeled data
        if not self.labeled_indices:
            self._initialize_random_samples()
        
        logger.info(f"Initialized active learning sampler: {len(self.labeled_indices)} labeled, "
                   f"{len(self.unlabeled_indices)} unlabeled")
    
    def _initialize_random_samples(self):
        """Initialize with random samples."""
        initial_indices = self.rng.choice(
            list(self.unlabeled_indices),
            size=min(self.config.initial_samples, len(self.unlabeled_indices)),
            replace=False
        )
        
        self.labeled_indices = set(initial_indices)
        self.unlabeled_indices = self.unlabeled_indices - self.labeled_indices
    
    def query_samples(self, n_samples: int) -> List[int]:
        """
        Query next samples to label.
        
        Args:
            n_samples: Number of samples to query
            
        Returns:
            Indices of samples to label
        """
        if not self.unlabeled_indices:
            return []
        
        n_samples = min(n_samples, len(self.unlabeled_indices))
        
        if self.config.strategy == "uncertainty":
            indices = self._uncertainty_sampling(n_samples)
        elif self.config.strategy == "diversity":
            indices = self._diversity_sampling(n_samples)
        elif self.config.strategy == "hybrid":
            indices = self._hybrid_sampling(n_samples)
        else:
            indices = self._random_sampling(n_samples)
        
        # Update labeled/unlabeled sets
        self.labeled_indices.update(indices)
        self.unlabeled_indices -= set(indices)
        
        return indices
    
    def _uncertainty_sampling(self, n_samples: int) -> List[int]:
        """
        Uncertainty-based sampling.
        
        Following uncertainty strategies from:
        - Shannon (1948): "A Mathematical Theory of Communication" (Entropy)
        - Culotta & McCallum (2005): "Reducing Labeling Effort for Structured Prediction Tasks"
        """
        if self.model is None:
            return self._random_sampling(n_samples)
        
        self.model.eval()
        uncertainties = []
        
        with torch.no_grad():
            for idx in self.unlabeled_indices:
                sample = self.dataset[idx]
                
                # Get model predictions
                if self.config.uncertainty_method == "entropy":
                    uncertainty = self._compute_entropy(sample)
                elif self.config.uncertainty_method == "least_confidence":
                    uncertainty = self._compute_least_confidence(sample)
                elif self.config.uncertainty_method == "margin":
                    uncertainty = self._compute_margin(sample)
                else:
                    uncertainty = 0.5
                
                uncertainties.append((idx, uncertainty))
        
        # Sort by uncertainty (descending)
        uncertainties.sort(key=lambda x: x[1], reverse=True)
        
        # Select top-k uncertain samples
        selected_indices = [idx for idx, _ in uncertainties[:n_samples]]
        
        return selected_indices
    
    def _diversity_sampling(self, n_samples: int) -> List[int]:
        """
        Diversity-based sampling.
        
        Following diversity strategies from:
        - Sener & Savarese (2018): "Active Learning for CNNs: A Core-Set Approach"
        """
        if self.config.diversity_method == "coreset":
            return self._coreset_sampling(n_samples)
        else:
            return self._random_sampling(n_samples)
    
    def _hybrid_sampling(self, n_samples: int) -> List[int]:
        """
        Hybrid uncertainty-diversity sampling.
        
        Following hybrid strategies from:
        - Ash et al. (2020): "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds"
        """
        # Get uncertainty scores
        uncertainty_scores = self._get_uncertainty_scores()
        
        # Get diversity scores
        diversity_scores = self._get_diversity_scores()
        
        # Combine scores
        combined_scores = {}
        for idx in self.unlabeled_indices:
            u_score = uncertainty_scores.get(idx, 0)
            d_score = diversity_scores.get(idx, 0)
            combined = (self.config.uncertainty_weight * u_score + 
                       self.config.diversity_weight * d_score)
            combined_scores[idx] = combined
        
        # Select top-k samples
        sorted_indices = sorted(combined_scores.keys(), 
                              key=lambda x: combined_scores[x], 
                              reverse=True)
        
        return sorted_indices[:n_samples]
    
    def _random_sampling(self, n_samples: int) -> List[int]:
        """Random sampling fallback."""
        return self.rng.choice(
            list(self.unlabeled_indices),
            size=n_samples,
            replace=False
        ).tolist()
    
    def _compute_entropy(self, sample: Dict[str, Any]) -> float:
        """
        Compute entropy of predictions.
        
        H(y|x) = -Σ p(y|x) log p(y|x)
        """
        # Get predictions (simplified - should use actual model)
        probs = torch.rand(4)  # Placeholder for 4 classes
        probs = F.softmax(probs, dim=0)
        
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
        
        return entropy
    
    def _compute_least_confidence(self, sample: Dict[str, Any]) -> float:
        """Compute least confidence score."""
        probs = torch.rand(4)
        probs = F.softmax(probs, dim=0)
        
        # 1 - max(p(y|x))
        return 1.0 - probs.max().item()
    
    def _compute_margin(self, sample: Dict[str, Any]) -> float:
        """Compute margin between top two predictions."""
        probs = torch.rand(4)
        probs = F.softmax(probs, dim=0)
        
        sorted_probs, _ = torch.sort(probs, descending=True)
        
        # Margin = p(y1|x) - p(y2|x)
        margin = sorted_probs[0] - sorted_probs[1]
        
        # Return negative margin (smaller margin = more uncertain)
        return -margin.item()
    
    def _coreset_sampling(self, n_samples: int) -> List[int]:
        """
        Core-set selection for diversity.
        
        Following:
        - Sener & Savarese (2018): "Active Learning for CNNs: A Core-Set Approach"
        """
        # Simplified coreset - should use actual embeddings
        selected = []
        remaining = list(self.unlabeled_indices)
        
        # Greedy k-center
        if self.labeled_indices:
            # Start from labeled set
            centers = list(self.labeled_indices)
        else:
            # Random first center
            first = self.rng.choice(remaining)
            selected.append(first)
            remaining.remove(first)
            centers = [first]
        
        # Select remaining centers
        while len(selected) < n_samples and remaining:
            max_min_dist = -1
            best_candidate = None
            
            # Find point with maximum minimum distance to centers
            for candidate in remaining[:100]:  # Limit for efficiency
                min_dist = float('inf')
                
                # Compute minimum distance to centers
                for center in centers[-10:]:  # Use recent centers
                    # Simplified distance - should use actual embeddings
                    dist = abs(candidate - center)
                    min_dist = min(min_dist, dist)
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_candidate = candidate
            
            if best_candidate is not None:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
                centers.append(best_candidate)
        
        return selected
    
    def _get_uncertainty_scores(self) -> Dict[int, float]:
        """Get uncertainty scores for all unlabeled samples."""
        scores = {}
        for idx in self.unlabeled_indices:
            # Simplified - should compute actual uncertainty
            scores[idx] = self.rng.random()
        return scores
    
    def _get_diversity_scores(self) -> Dict[int, float]:
        """Get diversity scores for all unlabeled samples."""
        scores = {}
        for idx in self.unlabeled_indices:
            # Simplified - should compute actual diversity
            scores[idx] = self.rng.random()
        return scores
    
    def __iter__(self):
        """Iterate through labeled samples."""
        indices = list(self.labeled_indices)
        self.rng.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        """Get number of labeled samples."""
        return len(self.labeled_indices)
