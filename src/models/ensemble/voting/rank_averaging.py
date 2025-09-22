"""
Rank-Based Averaging Ensemble for AG News Classification
=========================================================

Implementation of rank-based ensemble methods that combine predictions based on
ranking rather than raw probabilities, following:
- Melville et al. (2009): "Creating Diversity in Ensembles Using Artificial Data"
- Brodersen et al. (2010): "The Balanced Accuracy and Its Posterior Distribution"
- Bonab & Can (2016): "A Theoretical Framework on the Ideal Number of Classifiers"

Rank-based methods are more robust to calibration differences between models
and can better handle models with different output scales.

Mathematical Foundation:
For predictions p_i(c|x) from model i for class c:
rank_i(c) = rank of class c in model i's predictions
final_score(c) = aggregate_function(rank_1(c), ..., rank_n(c))

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import stats
from scipy.stats import rankdata

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.models.ensemble.base_ensemble import BaseEnsemble
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RankAggregationMethod(Enum):
    """Methods for aggregating ranks across models"""
    BORDA_COUNT = "borda"  # Sum of ranks
    MEDIAN_RANK = "median"  # Median of ranks
    GEOMETRIC_MEAN = "geometric"  # Geometric mean of ranks
    HARMONIC_MEAN = "harmonic"  # Harmonic mean of ranks
    TRIMMED_MEAN = "trimmed"  # Trimmed mean (remove outliers)
    WEIGHTED_BORDA = "weighted_borda"  # Weighted sum of ranks
    KEMENY_YOUNG = "kemeny"  # Kemeny-Young method
    SCHULZE = "schulze"  # Schulze method


@dataclass
class RankAveragingConfig:
    """Configuration for rank-based averaging ensemble"""
    
    # Ranking configuration
    aggregation_method: RankAggregationMethod = RankAggregationMethod.BORDA_COUNT
    rank_type: str = "ordinal"  # "ordinal", "fractional", "dense"
    higher_is_better: bool = True  # Whether higher scores are better
    
    # Weighting
    use_model_weights: bool = True
    model_weights: Optional[List[float]] = None
    learn_weights: bool = False
    
    # Normalization
    normalize_ranks: bool = True
    rank_scaling: str = "linear"  # "linear", "exponential", "logarithmic"
    
    # Robustness
    trim_percentage: float = 0.1  # For trimmed mean
    use_confidence_weighting: bool = False
    min_confidence: float = 0.1
    
    # Advanced options
    use_pairwise_preferences: bool = False  # For Kemeny-Young, Schulze
    temperature: float = 1.0  # Temperature for softmax on ranks
    
    # Computational
    cache_ranks: bool = True
    parallel_processing: bool = True


class RankTransformer:
    """
    Transform probabilities to ranks with various strategies.
    
    Handles different ranking schemes and normalization approaches
    to ensure fair comparison across models.
    """
    
    def __init__(self, config: RankAveragingConfig):
        """
        Initialize rank transformer.
        
        Args:
            config: Rank averaging configuration
        """
        self.config = config
        
    def transform_to_ranks(
        self,
        scores: torch.Tensor,
        method: str = "ordinal"
    ) -> torch.Tensor:
        """
        Transform scores to ranks.
        
        Args:
            scores: Prediction scores [batch_size, num_classes]
            method: Ranking method
            
        Returns:
            Ranks tensor [batch_size, num_classes]
        """
        batch_size, num_classes = scores.shape
        ranks = torch.zeros_like(scores)
        
        for i in range(batch_size):
            if method == "ordinal":
                # Standard ranking (1, 2, 3, ...)
                ranks[i] = self._ordinal_rank(scores[i])
            elif method == "fractional":
                # Average rank for ties
                ranks[i] = self._fractional_rank(scores[i])
            elif method == "dense":
                # Dense ranking (1, 1, 2, ...)
                ranks[i] = self._dense_rank(scores[i])
            else:
                ranks[i] = self._ordinal_rank(scores[i])
        
        # Normalize if configured
        if self.config.normalize_ranks:
            ranks = self._normalize_ranks(ranks, num_classes)
        
        return ranks
    
    def _ordinal_rank(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute ordinal ranks"""
        if self.config.higher_is_better:
            # Higher scores get lower ranks (1 is best)
            return (-scores).argsort().argsort().float() + 1
        else:
            return scores.argsort().argsort().float() + 1
    
    def _fractional_rank(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute fractional ranks (average for ties)"""
        scores_np = scores.cpu().numpy()
        method = 'max' if self.config.higher_is_better else 'min'
        ranks = rankdata(-scores_np if self.config.higher_is_better else scores_np, method='average')
        return torch.tensor(ranks, device=scores.device, dtype=torch.float32)
    
    def _dense_rank(self, scores: torch.Tensor) -> torch.Tensor:
        """Compute dense ranks"""
        scores_np = scores.cpu().numpy()
        ranks = rankdata(-scores_np if self.config.higher_is_better else scores_np, method='dense')
        return torch.tensor(ranks, device=scores.device, dtype=torch.float32)
    
    def _normalize_ranks(
        self,
        ranks: torch.Tensor,
        num_classes: int
    ) -> torch.Tensor:
        """
        Normalize ranks to [0, 1] range.
        
        Args:
            ranks: Raw ranks
            num_classes: Number of classes
            
        Returns:
            Normalized ranks
        """
        if self.config.rank_scaling == "linear":
            # Linear normalization
            normalized = (num_classes - ranks) / (num_classes - 1)
        elif self.config.rank_scaling == "exponential":
            # Exponential decay
            normalized = torch.exp(-(ranks - 1) / self.config.temperature)
        elif self.config.rank_scaling == "logarithmic":
            # Logarithmic scaling
            normalized = 1.0 / torch.log(ranks + 1)
        else:
            normalized = ranks
        
        return normalized


class RankAggregator:
    """
    Aggregate ranks from multiple models using various voting schemes.
    
    Implements different rank aggregation methods from social choice theory
    and ensemble learning literature.
    """
    
    def __init__(self, config: RankAveragingConfig):
        """
        Initialize rank aggregator.
        
        Args:
            config: Configuration
        """
        self.config = config
        self.aggregation_cache = {}
        
    def aggregate(
        self,
        ranks: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate ranks from multiple models.
        
        Args:
            ranks: Ranks from all models [num_models, batch_size, num_classes]
            weights: Model weights [num_models]
            
        Returns:
            Aggregated scores [batch_size, num_classes]
        """
        method = self.config.aggregation_method
        
        if method == RankAggregationMethod.BORDA_COUNT:
            return self._borda_count(ranks, weights)
        elif method == RankAggregationMethod.MEDIAN_RANK:
            return self._median_rank(ranks)
        elif method == RankAggregationMethod.GEOMETRIC_MEAN:
            return self._geometric_mean_rank(ranks, weights)
        elif method == RankAggregationMethod.HARMONIC_MEAN:
            return self._harmonic_mean_rank(ranks, weights)
        elif method == RankAggregationMethod.TRIMMED_MEAN:
            return self._trimmed_mean_rank(ranks)
        elif method == RankAggregationMethod.WEIGHTED_BORDA:
            return self._weighted_borda(ranks, weights)
        elif method == RankAggregationMethod.KEMENY_YOUNG:
            return self._kemeny_young(ranks)
        elif method == RankAggregationMethod.SCHULZE:
            return self._schulze_method(ranks)
        else:
            return self._borda_count(ranks, weights)
    
    def _borda_count(
        self,
        ranks: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Borda count aggregation.
        
        Each position gets points: n-1 for 1st, n-2 for 2nd, etc.
        """
        num_models, batch_size, num_classes = ranks.shape
        
        # Convert ranks to Borda scores
        borda_scores = num_classes - ranks
        
        if weights is not None:
            # Apply weights
            weights = weights.view(-1, 1, 1)
            borda_scores = borda_scores * weights
        
        # Sum across models
        aggregated = borda_scores.sum(dim=0)
        
        return aggregated
    
    def _median_rank(self, ranks: torch.Tensor) -> torch.Tensor:
        """Median rank aggregation"""
        # Take median across models
        median_ranks = torch.median(ranks, dim=0)[0]
        
        # Convert to scores (lower rank = higher score)
        num_classes = ranks.shape[2]
        scores = num_classes - median_ranks
        
        return scores
    
    def _geometric_mean_rank(
        self,
        ranks: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Geometric mean of ranks"""
        # Add small epsilon to avoid log(0)
        epsilon = 1e-8
        
        if weights is not None:
            # Weighted geometric mean
            weights = weights.view(-1, 1, 1)
            log_ranks = torch.log(ranks + epsilon) * weights
            geometric_mean = torch.exp(log_ranks.sum(dim=0) / weights.sum())
        else:
            # Unweighted geometric mean
            log_ranks = torch.log(ranks + epsilon)
            geometric_mean = torch.exp(log_ranks.mean(dim=0))
        
        # Convert to scores
        num_classes = ranks.shape[2]
        scores = num_classes - geometric_mean
        
        return scores
    
    def _harmonic_mean_rank(
        self,
        ranks: torch.Tensor,
        weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Harmonic mean of ranks"""
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        
        if weights is not None:
            weights = weights.view(-1, 1, 1)
            reciprocal_ranks = weights / (ranks + epsilon)
            harmonic_mean = weights.sum() / reciprocal_ranks.sum(dim=0)
        else:
            reciprocal_ranks = 1.0 / (ranks + epsilon)
            harmonic_mean = ranks.shape[0] / reciprocal_ranks.sum(dim=0)
        
        # Convert to scores
        num_classes = ranks.shape[2]
        scores = num_classes - harmonic_mean
        
        return scores
    
    def _trimmed_mean_rank(self, ranks: torch.Tensor) -> torch.Tensor:
        """Trimmed mean (remove outliers)"""
        num_models = ranks.shape[0]
        trim_count = int(num_models * self.config.trim_percentage)
        
        if trim_count > 0 and num_models > 2 * trim_count:
            # Sort ranks along model dimension
            sorted_ranks, _ = torch.sort(ranks, dim=0)
            
            # Remove top and bottom trim_count models
            trimmed = sorted_ranks[trim_count:-trim_count]
            
            # Take mean of remaining
            mean_ranks = trimmed.mean(dim=0)
        else:
            # Not enough models to trim
            mean_ranks = ranks.mean(dim=0)
        
        # Convert to scores
        num_classes = ranks.shape[2]
        scores = num_classes - mean_ranks
        
        return scores
    
    def _weighted_borda(
        self,
        ranks: torch.Tensor,
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Weighted Borda count with confidence adjustment"""
        # Standard Borda scores
        num_classes = ranks.shape[2]
        borda_scores = num_classes - ranks
        
        # Apply model weights
        if weights is not None:
            weights = weights.view(-1, 1, 1)
            
            # Optional: Adjust weights by confidence
            if self.config.use_confidence_weighting:
                # Calculate confidence as spread of ranks
                rank_std = ranks.std(dim=2, keepdim=True)
                confidence = 1.0 / (1.0 + rank_std)
                weights = weights * confidence
            
            borda_scores = borda_scores * weights
        
        # Aggregate
        aggregated = borda_scores.sum(dim=0)
        
        return aggregated
    
    def _kemeny_young(self, ranks: torch.Tensor) -> torch.Tensor:
        """
        Kemeny-Young method (computationally expensive).
        
        Finds the ranking that minimizes disagreement with all input rankings.
        """
        # Simplified implementation for efficiency
        # Full K-Y is NP-hard for large number of alternatives
        
        # Use Borda count as approximation
        logger.warning("Kemeny-Young using Borda approximation for efficiency")
        return self._borda_count(ranks)
    
    def _schulze_method(self, ranks: torch.Tensor) -> torch.Tensor:
        """
        Schulze method based on pairwise preferences.
        
        Constructs pairwise preference matrix and finds strongest paths.
        """
        num_models, batch_size, num_classes = ranks.shape
        
        # Build pairwise preference matrix
        preferences = torch.zeros(batch_size, num_classes, num_classes, device=ranks.device)
        
        for b in range(batch_size):
            for i in range(num_classes):
                for j in range(num_classes):
                    if i != j:
                        # Count how many models prefer i to j
                        prefer_i = (ranks[:, b, i] < ranks[:, b, j]).sum()
                        preferences[b, i, j] = prefer_i
        
        # Find strongest paths (simplified)
        # Full Schulze requires Floyd-Warshall algorithm
        scores = preferences.sum(dim=2) - preferences.sum(dim=1)
        
        return scores


@MODELS.register("rank_averaging")
class RankAveragingEnsemble(BaseEnsemble):
    """
    Rank-Based Averaging Ensemble Model.
    
    Combines predictions from multiple models using rank-based aggregation,
    which is more robust to calibration differences and outliers than
    probability averaging.
    
    The ensemble supports multiple rank aggregation methods from
    social choice theory and can learn optimal aggregation weights.
    """
    
    def __init__(
        self,
        models: List[AGNewsBaseModel],
        config: Optional[RankAveragingConfig] = None
    ):
        """
        Initialize rank averaging ensemble.
        
        Args:
            models: List of base models
            config: Ensemble configuration
        """
        super().__init__(models)
        
        self.config = config or RankAveragingConfig()
        self.n_models = len(models)
        
        # Initialize components
        self.rank_transformer = RankTransformer(self.config)
        self.rank_aggregator = RankAggregator(self.config)
        
        # Initialize model weights
        self._init_weights()
        
        # Cache for storing ranks
        if self.config.cache_ranks:
            self.rank_cache = {}
        
        # Statistics tracking
        self.aggregation_stats = {
            'rank_correlations': [],
            'agreement_scores': [],
            'confidence_scores': []
        }
        
        logger.info(
            f"Initialized Rank Averaging Ensemble with {self.n_models} models "
            f"using {self.config.aggregation_method.value} aggregation"
        )
    
    def _init_weights(self):
        """Initialize model weights"""
        if self.config.model_weights is not None:
            # Use provided weights
            weights = torch.tensor(self.config.model_weights)
        else:
            # Initialize uniform weights
            weights = torch.ones(self.n_models) / self.n_models
        
        if self.config.learn_weights:
            self.model_weights = nn.Parameter(weights)
        else:
            self.register_buffer('model_weights', weights)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_all_ranks: bool = False,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass with rank-based aggregation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            return_all_ranks: Whether to return individual model ranks
            **kwargs: Additional arguments
            
        Returns:
            Ensemble predictions with rank-based aggregation
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Collect predictions from all models
        all_predictions = []
        all_logits = []
        
        for i, model in enumerate(self.models):
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    # Convert to probabilities for ranking
                    probs = F.softmax(logits, dim=-1)
                else:
                    # Fallback
                    probs = torch.ones(batch_size, 4, device=device) / 4
                    logits = torch.zeros(batch_size, 4, device=device)
                
                all_predictions.append(probs)
                all_logits.append(logits)
        
        # Stack predictions
        all_predictions = torch.stack(all_predictions)  # [n_models, batch_size, n_classes]
        
        # Transform to ranks
        all_ranks = []
        for i in range(self.n_models):
            ranks = self.rank_transformer.transform_to_ranks(
                all_predictions[i],
                method=self.config.rank_type
            )
            all_ranks.append(ranks)
        
        all_ranks = torch.stack(all_ranks)  # [n_models, batch_size, n_classes]
        
        # Get model weights
        weights = None
        if self.config.use_model_weights:
            weights = F.softmax(self.model_weights, dim=0) if self.config.learn_weights else self.model_weights
        
        # Aggregate ranks
        aggregated_scores = self.rank_aggregator.aggregate(all_ranks, weights)
        
        # Convert aggregated scores to probabilities
        ensemble_probs = F.softmax(aggregated_scores / self.config.temperature, dim=-1)
        
        # Convert to logits
        ensemble_logits = torch.log(ensemble_probs + 1e-8)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(ensemble_logits, labels)
        
        # Update statistics
        self._update_statistics(all_ranks, all_predictions)
        
        # Prepare metadata
        metadata = {
            'aggregation_method': self.config.aggregation_method.value,
            'model_weights': weights.detach().cpu().numpy() if weights is not None else None,
            'rank_correlation': self._calculate_rank_correlation(all_ranks),
            'rank_agreement': self._calculate_rank_agreement(all_ranks)
        }
        
        if return_all_ranks:
            metadata['all_ranks'] = all_ranks.detach().cpu().numpy()
        
        return ModelOutputs(
            loss=loss,
            logits=ensemble_logits,
            metadata=metadata
        )
    
    def _update_statistics(
        self,
        ranks: torch.Tensor,
        predictions: torch.Tensor
    ):
        """Update ensemble statistics"""
        # Calculate rank correlation
        correlation = self._calculate_rank_correlation(ranks)
        self.aggregation_stats['rank_correlations'].append(correlation)
        
        # Calculate agreement
        agreement = self._calculate_rank_agreement(ranks)
        self.aggregation_stats['agreement_scores'].append(agreement)
        
        # Calculate confidence
        confidence = predictions.max(dim=2)[0].mean().item()
        self.aggregation_stats['confidence_scores'].append(confidence)
    
    def _calculate_rank_correlation(self, ranks: torch.Tensor) -> float:
        """Calculate average Spearman correlation between model rankings"""
        n_models = ranks.shape[0]
        correlations = []
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Calculate correlation for each sample
                for b in range(ranks.shape[1]):
                    corr = stats.spearmanr(
                        ranks[i, b].cpu().numpy(),
                        ranks[j, b].cpu().numpy()
                    )[0]
                    if not np.isnan(corr):
                        correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_rank_agreement(self, ranks: torch.Tensor) -> float:
        """Calculate agreement on top-ranked class"""
        # Get top-ranked class for each model
        top_ranked = ranks.argmin(dim=2)  # Lower rank = better
        
        # Calculate agreement
        mode_result = torch.mode(top_ranked, dim=0)
        agreement = (top_ranked == mode_result.values.unsqueeze(0)).float().mean()
        
        return agreement.item()
    
    def get_rank_analysis(self) -> Dict[str, Any]:
        """Get detailed analysis of rank aggregation"""
        if not self.aggregation_stats['rank_correlations']:
            return {}
        
        return {
            'avg_rank_correlation': np.mean(self.aggregation_stats['rank_correlations']),
            'avg_agreement': np.mean(self.aggregation_stats['agreement_scores']),
            'avg_confidence': np.mean(self.aggregation_stats['confidence_scores']),
            'correlation_std': np.std(self.aggregation_stats['rank_correlations']),
            'agreement_std': np.std(self.aggregation_stats['agreement_scores'])
        }
