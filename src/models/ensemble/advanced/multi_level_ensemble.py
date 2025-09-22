"""
Multi-Level Hierarchical Ensemble for AG News Classification
=============================================================

Implementation of multi-level ensemble that combines models hierarchically,
based on:
- Zhou (2012): "Ensemble Methods: Foundations and Algorithms"
- Sagi & Rokach (2018): "Ensemble Learning: A Survey"
- Dong et al. (2020): "A Survey on Ensemble Learning"

The multi-level ensemble organizes models in a hierarchy where:
- Level 1: Base models make initial predictions
- Level 2: Mid-level ensembles combine subsets of base models
- Level 3: Top-level ensemble combines mid-level predictions

Mathematical Foundation:
Level 1: h_i(x) for i in base_models
Level 2: g_j(x) = combine({h_i(x) | i in subset_j})
Level 3: f(x) = combine({g_j(x) | j in mid_ensembles})

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.models.ensemble.base_ensemble import BaseEnsemble
from src.models.ensemble.voting.soft_voting import SoftVotingEnsemble
from src.models.ensemble.stacking.stacking_classifier import StackingClassifier
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MultiLevelConfig:
    """Configuration for multi-level ensemble"""
    
    # Architecture configuration
    num_levels: int = 3
    models_per_group: int = 3  # Models per mid-level ensemble
    
    # Level strategies
    level1_strategy: str = "diverse"  # "diverse", "random", "clustered"
    level2_strategy: str = "voting"  # "voting", "stacking", "blending"
    level3_strategy: str = "stacking"  # "voting", "stacking", "meta"
    
    # Grouping configuration
    grouping_method: str = "performance"  # "performance", "diversity", "type"
    overlap_allowed: bool = False  # Allow models in multiple groups
    
    # Training configuration
    train_levels_separately: bool = True
    freeze_lower_levels: bool = True
    level_dropout: float = 0.0  # Dropout between levels
    
    # Combination weights
    use_weighted_combination: bool = True
    learn_combination_weights: bool = True
    
    # Regularization
    diversity_regularization: float = 0.01
    complexity_penalty: float = 0.001
    
    # Performance optimization
    prune_weak_paths: bool = True
    pruning_threshold: float = 0.6
    
    # Monitoring
    track_level_contributions: bool = True
    visualize_hierarchy: bool = False


class ModelGrouper:
    """
    Groups base models for mid-level ensembles.
    
    Implements various strategies for grouping models based on
    performance, diversity, or model characteristics.
    """
    
    def __init__(self, config: MultiLevelConfig):
        """
        Initialize model grouper.
        
        Args:
            config: Multi-level configuration
        """
        self.config = config
        
    def group_models(
        self,
        models: List[AGNewsBaseModel],
        validation_predictions: Optional[torch.Tensor] = None,
        validation_labels: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        Group models into subsets for mid-level ensembles.
        
        Args:
            models: List of base models
            validation_predictions: Validation predictions for grouping
            validation_labels: Validation labels
            
        Returns:
            List of model indices for each group
        """
        n_models = len(models)
        models_per_group = self.config.models_per_group
        
        if self.config.grouping_method == "performance":
            groups = self._group_by_performance(
                models, validation_predictions, validation_labels
            )
        elif self.config.grouping_method == "diversity":
            groups = self._group_by_diversity(
                models, validation_predictions
            )
        elif self.config.grouping_method == "type":
            groups = self._group_by_type(models)
        else:
            # Default: Sequential grouping
            groups = []
            for i in range(0, n_models, models_per_group):
                group = list(range(i, min(i + models_per_group, n_models)))
                if len(group) >= 2:  # Minimum 2 models per group
                    groups.append(group)
        
        logger.info(f"Created {len(groups)} model groups")
        return groups
    
    def _group_by_performance(
        self,
        models: List[AGNewsBaseModel],
        predictions: torch.Tensor,
        labels: torch.Tensor
    ) -> List[List[int]]:
        """Group models by performance tiers"""
        if predictions is None or labels is None:
            # Fallback to sequential grouping
            return self._sequential_grouping(len(models))
        
        # Calculate accuracy for each model
        accuracies = []
        for i in range(len(models)):
            pred_labels = predictions[i].argmax(dim=1)
            accuracy = (pred_labels == labels).float().mean().item()
            accuracies.append(accuracy)
        
        # Sort models by accuracy
        sorted_indices = np.argsort(accuracies)[::-1]
        
        # Create groups with mixed performance levels
        groups = []
        models_per_group = self.config.models_per_group
        
        # Stratified grouping: one from each performance tier
        n_tiers = models_per_group
        tier_size = len(models) // n_tiers
        
        for i in range(0, tier_size):
            group = []
            for tier in range(n_tiers):
                idx = tier * tier_size + i
                if idx < len(sorted_indices):
                    group.append(sorted_indices[idx])
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _group_by_diversity(
        self,
        models: List[AGNewsBaseModel],
        predictions: torch.Tensor
    ) -> List[List[int]]:
        """Group models by prediction diversity"""
        if predictions is None:
            return self._sequential_grouping(len(models))
        
        n_models = len(models)
        
        # Calculate pairwise disagreement matrix
        disagreement_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                # Calculate disagreement rate
                pred_i = predictions[i].argmax(dim=1)
                pred_j = predictions[j].argmax(dim=1)
                disagreement = (pred_i != pred_j).float().mean().item()
                disagreement_matrix[i, j] = disagreement
                disagreement_matrix[j, i] = disagreement
        
        # Greedy grouping to maximize diversity within groups
        groups = []
        used = set()
        
        while len(used) < n_models:
            # Start new group with unused model
            available = [i for i in range(n_models) if i not in used]
            if not available:
                break
            
            group = [available[0]]
            used.add(available[0])
            
            # Add diverse models to group
            while len(group) < self.config.models_per_group and len(used) < n_models:
                # Find model with maximum disagreement with current group
                max_disagreement = -1
                best_model = None
                
                for i in range(n_models):
                    if i not in used:
                        avg_disagreement = np.mean([disagreement_matrix[i, j] for j in group])
                        if avg_disagreement > max_disagreement:
                            max_disagreement = avg_disagreement
                            best_model = i
                
                if best_model is not None:
                    group.append(best_model)
                    used.add(best_model)
                else:
                    break
            
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _group_by_type(self, models: List[AGNewsBaseModel]) -> List[List[int]]:
        """Group models by their type/architecture"""
        model_types = defaultdict(list)
        
        for i, model in enumerate(models):
            model_type = model.__class__.__name__
            model_types[model_type].append(i)
        
        # Create mixed groups with different model types
        groups = []
        max_groups = max(len(indices) for indices in model_types.values())
        
        for i in range(max_groups):
            group = []
            for model_type, indices in model_types.items():
                if i < len(indices):
                    group.append(indices[i])
            if len(group) >= 2:
                groups.append(group)
        
        return groups
    
    def _sequential_grouping(self, n_models: int) -> List[List[int]]:
        """Simple sequential grouping"""
        groups = []
        models_per_group = self.config.models_per_group
        
        for i in range(0, n_models, models_per_group):
            group = list(range(i, min(i + models_per_group, n_models)))
            if len(group) >= 2:
                groups.append(group)
        
        return groups


class HierarchicalCombiner(nn.Module):
    """
    Combines predictions across hierarchical levels.
    
    Implements the combination logic for multi-level ensemble,
    managing information flow between levels.
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        """
        Initialize hierarchical combiner.
        
        Args:
            num_inputs: Number of input predictions
            num_classes: Number of output classes
            hidden_dim: Hidden dimension for combination network
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_classes = num_classes
        
        # Attention mechanism for weighted combination
        self.attention = nn.Sequential(
            nn.Linear(num_inputs * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_inputs),
            nn.Softmax(dim=1)
        )
        
        # Non-linear transformation
        self.transform = nn.Sequential(
            nn.Linear(num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Residual connection weight
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(
        self,
        predictions: torch.Tensor,
        return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Combine predictions hierarchically.
        
        Args:
            predictions: Input predictions [batch_size, num_inputs, num_classes]
            return_weights: Whether to return attention weights
            
        Returns:
            Combined predictions and optionally attention weights
        """
        batch_size = predictions.shape[0]
        
        # Calculate attention weights
        flat_predictions = predictions.view(batch_size, -1)
        attention_weights = self.attention(flat_predictions)
        
        # Apply attention
        attention_weights = attention_weights.unsqueeze(2)
        weighted_predictions = (predictions * attention_weights).sum(dim=1)
        
        # Non-linear transformation
        transformed = self.transform(weighted_predictions)
        
        # Residual connection with mean prediction
        mean_prediction = predictions.mean(dim=1)
        output = transformed + self.residual_weight * mean_prediction
        
        if return_weights:
            return output, attention_weights.squeeze(2)
        return output


@MODELS.register("multi_level_ensemble")
class MultiLevelEnsemble(BaseEnsemble):
    """
    Multi-Level Hierarchical Ensemble.
    
    Organizes models in a multi-level hierarchy where predictions
    flow from base models through intermediate ensembles to a
    final meta-ensemble, allowing for complex interaction patterns
    and improved generalization.
    
    The architecture enables:
    1. Specialization at different levels
    2. Error correction through multiple stages
    3. Automatic feature extraction from predictions
    4. Reduced overfitting through hierarchical regularization
    """
    
    def __init__(
        self,
        models: List[AGNewsBaseModel],
        config: Optional[MultiLevelConfig] = None
    ):
        """
        Initialize multi-level ensemble.
        
        Args:
            models: List of base models (Level 1)
            config: Multi-level configuration
        """
        super().__init__(models)
        
        self.config = config or MultiLevelConfig()
        self.base_models = models  # Level 1
        
        # Initialize model grouper
        self.grouper = ModelGrouper(self.config)
        
        # Create model groups
        self.model_groups = None  # Will be created during first forward
        
        # Initialize mid-level ensembles (Level 2)
        self.mid_ensembles = nn.ModuleList()
        
        # Initialize top-level combiner (Level 3)
        self.top_combiner = None
        
        # Statistics tracking
        self.level_stats = {
            'level1_outputs': [],
            'level2_outputs': [],
            'level3_outputs': [],
            'pruned_paths': set()
        }
        
        logger.info(
            f"Initialized Multi-Level Ensemble with {len(models)} base models, "
            f"{self.config.num_levels} levels"
        )
    
    def _create_hierarchy(
        self,
        validation_predictions: Optional[torch.Tensor] = None,
        validation_labels: Optional[torch.Tensor] = None
    ):
        """
        Create the hierarchical structure.
        
        Args:
            validation_predictions: Validation predictions for grouping
            validation_labels: Validation labels
        """
        # Create model groups
        self.model_groups = self.grouper.group_models(
            self.base_models,
            validation_predictions,
            validation_labels
        )
        
        # Create mid-level ensembles
        for group_indices in self.model_groups:
            group_models = [self.base_models[i] for i in group_indices]
            
            if self.config.level2_strategy == "voting":
                ensemble = SoftVotingEnsemble(group_models)
            elif self.config.level2_strategy == "stacking":
                ensemble = StackingClassifier(group_models)
            else:
                # Default to voting
                ensemble = SoftVotingEnsemble(group_models)
            
            self.mid_ensembles.append(ensemble)
        
        # Create top-level combiner
        num_mid_ensembles = len(self.mid_ensembles)
        self.top_combiner = HierarchicalCombiner(
            num_inputs=num_mid_ensembles,
            num_classes=4,  # AG News classes
            dropout=self.config.level_dropout
        )
        
        logger.info(
            f"Created hierarchy: {len(self.model_groups)} mid-level ensembles"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_all_levels: bool = False,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through multi-level ensemble.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            return_all_levels: Return predictions from all levels
            **kwargs: Additional arguments
            
        Returns:
            Hierarchical ensemble predictions
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Create hierarchy if not exists
        if self.model_groups is None:
            self._create_hierarchy()
        
        # Level 1: Base model predictions
        level1_predictions = []
        
        for model in self.base_models:
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                
                if hasattr(outputs, 'logits'):
                    probs = F.softmax(outputs.logits, dim=-1)
                else:
                    probs = torch.ones(batch_size, 4, device=device) / 4
                
                level1_predictions.append(probs)
        
        # Level 2: Mid-level ensemble predictions
        level2_predictions = []
        
        for i, ensemble in enumerate(self.mid_ensembles):
            # Get predictions from this ensemble's models
            group_indices = self.model_groups[i]
            group_predictions = [level1_predictions[j] for j in group_indices]
            
            # Combine using mid-level strategy
            if self.config.level2_strategy == "voting":
                # Simple average
                mid_pred = torch.stack(group_predictions).mean(dim=0)
            else:
                # Use ensemble's forward method
                with torch.no_grad() if not self.training else torch.enable_grad():
                    ensemble_output = ensemble(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    mid_pred = F.softmax(ensemble_output.logits, dim=-1)
            
            level2_predictions.append(mid_pred)
        
        # Level 3: Top-level combination
        level2_tensor = torch.stack(level2_predictions, dim=1)
        final_predictions = self.top_combiner(level2_tensor)
        
        # Convert to logits
        ensemble_logits = torch.log(final_predictions + 1e-8)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(ensemble_logits, labels)
            
            # Add diversity regularization
            if self.config.diversity_regularization > 0:
                diversity_loss = self._calculate_diversity_loss(level2_predictions)
                loss = loss + self.config.diversity_regularization * diversity_loss
        
        # Update statistics
        self._update_statistics(level1_predictions, level2_predictions, final_predictions)
        
        # Prepare metadata
        metadata = {
            'num_levels': self.config.num_levels,
            'num_mid_ensembles': len(self.mid_ensembles),
            'hierarchy_depth': 3
        }
        
        if return_all_levels:
            metadata['level1_predictions'] = torch.stack(level1_predictions)
            metadata['level2_predictions'] = torch.stack(level2_predictions)
            metadata['level3_predictions'] = final_predictions
        
        return ModelOutputs(
            loss=loss,
            logits=ensemble_logits,
            metadata=metadata
        )
    
    def _calculate_diversity_loss(
        self,
        predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """Calculate diversity loss to encourage diverse predictions"""
        # Stack predictions
        preds = torch.stack(predictions)
        
        # Calculate pairwise correlation
        n_models = len(predictions)
        correlation_sum = 0
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                # Correlation between predictions
                corr = torch.mean(predictions[i] * predictions[j])
                correlation_sum += corr
        
        # Normalize
        num_pairs = n_models * (n_models - 1) / 2
        avg_correlation = correlation_sum / num_pairs
        
        # Loss increases with correlation (encourage diversity)
        return avg_correlation
    
    def _update_statistics(
        self,
        level1_preds: List[torch.Tensor],
        level2_preds: List[torch.Tensor],
        level3_preds: torch.Tensor
    ):
        """Update level-wise statistics"""
        # Store recent predictions
        max_history = 100
        
        self.level_stats['level1_outputs'].append(torch.stack(level1_preds).mean().item())
        self.level_stats['level2_outputs'].append(torch.stack(level2_preds).mean().item())
        self.level_stats['level3_outputs'].append(level3_preds.mean().item())
        
        # Limit history size
        for key in ['level1_outputs', 'level2_outputs', 'level3_outputs']:
            if len(self.level_stats[key]) > max_history:
                self.level_stats[key] = self.level_stats[key][-max_history:]
    
    def prune_weak_paths(self, threshold: float = 0.6):
        """
        Prune weak prediction paths based on performance.
        
        Args:
            threshold: Performance threshold for pruning
        """
        if not self.config.prune_weak_paths:
            return
        
        # Analyze path contributions
        # This would require validation data to properly implement
        # For now, log the intention
        logger.info(f"Pruning paths with performance below {threshold}")
    
    def visualize_hierarchy(self) -> Dict[str, Any]:
        """
        Generate visualization data for the hierarchy.
        
        Returns:
            Dictionary with hierarchy structure
        """
        return {
            'num_base_models': len(self.base_models),
            'model_groups': self.model_groups,
            'num_mid_ensembles': len(self.mid_ensembles),
            'level_stats': {
                'level1_mean': np.mean(self.level_stats['level1_outputs'][-10:])
                              if self.level_stats['level1_outputs'] else 0,
                'level2_mean': np.mean(self.level_stats['level2_outputs'][-10:])
                              if self.level_stats['level2_outputs'] else 0,
                'level3_mean': np.mean(self.level_stats['level3_outputs'][-10:])
                              if self.level_stats['level3_outputs'] else 0
            }
        }


# Export classes
__all__ = [
    'MultiLevelConfig',
    'ModelGrouper',
    'HierarchicalCombiner',
    'MultiLevelEnsemble'
]
