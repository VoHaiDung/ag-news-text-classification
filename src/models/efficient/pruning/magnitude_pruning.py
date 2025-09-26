"""
Magnitude-based Neural Network Pruning Implementation
======================================================

This module implements magnitude-based pruning following methodologies from:
- Han et al. (2015): "Learning both Weights and Connections for Efficient Neural Networks"
- Frankle & Carbin (2019): "The Lottery Ticket Hypothesis"
- Louizos et al. (2018): "Learning Sparse Neural Networks through L0 Regularization"
- Renda et al. (2020): "Comparing Rewinding and Fine-tuning in Neural Network Pruning"

Mathematical Foundation:
Magnitude pruning removes weights w where |w| < τ, with threshold τ determined by:
- Global: τ = percentile(|W|, sparsity * 100)
- Layer-wise: τ_l = percentile(|W_l|, sparsity * 100) for each layer l
- Structured: Remove entire neurons/channels based on Σ|w|

The pruning objective minimizes:
L_pruned = L_task(f(x; W ⊙ M)) + λ||W ⊙ M||_0
where M is the binary mask, ⊙ is element-wise multiplication

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class PruningMethod(Enum):
    """Enumeration of pruning methods."""
    MAGNITUDE = "magnitude"
    RANDOM = "random"
    L1 = "l1"
    L2 = "l2"
    GRADIENT = "gradient"
    TAYLOR = "taylor"
    STRUCTURED = "structured"


class PruningSchedule(Enum):
    """Enumeration of pruning schedules."""
    ONESHOT = "oneshot"
    ITERATIVE = "iterative"
    GRADUAL = "gradual"
    POLYNOMIAL = "polynomial"
    EXPONENTIAL = "exponential"


@dataclass
class PruningConfig:
    """Configuration for magnitude pruning."""
    
    # Pruning parameters
    sparsity: float = 0.5  # Target sparsity level
    method: PruningMethod = PruningMethod.MAGNITUDE
    structured: bool = False  # Structured vs unstructured pruning
    
    # Schedule
    schedule: PruningSchedule = PruningSchedule.ONESHOT
    start_epoch: int = 0
    end_epoch: int = 10
    frequency: int = 1  # Pruning frequency in epochs
    
    # Granularity
    global_pruning: bool = False  # Global vs layer-wise
    
    # Fine-tuning
    fine_tune_epochs: int = 10
    rewinding: bool = False  # Lottery ticket rewinding
    
    # Layer selection
    exclude_layers: List[str] = None
    include_layers: List[str] = None
    
    # Initialization
    init_sparsity: float = 0.0
    final_sparsity: float = 0.9


class MagnitudePruner:
    """
    Magnitude-based pruning for neural networks.
    
    Implements various pruning strategies based on weight magnitudes,
    following the principle that smaller weights contribute less to
    the network's function and can be removed.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[PruningConfig] = None
    ):
        """
        Initialize magnitude pruner.
        
        Args:
            model: Model to prune
            config: Pruning configuration
        """
        self.model = model
        self.config = config or PruningConfig()
        
        # Initialize masks
        self.masks = {}
        self.original_weights = {}
        
        # Track pruning history
        self.pruning_history = {
            'sparsity': [],
            'accuracy': [],
            'loss': []
        }
        
        # Identify prunable layers
        self.prunable_layers = self._identify_prunable_layers()
        
        logger.info(
            f"Initialized MagnitudePruner with {len(self.prunable_layers)} "
            f"prunable layers, target sparsity: {self.config.sparsity}"
        )
    
    def _identify_prunable_layers(self) -> Dict[str, nn.Module]:
        """
        Identify layers that can be pruned.
        
        Returns:
            Dictionary of prunable layers
        """
        prunable_layers = {}
        
        for name, module in self.model.named_modules():
            # Check if layer should be excluded
            if self.config.exclude_layers and name in self.config.exclude_layers:
                continue
            
            # Check if layer should be included
            if self.config.include_layers and name not in self.config.include_layers:
                continue
            
            # Check if layer is prunable (has weight parameter)
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                prunable_layers[name] = module
                logger.debug(f"Layer {name} marked as prunable")
        
        return prunable_layers
    
    def compute_importance_scores(
        self,
        layer: nn.Module,
        method: PruningMethod = PruningMethod.MAGNITUDE
    ) -> torch.Tensor:
        """
        Compute importance scores for weights.
        
        Implements various importance metrics following:
        - Molchanov et al. (2017): "Variational Dropout Sparsifies DNNs"
        
        Args:
            layer: Layer to compute scores for
            method: Method for computing importance
            
        Returns:
            Importance scores tensor
        """
        weight = layer.weight.data
        
        if method == PruningMethod.MAGNITUDE:
            # L1 magnitude
            scores = torch.abs(weight)
            
        elif method == PruningMethod.L1:
            # L1 norm per neuron/filter
            if len(weight.shape) == 4:  # Conv2d
                scores = torch.sum(torch.abs(weight), dim=(1, 2, 3))
            elif len(weight.shape) == 2:  # Linear
                scores = torch.sum(torch.abs(weight), dim=1)
            else:
                scores = torch.abs(weight)
                
        elif method == PruningMethod.L2:
            # L2 norm
            if len(weight.shape) == 4:  # Conv2d
                scores = torch.sqrt(torch.sum(weight ** 2, dim=(1, 2, 3)))
            elif len(weight.shape) == 2:  # Linear
                scores = torch.sqrt(torch.sum(weight ** 2, dim=1))
            else:
                scores = weight ** 2
                
        elif method == PruningMethod.GRADIENT:
            # Gradient-based importance (requires gradients)
            if layer.weight.grad is not None:
                scores = torch.abs(weight * layer.weight.grad)
            else:
                logger.warning("No gradients available, falling back to magnitude")
                scores = torch.abs(weight)
                
        elif method == PruningMethod.TAYLOR:
            # First-order Taylor expansion importance
            if layer.weight.grad is not None:
                scores = torch.abs(weight * layer.weight.grad)
                # Second-order approximation if Hessian available
                if hasattr(layer, 'weight_hessian'):
                    scores += 0.5 * torch.abs(weight ** 2 * layer.weight_hessian)
            else:
                scores = torch.abs(weight)
                
        elif method == PruningMethod.RANDOM:
            # Random scores for comparison
            scores = torch.rand_like(weight)
            
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        return scores
    
    def compute_threshold(
        self,
        scores: torch.Tensor,
        sparsity: float
    ) -> float:
        """
        Compute pruning threshold for given sparsity.
        
        Args:
            scores: Importance scores
            sparsity: Target sparsity level
            
        Returns:
            Threshold value
        """
        if sparsity <= 0:
            return 0.0
        if sparsity >= 1:
            return float('inf')
        
        # Flatten scores and compute percentile
        scores_flat = scores.flatten()
        k = int(sparsity * scores_flat.numel())
        
        if k == 0:
            return 0.0
        
        # Get k-th smallest value
        threshold = torch.kthvalue(scores_flat, k)[0].item()
        
        return threshold
    
    def create_mask(
        self,
        layer: nn.Module,
        sparsity: float
    ) -> torch.Tensor:
        """
        Create pruning mask for layer.
        
        Args:
            layer: Layer to create mask for
            sparsity: Sparsity level
            
        Returns:
            Binary mask tensor
        """
        # Compute importance scores
        scores = self.compute_importance_scores(layer, self.config.method)
        
        if self.config.structured:
            # Structured pruning (remove entire neurons/channels)
            mask = self._create_structured_mask(layer, scores, sparsity)
        else:
            # Unstructured pruning (remove individual weights)
            threshold = self.compute_threshold(scores, sparsity)
            mask = (scores > threshold).float()
        
        return mask
    
    def _create_structured_mask(
        self,
        layer: nn.Module,
        scores: torch.Tensor,
        sparsity: float
    ) -> torch.Tensor:
        """
        Create structured pruning mask.
        
        Removes entire neurons or channels following:
        - Liu et al. (2017): "Learning Efficient CNNs through Network Slimming"
        
        Args:
            layer: Layer to prune
            scores: Importance scores
            sparsity: Sparsity level
            
        Returns:
            Structured mask
        """
        weight = layer.weight.data
        
        if isinstance(layer, nn.Conv2d):
            # Prune entire filters (output channels)
            if len(scores.shape) == 4:
                scores = torch.sum(torch.abs(scores), dim=(1, 2, 3))
            
            num_filters = scores.shape[0]
            num_prune = int(sparsity * num_filters)
            
            # Get indices of filters to keep
            _, keep_indices = torch.topk(scores, num_filters - num_prune)
            
            # Create mask
            mask = torch.zeros_like(weight)
            mask[keep_indices] = 1.0
            
        elif isinstance(layer, nn.Linear):
            # Prune entire neurons
            if len(scores.shape) == 2:
                scores = torch.sum(torch.abs(scores), dim=1)
            
            num_neurons = scores.shape[0]
            num_prune = int(sparsity * num_neurons)
            
            # Get indices of neurons to keep
            _, keep_indices = torch.topk(scores, num_neurons - num_prune)
            
            # Create mask
            mask = torch.zeros_like(weight)
            mask[keep_indices] = 1.0
            
        else:
            # Fallback to unstructured
            threshold = self.compute_threshold(scores, sparsity)
            mask = (scores > threshold).float()
        
        return mask
    
    def apply_mask(self, layer: nn.Module, mask: torch.Tensor):
        """
        Apply pruning mask to layer.
        
        Args:
            layer: Layer to apply mask to
            mask: Binary mask
        """
        with torch.no_grad():
            layer.weight.data *= mask
    
    def prune_layer(
        self,
        layer_name: str,
        layer: nn.Module,
        sparsity: float
    ):
        """
        Prune a single layer.
        
        Args:
            layer_name: Name of the layer
            layer: Layer module
            sparsity: Sparsity level
        """
        # Create and store mask
        mask = self.create_mask(layer, sparsity)
        self.masks[layer_name] = mask
        
        # Apply mask
        self.apply_mask(layer, mask)
        
        # Compute actual sparsity
        actual_sparsity = 1.0 - (mask.sum() / mask.numel())
        
        logger.debug(
            f"Pruned layer {layer_name}: "
            f"target sparsity={sparsity:.2%}, "
            f"actual sparsity={actual_sparsity:.2%}"
        )
    
    def prune(self, sparsity: Optional[float] = None):
        """
        Prune the entire model.
        
        Args:
            sparsity: Override sparsity level
        """
        sparsity = sparsity or self.config.sparsity
        
        if self.config.global_pruning:
            self._global_prune(sparsity)
        else:
            self._layer_wise_prune(sparsity)
        
        # Update pruning history
        self.pruning_history['sparsity'].append(self.get_sparsity())
        
        logger.info(f"Model pruned to {self.get_sparsity():.2%} sparsity")
    
    def _global_prune(self, sparsity: float):
        """
        Apply global magnitude pruning.
        
        Prunes weights globally across all layers following:
        - Elsen et al. (2020): "Fast Sparse ConvNets"
        
        Args:
            sparsity: Target sparsity
        """
        # Collect all weights and scores
        all_scores = []
        layer_info = []
        
        for name, layer in self.prunable_layers.items():
            scores = self.compute_importance_scores(layer, self.config.method)
            all_scores.append(scores.flatten())
            layer_info.append((name, layer, scores.shape))
        
        # Concatenate all scores
        all_scores = torch.cat(all_scores)
        
        # Compute global threshold
        threshold = self.compute_threshold(all_scores, sparsity)
        
        # Apply threshold to each layer
        for name, layer, shape in layer_info:
            scores = self.compute_importance_scores(layer, self.config.method)
            mask = (scores > threshold).float()
            self.masks[name] = mask
            self.apply_mask(layer, mask)
    
    def _layer_wise_prune(self, sparsity: float):
        """
        Apply layer-wise magnitude pruning.
        
        Args:
            sparsity: Target sparsity per layer
        """
        for name, layer in self.prunable_layers.items():
            self.prune_layer(name, layer, sparsity)
    
    def gradual_prune(
        self,
        current_epoch: int,
        total_epochs: int
    ) -> float:
        """
        Apply gradual magnitude pruning.
        
        Implements gradual pruning schedule from:
        - Zhu & Gupta (2018): "To Prune, or Not to Prune"
        
        Args:
            current_epoch: Current training epoch
            total_epochs: Total training epochs
            
        Returns:
            Current sparsity level
        """
        if current_epoch < self.config.start_epoch:
            return self.config.init_sparsity
        
        if current_epoch >= self.config.end_epoch:
            return self.config.final_sparsity
        
        # Compute sparsity based on schedule
        if self.config.schedule == PruningSchedule.POLYNOMIAL:
            # Polynomial decay
            progress = (current_epoch - self.config.start_epoch) / \
                      (self.config.end_epoch - self.config.start_epoch)
            sparsity = self.config.final_sparsity + \
                      (self.config.init_sparsity - self.config.final_sparsity) * \
                      (1 - progress) ** 3
                      
        elif self.config.schedule == PruningSchedule.EXPONENTIAL:
            # Exponential decay
            progress = (current_epoch - self.config.start_epoch) / \
                      (self.config.end_epoch - self.config.start_epoch)
            sparsity = self.config.final_sparsity + \
                      (self.config.init_sparsity - self.config.final_sparsity) * \
                      math.exp(-5 * progress)
                      
        else:
            # Linear
            progress = (current_epoch - self.config.start_epoch) / \
                      (self.config.end_epoch - self.config.start_epoch)
            sparsity = self.config.init_sparsity + \
                      (self.config.final_sparsity - self.config.init_sparsity) * \
                      progress
        
        # Apply pruning
        self.prune(sparsity)
        
        return sparsity
    
    def get_sparsity(self) -> float:
        """
        Calculate current model sparsity.
        
        Returns:
            Overall sparsity level
        """
        total_params = 0
        pruned_params = 0
        
        for name, layer in self.prunable_layers.items():
            weight = layer.weight.data
            total_params += weight.numel()
            
            if name in self.masks:
                pruned_params += (self.masks[name] == 0).sum().item()
            else:
                pruned_params += (weight == 0).sum().item()
        
        if total_params == 0:
            return 0.0
        
        return pruned_params / total_params
    
    def lottery_ticket_rewind(self):
        """
        Implement lottery ticket hypothesis rewinding.
        
        Resets weights to original initialization following:
        - Frankle & Carbin (2019): "The Lottery Ticket Hypothesis"
        """
        if not self.original_weights:
            logger.warning("No original weights stored for rewinding")
            return
        
        for name, layer in self.prunable_layers.items():
            if name in self.original_weights:
                # Reset to original weights
                layer.weight.data = self.original_weights[name].clone()
                
                # Apply current mask
                if name in self.masks:
                    self.apply_mask(layer, self.masks[name])
        
        logger.info("Weights rewound to original initialization")
    
    def save_original_weights(self):
        """Save original weights for lottery ticket rewinding."""
        for name, layer in self.prunable_layers.items():
            self.original_weights[name] = layer.weight.data.clone()
        
        logger.info("Original weights saved for rewinding")
    
    def remove_pruning(self):
        """Remove pruning and restore full weights."""
        for name, layer in self.prunable_layers.items():
            if hasattr(layer, 'weight_orig'):
                # PyTorch pruning
                prune.remove(layer, 'weight')
            
            # Clear masks
            if name in self.masks:
                del self.masks[name]
        
        logger.info("Pruning removed, weights restored")
    
    def make_pruning_permanent(self):
        """Make pruning permanent by removing masked weights."""
        for name, layer in self.prunable_layers.items():
            if name in self.masks:
                # Apply mask permanently
                with torch.no_grad():
                    layer.weight.data *= self.masks[name]
                
                # Remove mask tracking
                del self.masks[name]
        
        logger.info("Pruning made permanent")
    
    def get_pruning_summary(self) -> Dict[str, Any]:
        """
        Get summary of pruning statistics.
        
        Returns:
            Dictionary with pruning statistics
        """
        summary = {
            'overall_sparsity': self.get_sparsity(),
            'method': self.config.method.value,
            'structured': self.config.structured,
            'layers': {}
        }
        
        for name, layer in self.prunable_layers.items():
            weight = layer.weight.data
            
            if name in self.masks:
                mask = self.masks[name]
                layer_sparsity = 1.0 - (mask.sum() / mask.numel())
            else:
                layer_sparsity = (weight == 0).sum().item() / weight.numel()
            
            summary['layers'][name] = {
                'sparsity': layer_sparsity,
                'total_params': weight.numel(),
                'pruned_params': int(layer_sparsity * weight.numel())
            }
        
        return summary
