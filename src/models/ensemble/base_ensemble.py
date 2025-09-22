"""
Base Ensemble Model Implementation
===================================

This module provides base classes for ensemble methods following principles from:
- Dietterich (2000): "Ensemble Methods in Machine Learning"
- Zhou (2012): "Ensemble Methods: Foundations and Algorithms"
- Sagi & Rokach (2018): "Ensemble learning: A survey"

Mathematical Foundation:
For M models with predictions f_m(x), ensemble prediction is:
F(x) = Φ(f_1(x), f_2(x), ..., f_M(x))
where Φ is the combination function.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Union, Dict, Any, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.exceptions import ModelError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EnsembleConfig:
    """Configuration for ensemble models."""
    ensemble_method: str = "voting"
    num_classes: int = 4
    weights: Optional[List[float]] = None
    use_confidence_weighting: bool = False
    calibrate_predictions: bool = False
    temperature_scaling: float = 1.0
    optimize_weights: bool = False
    diversity_penalty: float = 0.0
    min_agreement: float = 0.0


class BaseEnsemble(AGNewsBaseModel, ABC):
    """
    Abstract base class for ensemble models.
    
    Provides common functionality for ensemble methods including:
    1. Model management
    2. Prediction aggregation
    3. Diversity measurement
    4. Weight optimization
    """
    
    def __init__(
        self,
        models: List[AGNewsBaseModel],
        config: Optional[EnsembleConfig] = None
    ):
        """
        Initialize base ensemble.
        
        Args:
            models: List of base models
            config: Ensemble configuration
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.config = config or EnsembleConfig()
        self.num_models = len(models)
        self.num_classes = self.config.num_classes
        
        # Initialize weights
        self._init_weights()
        
        # Calibration parameters
        if self.config.calibrate_predictions:
            self.temperature = nn.Parameter(
                torch.tensor(self.config.temperature_scaling)
            )
        
        logger.info(f"Initialized {self.__class__.__name__} with {self.num_models} models")
    
    def _init_weights(self):
        """Initialize ensemble weights."""
        if self.config.weights is not None:
            assert len(self.config.weights) == self.num_models
            self.weights = nn.Parameter(
                torch.tensor(self.config.weights, dtype=torch.float32)
            )
        else:
            # Equal weights by default
            self.weights = nn.Parameter(
                torch.ones(self.num_models, dtype=torch.float32) / self.num_models
            )
        
        # Make weights trainable if optimization is enabled
        self.weights.requires_grad = self.config.optimize_weights
    
    @abstractmethod
    def combine_predictions(
        self,
        predictions: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Combine predictions from individual models.
        
        Args:
            predictions: List of model predictions
            **kwargs: Additional arguments
            
        Returns:
            Combined predictions
        """
        pass
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_individual: bool = False,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through ensemble.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            return_individual: Return individual model predictions
            **kwargs: Additional arguments
            
        Returns:
            Ensemble predictions
        """
        # Collect predictions from all models
        all_outputs = []
        all_logits = []
        all_probs = []
        
        for i, model in enumerate(self.models):
            # Get model predictions
            with torch.set_grad_enabled(self.training):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                all_outputs.append(outputs)
                all_logits.append(outputs.logits)
                
                # Calculate probabilities
                if self.config.calibrate_predictions:
                    probs = F.softmax(outputs.logits / self.temperature, dim=-1)
                else:
                    probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs)
        
        # Combine predictions
        ensemble_logits = self.combine_predictions(all_logits)
        
        # Calculate loss
        loss = None
        if labels is not None:
            loss = self.compute_ensemble_loss(
                ensemble_logits,
                labels,
                all_logits
            )
        
        # Prepare outputs
        outputs = ModelOutputs(
            logits=ensemble_logits,
            loss=loss
        )
        
        if return_individual:
            outputs.metadata = {
                "individual_logits": all_logits,
                "individual_probs": all_probs,
                "ensemble_weights": self.weights.detach()
            }
        
        return outputs
    
    def compute_ensemble_loss(
        self,
        ensemble_logits: torch.Tensor,
        labels: torch.Tensor,
        individual_logits: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute ensemble loss with optional diversity penalty.
        
        Args:
            ensemble_logits: Combined predictions
            labels: Target labels
            individual_logits: Individual model predictions
            
        Returns:
            Total loss
        """
        # Main classification loss
        loss_fct = nn.CrossEntropyLoss()
        ensemble_loss = loss_fct(ensemble_logits, labels)
        
        # Add diversity penalty if configured
        if self.config.diversity_penalty > 0:
            diversity_loss = self.compute_diversity_loss(individual_logits)
            ensemble_loss = ensemble_loss - self.config.diversity_penalty * diversity_loss
        
        return ensemble_loss
    
    def compute_diversity_loss(
        self,
        predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute diversity loss to encourage diverse predictions.
        
        Based on negative correlation learning:
        L_div = -Σ_i Σ_j≠i corr(f_i, f_j)
        """
        diversity = 0.0
        num_pairs = 0
        
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # Compute correlation between predictions
                pred_i = F.softmax(predictions[i], dim=-1)
                pred_j = F.softmax(predictions[j], dim=-1)
                
                # Negative correlation
                correlation = torch.mean(pred_i * pred_j)
                diversity += correlation
                num_pairs += 1
        
        return diversity / max(num_pairs, 1)
    
    def measure_diversity(
        self,
        predictions: List[torch.Tensor]
    ) -> Dict[str, float]:
        """
        Measure ensemble diversity metrics.
        
        Returns various diversity measures:
        - Disagreement measure
        - Q-statistic
        - Correlation coefficient
        - Entropy
        """
        with torch.no_grad():
            # Convert to probabilities
            probs = [F.softmax(p, dim=-1) for p in predictions]
            
            # Get predicted classes
            preds = [torch.argmax(p, dim=-1) for p in probs]
            
            # Pairwise disagreement
            disagreements = []
            for i in range(len(preds)):
                for j in range(i + 1, len(preds)):
                    disagree = (preds[i] != preds[j]).float().mean()
                    disagreements.append(disagree.item())
            
            # Average entropy
            entropies = []
            for p in probs:
                entropy = -torch.sum(p * torch.log(p + 1e-8), dim=-1).mean()
                entropies.append(entropy.item())
            
            return {
                "avg_disagreement": np.mean(disagreements) if disagreements else 0.0,
                "avg_entropy": np.mean(entropies),
                "max_disagreement": np.max(disagreements) if disagreements else 0.0,
                "min_disagreement": np.min(disagreements) if disagreements else 0.0
            }
    
    def optimize_weights(
        self,
        validation_data: torch.Tensor,
        validation_labels: torch.Tensor,
        num_iterations: int = 100
    ):
        """
        Optimize ensemble weights using validation data.
        
        Uses gradient-based optimization to find optimal weights.
        """
        optimizer = torch.optim.Adam([self.weights], lr=0.01)
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            
            # Forward pass
            outputs = self.forward(validation_data, labels=validation_labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Ensure weights are positive and sum to 1
            with torch.no_grad():
                self.weights.data = F.softmax(self.weights, dim=0)
        
        logger.info(f"Optimized weights: {self.weights.tolist()}")
    
    def get_model_contributions(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get contribution of each model to final prediction.
        
        Returns:
            Dictionary with model indices and their contributions
        """
        with torch.no_grad():
            outputs = self.forward(
                input_ids,
                attention_mask,
                return_individual=True
            )
            
            individual_probs = outputs.metadata["individual_probs"]
            weights = outputs.metadata["ensemble_weights"]
            
            contributions = {}
            for i, (prob, weight) in enumerate(zip(individual_probs, weights)):
                contributions[f"model_{i}"] = prob * weight
            
            return contributions
