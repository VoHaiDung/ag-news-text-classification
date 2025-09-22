"""
Soft Voting Ensemble Implementation
====================================

This module implements soft voting ensemble for combining multiple models'
probabilistic predictions, following theoretical foundations from:
- Kittler et al. (1998): "On Combining Classifiers"
- Kuncheva (2004): "Combining Pattern Classifiers: Methods and Algorithms"
- Ruta & Gabrys (2005): "Classifier selection for majority voting"

Mathematical Foundation:
Soft voting combines probability distributions:
P(y=c|x) = Σ_m w_m * P_m(y=c|x)
where w_m are model weights and P_m are individual model probabilities.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Union, Dict, Any, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.ensemble.base_ensemble import BaseEnsemble, EnsembleConfig
from src.core.registry import ENSEMBLES
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@ENSEMBLES.register("soft_voting", aliases=["soft_vote", "weighted_average"])
class SoftVotingEnsemble(BaseEnsemble):
    """
    Soft voting ensemble that combines probability distributions.
    
    This implementation provides:
    1. Weighted averaging of probabilities
    2. Confidence-based dynamic weighting
    3. Temperature scaling for calibration
    4. Entropy-based uncertainty estimation
    
    The soft voting approach is particularly effective when:
    - Models output well-calibrated probabilities
    - Models have similar performance levels
    - Uncertainty estimation is important
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[EnsembleConfig] = None,
        use_log_probs: bool = False,
        confidence_threshold: float = 0.0,
        entropy_weighting: bool = False
    ):
        """
        Initialize soft voting ensemble.
        
        Args:
            models: List of base models
            config: Ensemble configuration
            use_log_probs: Combine in log-probability space
            confidence_threshold: Minimum confidence for voting
            entropy_weighting: Weight by inverse entropy
        """
        super().__init__(models, config)
        
        self.use_log_probs = use_log_probs
        self.confidence_threshold = confidence_threshold
        self.entropy_weighting = entropy_weighting
        
        # Initialize confidence estimator if needed
        if self.config.use_confidence_weighting:
            self._init_confidence_estimator()
        
        logger.info(
            f"Initialized SoftVotingEnsemble with {self.num_models} models, "
            f"log_probs={use_log_probs}, entropy_weighting={entropy_weighting}"
        )
    
    def _init_confidence_estimator(self):
        """Initialize confidence estimation components."""
        # Learnable confidence parameters for each model
        self.confidence_params = nn.ParameterList([
            nn.Parameter(torch.ones(1)) for _ in range(self.num_models)
        ])
    
    def combine_predictions(
        self,
        predictions: List[torch.Tensor],
        return_uncertainty: bool = False,
        **kwargs
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Combine predictions using soft voting.
        
        Implements weighted averaging of probability distributions with
        optional confidence weighting and entropy-based adjustments.
        
        Args:
            predictions: List of logits from each model [batch_size, num_classes]
            return_uncertainty: Return uncertainty estimates
            
        Returns:
            Combined logits and optionally uncertainty estimates
            
        Mathematical Description:
            P_ensemble = Σ_i w_i * softmax(logits_i / T)
            where w_i are normalized weights and T is temperature
        """
        batch_size = predictions[0].size(0)
        
        # Convert logits to probabilities
        if self.use_log_probs:
            # Work in log-probability space for numerical stability
            log_probs = [F.log_softmax(pred / self.temperature 
                         if hasattr(self, 'temperature') else pred, dim=-1) 
                         for pred in predictions]
        else:
            probs = [F.softmax(pred / self.temperature 
                     if hasattr(self, 'temperature') else pred, dim=-1) 
                     for pred in predictions]
        
        # Calculate dynamic weights if needed
        weights = self._calculate_weights(predictions)
        
        # Combine predictions
        if self.use_log_probs:
            # Weighted geometric mean in log space
            combined_log_probs = torch.zeros_like(log_probs[0])
            for i, (log_prob, weight) in enumerate(zip(log_probs, weights)):
                combined_log_probs += weight.unsqueeze(-1) * log_prob
            
            # Convert back to logits
            combined_logits = combined_log_probs
        else:
            # Weighted arithmetic mean
            combined_probs = torch.zeros_like(probs[0])
            for i, (prob, weight) in enumerate(zip(probs, weights)):
                combined_probs += weight.unsqueeze(-1) * prob
            
            # Convert to logits
            combined_logits = torch.log(combined_probs + 1e-10)
        
        if return_uncertainty:
            uncertainty = self._estimate_uncertainty(predictions, combined_logits)
            return combined_logits, uncertainty
        
        return combined_logits
    
    def _calculate_weights(
        self,
        predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Calculate dynamic weights for each model.
        
        Implements various weighting strategies:
        1. Fixed weights
        2. Confidence-based weights
        3. Entropy-based weights
        4. Performance-based weights
        """
        batch_size = predictions[0].size(0)
        
        # Start with base weights
        weights = self.weights.unsqueeze(0).expand(batch_size, -1)
        
        # Apply confidence weighting
        if self.config.use_confidence_weighting:
            confidences = self._estimate_confidences(predictions)
            weights = weights * confidences
        
        # Apply entropy weighting (inverse entropy)
        if self.entropy_weighting:
            entropies = []
            for pred in predictions:
                prob = F.softmax(pred, dim=-1)
                entropy = -torch.sum(prob * torch.log(prob + 1e-10), dim=-1)
                entropies.append(entropy)
            
            entropies = torch.stack(entropies, dim=1)  # [batch_size, num_models]
            
            # Inverse entropy as weight (lower entropy = higher confidence)
            entropy_weights = 1.0 / (1.0 + entropies)
            weights = weights * entropy_weights
        
        # Apply confidence threshold
        if self.confidence_threshold > 0:
            max_probs = []
            for pred in predictions:
                prob = F.softmax(pred, dim=-1)
                max_prob, _ = torch.max(prob, dim=-1)
                max_probs.append(max_prob)
            
            max_probs = torch.stack(max_probs, dim=1)
            
            # Zero out weights for low confidence predictions
            confidence_mask = (max_probs > self.confidence_threshold).float()
            weights = weights * confidence_mask
        
        # Normalize weights
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-10)
        
        return weights
    
    def _estimate_confidences(
        self,
        predictions: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Estimate confidence scores for each model's predictions.
        
        Uses various confidence metrics:
        - Maximum probability
        - Entropy
        - Prediction margin
        """
        confidences = []
        
        for i, pred in enumerate(predictions):
            prob = F.softmax(pred, dim=-1)
            
            # Maximum probability as confidence
            max_prob, _ = torch.max(prob, dim=-1)
            
            # Scale by learnable parameter
            if hasattr(self, 'confidence_params'):
                confidence = max_prob * torch.sigmoid(self.confidence_params[i])
            else:
                confidence = max_prob
            
            confidences.append(confidence)
        
        return torch.stack(confidences, dim=1)
    
    def _estimate_uncertainty(
        self,
        predictions: List[torch.Tensor],
        combined_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate prediction uncertainty.
        
        Computes multiple uncertainty metrics:
        1. Predictive entropy
        2. Mutual information
        3. Variation ratio
        4. Model disagreement
        """
        # Convert to probabilities
        probs = [F.softmax(pred, dim=-1) for pred in predictions]
        combined_probs = F.softmax(combined_logits, dim=-1)
        
        # Predictive entropy
        entropy = -torch.sum(combined_probs * torch.log(combined_probs + 1e-10), dim=-1)
        
        # Expected entropy (aleatoric uncertainty)
        expected_entropy = torch.zeros_like(entropy)
        for prob in probs:
            expected_entropy += -torch.sum(prob * torch.log(prob + 1e-10), dim=-1)
        expected_entropy /= len(probs)
        
        # Mutual information (epistemic uncertainty)
        mutual_info = entropy - expected_entropy
        
        # Variation ratio
        _, predicted_class = torch.max(combined_probs, dim=-1)
        agreement_counts = torch.zeros_like(entropy)
        for prob in probs:
            _, pred_class = torch.max(prob, dim=-1)
            agreement_counts += (pred_class == predicted_class).float()
        variation_ratio = 1.0 - agreement_counts / len(probs)
        
        return {
            "entropy": entropy,
            "mutual_information": mutual_info,
            "expected_entropy": expected_entropy,
            "variation_ratio": variation_ratio,
            "total_uncertainty": entropy
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass with soft voting.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            return_uncertainty: Return uncertainty estimates
            **kwargs: Additional arguments
            
        Returns:
            Ensemble predictions with optional uncertainty
        """
        from src.models.base.base_model import ModelOutputs
        
        # Get base outputs
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_individual=True,
            **kwargs
        )
        
        # Add uncertainty if requested
        if return_uncertainty:
            predictions = outputs.metadata["individual_logits"]
            uncertainty = self._estimate_uncertainty(predictions, outputs.logits)
            outputs.metadata["uncertainty"] = uncertainty
        
        return outputs


@ENSEMBLES.register("weighted_voting")
class WeightedVotingEnsemble(SoftVotingEnsemble):
    """
    Weighted voting with learnable weights.
    
    Extends soft voting with:
    1. Learnable weight optimization
    2. Performance-based weight adjustment
    3. Online weight adaptation
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        config: Optional[EnsembleConfig] = None,
        adapt_weights: bool = True,
        weight_decay: float = 0.01
    ):
        """
        Initialize weighted voting ensemble.
        
        Args:
            models: List of base models
            config: Ensemble configuration
            adapt_weights: Enable online weight adaptation
            weight_decay: Weight decay for regularization
        """
        super().__init__(models, config)
        
        self.adapt_weights = adapt_weights
        self.weight_decay = weight_decay
        
        # Performance tracking for weight adaptation
        if adapt_weights:
            self.model_performances = torch.zeros(self.num_models)
            self.update_counts = torch.zeros(self.num_models)
    
    def update_weights(
        self,
        model_correct: torch.Tensor
    ):
        """
        Update weights based on model performance.
        
        Implements exponential moving average of accuracy.
        
        Args:
            model_correct: Boolean tensor of correct predictions per model
        """
        if not self.adapt_weights:
            return
        
        # Update performance statistics
        alpha = 0.1  # Learning rate for EMA
        for i in range(self.num_models):
            accuracy = model_correct[i].float().mean()
            self.model_performances[i] = (
                alpha * accuracy + 
                (1 - alpha) * self.model_performances[i]
            )
            self.update_counts[i] += 1
        
        # Update weights based on performance
        with torch.no_grad():
            # Use softmax of performances as weights
            self.weights.data = F.softmax(
                self.model_performances / 0.1,  # Temperature
                dim=0
            )
        
        logger.debug(f"Updated weights: {self.weights.tolist()}")
