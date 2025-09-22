"""
Weighted Voting Ensemble for AG News Classification
========================================================

Implementation of weighted voting mechanisms for ensemble learning, based on:
- Dietterich (2000): "Ensemble Methods in Machine Learning"
- Kuncheva (2004): "Combining Pattern Classifiers: Methods and Algorithms"
- Zhou (2012): "Ensemble Methods: Foundations and Algorithms"

Weighted voting assigns different importance weights to base models based on their
individual performance, confidence, or learned optimal weights.

Mathematical Foundation:
For predictions h_i(x) with weights w_i:
H(x) = argmax_c Σ(w_i * I[h_i(x) = c]) for hard voting
H(x) = argmax_c Σ(w_i * P_i(c|x)) for soft voting

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, log_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.models.ensemble.base_ensemble import BaseEnsemble
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class WeightingStrategy(Enum):
    """Strategies for determining voting weights"""
    UNIFORM = "uniform"  # Equal weights
    ACCURACY = "accuracy"  # Based on validation accuracy
    F1_SCORE = "f1_score"  # Based on F1 scores
    CONFIDENCE = "confidence"  # Based on prediction confidence
    LEARNED = "learned"  # Learned through optimization
    DYNAMIC = "dynamic"  # Dynamic per-sample weights
    BAYESIAN = "bayesian"  # Bayesian model averaging
    ENTROPY = "entropy"  # Inverse entropy weighting


@dataclass
class WeightedVotingConfig:
    """Configuration for weighted voting ensemble"""
    
    # Voting strategy
    voting_type: str = "soft"  # "soft" or "hard"
    weighting_strategy: WeightingStrategy = WeightingStrategy.LEARNED
    
    # Weight constraints
    normalize_weights: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    weight_regularization: float = 0.01
    
    # Optimization settings
    optimization_method: str = "SLSQP"  # For learned weights
    optimization_metric: str = "accuracy"
    max_iterations: int = 1000
    
    # Dynamic weighting
    use_confidence_threshold: bool = True
    confidence_threshold: float = 0.7
    
    # Performance tracking
    track_individual_performance: bool = True
    use_validation_weights: bool = True
    
    # Advanced options
    use_weight_decay: bool = False
    weight_decay_rate: float = 0.99
    adaptive_weights: bool = False
    temperature: float = 1.0  # For softmax scaling
    
    # Bayesian averaging
    use_model_uncertainty: bool = False
    prior_strength: float = 1.0


class WeightOptimizer:
    """
    Optimizer for learning ensemble weights
    
    Implements various optimization strategies to find optimal
    combination weights for ensemble members.
    """
    
    def __init__(self, config: WeightedVotingConfig):
        """
        Initialize weight optimizer.
        
        Args:
            config: Weighted voting configuration
        """
        self.config = config
        self.best_weights = None
        self.optimization_history = []
        
    def optimize_weights(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        initial_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Optimize ensemble weights based on validation performance.
        
        Args:
            predictions: Model predictions [n_models, n_samples, n_classes]
            labels: True labels [n_samples]
            initial_weights: Initial weight values
            
        Returns:
            Optimized weights [n_models]
        """
        n_models = predictions.shape[0]
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = np.ones(n_models) / n_models
        
        # Define objective function
        def objective(weights):
            """Objective function to minimize"""
            # Apply weights to predictions
            if self.config.voting_type == "soft":
                weighted_pred = np.sum(
                    predictions * weights.reshape(-1, 1, 1),
                    axis=0
                )
                pred_labels = np.argmax(weighted_pred, axis=1)
            else:
                # Hard voting
                pred_labels = []
                for i in range(predictions.shape[1]):
                    votes = predictions[:, i, :].argmax(axis=1)
                    weighted_votes = np.bincount(
                        votes,
                        weights=weights,
                        minlength=predictions.shape[2]
                    )
                    pred_labels.append(np.argmax(weighted_votes))
                pred_labels = np.array(pred_labels)
            
            # Calculate loss
            if self.config.optimization_metric == "accuracy":
                score = -accuracy_score(labels, pred_labels)
            elif self.config.optimization_metric == "log_loss":
                if self.config.voting_type == "soft":
                    score = log_loss(labels, weighted_pred)
                else:
                    score = -accuracy_score(labels, pred_labels)
            else:
                score = -accuracy_score(labels, pred_labels)
            
            # Add regularization
            if self.config.weight_regularization > 0:
                score += self.config.weight_regularization * np.sum(weights ** 2)
            
            return score
        
        # Constraints
        constraints = []
        
        # Weights sum to 1
        if self.config.normalize_weights:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w) - 1.0
            })
        
        # Bounds
        bounds = [(self.config.min_weight, self.config.max_weight)] * n_models
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method=self.config.optimization_method,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.config.max_iterations}
        )
        
        if result.success:
            self.best_weights = result.x
            logger.info(f"Weight optimization converged: {self.best_weights}")
        else:
            logger.warning(f"Weight optimization failed: {result.message}")
            self.best_weights = initial_weights
        
        return self.best_weights


class DynamicWeightCalculator:
    """
    Calculator for dynamic, instance-specific weights
    
    Computes different weights for each sample based on model
    confidence and reliability.
    """
    
    def __init__(self, config: WeightedVotingConfig):
        """
        Initialize dynamic weight calculator.
        
        Args:
            config: Configuration
        """
        self.config = config
        self.model_reliability = {}
        
    def calculate_confidence_weights(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate weights based on prediction confidence.
        
        Args:
            predictions: Model predictions [n_models, batch_size, n_classes]
            
        Returns:
            Confidence weights [n_models, batch_size]
        """
        # Calculate confidence as max probability
        confidences = torch.max(predictions, dim=2)[0]  # [n_models, batch_size]
        
        # Apply temperature scaling
        confidences = confidences / self.config.temperature
        
        # Apply threshold
        if self.config.use_confidence_threshold:
            mask = confidences > self.config.confidence_threshold
            confidences = confidences * mask.float()
        
        # Normalize across models for each sample
        if self.config.normalize_weights:
            weight_sum = confidences.sum(dim=0, keepdim=True)
            weights = confidences / (weight_sum + 1e-8)
        else:
            weights = confidences
        
        return weights
    
    def calculate_entropy_weights(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate weights based on prediction entropy (uncertainty).
        
        Lower entropy = higher confidence = higher weight
        
        Args:
            predictions: Model predictions [n_models, batch_size, n_classes]
            
        Returns:
            Entropy-based weights [n_models, batch_size]
        """
        # Calculate entropy
        entropy = -torch.sum(
            predictions * torch.log(predictions + 1e-8),
            dim=2
        )  # [n_models, batch_size]
        
        # Invert entropy (lower entropy = higher weight)
        max_entropy = np.log(predictions.shape[2])  # Maximum possible entropy
        weights = (max_entropy - entropy) / max_entropy
        
        # Normalize
        if self.config.normalize_weights:
            weight_sum = weights.sum(dim=0, keepdim=True)
            weights = weights / (weight_sum + 1e-8)
        
        return weights
    
    def calculate_reliability_weights(
        self,
        predictions: torch.Tensor,
        model_ids: List[str]
    ) -> torch.Tensor:
        """
        Calculate weights based on historical model reliability.
        
        Args:
            predictions: Model predictions
            model_ids: Model identifiers
            
        Returns:
            Reliability weights
        """
        weights = []
        
        for model_id in model_ids:
            if model_id in self.model_reliability:
                weight = self.model_reliability[model_id]
            else:
                weight = 1.0 / len(model_ids)  # Default uniform weight
            weights.append(weight)
        
        weights = torch.tensor(weights).unsqueeze(1)  # [n_models, 1]
        
        # Broadcast to batch size
        batch_size = predictions.shape[1]
        weights = weights.expand(-1, batch_size)
        
        return weights
    
    def update_reliability(
        self,
        model_id: str,
        correct: bool,
        alpha: float = 0.1
    ):
        """
        Update model reliability based on prediction outcome.
        
        Args:
            model_id: Model identifier
            correct: Whether prediction was correct
            alpha: Learning rate for update
        """
        if model_id not in self.model_reliability:
            self.model_reliability[model_id] = 0.5
        
        # Exponential moving average update
        self.model_reliability[model_id] = (
            (1 - alpha) * self.model_reliability[model_id] +
            alpha * float(correct)
        )


@MODELS.register("weighted_voting")
class WeightedVotingEnsemble(BaseEnsemble):
    """
    Weighted Voting Ensemble Model
    
    Combines predictions from multiple models using weighted voting,
    where weights can be learned, dynamic, or based on model performance.
    
    The ensemble supports:
    1. Multiple weighting strategies
    2. Dynamic per-sample weights
    3. Confidence-based voting
    4. Bayesian model averaging
    5. Adaptive weight learning
    """
    
    def __init__(
        self,
        models: List[AGNewsBaseModel],
        config: Optional[WeightedVotingConfig] = None
    ):
        """
        Initialize weighted voting ensemble.
        
        Args:
            models: List of base models
            config: Ensemble configuration
        """
        super().__init__(models)
        
        self.config = config or WeightedVotingConfig()
        self.n_models = len(models)
        
        # Initialize components
        self.weight_optimizer = WeightOptimizer(self.config)
        self.dynamic_calculator = DynamicWeightCalculator(self.config)
        
        # Initialize weights
        self._init_weights()
        
        # Performance tracking
        self.model_performances = {}
        self.voting_history = []
        
        logger.info(
            f"Initialized Weighted Voting Ensemble with {self.n_models} models "
            f"using {self.config.weighting_strategy.value} strategy"
        )
    
    def _init_weights(self):
        """Initialize ensemble weights based on strategy"""
        if self.config.weighting_strategy == WeightingStrategy.UNIFORM:
            # Equal weights
            weights = torch.ones(self.n_models) / self.n_models
            
        elif self.config.weighting_strategy == WeightingStrategy.LEARNED:
            # Will be learned during training
            weights = torch.ones(self.n_models) / self.n_models
            self.weights = Parameter(weights)
            
        else:
            # Start with uniform, will be updated
            weights = torch.ones(self.n_models) / self.n_models
        
        if not hasattr(self, 'weights'):
            self.register_buffer('weights', weights)
        
        logger.info(f"Initialized weights: {self.weights.detach().cpu().numpy()}")
    
    def update_weights(
        self,
        validation_predictions: torch.Tensor,
        validation_labels: torch.Tensor
    ):
        """
        Update ensemble weights based on validation performance.
        
        Args:
            validation_predictions: Predictions on validation set
            validation_labels: True validation labels
        """
        if self.config.weighting_strategy == WeightingStrategy.ACCURACY:
            # Weight by individual accuracy
            weights = []
            for i in range(self.n_models):
                pred_labels = validation_predictions[i].argmax(dim=1)
                accuracy = (pred_labels == validation_labels).float().mean()
                weights.append(accuracy)
            
            weights = torch.tensor(weights)
            
        elif self.config.weighting_strategy == WeightingStrategy.F1_SCORE:
            # Weight by F1 scores
            from sklearn.metrics import f1_score
            weights = []
            
            for i in range(self.n_models):
                pred_labels = validation_predictions[i].argmax(dim=1).cpu().numpy()
                f1 = f1_score(
                    validation_labels.cpu().numpy(),
                    pred_labels,
                    average='weighted'
                )
                weights.append(f1)
            
            weights = torch.tensor(weights)
            
        elif self.config.weighting_strategy == WeightingStrategy.LEARNED:
            # Optimize weights
            predictions_np = validation_predictions.cpu().numpy()
            labels_np = validation_labels.cpu().numpy()
            
            optimized_weights = self.weight_optimizer.optimize_weights(
                predictions_np,
                labels_np
            )
            weights = torch.tensor(optimized_weights)
            
        elif self.config.weighting_strategy == WeightingStrategy.ENTROPY:
            # Weight by inverse entropy
            weights = []
            for i in range(self.n_models):
                entropy = -torch.sum(
                    validation_predictions[i] * torch.log(validation_predictions[i] + 1e-8),
                    dim=1
                ).mean()
                weights.append(1.0 / (entropy + 1e-8))
            
            weights = torch.tensor(weights)
        
        else:
            return  # Keep current weights
        
        # Normalize weights if needed
        if self.config.normalize_weights:
            weights = weights / weights.sum()
        
        # Apply constraints
        weights = torch.clamp(weights, self.config.min_weight, self.config.max_weight)
        
        # Update weights
        if isinstance(self.weights, Parameter):
            self.weights.data = weights.to(self.weights.device)
        else:
            self.weights = weights.to(self.weights.device)
        
        logger.info(f"Updated weights: {weights.numpy()}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_individual: bool = False,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass with weighted voting.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            return_individual: Return individual model predictions
            **kwargs: Additional arguments
            
        Returns:
            Ensemble predictions with metadata
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Collect predictions from all models
        all_predictions = []
        all_logits = []
        individual_outputs = []
        
        for i, model in enumerate(self.models):
            with torch.no_grad() if not self.training else torch.enable_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                
                # Get probabilities
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=-1)
                else:
                    # Fallback if no logits
                    probs = torch.ones(batch_size, 4, device=device) / 4
                    logits = torch.zeros(batch_size, 4, device=device)
                
                all_predictions.append(probs)
                all_logits.append(logits)
                
                if return_individual:
                    individual_outputs.append(outputs)
        
        # Stack predictions [n_models, batch_size, n_classes]
        all_predictions = torch.stack(all_predictions)
        all_logits = torch.stack(all_logits)
        
        # Calculate weights based on strategy
        if self.config.weighting_strategy == WeightingStrategy.DYNAMIC:
            # Dynamic per-sample weights
            weights = self.dynamic_calculator.calculate_confidence_weights(all_predictions)
        elif self.config.weighting_strategy == WeightingStrategy.CONFIDENCE:
            # Confidence-based weights
            weights = self.dynamic_calculator.calculate_entropy_weights(all_predictions)
        else:
            # Use fixed weights, broadcast to batch size
            weights = self.weights.unsqueeze(1).expand(-1, batch_size)
        
        # Apply weighted voting
        if self.config.voting_type == "soft":
            # Soft voting: weighted average of probabilities
            weights = weights.unsqueeze(2)  # [n_models, batch_size, 1]
            ensemble_probs = torch.sum(all_predictions * weights, dim=0)
            
            # Ensure normalization
            ensemble_probs = ensemble_probs / ensemble_probs.sum(dim=1, keepdim=True)
            
            # Convert to logits
            ensemble_logits = torch.log(ensemble_probs + 1e-8)
            
        else:
            # Hard voting: weighted majority vote
            predictions = all_predictions.argmax(dim=2)  # [n_models, batch_size]
            ensemble_predictions = []
            
            for b in range(batch_size):
                votes = predictions[:, b]
                vote_weights = weights[:, b]
                
                # Weighted vote counting
                weighted_votes = torch.zeros(4, device=device)
                for vote, weight in zip(votes, vote_weights):
                    weighted_votes[vote] += weight
                
                ensemble_predictions.append(weighted_votes.argmax())
            
            ensemble_predictions = torch.stack(ensemble_predictions)
            
            # Create one-hot logits
            ensemble_logits = torch.zeros(batch_size, 4, device=device)
            ensemble_logits[torch.arange(batch_size), ensemble_predictions] = 10.0
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(ensemble_logits, labels)
        
        # Track voting statistics
        self.voting_history.append({
            'weights': weights.mean(dim=1).detach().cpu().numpy(),
            'agreement': self._calculate_agreement(all_predictions.argmax(dim=2))
        })
        
        # Prepare metadata
        metadata = {
            'ensemble_weights': weights.mean(dim=1).detach().cpu().numpy(),
            'voting_type': self.config.voting_type,
            'weighting_strategy': self.config.weighting_strategy.value,
            'model_agreement': self._calculate_agreement(all_predictions.argmax(dim=2))
        }
        
        if return_individual:
            metadata['individual_outputs'] = individual_outputs
            metadata['individual_predictions'] = all_predictions
        
        return ModelOutputs(
            loss=loss,
            logits=ensemble_logits,
            metadata=metadata
        )
    
    def _calculate_agreement(self, predictions: torch.Tensor) -> float:
        """
        Calculate agreement rate among models.
        
        Args:
            predictions: Model predictions [n_models, batch_size]
            
        Returns:
            Average agreement rate
        """
        n_models = predictions.shape[0]
        batch_size = predictions.shape[1]
        
        agreement_scores = []
        for b in range(batch_size):
            votes = predictions[:, b]
            # Count most common prediction
            mode_count = torch.mode(votes)[0].sum().item()
            agreement = mode_count / n_models
            agreement_scores.append(agreement)
        
        return np.mean(agreement_scores)
    
    def get_weight_analysis(self) -> Dict[str, Any]:
        """
        Analyze ensemble weights and their impact
        
        Returns:
            Weight analysis dictionary
        """
        analysis = {
            'current_weights': self.weights.detach().cpu().numpy(),
            'weight_entropy': float(stats.entropy(self.weights.detach().cpu().numpy())),
            'effective_models': float(1.0 / torch.sum(self.weights ** 2).item()),
            'weight_variance': float(torch.var(self.weights).item()),
            'max_weight': float(torch.max(self.weights).item()),
            'min_weight': float(torch.min(self.weights).item())
        }
        
        # Add voting history statistics if available
        if self.voting_history:
            recent_history = self.voting_history[-100:]  # Last 100 votes
            analysis['average_agreement'] = np.mean([h['agreement'] for h in recent_history])
            analysis['weight_stability'] = np.std([h['weights'] for h in recent_history], axis=0).mean()
        
        return analysis
    
    def prune_weak_models(self, threshold: float = 0.1):
        """
        Remove models with consistently low weights
        
        Args:
            threshold: Minimum weight threshold
        """
        keep_indices = torch.where(self.weights > threshold)[0]
        
        if len(keep_indices) < 2:
            logger.warning("Cannot prune: would leave fewer than 2 models")
            return
        
        # Keep only selected models
        self.models = [self.models[i] for i in keep_indices]
        self.weights = self.weights[keep_indices]
        self.weights = self.weights / self.weights.sum()  # Renormalize
        self.n_models = len(self.models)
        
        logger.info(f"Pruned ensemble to {self.n_models} models")
