"""
Bayesian Ensemble Model Implementation
=======================================

Implementation of Bayesian model averaging and uncertainty quantification for ensembles,
based on:
- Hoeting et al. (1999): "Bayesian Model Averaging: A Tutorial"
- Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
- Lakshminarayanan et al. (2017): "Simple and Scalable Predictive Uncertainty Estimation"
- Wilson & Izmailov (2020): "Bayesian Deep Learning and a Probabilistic Perspective"

Mathematical Foundation:
Bayesian Model Averaging: P(y|x,D) = Σ_m P(y|x,m,D) P(m|D)
where P(m|D) is the posterior probability of model m given data D.

Uncertainty decomposition:
Total uncertainty = Aleatoric uncertainty + Epistemic uncertainty

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from scipy.stats import entropy

from src.models.ensemble.base_ensemble import BaseEnsemble, EnsembleConfig
from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import ENSEMBLES
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BayesianEnsembleConfig(EnsembleConfig):
    """Configuration for Bayesian ensemble."""
    # Model weighting
    use_posterior_weights: bool = True  # Use Bayesian posterior weights
    prior_type: str = "uniform"  # "uniform", "dirichlet", "learned"
    
    # Uncertainty estimation
    uncertainty_method: str = "entropy"  # "entropy", "variance", "mutual_info"
    mc_dropout: bool = True  # Use MC dropout for uncertainty
    num_mc_samples: int = 10  # Number of MC dropout samples
    
    # Temperature scaling
    use_temperature_scaling: bool = True
    initial_temperature: float = 1.0
    learn_temperature: bool = True
    
    # Evidence approximation
    evidence_method: str = "marginal_likelihood"  # "marginal_likelihood", "bic", "aic"
    
    # Calibration
    calibration_method: str = "platt"  # "platt", "isotonic", "temperature"
    
    # Deep ensembles
    use_deep_ensembles: bool = True
    ensemble_size: int = 5
    
    # Uncertainty thresholds
    uncertainty_threshold: float = 0.5
    abstention_allowed: bool = False


class BayesianModelWeight(nn.Module):
    """
    Bayesian model weighting with uncertainty.
    
    Learns posterior distribution over model weights using
    variational inference or MCMC approximation.
    """
    
    def __init__(
        self,
        num_models: int,
        prior_type: str = "uniform"
    ):
        """
        Initialize Bayesian model weights.
        
        Args:
            num_models: Number of models in ensemble
            prior_type: Type of prior distribution
        """
        super().__init__()
        
        self.num_models = num_models
        self.prior_type = prior_type
        
        # Variational parameters for weights
        self.weight_mean = nn.Parameter(torch.ones(num_models) / num_models)
        self.weight_log_var = nn.Parameter(torch.zeros(num_models))
        
        # Prior parameters
        if prior_type == "dirichlet":
            self.prior_alpha = nn.Parameter(torch.ones(num_models))
        
    def sample_weights(self, num_samples: int = 1) -> torch.Tensor:
        """
        Sample weights from posterior distribution.
        
        Args:
            num_samples: Number of weight samples
            
        Returns:
            Sampled weights [num_samples, num_models]
        """
        # Sample from variational distribution
        std = torch.exp(0.5 * self.weight_log_var)
        eps = torch.randn(num_samples, self.num_models)
        weights = self.weight_mean + eps * std
        
        # Apply softmax to ensure valid probability distribution
        weights = F.softmax(weights, dim=-1)
        
        return weights
    
    def kl_divergence(self) -> torch.Tensor:
        """
        Compute KL divergence between posterior and prior.
        
        Returns:
            KL divergence value
        """
        if self.prior_type == "uniform":
            # KL(q||p) for uniform prior
            kl = -0.5 * torch.sum(1 + self.weight_log_var - self.weight_mean.pow(2) - self.weight_log_var.exp())
        elif self.prior_type == "dirichlet":
            # Approximate KL for Dirichlet prior
            kl = torch.sum(self.weight_mean * (torch.log(self.weight_mean + 1e-10) - torch.digamma(self.prior_alpha)))
        else:
            kl = torch.tensor(0.0)
        
        return kl


class UncertaintyEstimator(nn.Module):
    """
    Uncertainty estimation for Bayesian ensembles.
    
    Implements various uncertainty quantification methods including
    predictive entropy, mutual information, and variance-based measures.
    """
    
    def __init__(self, config: BayesianEnsembleConfig):
        """
        Initialize uncertainty estimator.
        
        Args:
            config: Bayesian ensemble configuration
        """
        super().__init__()
        self.config = config
    
    def estimate_uncertainty(
        self,
        predictions: torch.Tensor,
        method: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Estimate various uncertainty measures.
        
        Args:
            predictions: Model predictions [batch_size, num_models, num_classes]
            method: Specific uncertainty method to use
            
        Returns:
            Dictionary of uncertainty measures
        """
        method = method or self.config.uncertainty_method
        
        # Convert to probabilities
        probs = F.softmax(predictions, dim=-1)
        
        uncertainties = {}
        
        # Predictive entropy (total uncertainty)
        mean_probs = probs.mean(dim=1)
        predictive_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-10), dim=-1)
        uncertainties["total_uncertainty"] = predictive_entropy
        
        # Expected entropy (aleatoric uncertainty)
        individual_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        expected_entropy = individual_entropy.mean(dim=1)
        uncertainties["aleatoric_uncertainty"] = expected_entropy
        
        # Mutual information (epistemic uncertainty)
        mutual_info = predictive_entropy - expected_entropy
        uncertainties["epistemic_uncertainty"] = mutual_info
        
        # Variance-based uncertainty
        variance = probs.var(dim=1).mean(dim=-1)
        uncertainties["variance_uncertainty"] = variance
        
        # BALD (Bayesian Active Learning by Disagreement)
        bald = mutual_info
        uncertainties["bald"] = bald
        
        # Variation ratio
        predictions_class = probs.argmax(dim=-1)
        mode_count = torch.zeros_like(predictions_class[:, 0])
        for i in range(predictions_class.size(1)):
            mode_count += (predictions_class[:, i] == predictions_class[:, 0]).float()
        variation_ratio = 1.0 - mode_count / predictions_class.size(1)
        uncertainties["variation_ratio"] = variation_ratio
        
        return uncertainties


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for calibrated predictions.
    
    Based on Guo et al. (2017): "On Calibration of Modern Neural Networks"
    """
    
    def __init__(self, initial_temperature: float = 1.0):
        """
        Initialize temperature scaling.
        
        Args:
            initial_temperature: Initial temperature value
        """
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits
            
        Returns:
            Temperature-scaled logits
        """
        return logits / self.temperature
    
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ):
        """
        Optimize temperature on validation set.
        
        Args:
            logits: Validation logits
            labels: Validation labels
            lr: Learning rate
            max_iter: Maximum iterations
        """
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)
        
        def eval():
            loss = F.cross_entropy(self.forward(logits), labels)
            loss.backward()
            return loss
        
        optimizer.step(eval)


@ENSEMBLES.register("bayesian", aliases=["bayesian_ensemble", "bma"])
class BayesianEnsemble(BaseEnsemble):
    """
    Bayesian ensemble with uncertainty quantification.
    
    Implements Bayesian Model Averaging (BMA) with:
    1. Posterior weighting of models
    2. Uncertainty decomposition (aleatoric vs epistemic)
    3. Calibrated predictions
    4. MC Dropout approximation
    5. Deep ensembles for robustness
    
    The ensemble provides principled uncertainty estimates
    crucial for safety-critical applications.
    """
    
    def __init__(
        self,
        models: List[AGNewsBaseModel],
        config: Optional[BayesianEnsembleConfig] = None
    ):
        """
        Initialize Bayesian ensemble.
        
        Args:
            models: List of base models
            config: Bayesian ensemble configuration
        """
        super().__init__(models, config)
        
        self.config = config or BayesianEnsembleConfig()
        
        # Initialize Bayesian components
        self._init_bayesian_weights()
        self._init_uncertainty_estimator()
        self._init_calibration()
        
        # Model evidence cache
        self.model_evidence = torch.zeros(self.num_models)
        
        logger.info(
            f"Initialized BayesianEnsemble with {self.num_models} models "
            f"using {self.config.uncertainty_method} uncertainty"
        )
    
    def _init_bayesian_weights(self):
        """Initialize Bayesian model weights."""
        if self.config.use_posterior_weights:
            self.bayesian_weights = BayesianModelWeight(
                self.num_models,
                self.config.prior_type
            )
        else:
            # Use fixed weights
            self.weights = nn.Parameter(
                torch.ones(self.num_models) / self.num_models,
                requires_grad=False
            )
    
    def _init_uncertainty_estimator(self):
        """Initialize uncertainty estimation components."""
        self.uncertainty_estimator = UncertaintyEstimator(self.config)
    
    def _init_calibration(self):
        """Initialize calibration components."""
        if self.config.use_temperature_scaling:
            self.temperature_scaling = TemperatureScaling(
                self.config.initial_temperature
            )
    
    def _compute_model_evidence(
        self,
        model: AGNewsBaseModel,
        data_loader,
        method: str = "marginal_likelihood"
    ) -> float:
        """
        Compute model evidence for Bayesian model averaging.
        
        Args:
            model: Model to evaluate
            data_loader: Validation data
            method: Evidence computation method
            
        Returns:
            Model evidence value
        """
        model.eval()
        total_log_likelihood = 0
        num_samples = 0
        num_params = sum(p.numel() for p in model.parameters())
        
        with torch.no_grad():
            for batch in data_loader:
                outputs = model(
                    batch['input_ids'],
                    batch.get('attention_mask')
                )
                
                # Log likelihood
                log_probs = F.log_softmax(outputs.logits, dim=-1)
                log_likelihood = log_probs[
                    torch.arange(len(batch['labels'])),
                    batch['labels']
                ].sum()
                
                total_log_likelihood += log_likelihood.item()
                num_samples += len(batch['labels'])
        
        if method == "marginal_likelihood":
            # Approximate marginal likelihood
            evidence = total_log_likelihood
        elif method == "bic":
            # Bayesian Information Criterion
            evidence = total_log_likelihood - 0.5 * num_params * np.log(num_samples)
        elif method == "aic":
            # Akaike Information Criterion
            evidence = total_log_likelihood - num_params
        else:
            evidence = total_log_likelihood
        
        return evidence
    
    def update_posterior_weights(self, validation_loader):
        """
        Update posterior weights based on validation data.
        
        Args:
            validation_loader: Validation data loader
        """
        # Compute model evidence
        for i, model in enumerate(self.models):
            self.model_evidence[i] = self._compute_model_evidence(
                model,
                validation_loader,
                self.config.evidence_method
            )
        
        # Convert to posterior probabilities
        posterior_weights = F.softmax(self.model_evidence, dim=0)
        
        if self.config.use_posterior_weights:
            # Update Bayesian weights
            self.bayesian_weights.weight_mean.data = posterior_weights
        else:
            self.weights.data = posterior_weights
        
        logger.info(f"Updated posterior weights: {posterior_weights.tolist()}")
    
    def forward_with_uncertainty(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_samples: int = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            num_samples: Number of MC samples
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        num_samples = num_samples or self.config.num_mc_samples
        batch_size = input_ids.size(0)
        
        # Collect MC samples
        all_predictions = []
        
        for sample_idx in range(num_samples):
            # Sample model weights
            if self.config.use_posterior_weights:
                weights = self.bayesian_weights.sample_weights(1).squeeze(0)
            else:
                weights = self.weights
            
            # Get predictions from each model
            model_predictions = []
            
            for model_idx, model in enumerate(self.models):
                # Enable dropout for MC sampling
                if self.config.mc_dropout:
                    model.train()
                else:
                    model.eval()
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Apply temperature scaling
                if self.config.use_temperature_scaling:
                    logits = self.temperature_scaling(outputs.logits)
                else:
                    logits = outputs.logits
                
                model_predictions.append(logits)
            
            # Weighted combination
            model_predictions = torch.stack(model_predictions, dim=1)
            weighted_predictions = torch.sum(
                model_predictions * weights.view(1, -1, 1),
                dim=1
            )
            all_predictions.append(weighted_predictions)
        
        # Stack all MC samples
        all_predictions = torch.stack(all_predictions, dim=1)
        
        # Compute uncertainties
        uncertainties = self.uncertainty_estimator.estimate_uncertainty(
            all_predictions
        )
        
        # Mean prediction
        mean_prediction = all_predictions.mean(dim=1)
        
        return mean_prediction, uncertainties
    
    def combine_predictions(
        self,
        predictions: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Combine predictions using Bayesian model averaging.
        
        Args:
            predictions: List of model predictions
            
        Returns:
            Combined predictions
        """
        # Stack predictions
        stacked = torch.stack(predictions, dim=1)
        
        # Get weights
        if self.config.use_posterior_weights:
            weights = self.bayesian_weights.weight_mean
        else:
            weights = self.weights
        
        # Weighted average
        weights = weights.view(1, -1, 1)
        combined = torch.sum(stacked * weights, dim=1)
        
        # Apply temperature scaling
        if self.config.use_temperature_scaling:
            combined = self.temperature_scaling(combined)
        
        return combined
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_uncertainty: bool = True,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through Bayesian ensemble.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            return_uncertainty: Return uncertainty estimates
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with predictions and uncertainties
        """
        if return_uncertainty:
            # Forward with uncertainty
            logits, uncertainties = self.forward_with_uncertainty(
                input_ids,
                attention_mask
            )
            
            # Check abstention threshold
            if self.config.abstention_allowed:
                total_uncertainty = uncertainties["total_uncertainty"]
                abstain_mask = total_uncertainty > self.config.uncertainty_threshold
                
                # Set low confidence for uncertain predictions
                logits[abstain_mask] = 0.0
        else:
            # Standard forward
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None,
                return_individual=True,
                **kwargs
            )
            logits = outputs.logits
            uncertainties = None
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            
            # Add KL divergence if using Bayesian weights
            if self.config.use_posterior_weights:
                kl_loss = self.bayesian_weights.kl_divergence()
                loss = loss + 0.01 * kl_loss
        
        return ModelOutputs(
            logits=logits,
            loss=loss,
            metadata={
                "uncertainties": uncertainties,
                "model_evidence": self.model_evidence.tolist(),
                "posterior_weights": self.bayesian_weights.weight_mean.tolist() 
                    if self.config.use_posterior_weights else self.weights.tolist()
            }
        )
