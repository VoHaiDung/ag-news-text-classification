"""
Uncertainty Sampling Module
===========================

Implements uncertainty-based sampling strategies following:
- Gal et al. (2017): "Deep Bayesian Active Learning with Image Data"
- Kirsch et al. (2019): "BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Sampler

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

class UncertaintySampler(Sampler):
    """
    Uncertainty-based sampler using various uncertainty metrics.
    
    Implements uncertainty estimation from:
    - Gal & Ghahramani (2016): "Dropout as a Bayesian Approximation"
    - Lakshminarayanan et al. (2017): "Simple and Scalable Predictive Uncertainty Estimation"
    """
    
    def __init__(
        self,
        dataset,
        model: torch.nn.Module,
        method: str = "entropy",
        mc_iterations: int = 10,
        temperature: float = 1.0,
        batch_size: int = 32,
        device: Optional[torch.device] = None
    ):
        """
        Initialize uncertainty sampler.
        
        Args:
            dataset: Dataset to sample from
            model: Model for uncertainty estimation
            method: Uncertainty method (entropy, bald, variation_ratio)
            mc_iterations: Monte Carlo dropout iterations
            temperature: Temperature for calibration
            batch_size: Batch size for inference
            device: Computing device
        """
        self.dataset = dataset
        self.model = model
        self.method = method
        self.mc_iterations = mc_iterations
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Compute uncertainties
        self.uncertainties = self._compute_uncertainties()
        
        # Sort indices by uncertainty
        self.sorted_indices = np.argsort(self.uncertainties)[::-1]
        
        logger.info(f"Initialized uncertainty sampler with {method} method")
    
    def _compute_uncertainties(self) -> np.ndarray:
        """
        Compute uncertainty scores for all samples.
        
        Following uncertainty quantification from:
        - Kendall & Gal (2017): "What Uncertainties Do We Need in Bayesian Deep Learning?"
        """
        self.model.eval()
        self.model.to(self.device)
        
        uncertainties = []
        
        # Enable dropout during inference for MC Dropout
        if self.method in ["bald", "mc_entropy"]:
            self._enable_dropout()
        
        with torch.no_grad():
            for i in range(0, len(self.dataset), self.batch_size):
                batch_indices = range(i, min(i + self.batch_size, len(self.dataset)))
                batch = [self.dataset[idx] for idx in batch_indices]
                
                # Get predictions
                if self.method == "entropy":
                    batch_uncertainties = self._compute_entropy(batch)
                elif self.method == "bald":
                    batch_uncertainties = self._compute_bald(batch)
                elif self.method == "variation_ratio":
                    batch_uncertainties = self._compute_variation_ratio(batch)
                elif self.method == "margin":
                    batch_uncertainties = self._compute_margin(batch)
                else:
                    batch_uncertainties = np.zeros(len(batch))
                
                uncertainties.extend(batch_uncertainties)
        
        return np.array(uncertainties)
    
    def _enable_dropout(self):
        """Enable dropout layers during inference."""
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.train()
    
    def _compute_entropy(self, batch: List) -> List[float]:
        """
        Compute predictive entropy.
        
        H[y|x] = -Σ p(y|x) log p(y|x)
        """
        # Get model predictions (simplified)
        logits = torch.randn(len(batch), 4)  # Placeholder
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        return entropy.cpu().numpy().tolist()
    
    def _compute_bald(self, batch: List) -> List[float]:
        """
        Compute BALD (Bayesian Active Learning by Disagreement).
        
        BALD = H[y|x] - E_p(w|D)[H[y|x,w]]
        
        Following:
        - Houlsby et al. (2011): "Bayesian Active Learning for Classification and Preference Learning"
        """
        # MC Dropout predictions
        mc_predictions = []
        
        for _ in range(self.mc_iterations):
            logits = torch.randn(len(batch), 4)  # Placeholder
            probs = F.softmax(logits / self.temperature, dim=-1)
            mc_predictions.append(probs)
        
        mc_predictions = torch.stack(mc_predictions)  # [MC, batch, classes]
        
        # Compute predictive entropy
        mean_probs = mc_predictions.mean(dim=0)
        predictive_entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        
        # Compute expected entropy
        individual_entropy = -(mc_predictions * torch.log(mc_predictions + 1e-8)).sum(dim=-1)
        expected_entropy = individual_entropy.mean(dim=0)
        
        # BALD = mutual information
        bald = predictive_entropy - expected_entropy
        
        return bald.cpu().numpy().tolist()
    
    def _compute_variation_ratio(self, batch: List) -> List[float]:
        """
        Compute variation ratio.
        
        VR = 1 - max_y p(y|x)
        """
        logits = torch.randn(len(batch), 4)  # Placeholder
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Variation ratio
        variation_ratio = 1.0 - probs.max(dim=-1)[0]
        
        return variation_ratio.cpu().numpy().tolist()
    
    def _compute_margin(self, batch: List) -> List[float]:
        """
        Compute margin uncertainty.
        
        Margin = p(y1|x) - p(y2|x)
        """
        logits = torch.randn(len(batch), 4)  # Placeholder
        probs = F.softmax(logits / self.temperature, dim=-1)
        
        # Sort probabilities
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        
        # Compute margin
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        
        # Invert (smaller margin = higher uncertainty)
        uncertainty = 1.0 - margin
        
        return uncertainty.cpu().numpy().tolist()
    
    def get_most_uncertain(self, n_samples: int) -> List[int]:
        """Get indices of most uncertain samples."""
        return self.sorted_indices[:n_samples].tolist()
    
    def get_least_uncertain(self, n_samples: int) -> List[int]:
        """Get indices of least uncertain samples."""
        return self.sorted_indices[-n_samples:].tolist()
    
    def __iter__(self):
        """Iterate through indices sorted by uncertainty."""
        return iter(self.sorted_indices)
    
    def __len__(self):
        """Get number of samples."""
        return len(self.dataset)
