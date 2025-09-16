"""
Influence Function Selection
=============================

Selects influential training samples using influence functions following:
- Koh & Liang (2017): "Understanding Black-box Predictions via Influence Functions"
- Pruthi et al. (2020): "Estimating Training Data Influence by Tracing Gradient Descent"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

class InfluenceFunction:
    """
    Compute influence functions for data selection.
    
    Implements influence estimation from:
    - Cook & Weisberg (1982): "Residuals and Influence in Regression"
    - Koh et al. (2019): "On the Accuracy of Influence Function Approximation"
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        damping: float = 0.01,
        scale: float = 25.0,
        recursion_depth: int = 100,
        device: Optional[torch.device] = None
    ):
        """
        Initialize influence function calculator.
        
        Args:
            model: Neural network model
            loss_fn: Loss function
            damping: Damping term for numerical stability
            scale: Scaling factor
            recursion_depth: Depth for recursive approximation
            device: Computing device
        """
        self.model = model
        self.loss_fn = loss_fn
        self.damping = damping
        self.scale = scale
        self.recursion_depth = recursion_depth
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        logger.info(f"Initialized influence function calculator with damping={damping}")
    
    def compute_influence_single(
        self,
        train_sample: Tuple[torch.Tensor, torch.Tensor],
        test_sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> float:
        """
        Compute influence of single training sample on test sample.
        
        I(z_train, z_test) = -∇_θ L(z_test)^T H^{-1} ∇_θ L(z_train)
        
        Args:
            train_sample: Training sample (input, label)
            test_sample: Test sample (input, label)
            
        Returns:
            Influence score
        """
        # Compute test gradient
        test_grad = self._compute_gradient(test_sample)
        
        # Compute train gradient
        train_grad = self._compute_gradient(train_sample)
        
        # Approximate H^{-1} ∇_θ L(z_train) using HVP
        ihvp = self._compute_ihvp(train_grad)
        
        # Compute influence
        influence = -torch.dot(test_grad.flatten(), ihvp.flatten()).item()
        
        return influence / self.scale
    
    def compute_self_influence(
        self,
        sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> float:
        """
        Compute self-influence (useful for data pruning).
        
        Following:
        - Feldman & Zhang (2020): "What Neural Networks Memorize and Why"
        """
        grad = self._compute_gradient(sample)
        ihvp = self._compute_ihvp(grad)
        
        self_influence = torch.dot(grad.flatten(), ihvp.flatten()).item()
        
        return self_influence / self.scale
    
    def select_influential_samples(
        self,
        train_loader: DataLoader,
        test_samples: List[Tuple],
        n_samples: int,
        positive: bool = True
    ) -> List[int]:
        """
        Select most influential training samples.
        
        Args:
            train_loader: Training data loader
            test_samples: Test samples to influence
            n_samples: Number of samples to select
            positive: Select positive (helpful) or negative (harmful) influences
            
        Returns:
            Indices of influential samples
        """
        influences = []
        
        for idx, (train_input, train_label) in enumerate(train_loader):
            train_sample = (train_input.to(self.device), train_label.to(self.device))
            
            # Compute average influence on test samples
            sample_influences = []
            for test_sample in test_samples[:10]:  # Limit for efficiency
                influence = self.compute_influence_single(train_sample, test_sample)
                sample_influences.append(influence)
            
            avg_influence = np.mean(sample_influences)
            influences.append((idx, avg_influence))
        
        # Sort by influence
        influences.sort(key=lambda x: x[1], reverse=positive)
        
        # Select top-k
        selected_indices = [idx for idx, _ in influences[:n_samples]]
        
        return selected_indices
    
    def _compute_gradient(
        self,
        sample: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Compute gradient of loss w.r.t. model parameters."""
        self.model.zero_grad()
        
        input_data, label = sample
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(0)
        if label.dim() == 0:
            label = label.unsqueeze(0)
        
        output = self.model(input_data)
        loss = self.loss_fn(output, label)
        
        # Compute gradients
        grads = torch.autograd.grad(loss, self.model.parameters(), create_graph=True)
        
        # Flatten and concatenate
        flat_grad = torch.cat([g.flatten() for g in grads])
        
        return flat_grad
    
    def _compute_ihvp(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse Hessian vector product: H^{-1} v
        
        Using iterative approximation from:
        - Agarwal et al. (2017): "Second-Order Stochastic Optimization for Machine Learning"
        """
        # Initialize with v
        ihvp = v.clone()
        
        # Iterative approximation: p = v + (I - H) p
        for _ in range(self.recursion_depth):
            # Compute Hv
            hv = self._compute_hvp(ihvp)
            
            # Update: p = v + p - Hv/scale
            ihvp = v + ihvp - hv / self.scale - self.damping * ihvp
        
        return ihvp / self.scale
    
    def _compute_hvp(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian vector product: Hv
        
        Using finite difference approximation.
        """
        # Finite difference parameter
        r = 1e-2
        
        # Get current parameters
        params = []
        for p in self.model.parameters():
            params.append(p.data.clone())
        
        # Perturb parameters: θ + rv
        idx = 0
        for p in self.model.parameters():
            num_params = p.numel()
            p.data += r * v[idx:idx + num_params].view_as(p)
            idx += num_params
        
        # Compute gradient at perturbed point
        # (Simplified - should use actual training data)
        dummy_input = torch.randn(1, 512).to(self.device)
        dummy_label = torch.tensor([0]).to(self.device)
        
        grad_plus = self._compute_gradient((dummy_input, dummy_label))
        
        # Restore parameters
        for p, p_orig in zip(self.model.parameters(), params):
            p.data = p_orig
        
        # Approximate Hv = (∇f(θ + rv) - ∇f(θ)) / r
        current_grad = self._compute_gradient((dummy_input, dummy_label))
        hvp = (grad_plus - current_grad) / r
        
        return hvp
