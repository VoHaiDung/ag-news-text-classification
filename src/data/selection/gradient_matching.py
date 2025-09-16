"""
Gradient Matching Selection
============================

Selects training samples that match target gradients following:
- Mirzasoleiman et al. (2020): "Coresets for Data-efficient Training of Machine Learning Models"
- Killamsetty et al. (2021): "GRAD-MATCH: Gradient Matching based Data Subset Selection"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

class GradientMatching:
    """
    Gradient matching for subset selection.
    
    Implements CRAIG and GRAD-MATCH algorithms from:
    - Mirzasoleiman et al. (2020): "Coresets for Data-efficient Training"
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        device: Optional[torch.device] = None,
        per_class: bool = True
    ):
        """
        Initialize gradient matching selector.
        
        Args:
            model: Neural network model
            loss_fn: Loss function
            device: Computing device
            per_class: Perform per-class selection
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.per_class = per_class
        
        self.model.to(self.device)
        
        logger.info(f"Initialized gradient matching selector (per_class={per_class})")
    
    def select_subset(
        self,
        train_loader: DataLoader,
        subset_fraction: float = 0.1,
        val_loader: Optional[DataLoader] = None
    ) -> List[int]:
        """
        Select subset using gradient matching.
        
        Minimize: ||∇L_full - ∇L_subset||²
        
        Args:
            train_loader: Full training data loader
            subset_fraction: Fraction of data to select
            val_loader: Validation loader for gradient target
            
        Returns:
            Selected indices
        """
        n_select = int(len(train_loader.dataset) * subset_fraction)
        
        # Compute target gradient (from validation or full set)
        if val_loader:
            target_gradient = self._compute_dataset_gradient(val_loader)
        else:
            target_gradient = self._compute_dataset_gradient(train_loader)
        
        # Compute individual gradients
        sample_gradients = self._compute_sample_gradients(train_loader)
        
        # Greedy selection
        selected_indices = self._greedy_selection(
            sample_gradients,
            target_gradient,
            n_select
        )
        
        logger.info(f"Selected {len(selected_indices)} samples via gradient matching")
        
        return selected_indices
    
    def _compute_dataset_gradient(self, loader: DataLoader) -> torch.Tensor:
        """Compute average gradient over dataset."""
        self.model.eval()
        total_gradient = None
        n_samples = 0
        
        for batch_input, batch_label in loader:
            batch_input = batch_input.to(self.device)
            batch_label = batch_label.to(self.device)
            
            # Compute gradient
            self.model.zero_grad()
            output = self.model(batch_input)
            loss = self.loss_fn(output, batch_label)
            loss.backward()
            
            # Accumulate gradients
            if total_gradient is None:
                total_gradient = []
                for param in self.model.parameters():
                    if param.grad is not None:
                        total_gradient.append(param.grad.data.clone())
                    else:
                        total_gradient.append(torch.zeros_like(param.data))
            else:
                for i, param in enumerate(self.model.parameters()):
                    if param.grad is not None:
                        total_gradient[i] += param.grad.data
            
            n_samples += batch_input.size(0)
        
        # Average gradient
        for i in range(len(total_gradient)):
            total_gradient[i] /= n_samples
        
        # Flatten
        flat_gradient = torch.cat([g.flatten() for g in total_gradient])
        
        return flat_gradient
    
    def _compute_sample_gradients(self, loader: DataLoader) -> List[torch.Tensor]:
        """Compute gradient for each sample."""
        self.model.eval()
        sample_gradients = []
        
        for idx, (input_data, label) in enumerate(loader):
            if input_data.size(0) != 1:
                # Process single samples
                for i in range(input_data.size(0)):
                    single_input = input_data[i:i+1].to(self.device)
                    single_label = label[i:i+1].to(self.device)
                    
                    grad = self._compute_single_gradient(single_input, single_label)
                    sample_gradients.append(grad)
            else:
                input_data = input_data.to(self.device)
                label = label.to(self.device)
                
                grad = self._compute_single_gradient(input_data, label)
                sample_gradients.append(grad)
        
        return sample_gradients
    
    def _compute_single_gradient(
        self,
        input_data: torch.Tensor,
        label: torch.Tensor
    ) -> torch.Tensor:
        """Compute gradient for single sample."""
        self.model.zero_grad()
        
        output = self.model(input_data)
        loss = self.loss_fn(output, label)
        loss.backward()
        
        # Collect gradients
        gradients = []
        for param in self.model.parameters():
            if param.grad is not None:
                gradients.append(param.grad.data.clone().flatten())
            else:
                gradients.append(torch.zeros_like(param.data).flatten())
        
        return torch.cat(gradients)
    
    def _greedy_selection(
        self,
        sample_gradients: List[torch.Tensor],
        target_gradient: torch.Tensor,
        n_select: int
    ) -> List[int]:
        """
        Greedy selection to approximate target gradient.
        
        Following submodular optimization from:
        - Nemhauser et al. (1978): "An Analysis of Approximations for Maximizing Submodular Set Functions"
        """
        selected_indices = []
        selected_gradient = torch.zeros_like(target_gradient)
        remaining_indices = list(range(len(sample_gradients)))
        
        for _ in range(n_select):
            best_idx = None
            best_score = float('inf')
            
            # Find sample that best reduces gradient difference
            for idx in remaining_indices[:100]:  # Limit for efficiency
                # Compute new gradient if this sample is added
                new_gradient = selected_gradient + sample_gradients[idx]
                
                # Compute distance to target
                distance = torch.norm(target_gradient - new_gradient / (len(selected_indices) + 1))
                
                if distance < best_score:
                    best_score = distance
                    best_idx = idx
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_gradient += sample_gradients[best_idx]
                remaining_indices.remove(best_idx)
        
        return selected_indices
