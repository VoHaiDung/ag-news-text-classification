"""
Snapshot Ensemble Implementation
=================================

Implementation of snapshot ensembling for creating diverse models from a single
training run, based on:
- Huang et al. (2017): "Snapshot Ensembles: Train 1, get M for free"
- Loshchilov & Hutter (2017): "SGDR: Stochastic Gradient Descent with Warm Restarts"
- Garipov et al. (2018): "Loss Surfaces, Mode Connectivity, and Fast Ensembling"

Mathematical Foundation:
Snapshot ensembling uses cyclic learning rate schedules:
η(t) = η_min + 0.5(η_max - η_min)(1 + cos(π * t_cur/T_cur))

This encourages the model to converge to different local minima,
creating diverse snapshots for ensembling.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import math
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from src.models.ensemble.base_ensemble import BaseEnsemble, EnsembleConfig
from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import ENSEMBLES
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class SnapshotEnsembleConfig(EnsembleConfig):
    """Configuration for snapshot ensemble."""
    # Training schedule
    num_cycles: int = 5  # Number of cosine annealing cycles
    cycle_length: int = 50  # Epochs per cycle
    lr_max: float = 0.1  # Maximum learning rate
    lr_min: float = 0.0001  # Minimum learning rate
    
    # Snapshot collection
    snapshot_interval: int = 1  # Collect snapshot every N cycles
    start_collecting: int = 1  # Start collecting after N cycles
    max_snapshots: int = 5  # Maximum snapshots to keep
    
    # Fast Geometric Ensembling (FGE)
    use_fge: bool = False  # Use FGE for connecting snapshots
    fge_samples: int = 3  # Number of samples along path
    
    # Stochastic Weight Averaging (SWA)
    use_swa: bool = True  # Use SWA for final model
    swa_start: int = 3  # Start SWA after N cycles
    swa_lr: float = 0.001  # SWA learning rate
    
    # Ensemble combination
    combination_method: str = "average"  # "average", "weighted", "voting"
    use_model_soup: bool = False  # Average weights directly
    
    # Diversity encouragement
    diversity_weight: float = 0.01
    use_different_augmentations: bool = True
    
    # Memory efficiency
    save_to_disk: bool = True
    snapshot_dir: str = "./snapshots"


class CosineAnnealingWithRestarts(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warm restarts.
    
    Implements SGDR (Stochastic Gradient Descent with Warm Restarts)
    for snapshot ensembling.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0,
        last_epoch: int = -1
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: Optimizer
            T_0: Initial cycle length
            T_mult: Cycle length multiplier
            eta_min: Minimum learning rate
            last_epoch: Last epoch
        """
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = 0
        self.T_i = T_0
        self.cycle = 0
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate for current step."""
        return [
            self.eta_min + (base_lr - self.eta_min) *
            (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
            for base_lr in self.base_lrs
        ]
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            
            if self.T_cur >= self.T_i:
                # Restart
                self.cycle += 1
                self.T_cur = 0
                self.T_i = self.T_i * self.T_mult
        else:
            if epoch < 0:
                raise ValueError("Epoch must be non-negative")
            
            self.cycle = 0
            self.T_i = self.T_0
            self.T_cur = epoch
            
            while self.T_cur >= self.T_i:
                self.T_cur -= self.T_i
                self.cycle += 1
                self.T_i *= self.T_mult
        
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
    
    def is_snapshot_point(self) -> bool:
        """Check if current point is good for snapshot."""
        # Take snapshot at the end of each cycle (minimum LR)
        return self.T_cur == self.T_i - 1


class StochasticWeightAveraging:
    """
    Stochastic Weight Averaging for improved generalization.
    
    Based on Izmailov et al. (2018): "Averaging Weights Leads to Wider Optima"
    """
    
    def __init__(self, base_model: nn.Module, swa_lr: float = 0.001):
        """
        Initialize SWA.
        
        Args:
            base_model: Base model to average
            swa_lr: SWA learning rate
        """
        self.base_model = base_model
        self.swa_model = copy.deepcopy(base_model)
        self.swa_n = 0
        self.swa_lr = swa_lr
    
    def update(self, model: nn.Module):
        """
        Update SWA model with new weights.
        
        Args:
            model: Model with new weights
        """
        self.swa_n += 1
        
        # Running average of weights
        for swa_param, param in zip(self.swa_model.parameters(), model.parameters()):
            swa_param.data = (
                swa_param.data * (self.swa_n - 1) + param.data
            ) / self.swa_n
    
    def get_model(self) -> nn.Module:
        """Get SWA averaged model."""
        return self.swa_model


class FastGeometricEnsembling:
    """
    Fast Geometric Ensembling for connecting model snapshots.
    
    Finds low-loss paths between snapshots in weight space.
    """
    
    def __init__(self, num_samples: int = 3):
        """
        Initialize FGE.
        
        Args:
            num_samples: Number of samples along path
        """
        self.num_samples = num_samples
    
    def connect_models(
        self,
        model1: nn.Module,
        model2: nn.Module,
        alpha_range: Tuple[float, float] = (0.2, 0.8)
    ) -> List[nn.Module]:
        """
        Connect two models with intermediate points.
        
        Args:
            model1: First model
            model2: Second model
            alpha_range: Range for interpolation
            
        Returns:
            List of interpolated models
        """
        models = []
        
        alphas = np.linspace(alpha_range[0], alpha_range[1], self.num_samples)
        
        for alpha in alphas:
            # Linear interpolation in weight space
            interpolated = copy.deepcopy(model1)
            
            for p_inter, p1, p2 in zip(
                interpolated.parameters(),
                model1.parameters(),
                model2.parameters()
            ):
                p_inter.data = (1 - alpha) * p1.data + alpha * p2.data
            
            models.append(interpolated)
        
        return models


@ENSEMBLES.register("snapshot", aliases=["snapshot_ensemble", "sgdr"])
class SnapshotEnsemble(BaseEnsemble):
    """
    Snapshot ensemble that collects models during cyclic training.
    
    Key features:
    1. Cyclic learning rate schedule with warm restarts
    2. Automatic snapshot collection at convergence points
    3. Optional SWA for final model averaging
    4. FGE for finding intermediate models
    5. Memory-efficient snapshot storage
    
    The ensemble achieves diversity through exploring different
    local minima during training cycles.
    """
    
    def __init__(
        self,
        base_model: AGNewsBaseModel,
        config: Optional[SnapshotEnsembleConfig] = None
    ):
        """
        Initialize snapshot ensemble.
        
        Args:
            base_model: Base model to train
            config: Snapshot ensemble configuration
        """
        # Initialize with empty model list (will be filled during training)
        super().__init__([], config)
        
        self.config = config or SnapshotEnsembleConfig()
        self.base_model = base_model
        
        # Snapshot storage
        self.snapshots = []
        self.snapshot_epochs = []
        self.snapshot_metrics = []
        
        # Initialize components
        self._init_scheduler_params()
        self._init_swa()
        self._init_fge()
        
        # Create snapshot directory
        if self.config.save_to_disk:
            self.snapshot_dir = Path(self.config.snapshot_dir)
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"Initialized SnapshotEnsemble with {config.num_cycles} cycles, "
            f"collecting up to {config.max_snapshots} snapshots"
        )
    
    def _init_scheduler_params(self):
        """Initialize scheduler parameters."""
        self.current_cycle = 0
        self.current_epoch = 0
        self.snapshots_collected = 0
    
    def _init_swa(self):
        """Initialize Stochastic Weight Averaging."""
        if self.config.use_swa:
            self.swa = StochasticWeightAveraging(
                self.base_model,
                self.config.swa_lr
            )
    
    def _init_fge(self):
        """Initialize Fast Geometric Ensembling."""
        if self.config.use_fge:
            self.fge = FastGeometricEnsembling(self.config.fge_samples)
    
    def create_scheduler(self, optimizer: Optimizer) -> _LRScheduler:
        """
        Create learning rate scheduler for snapshot training.
        
        Args:
            optimizer: Optimizer
            
        Returns:
            Learning rate scheduler
        """
        scheduler = CosineAnnealingWithRestarts(
            optimizer,
            T_0=self.config.cycle_length,
            T_mult=1,
            eta_min=self.config.lr_min
        )
        
        return scheduler
    
    def should_take_snapshot(self, epoch: int, scheduler: _LRScheduler) -> bool:
        """
        Determine if snapshot should be taken.
        
        Args:
            epoch: Current epoch
            scheduler: Learning rate scheduler
            
        Returns:
            Whether to take snapshot
        """
        # Check if at snapshot point in schedule
        if not scheduler.is_snapshot_point():
            return False
        
        # Check if should start collecting
        current_cycle = epoch // self.config.cycle_length
        if current_cycle < self.config.start_collecting:
            return False
        
        # Check snapshot interval
        if (current_cycle - self.config.start_collecting) % self.config.snapshot_interval != 0:
            return False
        
        # Check maximum snapshots
        if len(self.snapshots) >= self.config.max_snapshots:
            return False
        
        return True
    
    def take_snapshot(
        self,
        model: nn.Module,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Take a snapshot of current model.
        
        Args:
            model: Model to snapshot
            epoch: Current epoch
            metrics: Optional metrics for this snapshot
        """
        # Create snapshot
        snapshot = copy.deepcopy(model)
        
        # Store snapshot
        if self.config.save_to_disk:
            # Save to disk for memory efficiency
            snapshot_path = self.snapshot_dir / f"snapshot_epoch_{epoch}.pt"
            torch.save(snapshot.state_dict(), snapshot_path)
            self.snapshots.append(str(snapshot_path))
        else:
            # Keep in memory
            self.snapshots.append(snapshot)
        
        self.snapshot_epochs.append(epoch)
        self.snapshot_metrics.append(metrics or {})
        self.snapshots_collected += 1
        
        # Update SWA if enabled
        if self.config.use_swa and epoch >= self.config.swa_start * self.config.cycle_length:
            self.swa.update(model)
        
        logger.info(f"Took snapshot {self.snapshots_collected} at epoch {epoch}")
    
    def load_snapshots(self) -> List[nn.Module]:
        """
        Load all snapshots into memory.
        
        Returns:
            List of snapshot models
        """
        models = []
        
        for snapshot in self.snapshots:
            if isinstance(snapshot, str):
                # Load from disk
                model = copy.deepcopy(self.base_model)
                model.load_state_dict(torch.load(snapshot))
                models.append(model)
            else:
                # Already in memory
                models.append(snapshot)
        
        # Add FGE models if enabled
        if self.config.use_fge and len(models) > 1:
            fge_models = []
            for i in range(len(models) - 1):
                intermediate = self.fge.connect_models(models[i], models[i+1])
                fge_models.extend(intermediate)
            models.extend(fge_models)
        
        # Add SWA model if available
        if self.config.use_swa and hasattr(self, 'swa'):
            models.append(self.swa.get_model())
        
        return models
    
    def combine_predictions(
        self,
        predictions: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Combine snapshot predictions.
        
        Args:
            predictions: List of model predictions
            
        Returns:
            Combined predictions
        """
        if self.config.combination_method == "average":
            # Simple average
            combined = torch.stack(predictions).mean(dim=0)
            
        elif self.config.combination_method == "weighted":
            # Weight by validation metrics if available
            if self.snapshot_metrics and all('accuracy' in m for m in self.snapshot_metrics):
                weights = torch.tensor([m['accuracy'] for m in self.snapshot_metrics])
                weights = F.softmax(weights, dim=0)
                
                stacked = torch.stack(predictions)
                combined = torch.sum(stacked * weights.view(-1, 1, 1), dim=0)
            else:
                # Fall back to average
                combined = torch.stack(predictions).mean(dim=0)
                
        elif self.config.combination_method == "voting":
            # Majority voting
            votes = torch.stack([p.argmax(dim=-1) for p in predictions])
            combined_classes = torch.mode(votes, dim=0)[0]
            
            # Convert to logits
            num_classes = predictions[0].size(-1)
            combined = F.one_hot(combined_classes, num_classes).float()
            combined = torch.log(combined + 1e-10)
        
        else:
            combined = torch.stack(predictions).mean(dim=0)
        
        return combined
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through snapshot ensemble.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            **kwargs: Additional arguments
            
        Returns:
            Ensemble predictions
        """
        # Load snapshots if not already loaded
        if not self.models:
            self.models = nn.ModuleList(self.load_snapshots())
            self.num_models = len(self.models)
        
        # Get predictions from all snapshots
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def create_model_soup(self) -> nn.Module:
        """
        Create model soup by averaging snapshot weights.
        
        Returns:
            Model with averaged weights
        """
        models = self.load_snapshots()
        
        if not models:
            raise ValueError("No snapshots available for model soup")
        
        # Average weights directly
        soup_model = copy.deepcopy(models[0])
        
        for param in soup_model.parameters():
            param.data.zero_()
        
        for model in models:
            for soup_param, param in zip(soup_model.parameters(), model.parameters()):
                soup_param.data += param.data / len(models)
        
        logger.info(f"Created model soup from {len(models)} snapshots")
        
        return soup_model
