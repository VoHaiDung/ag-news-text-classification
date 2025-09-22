"""
Curriculum Learning Implementation
===================================

Implementation of curriculum learning for improved training efficiency,
based on:
- Bengio et al. (2009): "Curriculum Learning"
- Hacohen & Weinshall (2019): "On The Power of Curriculum Learning in Training Deep Networks"
- Platanios et al. (2019): "Competence-based Curriculum Learning"

Mathematical Foundation:
Curriculum learning optimizes: L_t = Σ_i∈D_t w_i ℓ(f_θ(x_i), y_i)
where D_t is the subset of training data at time t, ordered by difficulty.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from abc import ABC, abstractmethod

from src.training.trainers.base_trainer import BaseTrainer, TrainerConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CurriculumConfig(TrainerConfig):
    """Configuration for curriculum learning."""
    # Curriculum strategy
    curriculum_type: str = "fixed"  # "fixed", "adaptive", "self_paced"
    
    # Pacing function
    pacing_function: str = "linear"  # "linear", "exponential", "step", "root"
    initial_competence: float = 0.2  # Start with 20% easiest samples
    final_competence: float = 1.0  # End with all samples
    
    # Difficulty scoring
    difficulty_metric: str = "loss"  # "loss", "length", "confidence", "gradient"
    difficulty_percentile: bool = True  # Use percentile-based difficulty
    
    # Adaptive curriculum
    adaptive_window: int = 100  # Window for adaptive difficulty estimation
    competence_threshold: float = 0.9  # Threshold for increasing difficulty
    
    # Self-paced learning
    self_paced_weight: float = 1.0  # Weight for self-paced regularization
    self_paced_step_size: float = 0.1  # Step size for self-paced updates
    
    # Curriculum schedule
    curriculum_epochs: int = 5  # Epochs for curriculum before full training
    warmup_epochs: int = 1  # Initial warmup with easiest samples


class DifficultyScorer(ABC):
    """
    Abstract base class for sample difficulty scoring.
    
    Defines interface for computing difficulty scores for training samples.
    """
    
    @abstractmethod
    def score(
        self,
        model: nn.Module,
        data: Union[Dataset, DataLoader],
        device: str = "cuda"
    ) -> np.ndarray:
        """
        Compute difficulty scores for samples.
        
        Args:
            model: Model to use for scoring
            data: Dataset or dataloader
            device: Device to use
            
        Returns:
            Array of difficulty scores
        """
        pass


class LossDifficultyScorer(DifficultyScorer):
    """
    Score difficulty based on model loss.
    
    Higher loss indicates higher difficulty.
    """
    
    def score(
        self,
        model: nn.Module,
        data: Union[Dataset, DataLoader],
        device: str = "cuda"
    ) -> np.ndarray:
        """Score samples by loss."""
        model.eval()
        model = model.to(device)
        
        # Create dataloader if needed
        if isinstance(data, Dataset):
            dataloader = DataLoader(data, batch_size=32, shuffle=False)
        else:
            dataloader = data
        
        scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                inputs = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # Compute loss per sample
                outputs = model(inputs, attention_mask=attention_mask, labels=labels)
                
                # Get per-sample loss
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(reduction='none')
                per_sample_loss = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1)
                )
                
                scores.extend(per_sample_loss.cpu().numpy())
        
        return np.array(scores)


class LengthDifficultyScorer(DifficultyScorer):
    """
    Score difficulty based on sequence length.
    
    Longer sequences are considered more difficult.
    """
    
    def score(
        self,
        model: nn.Module,
        data: Union[Dataset, DataLoader],
        device: str = "cuda"
    ) -> np.ndarray:
        """Score samples by length."""
        if isinstance(data, DataLoader):
            dataset = data.dataset
        else:
            dataset = data
        
        scores = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            # Count non-padding tokens
            if isinstance(sample["input_ids"], torch.Tensor):
                length = (sample["input_ids"] != 0).sum().item()
            else:
                length = len([t for t in sample["input_ids"] if t != 0])
            scores.append(length)
        
        return np.array(scores)


class ConfidenceDifficultyScorer(DifficultyScorer):
    """
    Score difficulty based on model confidence.
    
    Lower confidence (higher entropy) indicates higher difficulty.
    """
    
    def score(
        self,
        model: nn.Module,
        data: Union[Dataset, DataLoader],
        device: str = "cuda"
    ) -> np.ndarray:
        """Score samples by prediction confidence."""
        model.eval()
        model = model.to(device)
        
        if isinstance(data, Dataset):
            dataloader = DataLoader(data, batch_size=32, shuffle=False)
        else:
            dataloader = data
        
        scores = []
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch["input_ids"].to(device)
                attention_mask = batch.get("attention_mask")
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                outputs = model(inputs, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # Compute entropy as difficulty measure
                entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                scores.extend(entropy.cpu().numpy())
        
        return np.array(scores)


class PacingFunction:
    """
    Pacing function for curriculum learning.
    
    Controls the rate at which difficulty increases during training.
    """
    
    @staticmethod
    def linear(t: float, initial: float = 0.2, final: float = 1.0) -> float:
        """Linear pacing: competence increases linearly."""
        return initial + (final - initial) * t
    
    @staticmethod
    def exponential(t: float, initial: float = 0.2, final: float = 1.0) -> float:
        """Exponential pacing: slow start, rapid increase."""
        return initial + (final - initial) * (np.exp(t) - 1) / (np.e - 1)
    
    @staticmethod
    def step(t: float, initial: float = 0.2, final: float = 1.0, steps: int = 5) -> float:
        """Step pacing: discrete difficulty levels."""
        step_size = (final - initial) / steps
        current_step = int(t * steps)
        return min(initial + current_step * step_size, final)
    
    @staticmethod
    def root(t: float, initial: float = 0.2, final: float = 1.0, power: float = 2.0) -> float:
        """Root pacing: rapid start, slow increase."""
        return initial + (final - initial) * (t ** (1 / power))


class CurriculumLearning(BaseTrainer):
    """
    Curriculum learning trainer.
    
    Implements various curriculum learning strategies:
    1. Fixed curriculum: Pre-defined difficulty ordering
    2. Adaptive curriculum: Dynamic difficulty adjustment
    3. Self-paced learning: Model determines its own curriculum
    
    The trainer gradually increases training difficulty for improved
    convergence and generalization.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[CurriculumConfig] = None,
        difficulty_scorer: Optional[DifficultyScorer] = None,
        **kwargs
    ):
        """
        Initialize curriculum learning trainer.
        
        Args:
            model: Model to train
            config: Curriculum configuration
            difficulty_scorer: Difficulty scoring strategy
            **kwargs: Additional trainer arguments
        """
        config = config or CurriculumConfig()
        super().__init__(model, config, **kwargs)
        
        self.config: CurriculumConfig = config
        
        # Initialize difficulty scorer
        if difficulty_scorer is None:
            if config.difficulty_metric == "loss":
                self.difficulty_scorer = LossDifficultyScorer()
            elif config.difficulty_metric == "length":
                self.difficulty_scorer = LengthDifficultyScorer()
            elif config.difficulty_metric == "confidence":
                self.difficulty_scorer = ConfidenceDifficultyScorer()
            else:
                self.difficulty_scorer = LossDifficultyScorer()
        else:
            self.difficulty_scorer = difficulty_scorer
        
        # Initialize pacing function
        self.pacing_functions = {
            "linear": PacingFunction.linear,
            "exponential": PacingFunction.exponential,
            "step": PacingFunction.step,
            "root": PacingFunction.root
        }
        self.pacing_function = self.pacing_functions.get(
            config.pacing_function,
            PacingFunction.linear
        )
        
        # Curriculum state
        self.sample_difficulties = None
        self.curriculum_indices = None
        self.current_competence = config.initial_competence
        
        logger.info(
            f"Initialized CurriculumLearning with "
            f"{config.curriculum_type} curriculum and "
            f"{config.difficulty_metric} difficulty scoring"
        )
    
    def _compute_sample_difficulties(self):
        """Compute difficulty scores for all training samples."""
        logger.info("Computing sample difficulties...")
        
        # Score all samples
        self.sample_difficulties = self.difficulty_scorer.score(
            self.model,
            self.train_loader.dataset,
            self.device
        )
        
        # Sort indices by difficulty
        self.curriculum_indices = np.argsort(self.sample_difficulties)
        
        logger.info(
            f"Computed difficulties for {len(self.sample_difficulties)} samples"
        )
    
    def _get_curriculum_subset(self, epoch: int) -> Subset:
        """
        Get training subset based on curriculum.
        
        Args:
            epoch: Current epoch
            
        Returns:
            Subset of training data
        """
        # Compute current competence level
        if epoch < self.config.curriculum_epochs:
            t = epoch / self.config.curriculum_epochs
            self.current_competence = self.pacing_function(
                t,
                self.config.initial_competence,
                self.config.final_competence
            )
        else:
            self.current_competence = self.config.final_competence
        
        # Select samples based on competence
        num_samples = int(len(self.curriculum_indices) * self.current_competence)
        selected_indices = self.curriculum_indices[:num_samples]
        
        # Create subset
        subset = Subset(self.train_loader.dataset, selected_indices.tolist())
        
        logger.info(
            f"Curriculum at epoch {epoch}: "
            f"competence={self.current_competence:.2f}, "
            f"samples={num_samples}/{len(self.curriculum_indices)}"
        )
        
        return subset
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with curriculum.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics for the epoch
        """
        # Compute difficulties if not done
        if self.sample_difficulties is None:
            self._compute_sample_difficulties()
        
        # Get curriculum subset
        if self.config.curriculum_type == "adaptive":
            # Update difficulties periodically
            if epoch % self.config.adaptive_window == 0 and epoch > 0:
                self._compute_sample_difficulties()
        
        subset = self._get_curriculum_subset(epoch)
        
        # Create temporary dataloader with subset
        original_loader = self.train_loader
        self.train_loader = DataLoader(
            subset,
            batch_size=original_loader.batch_size,
            shuffle=True,
            num_workers=original_loader.num_workers,
            pin_memory=original_loader.pin_memory
        )
        
        # Train on curriculum subset
        metrics = super()._train_epoch(epoch)
        
        # Restore original loader
        self.train_loader = original_loader
        
        # Add curriculum-specific metrics
        metrics["curriculum_competence"] = self.current_competence
        metrics["curriculum_samples"] = len(subset)
        
        return metrics
    
    def self_paced_update(self, epoch: int):
        """
        Update curriculum using self-paced learning.
        
        The model determines which samples to include based on current performance.
        """
        if self.config.curriculum_type != "self_paced":
            return
        
        # Re-score samples with current model
        self._compute_sample_difficulties()
        
        # Self-paced learning: include samples with loss below threshold
        threshold = np.percentile(
            self.sample_difficulties,
            self.current_competence * 100
        )
        
        # Gradually increase threshold
        self.current_competence = min(
            self.current_competence + self.config.self_paced_step_size,
            self.config.final_competence
        )
        
        logger.info(
            f"Self-paced update: threshold={threshold:.4f}, "
            f"competence={self.current_competence:.2f}"
        )
