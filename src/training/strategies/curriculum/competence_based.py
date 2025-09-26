"""
Competence-based Curriculum Learning Implementation
====================================================

Implementation of competence-based curriculum learning for adaptive training,
based on:
- Platanios et al. (2019): "Competence-based Curriculum Learning for Neural Machine Translation"
- Graves et al. (2017): "Automated Curriculum Learning for Neural Networks"
- Soviany et al. (2021): "Curriculum Learning: A Survey"

Mathematical Foundation:
Competence function c(t) determines data sampling:
p(x|t) ∝ exp(-difficulty(x) / c(t))
where c(t) increases over time, allowing harder samples.

Key Innovation: Automatic difficulty estimation and competence progression
based on model performance.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from scipy.stats import norm
import math

from src.training.trainers.base_trainer import BaseTrainer, TrainerConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CompetenceConfig(TrainerConfig):
    """Configuration for competence-based curriculum learning."""
    
    # Competence function
    competence_type: str = "linear"  # "linear", "root", "geometric", "adaptive"
    initial_competence: float = 0.01  # Starting competence level
    target_competence: float = 1.0  # Final competence level
    
    # Competence progression
    progression_type: str = "performance"  # "fixed", "performance", "uncertainty"
    competence_increment: float = 0.05  # Increment per update
    competence_patience: int = 3  # Epochs before increasing competence
    performance_threshold: float = 0.85  # Performance threshold for progression
    
    # Difficulty estimation
    difficulty_estimator: str = "neural"  # "neural", "statistical", "hybrid"
    difficulty_features: List[str] = field(default_factory=lambda: [
        "length", "vocabulary", "syntax", "semantics"
    ])
    difficulty_update_freq: int = 5  # Update difficulty every N epochs
    
    # Sampling strategy
    sampling_type: str = "probabilistic"  # "threshold", "probabilistic", "ranked"
    sampling_temperature: float = 1.0  # Temperature for probabilistic sampling
    min_batch_competence: float = 0.5  # Minimum competence for batch
    
    # Competence metrics
    use_validation_competence: bool = True  # Use validation for competence
    competence_window: int = 5  # Window for competence estimation
    
    # Advanced features
    multi_objective: bool = False  # Multi-objective competence
    task_weights: Optional[Dict[str, float]] = None  # Task-specific weights
    
    # Regularization
    competence_smoothing: float = 0.1  # Smoothing factor for competence
    difficulty_smoothing: float = 0.1  # Smoothing factor for difficulty


class CompetenceFunction:
    """
    Competence functions for curriculum progression.
    
    Controls how quickly the model's competence increases.
    """
    
    @staticmethod
    def linear(t: float, c0: float = 0.01, c1: float = 1.0) -> float:
        """Linear competence growth: c(t) = c0 + (c1 - c0) * t"""
        return c0 + (c1 - c0) * t
    
    @staticmethod
    def root(t: float, c0: float = 0.01, c1: float = 1.0, p: float = 2.0) -> float:
        """Root competence growth: c(t) = c0 + (c1 - c0) * t^(1/p)"""
        return c0 + (c1 - c0) * (t ** (1.0 / p))
    
    @staticmethod
    def geometric(t: float, c0: float = 0.01, c1: float = 1.0) -> float:
        """Geometric competence growth: c(t) = c0 * (c1/c0)^t"""
        if c0 <= 0:
            c0 = 0.01
        return c0 * ((c1 / c0) ** t)
    
    @staticmethod
    def adaptive(
        t: float,
        performance: float,
        c_current: float,
        increment: float = 0.05
    ) -> float:
        """
        Adaptive competence based on performance.
        
        Increases competence when model performs well.
        """
        if performance > 0.85:
            return min(c_current + increment * 2, 1.0)
        elif performance > 0.7:
            return min(c_current + increment, 1.0)
        else:
            return c_current  # No increase if performance is low


class DifficultyEstimator:
    """
    Estimates sample difficulty for competence-based selection.
    
    Uses various features to determine how difficult a sample is.
    """
    
    def __init__(self, estimator_type: str = "neural", features: List[str] = None):
        """
        Initialize difficulty estimator.
        
        Args:
            estimator_type: Type of estimator
            features: Features to use for difficulty
        """
        self.estimator_type = estimator_type
        self.features = features or ["length", "vocabulary", "syntax"]
        
        # Neural difficulty estimator
        if estimator_type == "neural":
            self.difficulty_net = nn.Sequential(
                nn.Linear(len(self.features), 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
    
    def extract_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract difficulty features from batch.
        
        Args:
            batch: Input batch
            
        Returns:
            Feature tensor [batch_size, num_features]
        """
        features = []
        
        # Length feature
        if "length" in self.features:
            lengths = (batch["input_ids"] != 0).sum(dim=1).float()
            lengths = lengths / lengths.max()  # Normalize
            features.append(lengths.unsqueeze(1))
        
        # Vocabulary complexity (rare tokens)
        if "vocabulary" in self.features:
            # Simple proxy: high token IDs are often rarer
            vocab_complexity = (batch["input_ids"] > 10000).float().mean(dim=1)
            features.append(vocab_complexity.unsqueeze(1))
        
        # Syntactic complexity (proxy: punctuation density)
        if "syntax" in self.features:
            # Punctuation tokens (simplified)
            punct_tokens = [1012, 1013, 1014, 1025, 1026]  # Common punctuation IDs
            syntax_complexity = torch.zeros(batch["input_ids"].size(0))
            for token_id in punct_tokens:
                syntax_complexity += (batch["input_ids"] == token_id).float().sum(dim=1)
            syntax_complexity = syntax_complexity / batch["input_ids"].size(1)
            features.append(syntax_complexity.unsqueeze(1))
        
        # Semantic complexity (proxy: attention entropy)
        if "semantics" in self.features and "attention_mask" in batch:
            # Use attention mask density as proxy
            semantic_complexity = batch["attention_mask"].float().mean(dim=1)
            features.append(semantic_complexity.unsqueeze(1))
        
        return torch.cat(features, dim=1)
    
    def estimate(
        self,
        batch: Dict[str, torch.Tensor],
        model_loss: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Estimate difficulty for samples.
        
        Args:
            batch: Input batch
            model_loss: Optional model loss for hybrid estimation
            
        Returns:
            Difficulty scores [batch_size]
        """
        if self.estimator_type == "neural":
            features = self.extract_features(batch)
            difficulties = self.difficulty_net(features).squeeze()
            
        elif self.estimator_type == "statistical":
            features = self.extract_features(batch)
            # Simple statistical combination
            difficulties = features.mean(dim=1)
            
        elif self.estimator_type == "hybrid":
            features = self.extract_features(batch)
            feature_difficulty = features.mean(dim=1)
            
            if model_loss is not None:
                # Combine feature-based and loss-based difficulty
                loss_difficulty = torch.sigmoid(model_loss)
                difficulties = 0.5 * feature_difficulty + 0.5 * loss_difficulty
            else:
                difficulties = feature_difficulty
        
        else:
            # Default: use length as difficulty
            difficulties = (batch["input_ids"] != 0).sum(dim=1).float()
            difficulties = difficulties / difficulties.max()
        
        return difficulties


class CompetenceBasedCurriculum(BaseTrainer):
    """
    Competence-based curriculum learning trainer.
    
    Automatically adjusts curriculum based on model competence,
    selecting samples that match current learning capacity.
    
    Key innovations:
    1. Automatic difficulty estimation
    2. Performance-based competence progression
    3. Probabilistic sampling based on competence
    4. Multi-objective competence for multiple tasks
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Optional[CompetenceConfig] = None,
        **kwargs
    ):
        """
        Initialize competence-based curriculum trainer.
        
        Args:
            model: Model to train
            config: Competence configuration
            **kwargs: Additional trainer arguments
        """
        config = config or CompetenceConfig()
        super().__init__(model, config, **kwargs)
        
        self.config: CompetenceConfig = config
        
        # Initialize competence function
        self.competence_functions = {
            "linear": CompetenceFunction.linear,
            "root": CompetenceFunction.root,
            "geometric": CompetenceFunction.geometric,
            "adaptive": CompetenceFunction.adaptive
        }
        self.competence_function = self.competence_functions[config.competence_type]
        
        # Initialize difficulty estimator
        self.difficulty_estimator = DifficultyEstimator(
            config.difficulty_estimator,
            config.difficulty_features
        )
        
        # Competence state
        self.current_competence = config.initial_competence
        self.competence_history = []
        self.performance_history = []
        self.patience_counter = 0
        
        # Sample difficulties cache
        self.sample_difficulties = {}
        self.difficulty_update_counter = 0
        
        logger.info(
            f"Initialized CompetenceBasedCurriculum with "
            f"{config.competence_type} competence function, "
            f"initial competence={self.current_competence:.3f}"
        )
    
    def update_competence(self, performance: float, epoch: int):
        """
        Update model competence based on performance.
        
        Args:
            performance: Current model performance
            epoch: Current epoch
        """
        self.performance_history.append(performance)
        
        if self.config.progression_type == "fixed":
            # Fixed progression
            progress = epoch / self.config.num_epochs
            self.current_competence = self.competence_function(
                progress,
                self.config.initial_competence,
                self.config.target_competence
            )
            
        elif self.config.progression_type == "performance":
            # Performance-based progression
            if performance >= self.config.performance_threshold:
                self.patience_counter += 1
                
                if self.patience_counter >= self.config.competence_patience:
                    # Increase competence
                    self.current_competence = min(
                        self.current_competence + self.config.competence_increment,
                        self.config.target_competence
                    )
                    self.patience_counter = 0
                    logger.info(
                        f"Increased competence to {self.current_competence:.3f} "
                        f"(performance: {performance:.3f})"
                    )
            else:
                self.patience_counter = 0
                
        elif self.config.progression_type == "uncertainty":
            # Uncertainty-based progression
            if len(self.performance_history) >= self.config.competence_window:
                recent_performance = self.performance_history[-self.config.competence_window:]
                performance_std = np.std(recent_performance)
                
                # Low uncertainty means stable learning
                if performance_std < 0.05 and performance > 0.7:
                    self.current_competence = min(
                        self.current_competence + self.config.competence_increment,
                        self.config.target_competence
                    )
                    logger.info(
                        f"Increased competence to {self.current_competence:.3f} "
                        f"(low uncertainty: {performance_std:.3f})"
                    )
        
        # Apply smoothing
        if len(self.competence_history) > 0:
            smoothed_competence = (
                (1 - self.config.competence_smoothing) * self.current_competence +
                self.config.competence_smoothing * self.competence_history[-1]
            )
            self.current_competence = smoothed_competence
        
        self.competence_history.append(self.current_competence)
    
    def compute_sampling_probabilities(
        self,
        difficulties: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute sampling probabilities based on competence.
        
        Args:
            difficulties: Sample difficulties
            
        Returns:
            Sampling probabilities
        """
        if self.config.sampling_type == "threshold":
            # Threshold-based: select all samples below competence
            probabilities = (difficulties <= self.current_competence).float()
            
        elif self.config.sampling_type == "probabilistic":
            # Probabilistic sampling based on competence
            # p(x) ∝ exp(-difficulty(x) / (competence * temperature))
            scaled_difficulties = difficulties / (
                self.current_competence * self.config.sampling_temperature
            )
            probabilities = torch.exp(-scaled_difficulties)
            probabilities = probabilities / probabilities.sum()
            
        elif self.config.sampling_type == "ranked":
            # Ranked sampling: top-k based on competence
            k = int(len(difficulties) * self.current_competence)
            k = max(1, k)  # At least one sample
            
            _, indices = torch.topk(difficulties, k, largest=False)
            probabilities = torch.zeros_like(difficulties)
            probabilities[indices] = 1.0 / k
        
        else:
            # Uniform sampling as fallback
            probabilities = torch.ones_like(difficulties) / len(difficulties)
        
        return probabilities
    
    def update_sample_difficulties(self):
        """Update difficulty estimates for all samples."""
        logger.info("Updating sample difficulties...")
        
        self.model.eval()
        all_difficulties = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.train_loader):
                # Move to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                
                # Compute model loss if needed
                if self.config.difficulty_estimator == "hybrid":
                    outputs = self.model(**batch)
                    model_loss = outputs.loss if hasattr(outputs, 'loss') else None
                else:
                    model_loss = None
                
                # Estimate difficulties
                difficulties = self.difficulty_estimator.estimate(batch, model_loss)
                
                # Store difficulties
                batch_size = batch["input_ids"].size(0)
                start_idx = batch_idx * batch_size
                for i, diff in enumerate(difficulties):
                    self.sample_difficulties[start_idx + i] = diff.item()
                
                all_difficulties.extend(difficulties.cpu().numpy())
        
        # Apply smoothing
        if self.difficulty_update_counter > 0:
            for idx in self.sample_difficulties:
                old_diff = self.sample_difficulties[idx]
                new_diff = all_difficulties[idx] if idx < len(all_difficulties) else old_diff
                self.sample_difficulties[idx] = (
                    (1 - self.config.difficulty_smoothing) * new_diff +
                    self.config.difficulty_smoothing * old_diff
                )
        
        self.difficulty_update_counter += 1
        
        logger.info(
            f"Updated difficulties for {len(self.sample_difficulties)} samples"
        )
    
    def create_competence_dataloader(self) -> DataLoader:
        """
        Create dataloader with competence-based sampling.
        
        Returns:
            DataLoader with weighted sampling
        """
        # Get difficulties for all samples
        if not self.sample_difficulties or (
            self.difficulty_update_counter % self.config.difficulty_update_freq == 0
        ):
            self.update_sample_difficulties()
        
        # Convert difficulties to tensor
        num_samples = len(self.train_loader.dataset)
        difficulties = torch.zeros(num_samples)
        for idx, diff in self.sample_difficulties.items():
            if idx < num_samples:
                difficulties[idx] = diff
        
        # Compute sampling probabilities
        probabilities = self.compute_sampling_probabilities(difficulties)
        
        # Create weighted sampler
        sampler = WeightedRandomSampler(
            weights=probabilities,
            num_samples=len(self.train_loader.dataset),
            replacement=True
        )
        
        # Create new dataloader
        dataloader = DataLoader(
            self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            sampler=sampler,
            num_workers=self.train_loader.num_workers,
            pin_memory=self.train_loader.pin_memory
        )
        
        return dataloader
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with competence-based curriculum.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics
        """
        self.model.train()
        
        # Create competence-based dataloader
        competence_loader = self.create_competence_dataloader()
        
        epoch_metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "competence": self.current_competence
        }
        
        num_batches = 0
        
        for batch in competence_loader:
            # Standard training step
            metrics = self.train_step(batch)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key in epoch_metrics:
                    epoch_metrics[key] += value
            
            num_batches += 1
        
        # Average metrics
        for key in ["loss", "accuracy"]:
            if key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        
        # Validation performance
        if self.config.use_validation_competence and self.val_loader is not None:
            val_metrics = self.validate()
            performance = val_metrics.get("accuracy", val_metrics.get("f1", 0.0))
        else:
            performance = epoch_metrics.get("accuracy", 1.0 - epoch_metrics["loss"])
        
        # Update competence
        self.update_competence(performance, epoch)
        
        # Add competence metrics
        epoch_metrics["current_competence"] = self.current_competence
        epoch_metrics["performance"] = performance
        epoch_metrics["avg_difficulty"] = np.mean(list(self.sample_difficulties.values()))
        
        return epoch_metrics
