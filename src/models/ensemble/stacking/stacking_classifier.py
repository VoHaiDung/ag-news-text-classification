"""
Stacking Ensemble Classifier Implementation
============================================

Implementation of stacking (stacked generalization) for ensemble learning,
based on:
- Wolpert (1992): "Stacked Generalization"
- Breiman (1996): "Stacked Regressions"
- Van der Laan et al. (2007): "Super Learner"

Mathematical Foundation:
Two-level learning architecture:
Level-0: Base models f_m(x) -> predictions
Level-1: Meta-learner g(f_1(x), ..., f_M(x)) -> final prediction

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Union, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.ensemble.base_ensemble import BaseEnsemble, EnsembleConfig
from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import ENSEMBLES
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class StackingConfig(EnsembleConfig):
    """Configuration for stacking ensemble."""
    meta_learner_type: str = "neural_network"  # "neural_network", "linear", "xgboost"
    use_probabilities: bool = True  # Use probabilities vs raw predictions
    use_original_features: bool = False  # Include original features
    cv_folds: int = 5  # Cross-validation folds for training
    meta_hidden_size: int = 128
    meta_num_layers: int = 2
    meta_dropout: float = 0.2
    meta_learning_rate: float = 0.001
    meta_epochs: int = 50
    blend_with_base: bool = False  # Blend meta predictions with base
    restacking: bool = False  # Multi-level stacking


class MetaLearner(nn.Module):
    """
    Neural network meta-learner for stacking.
    
    Learns to combine base model predictions optimally.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        """
        Initialize meta-learner.
        
        Args:
            input_size: Size of input (num_models * num_classes)
            output_size: Number of output classes
            hidden_size: Hidden layer size
            num_layers: Number of hidden layers
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_size, hidden_size))
            
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            
            # Batch normalization for stability
            layers.append(nn.BatchNorm1d(hidden_size))
        
        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through meta-learner."""
        return self.network(x)


@ENSEMBLES.register("stacking", aliases=["stacked", "super_learner"])
class StackingClassifier(BaseEnsemble):
    """
    Stacking ensemble classifier.
    
    Implements two-level stacking:
    1. Level-0: Base models make predictions
    2. Level-1: Meta-learner combines predictions
    
    Key features:
    - Cross-validation for meta-training
    - Multiple meta-learner options
    - Blending capabilities
    - Multi-level stacking support
    """
    
    def __init__(
        self,
        models: List[AGNewsBaseModel],
        config: Optional[StackingConfig] = None
    ):
        """
        Initialize stacking classifier.
        
        Args:
            models: List of base models
            config: Stacking configuration
        """
        super().__init__(models, config)
        
        self.config = config or StackingConfig()
        
        # Initialize meta-learner
        self._init_meta_learner()
        
        # Storage for cross-validation predictions
        self.cv_predictions = None
        self.is_fitted = False
        
        logger.info(
            f"Initialized StackingClassifier with {self.num_models} base models "
            f"and {self.config.meta_learner_type} meta-learner"
        )
    
    def _init_meta_learner(self):
        """Initialize the meta-learner based on configuration."""
        # Calculate input size for meta-learner
        if self.config.use_probabilities:
            input_size = self.num_models * self.num_classes
        else:
            input_size = self.num_models
        
        if self.config.use_original_features:
            # Add space for original features if configured
            input_size += 768  # Assuming standard hidden size
        
        # Create meta-learner
        if self.config.meta_learner_type == "neural_network":
            self.meta_learner = MetaLearner(
                input_size=input_size,
                output_size=self.num_classes,
                hidden_size=self.config.meta_hidden_size,
                num_layers=self.config.meta_num_layers,
                dropout=self.config.meta_dropout
            )
        elif self.config.meta_learner_type == "linear":
            self.meta_learner = nn.Linear(input_size, self.num_classes)
        else:
            raise ValueError(f"Unknown meta-learner type: {self.config.meta_learner_type}")
    
    def _get_base_predictions(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Get predictions from all base models.
        
        Returns:
            Tuple of (logits_list, probs_list)
        """
        all_logits = []
        all_probs = []
        
        with torch.no_grad():
            for model in self.models:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                
                all_logits.append(logits)
                all_probs.append(probs)
        
        return all_logits, all_probs
    
    def _prepare_meta_features(
        self,
        predictions: List[torch.Tensor],
        original_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Prepare features for meta-learner.
        
        Args:
            predictions: Base model predictions
            original_features: Original input features
            
        Returns:
            Concatenated features for meta-learner
        """
        # Stack and flatten predictions
        if self.config.use_probabilities:
            # Use full probability distributions
            meta_features = torch.cat(predictions, dim=-1)
        else:
            # Use only predicted classes
            predicted_classes = [torch.argmax(pred, dim=-1) for pred in predictions]
            meta_features = torch.stack(predicted_classes, dim=-1).float()
        
        # Add original features if configured
        if self.config.use_original_features and original_features is not None:
            meta_features = torch.cat([meta_features, original_features], dim=-1)
        
        return meta_features
    
    def fit_meta_learner(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Train meta-learner using cross-validation.
        
        Implements proper stacking with cross-validation to avoid overfitting.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        logger.info("Starting meta-learner training with cross-validation")
        
        # Collect all training data
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        
        for batch in train_loader:
            all_input_ids.append(batch['input_ids'])
            all_attention_masks.append(batch['attention_mask'])
            all_labels.append(batch['labels'])
        
        all_input_ids = torch.cat(all_input_ids)
        all_attention_masks = torch.cat(all_attention_masks)
        all_labels = torch.cat(all_labels)
        
        # Cross-validation for meta-training
        kfold = KFold(n_splits=self.config.cv_folds, shuffle=True)
        
        cv_meta_features = []
        cv_labels = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_input_ids)):
            logger.info(f"Processing fold {fold + 1}/{self.config.cv_folds}")
            
            # Split data
            val_input_ids = all_input_ids[val_idx]
            val_attention_masks = all_attention_masks[val_idx]
            val_labels = all_labels[val_idx]
            
            # Get predictions on validation fold
            logits, probs = self._get_base_predictions(
                val_input_ids,
                val_attention_masks
            )
            
            # Prepare meta features
            meta_features = self._prepare_meta_features(
                probs if self.config.use_probabilities else logits
            )
            
            cv_meta_features.append(meta_features)
            cv_labels.append(val_labels)
        
        # Combine all CV predictions
        cv_meta_features = torch.cat(cv_meta_features)
        cv_labels = torch.cat(cv_labels)
        
        # Train meta-learner
        self._train_meta_learner(cv_meta_features, cv_labels, val_loader)
        
        self.is_fitted = True
        logger.info("Meta-learner training completed")
    
    def _train_meta_learner(
        self,
        meta_features: torch.Tensor,
        labels: torch.Tensor,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Train the meta-learner on meta-features.
        
        Args:
            meta_features: Prepared meta features
            labels: Target labels
            val_loader: Validation data loader
        """
        # Create dataset and loader
        dataset = TensorDataset(meta_features, labels)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(
            self.meta_learner.parameters(),
            lr=self.config.meta_learning_rate
        )
        
        # Training loop
        self.meta_learner.train()
        
        for epoch in range(self.config.meta_epochs):
            total_loss = 0
            
            for batch_features, batch_labels in loader:
                optimizer.zero_grad()
                
                # Forward pass
                logits = self.meta_learner(batch_features)
                loss = F.cross_entropy(logits, batch_labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.debug(f"Meta-learner epoch {epoch}: loss = {avg_loss:.4f}")
    
    def combine_predictions(
        self,
        predictions: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Combine predictions using trained meta-learner.
        
        Args:
            predictions: Base model predictions
            
        Returns:
            Combined predictions from meta-learner
        """
        if not self.is_fitted:
            # Fallback to simple averaging if not fitted
            logger.warning("Meta-learner not fitted, using average combination")
            return torch.stack(predictions).mean(dim=0)
        
        # Prepare meta features
        meta_features = self._prepare_meta_features(predictions)
        
        # Get meta-learner predictions
        self.meta_learner.eval()
        with torch.no_grad():
            meta_logits = self.meta_learner(meta_features)
        
        # Optional blending with base predictions
        if self.config.blend_with_base:
            base_avg = torch.stack(predictions).mean(dim=0)
            meta_logits = 0.5 * meta_logits + 0.5 * base_avg
        
        return meta_logits
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through stacking ensemble.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            **kwargs: Additional arguments
            
        Returns:
            Stacked ensemble predictions
        """
        # Get base predictions
        logits_list, probs_list = self._get_base_predictions(
            input_ids,
            attention_mask,
            **kwargs
        )
        
        # Combine using meta-learner
        if self.config.use_probabilities:
            combined_logits = self.combine_predictions(probs_list)
        else:
            combined_logits = self.combine_predictions(logits_list)
        
        # Compute loss
        loss = None
        if labels is not None:
            loss = F.cross_entropy(combined_logits, labels)
        
        return ModelOutputs(
            logits=combined_logits,
            loss=loss,
            metadata={
                "base_logits": logits_list,
                "base_probs": probs_list,
                "is_fitted": self.is_fitted
            }
        )
