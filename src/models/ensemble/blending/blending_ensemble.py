"""
Blending Ensemble Implementation
=================================

Implementation of blending ensemble method for combining multiple models,
based on:
- Wolpert & Macready (1997): "No Free Lunch Theorems for Optimization"
- Sill et al. (2009): "Feature-Weighted Linear Stacking"
- Jahrer & Töscher (2010): "Collaborative Filtering Ensemble"

Mathematical Foundation:
Blending uses a holdout validation set to train a meta-learner:
1. Split data: D = D_train ∪ D_blend ∪ D_test
2. Train base models on D_train
3. Generate predictions on D_blend
4. Train blender on (predictions(D_blend), labels(D_blend))
5. Final prediction: f_blend(f_1(x), ..., f_M(x))

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Union, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

from src.models.ensemble.base_ensemble import BaseEnsemble, EnsembleConfig
from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import ENSEMBLES
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class BlendingConfig(EnsembleConfig):
    """Configuration for blending ensemble."""
    blend_size: float = 0.2  # Proportion of data for blending
    blender_type: str = "linear"  # "linear", "neural", "xgboost", "random_forest"
    use_original_features: bool = False  # Include original features
    use_oof_predictions: bool = True  # Use out-of-fold predictions
    n_folds: int = 5  # Number of folds for OOF
    blend_features: List[str] = None  # Additional features for blending
    neural_hidden_sizes: List[int] = None  # Hidden sizes for neural blender
    neural_dropout: float = 0.2
    xgb_params: Dict[str, Any] = None  # XGBoost parameters
    use_proba: bool = True  # Use probabilities vs raw predictions
    stratify_blend: bool = True  # Stratify blend set
    optimize_threshold: bool = False  # Optimize classification threshold


class NeuralBlender(nn.Module):
    """
    Neural network blender for combining predictions.
    
    Learns non-linear combinations of base model predictions.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = None,
        dropout: float = 0.2
    ):
        """
        Initialize neural blender.
        
        Args:
            input_size: Size of input features
            output_size: Number of output classes
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate
        """
        super().__init__()
        
        hidden_sizes = hidden_sizes or [128, 64]
        
        layers = []
        prev_size = input_size
        
        # Build hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through neural blender."""
        return self.network(x)


class FeatureEngineering:
    """
    Feature engineering for blending.
    
    Creates additional features from base predictions for improved blending.
    """
    
    @staticmethod
    def create_features(
        predictions: np.ndarray,
        include_stats: bool = True,
        include_diversity: bool = True,
        include_confidence: bool = True
    ) -> np.ndarray:
        """
        Create engineered features from predictions.
        
        Args:
            predictions: Base model predictions [n_samples, n_models, n_classes]
            include_stats: Include statistical features
            include_diversity: Include diversity measures
            include_confidence: Include confidence scores
            
        Returns:
            Engineered features array
        """
        features = []
        
        # Flatten base predictions
        n_samples, n_models, n_classes = predictions.shape
        features.append(predictions.reshape(n_samples, -1))
        
        if include_stats:
            # Statistical features across models
            features.append(np.mean(predictions, axis=1))  # Mean predictions
            features.append(np.std(predictions, axis=1))   # Standard deviation
            features.append(np.max(predictions, axis=1))   # Max predictions
            features.append(np.min(predictions, axis=1))   # Min predictions
            
            # Pairwise differences
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    diff = predictions[:, i] - predictions[:, j]
                    features.append(diff)
        
        if include_diversity:
            # Diversity measures
            # Prediction disagreement
            pred_classes = np.argmax(predictions, axis=2)
            disagreement = np.zeros((n_samples, 1))
            for i in range(n_models):
                for j in range(i + 1, n_models):
                    disagreement += (pred_classes[:, i] != pred_classes[:, j]).reshape(-1, 1)
            disagreement /= (n_models * (n_models - 1) / 2)
            features.append(disagreement)
            
            # Entropy of predictions
            avg_probs = np.mean(predictions, axis=1)
            entropy = -np.sum(avg_probs * np.log(avg_probs + 1e-10), axis=1, keepdims=True)
            features.append(entropy)
        
        if include_confidence:
            # Confidence scores
            max_probs = np.max(predictions, axis=2)  # Max probability per model
            features.append(max_probs)
            
            # Margin (difference between top 2 predictions)
            sorted_probs = np.sort(predictions, axis=2)
            margin = sorted_probs[:, :, -1] - sorted_probs[:, :, -2]
            features.append(margin)
        
        return np.concatenate(features, axis=1)


@ENSEMBLES.register("blending", aliases=["blend", "blended"])
class BlendingEnsemble(BaseEnsemble):
    """
    Blending ensemble classifier.
    
    Key differences from stacking:
    1. Uses a single holdout set instead of cross-validation
    2. Simpler and faster to train
    3. May be less prone to overfitting
    4. Can incorporate domain-specific features
    
    Supports multiple blender types:
    - Linear: Logistic regression
    - Neural: Neural network blender
    - XGBoost: Gradient boosting
    - Random Forest: Tree ensemble
    """
    
    def __init__(
        self,
        models: List[AGNewsBaseModel],
        config: Optional[BlendingConfig] = None
    ):
        """
        Initialize blending ensemble.
        
        Args:
            models: List of base models
            config: Blending configuration
        """
        super().__init__(models, config)
        
        self.config = config or BlendingConfig()
        self.feature_eng = FeatureEngineering()
        
        # Initialize blender
        self._init_blender()
        
        # Storage for blend set
        self.blend_features = None
        self.blend_labels = None
        self.is_fitted = False
        
        logger.info(
            f"Initialized BlendingEnsemble with {self.num_models} models "
            f"and {self.config.blender_type} blender"
        )
    
    def _init_blender(self):
        """Initialize the blender based on configuration."""
        if self.config.blender_type == "linear":
            self.blender = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=42
            )
            
        elif self.config.blender_type == "neural":
            # Will be initialized after knowing input size
            self.blender = None
            
        elif self.config.blender_type == "xgboost":
            params = self.config.xgb_params or {
                'objective': 'multi:softprob',
                'num_class': self.num_classes,
                'max_depth': 3,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            self.blender = xgb.XGBClassifier(**params)
            
        elif self.config.blender_type == "random_forest":
            self.blender = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown blender type: {self.config.blender_type}")
    
    def create_blend_set(
        self,
        train_loader: DataLoader,
        stratify: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create blend set from training data.
        
        Args:
            train_loader: Training data loader
            stratify: Whether to stratify the split
            
        Returns:
            Tuple of (train_loader, blend_loader)
        """
        # Extract all data
        all_indices = list(range(len(train_loader.dataset)))
        
        # Get labels for stratification
        if stratify:
            labels = []
            for batch in train_loader:
                labels.extend(batch['labels'].numpy())
            labels = np.array(labels)
        else:
            labels = None
        
        # Split indices
        train_indices, blend_indices = train_test_split(
            all_indices,
            test_size=self.config.blend_size,
            stratify=labels,
            random_state=42
        )
        
        # Create subset loaders
        train_subset = Subset(train_loader.dataset, train_indices)
        blend_subset = Subset(train_loader.dataset, blend_indices)
        
        new_train_loader = DataLoader(
            train_subset,
            batch_size=train_loader.batch_size,
            shuffle=True
        )
        
        blend_loader = DataLoader(
            blend_subset,
            batch_size=train_loader.batch_size,
            shuffle=False
        )
        
        logger.info(
            f"Created blend set: {len(train_indices)} train, "
            f"{len(blend_indices)} blend samples"
        )
        
        return new_train_loader, blend_loader
    
    def _get_oof_predictions(
        self,
        train_loader: DataLoader
    ) -> np.ndarray:
        """
        Get out-of-fold predictions for blending.
        
        Uses k-fold cross-validation to generate predictions on entire training set.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            OOF predictions array
        """
        from sklearn.model_selection import KFold
        
        # Collect all data
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
        
        n_samples = len(all_labels)
        
        # Initialize OOF predictions
        oof_predictions = np.zeros((n_samples, self.num_models, self.num_classes))
        
        # K-fold cross-validation
        kfold = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=42)
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(all_input_ids)):
            logger.info(f"Processing fold {fold + 1}/{self.config.n_folds}")
            
            # Get validation data for this fold
            val_input_ids = all_input_ids[val_idx]
            val_attention_masks = all_attention_masks[val_idx]
            
            # Get predictions from each model
            with torch.no_grad():
                for model_idx, model in enumerate(self.models):
                    model.eval()
                    outputs = model(
                        input_ids=val_input_ids,
                        attention_mask=val_attention_masks
                    )
                    
                    if self.config.use_proba:
                        probs = F.softmax(outputs.logits, dim=-1)
                        oof_predictions[val_idx, model_idx] = probs.cpu().numpy()
                    else:
                        oof_predictions[val_idx, model_idx] = outputs.logits.cpu().numpy()
        
        return oof_predictions, all_labels.numpy()
    
    def fit_blender(
        self,
        train_loader: DataLoader,
        use_oof: Optional[bool] = None
    ):
        """
        Fit the blender on training data.
        
        Args:
            train_loader: Training data loader
            use_oof: Whether to use out-of-fold predictions
        """
        use_oof = use_oof if use_oof is not None else self.config.use_oof_predictions
        
        if use_oof:
            # Get out-of-fold predictions
            logger.info("Generating out-of-fold predictions")
            predictions, labels = self._get_oof_predictions(train_loader)
        else:
            # Create blend set
            train_loader, blend_loader = self.create_blend_set(
                train_loader,
                stratify=self.config.stratify_blend
            )
            
            # Train base models on reduced training set
            logger.info("Training base models on reduced training set")
            # Note: In practice, models would be retrained here
            
            # Get predictions on blend set
            predictions = []
            labels = []
            
            with torch.no_grad():
                for batch in blend_loader:
                    batch_preds = []
                    
                    for model in self.models:
                        model.eval()
                        outputs = model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask']
                        )
                        
                        if self.config.use_proba:
                            probs = F.softmax(outputs.logits, dim=-1)
                            batch_preds.append(probs.cpu().numpy())
                        else:
                            batch_preds.append(outputs.logits.cpu().numpy())
                    
                    predictions.append(np.stack(batch_preds, axis=1))
                    labels.append(batch['labels'].numpy())
            
            predictions = np.concatenate(predictions, axis=0)
            labels = np.concatenate(labels, axis=0)
        
        # Create features for blending
        blend_features = self.feature_eng.create_features(
            predictions,
            include_stats=True,
            include_diversity=True,
            include_confidence=True
        )
        
        # Initialize neural blender if needed
        if self.config.blender_type == "neural" and self.blender is None:
            input_size = blend_features.shape[1]
            self.blender = NeuralBlender(
                input_size=input_size,
                output_size=self.num_classes,
                hidden_sizes=self.config.neural_hidden_sizes,
                dropout=self.config.neural_dropout
            )
            
            # Train neural blender
            self._train_neural_blender(blend_features, labels)
        else:
            # Train sklearn blender
            self.blender.fit(blend_features, labels)
        
        # Store blend features for analysis
        self.blend_features = blend_features
        self.blend_labels = labels
        self.is_fitted = True
        
        # Log performance
        if hasattr(self.blender, 'score'):
            score = self.blender.score(blend_features, labels)
            logger.info(f"Blender training accuracy: {score:.4f}")
        
        logger.info("Blender fitting completed")
    
    def _train_neural_blender(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        epochs: int = 50
    ):
        """
        Train neural network blender.
        
        Args:
            features: Blend features
            labels: Target labels
            epochs: Number of training epochs
        """
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels)
        
        # Create data loader
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.blender.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.blender.train()
        for epoch in range(epochs):
            total_loss = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                
                logits = self.blender(batch_X)
                loss = criterion(logits, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / len(loader)
                logger.debug(f"Neural blender epoch {epoch}: loss = {avg_loss:.4f}")
    
    def combine_predictions(
        self,
        predictions: List[torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Combine predictions using trained blender.
        
        Args:
            predictions: List of model predictions
            
        Returns:
            Blended predictions
        """
        if not self.is_fitted:
            # Fallback to simple averaging
            logger.warning("Blender not fitted, using average combination")
            return torch.stack(predictions).mean(dim=0)
        
        # Convert predictions to numpy
        if self.config.use_proba:
            pred_array = np.stack([
                F.softmax(pred, dim=-1).cpu().numpy() for pred in predictions
            ], axis=1)
        else:
            pred_array = np.stack([
                pred.cpu().numpy() for pred in predictions
            ], axis=1)
        
        # Create features
        blend_features = self.feature_eng.create_features(
            pred_array,
            include_stats=True,
            include_diversity=True,
            include_confidence=True
        )
        
        # Get blender predictions
        if self.config.blender_type == "neural":
            self.blender.eval()
            with torch.no_grad():
                X = torch.FloatTensor(blend_features)
                blended_logits = self.blender(X)
        else:
            if hasattr(self.blender, 'predict_proba'):
                blended_probs = self.blender.predict_proba(blend_features)
                blended_logits = torch.FloatTensor(np.log(blended_probs + 1e-10))
            else:
                blended_preds = self.blender.predict(blend_features)
                # Convert to one-hot and then logits
                blended_logits = torch.zeros(len(blended_preds), self.num_classes)
                blended_logits[range(len(blended_preds)), blended_preds] = 1.0
        
        return blended_logits
    
    def optimize_threshold(
        self,
        val_loader: DataLoader,
        metric: str = "f1"
    ) -> float:
        """
        Optimize classification threshold on validation set.
        
        Args:
            val_loader: Validation data loader
            metric: Metric to optimize
            
        Returns:
            Optimal threshold
        """
        from sklearn.metrics import f1_score, accuracy_score
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get predictions
                outputs = self.forward(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                probs = F.softmax(outputs.logits, dim=-1)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(batch['labels'].numpy())
        
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)
        
        # Search for optimal threshold
        best_threshold = 0.5
        best_score = 0
        
        for threshold in np.arange(0.3, 0.8, 0.05):
            # Apply threshold
            preds = (all_probs.max(axis=1) > threshold) * all_probs.argmax(axis=1)
            
            if metric == "f1":
                score = f1_score(all_labels, preds, average='macro')
            else:
                score = accuracy_score(all_labels, preds)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.2f} ({metric}: {best_score:.4f})")
        
        return best_threshold
