"""
Cross-Validation Based Stacking Ensemble
=========================================

Implementation of stacking with cross-validation to prevent overfitting,
based on:
- Wolpert (1992): "Stacked Generalization"
- Breiman (1996): "Stacked Regressions"
- Ting & Witten (1999): "Issues in Stacked Generalization"

Cross-validation stacking creates out-of-fold predictions for training
the meta-learner, preventing information leakage and overfitting.

Mathematical Foundation:
For K-fold CV: Split data into K folds
For each fold k:
  - Train base models on folds ≠ k
  - Predict on fold k to create meta-features
Meta-learner trains on all out-of-fold predictions

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.base import BaseEstimator, ClassifierMixin, clone
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.models.ensemble.base_ensemble import BaseEnsemble
from src.models.ensemble.stacking.meta_learners import MetaLearnerFactory, MetaLearnerConfig
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CrossValidationConfig:
    """Configuration for cross-validation stacking"""
    
    # Cross-validation settings
    n_splits: int = 5  # Number of CV folds
    shuffle: bool = True
    random_state: int = 42
    stratified: bool = True  # Use stratified splits
    
    # Stacking configuration
    use_probabilities: bool = True  # Use probabilities vs predictions
    include_original_features: bool = False  # Include original features
    blend_test_predictions: bool = True  # Blend test predictions from CV models
    
    # Meta-learner configuration
    meta_learner_type: str = "logistic"  # Type of meta-learner
    meta_learner_config: Optional[MetaLearnerConfig] = None
    
    # Training configuration
    retrain_on_full: bool = True  # Retrain base models on full data after CV
    save_oof_predictions: bool = True  # Save out-of-fold predictions
    
    # Performance optimization
    parallel_cv: bool = False  # Parallel CV training
    cache_predictions: bool = True
    
    # Validation
    holdout_validation: float = 0.0  # Fraction for holdout validation
    early_stopping_rounds: int = 5
    
    # Advanced options
    dynamic_k: bool = False  # Dynamic number of folds based on data size
    nested_cv: bool = False  # Nested CV for hyperparameter tuning
    feature_selection: bool = False  # Select features from base models


class OutOfFoldPredictor:
    """
    Generates out-of-fold predictions for stacking.
    
    Ensures that meta-features are created using models that
    haven't seen the respective training examples.
    """
    
    def __init__(
        self,
        base_models: List[AGNewsBaseModel],
        config: CrossValidationConfig
    ):
        """
        Initialize out-of-fold predictor.
        
        Args:
            base_models: List of base models
            config: Cross-validation configuration
        """
        self.base_models = base_models
        self.config = config
        self.oof_predictions = {}
        self.test_predictions = {}
        
        # Setup cross-validation splitter
        if config.stratified:
            self.cv_splitter = StratifiedKFold(
                n_splits=config.n_splits,
                shuffle=config.shuffle,
                random_state=config.random_state
            )
        else:
            self.cv_splitter = KFold(
                n_splits=config.n_splits,
                shuffle=config.shuffle,
                random_state=config.random_state
            )
        
        logger.info(
            f"Initialized OOF predictor with {len(base_models)} models, "
            f"{config.n_splits} folds"
        )
    
    def generate_oof_predictions(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: Optional[torch.Tensor] = None,
        attention_mask_train: Optional[torch.Tensor] = None,
        attention_mask_test: Optional[torch.Tensor] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate out-of-fold predictions.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            attention_mask_train: Training attention masks
            attention_mask_test: Test attention masks
            
        Returns:
            Tuple of (OOF predictions, test predictions)
        """
        n_samples = X_train.shape[0]
        n_models = len(self.base_models)
        n_classes = 4  # AG News classes
        
        # Initialize arrays for predictions
        if self.config.use_probabilities:
            oof_preds = np.zeros((n_samples, n_models * n_classes))
            if X_test is not None:
                test_preds = np.zeros((X_test.shape[0], n_models * n_classes, self.config.n_splits))
        else:
            oof_preds = np.zeros((n_samples, n_models))
            if X_test is not None:
                test_preds = np.zeros((X_test.shape[0], n_models, self.config.n_splits))
        
        # Generate OOF predictions for each model
        for model_idx, model in enumerate(self.base_models):
            logger.info(f"Generating OOF predictions for model {model_idx + 1}/{n_models}")
            
            # Arrays for this model's predictions
            if self.config.use_probabilities:
                model_oof = np.zeros((n_samples, n_classes))
            else:
                model_oof = np.zeros(n_samples)
            
            # Cross-validation loop
            for fold_idx, (train_idx, val_idx) in enumerate(
                self.cv_splitter.split(X_train.cpu().numpy(), y_train.cpu().numpy())
            ):
                logger.debug(f"Processing fold {fold_idx + 1}/{self.config.n_splits}")
                
                # Split data
                X_fold_train = X_train[train_idx]
                y_fold_train = y_train[train_idx]
                X_fold_val = X_train[val_idx]
                
                mask_fold_train = attention_mask_train[train_idx] if attention_mask_train is not None else None
                mask_fold_val = attention_mask_train[val_idx] if attention_mask_train is not None else None
                
                # Clone model for this fold
                fold_model = self._clone_model(model)
                
                # Train model on fold
                self._train_fold_model(
                    fold_model,
                    X_fold_train,
                    y_fold_train,
                    mask_fold_train
                )
                
                # Generate OOF predictions for validation fold
                with torch.no_grad():
                    outputs = fold_model(
                        X_fold_val,
                        attention_mask=mask_fold_val
                    )
                    
                    if self.config.use_probabilities:
                        fold_preds = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                        model_oof[val_idx] = fold_preds
                    else:
                        fold_preds = outputs.logits.argmax(dim=-1).cpu().numpy()
                        model_oof[val_idx] = fold_preds
                
                # Generate test predictions if test data provided
                if X_test is not None:
                    with torch.no_grad():
                        test_outputs = fold_model(
                            X_test,
                            attention_mask=attention_mask_test
                        )
                        
                        if self.config.use_probabilities:
                            test_fold_preds = F.softmax(test_outputs.logits, dim=-1).cpu().numpy()
                            test_preds[:, model_idx*n_classes:(model_idx+1)*n_classes, fold_idx] = test_fold_preds
                        else:
                            test_fold_preds = test_outputs.logits.argmax(dim=-1).cpu().numpy()
                            test_preds[:, model_idx, fold_idx] = test_fold_preds
            
            # Store model's OOF predictions
            if self.config.use_probabilities:
                oof_preds[:, model_idx*n_classes:(model_idx+1)*n_classes] = model_oof
            else:
                oof_preds[:, model_idx] = model_oof
        
        # Blend test predictions across folds
        if X_test is not None and self.config.blend_test_predictions:
            test_preds_final = test_preds.mean(axis=2)
        else:
            test_preds_final = test_preds if X_test is not None else None
        
        # Save predictions if configured
        if self.config.save_oof_predictions:
            self.oof_predictions = oof_preds
            if test_preds_final is not None:
                self.test_predictions = test_preds_final
        
        return oof_preds, test_preds_final
    
    def _clone_model(self, model: AGNewsBaseModel) -> AGNewsBaseModel:
        """
        Clone a model for fold training.
        
        Args:
            model: Model to clone
            
        Returns:
            Cloned model
        """
        # Deep copy model
        import copy
        cloned = copy.deepcopy(model)
        
        # Reset parameters to initial state if possible
        if hasattr(cloned, 'reset_parameters'):
            cloned.reset_parameters()
        
        return cloned
    
    def _train_fold_model(
        self,
        model: AGNewsBaseModel,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        epochs: int = 5
    ):
        """
        Train model on fold data.
        
        Args:
            model: Model to train
            X_train: Training features
            y_train: Training labels
            attention_mask: Attention masks
            epochs: Number of training epochs
        """
        # Simple training loop (would be more sophisticated in practice)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        
        # Create dataset
        if attention_mask is not None:
            dataset = TensorDataset(X_train, attention_mask, y_train)
        else:
            dataset = TensorDataset(X_train, y_train)
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in dataloader:
                if attention_mask is not None:
                    inputs, masks, labels = batch
                    outputs = model(inputs, attention_mask=masks, labels=labels)
                else:
                    inputs, labels = batch
                    outputs = model(inputs, labels=labels)
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            logger.debug(f"Fold training - Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


@MODELS.register("cv_stacking")
class CrossValidationStacking(BaseEnsemble):
    """
    Stacking ensemble with cross-validation.
    
    Implements proper stacking methodology using out-of-fold predictions
    to train the meta-learner, preventing overfitting and ensuring
    generalization.
    
    The process:
    1. Generate OOF predictions using K-fold CV
    2. Train meta-learner on OOF predictions
    3. Optionally retrain base models on full data
    4. Make final predictions using meta-learner
    """
    
    def __init__(
        self,
        base_models: List[AGNewsBaseModel],
        config: Optional[CrossValidationConfig] = None
    ):
        """
        Initialize cross-validation stacking.
        
        Args:
            base_models: List of base models
            config: Configuration
        """
        super().__init__(base_models)
        
        self.config = config or CrossValidationConfig()
        
        # Initialize OOF predictor
        self.oof_predictor = OutOfFoldPredictor(base_models, self.config)
        
        # Initialize meta-learner
        meta_config = self.config.meta_learner_config or MetaLearnerConfig(
            learner_type=self.config.meta_learner_type
        )
        self.meta_learner = MetaLearnerFactory.create(meta_config)
        
        # Storage for predictions
        self.oof_predictions = None
        self.is_trained = False
        
        # Statistics
        self.cv_scores = []
        self.feature_importance = None
        
        logger.info(
            f"Initialized CV Stacking with {len(base_models)} base models, "
            f"{config.n_splits} folds, {config.meta_learner_type} meta-learner"
        )
    
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: Optional[torch.Tensor] = None,
        y_val: Optional[torch.Tensor] = None,
        attention_mask_train: Optional[torch.Tensor] = None,
        attention_mask_val: Optional[torch.Tensor] = None
    ):
        """
        Train stacking ensemble with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            attention_mask_train: Training attention masks
            attention_mask_val: Validation attention masks
        """
        logger.info("Training CV stacking ensemble...")
        
        # Generate out-of-fold predictions
        oof_preds, val_preds = self.oof_predictor.generate_oof_predictions(
            X_train,
            y_train,
            X_val,
            attention_mask_train,
            attention_mask_val
        )
        
        # Save OOF predictions
        self.oof_predictions = oof_preds
        
        # Optionally include original features
        if self.config.include_original_features:
            # Flatten and concatenate features
            X_meta = self._prepare_meta_features(X_train, oof_preds)
        else:
            X_meta = oof_preds
        
        # Train meta-learner
        logger.info("Training meta-learner on OOF predictions...")
        self.meta_learner.fit(X_meta, y_train.cpu().numpy())
        
        # Get feature importance if available
        self.feature_importance = self.meta_learner.get_feature_importance()
        
        # Retrain base models on full data if configured
        if self.config.retrain_on_full:
            logger.info("Retraining base models on full dataset...")
            self._retrain_base_models(X_train, y_train, attention_mask_train)
        
        # Validate if validation data provided
        if X_val is not None and y_val is not None:
            val_score = self._validate(X_val, y_val, attention_mask_val)
            logger.info(f"Validation score: {val_score:.4f}")
        
        self.is_trained = True
        logger.info("CV stacking training completed")
    
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
            Ensemble predictions
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before inference")
        
        batch_size = input_ids.shape[0]
        n_models = len(self.models)
        n_classes = 4
        
        # Get base model predictions
        base_predictions = []
        
        for model in self.models:
            with torch.no_grad():
                outputs = model(
                    input_ids,
                    attention_mask=attention_mask,
                    **kwargs
                )
                
                if self.config.use_probabilities:
                    probs = F.softmax(outputs.logits, dim=-1)
                    base_predictions.append(probs)
                else:
                    preds = outputs.logits.argmax(dim=-1)
                    base_predictions.append(preds)
        
        # Prepare meta-features
        if self.config.use_probabilities:
            # Stack probability predictions
            meta_features = torch.cat(base_predictions, dim=1).cpu().numpy()
        else:
            # Stack class predictions
            meta_features = torch.stack(base_predictions, dim=1).cpu().numpy()
        
        # Optionally include original features
        if self.config.include_original_features:
            meta_features = self._prepare_meta_features(input_ids, meta_features)
        
        # Get meta-learner predictions
        meta_predictions = self.meta_learner.predict_proba(meta_features)
        meta_logits = torch.tensor(
            np.log(meta_predictions + 1e-8),
            device=input_ids.device
        )
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(meta_logits, labels)
        
        return ModelOutputs(
            loss=loss,
            logits=meta_logits,
            metadata={
                'stacking_type': 'cross_validation',
                'n_folds': self.config.n_splits,
                'meta_learner': self.config.meta_learner_type,
                'n_base_models': n_models
            }
        )
    
    def _prepare_meta_features(
        self,
        original_features: torch.Tensor,
        predictions: np.ndarray
    ) -> np.ndarray:
        """
        Prepare meta-features by combining predictions with original features.
        
        Args:
            original_features: Original input features
            predictions: Base model predictions
            
        Returns:
            Combined meta-features
        """
        # Simple approach: use mean pooling of original features
        if len(original_features.shape) == 2:
            # Already 2D
            original_flat = original_features.cpu().numpy()
        else:
            # Pool over sequence dimension
            original_flat = original_features.mean(dim=1).cpu().numpy()
        
        # Concatenate
        meta_features = np.concatenate([predictions, original_flat], axis=1)
        
        return meta_features
    
    def _retrain_base_models(
        self,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Retrain base models on full training data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            attention_mask: Attention masks
        """
        for i, model in enumerate(self.models):
            logger.info(f"Retraining model {i+1}/{len(self.models)} on full data...")
            
            # Simple retraining (would be more sophisticated in practice)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
            
            dataset = TensorDataset(X_train, y_train)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            for epoch in range(5):
                for batch in dataloader:
                    inputs, labels = batch
                    outputs = model(inputs, labels=labels)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
    
    def _validate(
        self,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> float:
        """
        Validate ensemble on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            attention_mask: Attention masks
            
        Returns:
            Validation accuracy
        """
        with torch.no_grad():
            outputs = self.forward(X_val, attention_mask=attention_mask)
            predictions = outputs.logits.argmax(dim=-1)
            accuracy = (predictions == y_val).float().mean().item()
        
        return accuracy
    
    def get_cv_results(self) -> Dict[str, Any]:
        """
        Get cross-validation results and statistics.
        
        Returns:
            Dictionary of CV results
        """
        results = {
            'n_folds': self.config.n_splits,
            'meta_learner': self.config.meta_learner_type,
            'is_trained': self.is_trained
        }
        
        if self.oof_predictions is not None:
            results['oof_shape'] = self.oof_predictions.shape
        
        if self.feature_importance is not None:
            results['feature_importance'] = self.feature_importance
        
        if self.cv_scores:
            results['cv_scores'] = self.cv_scores
            results['mean_cv_score'] = np.mean(self.cv_scores)
            results['std_cv_score'] = np.std(self.cv_scores)
        
        return results
