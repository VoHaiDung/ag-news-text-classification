"""
Meta-Learners for Stacking Ensemble
====================================

Implementation of meta-learning algorithms for stacking ensemble models,
following the theoretical framework from:
- Wolpert (1992): "Stacked Generalization"
- Breiman (1996): "Stacked Regressions"
- Van der Laan et al. (2007): "Super Learner"

The meta-learner combines predictions from base models to produce final predictions,
learning optimal combination weights through various strategies.

Mathematical Foundation:
Given base model predictions h_1(x), ..., h_L(x), the meta-learner f learns:
y = f(h_1(x), ..., h_L(x)) where f optimizes the final prediction accuracy.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetaLearnerConfig:
    """Configuration for meta-learners."""
    
    learner_type: str = "logistic"  # Type of meta-learner
    use_probabilities: bool = True  # Use probabilities vs class predictions
    use_original_features: bool = False  # Include original features
    cv_folds: int = 5  # Cross-validation folds
    optimize_threshold: bool = True  # Optimize decision threshold
    ensemble_size: int = None  # Number of base models to select
    regularization: float = 1.0  # Regularization strength
    random_state: int = 42
    
    # Advanced options
    feature_selection: bool = False  # Select important features
    calibrate_probabilities: bool = True  # Calibrate base model probabilities
    weighted_average: bool = False  # Use weighted averaging
    
    # Neural network specific
    hidden_layers: List[int] = None  # Hidden layer sizes for NN meta-learner
    dropout_rate: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32


class BaseMetaLearner(ABC):
    """
    Abstract base class for meta-learners.
    
    Defines the interface that all meta-learners must implement,
    ensuring consistency across different meta-learning strategies.
    """
    
    def __init__(self, config: MetaLearnerConfig):
        """
        Initialize meta-learner.
        
        Args:
            config: Meta-learner configuration
        """
        self.config = config
        self.is_fitted = False
        self.feature_importances_ = None
        self.model = None
        
    @abstractmethod
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Fit meta-learner on base model predictions.
        
        Args:
            X: Base model predictions [n_samples, n_base_models * n_classes]
            y: True labels [n_samples]
            sample_weight: Sample weights
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using meta-learner.
        
        Args:
            X: Base model predictions
            
        Returns:
            Final predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Base model predictions
            
        Returns:
            Class probabilities
        """
        pass
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores if available."""
        return self.feature_importances_


class LogisticMetaLearner(BaseMetaLearner):
    """
    Logistic Regression meta-learner.
    
    Uses regularized logistic regression to combine base model predictions.
    Particularly effective when base models are well-calibrated.
    """
    
    def __init__(self, config: MetaLearnerConfig):
        """Initialize logistic meta-learner."""
        super().__init__(config)
        
        self.model = LogisticRegression(
            C=1.0 / config.regularization,
            random_state=config.random_state,
            max_iter=1000,
            solver='lbfgs',
            multi_class='multinomial'
        )
        
        logger.info("Initialized Logistic Regression meta-learner")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """Fit logistic regression on base predictions."""
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        
        # Extract feature importance (coefficients)
        if hasattr(self.model, 'coef_'):
            self.feature_importances_ = np.abs(self.model.coef_).mean(axis=0)
        
        logger.info(f"Fitted Logistic meta-learner on {X.shape[0]} samples")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict_proba(X)


class XGBoostMetaLearner(BaseMetaLearner):
    """
    XGBoost meta-learner.
    
    Uses gradient boosting to learn non-linear combinations of base predictions.
    Effective for capturing complex interactions between base models.
    """
    
    def __init__(self, config: MetaLearnerConfig):
        """Initialize XGBoost meta-learner."""
        super().__init__(config)
        
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=config.regularization,
            reg_lambda=config.regularization,
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        logger.info("Initialized XGBoost meta-learner")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """Fit XGBoost on base predictions."""
        self.model.fit(
            X, y,
            sample_weight=sample_weight,
            eval_set=[(X, y)],
            early_stopping_rounds=10,
            verbose=False
        )
        self.is_fitted = True
        
        # Extract feature importance
        self.feature_importances_ = self.model.feature_importances_
        
        logger.info(f"Fitted XGBoost meta-learner with {self.model.n_estimators} trees")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict_proba(X)


class NeuralMetaLearner(BaseMetaLearner):
    """
    Neural Network meta-learner.
    
    Uses a feedforward neural network to learn complex non-linear
    combinations of base model predictions.
    """
    
    def __init__(self, config: MetaLearnerConfig):
        """Initialize neural network meta-learner."""
        super().__init__(config)
        
        hidden_layers = config.hidden_layers or [50, 30]
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            alpha=config.regularization,
            learning_rate_init=config.learning_rate,
            max_iter=config.epochs,
            random_state=config.random_state,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        logger.info(f"Initialized Neural meta-learner with layers {hidden_layers}")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """Fit neural network on base predictions."""
        self.model.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True
        
        logger.info(f"Fitted Neural meta-learner for {self.model.n_iter_} iterations")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict_proba(X)


class LightGBMMetaLearner(BaseMetaLearner):
    """
    LightGBM meta-learner.
    
    Uses gradient boosting with leaf-wise tree growth for efficient
    and accurate combination of base predictions.
    """
    
    def __init__(self, config: MetaLearnerConfig):
        """Initialize LightGBM meta-learner."""
        super().__init__(config)
        
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            num_leaves=31,
            learning_rate=0.1,
            feature_fraction=0.8,
            bagging_fraction=0.8,
            bagging_freq=5,
            reg_alpha=config.regularization,
            reg_lambda=config.regularization,
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=-1
        )
        
        logger.info("Initialized LightGBM meta-learner")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """Fit LightGBM on base predictions."""
        self.model.fit(
            X, y,
            sample_weight=sample_weight,
            eval_set=[(X, y)],
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        self.is_fitted = True
        
        # Extract feature importance
        self.feature_importances_ = self.model.feature_importances_
        
        logger.info(f"Fitted LightGBM meta-learner with {self.model.n_estimators} trees")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict_proba(X)


class CatBoostMetaLearner(BaseMetaLearner):
    """
    CatBoost meta-learner.
    
    Uses gradient boosting with ordered boosting for reduced overfitting
    and better generalization.
    """
    
    def __init__(self, config: MetaLearnerConfig):
        """Initialize CatBoost meta-learner."""
        super().__init__(config)
        
        self.model = cb.CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            l2_leaf_reg=config.regularization,
            random_state=config.random_state,
            verbose=False,
            thread_count=-1
        )
        
        logger.info("Initialized CatBoost meta-learner")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """Fit CatBoost on base predictions."""
        self.model.fit(
            X, y,
            sample_weight=sample_weight,
            eval_set=(X, y),
            early_stopping_rounds=10,
            verbose=False
        )
        self.is_fitted = True
        
        # Extract feature importance
        self.feature_importances_ = self.model.feature_importances_
        
        logger.info(f"Fitted CatBoost meta-learner")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict_proba(X)


class BayesianMetaLearner(BaseMetaLearner):
    """
    Bayesian meta-learner using Gaussian Processes or Bayesian Neural Networks.
    
    Provides uncertainty estimates along with predictions, useful for
    active learning and uncertainty-aware decision making.
    """
    
    def __init__(self, config: MetaLearnerConfig):
        """Initialize Bayesian meta-learner."""
        super().__init__(config)
        
        # Use Gaussian Process Classifier for small datasets
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF, Matern
        
        kernel = 1.0 * RBF(length_scale=1.0)
        self.model = GaussianProcessClassifier(
            kernel=kernel,
            n_restarts_optimizer=5,
            random_state=config.random_state
        )
        
        logger.info("Initialized Bayesian (GP) meta-learner")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """Fit Gaussian Process on base predictions."""
        # GP doesn't support sample weights directly
        self.model.fit(X, y)
        self.is_fitted = True
        
        logger.info(f"Fitted Bayesian meta-learner on {X.shape[0]} samples")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with uncertainty."""
        if not self.is_fitted:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.model.predict_proba(X)
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Args:
            X: Base model predictions
            
        Returns:
            Tuple of (predictions, uncertainties)
        """
        # Get probability predictions
        proba = self.predict_proba(X)
        
        # Calculate entropy as uncertainty measure
        # H = -sum(p * log(p))
        epsilon = 1e-10
        uncertainty = -np.sum(proba * np.log(proba + epsilon), axis=1)
        
        predictions = np.argmax(proba, axis=1)
        
        return predictions, uncertainty


class AdaptiveMetaLearner(BaseMetaLearner):
    """
    Adaptive meta-learner that selects the best meta-learning strategy
    based on data characteristics.
    
    Implements an adaptive selection mechanism that chooses between
    different meta-learners based on cross-validation performance.
    """
    
    def __init__(self, config: MetaLearnerConfig):
        """Initialize adaptive meta-learner."""
        super().__init__(config)
        
        # Initialize candidate meta-learners
        self.candidates = {
            'logistic': LogisticMetaLearner(config),
            'xgboost': XGBoostMetaLearner(config),
            'lightgbm': LightGBMMetaLearner(config),
            'neural': NeuralMetaLearner(config)
        }
        
        self.best_learner = None
        self.best_score = -np.inf
        
        logger.info("Initialized Adaptive meta-learner with multiple candidates")
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ):
        """
        Fit by selecting best meta-learner through cross-validation.
        
        Args:
            X: Base model predictions
            y: True labels
            sample_weight: Sample weights
        """
        from sklearn.model_selection import cross_val_score
        
        best_score = -np.inf
        best_name = None
        
        # Evaluate each candidate
        for name, learner in self.candidates.items():
            try:
                # Fit learner
                learner.fit(X, y, sample_weight)
                
                # Cross-validation score
                scores = cross_val_score(
                    learner.model, X, y,
                    cv=min(3, X.shape[0] // 10),
                    scoring='accuracy'
                )
                mean_score = np.mean(scores)
                
                logger.info(f"{name} meta-learner CV score: {mean_score:.4f}")
                
                if mean_score > best_score:
                    best_score = mean_score
                    best_name = name
                    
            except Exception as e:
                logger.warning(f"Failed to evaluate {name}: {e}")
                continue
        
        # Select best learner
        if best_name:
            self.best_learner = self.candidates[best_name]
            self.model = self.best_learner.model
            self.best_score = best_score
            self.is_fitted = True
            
            logger.info(f"Selected {best_name} as best meta-learner (score: {best_score:.4f})")
        else:
            raise ValueError("No meta-learner could be fitted successfully")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using best meta-learner."""
        if not self.is_fitted or self.best_learner is None:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.best_learner.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using best meta-learner."""
        if not self.is_fitted or self.best_learner is None:
            raise ValueError("Meta-learner must be fitted before prediction")
        return self.best_learner.predict_proba(X)


class MetaLearnerFactory:
    """
    Factory class for creating meta-learners.
    
    Provides a unified interface for instantiating different types
    of meta-learners based on configuration.
    """
    
    _learners = {
        'logistic': LogisticMetaLearner,
        'xgboost': XGBoostMetaLearner,
        'lightgbm': LightGBMMetaLearner,
        'catboost': CatBoostMetaLearner,
        'neural': NeuralMetaLearner,
        'bayesian': BayesianMetaLearner,
        'adaptive': AdaptiveMetaLearner
    }
    
    @classmethod
    def create(cls, config: MetaLearnerConfig) -> BaseMetaLearner:
        """
        Create meta-learner instance.
        
        Args:
            config: Meta-learner configuration
            
        Returns:
            Meta-learner instance
            
        Raises:
            ValueError: If learner type is unknown
        """
        learner_type = config.learner_type.lower()
        
        if learner_type not in cls._learners:
            raise ValueError(
                f"Unknown meta-learner type: {learner_type}. "
                f"Available: {list(cls._learners.keys())}"
            )
        
        learner_class = cls._learners[learner_type]
        return learner_class(config)
    
    @classmethod
    def register(cls, name: str, learner_class: type):
        """
        Register new meta-learner type.
        
        Args:
            name: Name for the meta-learner
            learner_class: Meta-learner class
        """
        cls._learners[name] = learner_class
        logger.info(f"Registered meta-learner: {name}")


# Export public API
__all__ = [
    'MetaLearnerConfig',
    'BaseMetaLearner',
    'LogisticMetaLearner',
    'XGBoostMetaLearner',
    'LightGBMMetaLearner',
    'CatBoostMetaLearner',
    'NeuralMetaLearner',
    'BayesianMetaLearner',
    'AdaptiveMetaLearner',
    'MetaLearnerFactory'
]
