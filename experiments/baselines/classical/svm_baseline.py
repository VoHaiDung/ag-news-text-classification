"""
Support Vector Machine Baseline for AG News Text Classification
================================================================================
This module implements SVM classifiers as baseline models for text classification,
including Linear SVM and kernel-based variants with various optimization strategies.

SVMs provide robust classification by finding optimal hyperplanes that maximize
the margin between classes in high-dimensional feature spaces.

References:
    - Joachims, T. (1998). Text Categorization with Support Vector Machines
    - Fan, R. E., et al. (2008). LIBLINEAR: A Library for Large Linear Classification
    - Hsu, C. W., & Lin, C. J. (2002). A Comparison of Methods for Multiclass Support Vector Machines

Author: Võ Hải Dũng
License: MIT
"""

import logging
import pickle
from typing import Dict, Any, Optional, Tuple, Union, List
from pathlib import Path
import json
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, make_scorer
)
from scipy.stats import uniform, loguniform

from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class SVMBaseline:
    """
    Support Vector Machine baseline for text classification.
    
    This class implements:
    - Linear and kernel SVMs
    - Advanced text vectorization
    - Hyperparameter optimization
    - Probability calibration
    - Feature selection
    """
    
    def __init__(
        self,
        kernel: str = "linear",
        C: float = 1.0,
        gamma: Union[str, float] = "scale",
        degree: int = 3,
        coef0: float = 0.0,
        max_iter: int = 1000,
        class_weight: Optional[Union[str, dict]] = "balanced",
        probability: bool = True,
        vectorizer_type: str = "tfidf",
        max_features: int = 20000,
        ngram_range: Tuple[int, int] = (1, 3),
        min_df: int = 2,
        max_df: float = 0.95,
        use_idf: bool = True,
        sublinear_tf: bool = True,
        norm: Optional[str] = "l2",
        dual: bool = True,
        seed: int = 42
    ):
        """
        Initialize SVM baseline.
        
        Args:
            kernel: Kernel type ("linear", "rbf", "poly", "sigmoid")
            C: Regularization parameter
            gamma: Kernel coefficient
            degree: Degree for polynomial kernel
            coef0: Independent term in kernel
            max_iter: Maximum iterations
            class_weight: Class weight strategy
            probability: Whether to enable probability estimates
            vectorizer_type: Type of vectorizer
            max_features: Maximum number of features
            ngram_range: Range of n-grams
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_idf: Whether to use IDF
            sublinear_tf: Whether to use sublinear TF
            norm: Normalization type
            dual: Whether to solve dual problem
            seed: Random seed
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.probability = probability
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.dual = dual
        self.seed = seed
        
        # Initialize components
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None
        
        # Results storage
        self.results = {
            "train_metrics": {},
            "val_metrics": {},
            "test_metrics": {},
            "support_vectors": {},
            "training_time": 0
        }
        
        # Initialize registry and metrics
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        # Set random seed
        set_seed(seed)
        
        # Build model
        self._build_model()
        
        logger.info(f"Initialized SVM baseline with {kernel} kernel")
    
    def _build_model(self):
        """Build SVM model pipeline."""
        # Create vectorizer
        if self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                use_idf=self.use_idf,
                sublinear_tf=self.sublinear_tf,
                norm=self.norm,
                strip_accents='unicode',
                lowercase=True,
                stop_words='english',
                token_pattern=r'\b\w+\b'
            )
        else:  # count
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                strip_accents='unicode',
                lowercase=True,
                stop_words='english',
                token_pattern=r'\b\w+\b'
            )
        
        # Create classifier
        if self.kernel == "linear":
            # Use LinearSVC for efficiency
            base_classifier = LinearSVC(
                C=self.C,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                dual=self.dual,
                random_state=self.seed,
                loss='squared_hinge',
                penalty='l2',
                tol=1e-4
            )
            
            # Wrap with calibration for probability estimates
            if self.probability:
                self.classifier = CalibratedClassifierCV(
                    base_classifier,
                    cv=3,
                    method='sigmoid'
                )
            else:
                self.classifier = base_classifier
        else:
            # Use SVC for non-linear kernels
            self.classifier = SVC(
                C=self.C,
                kernel=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                max_iter=self.max_iter,
                class_weight=self.class_weight,
                probability=self.probability,
                random_state=self.seed,
                cache_size=500,
                decision_function_shape='ovr',
                tol=1e-3
            )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
    
    def train(
        self,
        train_texts: Union[list, np.ndarray],
        train_labels: Union[list, np.ndarray],
        val_texts: Optional[Union[list, np.ndarray]] = None,
        val_labels: Optional[Union[list, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train SVM model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            sample_weight: Sample weights for training
            
        Returns:
            Training results
        """
        logger.info("Training SVM model")
        
        import time
        start_time = time.time()
        
        # Fit model
        if sample_weight is not None:
            self.pipeline.fit(train_texts, train_labels, classifier__sample_weight=sample_weight)
        else:
            self.pipeline.fit(train_texts, train_labels)
        
        training_time = time.time() - start_time
        self.results["training_time"] = training_time
        
        # Evaluate on training set
        train_predictions = self.pipeline.predict(train_texts)
        
        if self.probability or isinstance(self.classifier, CalibratedClassifierCV):
            train_proba = self.pipeline.predict_proba(train_texts)
        else:
            # Use decision function as proxy for probabilities
            train_decision = self.pipeline.decision_function(train_texts)
            train_proba = self._decision_to_proba(train_decision)
        
        self.results["train_metrics"] = self._calculate_metrics(
            train_labels, train_predictions, train_proba
        )
        
        # Evaluate on validation set if provided
        if val_texts is not None and val_labels is not None:
            val_predictions = self.pipeline.predict(val_texts)
            
            if self.probability or isinstance(self.classifier, CalibratedClassifierCV):
                val_proba = self.pipeline.predict_proba(val_texts)
            else:
                val_decision = self.pipeline.decision_function(val_texts)
                val_proba = self._decision_to_proba(val_decision)
            
            self.results["val_metrics"] = self._calculate_metrics(
                val_labels, val_predictions, val_proba
            )
        
        # Extract support vector information
        self._extract_support_vectors()
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Train accuracy: {self.results['train_metrics']['accuracy']:.4f}")
        
        if val_texts is not None:
            logger.info(f"Val accuracy: {self.results['val_metrics']['accuracy']:.4f}")
        
        return self.results
    
    def predict(
        self,
        texts: Union[list, np.ndarray]
    ) -> np.ndarray:
        """
        Make predictions on texts.
        
        Args:
            texts: Input texts
            
        Returns:
            Predicted labels
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before prediction")
        
        return self.pipeline.predict(texts)
    
    def predict_proba(
        self,
        texts: Union[list, np.ndarray]
    ) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            texts: Input texts
            
        Returns:
            Prediction probabilities
        """
        if self.pipeline is None:
            raise ValueError("Model must be trained before prediction")
        
        if self.probability or isinstance(self.classifier, CalibratedClassifierCV):
            return self.pipeline.predict_proba(texts)
        else:
            # Convert decision function to probabilities
            decision = self.pipeline.decision_function(texts)
            return self._decision_to_proba(decision)
    
    def evaluate(
        self,
        test_texts: Union[list, np.ndarray],
        test_labels: Union[list, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.
        
        Args:
            test_texts: Test texts
            test_labels: Test labels
            
        Returns:
            Test metrics
        """
        logger.info("Evaluating SVM model")
        
        predictions = self.predict(test_texts)
        probabilities = self.predict_proba(test_texts)
        
        self.results["test_metrics"] = self._calculate_metrics(
            test_labels, predictions, probabilities
        )
        
        logger.info(f"Test accuracy: {self.results['test_metrics']['accuracy']:.4f}")
        
        return self.results["test_metrics"]
    
    def _calculate_metrics(
        self,
        true_labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calculate classification metrics.
        
        Args:
            true_labels: True labels
            predictions: Predicted labels
            probabilities: Prediction probabilities
            
        Returns:
            Metrics dictionary
        """
        metrics = {
            "accuracy": accuracy_score(true_labels, predictions),
            "precision_macro": precision_score(true_labels, predictions, average='macro', zero_division=0),
            "precision_weighted": precision_score(true_labels, predictions, average='weighted', zero_division=0),
            "recall_macro": recall_score(true_labels, predictions, average='macro', zero_division=0),
            "recall_weighted": recall_score(true_labels, predictions, average='weighted', zero_division=0),
            "f1_macro": f1_score(true_labels, predictions, average='macro', zero_division=0),
            "f1_weighted": f1_score(true_labels, predictions, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(true_labels, predictions).tolist()
        }
        
        # Add advanced metrics if probabilities available
        if probabilities is not None:
            advanced_metrics = self.metrics_calculator.calculate_advanced_metrics(
                true_labels, predictions, probabilities
            )
            metrics.update(advanced_metrics)
        
        return metrics
    
    def _decision_to_proba(self, decision_values: np.ndarray) -> np.ndarray:
        """
        Convert decision function values to probabilities.
        
        Args:
            decision_values: Decision function values
            
        Returns:
            Probability estimates
        """
        # Apply sigmoid transformation
        # For multiclass, use softmax
        if len(decision_values.shape) == 1:
            # Binary classification
            proba = 1 / (1 + np.exp(-decision_values))
            return np.column_stack([1 - proba, proba])
        else:
            # Multiclass - apply softmax
            exp_values = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
    def _extract_support_vectors(self):
        """Extract support vector information."""
        if hasattr(self.classifier, 'support_vectors_'):
            self.results["support_vectors"] = {
                "n_support": self.classifier.n_support_.tolist(),
                "support_indices": self.classifier.support_.tolist()[:100]  # Limit to first 100
            }
        elif hasattr(self.classifier, 'coef_'):
            # For LinearSVC
            coef = self.classifier.coef_
            self.results["support_vectors"] = {
                "coef_shape": coef.shape,
                "n_nonzero": np.count_nonzero(coef)
            }
    
    def hyperparameter_search(
        self,
        train_texts: Union[list, np.ndarray],
        train_labels: Union[list, np.ndarray],
        param_distributions: Optional[Dict[str, Any]] = None,
        n_iter: int = 20,
        cv: int = 3,
        scoring: str = 'f1_weighted',
        n_jobs: int = -1,
        search_type: str = "random"
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            param_distributions: Parameter distributions for search
            n_iter: Number of iterations for random search
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            search_type: Type of search ("random" or "grid")
            
        Returns:
            Best parameters and results
        """
        logger.info(f"Performing {search_type} hyperparameter search")
        
        if param_distributions is None:
            if self.kernel == "linear":
                param_distributions = {
                    'vectorizer__max_features': [10000, 20000, 30000],
                    'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
                    'vectorizer__min_df': [1, 2, 5],
                    'classifier__C': loguniform(0.001, 100) if search_type == "random" else [0.01, 0.1, 1, 10, 100]
                }
            else:
                param_distributions = {
                    'vectorizer__max_features': [10000, 20000],
                    'vectorizer__ngram_range': [(1, 1), (1, 2)],
                    'classifier__C': loguniform(0.1, 100) if search_type == "random" else [0.1, 1, 10, 100],
                    'classifier__gamma': loguniform(0.0001, 1) if search_type == "random" else [0.001, 0.01, 0.1, 1]
                }
        
        # Create search object
        if search_type == "random":
            search = RandomizedSearchCV(
                self.pipeline,
                param_distributions,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1,
                random_state=self.seed
            )
        else:
            search = GridSearchCV(
                self.pipeline,
                param_distributions,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
        
        # Fit search
        search.fit(train_texts, train_labels)
        
        # Update model with best parameters
        self.pipeline = search.best_estimator_
        
        # Store results
        hyperparameter_results = {
            "best_params": search.best_params_,
            "best_score": search.best_score_,
            "cv_results": pd.DataFrame(search.cv_results_).to_dict()
        }
        
        logger.info(f"Best parameters: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_:.4f}")
        
        return hyperparameter_results
    
    def save_model(self, filepath: str):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "pipeline": self.pipeline,
            "results": self.results,
            "config": {
                "kernel": self.kernel,
                "C": self.C,
                "gamma": self.gamma,
                "vectorizer_type": self.vectorizer_type,
                "max_features": self.max_features,
                "ngram_range": self.ngram_range
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model from file.
        
        Args:
            filepath: Path to load model from
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.pipeline = model_data["pipeline"]
        self.results = model_data["results"]
        
        # Update config
        config = model_data["config"]
        self.kernel = config["kernel"]
        self.C = config["C"]
        self.gamma = config["gamma"]
        self.vectorizer_type = config["vectorizer_type"]
        self.max_features = config["max_features"]
        self.ngram_range = config["ngram_range"]
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get model summary.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "model_type": f"SVM ({self.kernel} kernel)",
            "C": self.C,
            "gamma": self.gamma if self.kernel != "linear" else None,
            "vectorizer": self.vectorizer_type,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "training_time": self.results.get("training_time", 0),
            "performance": {
                "train_accuracy": self.results.get("train_metrics", {}).get("accuracy", 0),
                "val_accuracy": self.results.get("val_metrics", {}).get("accuracy", 0),
                "test_accuracy": self.results.get("test_metrics", {}).get("accuracy", 0)
            }
        }
        
        # Add support vector info
        if "support_vectors" in self.results:
            summary["support_vectors"] = self.results["support_vectors"]
        
        return summary


def run_svm_experiments():
    """Run comprehensive SVM experiments."""
    logger.info("Starting SVM experiments")
    
    # Load data
    dataset = AGNewsDataset()
    train_data, val_data, test_data = dataset.load_splits()
    
    results = {}
    
    # Test different kernels
    kernels = ["linear", "rbf"]
    
    for kernel in kernels:
        logger.info(f"\nTesting SVM with {kernel} kernel")
        
        # Create model
        model = SVMBaseline(
            kernel=kernel,
            C=1.0,
            vectorizer_type="tfidf",
            max_features=20000,
            ngram_range=(1, 2)
        )
        
        # Train
        model.train(
            train_data["texts"],
            train_data["labels"],
            val_data["texts"],
            val_data["labels"]
        )
        
        # Evaluate
        test_metrics = model.evaluate(
            test_data["texts"],
            test_data["labels"]
        )
        
        # Store results
        results[kernel] = {
            "metrics": test_metrics,
            "summary": model.get_summary()
        }
        
        # Save model
        model.save_model(f"outputs/models/svm_{kernel}.pkl")
    
    # Find best kernel
    best_kernel = max(results.keys(), key=lambda k: results[k]["metrics"]["accuracy"])
    logger.info(f"\nBest kernel: {best_kernel}")
    logger.info(f"Best accuracy: {results[best_kernel]['metrics']['accuracy']:.4f}")
    
    # Save results
    output_path = Path("outputs/results/svm_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_svm_experiments()
