"""
Logistic Regression Baseline for AG News Text Classification
================================================================================
This module implements Logistic Regression classifiers as baseline models for
text classification, providing linear models with various regularization options.

Logistic Regression offers interpretable linear classification with support for
L1, L2, and Elastic Net regularization for feature selection and overfitting control.

References:
    - Genkin, A., et al. (2007). Large-scale Bayesian Logistic Regression for Text Categorization
    - Yu, H. F., et al. (2011). Feature Engineering and Classifier Ensemble for Text Classification
    - Zou, H., & Hastie, T. (2005). Regularization and Variable Selection via the Elastic Net

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
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.feature_selection import SelectFromModel, chi2, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class LogisticRegressionBaseline:
    """
    Logistic Regression baseline for text classification.
    
    This class implements:
    - Multiple solvers (liblinear, lbfgs, saga, sag)
    - L1, L2, and Elastic Net regularization
    - Feature selection strategies
    - Cross-validated regularization selection
    - Stochastic gradient descent variant
    """
    
    def __init__(
        self,
        penalty: str = "l2",
        C: float = 1.0,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        multi_class: str = "auto",
        class_weight: Optional[Union[str, dict]] = "balanced",
        dual: bool = False,
        tol: float = 1e-4,
        l1_ratio: Optional[float] = None,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        warm_start: bool = False,
        n_jobs: int = -1,
        vectorizer_type: str = "tfidf",
        max_features: int = 20000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        use_idf: bool = True,
        sublinear_tf: bool = True,
        use_hashing: bool = False,
        n_features_hash: int = 2**18,
        feature_selection: Optional[str] = None,
        n_selected_features: int = 10000,
        use_sgd: bool = False,
        learning_rate: str = "optimal",
        seed: int = 42
    ):
        """
        Initialize Logistic Regression baseline.
        
        Args:
            penalty: Regularization type ("l1", "l2", "elasticnet", "none")
            C: Inverse regularization strength
            solver: Optimization algorithm
            max_iter: Maximum iterations
            multi_class: Multi-class strategy
            class_weight: Class weight strategy
            dual: Whether to solve dual problem
            tol: Tolerance for stopping
            l1_ratio: Elastic Net mixing parameter
            fit_intercept: Whether to fit intercept
            intercept_scaling: Intercept scaling
            warm_start: Whether to reuse previous solution
            n_jobs: Number of parallel jobs
            vectorizer_type: Type of text vectorizer
            max_features: Maximum vocabulary size
            ngram_range: Range of n-grams
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_idf: Whether to use IDF weighting
            sublinear_tf: Whether to use sublinear TF
            use_hashing: Whether to use hashing vectorizer
            n_features_hash: Number of features for hashing
            feature_selection: Feature selection method
            n_selected_features: Number of features to select
            use_sgd: Whether to use SGD variant
            learning_rate: Learning rate schedule for SGD
            seed: Random seed
        """
        self.penalty = penalty
        self.C = C
        self.solver = solver
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.class_weight = class_weight
        self.dual = dual
        self.tol = tol
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.use_hashing = use_hashing
        self.n_features_hash = n_features_hash
        self.feature_selection = feature_selection
        self.n_selected_features = n_selected_features
        self.use_sgd = use_sgd
        self.learning_rate = learning_rate
        self.seed = seed
        
        # Initialize components
        self.vectorizer = None
        self.selector = None
        self.classifier = None
        self.pipeline = None
        
        # Results storage
        self.results = {
            "train_metrics": {},
            "val_metrics": {},
            "test_metrics": {},
            "coefficients": {},
            "training_time": 0,
            "convergence_info": {}
        }
        
        # Initialize registry and metrics
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        # Set random seed
        set_seed(seed)
        
        # Build model
        self._build_model()
        
        logger.info(f"Initialized Logistic Regression baseline with {penalty} penalty")
    
    def _build_model(self):
        """Build Logistic Regression model pipeline."""
        # Create vectorizer
        if self.use_hashing:
            self.vectorizer = HashingVectorizer(
                n_features=self.n_features_hash,
                ngram_range=self.ngram_range,
                strip_accents='unicode',
                lowercase=True,
                stop_words='english',
                token_pattern=r'\b\w+\b',
                norm='l2',
                alternate_sign=False
            )
        elif self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                use_idf=self.use_idf,
                sublinear_tf=self.sublinear_tf,
                strip_accents='unicode',
                lowercase=True,
                stop_words='english',
                token_pattern=r'\b\w+\b',
                norm='l2'
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
        
        # Create feature selector if specified
        if self.feature_selection:
            if self.feature_selection == "l1":
                # Use L1 regularization for feature selection
                selector_model = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=0.1,
                    random_state=self.seed
                )
                self.selector = SelectFromModel(selector_model, max_features=self.n_selected_features)
            elif self.feature_selection == "chi2":
                self.selector = SelectKBest(chi2, k=self.n_selected_features)
        
        # Create classifier
        if self.use_sgd:
            # Use SGDClassifier for large-scale learning
            if self.penalty == "elasticnet":
                self.classifier = SGDClassifier(
                    loss='log',
                    penalty=self.penalty,
                    alpha=1.0 / self.C,
                    l1_ratio=self.l1_ratio or 0.15,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    learning_rate=self.learning_rate,
                    class_weight=self.class_weight,
                    n_jobs=self.n_jobs,
                    random_state=self.seed,
                    warm_start=self.warm_start
                )
            else:
                self.classifier = SGDClassifier(
                    loss='log',
                    penalty=self.penalty,
                    alpha=1.0 / self.C,
                    max_iter=self.max_iter,
                    tol=self.tol,
                    learning_rate=self.learning_rate,
                    class_weight=self.class_weight,
                    n_jobs=self.n_jobs,
                    random_state=self.seed,
                    warm_start=self.warm_start
                )
        else:
            # Use standard LogisticRegression
            if self.penalty == "elasticnet":
                self.classifier = LogisticRegression(
                    penalty=self.penalty,
                    C=self.C,
                    solver='saga',  # Only saga supports elastic net
                    l1_ratio=self.l1_ratio or 0.5,
                    max_iter=self.max_iter,
                    multi_class=self.multi_class,
                    class_weight=self.class_weight,
                    dual=False,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                    intercept_scaling=self.intercept_scaling,
                    warm_start=self.warm_start,
                    n_jobs=self.n_jobs,
                    random_state=self.seed
                )
            else:
                self.classifier = LogisticRegression(
                    penalty=self.penalty,
                    C=self.C,
                    solver=self.solver,
                    max_iter=self.max_iter,
                    multi_class=self.multi_class,
                    class_weight=self.class_weight,
                    dual=self.dual,
                    tol=self.tol,
                    fit_intercept=self.fit_intercept,
                    intercept_scaling=self.intercept_scaling,
                    warm_start=self.warm_start,
                    n_jobs=self.n_jobs,
                    random_state=self.seed
                )
        
        # Create pipeline
        pipeline_steps = [('vectorizer', self.vectorizer)]
        
        if self.selector is not None:
            pipeline_steps.append(('selector', self.selector))
        
        pipeline_steps.append(('classifier', self.classifier))
        
        self.pipeline = Pipeline(pipeline_steps)
    
    def train(
        self,
        train_texts: Union[list, np.ndarray],
        train_labels: Union[list, np.ndarray],
        val_texts: Optional[Union[list, np.ndarray]] = None,
        val_labels: Optional[Union[list, np.ndarray]] = None,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train Logistic Regression model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            sample_weight: Sample weights for training
            
        Returns:
            Training results
        """
        logger.info("Training Logistic Regression model")
        
        import time
        start_time = time.time()
        
        # Fit model
        if sample_weight is not None:
            self.pipeline.fit(train_texts, train_labels, classifier__sample_weight=sample_weight)
        else:
            self.pipeline.fit(train_texts, train_labels)
        
        training_time = time.time() - start_time
        self.results["training_time"] = training_time
        
        # Check convergence
        self._check_convergence()
        
        # Evaluate on training set
        train_predictions = self.pipeline.predict(train_texts)
        train_proba = self.pipeline.predict_proba(train_texts)
        
        self.results["train_metrics"] = self._calculate_metrics(
            train_labels, train_predictions, train_proba
        )
        
        # Evaluate on validation set if provided
        if val_texts is not None and val_labels is not None:
            val_predictions = self.pipeline.predict(val_texts)
            val_proba = self.pipeline.predict_proba(val_texts)
            
            self.results["val_metrics"] = self._calculate_metrics(
                val_labels, val_predictions, val_proba
            )
        
        # Extract coefficient information
        self._extract_coefficients()
        
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
        
        return self.pipeline.predict_proba(texts)
    
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
        logger.info("Evaluating Logistic Regression model")
        
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
    
    def _check_convergence(self):
        """Check and report convergence information."""
        if hasattr(self.classifier, 'n_iter_'):
            if isinstance(self.classifier.n_iter_, np.ndarray):
                n_iter = int(self.classifier.n_iter_[0])
            else:
                n_iter = int(self.classifier.n_iter_)
            
            self.results["convergence_info"] = {
                "n_iterations": n_iter,
                "converged": n_iter < self.max_iter,
                "max_iterations": self.max_iter
            }
            
            if n_iter >= self.max_iter:
                logger.warning(f"Model did not converge within {self.max_iter} iterations")
    
    def _extract_coefficients(self):
        """Extract and analyze model coefficients."""
        if not hasattr(self.classifier, 'coef_'):
            return
        
        # Get feature names if not using hashing
        if not self.use_hashing:
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Adjust for feature selection
            if self.selector is not None:
                selected_indices = self.selector.get_support(indices=True)
                feature_names = feature_names[selected_indices]
        else:
            feature_names = [f"hash_feature_{i}" for i in range(self.n_features_hash)]
        
        # Get coefficients
        coefficients = self.classifier.coef_
        
        # For multiclass, process each class
        if coefficients.shape[0] > 1:
            self.results["coefficients"] = {}
            
            for class_idx in range(coefficients.shape[0]):
                class_coef = coefficients[class_idx, :]
                
                # Get top positive and negative coefficients
                top_positive_idx = np.argsort(class_coef)[-20:][::-1]
                top_negative_idx = np.argsort(class_coef)[:20]
                
                self.results["coefficients"][f"class_{class_idx}"] = {
                    "top_positive": [
                        {"feature": feature_names[idx], "coef": float(class_coef[idx])}
                        for idx in top_positive_idx if class_coef[idx] > 0
                    ],
                    "top_negative": [
                        {"feature": feature_names[idx], "coef": float(class_coef[idx])}
                        for idx in top_negative_idx if class_coef[idx] < 0
                    ],
                    "n_nonzero": int(np.count_nonzero(class_coef)),
                    "sparsity": float(1 - np.count_nonzero(class_coef) / len(class_coef))
                }
        
        # Calculate overall sparsity
        total_nonzero = np.count_nonzero(coefficients)
        total_features = coefficients.size
        self.results["coefficients"]["overall_sparsity"] = 1 - total_nonzero / total_features
        logger.info(f"Model sparsity: {self.results['coefficients']['overall_sparsity']:.2%}")
    
    def cross_validate_C(
        self,
        train_texts: Union[list, np.ndarray],
        train_labels: Union[list, np.ndarray],
        Cs: List[float] = None,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Cross-validate to find optimal C parameter.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            Cs: List of C values to try
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        logger.info("Cross-validating C parameter")
        
        if Cs is None:
            Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        
        # Create LogisticRegressionCV
        lr_cv = LogisticRegressionCV(
            Cs=Cs,
            cv=cv,
            penalty=self.penalty,
            solver=self.solver,
            scoring=scoring,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.seed
        )
        
        # Create pipeline with CV classifier
        pipeline_cv = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', lr_cv)
        ])
        
        # Fit
        pipeline_cv.fit(train_texts, train_labels)
        
        # Get results
        cv_results = {
            "best_C": float(lr_cv.C_[0]) if isinstance(lr_cv.C_, np.ndarray) else float(lr_cv.C_),
            "scores": lr_cv.scores_[1].mean(axis=0).tolist() if len(lr_cv.scores_.shape) > 2 else lr_cv.scores_.mean(axis=0).tolist(),
            "Cs": Cs
        }
        
        # Update model with best C
        self.C = cv_results["best_C"]
        self._build_model()
        
        logger.info(f"Best C: {cv_results['best_C']}")
        
        return cv_results
    
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
                "penalty": self.penalty,
                "C": self.C,
                "solver": self.solver,
                "max_iter": self.max_iter,
                "vectorizer_type": self.vectorizer_type,
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "use_sgd": self.use_sgd
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
        for key, value in config.items():
            setattr(self, key, value)
        
        self.classifier = self.pipeline.named_steps['classifier']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get model summary.
        
        Returns:
            Summary dictionary
        """
        model_type = "SGD Logistic Regression" if self.use_sgd else "Logistic Regression"
        
        summary = {
            "model_type": model_type,
            "penalty": self.penalty,
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "vectorizer": self.vectorizer_type,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "feature_selection": self.feature_selection,
            "training_time": self.results.get("training_time", 0),
            "convergence_info": self.results.get("convergence_info", {}),
            "performance": {
                "train_accuracy": self.results.get("train_metrics", {}).get("accuracy", 0),
                "val_accuracy": self.results.get("val_metrics", {}).get("accuracy", 0),
                "test_accuracy": self.results.get("test_metrics", {}).get("accuracy", 0)
            }
        }
        
        # Add sparsity information if available
        if "coefficients" in self.results and "overall_sparsity" in self.results["coefficients"]:
            summary["sparsity"] = self.results["coefficients"]["overall_sparsity"]
        
        return summary


def run_logistic_regression_experiments():
    """Run comprehensive Logistic Regression experiments."""
    logger.info("Starting Logistic Regression experiments")
    
    # Load data
    dataset = AGNewsDataset()
    train_data, val_data, test_data = dataset.load_splits()
    
    results = {}
    
    # Test different regularization strategies
    configurations = [
        {"name": "lr_l2", "penalty": "l2", "C": 1.0, "solver": "lbfgs"},
        {"name": "lr_l1", "penalty": "l1", "C": 1.0, "solver": "liblinear"},
        {"name": "lr_elasticnet", "penalty": "elasticnet", "C": 1.0, "solver": "saga", "l1_ratio": 0.5},
        {"name": "lr_sgd", "penalty": "l2", "C": 1.0, "use_sgd": True}
    ]
    
    for config in configurations:
        logger.info(f"\nTesting configuration: {config['name']}")
        
        # Create model
        model = LogisticRegressionBaseline(
            penalty=config["penalty"],
            C=config["C"],
            solver=config.get("solver", "lbfgs"),
            l1_ratio=config.get("l1_ratio"),
            use_sgd=config.get("use_sgd", False),
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
        results[config["name"]] = {
            "config": config,
            "metrics": test_metrics,
            "summary": model.get_summary()
        }
        
        # Save model
        model.save_model(f"outputs/models/{config['name']}.pkl")
    
    # Find best configuration
    best_config = max(results.keys(), key=lambda k: results[k]["metrics"]["accuracy"])
    logger.info(f"\nBest configuration: {best_config}")
    logger.info(f"Best accuracy: {results[best_config]['metrics']['accuracy']:.4f}")
    
    # Save results
    output_path = Path("outputs/results/logistic_regression_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_logistic_regression_experiments()
