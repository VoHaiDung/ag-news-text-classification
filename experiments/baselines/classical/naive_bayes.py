"""
Naive Bayes Baseline for AG News Text Classification
================================================================================
This module implements Naive Bayes classifiers as baseline models for text
classification, including Multinomial, Bernoulli, and Complement variants.

Naive Bayes provides a probabilistic approach to classification based on
Bayes' theorem with strong independence assumptions between features.

References:
    - McCallum, A., & Nigam, K. (1998). A Comparison of Event Models for Naive Bayes Text Classification
    - Rennie, J. D., et al. (2003). Tackling the Poor Assumptions of Naive Bayes Text Classifiers
    - Kibriya, A. M., et al. (2004). Multinomial Naive Bayes for Text Categorization Revisited

Author: Võ Hải Dũng
License: MIT
"""

import logging
import pickle
from typing import Dict, Any, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class NaiveBayesBaseline:
    """
    Naive Bayes baseline models for text classification.
    
    This class implements:
    - Multiple Naive Bayes variants
    - TF-IDF and Count vectorization
    - Hyperparameter optimization
    - Feature selection
    - Ensemble of NB models
    """
    
    def __init__(
        self,
        variant: str = "multinomial",
        vectorizer_type: str = "tfidf",
        max_features: int = 10000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        use_idf: bool = True,
        sublinear_tf: bool = True,
        alpha: float = 1.0,
        fit_prior: bool = True,
        class_prior: Optional[np.ndarray] = None,
        seed: int = 42
    ):
        """
        Initialize Naive Bayes baseline.
        
        Args:
            variant: Type of Naive Bayes ("multinomial", "bernoulli", "complement")
            vectorizer_type: Type of vectorizer ("tfidf", "count")
            max_features: Maximum number of features
            ngram_range: Range of n-grams
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_idf: Whether to use IDF weighting
            sublinear_tf: Whether to use sublinear TF scaling
            alpha: Smoothing parameter
            fit_prior: Whether to learn class prior probabilities
            class_prior: Prior probabilities of classes
            seed: Random seed
        """
        self.variant = variant
        self.vectorizer_type = vectorizer_type
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
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
            "feature_importance": {},
            "training_time": 0
        }
        
        # Initialize registry and metrics
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        # Set random seed
        set_seed(seed)
        
        # Build model
        self._build_model()
        
        logger.info(f"Initialized {variant} Naive Bayes baseline")
    
    def _build_model(self):
        """Build Naive Bayes model pipeline."""
        # Create vectorizer
        if self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                use_idf=self.use_idf,
                sublinear_tf=self.sublinear_tf,
                strip_accents='unicode',
                lowercase=True,
                stop_words='english'
            )
        else:  # count
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                strip_accents='unicode',
                lowercase=True,
                stop_words='english'
            )
        
        # Create classifier
        if self.variant == "multinomial":
            self.classifier = MultinomialNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior
            )
        elif self.variant == "bernoulli":
            self.classifier = BernoulliNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior,
                binarize=0.0
            )
        elif self.variant == "complement":
            self.classifier = ComplementNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior,
                norm=False
            )
        else:
            raise ValueError(f"Unknown Naive Bayes variant: {self.variant}")
        
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
        val_labels: Optional[Union[list, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Train Naive Bayes model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            
        Returns:
            Training results
        """
        logger.info("Training Naive Bayes model")
        
        import time
        start_time = time.time()
        
        # Fit model
        self.pipeline.fit(train_texts, train_labels)
        
        training_time = time.time() - start_time
        self.results["training_time"] = training_time
        
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
        
        # Extract feature importance
        self._extract_feature_importance()
        
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
        logger.info("Evaluating Naive Bayes model")
        
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
            "precision_macro": precision_score(true_labels, predictions, average='macro'),
            "precision_weighted": precision_score(true_labels, predictions, average='weighted'),
            "recall_macro": recall_score(true_labels, predictions, average='macro'),
            "recall_weighted": recall_score(true_labels, predictions, average='weighted'),
            "f1_macro": f1_score(true_labels, predictions, average='macro'),
            "f1_weighted": f1_score(true_labels, predictions, average='weighted'),
            "confusion_matrix": confusion_matrix(true_labels, predictions).tolist()
        }
        
        # Add advanced metrics if probabilities available
        if probabilities is not None:
            advanced_metrics = self.metrics_calculator.calculate_advanced_metrics(
                true_labels, predictions, probabilities
            )
            metrics.update(advanced_metrics)
        
        return metrics
    
    def _extract_feature_importance(self):
        """Extract feature importance from trained model."""
        if self.pipeline is None:
            return
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get feature log probabilities
        if hasattr(self.classifier, 'feature_log_prob_'):
            # For Multinomial and Bernoulli NB
            feature_importance = np.exp(self.classifier.feature_log_prob_)
            
            # Calculate importance for each class
            self.results["feature_importance"] = {}
            
            for class_idx in range(feature_importance.shape[0]):
                class_importance = feature_importance[class_idx, :]
                
                # Get top features for this class
                top_indices = np.argsort(class_importance)[-20:][::-1]
                
                self.results["feature_importance"][f"class_{class_idx}"] = [
                    {
                        "feature": feature_names[idx],
                        "importance": float(class_importance[idx])
                    }
                    for idx in top_indices
                ]
    
    def hyperparameter_search(
        self,
        train_texts: Union[list, np.ndarray],
        train_labels: Union[list, np.ndarray],
        param_grid: Optional[Dict[str, list]] = None,
        cv: int = 5,
        scoring: str = 'accuracy',
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search using GridSearchCV.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            param_grid: Parameter grid for search
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Best parameters and results
        """
        logger.info("Performing hyperparameter search")
        
        if param_grid is None:
            param_grid = {
                'vectorizer__max_features': [5000, 10000, 20000],
                'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'vectorizer__min_df': [1, 2, 5],
                'vectorizer__max_df': [0.9, 0.95, 1.0],
                'classifier__alpha': [0.01, 0.1, 0.5, 1.0, 2.0]
            }
        
        # Create grid search
        grid_search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(train_texts, train_labels)
        
        # Update model with best parameters
        self.pipeline = grid_search.best_estimator_
        
        # Store results
        hyperparameter_results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": pd.DataFrame(grid_search.cv_results_).to_dict()
        }
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return hyperparameter_results
    
    def cross_validate(
        self,
        texts: Union[list, np.ndarray],
        labels: Union[list, np.ndarray],
        cv: int = 5,
        scoring: Union[str, list] = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.
        
        Args:
            texts: Input texts
            labels: Labels
            cv: Number of folds
            scoring: Scoring metric(s)
            
        Returns:
            Cross-validation results
        """
        logger.info(f"Performing {cv}-fold cross-validation")
        
        if isinstance(scoring, str):
            scores = cross_val_score(
                self.pipeline, texts, labels, 
                cv=cv, scoring=scoring
            )
            
            cv_results = {
                scoring: {
                    "scores": scores.tolist(),
                    "mean": float(np.mean(scores)),
                    "std": float(np.std(scores))
                }
            }
        else:
            from sklearn.model_selection import cross_validate as sklearn_cv
            
            scores = sklearn_cv(
                self.pipeline, texts, labels,
                cv=cv, scoring=scoring
            )
            
            cv_results = {}
            for metric in scoring:
                key = f"test_{metric}"
                if key in scores:
                    cv_results[metric] = {
                        "scores": scores[key].tolist(),
                        "mean": float(np.mean(scores[key])),
                        "std": float(np.std(scores[key]))
                    }
        
        logger.info(f"Cross-validation results: {cv_results}")
        
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
                "variant": self.variant,
                "vectorizer_type": self.vectorizer_type,
                "max_features": self.max_features,
                "ngram_range": self.ngram_range,
                "alpha": self.alpha
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
        self.variant = config["variant"]
        self.vectorizer_type = config["vectorizer_type"]
        self.max_features = config["max_features"]
        self.ngram_range = config["ngram_range"]
        self.alpha = config["alpha"]
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get model summary.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "model_type": f"Naive Bayes ({self.variant})",
            "vectorizer": self.vectorizer_type,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "alpha": self.alpha,
            "training_time": self.results.get("training_time", 0),
            "performance": {
                "train_accuracy": self.results.get("train_metrics", {}).get("accuracy", 0),
                "val_accuracy": self.results.get("val_metrics", {}).get("accuracy", 0),
                "test_accuracy": self.results.get("test_metrics", {}).get("accuracy", 0)
            }
        }
        
        # Add vocabulary size if model is trained
        if self.pipeline is not None and hasattr(self.vectorizer, 'vocabulary_'):
            summary["vocabulary_size"] = len(self.vectorizer.vocabulary_)
        
        return summary


def run_naive_bayes_experiments():
    """Run comprehensive Naive Bayes experiments."""
    logger.info("Starting Naive Bayes experiments")
    
    # Load data
    dataset = AGNewsDataset()
    train_data, val_data, test_data = dataset.load_splits()
    
    results = {}
    
    # Test different NB variants
    variants = ["multinomial", "bernoulli", "complement"]
    
    for variant in variants:
        logger.info(f"\nTesting {variant} Naive Bayes")
        
        # Create model
        model = NaiveBayesBaseline(
            variant=variant,
            vectorizer_type="tfidf",
            max_features=10000,
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
        results[variant] = {
            "metrics": test_metrics,
            "summary": model.get_summary()
        }
        
        # Save model
        model.save_model(f"outputs/models/naive_bayes_{variant}.pkl")
    
    # Find best variant
    best_variant = max(results.keys(), key=lambda k: results[k]["metrics"]["accuracy"])
    logger.info(f"\nBest variant: {best_variant}")
    logger.info(f"Best accuracy: {results[best_variant]['metrics']['accuracy']:.4f}")
    
    # Save results
    output_path = Path("outputs/results/naive_bayes_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_naive_bayes_experiments()
