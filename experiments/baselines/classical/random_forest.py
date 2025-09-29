"""
Random Forest Baseline for AG News Text Classification
================================================================================
This module implements Random Forest classifiers as baseline models for text
classification, providing ensemble learning with bagging and feature randomness.

Random Forests combine multiple decision trees to reduce overfitting and improve
generalization through bootstrap aggregation and random feature selection.

References:
    - Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32
    - Genuer, R., et al. (2010). Variable Selection Using Random Forests
    - Louppe, G., et al. (2013). Understanding Variable Importances in Forests of Randomized Trees

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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from scipy.stats import randint, uniform

from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class RandomForestBaseline:
    """
    Random Forest baseline for text classification.
    
    This class implements:
    - Random Forest and Extra Trees classifiers
    - Feature importance analysis
    - Feature selection strategies
    - Hyperparameter optimization
    - Ensemble configuration tuning
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Union[str, int, float] = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int = -1,
        class_weight: Optional[Union[str, dict]] = "balanced",
        criterion: str = "gini",
        min_impurity_decrease: float = 0.0,
        vectorizer_type: str = "tfidf",
        max_vocab_features: int = 15000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        use_idf: bool = True,
        sublinear_tf: bool = True,
        feature_selection: Optional[str] = None,
        n_selected_features: int = 10000,
        use_extra_trees: bool = False,
        seed: int = 42
    ):
        """
        Initialize Random Forest baseline.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples required at leaf
            max_features: Number of features for best split
            bootstrap: Whether to use bootstrap samples
            oob_score: Whether to use out-of-bag samples for scoring
            n_jobs: Number of parallel jobs
            class_weight: Class weight strategy
            criterion: Split criterion ("gini" or "entropy")
            min_impurity_decrease: Minimum impurity decrease for split
            vectorizer_type: Type of text vectorizer
            max_vocab_features: Maximum vocabulary size
            ngram_range: Range of n-grams
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_idf: Whether to use IDF weighting
            sublinear_tf: Whether to use sublinear TF scaling
            feature_selection: Feature selection method ("chi2", "mutual_info", None)
            n_selected_features: Number of features to select
            use_extra_trees: Whether to use Extra Trees instead of Random Forest
            seed: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease
        self.vectorizer_type = vectorizer_type
        self.max_vocab_features = max_vocab_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.sublinear_tf = sublinear_tf
        self.feature_selection = feature_selection
        self.n_selected_features = n_selected_features
        self.use_extra_trees = use_extra_trees
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
            "feature_importance": {},
            "oob_score": None,
            "training_time": 0
        }
        
        # Initialize registry and metrics
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        # Set random seed
        set_seed(seed)
        
        # Build model
        self._build_model()
        
        classifier_name = "Extra Trees" if use_extra_trees else "Random Forest"
        logger.info(f"Initialized {classifier_name} baseline with {n_estimators} trees")
    
    def _build_model(self):
        """Build Random Forest model pipeline."""
        # Create vectorizer
        if self.vectorizer_type == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_vocab_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                use_idf=self.use_idf,
                sublinear_tf=self.sublinear_tf,
                strip_accents='unicode',
                lowercase=True,
                stop_words='english',
                token_pattern=r'\b\w+\b'
            )
        else:  # count
            self.vectorizer = CountVectorizer(
                max_features=self.max_vocab_features,
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
            if self.feature_selection == "chi2":
                self.selector = SelectKBest(chi2, k=self.n_selected_features)
            elif self.feature_selection == "mutual_info":
                self.selector = SelectKBest(mutual_info_classif, k=self.n_selected_features)
        
        # Create classifier
        if self.use_extra_trees:
            self.classifier = ExtraTreesClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                class_weight=self.class_weight,
                criterion=self.criterion,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.seed,
                verbose=0
            )
        else:
            self.classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                bootstrap=self.bootstrap,
                oob_score=self.oob_score,
                n_jobs=self.n_jobs,
                class_weight=self.class_weight,
                criterion=self.criterion,
                min_impurity_decrease=self.min_impurity_decrease,
                random_state=self.seed,
                verbose=0
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
        Train Random Forest model.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            sample_weight: Sample weights for training
            
        Returns:
            Training results
        """
        logger.info("Training Random Forest model")
        
        import time
        start_time = time.time()
        
        # Fit model
        if sample_weight is not None:
            self.pipeline.fit(train_texts, train_labels, classifier__sample_weight=sample_weight)
        else:
            self.pipeline.fit(train_texts, train_labels)
        
        training_time = time.time() - start_time
        self.results["training_time"] = training_time
        
        # Get OOB score if available
        if self.oob_score and hasattr(self.classifier, 'oob_score_'):
            self.results["oob_score"] = self.classifier.oob_score_
            logger.info(f"OOB Score: {self.classifier.oob_score_:.4f}")
        
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
        logger.info("Evaluating Random Forest model")
        
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
    
    def _extract_feature_importance(self):
        """Extract and analyze feature importance."""
        if self.pipeline is None:
            return
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # If feature selection was used, get selected features
        if self.selector is not None:
            selected_indices = self.selector.get_support(indices=True)
            feature_names = feature_names[selected_indices]
        
        # Get feature importances from classifier
        feature_importances = self.classifier.feature_importances_
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importances
        }).sort_values('importance', ascending=False)
        
        # Store top features
        self.results["feature_importance"] = {
            "top_features": importance_df.head(50).to_dict('records'),
            "total_features": len(feature_names),
            "importance_stats": {
                "mean": float(np.mean(feature_importances)),
                "std": float(np.std(feature_importances)),
                "max": float(np.max(feature_importances)),
                "min": float(np.min(feature_importances))
            }
        }
        
        # Log top features
        logger.info("Top 10 most important features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
    
    def hyperparameter_search(
        self,
        train_texts: Union[list, np.ndarray],
        train_labels: Union[list, np.ndarray],
        param_distributions: Optional[Dict[str, Any]] = None,
        n_iter: int = 30,
        cv: int = 3,
        scoring: str = 'f1_weighted',
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter search using RandomizedSearchCV.
        
        Args:
            train_texts: Training texts
            train_labels: Training labels
            param_distributions: Parameter distributions for search
            n_iter: Number of iterations
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Best parameters and results
        """
        logger.info("Performing hyperparameter search")
        
        if param_distributions is None:
            param_distributions = {
                'vectorizer__max_features': [5000, 10000, 15000, 20000],
                'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3)],
                'vectorizer__min_df': [1, 2, 5],
                'vectorizer__max_df': [0.9, 0.95, 1.0],
                'classifier__n_estimators': randint(50, 300),
                'classifier__max_depth': [10, 20, 30, None],
                'classifier__min_samples_split': randint(2, 20),
                'classifier__min_samples_leaf': randint(1, 10),
                'classifier__max_features': ['sqrt', 'log2', 0.3, 0.5]
            }
            
            # Add feature selector parameters if used
            if self.selector is not None:
                param_distributions['selector__k'] = [5000, 10000, 15000, 20000]
        
        # Create randomized search
        random_search = RandomizedSearchCV(
            self.pipeline,
            param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=self.seed
        )
        
        # Fit search
        random_search.fit(train_texts, train_labels)
        
        # Update model with best parameters
        self.pipeline = random_search.best_estimator_
        self.classifier = self.pipeline.named_steps['classifier']
        
        # Store results
        hyperparameter_results = {
            "best_params": random_search.best_params_,
            "best_score": random_search.best_score_,
            "cv_results": pd.DataFrame(random_search.cv_results_).to_dict()
        }
        
        logger.info(f"Best parameters: {random_search.best_params_}")
        logger.info(f"Best CV score: {random_search.best_score_:.4f}")
        
        return hyperparameter_results
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the trees in the forest.
        
        Returns:
            Tree statistics
        """
        if self.classifier is None:
            return {}
        
        tree_depths = []
        tree_leaves = []
        
        for tree in self.classifier.estimators_:
            tree_depths.append(tree.tree_.max_depth)
            tree_leaves.append(tree.tree_.n_leaves)
        
        return {
            "n_trees": len(self.classifier.estimators_),
            "tree_depths": {
                "mean": float(np.mean(tree_depths)),
                "std": float(np.std(tree_depths)),
                "min": int(np.min(tree_depths)),
                "max": int(np.max(tree_depths))
            },
            "tree_leaves": {
                "mean": float(np.mean(tree_leaves)),
                "std": float(np.std(tree_leaves)),
                "min": int(np.min(tree_leaves)),
                "max": int(np.max(tree_leaves))
            }
        }
    
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
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "criterion": self.criterion,
                "vectorizer_type": self.vectorizer_type,
                "max_vocab_features": self.max_vocab_features,
                "ngram_range": self.ngram_range,
                "use_extra_trees": self.use_extra_trees
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
        model_type = "Extra Trees" if self.use_extra_trees else "Random Forest"
        
        summary = {
            "model_type": model_type,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "criterion": self.criterion,
            "max_features": self.max_features,
            "vectorizer": self.vectorizer_type,
            "max_vocab_features": self.max_vocab_features,
            "ngram_range": self.ngram_range,
            "feature_selection": self.feature_selection,
            "training_time": self.results.get("training_time", 0),
            "oob_score": self.results.get("oob_score"),
            "performance": {
                "train_accuracy": self.results.get("train_metrics", {}).get("accuracy", 0),
                "val_accuracy": self.results.get("val_metrics", {}).get("accuracy", 0),
                "test_accuracy": self.results.get("test_metrics", {}).get("accuracy", 0)
            }
        }
        
        # Add tree statistics if available
        if self.classifier is not None:
            summary["tree_statistics"] = self.get_tree_statistics()
        
        return summary


def run_random_forest_experiments():
    """Run comprehensive Random Forest experiments."""
    logger.info("Starting Random Forest experiments")
    
    # Load data
    dataset = AGNewsDataset()
    train_data, val_data, test_data = dataset.load_splits()
    
    results = {}
    
    # Test different configurations
    configurations = [
        {"name": "rf_basic", "n_estimators": 100, "max_depth": None, "use_extra_trees": False},
        {"name": "rf_deep", "n_estimators": 200, "max_depth": 30, "use_extra_trees": False},
        {"name": "extra_trees", "n_estimators": 200, "max_depth": None, "use_extra_trees": True}
    ]
    
    for config in configurations:
        logger.info(f"\nTesting configuration: {config['name']}")
        
        # Create model
        model = RandomForestBaseline(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            use_extra_trees=config["use_extra_trees"],
            vectorizer_type="tfidf",
            max_vocab_features=15000,
            ngram_range=(1, 2),
            oob_score=True
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
    output_path = Path("outputs/results/random_forest_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_random_forest_experiments()
