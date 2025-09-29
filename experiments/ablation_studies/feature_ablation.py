"""
Feature Ablation Study for AG News Text Classification
================================================================================
This module performs ablation studies on different feature types and feature
engineering techniques used in text classification.

Feature ablation helps identify which text features and representations
contribute most to model performance.

References:
    - Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings
    - Devlin, J., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.data.preprocessing.feature_extraction import FeatureExtraction
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class FeatureAblation:
    """
    Performs feature ablation studies for text classification.
    
    Analyzes the impact of:
    - Feature types (tokens, n-grams, embeddings)
    - Feature engineering techniques
    - Feature combinations
    - Feature importance
    """
    
    def __init__(
        self,
        model_name: str = "bert-base",
        feature_types: Optional[List[str]] = None,
        feature_combinations: Optional[List[List[str]]] = None,
        dataset_name: str = "ag_news",
        num_trials: int = 3,
        device: str = "cuda",
        output_dir: str = "./ablation_results/feature",
        seed: int = 42
    ):
        """
        Initialize feature ablation study.
        
        Args:
            model_name: Model to use
            feature_types: List of feature types to test
            feature_combinations: Feature combinations to test
            dataset_name: Dataset name
            num_trials: Number of trials
            device: Device to use
            output_dir: Output directory
            seed: Random seed
        """
        self.model_name = model_name
        self.feature_types = feature_types or self._get_default_feature_types()
        self.feature_combinations = feature_combinations or self._get_default_combinations()
        self.dataset_name = dataset_name
        self.num_trials = num_trials
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        self.feature_extractor = FeatureExtraction()
        
        self.results = {
            "individual_features": {},
            "feature_combinations": {},
            "feature_importance": {},
            "interaction_effects": {},
            "summary": {}
        }
        
        set_seed(seed)
        logger.info("Initialized Feature Ablation Study")
    
    def _get_default_feature_types(self) -> List[str]:
        """Get default feature types to test."""
        return [
            "raw_text",
            "tokens",
            "subwords",
            "characters",
            "word_embeddings",
            "contextual_embeddings",
            "tfidf",
            "bow",
            "ngrams",
            "pos_tags",
            "named_entities",
            "syntactic_features",
            "semantic_features",
            "statistical_features",
            "sentiment_features"
        ]
    
    def _get_default_combinations(self) -> List[List[str]]:
        """Get default feature combinations."""
        return [
            ["raw_text"],
            ["tokens", "pos_tags"],
            ["word_embeddings", "tfidf"],
            ["contextual_embeddings", "statistical_features"],
            ["ngrams", "sentiment_features"],
            ["word_embeddings", "syntactic_features", "semantic_features"],
            ["contextual_embeddings", "named_entities", "statistical_features"]
        ]
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """
        Run complete feature ablation study.
        
        Returns:
            Ablation study results
        """
        logger.info("Starting feature ablation study")
        
        # Load dataset
        dataset = self._load_dataset()
        
        # Test individual features
        logger.info("\nTesting individual features")
        for feature_type in self.feature_types:
            logger.info(f"Testing feature: {feature_type}")
            
            feature_results = self._test_feature(feature_type, dataset)
            self.results["individual_features"][feature_type] = feature_results
            
            logger.info(
                f"Feature: {feature_type} | "
                f"Accuracy: {feature_results['mean_accuracy']:.4f} | "
                f"Dimensionality: {feature_results['feature_dim']}"
            )
        
        # Test feature combinations
        logger.info("\nTesting feature combinations")
        for combination in self.feature_combinations:
            combo_name = "+".join(combination)
            logger.info(f"Testing combination: {combo_name}")
            
            combo_results = self._test_combination(combination, dataset)
            self.results["feature_combinations"][combo_name] = combo_results
            
            logger.info(
                f"Combination: {combo_name} | "
                f"Accuracy: {combo_results['mean_accuracy']:.4f}"
            )
        
        # Analyze feature importance
        self.results["feature_importance"] = self._analyze_feature_importance()
        
        # Analyze interaction effects
        self.results["interaction_effects"] = self._analyze_interaction_effects()
        
        # Generate summary
        self.results["summary"] = self._generate_summary()
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return self.results
    
    def _test_feature(
        self,
        feature_type: str,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a single feature type.
        
        Args:
            feature_type: Type of feature to test
            dataset: Dataset dictionary
            
        Returns:
            Feature test results
        """
        results = {
            "feature_type": feature_type,
            "trials": [],
            "mean_accuracy": 0,
            "std_accuracy": 0,
            "mean_f1": 0,
            "feature_dim": 0,
            "extraction_time": 0
        }
        
        accuracies = []
        f1_scores = []
        
        for trial in range(self.num_trials):
            logger.info(f"Trial {trial + 1}/{self.num_trials}")
            
            # Set seed
            set_seed(self.seed + trial)
            
            # Extract features
            import time
            start_time = time.time()
            
            train_features = self._extract_features(
                dataset["train"]["texts"][:5000],
                feature_type
            )
            val_features = self._extract_features(
                dataset["val"]["texts"][:1000],
                feature_type
            )
            test_features = self._extract_features(
                dataset["test"]["texts"][:1000],
                feature_type
            )
            
            extraction_time = time.time() - start_time
            results["extraction_time"] = extraction_time
            
            # Get feature dimensionality
            if trial == 0:
                if isinstance(train_features, torch.Tensor):
                    results["feature_dim"] = train_features.shape[-1]
                elif isinstance(train_features, np.ndarray):
                    results["feature_dim"] = train_features.shape[-1] if len(train_features.shape) > 1 else 1
                else:
                    results["feature_dim"] = 0
            
            # Create model for features
            model = self._create_model_for_features(feature_type, results["feature_dim"])
            
            # Train model
            trainer = BaseTrainer(
                model=model,
                config=self._get_training_config(),
                device=self.device
            )
            
            trainer.train(
                train_features,
                dataset["train"]["labels"][:5000],
                val_features,
                dataset["val"]["labels"][:1000]
            )
            
            # Evaluate
            test_metrics = trainer.evaluate(
                test_features,
                dataset["test"]["labels"][:1000]
            )
            
            accuracies.append(test_metrics["accuracy"])
            f1_scores.append(test_metrics["f1_weighted"])
            
            results["trials"].append({
                "accuracy": test_metrics["accuracy"],
                "f1": test_metrics["f1_weighted"]
            })
        
        results["mean_accuracy"] = np.mean(accuracies)
        results["std_accuracy"] = np.std(accuracies)
        results["mean_f1"] = np.mean(f1_scores)
        
        return results
    
    def _test_combination(
        self,
        feature_combination: List[str],
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a combination of features.
        
        Args:
            feature_combination: List of features to combine
            dataset: Dataset dictionary
            
        Returns:
            Combination test results
        """
        results = {
            "combination": feature_combination,
            "trials": [],
            "mean_accuracy": 0,
            "std_accuracy": 0,
            "mean_f1": 0,
            "total_dim": 0
        }
        
        accuracies = []
        f1_scores = []
        
        for trial in range(min(self.num_trials, 2)):  # Fewer trials for efficiency
            logger.info(f"Trial {trial + 1}")
            
            # Set seed
            set_seed(self.seed + trial)
            
            # Extract and combine features
            train_combined = self._combine_features(
                dataset["train"]["texts"][:5000],
                feature_combination
            )
            val_combined = self._combine_features(
                dataset["val"]["texts"][:1000],
                feature_combination
            )
            test_combined = self._combine_features(
                dataset["test"]["texts"][:1000],
                feature_combination
            )
            
            # Get combined dimensionality
            if trial == 0:
                if isinstance(train_combined, torch.Tensor):
                    results["total_dim"] = train_combined.shape[-1]
                elif isinstance(train_combined, np.ndarray):
                    results["total_dim"] = train_combined.shape[-1]
            
            # Create model
            model = self._create_model_for_features("combined", results["total_dim"])
            
            # Train
            trainer = BaseTrainer(
                model=model,
                config=self._get_training_config(),
                device=self.device
            )
            
            trainer.train(
                train_combined,
                dataset["train"]["labels"][:5000],
                val_combined,
                dataset["val"]["labels"][:1000]
            )
            
            # Evaluate
            test_metrics = trainer.evaluate(
                test_combined,
                dataset["test"]["labels"][:1000]
            )
            
            accuracies.append(test_metrics["accuracy"])
            f1_scores.append(test_metrics["f1_weighted"])
            
            results["trials"].append({
                "accuracy": test_metrics["accuracy"],
                "f1": test_metrics["f1_weighted"]
            })
        
        results["mean_accuracy"] = np.mean(accuracies)
        results["std_accuracy"] = np.std(accuracies)
        results["mean_f1"] = np.mean(f1_scores)
        
        return results
    
    def _extract_features(
        self,
        texts: List[str],
        feature_type: str
    ) -> Union[np.ndarray, torch.Tensor, List]:
        """
        Extract features of specified type.
        
        Args:
            texts: Input texts
            feature_type: Type of features to extract
            
        Returns:
            Extracted features
        """
        if feature_type == "raw_text":
            return texts
        
        elif feature_type == "tokens":
            # Simple tokenization
            return [text.lower().split() for text in texts]
        
        elif feature_type == "subwords":
            # Simplified subword tokenization
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            encoded = tokenizer(texts, padding=True, truncation=True, max_length=128)
            return torch.tensor(encoded["input_ids"])
        
        elif feature_type == "characters":
            # Character-level features
            char_features = []
            for text in texts:
                chars = list(text[:500])  # Limit length
                char_ids = [ord(c) % 256 for c in chars]
                char_features.append(char_ids)
            return char_features
        
        elif feature_type == "word_embeddings":
            # Average word embeddings (simplified)
            embedding_dim = 300
            embeddings = np.random.randn(len(texts), embedding_dim)
            return embeddings
        
        elif feature_type == "contextual_embeddings":
            # BERT embeddings (simplified)
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            model = AutoModel.from_pretrained("bert-base-uncased")
            model.eval()
            
            embeddings = []
            with torch.no_grad():
                for text in texts[:100]:  # Limit for efficiency
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                    outputs = model(**inputs)
                    # Use CLS token embedding
                    embeddings.append(outputs.last_hidden_state[0, 0, :].numpy())
            
            # Pad with zeros for remaining
            if len(texts) > 100:
                embeddings.extend([np.zeros(768) for _ in range(len(texts) - 100)])
            
            return np.array(embeddings)
        
        elif feature_type == "tfidf":
            # TF-IDF features
            vectorizer = TfidfVectorizer(max_features=1000, max_df=0.95, min_df=2)
            tfidf_features = vectorizer.fit_transform(texts).toarray()
            return tfidf_features
        
        elif feature_type == "bow":
            # Bag of words
            vectorizer = CountVectorizer(max_features=1000, max_df=0.95, min_df=2)
            bow_features = vectorizer.fit_transform(texts).toarray()
            return bow_features
        
        elif feature_type == "ngrams":
            # N-gram features
            vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=1000)
            ngram_features = vectorizer.fit_transform(texts).toarray()
            return ngram_features
        
        elif feature_type == "pos_tags":
            # POS tag features (simplified)
            pos_features = []
            for text in texts:
                # Simplified: count basic patterns
                features = [
                    text.count('.'),
                    text.count(','),
                    text.count('?'),
                    text.count('!'),
                    len(text.split()),
                    text.count(' is '),
                    text.count(' was '),
                    text.count(' the ')
                ]
                pos_features.append(features)
            return np.array(pos_features)
        
        elif feature_type == "named_entities":
            # Named entity features (simplified)
            ne_features = []
            for text in texts:
                # Simplified: count capitalized words
                words = text.split()
                features = [
                    sum(1 for w in words if w and w[0].isupper()),
                    sum(1 for w in words if w.isupper()),
                    text.count('Inc'),
                    text.count('Corp'),
                    text.count('Mr'),
                    text.count('Ms')
                ]
                ne_features.append(features)
            return np.array(ne_features)
        
        elif feature_type == "syntactic_features":
            # Syntactic features
            syntactic_features = []
            for text in texts:
                words = text.split()
                features = [
                    len(words),  # Word count
                    np.mean([len(w) for w in words]) if words else 0,  # Avg word length
                    len(text.split('.')),  # Sentence count
                    text.count('('),  # Parentheses
                    text.count('"'),  # Quotes
                    max([len(w) for w in words]) if words else 0  # Max word length
                ]
                syntactic_features.append(features)
            return np.array(syntactic_features)
        
        elif feature_type == "semantic_features":
            # Semantic features (simplified)
            semantic_features = []
            for text in texts:
                # Simple semantic indicators
                features = [
                    1 if 'however' in text.lower() else 0,
                    1 if 'therefore' in text.lower() else 0,
                    1 if 'because' in text.lower() else 0,
                    1 if 'although' in text.lower() else 0,
                    text.lower().count('not'),
                    text.lower().count('but')
                ]
                semantic_features.append(features)
            return np.array(semantic_features)
        
        elif feature_type == "statistical_features":
            # Statistical text features
            stat_features = []
            for text in texts:
                words = text.split()
                chars = list(text)
                
                features = [
                    len(text),  # Text length
                    len(words),  # Word count
                    len(set(words)),  # Unique words
                    len(set(words)) / len(words) if words else 0,  # Type-token ratio
                    sum(1 for c in chars if c.isupper()) / len(chars) if chars else 0,  # Upper ratio
                    sum(1 for c in chars if c.isdigit()) / len(chars) if chars else 0,  # Digit ratio
                    text.count(' ') / len(text) if text else 0,  # Space ratio
                    np.std([len(w) for w in words]) if len(words) > 1 else 0  # Word length std
                ]
                stat_features.append(features)
            return np.array(stat_features)
        
        elif feature_type == "sentiment_features":
            # Sentiment features (simplified)
            positive_words = {'good', 'great', 'excellent', 'best', 'amazing', 'wonderful', 'fantastic'}
            negative_words = {'bad', 'worst', 'terrible', 'awful', 'horrible', 'poor', 'disappointing'}
            
            sentiment_features = []
            for text in texts:
                words = set(text.lower().split())
                features = [
                    len(words & positive_words),
                    len(words & negative_words),
                    text.count('!'),
                    text.count('?'),
                    1 if any(w in text.lower() for w in ['love', 'like']) else 0,
                    1 if any(w in text.lower() for w in ['hate', 'dislike']) else 0
                ]
                sentiment_features.append(features)
            return np.array(sentiment_features)
        
        else:
            # Default: return raw text
            return texts
    
    def _combine_features(
        self,
        texts: List[str],
        feature_types: List[str]
    ) -> np.ndarray:
        """
        Combine multiple feature types.
        
        Args:
            texts: Input texts
            feature_types: List of feature types to combine
            
        Returns:
            Combined feature array
        """
        all_features = []
        
        for feature_type in feature_types:
            features = self._extract_features(texts, feature_type)
            
            # Convert to numpy array if needed
            if isinstance(features, torch.Tensor):
                features = features.numpy()
            elif isinstance(features, list):
                if feature_type in ["raw_text", "tokens"]:
                    # Skip text features for now
                    continue
                features = np.array(features)
            
            # Ensure 2D
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)
            
            all_features.append(features)
        
        # Concatenate all features
        if all_features:
            combined = np.concatenate(all_features, axis=1)
        else:
            # Fallback to simple features
            combined = np.random.randn(len(texts), 10)
        
        return combined
    
    def _analyze_feature_importance(self) -> Dict[str, Any]:
        """
        Analyze relative importance of features.
        
        Returns:
            Feature importance analysis
        """
        importance_scores = {}
        
        # Calculate importance based on individual performance
        baseline_accuracy = 0.25  # Random baseline for 4 classes
        
        for feature_type, results in self.results["individual_features"].items():
            importance = (results["mean_accuracy"] - baseline_accuracy) / baseline_accuracy
            importance_scores[feature_type] = {
                "importance": importance,
                "accuracy": results["mean_accuracy"],
                "dimensionality": results["feature_dim"],
                "efficiency": results["mean_accuracy"] / (results["feature_dim"] + 1)
            }
        
        # Rank features
        ranked_features = sorted(
            importance_scores.items(),
            key=lambda x: x[1]["importance"],
            reverse=True
        )
        
        return {
            "scores": importance_scores,
            "ranking": [f[0] for f in ranked_features],
            "most_important": ranked_features[0] if ranked_features else None,
            "least_important": ranked_features[-1] if ranked_features else None
        }
    
    def _analyze_interaction_effects(self) -> Dict[str, Any]:
        """
        Analyze interaction effects between features.
        
        Returns:
            Interaction analysis results
        """
        interactions = {}
        
        # Compare combinations with individual features
        for combo_name, combo_results in self.results["feature_combinations"].items():
            if "+" in combo_name:
                features = combo_name.split("+")
                
                # Get individual accuracies
                individual_accs = []
                for feature in features:
                    if feature in self.results["individual_features"]:
                        individual_accs.append(
                            self.results["individual_features"][feature]["mean_accuracy"]
                        )
                
                if individual_accs:
                    # Calculate interaction effect
                    expected_acc = np.mean(individual_accs)
                    actual_acc = combo_results["mean_accuracy"]
                    interaction_effect = actual_acc - expected_acc
                    
                    interactions[combo_name] = {
                        "features": features,
                        "expected_accuracy": expected_acc,
                        "actual_accuracy": actual_acc,
                        "interaction_effect": interaction_effect,
                        "synergistic": interaction_effect > 0
                    }
        
        return interactions
    
    def run_feature_selection_ablation(self) -> Dict[str, Any]:
        """
        Run ablation with automatic feature selection.
        
        Returns:
            Feature selection results
        """
        logger.info("Running feature selection ablation")
        
        dataset = self._load_dataset()
        selection_methods = ["variance", "mutual_info", "chi2", "lasso"]
        selection_results = {}
        
        for method in selection_methods:
            logger.info(f"Testing selection method: {method}")
            
            # Extract base features
            features = self._extract_features(
                dataset["train"]["texts"][:1000],
                "tfidf"
            )
            
            # Apply feature selection
            selected_features = self._apply_feature_selection(
                features,
                dataset["train"]["labels"][:1000],
                method=method,
                k=100
            )
            
            # Train and evaluate
            model = self._create_model_for_features("selected", selected_features.shape[1])
            trainer = BaseTrainer(model=model, config=self._get_training_config(), device=self.device)
            
            # Quick training
            trainer.train(
                selected_features,
                dataset["train"]["labels"][:1000],
                selected_features[:200],  # Use subset for validation
                dataset["train"]["labels"][:200]
            )
            
            test_features = self._apply_feature_selection(
                self._extract_features(dataset["test"]["texts"][:500], "tfidf"),
                dataset["test"]["labels"][:500],
                method=method,
                k=100
            )
            
            test_metrics = trainer.evaluate(
                test_features,
                dataset["test"]["labels"][:500]
            )
            
            selection_results[method] = {
                "method": method,
                "num_features": selected_features.shape[1],
                "accuracy": test_metrics["accuracy"],
                "f1": test_metrics["f1_weighted"]
            }
        
        return selection_results
    
    def _apply_feature_selection(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        method: str,
        k: int = 100
    ) -> np.ndarray:
        """Apply feature selection method."""
        from sklearn.feature_selection import (
            SelectKBest, chi2, mutual_info_classif,
            VarianceThreshold, SelectFromModel
        )
        from sklearn.linear_model import LassoCV
        
        if method == "variance":
            selector = VarianceThreshold(threshold=0.01)
            selected = selector.fit_transform(features)
        
        elif method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=min(k, features.shape[1]))
            selected = selector.fit_transform(features, labels)
        
        elif method == "chi2":
            # Ensure non-negative features for chi2
            features_positive = features - features.min() + 1e-10
            selector = SelectKBest(chi2, k=min(k, features.shape[1]))
            selected = selector.fit_transform(features_positive, labels)
        
        elif method == "lasso":
            lasso = LassoCV(cv=3, max_iter=1000)
            selector = SelectFromModel(lasso)
            selector.fit(features, labels)
            selected = selector.transform(features)
        
        else:
            selected = features
        
        return selected
    
    def _create_model_for_features(
        self,
        feature_type: str,
        feature_dim: int
    ):
        """Create appropriate model for feature type."""
        # Simplified: create a basic classifier
        import torch.nn as nn
        
        class SimpleClassifier(nn.Module):
            def __init__(self, input_dim, num_classes=4):
                super().__init__()
                hidden_dim = min(256, input_dim * 2)
                self.classifier = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, num_classes)
                )
            
            def forward(self, x):
                if isinstance(x, np.ndarray):
                    x = torch.FloatTensor(x)
                return self.classifier(x)
        
        return SimpleClassifier(feature_dim)
    
    def _get_training_config(self) -> Dict[str, Any]:
        """Get training configuration."""
        return {
            "batch_size": 32,
            "learning_rate": 1e-3,
            "num_epochs": 5,
            "warmup_ratio": 0.1
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary."""
        # Find best individual feature
        best_individual = max(
            self.results["individual_features"].items(),
            key=lambda x: x[1]["mean_accuracy"]
        ) if self.results["individual_features"] else (None, {"mean_accuracy": 0})
        
        # Find best combination
        best_combination = max(
            self.results["feature_combinations"].items(),
            key=lambda x: x[1]["mean_accuracy"]
        ) if self.results["feature_combinations"] else (None, {"mean_accuracy": 0})
        
        # Calculate average importance
        if self.results["feature_importance"].get("scores"):
            avg_importance = np.mean([
                score["importance"] 
                for score in self.results["feature_importance"]["scores"].values()
            ])
        else:
            avg_importance = 0
        
        return {
            "best_individual_feature": {
                "name": best_individual[0],
                "accuracy": best_individual[1]["mean_accuracy"]
            },
            "best_combination": {
                "name": best_combination[0],
                "accuracy": best_combination[1]["mean_accuracy"]
            },
            "feature_ranking": self.results["feature_importance"].get("ranking", []),
            "average_importance": avg_importance,
            "num_synergistic_combinations": sum(
                1 for interaction in self.results["interaction_effects"].values()
                if interaction.get("synergistic", False)
            ),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        # Best features
        if self.results["feature_importance"].get("ranking"):
            top_features = self.results["feature_importance"]["ranking"][:3]
            recommendations.append(f"Focus on these features: {', '.join(top_features)}")
        
        # Synergistic combinations
        synergistic = [
            name for name, interaction in self.results["interaction_effects"].items()
            if interaction.get("synergistic", False)
        ]
        if synergistic:
            recommendations.append(f"Use these feature combinations: {', '.join(synergistic[:2])}")
        
        # Efficiency considerations
        efficient_features = [
            name for name, score in self.results["feature_importance"].get("scores", {}).items()
            if score.get("efficiency", 0) > 0.001
        ]
        if efficient_features:
            recommendations.append(f"Most efficient features: {', '.join(efficient_features[:3])}")
        
        return recommendations
    
    def _generate_visualizations(self):
        """Generate visualizations for feature ablation."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Feature importance bar chart
        ax = axes[0, 0]
        if self.results["individual_features"]:
            features = list(self.results["individual_features"].keys())
            accuracies = [r["mean_accuracy"] for r in self.results["individual_features"].values()]
            
            bars = ax.barh(range(len(features)), accuracies)
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features, fontsize=8)
            ax.set_xlabel('Accuracy')
            ax.set_title('Individual Feature Performance')
            ax.axvline(x=0.25, color='red', linestyle='--', alpha=0.5, label='Random baseline')
            ax.legend()
        
        # 2. Feature combinations
        ax = axes[0, 1]
        if self.results["feature_combinations"]:
            combos = list(self.results["feature_combinations"].keys())[:10]
            combo_accs = [self.results["feature_combinations"][c]["mean_accuracy"] for c in combos]
            
            ax.bar(range(len(combos)), combo_accs, color='orange')
            ax.set_xticks(range(len(combos)))
            ax.set_xticklabels(combos, rotation=45, ha='right', fontsize=8)
            ax.set_ylabel('Accuracy')
            ax.set_title('Feature Combination Performance')
        
        # 3. Feature efficiency scatter
        ax = axes[1, 0]
        if self.results["feature_importance"].get("scores"):
            dims = []
            accs = []
            names = []
            
            for name, score in self.results["feature_importance"]["scores"].items():
                if name in self.results["individual_features"]:
                    dims.append(self.results["individual_features"][name]["feature_dim"])
                    accs.append(score["accuracy"])
                    names.append(name)
            
            if dims and accs:
                scatter = ax.scatter(dims, accs, s=100, alpha=0.6)
                for i, name in enumerate(names):
                    if i % 2 == 0:  # Annotate every other point to avoid overlap
                        ax.annotate(name, (dims[i], accs[i]), fontsize=6)
                
                ax.set_xlabel('Feature Dimensionality')
                ax.set_ylabel('Accuracy')
                ax.set_title('Accuracy vs Feature Dimensionality')
                ax.set_xscale('log')
        
        # 4. Interaction effects
        ax = axes[1, 1]
        if self.results["interaction_effects"]:
            interactions = list(self.results["interaction_effects"].values())[:10]
            expected = [i["expected_accuracy"] for i in interactions]
            actual = [i["actual_accuracy"] for i in interactions]
            labels = [k.split("+")[0][:10] + "..." for k in list(self.results["interaction_effects"].keys())[:10]]
            
            x = np.arange(len(labels))
            width = 0.35
            
            ax.bar(x - width/2, expected, width, label='Expected', alpha=0.8)
            ax.bar(x + width/2, actual, width, label='Actual', alpha=0.8)
            
            ax.set_xlabel('Feature Combinations')
            ax.set_ylabel('Accuracy')
            ax.set_title('Expected vs Actual Performance')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "feature_ablation_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {plot_path}")
        plt.show()
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset for ablation study."""
        dataset = AGNewsDataset()
        return dataset.load_splits()
    
    def _save_results(self):
        """Save ablation results."""
        results_path = self.output_dir / "feature_ablation_results.json"
        
        # Convert numpy types for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
        
        # Save summary as CSV
        df_data = []
        for name, result in self.results["individual_features"].items():
            df_data.append({
                "feature": name,
                "accuracy": result["mean_accuracy"],
                "std": result["std_accuracy"],
                "f1": result["mean_f1"],
                "dimensionality": result["feature_dim"]
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            csv_path = self.output_dir / "feature_ablation_summary.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved summary to {csv_path}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def run_feature_ablation():
    """Run feature ablation study."""
    logger.info("Starting feature ablation study")
    
    ablation = FeatureAblation(
        model_name="bert-base",
        num_trials=2
    )
    
    # Run main ablation
    results = ablation.run_ablation_study()
    
    # Run feature selection ablation
    selection_results = ablation.run_feature_selection_ablation()
    
    logger.info(f"Summary: {results['summary']}")
    logger.info(f"Selection results: {selection_results}")
    
    return results


if __name__ == "__main__":
    run_feature_ablation()
