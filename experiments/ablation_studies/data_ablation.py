"""
Data Ablation Study for AG News Text Classification
================================================================================
This module performs ablation studies on data-related aspects including data
size, augmentation techniques, and data quality filters.

Data ablation helps understand the impact of training data quantity and quality
on model performance, guiding data collection and augmentation strategies.

References:
    - Banko, M., & Brill, E. (2001). Scaling to Very Very Large Corpora
    - Wei, J., & Zou, K. (2019). EDA: Easy Data Augmentation Techniques

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.data.augmentation.base_augmenter import BaseAugmenter
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class DataAblation:
    """
    Performs data ablation studies for text classification.
    
    Analyzes the impact of:
    - Training data size
    - Data augmentation techniques
    - Data quality and filtering
    - Class balance
    """
    
    def __init__(
        self,
        model_name: str = "bert-base",
        model_config: Optional[Dict[str, Any]] = None,
        data_sizes: Optional[List[float]] = None,
        augmentation_methods: Optional[List[str]] = None,
        num_trials: int = 3,
        device: str = "cuda",
        output_dir: str = "./ablation_results/data",
        seed: int = 42
    ):
        """
        Initialize data ablation study.
        
        Args:
            model_name: Model to use
            model_config: Model configuration
            data_sizes: List of data size fractions to test
            augmentation_methods: List of augmentation methods
            num_trials: Number of trials
            device: Device to use
            output_dir: Output directory
            seed: Random seed
        """
        self.model_name = model_name
        self.model_config = model_config or self._get_default_config()
        self.data_sizes = data_sizes or [0.1, 0.25, 0.5, 0.75, 1.0]
        self.augmentation_methods = augmentation_methods or [
            "none",
            "synonym_replacement",
            "random_insertion",
            "random_swap",
            "random_deletion",
            "back_translation",
            "paraphrase",
            "mixup",
            "all"
        ]
        self.num_trials = num_trials
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        self.results = {
            "data_size": {},
            "augmentation": {},
            "quality": {},
            "class_balance": {},
            "summary": {}
        }
        
        set_seed(seed)
        logger.info("Initialized Data Ablation Study")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            "max_length": 256,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "warmup_ratio": 0.1
        }
    
    def run_data_size_ablation(self) -> Dict[str, Any]:
        """
        Test model performance with different training data sizes.
        
        Returns:
            Data size ablation results
        """
        logger.info("Running data size ablation")
        
        # Load full dataset
        dataset = AGNewsDataset()
        full_data = dataset.load_splits()
        
        for data_fraction in self.data_sizes:
            logger.info(f"\nTesting with {data_fraction*100}% of training data")
            
            size_results = {
                "fraction": data_fraction,
                "num_samples": 0,
                "trials": [],
                "mean_accuracy": 0,
                "std_accuracy": 0,
                "mean_f1": 0,
                "training_time": 0
            }
            
            accuracies = []
            f1_scores = []
            
            for trial in range(self.num_trials):
                logger.info(f"Trial {trial + 1}/{self.num_trials}")
                
                # Sample data
                sampled_data = self._sample_data(full_data, data_fraction)
                size_results["num_samples"] = len(sampled_data["train"]["texts"])
                
                # Train model
                model = self.factory.create_model(self.model_name, **self.model_config)
                trainer = BaseTrainer(model=model, config=self.model_config, device=self.device)
                
                # Training
                trainer.train(
                    sampled_data["train"]["texts"],
                    sampled_data["train"]["labels"],
                    sampled_data["val"]["texts"],
                    sampled_data["val"]["labels"]
                )
                
                # Evaluation
                test_metrics = trainer.evaluate(
                    full_data["test"]["texts"],
                    full_data["test"]["labels"]
                )
                
                accuracies.append(test_metrics["accuracy"])
                f1_scores.append(test_metrics["f1_weighted"])
                
                size_results["trials"].append({
                    "accuracy": test_metrics["accuracy"],
                    "f1": test_metrics["f1_weighted"]
                })
            
            size_results["mean_accuracy"] = np.mean(accuracies)
            size_results["std_accuracy"] = np.std(accuracies)
            size_results["mean_f1"] = np.mean(f1_scores)
            
            self.results["data_size"][f"{int(data_fraction*100)}%"] = size_results
            
            logger.info(
                f"Data fraction: {data_fraction} | "
                f"Accuracy: {size_results['mean_accuracy']:.4f} ± {size_results['std_accuracy']:.4f}"
            )
        
        return self.results["data_size"]
    
    def run_augmentation_ablation(self) -> Dict[str, Any]:
        """
        Test different data augmentation strategies.
        
        Returns:
            Augmentation ablation results
        """
        logger.info("Running augmentation ablation")
        
        dataset = AGNewsDataset()
        base_data = dataset.load_splits()
        
        # Use a smaller subset for augmentation experiments
        base_data = self._sample_data(base_data, 0.25)
        
        for aug_method in self.augmentation_methods:
            logger.info(f"\nTesting augmentation: {aug_method}")
            
            aug_results = {
                "method": aug_method,
                "trials": [],
                "mean_accuracy": 0,
                "std_accuracy": 0,
                "mean_f1": 0,
                "augmentation_ratio": 0
            }
            
            accuracies = []
            f1_scores = []
            
            for trial in range(self.num_trials):
                logger.info(f"Trial {trial + 1}/{self.num_trials}")
                
                # Apply augmentation
                augmented_data = self._apply_augmentation(base_data, aug_method)
                aug_results["augmentation_ratio"] = (
                    len(augmented_data["train"]["texts"]) / 
                    len(base_data["train"]["texts"])
                )
                
                # Train model
                model = self.factory.create_model(self.model_name, **self.model_config)
                trainer = BaseTrainer(model=model, config=self.model_config, device=self.device)
                
                trainer.train(
                    augmented_data["train"]["texts"],
                    augmented_data["train"]["labels"],
                    augmented_data["val"]["texts"],
                    augmented_data["val"]["labels"]
                )
                
                test_metrics = trainer.evaluate(
                    base_data["test"]["texts"],
                    base_data["test"]["labels"]
                )
                
                accuracies.append(test_metrics["accuracy"])
                f1_scores.append(test_metrics["f1_weighted"])
                
                aug_results["trials"].append({
                    "accuracy": test_metrics["accuracy"],
                    "f1": test_metrics["f1_weighted"]
                })
            
            aug_results["mean_accuracy"] = np.mean(accuracies)
            aug_results["std_accuracy"] = np.std(accuracies)
            aug_results["mean_f1"] = np.mean(f1_scores)
            
            self.results["augmentation"][aug_method] = aug_results
            
            logger.info(
                f"Augmentation: {aug_method} | "
                f"Accuracy: {aug_results['mean_accuracy']:.4f} ± {aug_results['std_accuracy']:.4f}"
            )
        
        return self.results["augmentation"]
    
    def run_quality_ablation(self) -> Dict[str, Any]:
        """
        Test impact of data quality filtering.
        
        Returns:
            Quality ablation results
        """
        logger.info("Running data quality ablation")
        
        dataset = AGNewsDataset()
        full_data = dataset.load_splits()
        
        quality_filters = [
            {"name": "none", "filter_fn": lambda x: True},
            {"name": "length_filter", "filter_fn": lambda x: 10 <= len(x.split()) <= 100},
            {"name": "no_special_chars", "filter_fn": lambda x: x.isascii()},
            {"name": "confidence_filter", "filter_fn": self._confidence_filter},
            {"name": "all_filters", "filter_fn": self._all_filters}
        ]
        
        for filter_config in quality_filters:
            logger.info(f"\nTesting quality filter: {filter_config['name']}")
            
            quality_results = {
                "filter": filter_config["name"],
                "samples_retained": 0,
                "retention_rate": 0,
                "mean_accuracy": 0,
                "std_accuracy": 0
            }
            
            # Apply filter
            filtered_data = self._apply_quality_filter(
                full_data,
                filter_config["filter_fn"]
            )
            
            quality_results["samples_retained"] = len(filtered_data["train"]["texts"])
            quality_results["retention_rate"] = (
                len(filtered_data["train"]["texts"]) / 
                len(full_data["train"]["texts"])
            )
            
            # Train and evaluate
            accuracies = []
            
            for trial in range(min(self.num_trials, 2)):  # Fewer trials for efficiency
                model = self.factory.create_model(self.model_name, **self.model_config)
                trainer = BaseTrainer(model=model, config=self.model_config, device=self.device)
                
                trainer.train(
                    filtered_data["train"]["texts"][:5000],
                    filtered_data["train"]["labels"][:5000],
                    filtered_data["val"]["texts"][:1000],
                    filtered_data["val"]["labels"][:1000]
                )
                
                test_metrics = trainer.evaluate(
                    full_data["test"]["texts"][:1000],
                    full_data["test"]["labels"][:1000]
                )
                
                accuracies.append(test_metrics["accuracy"])
            
            quality_results["mean_accuracy"] = np.mean(accuracies)
            quality_results["std_accuracy"] = np.std(accuracies)
            
            self.results["quality"][filter_config["name"]] = quality_results
            
            logger.info(
                f"Filter: {filter_config['name']} | "
                f"Retention: {quality_results['retention_rate']:.2%} | "
                f"Accuracy: {quality_results['mean_accuracy']:.4f}"
            )
        
        return self.results["quality"]
    
    def run_class_balance_ablation(self) -> Dict[str, Any]:
        """
        Test impact of class imbalance.
        
        Returns:
            Class balance ablation results
        """
        logger.info("Running class balance ablation")
        
        dataset = AGNewsDataset()
        full_data = dataset.load_splits()
        
        imbalance_ratios = [1.0, 0.5, 0.25, 0.1]  # Minority class ratios
        
        for ratio in imbalance_ratios:
            logger.info(f"\nTesting imbalance ratio: {ratio}")
            
            balance_results = {
                "ratio": ratio,
                "class_distribution": {},
                "mean_accuracy": 0,
                "mean_f1": 0,
                "per_class_f1": {}
            }
            
            # Create imbalanced dataset
            imbalanced_data = self._create_imbalanced_data(full_data, ratio)
            
            # Calculate class distribution
            unique, counts = np.unique(imbalanced_data["train"]["labels"], return_counts=True)
            balance_results["class_distribution"] = dict(zip(unique.tolist(), counts.tolist()))
            
            # Train and evaluate
            model = self.factory.create_model(self.model_name, **self.model_config)
            trainer = BaseTrainer(model=model, config=self.model_config, device=self.device)
            
            trainer.train(
                imbalanced_data["train"]["texts"],
                imbalanced_data["train"]["labels"],
                imbalanced_data["val"]["texts"],
                imbalanced_data["val"]["labels"]
            )
            
            test_metrics = trainer.evaluate(
                full_data["test"]["texts"],
                full_data["test"]["labels"]
            )
            
            balance_results["mean_accuracy"] = test_metrics["accuracy"]
            balance_results["mean_f1"] = test_metrics["f1_weighted"]
            
            self.results["class_balance"][f"ratio_{ratio}"] = balance_results
            
            logger.info(
                f"Imbalance ratio: {ratio} | "
                f"Accuracy: {balance_results['mean_accuracy']:.4f} | "
                f"F1: {balance_results['mean_f1']:.4f}"
            )
        
        return self.results["class_balance"]
    
    def _sample_data(
        self,
        data: Dict[str, Any],
        fraction: float
    ) -> Dict[str, Any]:
        """Sample a fraction of the data."""
        if fraction >= 1.0:
            return data
        
        sampled = {}
        
        for split in ["train", "val"]:
            n_samples = int(len(data[split]["texts"]) * fraction)
            indices = np.random.choice(
                len(data[split]["texts"]),
                size=n_samples,
                replace=False
            )
            
            sampled[split] = {
                "texts": [data[split]["texts"][i] for i in indices],
                "labels": data[split]["labels"][indices] if isinstance(data[split]["labels"], np.ndarray)
                         else [data[split]["labels"][i] for i in indices]
            }
        
        sampled["test"] = data["test"]
        
        return sampled
    
    def _apply_augmentation(
        self,
        data: Dict[str, Any],
        method: str
    ) -> Dict[str, Any]:
        """Apply augmentation to training data."""
        if method == "none":
            return data
        
        augmented = data.copy()
        augmented_texts = []
        augmented_labels = []
        
        # Simple augmentation implementations
        for text, label in zip(data["train"]["texts"], data["train"]["labels"]):
            augmented_texts.append(text)
            augmented_labels.append(label)
            
            if method == "synonym_replacement":
                # Simplified synonym replacement
                words = text.split()
                if len(words) > 3:
                    idx = np.random.randint(0, len(words))
                    words[idx] = f"SYN_{words[idx]}"
                    augmented_texts.append(" ".join(words))
                    augmented_labels.append(label)
            
            elif method == "random_insertion":
                words = text.split()
                idx = np.random.randint(0, len(words) + 1)
                words.insert(idx, "INSERTED")
                augmented_texts.append(" ".join(words))
                augmented_labels.append(label)
            
            elif method == "random_swap":
                words = text.split()
                if len(words) > 1:
                    idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                    words[idx1], words[idx2] = words[idx2], words[idx1]
                    augmented_texts.append(" ".join(words))
                    augmented_labels.append(label)
            
            elif method == "random_deletion":
                words = text.split()
                if len(words) > 2:
                    idx = np.random.randint(0, len(words))
                    del words[idx]
                    augmented_texts.append(" ".join(words))
                    augmented_labels.append(label)
        
        augmented["train"]["texts"] = augmented_texts
        augmented["train"]["labels"] = augmented_labels
        
        return augmented
    
    def _confidence_filter(self, text: str) -> bool:
        """Filter based on pseudo-confidence."""
        # Simplified: filter very short or very long texts
        word_count = len(text.split())
        return 5 <= word_count <= 200
    
    def _all_filters(self, text: str) -> bool:
        """Apply all quality filters."""
        word_count = len(text.split())
        return (5 <= word_count <= 200 and 
                text.isascii() and 
                not text.isupper())
    
    def _apply_quality_filter(
        self,
        data: Dict[str, Any],
        filter_fn
    ) -> Dict[str, Any]:
        """Apply quality filter to data."""
        filtered = {}
        
        for split in ["train", "val", "test"]:
            filtered_texts = []
            filtered_labels = []
            
            for text, label in zip(data[split]["texts"], data[split]["labels"]):
                if filter_fn(text):
                    filtered_texts.append(text)
                    filtered_labels.append(label)
            
            filtered[split] = {
                "texts": filtered_texts,
                "labels": filtered_labels
            }
        
        return filtered
    
    def _create_imbalanced_data(
        self,
        data: Dict[str, Any],
        minority_ratio: float
    ) -> Dict[str, Any]:
        """Create imbalanced dataset."""
        imbalanced = {}
        
        for split in ["train", "val"]:
            texts = []
            labels = []
            
            # Get unique classes
            unique_labels = np.unique(data[split]["labels"])
            
            for label_idx, label in enumerate(unique_labels):
                # Get samples for this class
                class_texts = [
                    text for text, l in zip(data[split]["texts"], data[split]["labels"])
                    if l == label
                ]
                
                # Determine number of samples
                if label_idx == 0:
                    # Keep all samples for first class
                    n_samples = len(class_texts)
                else:
                    # Reduce samples for other classes
                    n_samples = int(len(class_texts) * minority_ratio)
                
                # Sample
                sampled_texts = np.random.choice(class_texts, size=n_samples, replace=False)
                texts.extend(sampled_texts)
                labels.extend([label] * n_samples)
            
            imbalanced[split] = {
                "texts": texts,
                "labels": labels
            }
        
        imbalanced["test"] = data["test"]
        
        return imbalanced
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive report."""
        report = {
            "data_size_impact": self._analyze_data_size_impact(),
            "augmentation_effectiveness": self._analyze_augmentation_effectiveness(),
            "quality_importance": self._analyze_quality_importance(),
            "class_balance_sensitivity": self._analyze_class_balance_sensitivity(),
            "recommendations": self._generate_recommendations()
        }
        
        return report
    
    def _analyze_data_size_impact(self) -> Dict[str, Any]:
        """Analyze impact of data size."""
        if not self.results["data_size"]:
            return {}
        
        sizes = []
        accuracies = []
        
        for key, result in self.results["data_size"].items():
            sizes.append(result["num_samples"])
            accuracies.append(result["mean_accuracy"])
        
        # Fit logarithmic curve
        if len(sizes) > 1:
            coeffs = np.polyfit(np.log(sizes), accuracies, 1)
            
            return {
                "scaling_coefficient": float(coeffs[0]),
                "diminishing_returns_point": sizes[-2] if len(sizes) > 1 else 0,
                "data_efficiency": accuracies[0] / sizes[0] if sizes[0] > 0 else 0
            }
        
        return {}
    
    def _analyze_augmentation_effectiveness(self) -> Dict[str, Any]:
        """Analyze augmentation effectiveness."""
        if not self.results["augmentation"]:
            return {}
        
        baseline = self.results["augmentation"].get("none", {}).get("mean_accuracy", 0)
        
        improvements = {}
        for method, result in self.results["augmentation"].items():
            if method != "none":
                improvements[method] = result["mean_accuracy"] - baseline
        
        best_method = max(improvements.items(), key=lambda x: x[1]) if improvements else None
        
        return {
            "baseline_accuracy": baseline,
            "best_augmentation": best_method[0] if best_method else None,
            "best_improvement": best_method[1] if best_method else 0,
            "all_improvements": improvements
        }
    
    def _analyze_quality_importance(self) -> Dict[str, Any]:
        """Analyze importance of data quality."""
        if not self.results["quality"]:
            return {}
        
        baseline = self.results["quality"].get("none", {}).get("mean_accuracy", 0)
        
        quality_impact = {}
        for filter_name, result in self.results["quality"].items():
            if filter_name != "none":
                quality_impact[filter_name] = {
                    "accuracy_change": result["mean_accuracy"] - baseline,
                    "data_retention": result["retention_rate"]
                }
        
        return quality_impact
    
    def _analyze_class_balance_sensitivity(self) -> Dict[str, Any]:
        """Analyze sensitivity to class imbalance."""
        if not self.results["class_balance"]:
            return {}
        
        ratios = []
        accuracies = []
        f1_scores = []
        
        for key, result in self.results["class_balance"].items():
            ratios.append(result["ratio"])
            accuracies.append(result["mean_accuracy"])
            f1_scores.append(result["mean_f1"])
        
        return {
            "accuracy_drop_per_imbalance": (accuracies[0] - accuracies[-1]) / (ratios[0] - ratios[-1])
                                           if len(ratios) > 1 else 0,
            "f1_sensitivity": (f1_scores[0] - f1_scores[-1]) / (ratios[0] - ratios[-1])
                              if len(ratios) > 1 else 0,
            "robust_to_imbalance": accuracies[-1] > accuracies[0] * 0.9 if accuracies else False
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on ablation results."""
        recommendations = []
        
        # Data size recommendations
        if self.results["data_size"]:
            sizes = [r["num_samples"] for r in self.results["data_size"].values()]
            accs = [r["mean_accuracy"] for r in self.results["data_size"].values()]
            
            if len(sizes) > 1 and (accs[-1] - accs[-2]) < 0.01:
                recommendations.append(
                    f"Consider using {sizes[-2]} samples instead of {sizes[-1]} "
                    f"(only {(accs[-1] - accs[-2])*100:.1f}% improvement)"
                )
        
        # Augmentation recommendations
        if self.results["augmentation"]:
            best = max(self.results["augmentation"].items(), 
                      key=lambda x: x[1]["mean_accuracy"])
            if best[0] != "none":
                recommendations.append(f"Use {best[0]} augmentation for best results")
        
        # Quality recommendations
        if self.results["quality"]:
            for filter_name, result in self.results["quality"].items():
                if result["retention_rate"] > 0.8 and result["mean_accuracy"] > 0:
                    recommendations.append(f"Consider applying {filter_name} quality filter")
        
        # Class balance recommendations
        if self.results["class_balance"]:
            ratios = [r["ratio"] for r in self.results["class_balance"].values()]
            if min(ratios) < 0.5:
                recommendations.append("Model is sensitive to class imbalance - use class weights")
        
        return recommendations


def run_data_ablation():
    """Run data ablation study."""
    logger.info("Starting data ablation study")
    
    ablation = DataAblation(
        model_name="bert-base",
        data_sizes=[0.1, 0.25, 0.5, 1.0],
        num_trials=2
    )
    
    # Run different ablation studies
    ablation.run_data_size_ablation()
    ablation.run_augmentation_ablation()
    ablation.run_quality_ablation()
    ablation.run_class_balance_ablation()
    
    # Generate report
    report = ablation.generate_report()
    
    logger.info(f"Report: {report}")
    
    return ablation.results


if __name__ == "__main__":
    run_data_ablation()
