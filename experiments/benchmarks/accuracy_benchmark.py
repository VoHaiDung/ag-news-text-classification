"""
Accuracy Benchmark for AG News Text Classification
================================================================================
This module implements comprehensive accuracy benchmarking across different
models, configurations, and datasets following rigorous evaluation protocols.

The benchmark measures classification performance using multiple metrics and
statistical tests to ensure reliable comparisons.

References:
    - Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets
    - Raschka, S. (2018). Model Evaluation, Model Selection, and Algorithm Selection

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    matthews_corrcoef
)
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

from src.core.registry import Registry
from src.core.factory import Factory
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.evaluation.metrics.classification_metrics import ClassificationMetrics
from experiments.experiment_runner import ExperimentRunner, ExperimentConfig

logger = logging.getLogger(__name__)


class AccuracyBenchmark:
    """
    Comprehensive accuracy benchmarking for text classification models.
    
    This class provides:
    - Multi-metric evaluation
    - Statistical significance testing
    - Cross-validation assessment
    - Per-class performance analysis
    - Robustness evaluation
    """
    
    def __init__(
        self,
        models: List[str],
        dataset_name: str = "ag_news",
        num_runs: int = 5,
        seed: int = 42
    ):
        """
        Initialize accuracy benchmark.
        
        Args:
            models: List of model names to benchmark
            dataset_name: Name of dataset to use
            num_runs: Number of repeated runs for statistical validity
            seed: Random seed for reproducibility
        """
        self.models = models
        self.dataset_name = dataset_name
        self.num_runs = num_runs
        self.seed = seed
        
        self.registry = Registry()
        self.factory = Factory()
        self.metrics_calculator = ClassificationMetrics()
        self.results = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run complete accuracy benchmark.
        
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Starting accuracy benchmark")
        logger.info(f"Models: {self.models}")
        logger.info(f"Dataset: {self.dataset_name}")
        logger.info(f"Number of runs: {self.num_runs}")
        
        # Load dataset
        dataset = self._load_dataset()
        
        # Run benchmarks for each model
        for model_name in self.models:
            logger.info(f"\nBenchmarking model: {model_name}")
            self.results[model_name] = self._benchmark_model(
                model_name,
                dataset
            )
        
        # Perform statistical analysis
        statistical_results = self._statistical_analysis()
        
        # Generate ranking
        ranking = self._generate_ranking()
        
        # Create summary report
        summary = self._create_summary()
        
        return {
            "model_results": self.results,
            "statistical_analysis": statistical_results,
            "ranking": ranking,
            "summary": summary
        }
    
    def _benchmark_model(
        self,
        model_name: str,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Benchmark a single model.
        
        Args:
            model_name: Name of model to benchmark
            dataset: Dataset dictionary
            
        Returns:
            Benchmark results for the model
        """
        all_predictions = []
        all_labels = []
        all_scores = []
        run_metrics = []
        
        for run_idx in range(self.num_runs):
            # Set seed for reproducibility
            run_seed = self.seed + run_idx
            set_seed(run_seed)
            
            logger.info(f"Run {run_idx + 1}/{self.num_runs}")
            
            # Load or train model
            model = self._get_model(model_name, run_seed)
            
            # Make predictions
            predictions, scores = self._predict(model, dataset["test"])
            labels = dataset["test"]["labels"]
            
            # Calculate metrics
            metrics = self._calculate_metrics(predictions, labels, scores)
            
            # Store results
            all_predictions.append(predictions)
            all_labels.append(labels)
            all_scores.append(scores)
            run_metrics.append(metrics)
        
        # Aggregate results
        aggregated = self._aggregate_results(run_metrics)
        
        # Per-class analysis
        per_class = self._per_class_analysis(all_predictions, all_labels)
        
        # Confusion matrix
        confusion = self._compute_confusion_matrix(all_predictions, all_labels)
        
        return {
            "metrics": aggregated,
            "per_run": run_metrics,
            "per_class": per_class,
            "confusion_matrix": confusion,
            "predictions": all_predictions,
            "scores": all_scores
        }
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        scores: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.
        
        Args:
            predictions: Predicted labels
            labels: True labels
            scores: Prediction scores (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            # Basic metrics
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro"),
            "f1_micro": f1_score(labels, predictions, average="micro"),
            "f1_weighted": f1_score(labels, predictions, average="weighted"),
            "precision_macro": precision_score(labels, predictions, average="macro"),
            "precision_micro": precision_score(labels, predictions, average="micro"),
            "precision_weighted": precision_score(labels, predictions, average="weighted"),
            "recall_macro": recall_score(labels, predictions, average="macro"),
            "recall_micro": recall_score(labels, predictions, average="micro"),
            "recall_weighted": recall_score(labels, predictions, average="weighted"),
            
            # Agreement metrics
            "cohen_kappa": cohen_kappa_score(labels, predictions),
            "matthews_corrcoef": matthews_corrcoef(labels, predictions)
        }
        
        # Add advanced metrics if scores available
        if scores is not None:
            advanced_metrics = self.metrics_calculator.calculate_advanced_metrics(
                labels, predictions, scores
            )
            metrics.update(advanced_metrics)
        
        return metrics
    
    def _aggregate_results(
        self,
        run_metrics: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across runs.
        
        Args:
            run_metrics: List of metrics from each run
            
        Returns:
            Aggregated statistics
        """
        df = pd.DataFrame(run_metrics)
        
        aggregated = {
            "mean": df.mean().to_dict(),
            "std": df.std().to_dict(),
            "min": df.min().to_dict(),
            "max": df.max().to_dict(),
            "median": df.median().to_dict()
        }
        
        # Calculate confidence intervals
        aggregated["confidence_intervals"] = {}
        for metric in df.columns:
            values = df[metric].values
            ci = self._confidence_interval(values)
            aggregated["confidence_intervals"][metric] = ci
        
        return aggregated
    
    def _per_class_analysis(
        self,
        all_predictions: List[np.ndarray],
        all_labels: List[np.ndarray]
    ) -> Dict[str, Any]:
        """
        Perform per-class performance analysis.
        
        Args:
            all_predictions: List of predictions from all runs
            all_labels: List of labels from all runs
            
        Returns:
            Per-class metrics
        """
        # Flatten predictions and labels
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        
        # Get unique classes
        classes = np.unique(labels)
        
        per_class_metrics = {}
        for class_id in classes:
            # Create binary masks
            class_mask = labels == class_id
            class_predictions = predictions[class_mask]
            class_labels = labels[class_mask]
            
            # Calculate class-specific metrics
            per_class_metrics[f"class_{class_id}"] = {
                "support": np.sum(class_mask),
                "accuracy": accuracy_score(
                    class_labels == class_id,
                    class_predictions == class_id
                ),
                "precision": precision_score(
                    labels, predictions,
                    labels=[class_id], average="micro"
                ),
                "recall": recall_score(
                    labels, predictions,
                    labels=[class_id], average="micro"
                ),
                "f1": f1_score(
                    labels, predictions,
                    labels=[class_id], average="micro"
                )
            }
        
        return per_class_metrics
    
    def _statistical_analysis(self) -> Dict[str, Any]:
        """
        Perform statistical analysis comparing models.
        
        Returns:
            Statistical test results
        """
        if len(self.models) < 2:
            return {"message": "Need at least 2 models for comparison"}
        
        # Prepare data for statistical tests
        model_metrics = {}
        for model_name in self.models:
            if model_name in self.results:
                model_metrics[model_name] = [
                    run["accuracy"] for run in self.results[model_name]["per_run"]
                ]
        
        # Friedman test for multiple models
        if len(self.models) > 2:
            friedman_result = self._friedman_test(model_metrics)
        else:
            friedman_result = None
        
        # Pairwise comparisons
        pairwise_results = self._pairwise_comparisons(model_metrics)
        
        # Effect size calculations
        effect_sizes = self._calculate_effect_sizes(model_metrics)
        
        return {
            "friedman_test": friedman_result,
            "pairwise_comparisons": pairwise_results,
            "effect_sizes": effect_sizes
        }
    
    def _friedman_test(
        self,
        model_metrics: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Perform Friedman test for multiple model comparison.
        
        Args:
            model_metrics: Dictionary of model metrics
            
        Returns:
            Friedman test results
        """
        # Prepare data matrix
        data = []
        for model_name in self.models:
            if model_name in model_metrics:
                data.append(model_metrics[model_name])
        
        data = np.array(data).T
        
        # Perform Friedman test
        statistic, p_value = stats.friedmanchisquare(*data.T)
        
        return {
            "statistic": statistic,
            "p_value": p_value,
            "significant": p_value < 0.05,
            "interpretation": "Models differ significantly" if p_value < 0.05 
                            else "No significant difference between models"
        }
    
    def _pairwise_comparisons(
        self,
        model_metrics: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Perform pairwise statistical comparisons.
        
        Args:
            model_metrics: Dictionary of model metrics
            
        Returns:
            Pairwise comparison results
        """
        comparisons = {}
        model_names = list(model_metrics.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                # Wilcoxon signed-rank test
                statistic, p_value = stats.wilcoxon(
                    model_metrics[model1],
                    model_metrics[model2]
                )
                
                # Apply Bonferroni correction
                n_comparisons = len(model_names) * (len(model_names) - 1) / 2
                corrected_p_value = p_value * n_comparisons
                
                comparisons[f"{model1}_vs_{model2}"] = {
                    "statistic": statistic,
                    "p_value": p_value,
                    "corrected_p_value": min(corrected_p_value, 1.0),
                    "significant": corrected_p_value < 0.05,
                    "better_model": model1 if np.mean(model_metrics[model1]) > 
                                   np.mean(model_metrics[model2]) else model2
                }
        
        return comparisons
    
    def _calculate_effect_sizes(
        self,
        model_metrics: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Calculate effect sizes for model comparisons.
        
        Args:
            model_metrics: Dictionary of model metrics
            
        Returns:
            Effect size calculations
        """
        effect_sizes = {}
        model_names = list(model_metrics.keys())
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                # Cohen's d
                mean1 = np.mean(model_metrics[model1])
                mean2 = np.mean(model_metrics[model2])
                std1 = np.std(model_metrics[model1])
                std2 = np.std(model_metrics[model2])
                
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                cohens_d = (mean1 - mean2) / pooled_std
                
                # Interpret effect size
                if abs(cohens_d) < 0.2:
                    interpretation = "negligible"
                elif abs(cohens_d) < 0.5:
                    interpretation = "small"
                elif abs(cohens_d) < 0.8:
                    interpretation = "medium"
                else:
                    interpretation = "large"
                
                effect_sizes[f"{model1}_vs_{model2}"] = {
                    "cohens_d": cohens_d,
                    "interpretation": interpretation
                }
        
        return effect_sizes
    
    def _generate_ranking(self) -> List[Dict[str, Any]]:
        """
        Generate model ranking based on performance.
        
        Returns:
            Ranked list of models
        """
        ranking = []
        
        for model_name in self.models:
            if model_name in self.results:
                mean_accuracy = self.results[model_name]["metrics"]["mean"]["accuracy"]
                std_accuracy = self.results[model_name]["metrics"]["std"]["accuracy"]
                
                ranking.append({
                    "rank": 0,  # Will be assigned after sorting
                    "model": model_name,
                    "mean_accuracy": mean_accuracy,
                    "std_accuracy": std_accuracy,
                    "score": mean_accuracy - std_accuracy  # Conservative score
                })
        
        # Sort by score
        ranking = sorted(ranking, key=lambda x: x["score"], reverse=True)
        
        # Assign ranks
        for i, item in enumerate(ranking):
            item["rank"] = i + 1
        
        return ranking
    
    def _confidence_interval(
        self,
        values: np.ndarray,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Calculate confidence interval.
        
        Args:
            values: Array of values
            confidence: Confidence level
            
        Returns:
            Lower and upper bounds
        """
        n = len(values)
        mean = np.mean(values)
        std_err = stats.sem(values)
        
        interval = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        
        return (mean - interval, mean + interval)
    
    def _compute_confusion_matrix(
        self,
        all_predictions: List[np.ndarray],
        all_labels: List[np.ndarray]
    ) -> np.ndarray:
        """
        Compute aggregated confusion matrix.
        
        Args:
            all_predictions: List of predictions
            all_labels: List of labels
            
        Returns:
            Confusion matrix
        """
        # Flatten predictions and labels
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels)
        
        return confusion_matrix(labels, predictions)
    
    def _create_summary(self) -> Dict[str, Any]:
        """
        Create benchmark summary.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "dataset": self.dataset_name,
            "num_models": len(self.models),
            "num_runs": self.num_runs,
            "best_model": None,
            "best_accuracy": 0.0,
            "average_accuracy": 0.0
        }
        
        # Find best model
        accuracies = []
        for model_name in self.models:
            if model_name in self.results:
                mean_acc = self.results[model_name]["metrics"]["mean"]["accuracy"]
                accuracies.append(mean_acc)
                
                if mean_acc > summary["best_accuracy"]:
                    summary["best_model"] = model_name
                    summary["best_accuracy"] = mean_acc
        
        if accuracies:
            summary["average_accuracy"] = np.mean(accuracies)
        
        return summary
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset for benchmarking."""
        dataset = AGNewsDataset()
        return dataset.load_splits()
    
    def _get_model(self, model_name: str, seed: int):
        """Load or create model for benchmarking."""
        return self.factory.create_model(model_name, seed=seed)
    
    def _predict(self, model, data):
        """Make predictions with model."""
        # Implementation depends on model interface
        predictions = model.predict(data["texts"])
        scores = model.predict_proba(data["texts"])
        return predictions, scores
    
    def visualize_results(self, save_path: Optional[str] = None):
        """
        Visualize benchmark results.
        
        Args:
            save_path: Path to save visualization
        """
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Accuracy comparison
        self._plot_accuracy_comparison(axes[0, 0])
        
        # Plot 2: Per-class performance
        self._plot_per_class_performance(axes[0, 1])
        
        # Plot 3: Statistical significance
        self._plot_statistical_significance(axes[1, 0])
        
        # Plot 4: Confusion matrices
        self._plot_confusion_matrices(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def _plot_accuracy_comparison(self, ax):
        """Plot accuracy comparison across models."""
        model_names = []
        accuracies = []
        errors = []
        
        for model_name in self.models:
            if model_name in self.results:
                model_names.append(model_name)
                mean_acc = self.results[model_name]["metrics"]["mean"]["accuracy"]
                std_acc = self.results[model_name]["metrics"]["std"]["accuracy"]
                accuracies.append(mean_acc)
                errors.append(std_acc)
        
        ax.bar(model_names, accuracies, yerr=errors, capsize=5)
        ax.set_xlabel("Model")
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Accuracy Comparison")
        ax.set_ylim([0, 1])
        
        # Add value labels
        for i, (acc, err) in enumerate(zip(accuracies, errors)):
            ax.text(i, acc + err + 0.01, f"{acc:.3f}±{err:.3f}",
                   ha="center", va="bottom")
    
    def _plot_per_class_performance(self, ax):
        """Plot per-class performance heatmap."""
        # Prepare data for heatmap
        data = []
        model_names = []
        
        for model_name in self.models[:5]:  # Limit to 5 models for clarity
            if model_name in self.results:
                model_names.append(model_name)
                per_class = self.results[model_name]["per_class"]
                
                class_f1_scores = [
                    per_class[f"class_{i}"]["f1"]
                    for i in range(len(per_class))
                ]
                data.append(class_f1_scores)
        
        if data:
            sns.heatmap(data, annot=True, fmt=".3f", cmap="YlOrRd",
                       xticklabels=[f"Class {i}" for i in range(len(data[0]))],
                       yticklabels=model_names, ax=ax)
            ax.set_title("Per-Class F1 Scores")
    
    def _plot_statistical_significance(self, ax):
        """Plot statistical significance matrix."""
        # This would show pairwise comparison results
        ax.text(0.5, 0.5, "Statistical Significance Matrix",
               ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Statistical Comparisons")
    
    def _plot_confusion_matrices(self, ax):
        """Plot confusion matrices."""
        # This would show confusion matrices for best model
        ax.text(0.5, 0.5, "Confusion Matrix",
               ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Best Model Confusion Matrix")
    
    def save_results(self, filepath: str):
        """
        Save benchmark results to file.
        
        Args:
            filepath: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
