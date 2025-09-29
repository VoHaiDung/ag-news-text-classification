"""
Result Aggregator for AG News Text Classification
================================================================================
This module provides comprehensive result aggregation functionality for combining
and analyzing results from multiple experiments, models, and evaluation metrics.

The aggregator implements statistical analysis, visualization, and report generation
for experiment results following best practices in machine learning evaluation.

References:
    - Demšar, J. (2006). Statistical Comparisons of Classifiers over Multiple Data Sets
    - Bergstra, J., & Bengio, Y. (2012). Random Search for Hyper-Parameter Optimization

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.utils.io_utils import save_json, load_json, save_pickle, load_pickle
from experiments.results.experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class ResultAggregator:
    """
    Aggregates and analyzes results from multiple experiments.
    
    This class provides:
    - Multi-experiment result aggregation
    - Statistical analysis and hypothesis testing
    - Performance trend analysis
    - Best configuration identification
    - Comprehensive report generation
    """
    
    def __init__(
        self,
        results_dir: str = "outputs/results",
        experiments_file: Optional[str] = None,
        confidence_level: float = 0.95
    ):
        """
        Initialize result aggregator.
        
        Args:
            results_dir: Directory containing experiment results
            experiments_file: Optional file listing experiment IDs
            confidence_level: Confidence level for statistical tests
        """
        self.results_dir = Path(results_dir)
        self.experiments_file = experiments_file
        self.confidence_level = confidence_level
        
        self.experiments = {}
        self.aggregated_results = {}
        self.statistical_results = {}
        
        # Load experiments if file provided
        if experiments_file:
            self._load_experiments_list()
        
        logger.info(f"Initialized ResultAggregator with results from {results_dir}")
    
    def load_experiments(
        self,
        experiment_ids: Optional[List[str]] = None,
        pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load experiment results.
        
        Args:
            experiment_ids: Specific experiment IDs to load
            pattern: Pattern to match experiment directories
            
        Returns:
            Dictionary of loaded experiments
        """
        if experiment_ids:
            # Load specific experiments
            for exp_id in experiment_ids:
                exp_path = self.results_dir / exp_id
                if exp_path.exists():
                    self.experiments[exp_id] = self._load_experiment(exp_path)
                else:
                    logger.warning(f"Experiment not found: {exp_id}")
        
        elif pattern:
            # Load experiments matching pattern
            for exp_path in self.results_dir.glob(pattern):
                if exp_path.is_dir():
                    exp_id = exp_path.name
                    self.experiments[exp_id] = self._load_experiment(exp_path)
        
        else:
            # Load all experiments
            for exp_path in self.results_dir.iterdir():
                if exp_path.is_dir():
                    exp_id = exp_path.name
                    self.experiments[exp_id] = self._load_experiment(exp_path)
        
        logger.info(f"Loaded {len(self.experiments)} experiments")
        return self.experiments
    
    def aggregate_metrics(
        self,
        metrics: List[str],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate metrics across experiments.
        
        Args:
            metrics: List of metrics to aggregate
            group_by: Optional grouping variable
            
        Returns:
            DataFrame with aggregated metrics
        """
        data = []
        
        for exp_id, exp_data in self.experiments.items():
            row = {"experiment_id": exp_id}
            
            # Extract metrics
            for metric in metrics:
                value = self._extract_metric(exp_data, metric)
                row[metric] = value
            
            # Add grouping variable if specified
            if group_by:
                row[group_by] = exp_data.get("config", {}).get(group_by)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate summary statistics
        if group_by and group_by in df.columns:
            # Group-wise aggregation
            aggregated = df.groupby(group_by)[metrics].agg([
                'mean', 'std', 'min', 'max', 'median'
            ])
        else:
            # Overall aggregation
            aggregated = df[metrics].agg([
                'mean', 'std', 'min', 'max', 'median'
            ])
        
        self.aggregated_results = {
            "raw_data": df,
            "summary": aggregated
        }
        
        return aggregated
    
    def find_best_configuration(
        self,
        metric: str,
        constraints: Optional[Dict[str, Any]] = None,
        maximize: bool = True
    ) -> Dict[str, Any]:
        """
        Find best configuration based on metric.
        
        Args:
            metric: Metric to optimize
            constraints: Optional constraints on configurations
            maximize: Whether to maximize or minimize metric
            
        Returns:
            Best configuration and its performance
        """
        best_config = None
        best_value = -np.inf if maximize else np.inf
        best_experiment = None
        
        for exp_id, exp_data in self.experiments.items():
            # Check constraints
            if constraints:
                if not self._check_constraints(exp_data, constraints):
                    continue
            
            # Get metric value
            value = self._extract_metric(exp_data, metric)
            
            if value is None:
                continue
            
            # Update best if better
            if (maximize and value > best_value) or (not maximize and value < best_value):
                best_value = value
                best_config = exp_data.get("config", {})
                best_experiment = exp_id
        
        return {
            "experiment_id": best_experiment,
            "configuration": best_config,
            "metric_value": best_value,
            "metric_name": metric,
            "optimization": "maximize" if maximize else "minimize"
        }
    
    def statistical_comparison(
        self,
        metrics: List[str],
        models: Optional[List[str]] = None,
        test_type: str = "friedman"
    ) -> Dict[str, Any]:
        """
        Perform statistical comparison between models.
        
        Args:
            metrics: Metrics to compare
            models: Optional list of models to compare
            test_type: Type of statistical test
            
        Returns:
            Statistical test results
        """
        results = {}
        
        for metric in metrics:
            # Prepare data for comparison
            data = self._prepare_comparison_data(metric, models)
            
            if len(data) < 2:
                logger.warning(f"Not enough data for statistical comparison of {metric}")
                continue
            
            # Perform statistical test
            if test_type == "friedman":
                stat_result = self._friedman_test(data, metric)
            elif test_type == "nemenyi":
                stat_result = self._nemenyi_test(data, metric)
            elif test_type == "wilcoxon":
                stat_result = self._wilcoxon_test(data, metric)
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            results[metric] = stat_result
        
        self.statistical_results = results
        return results
    
    def _friedman_test(
        self,
        data: Dict[str, List[float]],
        metric: str
    ) -> Dict[str, Any]:
        """
        Perform Friedman test for multiple model comparison.
        
        Args:
            data: Model performance data
            metric: Metric name
            
        Returns:
            Test results
        """
        # Prepare data matrix
        models = list(data.keys())
        values = [data[model] for model in models]
        
        # Ensure equal length
        min_length = min(len(v) for v in values)
        values = [v[:min_length] for v in values]
        
        # Perform Friedman test
        statistic, p_value = friedmanchisquare(*values)
        
        # Calculate average ranks
        ranks = np.array([rankdata(-np.array(v)) for v in values]).T
        avg_ranks = np.mean(ranks, axis=0)
        
        # Post-hoc analysis if significant
        post_hoc = None
        if p_value < (1 - self.confidence_level):
            post_hoc = self._friedman_posthoc(data, models)
        
        return {
            "test": "Friedman",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < (1 - self.confidence_level),
            "models": models,
            "average_ranks": {m: float(r) for m, r in zip(models, avg_ranks)},
            "post_hoc": post_hoc
        }
    
    def _friedman_posthoc(
        self,
        data: Dict[str, List[float]],
        models: List[str]
    ) -> Dict[str, Any]:
        """
        Perform post-hoc analysis after Friedman test.
        
        Args:
            data: Model performance data
            models: List of model names
            
        Returns:
            Post-hoc test results
        """
        n_models = len(models)
        p_values = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                # Wilcoxon signed-rank test for pairwise comparison
                statistic, p_value = wilcoxon(data[models[i]], data[models[j]])
                p_values[i, j] = p_value
                p_values[j, i] = p_value
        
        # Apply Bonferroni correction
        n_comparisons = n_models * (n_models - 1) / 2
        corrected_p_values = np.minimum(p_values * n_comparisons, 1.0)
        
        return {
            "method": "Wilcoxon with Bonferroni correction",
            "p_values": p_values.tolist(),
            "corrected_p_values": corrected_p_values.tolist(),
            "model_names": models
        }
    
    def _nemenyi_test(
        self,
        data: Dict[str, List[float]],
        metric: str
    ) -> Dict[str, Any]:
        """
        Perform Nemenyi test for multiple model comparison.
        
        Args:
            data: Model performance data
            metric: Metric name
            
        Returns:
            Test results
        """
        # Calculate critical difference
        models = list(data.keys())
        n_models = len(models)
        n_datasets = len(next(iter(data.values())))
        
        # Calculate average ranks
        values = [data[model] for model in models]
        ranks = np.array([rankdata(-np.array(v)) for v in values]).T
        avg_ranks = np.mean(ranks, axis=0)
        
        # Critical difference (simplified version)
        q_alpha = 2.569  # For alpha=0.05, k=5 models
        cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
        
        # Determine significant differences
        significant_pairs = []
        for i in range(n_models):
            for j in range(i + 1, n_models):
                if abs(avg_ranks[i] - avg_ranks[j]) > cd:
                    significant_pairs.append((models[i], models[j]))
        
        return {
            "test": "Nemenyi",
            "critical_difference": float(cd),
            "average_ranks": {m: float(r) for m, r in zip(models, avg_ranks)},
            "significant_pairs": significant_pairs
        }
    
    def _wilcoxon_test(
        self,
        data: Dict[str, List[float]],
        metric: str
    ) -> Dict[str, Any]:
        """
        Perform pairwise Wilcoxon signed-rank tests.
        
        Args:
            data: Model performance data
            metric: Metric name
            
        Returns:
            Test results
        """
        models = list(data.keys())
        
        if len(models) != 2:
            raise ValueError("Wilcoxon test requires exactly 2 models")
        
        statistic, p_value = wilcoxon(data[models[0]], data[models[1]])
        
        # Calculate effect size (r = Z / sqrt(N))
        n = len(data[models[0]])
        z_score = stats.norm.ppf(1 - p_value / 2)
        effect_size = abs(z_score) / np.sqrt(n)
        
        return {
            "test": "Wilcoxon signed-rank",
            "models": models,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < (1 - self.confidence_level),
            "effect_size": float(effect_size),
            "better_model": models[0] if np.mean(data[models[0]]) > np.mean(data[models[1]]) else models[1]
        }
    
    def analyze_hyperparameter_importance(
        self,
        target_metric: str,
        hyperparameters: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze hyperparameter importance for target metric.
        
        Args:
            target_metric: Target metric to analyze
            hyperparameters: List of hyperparameters to analyze
            
        Returns:
            Hyperparameter importance analysis
        """
        # Prepare data
        data = []
        
        for exp_id, exp_data in self.experiments.items():
            row = {}
            
            # Get metric value
            row["target"] = self._extract_metric(exp_data, target_metric)
            
            if row["target"] is None:
                continue
            
            # Get hyperparameters
            config = exp_data.get("config", {})
            
            if hyperparameters:
                for hp in hyperparameters:
                    row[hp] = config.get(hp)
            else:
                # Use all config parameters
                row.update(config)
            
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Calculate correlations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}
        
        for col in numeric_cols:
            if col != "target":
                corr, p_value = stats.pearsonr(df[col].dropna(), df["target"].dropna())
                correlations[col] = {
                    "correlation": float(corr),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05
                }
        
        # Feature importance using Random Forest
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare features
        feature_cols = [col for col in df.columns if col != "target"]
        X = df[feature_cols].copy()
        y = df["target"].values
        
        # Encode categorical variables
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = X[col].fillna("missing")
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
        
        # Fill missing values
        X = X.fillna(X.mean())
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        
        # Get feature importance
        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": rf.feature_importances_
        }).sort_values("importance", ascending=False)
        
        return {
            "correlations": correlations,
            "feature_importance": importance.to_dict("records"),
            "best_hyperparameters": self._find_best_hyperparameters(df, target_metric)
        }
    
    def _find_best_hyperparameters(
        self,
        df: pd.DataFrame,
        target_metric: str
    ) -> Dict[str, Any]:
        """
        Find best hyperparameter values.
        
        Args:
            df: DataFrame with hyperparameters and target metric
            target_metric: Target metric name
            
        Returns:
            Best hyperparameter values
        """
        # Find row with best target value
        best_idx = df["target"].idxmax()
        best_row = df.loc[best_idx]
        
        # Extract hyperparameters
        best_params = {}
        for col in df.columns:
            if col != "target":
                best_params[col] = best_row[col]
        
        return {
            "parameters": best_params,
            "metric_value": float(best_row["target"])
        }
    
    def generate_report(
        self,
        output_path: str,
        include_plots: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive report of aggregated results.
        
        Args:
            output_path: Path to save report
            include_plots: Whether to include visualizations
            
        Returns:
            Report data
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(self.experiments),
            "experiments": list(self.experiments.keys()),
            "aggregated_metrics": {},
            "statistical_analysis": self.statistical_results,
            "best_configurations": {},
            "summary": {}
        }
        
        # Add aggregated metrics
        if self.aggregated_results:
            if isinstance(self.aggregated_results.get("summary"), pd.DataFrame):
                report["aggregated_metrics"] = self.aggregated_results["summary"].to_dict()
            else:
                report["aggregated_metrics"] = self.aggregated_results.get("summary", {})
        
        # Find best configurations for common metrics
        common_metrics = ["accuracy", "f1_score", "precision", "recall"]
        for metric in common_metrics:
            try:
                best = self.find_best_configuration(metric)
                report["best_configurations"][metric] = best
            except Exception as e:
                logger.warning(f"Could not find best configuration for {metric}: {e}")
        
        # Generate summary statistics
        report["summary"] = self._generate_summary_statistics()
        
        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        save_json(report, output_path)
        
        # Generate plots if requested
        if include_plots:
            plot_dir = output_path.parent / "plots"
            plot_dir.mkdir(exist_ok=True)
            self._generate_plots(plot_dir)
        
        logger.info(f"Report saved to {output_path}")
        return report
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate summary statistics.
        
        Returns:
            Summary statistics
        """
        summary = {}
        
        # Collect all metrics
        all_metrics = defaultdict(list)
        
        for exp_data in self.experiments.values():
            metrics = exp_data.get("metrics", {})
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[metric_name].append(value)
        
        # Calculate statistics for each metric
        for metric_name, values in all_metrics.items():
            if values:
                summary[metric_name] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "median": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75))
                }
        
        return summary
    
    def _generate_plots(self, output_dir: Path):
        """
        Generate visualization plots.
        
        Args:
            output_dir: Directory to save plots
        """
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Metric distributions
        self._plot_metric_distributions(output_dir / "metric_distributions.png")
        
        # 2. Model comparison
        self._plot_model_comparison(output_dir / "model_comparison.png")
        
        # 3. Hyperparameter analysis
        self._plot_hyperparameter_analysis(output_dir / "hyperparameter_analysis.png")
        
        # 4. Performance over time
        self._plot_performance_timeline(output_dir / "performance_timeline.png")
    
    def _plot_metric_distributions(self, output_path: Path):
        """Plot distributions of metrics across experiments."""
        # Collect metrics
        metrics_data = defaultdict(list)
        
        for exp_data in self.experiments.values():
            metrics = exp_data.get("metrics", {})
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metrics_data[metric_name].append(value)
        
        # Create subplots
        n_metrics = len(metrics_data)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for idx, (metric_name, values) in enumerate(metrics_data.items()):
            if idx < len(axes):
                ax = axes[idx]
                ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
                ax.set_title(metric_name)
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.axvline(np.mean(values), color='red', linestyle='--', label=f'Mean: {np.mean(values):.3f}')
                ax.legend()
        
        # Remove empty subplots
        for idx in range(len(metrics_data), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_comparison(self, output_path: Path):
        """Plot model comparison."""
        if not self.aggregated_results:
            return
        
        df = self.aggregated_results.get("raw_data")
        if df is None or df.empty:
            return
        
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        metric_cols = [col for col in numeric_cols if col != "experiment_id"]
        
        if not metric_cols:
            return
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Normalize data for heatmap
        data_for_heatmap = df[metric_cols].T
        sns.heatmap(data_for_heatmap, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax)
        
        ax.set_title("Model Performance Comparison")
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Metric")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_hyperparameter_analysis(self, output_path: Path):
        """Plot hyperparameter importance analysis."""
        # This is a placeholder - actual implementation would depend on available data
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.text(0.5, 0.5, "Hyperparameter Analysis\n(Requires hyperparameter data)",
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title("Hyperparameter Importance")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_timeline(self, output_path: Path):
        """Plot performance over time."""
        # Extract timestamps and metrics
        timeline_data = []
        
        for exp_id, exp_data in self.experiments.items():
            timestamp = exp_data.get("timestamp")
            if timestamp:
                metrics = exp_data.get("metrics", {})
                if "accuracy" in metrics:
                    timeline_data.append({
                        "timestamp": pd.to_datetime(timestamp),
                        "accuracy": metrics["accuracy"],
                        "experiment": exp_id
                    })
        
        if not timeline_data:
            return
        
        df = pd.DataFrame(timeline_data).sort_values("timestamp")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(df["timestamp"], df["accuracy"], marker='o', linewidth=2)
        
        # Add experiment labels
        for idx, row in df.iterrows():
            ax.annotate(row["experiment"][:10], 
                       (row["timestamp"], row["accuracy"]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, rotation=45)
        
        ax.set_xlabel("Time")
        ax.set_ylabel("Accuracy")
        ax.set_title("Performance Timeline")
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _load_experiment(self, exp_path: Path) -> Dict[str, Any]:
        """
        Load experiment data from directory.
        
        Args:
            exp_path: Path to experiment directory
            
        Returns:
            Experiment data
        """
        exp_data = {}
        
        # Load config
        config_path = exp_path / "config.json"
        if config_path.exists():
            exp_data["config"] = load_json(config_path)
        
        # Load metrics
        metrics_path = exp_path / "metrics.json"
        if metrics_path.exists():
            exp_data["metrics"] = load_json(metrics_path)
        
        # Load results
        results_path = exp_path / "results.json"
        if results_path.exists():
            exp_data["results"] = load_json(results_path)
        
        # Add metadata
        exp_data["experiment_id"] = exp_path.name
        exp_data["path"] = str(exp_path)
        
        # Try to get timestamp from directory name or files
        try:
            exp_data["timestamp"] = datetime.fromtimestamp(exp_path.stat().st_mtime)
        except:
            exp_data["timestamp"] = None
        
        return exp_data
    
    def _load_experiments_list(self):
        """Load list of experiments from file."""
        if self.experiments_file and Path(self.experiments_file).exists():
            with open(self.experiments_file, 'r') as f:
                experiment_ids = [line.strip() for line in f if line.strip()]
                self.load_experiments(experiment_ids)
    
    def _extract_metric(
        self,
        exp_data: Dict[str, Any],
        metric: str
    ) -> Optional[float]:
        """
        Extract metric value from experiment data.
        
        Args:
            exp_data: Experiment data
            metric: Metric name
            
        Returns:
            Metric value or None
        """
        # Try different locations for metric
        locations = [
            exp_data.get("metrics", {}),
            exp_data.get("results", {}),
            exp_data.get("results", {}).get("metrics", {}),
            exp_data.get("results", {}).get("test", {})
        ]
        
        for location in locations:
            if metric in location:
                value = location[metric]
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, dict) and "mean" in value:
                    return float(value["mean"])
        
        return None
    
    def _check_constraints(
        self,
        exp_data: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> bool:
        """
        Check if experiment satisfies constraints.
        
        Args:
            exp_data: Experiment data
            constraints: Constraints to check
            
        Returns:
            True if constraints are satisfied
        """
        config = exp_data.get("config", {})
        
        for key, value in constraints.items():
            if key not in config:
                return False
            
            if isinstance(value, (list, tuple)):
                # Value should be in list
                if config[key] not in value:
                    return False
            elif isinstance(value, dict):
                # Range constraint
                if "min" in value and config[key] < value["min"]:
                    return False
                if "max" in value and config[key] > value["max"]:
                    return False
            else:
                # Exact match
                if config[key] != value:
                    return False
        
        return True
    
    def _prepare_comparison_data(
        self,
        metric: str,
        models: Optional[List[str]]
    ) -> Dict[str, List[float]]:
        """
        Prepare data for statistical comparison.
        
        Args:
            metric: Metric to compare
            models: Optional list of models
            
        Returns:
            Dictionary mapping models to metric values
        """
        data = defaultdict(list)
        
        for exp_id, exp_data in self.experiments.items():
            # Determine model name
            model_name = exp_data.get("config", {}).get("model_name")
            
            if models and model_name not in models:
                continue
            
            # Get metric value
            value = self._extract_metric(exp_data, metric)
            
            if value is not None:
                data[model_name].append(value)
        
        return dict(data)
