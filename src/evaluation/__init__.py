"""
Evaluation Module for AG News Text Classification
==================================================

This module provides comprehensive evaluation functionality following best practices from:
- Sokolova & Lapalme (2009): "A systematic analysis of performance measures"
- Flach (2019): "Performance Evaluation in Machine Learning"
- Raschka (2018): "Model Evaluation, Model Selection, and Algorithm Selection"

The module implements various evaluation paradigms:
1. Standard metrics computation (accuracy, precision, recall, F1)
2. Advanced metrics (MCC, Cohen's Kappa, calibration)
3. Statistical significance testing
4. Error analysis and interpretability
5. Robustness evaluation

Design Principles:
- Comprehensive metric coverage
- Statistical rigor in comparisons
- Interpretable evaluation reports
- Reproducible evaluation protocols

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from pathlib import Path
import numpy as np
import pandas as pd

# Import metrics
from src.evaluation.metrics.classification_metrics import (
    ClassificationMetrics,
    MetricsConfig,
    create_metrics_summary
)

# Import additional metrics (to be implemented)
try:
    from src.evaluation.metrics.ensemble_metrics import EnsembleMetrics
    from src.evaluation.metrics.robustness_metrics import RobustnessMetrics
    from src.evaluation.metrics.fairness_metrics import FairnessMetrics
    from src.evaluation.metrics.efficiency_metrics import EfficiencyMetrics
    ADVANCED_METRICS = True
except ImportError:
    ADVANCED_METRICS = False

# Import analysis tools
try:
    from src.evaluation.analysis.error_analysis import ErrorAnalyzer
    from src.evaluation.analysis.confusion_analysis import ConfusionAnalyzer
    from src.evaluation.analysis.class_wise_analysis import ClassWiseAnalyzer
    from src.evaluation.analysis.failure_case_analysis import FailureCaseAnalyzer
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

# Import interpretability tools
try:
    from src.evaluation.interpretability.attention_analysis import AttentionAnalyzer
    from src.evaluation.interpretability.shap_interpreter import SHAPInterpreter
    from src.evaluation.interpretability.lime_interpreter import LIMEInterpreter
    INTERPRETABILITY_AVAILABLE = True
except ImportError:
    INTERPRETABILITY_AVAILABLE = False

# Import statistical tools
try:
    from src.evaluation.statistical.significance_tests import SignificanceTests
    from src.evaluation.statistical.bootstrap_confidence import BootstrapConfidence
    from src.evaluation.statistical.mcnemar_test import McNemarTest
    STATISTICAL_AVAILABLE = True
except ImportError:
    STATISTICAL_AVAILABLE = False

# Import visualization tools
try:
    from src.evaluation.visualization.performance_plots import PerformancePlotter
    from src.evaluation.visualization.learning_curves import LearningCurvePlotter
    from src.evaluation.visualization.report_generator import ReportGenerator
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)

# Module version
__version__ = "1.0.0"

# ============================================================================
# Comprehensive Evaluator
# ============================================================================

class Evaluator:
    """
    Comprehensive evaluation framework for classification models.
    
    Provides unified interface for model evaluation with multiple metrics,
    statistical testing, and visualization capabilities.
    """
    
    def __init__(
        self,
        config: Optional[MetricsConfig] = None,
        save_dir: Optional[Union[str, Path]] = None
    ):
        """
        Initialize evaluator.
        
        Args:
            config: Metrics configuration
            save_dir: Directory to save evaluation results
        """
        self.config = config or MetricsConfig()
        self.save_dir = Path(save_dir) if save_dir else Path("./evaluation_results")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.metrics = ClassificationMetrics(config)
        
        # Initialize advanced components if available
        if ADVANCED_METRICS:
            self.ensemble_metrics = EnsembleMetrics()
            self.robustness_metrics = RobustnessMetrics()
            self.fairness_metrics = FairnessMetrics()
            self.efficiency_metrics = EfficiencyMetrics()
        
        if ANALYSIS_AVAILABLE:
            self.error_analyzer = ErrorAnalyzer()
            self.confusion_analyzer = ConfusionAnalyzer()
        
        if STATISTICAL_AVAILABLE:
            self.significance_tests = SignificanceTests()
            self.bootstrap_confidence = BootstrapConfidence()
        
        if VISUALIZATION_AVAILABLE:
            self.plotter = PerformancePlotter()
            self.report_generator = ReportGenerator()
        
        logger.info(f"Initialized Evaluator with save_dir={self.save_dir}")
    
    def evaluate(
        self,
        y_true: Union[np.ndarray, List],
        y_pred: Union[np.ndarray, List],
        y_prob: Optional[Union[np.ndarray, List]] = None,
        model_name: str = "model",
        save_results: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities
            model_name: Model identifier
            save_results: Whether to save results
            generate_report: Whether to generate HTML report
            
        Returns:
            Evaluation results dictionary
        """
        logger.info(f"Evaluating {model_name}")
        
        # Compute all metrics
        results = self.metrics.compute_all_metrics(y_true, y_pred, y_prob)
        results["model_name"] = model_name
        
        # Add advanced metrics if available
        if ADVANCED_METRICS and y_prob is not None:
            # Robustness metrics
            if hasattr(self, 'robustness_metrics'):
                results["robustness"] = self.robustness_metrics.evaluate(
                    y_true, y_pred, y_prob
                )
            
            # Fairness metrics
            if hasattr(self, 'fairness_metrics'):
                results["fairness"] = self.fairness_metrics.evaluate(
                    y_true, y_pred
                )
        
        # Perform error analysis
        if ANALYSIS_AVAILABLE and hasattr(self, 'error_analyzer'):
            results["error_analysis"] = self.error_analyzer.analyze(
                y_true, y_pred, y_prob
            )
        
        # Statistical significance
        if STATISTICAL_AVAILABLE and hasattr(self, 'bootstrap_confidence'):
            results["confidence_intervals"] = self.bootstrap_confidence.compute(
                y_true, y_pred
            )
        
        # Save results
        if save_results:
            self._save_results(results, model_name)
        
        # Generate report
        if generate_report and VISUALIZATION_AVAILABLE:
            self._generate_report(results, model_name)
        
        logger.info(f"Evaluation completed for {model_name}")
        
        return results
    
    def compare_models(
        self,
        models_results: Dict[str, Dict[str, Any]],
        baseline: Optional[str] = None,
        metrics_to_compare: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models_results: Dictionary of model results
            baseline: Baseline model name
            metrics_to_compare: Metrics to include in comparison
            
        Returns:
            Comparison dataframe
        """
        if not metrics_to_compare:
            metrics_to_compare = [
                "accuracy", "f1_macro", "f1_weighted",
                "precision_macro", "recall_macro", "mcc"
            ]
        
        # Create comparison dataframe
        comparison_data = []
        
        for model_name, results in models_results.items():
            row = {"model": model_name}
            for metric in metrics_to_compare:
                if metric in results:
                    value = results[metric]
                    row[metric] = value
                    
                    # Add relative improvement if baseline
                    if baseline and baseline != model_name and baseline in models_results:
                        baseline_value = models_results[baseline].get(metric, 0)
                        if baseline_value > 0:
                            improvement = ((value - baseline_value) / baseline_value) * 100
                            row[f"{metric}_improvement_%"] = improvement
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by primary metric
        primary_metric = metrics_to_compare[0]
        df = df.sort_values(primary_metric, ascending=False)
        
        # Save comparison
        comparison_path = self.save_dir / "model_comparison.csv"
        df.to_csv(comparison_path, index=False)
        logger.info(f"Saved model comparison to {comparison_path}")
        
        return df
    
    def evaluate_robustness(
        self,
        model,
        test_sets: Dict[str, Any],
        perturbation_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model robustness on perturbed data.
        
        Args:
            model: Model to evaluate
            test_sets: Dictionary of test sets
            perturbation_types: Types of perturbations to test
            
        Returns:
            Robustness evaluation results
        """
        if not perturbation_types:
            perturbation_types = ["typos", "synonyms", "paraphrase", "adversarial"]
        
        robustness_results = {}
        
        for test_name, test_data in test_sets.items():
            logger.info(f"Evaluating on {test_name}")
            
            # Get predictions
            y_true = test_data["labels"]
            y_pred = model.predict(test_data["inputs"])
            
            # Compute metrics
            metrics = self.metrics.compute_basic_metrics(y_true, y_pred)
            robustness_results[test_name] = metrics
        
        # Compute robustness score
        if len(robustness_results) > 1:
            scores = [r.get("f1_macro", 0) for r in robustness_results.values()]
            robustness_results["robustness_score"] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            }
        
        return robustness_results
    
    def _save_results(self, results: Dict[str, Any], model_name: str):
        """Save evaluation results."""
        import json
        
        # Save JSON results
        results_path = self.save_dir / f"{model_name}_results.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        serializable_results = convert_numpy(results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
    
    def _generate_report(self, results: Dict[str, Any], model_name: str):
        """Generate evaluation report."""
        if hasattr(self, 'report_generator'):
            report_path = self.save_dir / f"{model_name}_report.html"
            self.report_generator.generate(results, report_path)
            logger.info(f"Generated report at {report_path}")


# ============================================================================
# Evaluation Utilities
# ============================================================================

def evaluate_model(
    model,
    test_loader,
    device: str = "cuda",
    return_predictions: bool = False
) -> Dict[str, Any]:
    """
    Evaluate model on test data.
    
    Args:
        model: Model to evaluate
        test_loader: Test data loader
        device: Device to use
        return_predictions: Whether to return predictions
        
    Returns:
        Evaluation results
    """
    import torch
    
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Move to device
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch.get("attention_mask", None)
            
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(inputs, attention_mask=attention_mask)
            
            # Get predictions
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            
            # Collect results
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probs.cpu().numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Compute metrics
    evaluator = Evaluator()
    results = evaluator.evaluate(
        all_labels,
        all_predictions,
        all_probabilities,
        save_results=False,
        generate_report=False
    )
    
    if return_predictions:
        results["predictions"] = all_predictions
        results["probabilities"] = all_probabilities
        results["labels"] = all_labels
    
    return results


def cross_validate(
    model_class,
    data,
    n_folds: int = 5,
    config: Optional[Any] = None,
    stratified: bool = True
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.
    
    Args:
        model_class: Model class to instantiate
        data: Data to use
        n_folds: Number of folds
        config: Model configuration
        stratified: Use stratified splits
        
    Returns:
        Cross-validation results
    """
    from sklearn.model_selection import KFold, StratifiedKFold
    
    # Choose splitter
    if stratified:
        splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    else:
        splitter = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(splitter.split(data.X, data.y)):
        logger.info(f"Fold {fold + 1}/{n_folds}")
        
        # Split data
        X_train, X_val = data.X[train_idx], data.X[val_idx]
        y_train, y_val = data.y[train_idx], data.y[val_idx]
        
        # Create model
        model = model_class(config)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_val)
        
        # Compute metrics
        metrics = ClassificationMetrics().compute_basic_metrics(y_val, y_pred)
        fold_results.append(metrics)
    
    # Aggregate results
    aggregated = {}
    for metric in fold_results[0].keys():
        values = [r[metric] for r in fold_results]
        aggregated[f"{metric}_mean"] = np.mean(values)
        aggregated[f"{metric}_std"] = np.std(values)
    
    return {
        "fold_results": fold_results,
        "aggregated": aggregated,
        "n_folds": n_folds
    }


class EvaluationPipeline:
    """
    Complete evaluation pipeline for systematic model assessment.
    
    Orchestrates comprehensive evaluation including metrics computation,
    statistical testing, and report generation.
    """
    
    def __init__(
        self,
        models: Dict[str, Any],
        test_data,
        config: Optional[MetricsConfig] = None
    ):
        """
        Initialize evaluation pipeline.
        
        Args:
            models: Dictionary of models to evaluate
            test_data: Test dataset
            config: Evaluation configuration
        """
        self.models = models
        self.test_data = test_data
        self.evaluator = Evaluator(config)
        
        logger.info(f"Initialized EvaluationPipeline with {len(models)} models")
    
    def run(self) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline.
        
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting evaluation pipeline")
        
        results = {}
        
        # Evaluate each model
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}")
            
            # Get predictions
            model_results = evaluate_model(
                model,
                self.test_data,
                return_predictions=True
            )
            
            results[model_name] = model_results
        
        # Compare models
        if len(self.models) > 1:
            comparison = self.evaluator.compare_models(results)
            results["comparison"] = comparison
        
        # Statistical significance testing
        if len(self.models) == 2 and STATISTICAL_AVAILABLE:
            model_names = list(self.models.keys())
            results["significance"] = self._test_significance(
                results[model_names[0]],
                results[model_names[1]]
            )
        
        logger.info("Evaluation pipeline completed")
        
        return results
    
    def _test_significance(self, results1: Dict, results2: Dict) -> Dict[str, Any]:
        """Test statistical significance between two models."""
        if not STATISTICAL_AVAILABLE:
            return {}
        
        # McNemar test
        mcnemar = McNemarTest()
        p_value = mcnemar.test(
            results1["predictions"],
            results2["predictions"],
            results1["labels"]
        )
        
        return {
            "mcnemar_p_value": p_value,
            "significant": p_value < 0.05
        }


# ============================================================================
# Export Public API
# ============================================================================

__all__ = [
    # Core classes
    "Evaluator",
    "EvaluationPipeline",
    "ClassificationMetrics",
    "MetricsConfig",
    
    # Utilities
    "evaluate_model",
    "cross_validate",
    "create_metrics_summary",
    
    # Version
    "__version__"
]

# Log module initialization
logger.info(f"Evaluation module initialized (v{__version__})")
if ADVANCED_METRICS:
    logger.info("Advanced metrics available")
if ANALYSIS_AVAILABLE:
    logger.info("Analysis tools available")
if INTERPRETABILITY_AVAILABLE:
    logger.info("Interpretability tools available")
if STATISTICAL_AVAILABLE:
    logger.info("Statistical testing available")
if VISUALIZATION_AVAILABLE:
    logger.info("Visualization tools available")
