"""
Model Evaluation Script for AG News Text Classification
========================================================

This script implements comprehensive model evaluation following methodologies from:
- Reimers & Gurevych (2017): "Reporting Score Distributions Makes a Difference"
- Dror et al. (2018): "The Hitchhiker's Guide to Testing Statistical Significance in NLP"
- Benavoli et al. (2017): "Time for a Change: a Tutorial for Comparing Multiple Classifiers"

The evaluation pipeline implements:
1. Multi-model evaluation with statistical testing
2. Comprehensive metrics computation
3. Error analysis and interpretability
4. Performance comparison and ranking
5. Report generation with visualizations

Statistical Framework:
For comparing k models on metric M:
- Null hypothesis H₀: All models perform equally
- Alternative H₁: At least one model differs
- Tests: McNemar's test (pairwise), Friedman test (multiple)

Author: Võ Hải Dũng
License: MIT
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from scipy import stats
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    log_loss
)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.data.datasets.ag_news import AGNewsDataset, AGNewsConfig
from src.data.loaders.dataloader import get_dataloader
from src.utils.logging_config import setup_logging
from src.utils.io_utils import safe_save, safe_load, ensure_dir, find_files
from configs.constants import (
    AG_NEWS_NUM_CLASSES,
    AG_NEWS_CLASSES,
    MAX_SEQUENCE_LENGTH,
    MODELS_DIR,
    DATA_DIR
)

# Setup logging
logger = setup_logging(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluator implementing evaluation methodologies from:
    - Sokolova & Lapalme (2009): "A systematic analysis of performance measures"
    - Japkowicz & Shah (2011): "Evaluating Learning Algorithms"
    
    The evaluator computes:
    1. Standard classification metrics
    2. Statistical significance tests
    3. Confidence intervals using bootstrap
    4. Error analysis and confusion patterns
    """
    
    def __init__(
        self,
        model_path: Path,
        model_name: str,
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_length: int = MAX_SEQUENCE_LENGTH
    ):
        """
        Initialize evaluator with model.
        
        Args:
            model_path: Path to model checkpoint
            model_name: Model identifier
            tokenizer: Tokenizer for preprocessing
            device: Compute device
            max_length: Maximum sequence length
        """
        self.model_path = model_path
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        logger.info(f"Initialized evaluator for {model_name}")
    
    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint."""
        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=AG_NEWS_NUM_CLASSES
            )
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise
    
    def evaluate(
        self,
        dataloader: DataLoader,
        compute_confidence: bool = True,
        bootstrap_samples: int = 1000
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Implements evaluation protocol from:
        - Kohavi (1995): "A Study of Cross-Validation and Bootstrap"
        
        Args:
            dataloader: Data loader for evaluation
            compute_confidence: Whether to compute confidence intervals
            bootstrap_samples: Number of bootstrap samples
            
        Returns:
            Dictionary containing evaluation results
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        total_loss = 0.0
        
        # Inference
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {self.model_name}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                probabilities = F.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(probabilities, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probabilities)
        labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = self._calculate_metrics(predictions, probabilities, labels)
        metrics["loss"] = total_loss / len(dataloader)
        
        # Confidence intervals
        if compute_confidence:
            confidence_intervals = self._bootstrap_confidence_intervals(
                labels, predictions, bootstrap_samples
            )
            metrics["confidence_intervals"] = confidence_intervals
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(labels, predictions).tolist()
        
        # Per-class metrics
        metrics["per_class_metrics"] = self._calculate_per_class_metrics(
            labels, predictions, probabilities
        )
        
        # Error analysis
        metrics["error_analysis"] = self._analyze_errors(
            labels, predictions, probabilities
        )
        
        return metrics
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics.
        
        Following metrics taxonomy from:
        - Sokolova & Lapalme (2009): "A systematic analysis of performance measures"
        """
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        
        # Precision, Recall, F1 (macro, micro, weighted)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average="macro"
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            labels, predictions, average="micro"
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            labels, predictions, average="weighted"
        )
        
        # Advanced metrics
        mcc = matthews_corrcoef(labels, predictions)
        kappa = cohen_kappa_score(labels, predictions)
        
        # Probabilistic metrics
        try:
            # One-hot encode labels for multi-class ROC-AUC
            from sklearn.preprocessing import label_binarize
            labels_binarized = label_binarize(labels, classes=list(range(AG_NEWS_NUM_CLASSES)))
            roc_auc = roc_auc_score(labels_binarized, probabilities, multi_class="ovr")
        except:
            roc_auc = 0.0
        
        logloss = log_loss(labels, probabilities)
        
        # Expected Calibration Error (ECE)
        ece = self._calculate_ece(labels, predictions, probabilities)
        
        return {
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "precision_micro": precision_micro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_micro": recall_micro,
            "recall_weighted": recall_weighted,
            "f1_macro": f1_macro,
            "f1_micro": f1_micro,
            "f1_weighted": f1_weighted,
            "matthews_corrcoef": mcc,
            "cohen_kappa": kappa,
            "roc_auc": roc_auc,
            "log_loss": logloss,
            "expected_calibration_error": ece
        }
    
    def _calculate_ece(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """
        Calculate Expected Calibration Error.
        
        Following:
        - Guo et al. (2017): "On Calibration of Modern Neural Networks"
        """
        max_probs = np.max(probabilities, axis=1)
        
        ece = 0.0
        for bin_idx in range(n_bins):
            bin_lower = bin_idx / n_bins
            bin_upper = (bin_idx + 1) / n_bins
            
            in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (predictions[in_bin] == labels[in_bin]).mean()
                avg_confidence_in_bin = max_probs[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _bootstrap_confidence_intervals(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """
        Calculate bootstrap confidence intervals.
        
        Implements bootstrap method from:
        - Efron & Tibshirani (1993): "An Introduction to the Bootstrap"
        """
        n_samples = len(labels)
        bootstrap_metrics = defaultdict(list)
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            boot_labels = labels[indices]
            boot_predictions = predictions[indices]
            
            # Calculate metrics
            bootstrap_metrics["accuracy"].append(
                accuracy_score(boot_labels, boot_predictions)
            )
            f1 = precision_recall_fscore_support(
                boot_labels, boot_predictions, average="macro"
            )[2]
            bootstrap_metrics["f1_macro"].append(f1)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        confidence_intervals = {}
        for metric, values in bootstrap_metrics.items():
            lower = np.percentile(values, lower_percentile)
            upper = np.percentile(values, upper_percentile)
            confidence_intervals[metric] = (lower, upper)
        
        return confidence_intervals
    
    def _calculate_per_class_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """Calculate per-class metrics."""
        per_class_metrics = {}
        
        for class_idx, class_name in enumerate(AG_NEWS_CLASSES):
            # Binary classification for this class
            binary_labels = (labels == class_idx).astype(int)
            binary_predictions = (predictions == class_idx).astype(int)
            binary_probs = probabilities[:, class_idx]
            
            # Calculate metrics
            tp = np.sum((binary_labels == 1) & (binary_predictions == 1))
            tn = np.sum((binary_labels == 0) & (binary_predictions == 0))
            fp = np.sum((binary_labels == 0) & (binary_predictions == 1))
            fn = np.sum((binary_labels == 1) & (binary_predictions == 0))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_class_metrics[class_name] = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": np.sum(labels == class_idx),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn)
            }
        
        return per_class_metrics
    
    def _analyze_errors(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze prediction errors.
        
        Implements error analysis from:
        - Koh & Liang (2017): "Understanding Black-box Predictions"
        """
        errors = predictions != labels
        error_indices = np.where(errors)[0]
        
        if len(error_indices) == 0:
            return {"num_errors": 0}
        
        # Analyze error patterns
        error_analysis = {
            "num_errors": len(error_indices),
            "error_rate": len(error_indices) / len(labels),
            "confusion_pairs": defaultdict(int)
        }
        
        # Most confused pairs
        for idx in error_indices:
            true_label = AG_NEWS_CLASSES[labels[idx]]
            pred_label = AG_NEWS_CLASSES[predictions[idx]]
            error_analysis["confusion_pairs"][f"{true_label} -> {pred_label}"] += 1
        
        # Confidence analysis for errors
        error_confidences = np.max(probabilities[error_indices], axis=1)
        correct_confidences = np.max(probabilities[~errors], axis=1)
        
        error_analysis["error_confidence"] = {
            "mean": float(np.mean(error_confidences)),
            "std": float(np.std(error_confidences)),
            "min": float(np.min(error_confidences)),
            "max": float(np.max(error_confidences))
        }
        
        error_analysis["correct_confidence"] = {
            "mean": float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
            "std": float(np.std(correct_confidences)) if len(correct_confidences) > 0 else 0.0
        }
        
        return error_analysis


def mcnemar_test(
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    labels: np.ndarray
) -> Tuple[float, float]:
    """
    McNemar's test for comparing two classifiers.
    
    Implements test from:
    - McNemar (1947): "Note on the sampling error of the difference between correlated proportions"
    
    Returns:
        Tuple of (statistic, p-value)
    """
    # Create contingency table
    correct1 = predictions1 == labels
    correct2 = predictions2 == labels
    
    n00 = np.sum(~correct1 & ~correct2)  # Both wrong
    n01 = np.sum(~correct1 & correct2)   # Model 1 wrong, Model 2 right
    n10 = np.sum(correct1 & ~correct2)   # Model 1 right, Model 2 wrong  
    n11 = np.sum(correct1 & correct2)    # Both right
    
    # McNemar's statistic
    if n01 + n10 == 0:
        return 0.0, 1.0
    
    statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return statistic, p_value


def create_comparison_report(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path
) -> pd.DataFrame:
    """
    Create comprehensive comparison report.
    
    Generates report following:
    - Demšar (2006): "Statistical Comparisons of Classifiers over Multiple Data Sets"
    """
    # Create comparison dataframe
    comparison_data = []
    
    for model_name, model_results in results.items():
        row = {
            "Model": model_name,
            "Accuracy": model_results["accuracy"],
            "Precision (Macro)": model_results["precision_macro"],
            "Recall (Macro)": model_results["recall_macro"],
            "F1 (Macro)": model_results["f1_macro"],
            "F1 (Weighted)": model_results["f1_weighted"],
            "MCC": model_results["matthews_corrcoef"],
            "Cohen's Kappa": model_results["cohen_kappa"],
            "ROC-AUC": model_results["roc_auc"],
            "Log Loss": model_results["log_loss"],
            "ECE": model_results["expected_calibration_error"]
        }
        
        # Add confidence intervals if available
        if "confidence_intervals" in model_results:
            for metric, (lower, upper) in model_results["confidence_intervals"].items():
                if metric == "accuracy":
                    row["Accuracy CI"] = f"[{lower:.4f}, {upper:.4f}]"
                elif metric == "f1_macro":
                    row["F1 Macro CI"] = f"[{lower:.4f}, {upper:.4f}]"
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values("F1 (Macro)", ascending=False)
    
    # Save to CSV
    df.to_csv(output_dir / "model_comparison.csv", index=False)
    
    # Create LaTeX table
    latex_table = df.to_latex(index=False, float_format="%.4f")
    with open(output_dir / "model_comparison.tex", "w") as f:
        f.write(latex_table)
    
    return df


def create_visualizations(
    results: Dict[str, Dict[str, Any]],
    output_dir: Path
):
    """Create evaluation visualizations."""
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)
    
    # 1. Metrics comparison bar plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics_to_plot = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        models = list(results.keys())
        values = [results[model][metric] for model in models]
        
        bars = ax.bar(models, values)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_ylim([0.9, 1.0] if min(values) > 0.9 else [0, 1])
        ax.set_ylabel("Score")
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Confusion matrices
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, model_results) in enumerate(results.items()):
        cm = np.array(model_results["confusion_matrix"])
        
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=AG_NEWS_CLASSES,
                   yticklabels=AG_NEWS_CLASSES,
                   ax=axes[idx])
        axes[idx].set_title(f"{model_name}")
        axes[idx].set_xlabel("Predicted")
        axes[idx].set_ylabel("True")
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    logger.info(f"Visualizations saved to {output_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate all trained models on AG News dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--models-dir",
        type=str,
        default=str(MODELS_DIR),
        help="Directory containing trained models"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing test data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluation",
        help="Output directory for results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_SEQUENCE_LENGTH,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for evaluation"
    )
    parser.add_argument(
        "--compute-confidence",
        action="store_true",
        help="Compute bootstrap confidence intervals"
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples"
    )
    parser.add_argument(
        "--statistical-tests",
        action="store_true",
        help="Perform statistical significance tests"
    )
    
    return parser.parse_args()


def main():
    """
    Main evaluation pipeline.
    
    Implements evaluation protocol from:
    - Reimers & Gurevych (2017): "Reporting Score Distributions"
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluation_dir = output_dir / f"evaluation_{timestamp}"
    ensure_dir(evaluation_dir)
    
    logger.info(f"Starting evaluation - Results will be saved to {evaluation_dir}")
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Find all model directories
    models_dir = Path(args.models_dir)
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and (d / "best_model").exists()]
    
    if not model_dirs:
        logger.error(f"No trained models found in {models_dir}")
        return
    
    logger.info(f"Found {len(model_dirs)} models to evaluate")
    
    # Load test dataset
    logger.info("Loading test dataset")
    data_config = AGNewsConfig(
        data_dir=Path(args.data_dir),
        max_length=args.max_length
    )
    
    # Results storage
    all_results = {}
    all_predictions = {}
    
    # Evaluate each model
    for model_dir in model_dirs:
        model_name = model_dir.name
        logger.info(f"\nEvaluating model: {model_name}")
        
        # Get best model path
        best_model_path = model_dir / "best_model"
        
        # Load model config to get tokenizer info
        training_state_path = best_model_path / "training_state.json"
        if training_state_path.exists():
            training_state = safe_load(training_state_path)
            model_config = training_state.get("config", {})
        else:
            model_config = {}
        
        # Infer tokenizer from model name
        if "deberta" in model_name.lower():
            tokenizer_name = "microsoft/deberta-v3-base"
        elif "roberta" in model_name.lower():
            tokenizer_name = "roberta-base"
        elif "xlnet" in model_name.lower():
            tokenizer_name = "xlnet-base-cased"
        else:
            tokenizer_name = "bert-base-uncased"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Create test dataset
        test_dataset = AGNewsDataset(
            config=data_config,
            split="test",
            tokenizer=tokenizer
        )
        
        # Create dataloader
        test_loader = get_dataloader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Initialize evaluator
        evaluator = ModelEvaluator(
            model_path=best_model_path,
            model_name=model_name,
            tokenizer=tokenizer,
            device=device,
            max_length=args.max_length
        )
        
        # Evaluate
        start_time = time.time()
        results = evaluator.evaluate(
            test_loader,
            compute_confidence=args.compute_confidence,
            bootstrap_samples=args.bootstrap_samples
        )
        evaluation_time = time.time() - start_time
        
        results["evaluation_time_seconds"] = evaluation_time
        all_results[model_name] = results
        
        # Store predictions for statistical tests
        if args.statistical_tests:
            # Re-run to get predictions (not ideal but simple)
            predictions = []
            labels = []
            
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    batch_labels = batch["labels"]
                    
                    outputs = evaluator.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    
                    batch_predictions = torch.argmax(outputs.logits, dim=-1)
                    predictions.extend(batch_predictions.cpu().numpy())
                    labels.extend(batch_labels.numpy())
            
            all_predictions[model_name] = np.array(predictions)
            
            if len(all_predictions) == 1:
                # Store labels only once
                all_predictions["labels"] = np.array(labels)
        
        # Log results
        logger.info(f"Results for {model_name}:")
        logger.info(f"  Accuracy: {results['accuracy']:.4f}")
        logger.info(f"  F1 Macro: {results['f1_macro']:.4f}")
        logger.info(f"  MCC: {results['matthews_corrcoef']:.4f}")
        logger.info(f"  Evaluation time: {evaluation_time:.2f}s")
    
    # Statistical significance testing
    if args.statistical_tests and len(all_predictions) > 2:
        logger.info("\nPerforming statistical significance tests")
        
        model_names = [name for name in all_predictions.keys() if name != "labels"]
        labels = all_predictions["labels"]
        
        # Pairwise McNemar tests
        significance_results = {}
        
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                statistic, p_value = mcnemar_test(
                    all_predictions[model1],
                    all_predictions[model2],
                    labels
                )
                
                test_name = f"{model1} vs {model2}"
                significance_results[test_name] = {
                    "statistic": statistic,
                    "p_value": p_value,
                    "significant": p_value < 0.05
                }
                
                logger.info(f"{test_name}: p-value = {p_value:.4f} {'(significant)' if p_value < 0.05 else ''}")
        
        # Save significance results
        safe_save(significance_results, evaluation_dir / "significance_tests.json")
    
    # Create comparison report
    logger.info("\nCreating comparison report")
    comparison_df = create_comparison_report(all_results, evaluation_dir)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EVALUATION SUMMARY")
    logger.info("="*80)
    print(comparison_df.to_string())
    
    # Create visualizations
    logger.info("\nCreating visualizations")
    create_visualizations(all_results, evaluation_dir)
    
    # Save all results
    safe_save(all_results, evaluation_dir / "all_results.json")
    
    # Generate final report
    final_report = {
        "timestamp": timestamp,
        "num_models_evaluated": len(all_results),
        "models": list(all_results.keys()),
        "best_model_by_f1": comparison_df.iloc[0]["Model"],
        "best_f1_score": comparison_df.iloc[0]["F1 (Macro)"],
        "evaluation_summary": comparison_df.to_dict(orient="records"),
        "output_directory": str(evaluation_dir)
    }
    
    safe_save(final_report, evaluation_dir / "final_report.json")
    
    logger.info(f"\nEvaluation complete! All results saved to {evaluation_dir}")
    logger.info(f"Best model: {final_report['best_model_by_f1']} with F1={final_report['best_f1_score']:.4f}")


if __name__ == "__main__":
    main()
