"""
Classification Metrics Implementation
======================================

Comprehensive metrics for evaluating text classification models, based on:
- Sokolova & Lapalme (2009): "A systematic analysis of performance measures"
- Opitz & Burst (2019): "Macro F1 and Macro F1"
- Buckland & Gey (1994): "The relationship between Recall and Precision"

Mathematical foundations for key metrics:
- Precision: P = TP / (TP + FP)
- Recall: R = TP / (TP + FN)
- F1-Score: F1 = 2PR / (P + R)
- Matthews Correlation: MCC = (TP×TN - FP×FN) / √[(TP+FP)(TP+FN)(TN+FP)(TN+FN)]

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import numpy as np
import warnings
from scipy import stats
from sklearn import metrics as sklearn_metrics
import torch

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics computation."""
    # Basic settings
    average: str = "macro"  # "macro", "micro", "weighted", "samples"
    labels: Optional[List[int]] = None
    pos_label: int = 1  # For binary classification
    
    # Advanced metrics
    compute_confusion_matrix: bool = True
    compute_classification_report: bool = True
    compute_roc_auc: bool = True
    compute_pr_curve: bool = True
    
    # Statistical tests
    compute_confidence_intervals: bool = True
    confidence_level: float = 0.95
    n_bootstrap: int = 1000
    
    # Calibration
    compute_calibration: bool = True
    n_bins: int = 10
    
    # Per-class analysis
    compute_per_class: bool = True
    class_names: Optional[List[str]] = None


class ClassificationMetrics:
    """
    Comprehensive metrics calculator for classification tasks.
    
    Provides a wide range of metrics including:
    1. Basic metrics (accuracy, precision, recall, F1)
    2. Advanced metrics (MCC, Cohen's kappa, AUC)
    3. Calibration metrics (ECE, MCE, Brier score)
    4. Statistical significance tests
    5. Per-class detailed analysis
    """
    
    def __init__(self, config: Optional[MetricsConfig] = None):
        """
        Initialize metrics calculator.
        
        Args:
            config: Metrics configuration
        """
        self.config = config or MetricsConfig()
        
        # AG News class names
        self.default_class_names = ["World", "Sports", "Business", "Sci/Tech"]
        
    def compute_all_metrics(
        self,
        y_true: Union[np.ndarray, torch.Tensor, List],
        y_pred: Union[np.ndarray, torch.Tensor, List],
        y_prob: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Compute all configured metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Prediction probabilities (optional)
            
        Returns:
            Dictionary containing all computed metrics
        """
        # Convert to numpy arrays
        y_true = self._to_numpy(y_true)
        y_pred = self._to_numpy(y_pred)
        if y_prob is not None:
            y_prob = self._to_numpy(y_prob)
        
        metrics_dict = {}
        
        # Basic metrics
        metrics_dict.update(self.compute_basic_metrics(y_true, y_pred))
        
        # Advanced metrics
        metrics_dict.update(self.compute_advanced_metrics(y_true, y_pred))
        
        # Probability-based metrics
        if y_prob is not None:
            metrics_dict.update(self.compute_probability_metrics(y_true, y_prob))
            
            if self.config.compute_calibration:
                metrics_dict.update(self.compute_calibration_metrics(y_true, y_prob))
        
        # Per-class metrics
        if self.config.compute_per_class:
            metrics_dict["per_class"] = self.compute_per_class_metrics(y_true, y_pred)
        
        # Confusion matrix
        if self.config.compute_confusion_matrix:
            metrics_dict["confusion_matrix"] = self.compute_confusion_matrix(y_true, y_pred)
        
        # Classification report
        if self.config.compute_classification_report:
            metrics_dict["classification_report"] = self.compute_classification_report(
                y_true, y_pred
            )
        
        # Confidence intervals
        if self.config.compute_confidence_intervals:
            metrics_dict["confidence_intervals"] = self.compute_confidence_intervals(
                y_true, y_pred
            )
        
        return metrics_dict
    
    def compute_basic_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of basic metrics
        """
        metrics = {}
        
        # Accuracy
        metrics["accuracy"] = sklearn_metrics.accuracy_score(y_true, y_pred)
        
        # Precision, Recall, F1
        precision, recall, f1, support = sklearn_metrics.precision_recall_fscore_support(
            y_true, y_pred,
            average=self.config.average,
            labels=self.config.labels,
            zero_division=0
        )
        
        metrics[f"precision_{self.config.average}"] = precision
        metrics[f"recall_{self.config.average}"] = recall
        metrics[f"f1_{self.config.average}"] = f1
        
        # Additional averages
        for avg in ["micro", "macro", "weighted"]:
            if avg != self.config.average:
                _, _, f1_avg, _ = sklearn_metrics.precision_recall_fscore_support(
                    y_true, y_pred,
                    average=avg,
                    labels=self.config.labels,
                    zero_division=0
                )
                metrics[f"f1_{avg}"] = f1_avg
        
        # Balanced accuracy
        metrics["balanced_accuracy"] = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
        
        return metrics
    
    def compute_advanced_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute advanced classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of advanced metrics
        """
        metrics = {}
        
        # Matthews Correlation Coefficient
        metrics["mcc"] = sklearn_metrics.matthews_corrcoef(y_true, y_pred)
        
        # Cohen's Kappa
        metrics["cohen_kappa"] = sklearn_metrics.cohen_kappa_score(y_true, y_pred)
        
        # Hamming Loss
        metrics["hamming_loss"] = sklearn_metrics.hamming_loss(y_true, y_pred)
        
        # Jaccard Score
        metrics["jaccard_score"] = sklearn_metrics.jaccard_score(
            y_true, y_pred,
            average=self.config.average
        )
        
        return metrics
    
    def compute_probability_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute metrics requiring probability predictions.
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of probability-based metrics
        """
        metrics = {}
        
        # Ensure y_prob is 2D
        if y_prob.ndim == 1:
            y_prob = np.vstack([1 - y_prob, y_prob]).T
        
        # Log loss
        metrics["log_loss"] = sklearn_metrics.log_loss(y_true, y_prob)
        
        # ROC-AUC
        if self.config.compute_roc_auc:
            try:
                if y_prob.shape[1] == 2:
                    # Binary classification
                    metrics["roc_auc"] = sklearn_metrics.roc_auc_score(
                        y_true, y_prob[:, 1]
                    )
                else:
                    # Multi-class
                    metrics["roc_auc_ovr"] = sklearn_metrics.roc_auc_score(
                        y_true, y_prob,
                        multi_class="ovr",
                        average=self.config.average
                    )
                    metrics["roc_auc_ovo"] = sklearn_metrics.roc_auc_score(
                        y_true, y_prob,
                        multi_class="ovo",
                        average=self.config.average
                    )
            except ValueError as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
        
        # Average Precision
        if y_prob.shape[1] == 2:
            metrics["average_precision"] = sklearn_metrics.average_precision_score(
                y_true, y_prob[:, 1]
            )
        
        # Top-k accuracy
        for k in [2, 3]:
            if y_prob.shape[1] > k:
                metrics[f"top_{k}_accuracy"] = self._top_k_accuracy(y_true, y_prob, k)
        
        return metrics
    
    def compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute calibration metrics.
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            
        Returns:
            Dictionary of calibration metrics
        """
        metrics = {}
        
        # Expected Calibration Error (ECE)
        metrics["ece"] = self._expected_calibration_error(y_true, y_prob)
        
        # Maximum Calibration Error (MCE)
        metrics["mce"] = self._maximum_calibration_error(y_true, y_prob)
        
        # Brier Score
        if y_prob.shape[1] == 2:
            metrics["brier_score"] = sklearn_metrics.brier_score_loss(
                y_true, y_prob[:, 1]
            )
        else:
            # Multi-class Brier score
            metrics["brier_score"] = self._multiclass_brier_score(y_true, y_prob)
        
        return metrics
    
    def compute_per_class_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of per-class metrics
        """
        class_names = self.config.class_names or self.default_class_names
        unique_labels = np.unique(np.concatenate([y_true, y_pred]))
        
        per_class = {}
        
        for label in unique_labels:
            if label < len(class_names):
                class_name = class_names[label]
            else:
                class_name = f"Class_{label}"
            
            # Binary metrics for this class
            y_true_binary = (y_true == label).astype(int)
            y_pred_binary = (y_pred == label).astype(int)
            
            per_class[class_name] = {
                "precision": sklearn_metrics.precision_score(
                    y_true_binary, y_pred_binary, zero_division=0
                ),
                "recall": sklearn_metrics.recall_score(
                    y_true_binary, y_pred_binary, zero_division=0
                ),
                "f1": sklearn_metrics.f1_score(
                    y_true_binary, y_pred_binary, zero_division=0
                ),
                "support": int((y_true == label).sum())
            }
        
        return per_class
    
    def compute_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        normalize: Optional[str] = None
    ) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            normalize: Normalization mode ('true', 'pred', 'all')
            
        Returns:
            Confusion matrix
        """
        cm = sklearn_metrics.confusion_matrix(
            y_true, y_pred,
            labels=self.config.labels,
            normalize=normalize
        )
        
        return cm
    
    def compute_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """
        Generate classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        class_names = self.config.class_names or self.default_class_names
        
        report = sklearn_metrics.classification_report(
            y_true, y_pred,
            labels=self.config.labels,
            target_names=class_names[:len(np.unique(y_true))],
            zero_division=0
        )
        
        return report
    
    def compute_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of confidence intervals
        """
        n_samples = len(y_true)
        rng = np.random.RandomState(42)
        
        # Bootstrap sampling
        bootstrap_metrics = {
            "accuracy": [],
            "f1_macro": [],
            "precision_macro": [],
            "recall_macro": []
        }
        
        for _ in range(self.config.n_bootstrap):
            # Sample with replacement
            indices = rng.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Compute metrics
            bootstrap_metrics["accuracy"].append(
                sklearn_metrics.accuracy_score(y_true_boot, y_pred_boot)
            )
            
            p, r, f, _ = sklearn_metrics.precision_recall_fscore_support(
                y_true_boot, y_pred_boot,
                average="macro",
                zero_division=0
            )
            bootstrap_metrics["f1_macro"].append(f)
            bootstrap_metrics["precision_macro"].append(p)
            bootstrap_metrics["recall_macro"].append(r)
        
        # Compute confidence intervals
        confidence_intervals = {}
        alpha = 1 - self.config.confidence_level
        
        for metric_name, values in bootstrap_metrics.items():
            lower = np.percentile(values, 100 * alpha / 2)
            upper = np.percentile(values, 100 * (1 - alpha / 2))
            confidence_intervals[metric_name] = (lower, upper)
        
        return confidence_intervals
    
    def _expected_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: Optional[int] = None
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = Σ_m (n_m/n) |acc_m - conf_m|
        where n_m is number of samples in bin m, acc_m is accuracy in bin m,
        conf_m is average confidence in bin m.
        """
        n_bins = n_bins or self.config.n_bins
        
        # Get predicted class and confidence
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                # Accuracy in bin
                accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
                # Average confidence in bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                # ECE contribution
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _maximum_calibration_error(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: Optional[int] = None
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE = max_m |acc_m - conf_m|
        """
        n_bins = n_bins or self.config.n_bins
        
        # Get predicted class and confidence
        y_pred = np.argmax(y_prob, axis=1)
        confidences = np.max(y_prob, axis=1)
        
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                # Accuracy in bin
                accuracy_in_bin = (y_pred[in_bin] == y_true[in_bin]).mean()
                # Average confidence in bin
                avg_confidence_in_bin = confidences[in_bin].mean()
                # MCE update
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _multiclass_brier_score(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> float:
        """
        Compute multi-class Brier score.
        
        BS = (1/N) Σ_i Σ_j (p_ij - y_ij)²
        where p_ij is predicted probability and y_ij is one-hot true label.
        """
        n_classes = y_prob.shape[1]
        y_true_onehot = np.eye(n_classes)[y_true]
        
        return np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))
    
    def _top_k_accuracy(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        k: int
    ) -> float:
        """
        Compute top-k accuracy.
        
        Args:
            y_true: True labels
            y_prob: Prediction probabilities
            k: Number of top predictions to consider
            
        Returns:
            Top-k accuracy
        """
        top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]
        correct = np.any(top_k_preds == y_true.reshape(-1, 1), axis=1)
        return correct.mean()
    
    def _to_numpy(self, data: Union[np.ndarray, torch.Tensor, List]) -> np.ndarray:
        """Convert data to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        elif isinstance(data, list):
            return np.array(data)
        return data


def create_metrics_summary(
    metrics_dict: Dict[str, Any],
    format: str = "text"
) -> str:
    """
    Create formatted metrics summary.
    
    Args:
        metrics_dict: Dictionary of metrics
        format: Output format ("text", "markdown", "latex")
        
    Returns:
        Formatted metrics summary
    """
    if format == "text":
        summary = "Classification Metrics Summary\n"
        summary += "=" * 40 + "\n"
        
        # Main metrics
        summary += f"Accuracy: {metrics_dict.get('accuracy', 0):.4f}\n"
        summary += f"F1 (Macro): {metrics_dict.get('f1_macro', 0):.4f}\n"
        summary += f"F1 (Weighted): {metrics_dict.get('f1_weighted', 0):.4f}\n"
        summary += f"Precision (Macro): {metrics_dict.get('precision_macro', 0):.4f}\n"
        summary += f"Recall (Macro): {metrics_dict.get('recall_macro', 0):.4f}\n"
        
        # Advanced metrics
        if "mcc" in metrics_dict:
            summary += f"MCC: {metrics_dict['mcc']:.4f}\n"
        if "cohen_kappa" in metrics_dict:
            summary += f"Cohen's Kappa: {metrics_dict['cohen_kappa']:.4f}\n"
        
        # Calibration
        if "ece" in metrics_dict:
            summary += f"ECE: {metrics_dict['ece']:.4f}\n"
        
        # Confidence intervals
        if "confidence_intervals" in metrics_dict:
            summary += "\n95% Confidence Intervals:\n"
            for metric, (lower, upper) in metrics_dict["confidence_intervals"].items():
                summary += f"  {metric}: [{lower:.4f}, {upper:.4f}]\n"
        
        # Classification report
        if "classification_report" in metrics_dict:
            summary += "\nClassification Report:\n"
            summary += metrics_dict["classification_report"]
        
    elif format == "markdown":
        summary = "## Classification Metrics Summary\n\n"
        summary += "| Metric | Value |\n"
        summary += "|--------|-------|\n"
        
        for key, value in metrics_dict.items():
            if isinstance(value, (int, float)):
                summary += f"| {key} | {value:.4f} |\n"
    
    else:
        summary = str(metrics_dict)
    
    return summary
