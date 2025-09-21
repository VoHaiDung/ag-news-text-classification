#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple Evaluation Script for AG News Classification
====================================================

This script provides a simplified evaluation pipeline for trained models,
suitable for quick assessment and model comparison.

Following evaluation methodologies from:
- Dodge et al. (2019): "Show Your Work: Improved Reporting of Experimental Results"
- Reimers & Gurevych (2017): "Reporting Score Distributions Makes a Difference"

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from configs.constants import AG_NEWS_CLASSES, MAX_SEQUENCE_LENGTH
from quickstart.train_simple import AGNewsDataset

# Setup logging
logger = setup_logging(name=__name__)

class SimpleEvaluator:
    """
    Simple evaluator for AG News models.
    
    Implements comprehensive evaluation following:
    - Dror et al. (2018): "The Hitchhiker's Guide to Testing Statistical Significance"
    """
    
    def __init__(
        self,
        model_path: Path,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_length: int = 256
    ):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            device: Device to use
            batch_size: Batch size for evaluation
            max_length: Maximum sequence length
        """
        self.model_path = Path(model_path)
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model and tokenizer
        self._load_model()
        
        # Results storage
        self.results = {}
    
    def _load_model(self):
        """Load model and tokenizer from path."""
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        save_predictions: bool = True,
        compute_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate model on test data.
        
        Args:
            test_data: Test DataFrame with 'text' and 'label' columns
            save_predictions: Whether to save predictions
            compute_confidence: Whether to compute confidence scores
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating on {len(test_data)} samples")
        
        # Create dataset and dataloader
        dataset = AGNewsDataset(
            test_data["text"].values,
            test_data["label"].values,
            self.tokenizer,
            self.max_length
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Collect predictions
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                logits = outputs.logits
                
                # Get predictions
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(batch["labels"].cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Compute metrics
        metrics = self._compute_metrics(all_labels, all_predictions, all_probabilities)
        
        # Store results
        self.results = {
            "metrics": metrics,
            "predictions": all_predictions,
            "labels": all_labels,
            "probabilities": all_probabilities
        }
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(test_data, all_predictions, all_probabilities)
        
        return metrics
    
    def _compute_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute comprehensive evaluation metrics.
        
        Following metric recommendations from:
        - Sokolova & Lapalme (2009): "A systematic analysis of performance measures"
        """
        # Basic metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None
        )
        
        # Macro/micro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average="macro"
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            labels, predictions, average="micro"
        )
        
        # Additional metrics
        mcc = matthews_corrcoef(labels, predictions)
        kappa = cohen_kappa_score(labels, predictions)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Confidence statistics
        confidence_correct = probabilities[np.arange(len(labels)), labels].mean()
        confidence_predicted = probabilities[np.arange(len(predictions)), predictions].mean()
        max_confidence = probabilities.max(axis=1).mean()
        
        metrics = {
            # Overall metrics
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_micro": float(f1_micro),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "matthews_corrcoef": float(mcc),
            "cohen_kappa": float(kappa),
            
            # Per-class metrics
            "per_class": {
                AG_NEWS_CLASSES[i]: {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1": float(f1[i]),
                    "support": int(support[i])
                }
                for i in range(len(AG_NEWS_CLASSES))
            },
            
            # Confusion matrix
            "confusion_matrix": cm.tolist(),
            
            # Confidence metrics
            "confidence": {
                "mean_confidence_correct": float(confidence_correct),
                "mean_confidence_predicted": float(confidence_predicted),
                "mean_max_confidence": float(max_confidence)
            }
        }
        
        return metrics
    
    def _save_predictions(
        self,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ):
        """Save predictions to file."""
        output_dir = self.model_path / "evaluation"
        output_dir.mkdir(exist_ok=True)
        
        # Create predictions DataFrame
        pred_df = test_data.copy()
        pred_df["predicted_label"] = predictions
        pred_df["predicted_class"] = [AG_NEWS_CLASSES[p] for p in predictions]
        pred_df["true_class"] = [AG_NEWS_CLASSES[l] for l in test_data["label"]]
        
        # Add probabilities
        for i, class_name in enumerate(AG_NEWS_CLASSES):
            pred_df[f"prob_{class_name}"] = probabilities[:, i]
        
        pred_df["max_probability"] = probabilities.max(axis=1)
        pred_df["correct"] = pred_df["label"] == pred_df["predicted_label"]
        
        # Save to CSV
        pred_path = output_dir / "predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        logger.info(f"Predictions saved to {pred_path}")
    
    def plot_confusion_matrix(self, save_path: Optional[Path] = None):
        """Plot confusion matrix."""
        if "confusion_matrix" not in self.results.get("metrics", {}):
            logger.warning("No confusion matrix to plot")
            return
        
        cm = np.array(self.results["metrics"]["confusion_matrix"])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt="d",
            cmap="Blues",
            xticklabels=AG_NEWS_CLASSES,
            yticklabels=AG_NEWS_CLASSES
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_class_performance(self, save_path: Optional[Path] = None):
        """Plot per-class performance."""
        if "per_class" not in self.results.get("metrics", {}):
            logger.warning("No per-class metrics to plot")
            return
        
        per_class = self.results["metrics"]["per_class"]
        
        # Prepare data
        classes = list(per_class.keys())
        precisions = [per_class[c]["precision"] for c in classes]
        recalls = [per_class[c]["recall"] for c in classes]
        f1_scores = [per_class[c]["f1"] for c in classes]
        
        # Create plot
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precisions, width, label="Precision")
        ax.bar(x, recalls, width, label="Recall")
        ax.bar(x + width, f1_scores, width, label="F1-Score")
        
        ax.set_xlabel("Classes")
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Performance")
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (p, r, f) in enumerate(zip(precisions, recalls, f1_scores)):
            ax.text(i - width, p + 0.01, f"{p:.3f}", ha="center", fontsize=9)
            ax.text(i, r + 0.01, f"{r:.3f}", ha="center", fontsize=9)
            ax.text(i + width, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Class performance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def print_report(self):
        """Print detailed evaluation report."""
        if not self.results:
            logger.warning("No results to report")
            return
        
        metrics = self.results["metrics"]
        
        print("\n" + "=" * 80)
        print("EVALUATION REPORT")
        print("=" * 80)
        
        # Overall metrics
        print("\nOverall Performance:")
        print(f"  Accuracy:          {metrics['accuracy']:.4f}")
        print(f"  F1-Score (Macro):  {metrics['f1_macro']:.4f}")
        print(f"  F1-Score (Micro):  {metrics['f1_micro']:.4f}")
        print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
        print(f"  Recall (Macro):    {metrics['recall_macro']:.4f}")
        print(f"  Matthews Corr.:    {metrics['matthews_corrcoef']:.4f}")
        print(f"  Cohen's Kappa:     {metrics['cohen_kappa']:.4f}")
        
        # Confidence metrics
        print("\nConfidence Analysis:")
        conf = metrics["confidence"]
        print(f"  Mean confidence (correct):   {conf['mean_confidence_correct']:.4f}")
        print(f"  Mean confidence (predicted): {conf['mean_confidence_predicted']:.4f}")
        print(f"  Mean max confidence:         {conf['mean_max_confidence']:.4f}")
        
        # Per-class metrics
        print("\nPer-Class Performance:")
        print("-" * 60)
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<10}")
        print("-" * 60)
        
        for class_name, class_metrics in metrics["per_class"].items():
            print(f"{class_name:<15} "
                  f"{class_metrics['precision']:<10.4f} "
                  f"{class_metrics['recall']:<10.4f} "
                  f"{class_metrics['f1']:<10.4f} "
                  f"{class_metrics['support']:<10}")
        
        # Classification report
        if "labels" in self.results and "predictions" in self.results:
            print("\nDetailed Classification Report:")
            print("-" * 60)
            print(classification_report(
                self.results["labels"],
                self.results["predictions"],
                target_names=AG_NEWS_CLASSES
            ))
        
        print("=" * 80)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Simple evaluation script for AG News models"
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model directory"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "test.csv",
        help="Path to test data CSV"
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
        default=256,
        help="Maximum sequence length"
    )
    
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save visualization plots"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Ensure reproducibility
    ensure_reproducibility(seed=args.seed)
    
    # Load test data
    logger.info(f"Loading test data from {args.data_path}")
    test_data = pd.read_csv(args.data_path)
    logger.info(f"Loaded {len(test_data)} test samples")
    
    # Create evaluator
    evaluator = SimpleEvaluator(
        model_path=args.model_path,
        batch_size=args.batch_size,
        max_length=args.max_length
    )
    
    # Run evaluation
    metrics = evaluator.evaluate(test_data)
    
    # Print report
    evaluator.print_report()
    
    # Save plots if requested
    if args.save_plots:
        output_dir = args.model_path / "evaluation"
        output_dir.mkdir(exist_ok=True)
        
        evaluator.plot_confusion_matrix(output_dir / "confusion_matrix.png")
        evaluator.plot_class_performance(output_dir / "class_performance.png")
    
    # Save metrics
    metrics_path = args.model_path / "evaluation" / "metrics.json"
    metrics_path.parent.mkdir(exist_ok=True)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Evaluation complete! Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()
