"""
Single Model State-of-the-Art Experiments for AG News Text Classification
================================================================================
This module implements SOTA experiments with individual transformer models,
testing various architectures and configurations to achieve best single-model performance.

The experiments include advanced training techniques, hyperparameter optimization,
and comprehensive evaluation to push the boundaries of single-model performance.

References:
    - He, P., et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention
    - Liu, Y., et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach
    - Beltagy, I., et al. (2020). Longformer: The Long-Document Transformer

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import time
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import optuna
from tqdm import tqdm

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.utils.experiment_tracking import ExperimentTracker
from src.data.datasets.ag_news import AGNewsDataset
from src.data.augmentation.mixup import MixupAugmentation
from src.models.transformers.deberta.deberta_v3 import DeBERTaV3Classifier
from src.models.transformers.roberta.roberta_enhanced import RoBERTaEnhanced
from src.models.transformers.xlnet.xlnet_classifier import XLNetClassifier
from src.models.transformers.electra.electra_discriminator import ElectraDiscriminator
from src.models.transformers.longformer.longformer_global import LongformerGlobal
from src.training.trainers.standard_trainer import StandardTrainer
from src.training.strategies.adversarial.fgm import FGMAdversarial
from src.training.callbacks.model_checkpoint import ModelCheckpoint
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


@dataclass
class SOTAConfig:
    """Configuration for SOTA experiments."""
    model_name: str
    model_type: str
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    label_smoothing: float = 0.1
    dropout: float = 0.1
    use_adversarial: bool = True
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_swa: bool = True  # Stochastic Weight Averaging
    use_lookahead: bool = True
    use_gradient_checkpointing: bool = True
    seed: int = 42


class SingleModelSOTA:
    """
    Implements state-of-the-art single model experiments.
    
    Tests various transformer architectures with advanced training techniques
    to achieve best possible single-model performance.
    """
    
    def __init__(
        self,
        experiment_name: str = "single_model_sota",
        models_to_test: Optional[List[str]] = None,
        output_dir: str = "./outputs/sota_experiments/single_model",
        use_hyperopt: bool = True,
        num_trials: int = 3,
        device: str = "cuda"
    ):
        """
        Initialize single model SOTA experiments.
        
        Args:
            experiment_name: Name of experiment
            models_to_test: List of models to test
            output_dir: Output directory
            use_hyperopt: Whether to use hyperparameter optimization
            num_trials: Number of trials per model
            device: Device to use
        """
        self.experiment_name = experiment_name
        self.models_to_test = models_to_test or self._get_default_models()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_hyperopt = use_hyperopt
        self.num_trials = num_trials
        self.device = device if torch.cuda.is_available() else "cpu"
        
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        self.experiment_tracker = ExperimentTracker(
            experiment_name=experiment_name,
            tracking_uri="./mlruns"
        )
        
        self.results = {
            "models": {},
            "best_model": None,
            "best_score": 0,
            "training_times": {},
            "inference_times": {},
            "hyperparameters": {}
        }
        
        logger.info(f"Initialized Single Model SOTA experiments")
    
    def _get_default_models(self) -> List[str]:
        """Get default SOTA models to test."""
        return [
            "microsoft/deberta-v3-large",
            "roberta-large",
            "xlnet-large-cased",
            "google/electra-large-discriminator",
            "allenai/longformer-large-4096",
            "microsoft/deberta-v3-xlarge",  # Extra large variant
            "roberta-large-mnli",  # Pre-trained on MNLI
            "albert-xxlarge-v2"  # ALBERT variant
        ]
    
    def run_experiments(self) -> Dict[str, Any]:
        """
        Run SOTA experiments for all models.
        
        Returns:
            Experiment results
        """
        logger.info("Starting Single Model SOTA experiments")
        
        # Load dataset
        dataset = self._load_and_prepare_dataset()
        
        for model_name in self.models_to_test:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing model: {model_name}")
            logger.info(f"{'='*60}")
            
            model_results = {
                "trials": [],
                "best_accuracy": 0,
                "best_f1": 0,
                "avg_accuracy": 0,
                "avg_f1": 0,
                "std_accuracy": 0,
                "training_time": 0,
                "inference_time": 0,
                "model_size": 0,
                "best_hyperparams": {}
            }
            
            # Hyperparameter optimization
            if self.use_hyperopt:
                logger.info("Running hyperparameter optimization...")
                best_params = self._optimize_hyperparameters(model_name, dataset)
                model_results["best_hyperparams"] = best_params
            else:
                best_params = self._get_default_hyperparameters(model_name)
            
            # Run multiple trials with best parameters
            accuracies = []
            f1_scores = []
            
            for trial in range(self.num_trials):
                logger.info(f"\nTrial {trial + 1}/{self.num_trials}")
                
                trial_result = self._run_single_trial(
                    model_name,
                    dataset,
                    best_params,
                    trial
                )
                
                model_results["trials"].append(trial_result)
                accuracies.append(trial_result["accuracy"])
                f1_scores.append(trial_result["f1_weighted"])
                
                # Track best results
                if trial_result["accuracy"] > model_results["best_accuracy"]:
                    model_results["best_accuracy"] = trial_result["accuracy"]
                    model_results["best_f1"] = trial_result["f1_weighted"]
                    
                    # Save best model
                    self._save_best_model(model_name, trial_result["model_path"])
            
            # Calculate statistics
            model_results["avg_accuracy"] = np.mean(accuracies)
            model_results["avg_f1"] = np.mean(f1_scores)
            model_results["std_accuracy"] = np.std(accuracies)
            
            # Store results
            self.results["models"][model_name] = model_results
            
            # Update best overall model
            if model_results["best_accuracy"] > self.results["best_score"]:
                self.results["best_score"] = model_results["best_accuracy"]
                self.results["best_model"] = model_name
            
            logger.info(f"\nModel: {model_name}")
            logger.info(f"Best Accuracy: {model_results['best_accuracy']:.4f}")
            logger.info(f"Avg Accuracy: {model_results['avg_accuracy']:.4f} ± {model_results['std_accuracy']:.4f}")
            
            # Log to experiment tracker
            self.experiment_tracker.log_metrics({
                "model": model_name,
                "best_accuracy": model_results["best_accuracy"],
                "avg_accuracy": model_results["avg_accuracy"],
                "best_f1": model_results["best_f1"]
            })
        
        # Generate final report
        self._generate_report()
        
        return self.results
    
    def _run_single_trial(
        self,
        model_name: str,
        dataset: Dict[str, Any],
        hyperparams: Dict[str, Any],
        trial_num: int
    ) -> Dict[str, Any]:
        """
        Run a single trial with given model and hyperparameters.
        
        Args:
            model_name: Model name
            dataset: Dataset
            hyperparams: Hyperparameters
            trial_num: Trial number
            
        Returns:
            Trial results
        """
        set_seed(hyperparams.get("seed", 42) + trial_num)
        
        # Create config
        config = SOTAConfig(
            model_name=model_name,
            model_type=self._get_model_type(model_name),
            **hyperparams
        )
        
        # Initialize model
        model = self._create_model(config)
        
        # Setup training
        trainer = self._create_trainer(model, config, dataset)
        
        # Training with timing
        start_time = time.time()
        
        training_history = trainer.train(
            dataset["train"],
            dataset["val"]
        )
        
        training_time = time.time() - start_time
        
        # Evaluation
        start_time = time.time()
        
        test_results = trainer.evaluate(dataset["test"])
        
        inference_time = time.time() - start_time
        
        # Save model
        model_path = self.output_dir / f"{model_name.replace('/', '_')}_trial_{trial_num}.pt"
        torch.save(model.state_dict(), model_path)
        
        # Calculate additional metrics
        predictions = trainer.predict(dataset["test"])
        detailed_metrics = self._calculate_detailed_metrics(
            predictions,
            dataset["test"]["labels"]
        )
        
        return {
            "accuracy": test_results["accuracy"],
            "f1_weighted": test_results["f1_weighted"],
            "f1_macro": test_results["f1_macro"],
            "precision": test_results["precision_weighted"],
            "recall": test_results["recall_weighted"],
            "training_time": training_time,
            "inference_time": inference_time,
            "training_history": training_history,
            "detailed_metrics": detailed_metrics,
            "model_path": str(model_path),
            "hyperparameters": asdict(config)
        }
    
    def _optimize_hyperparameters(
        self,
        model_name: str,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            model_name: Model name
            dataset: Dataset
            
        Returns:
            Best hyperparameters
        """
        def objective(trial):
            # Suggest hyperparameters
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
                "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3),
                "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.2),
                "mixup_alpha": trial.suggest_float("mixup_alpha", 0.0, 0.4)
            }
            
            # Use subset for faster optimization
            subset_dataset = self._create_subset_dataset(dataset, fraction=0.2)
            
            # Create and train model
            config = SOTAConfig(
                model_name=model_name,
                model_type=self._get_model_type(model_name),
                num_epochs=3,  # Fewer epochs for optimization
                **params
            )
            
            model = self._create_model(config)
            trainer = self._create_trainer(model, config, subset_dataset)
            
            # Train
            trainer.train(subset_dataset["train"], subset_dataset["val"])
            
            # Evaluate
            val_results = trainer.evaluate(subset_dataset["val"])
            
            return val_results["accuracy"]
        
        # Create study
        study = optuna.create_study(
            direction="maximize",
            study_name=f"{model_name}_hyperopt"
        )
        
        # Optimize
        study.optimize(objective, n_trials=20, timeout=3600)  # 1 hour timeout
        
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation accuracy: {study.best_value:.4f}")
        
        return study.best_params
    
    def _get_default_hyperparameters(self, model_name: str) -> Dict[str, Any]:
        """Get default hyperparameters for model."""
        # Model-specific defaults
        if "deberta-v3-xlarge" in model_name:
            return {
                "learning_rate": 1e-5,
                "batch_size": 8,
                "gradient_accumulation_steps": 4,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "dropout": 0.1,
                "label_smoothing": 0.1,
                "mixup_alpha": 0.2
            }
        elif "roberta-large" in model_name:
            return {
                "learning_rate": 2e-5,
                "batch_size": 16,
                "gradient_accumulation_steps": 2,
                "warmup_ratio": 0.06,
                "weight_decay": 0.01,
                "dropout": 0.1,
                "label_smoothing": 0.05,
                "mixup_alpha": 0.1
            }
        else:
            return {
                "learning_rate": 2e-5,
                "batch_size": 16,
                "gradient_accumulation_steps": 2,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "dropout": 0.1,
                "label_smoothing": 0.0,
                "mixup_alpha": 0.0
            }
    
    def _create_model(self, config: SOTAConfig):
        """Create model based on configuration."""
        model_type = config.model_type
        
        if model_type == "deberta":
            return DeBERTaV3Classifier(
                model_name=config.model_name,
                num_labels=4,
                dropout=config.dropout,
                use_gradient_checkpointing=config.use_gradient_checkpointing
            )
        elif model_type == "roberta":
            return RoBERTaEnhanced(
                model_name=config.model_name,
                num_labels=4,
                dropout=config.dropout
            )
        elif model_type == "xlnet":
            return XLNetClassifier(
                model_name=config.model_name,
                num_labels=4,
                dropout=config.dropout
            )
        elif model_type == "electra":
            return ElectraDiscriminator(
                model_name=config.model_name,
                num_labels=4,
                dropout=config.dropout
            )
        elif model_type == "longformer":
            return LongformerGlobal(
                model_name=config.model_name,
                num_labels=4,
                dropout=config.dropout
            )
        else:
            # Default to AutoModel
            return AutoModelForSequenceClassification.from_pretrained(
                config.model_name,
                num_labels=4
            )
    
    def _create_trainer(
        self,
        model,
        config: SOTAConfig,
        dataset: Dict[str, Any]
    ):
        """Create trainer with advanced techniques."""
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                save_dir=self.output_dir / "checkpoints",
                save_best_only=True,
                monitor="val_accuracy"
            ),
            EarlyStoppingCallback(early_stopping_patience=3)
        ]
        
        # Setup adversarial training if enabled
        adversarial_trainer = None
        if config.use_adversarial:
            adversarial_trainer = FGMAdversarial(epsilon=0.5)
        
        # Create trainer
        trainer = StandardTrainer(
            model=model,
            config=asdict(config),
            device=self.device,
            callbacks=callbacks,
            adversarial_trainer=adversarial_trainer,
            use_amp=config.fp16,
            gradient_accumulation_steps=config.gradient_accumulation_steps
        )
        
        return trainer
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine model type from name."""
        if "deberta" in model_name.lower():
            return "deberta"
        elif "roberta" in model_name.lower():
            return "roberta"
        elif "xlnet" in model_name.lower():
            return "xlnet"
        elif "electra" in model_name.lower():
            return "electra"
        elif "longformer" in model_name.lower():
            return "longformer"
        elif "albert" in model_name.lower():
            return "albert"
        else:
            return "auto"
    
    def _load_and_prepare_dataset(self) -> Dict[str, Any]:
        """Load and prepare dataset with augmentation."""
        dataset = AGNewsDataset()
        data = dataset.load_splits()
        
        # Apply augmentation to training data
        augmenter = MixupAugmentation(alpha=0.2)
        
        augmented_train = augmenter.augment(
            data["train"]["texts"],
            data["train"]["labels"]
        )
        
        data["train"]["texts"].extend(augmented_train["texts"])
        data["train"]["labels"] = np.concatenate([
            data["train"]["labels"],
            augmented_train["labels"]
        ])
        
        logger.info(f"Dataset size after augmentation: {len(data['train']['texts'])}")
        
        return data
    
    def _create_subset_dataset(
        self,
        dataset: Dict[str, Any],
        fraction: float
    ) -> Dict[str, Any]:
        """Create subset of dataset for faster experimentation."""
        subset = {}
        
        for split in ["train", "val", "test"]:
            n_samples = int(len(dataset[split]["texts"]) * fraction)
            indices = np.random.choice(
                len(dataset[split]["texts"]),
                size=n_samples,
                replace=False
            )
            
            subset[split] = {
                "texts": [dataset[split]["texts"][i] for i in indices],
                "labels": dataset[split]["labels"][indices]
            }
        
        return subset
    
    def _calculate_detailed_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate detailed metrics for analysis."""
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Per-class metrics
        per_class_metrics = {}
        for class_idx in range(4):
            class_mask = labels == class_idx
            class_preds = predictions[class_mask]
            class_labels = labels[class_mask]
            
            if len(class_labels) > 0:
                per_class_metrics[f"class_{class_idx}"] = {
                    "accuracy": accuracy_score(class_labels, class_preds),
                    "f1": f1_score(class_labels, class_preds, average="weighted"),
                    "support": len(class_labels)
                }
        
        return {
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": per_class_metrics
        }
    
    def _save_best_model(self, model_name: str, model_path: str):
        """Save best model with metadata."""
        metadata = {
            "model_name": model_name,
            "accuracy": self.results["models"][model_name]["best_accuracy"],
            "f1_score": self.results["models"][model_name]["best_f1"],
            "hyperparameters": self.results["models"][model_name]["best_hyperparams"],
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = Path(model_path).parent / f"{Path(model_path).stem}_metadata.json"
        
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved best model metadata to {metadata_path}")
    
    def _generate_report(self):
        """Generate comprehensive experiment report."""
        report = {
            "experiment_name": self.experiment_name,
            "timestamp": datetime.now().isoformat(),
            "best_model": self.results["best_model"],
            "best_accuracy": self.results["best_score"],
            "model_rankings": [],
            "recommendations": []
        }
        
        # Rank models by performance
        ranked_models = sorted(
            self.results["models"].items(),
            key=lambda x: x[1]["best_accuracy"],
            reverse=True
        )
        
        for rank, (model_name, model_results) in enumerate(ranked_models, 1):
            report["model_rankings"].append({
                "rank": rank,
                "model": model_name,
                "best_accuracy": model_results["best_accuracy"],
                "avg_accuracy": model_results["avg_accuracy"],
                "std_accuracy": model_results["std_accuracy"],
                "best_f1": model_results["best_f1"]
            })
        
        # Generate recommendations
        if self.results["best_score"] > 0.95:
            report["recommendations"].append(
                "Excellent performance achieved. Consider ensemble methods for marginal improvements."
            )
        elif self.results["best_score"] > 0.92:
            report["recommendations"].append(
                "Good performance. Try domain adaptation or larger models for improvement."
            )
        else:
            report["recommendations"].append(
                "Consider data augmentation and hyperparameter tuning for better results."
            )
        
        # Save report
        report_path = self.output_dir / "single_model_sota_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Generated report: {report_path}")
        
        # Also save as markdown
        self._save_markdown_report(report)
    
    def _save_markdown_report(self, report: Dict[str, Any]):
        """Save report as markdown."""
        md_path = self.output_dir / "single_model_sota_report.md"
        
        with open(md_path, "w") as f:
            f.write("# Single Model SOTA Experiment Report\n\n")
            f.write(f"**Date**: {report['timestamp']}\n\n")
            f.write(f"**Best Model**: {report['best_model']}\n")
            f.write(f"**Best Accuracy**: {report['best_accuracy']:.4f}\n\n")
            
            f.write("## Model Rankings\n\n")
            f.write("| Rank | Model | Best Acc | Avg Acc | Std | F1 Score |\n")
            f.write("|------|-------|----------|---------|-----|----------|\n")
            
            for model in report["model_rankings"]:
                f.write(
                    f"| {model['rank']} | {model['model']} | "
                    f"{model['best_accuracy']:.4f} | "
                    f"{model['avg_accuracy']:.4f} | "
                    f"{model['std_accuracy']:.4f} | "
                    f"{model['best_f1']:.4f} |\n"
                )
            
            f.write("\n## Recommendations\n\n")
            for rec in report["recommendations"]:
                f.write(f"- {rec}\n")
        
        logger.info(f"Saved markdown report: {md_path}")


def run_single_model_sota():
    """Run single model SOTA experiments."""
    logger.info("Starting Single Model SOTA Experiments")
    
    experiment = SingleModelSOTA(
        experiment_name="ag_news_single_sota",
        use_hyperopt=True,
        num_trials=3
    )
    
    results = experiment.run_experiments()
    
    logger.info(f"\nBest Model: {results['best_model']}")
    logger.info(f"Best Accuracy: {results['best_score']:.4f}")
    
    return results


if __name__ == "__main__":
    run_single_model_sota()
