"""
Ensemble Model Training Script for AG News Text Classification
===============================================================

This script implements comprehensive ensemble training following methodologies from:
- Dietterich (2000): "Ensemble Methods in Machine Learning"
- Zhou (2012): "Ensemble Methods: Foundations and Algorithms"
- Sagi & Rokach (2018): "Ensemble learning: A survey"

The ensemble training pipeline implements:
1. Multiple base model training with diversity
2. Ensemble combination strategies (voting, stacking, blending)
3. Cross-validation for meta-learner training
4. Ensemble pruning for efficiency

Mathematical Framework:
Ensemble prediction: f_ensemble(x) = Σ_i w_i * f_i(x)
where w_i are combination weights and f_i are base models.

Author: Võ Hải Dũng
License: MIT
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import time
import pickle
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.data.datasets.ag_news import AGNewsDataset, create_ag_news_datasets
from src.data.loaders.dataloader import create_train_val_test_loaders
from src.models.ensemble.voting.soft_voting import SoftVotingEnsemble
from src.models.ensemble.voting.weighted_voting import WeightedVotingEnsemble
from src.models.ensemble.stacking.stacking_classifier import StackingEnsemble
from src.models.ensemble.blending.blending_ensemble import BlendingEnsemble
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from src.utils.experiment_tracking import create_experiment, log_metrics
from src.utils.io_utils import safe_save, safe_load, ensure_dir
from configs.config_loader import load_ensemble_config
from configs.constants import (
    AG_NEWS_NUM_CLASSES,
    MAX_SEQUENCE_LENGTH,
    MODELS_DIR,
    AG_NEWS_CLASSES
)

logger = setup_logging(__name__)


class EnsembleTrainer:
    """
    Comprehensive ensemble trainer implementing strategies from:
    - Breiman (1996): "Bagging Predictors"
    - Wolpert (1992): "Stacked Generalization"
    - Ju et al. (2018): "The Ensemble of Ensemble Models"
    
    Implements diverse ensemble strategies:
    1. Homogeneous ensembles with different initializations
    2. Heterogeneous ensembles with different architectures
    3. Multi-level stacking ensembles
    """
    
    def __init__(
        self,
        base_models: List[str],
        ensemble_type: str,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        experiment_name: str
    ):
        """
        Initialize ensemble trainer.
        
        Args:
            base_models: List of base model names/paths
            ensemble_type: Type of ensemble (voting, stacking, blending)
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            config: Training configuration
            device: Compute device
            experiment_name: Name for experiment tracking
        """
        self.base_models = base_models
        self.ensemble_type = ensemble_type
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        self.experiment_name = experiment_name
        
        # Ensemble components
        self.trained_models = []
        self.model_weights = []
        self.meta_learner = None
        
        # Training history
        self.training_history = defaultdict(list)
        
        logger.info(
            f"Initialized {ensemble_type} ensemble trainer with "
            f"{len(base_models)} base models"
        )
    
    def train_base_model(
        self,
        model_name: str,
        model_idx: int,
        data_subset: Optional[Subset] = None
    ) -> Tuple[nn.Module, Dict[str, float]]:
        """
        Train a single base model.
        
        Implements diversity injection following:
        - Kuncheva & Whitaker (2003): "Measures of Diversity in Classifier Ensembles"
        
        Args:
            model_name: Name/path of model to train
            model_idx: Index of model in ensemble
            data_subset: Optional data subset for bagging
            
        Returns:
            Tuple of (trained model, validation metrics)
        """
        logger.info(f"Training base model {model_idx}: {model_name}")
        
        # Load model with different initialization seed
        model_seed = self.config.get("seed", 42) + model_idx
        ensure_reproducibility(seed=model_seed)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=AG_NEWS_NUM_CLASSES
        )
        model.to(self.device)
        
        # Create optimizer with potentially different hyperparameters
        lr_multiplier = 1.0 + (model_idx * 0.1)  # Vary learning rate
        learning_rate = self.config.get("learning_rate", 2e-5) * lr_multiplier
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=self.config.get("weight_decay", 0.01)
        )
        
        # Training loop
        num_epochs = self.config.get("base_model_epochs", 5)
        best_val_metric = 0.0
        best_model_state = None
        
        # Use data subset if provided (bagging)
        loader = DataLoader(
            data_subset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=True
        ) if data_subset else self.train_loader
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            total_loss = 0.0
            
            for batch in tqdm(loader, desc=f"Model {model_idx} Epoch {epoch}"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.get("max_grad_norm", 1.0)
                )
                optimizer.step()
                
                total_loss += loss.item()
            
            # Validation
            val_metrics = self.evaluate_model(model, self.val_loader)
            
            logger.info(
                f"Model {model_idx} Epoch {epoch}: "
                f"Loss={total_loss/len(loader):.4f}, "
                f"Val Acc={val_metrics['accuracy']:.4f}"
            )
            
            # Save best model
            if val_metrics["f1_macro"] > best_val_metric:
                best_val_metric = val_metrics["f1_macro"]
                best_model_state = model.state_dict()
        
        # Load best model state
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        return model, {"best_f1": best_val_metric}
    
    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate a single model.
        
        Args:
            model: Model to evaluate
            dataloader: Data loader for evaluation
            
        Returns:
            Dictionary of metrics
        """
        model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                probs = F.softmax(outputs.logits, dim=-1)
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                all_probs.append(probs.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="macro"
        )
        
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_macro": f1,
            "probabilities": np.concatenate(all_probs) if all_probs else None
        }
    
    def train_stacking_meta_learner(self):
        """
        Train meta-learner for stacking ensemble.
        
        Implements stacked generalization following:
        - Wolpert (1992): "Stacked Generalization"
        - Ting & Witten (1999): "Issues in Stacked Generalization"
        """
        logger.info("Training stacking meta-learner")
        
        # Collect base model predictions using cross-validation
        n_splits = self.config.get("cv_folds", 5)
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Get all training labels
        all_labels = []
        for batch in self.train_loader:
            all_labels.extend(batch["labels"].numpy())
        all_labels = np.array(all_labels)
        
        # Generate out-of-fold predictions
        oof_predictions = np.zeros(
            (len(all_labels), len(self.trained_models), AG_NEWS_NUM_CLASSES)
        )
        
        for fold_idx, (train_idx, val_idx) in enumerate(
            kfold.split(np.zeros(len(all_labels)), all_labels)
        ):
            logger.info(f"Processing fold {fold_idx + 1}/{n_splits}")
            
            # Create subset dataloaders
            val_subset = Subset(self.train_loader.dataset, val_idx)
            val_fold_loader = DataLoader(
                val_subset,
                batch_size=self.config.get("batch_size", 32),
                shuffle=False
            )
            
            # Get predictions from each base model
            for model_idx, model in enumerate(self.trained_models):
                val_metrics = self.evaluate_model(model, val_fold_loader)
                oof_predictions[val_idx, model_idx, :] = val_metrics["probabilities"]
        
        # Reshape for meta-learner training
        X_meta = oof_predictions.reshape(len(all_labels), -1)
        y_meta = all_labels
        
        # Train meta-learner
        meta_learner_type = self.config.get("meta_learner", "logistic")
        
        if meta_learner_type == "logistic":
            self.meta_learner = LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        elif meta_learner_type == "random_forest":
            self.meta_learner = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
        elif meta_learner_type == "xgboost":
            self.meta_learner = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown meta-learner: {meta_learner_type}")
        
        self.meta_learner.fit(X_meta, y_meta)
        
        logger.info(f"Meta-learner trained: {meta_learner_type}")
    
    def optimize_ensemble_weights(self):
        """
        Optimize ensemble weights for weighted voting.
        
        Implements weight optimization following:
        - Kuncheva (2014): "Combining Pattern Classifiers"
        """
        logger.info("Optimizing ensemble weights")
        
        # Get validation predictions from all models
        val_predictions = []
        val_labels = []
        
        for batch in self.val_loader:
            labels = batch["labels"].numpy()
            val_labels.extend(labels)
        val_labels = np.array(val_labels)
        
        # Collect predictions
        model_predictions = []
        for model in self.trained_models:
            metrics = self.evaluate_model(model, self.val_loader)
            model_predictions.append(metrics["probabilities"])
        
        # Grid search for optimal weights
        best_accuracy = 0.0
        best_weights = None
        
        # Generate weight candidates
        weight_candidates = []
        n_models = len(self.trained_models)
        
        # Try different weight combinations
        for i in range(100):
            weights = np.random.dirichlet(np.ones(n_models))
            weight_candidates.append(weights)
        
        for weights in weight_candidates:
            # Weighted average of predictions
            ensemble_probs = np.zeros_like(model_predictions[0])
            for i, probs in enumerate(model_predictions):
                ensemble_probs += weights[i] * probs
            
            ensemble_preds = np.argmax(ensemble_probs, axis=1)
            accuracy = accuracy_score(val_labels, ensemble_preds)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_weights = weights
        
        self.model_weights = best_weights
        logger.info(f"Optimal weights: {best_weights}")
        logger.info(f"Validation accuracy with optimal weights: {best_accuracy:.4f}")
    
    def train(self) -> Dict[str, Any]:
        """
        Complete ensemble training pipeline.
        
        Returns:
            Training results and ensemble performance
        """
        logger.info(f"Starting ensemble training: {self.ensemble_type}")
        start_time = time.time()
        
        # Step 1: Train base models
        logger.info("Step 1: Training base models")
        
        for idx, model_name in enumerate(self.base_models):
            # Optional: Use bagging for diversity
            if self.config.get("use_bagging", False):
                # Sample with replacement
                n_samples = len(self.train_loader.dataset)
                sample_indices = np.random.choice(
                    n_samples,
                    size=int(n_samples * self.config.get("bagging_ratio", 0.8)),
                    replace=True
                )
                data_subset = Subset(self.train_loader.dataset, sample_indices)
            else:
                data_subset = None
            
            model, metrics = self.train_base_model(model_name, idx, data_subset)
            self.trained_models.append(model)
            
            # Log individual model performance
            log_metrics(
                {f"model_{idx}_{k}": v for k, v in metrics.items()},
                step=idx
            )
        
        # Step 2: Train ensemble combination
        logger.info("Step 2: Training ensemble combination")
        
        if self.ensemble_type == "voting":
            # Simple or weighted voting
            if self.config.get("use_weighted_voting", True):
                self.optimize_ensemble_weights()
            else:
                # Equal weights
                self.model_weights = [1.0 / len(self.trained_models)] * len(self.trained_models)
        
        elif self.ensemble_type == "stacking":
            # Train meta-learner
            self.train_stacking_meta_learner()
        
        elif self.ensemble_type == "blending":
            # Similar to stacking but with holdout validation
            self.train_blending_meta_learner()
        
        # Step 3: Evaluate ensemble
        logger.info("Step 3: Evaluating ensemble")
        
        ensemble_metrics = self.evaluate_ensemble()
        
        training_time = time.time() - start_time
        
        # Prepare results
        results = {
            "ensemble_type": self.ensemble_type,
            "num_base_models": len(self.trained_models),
            "base_models": self.base_models,
            "model_weights": self.model_weights,
            "ensemble_metrics": ensemble_metrics,
            "training_time_seconds": training_time,
            "config": self.config
        }
        
        logger.info(f"Ensemble training completed in {training_time:.2f} seconds")
        logger.info(f"Ensemble Test Accuracy: {ensemble_metrics['test_accuracy']:.4f}")
        logger.info(f"Ensemble Test F1: {ensemble_metrics['test_f1_macro']:.4f}")
        
        return results
    
    def train_blending_meta_learner(self):
        """
        Train meta-learner for blending ensemble using holdout validation.
        """
        logger.info("Training blending meta-learner")
        
        # Split validation set for blending
        val_size = len(self.val_loader.dataset)
        blend_size = int(val_size * 0.5)
        
        blend_indices = np.random.choice(val_size, size=blend_size, replace=False)
        holdout_indices = np.setdiff1d(np.arange(val_size), blend_indices)
        
        blend_subset = Subset(self.val_loader.dataset, blend_indices)
        blend_loader = DataLoader(
            blend_subset,
            batch_size=self.config.get("batch_size", 32),
            shuffle=False
        )
        
        # Get predictions on blend set
        blend_predictions = []
        blend_labels = []
        
        for batch in blend_loader:
            blend_labels.extend(batch["labels"].numpy())
        blend_labels = np.array(blend_labels)
        
        # Collect model predictions
        for model in self.trained_models:
            metrics = self.evaluate_model(model, blend_loader)
            blend_predictions.append(metrics["probabilities"])
        
        # Prepare meta-learner input
        X_blend = np.concatenate(
            [pred.reshape(len(blend_labels), -1) for pred in blend_predictions],
            axis=1
        )
        
        # Train meta-learner
        self.meta_learner = LogisticRegression(max_iter=1000, random_state=42)
        self.meta_learner.fit(X_blend, blend_labels)
        
        logger.info("Blending meta-learner trained")
    
    def evaluate_ensemble(self) -> Dict[str, float]:
        """
        Evaluate complete ensemble on test set.
        
        Returns:
            Dictionary of ensemble metrics
        """
        logger.info("Evaluating ensemble on test set")
        
        test_predictions = []
        test_labels = []
        
        # Collect true labels
        for batch in self.test_loader:
            test_labels.extend(batch["labels"].numpy())
        test_labels = np.array(test_labels)
        
        # Get predictions from all base models
        model_predictions = []
        for model in self.trained_models:
            metrics = self.evaluate_model(model, self.test_loader)
            model_predictions.append(metrics["probabilities"])
        
        # Combine predictions based on ensemble type
        if self.ensemble_type == "voting":
            # Weighted voting
            ensemble_probs = np.zeros_like(model_predictions[0])
            for i, probs in enumerate(model_predictions):
                ensemble_probs += self.model_weights[i] * probs
            test_predictions = np.argmax(ensemble_probs, axis=1)
        
        elif self.ensemble_type == "stacking":
            # Use meta-learner
            X_test = np.concatenate(
                [pred.reshape(len(test_labels), -1) for pred in model_predictions],
                axis=1
            )
            test_predictions = self.meta_learner.predict(X_test)
        
        elif self.ensemble_type == "blending":
            # Similar to stacking
            X_test = np.concatenate(
                [pred.reshape(len(test_labels), -1) for pred in model_predictions],
                axis=1
            )
            test_predictions = self.meta_learner.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, test_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_predictions, average="macro"
        )
        
        # Generate classification report
        report = classification_report(
            test_labels,
            test_predictions,
            target_names=AG_NEWS_CLASSES,
            digits=4
        )
        
        logger.info(f"\nEnsemble Classification Report:\n{report}")
        
        return {
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_macro": f1,
            "classification_report": report
        }
    
    def save_ensemble(self):
        """Save ensemble models and configuration."""
        save_dir = Path(self.config.get("output_dir", MODELS_DIR)) / self.experiment_name
        ensure_dir(save_dir)
        
        # Save base models
        for idx, model in enumerate(self.trained_models):
            model_path = save_dir / f"base_model_{idx}"
            model.save_pretrained(model_path)
        
        # Save meta-learner if exists
        if self.meta_learner:
            joblib.dump(self.meta_learner, save_dir / "meta_learner.pkl")
        
        # Save ensemble configuration
        ensemble_config = {
            "ensemble_type": self.ensemble_type,
            "base_models": self.base_models,
            "model_weights": self.model_weights,
            "config": self.config
        }
        safe_save(ensemble_config, save_dir / "ensemble_config.json")
        
        logger.info(f"Ensemble saved to {save_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ensemble models on AG News dataset"
    )
    
    # Model arguments
    parser.add_argument(
        "--base-models",
        type=str,
        nargs="+",
        default=["roberta-base", "bert-base-uncased", "distilbert-base-uncased"],
        help="List of base model names"
    )
    parser.add_argument(
        "--ensemble-type",
        type=str,
        default="voting",
        choices=["voting", "stacking", "blending"],
        help="Type of ensemble"
    )
    parser.add_argument(
        "--ensemble-config",
        type=str,
        default="configs/models/ensemble/voting_ensemble.yaml",
        help="Path to ensemble configuration"
    )
    
    # Training arguments
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of epochs for base models"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size"
    )
    
    # Other arguments
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    
    return parser.parse_args()


def main():
    """Main ensemble training pipeline."""
    args = parse_arguments()
    
    # Setup reproducibility
    ensure_reproducibility(seed=args.seed)
    
    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"ensemble_{args.ensemble_type}_{timestamp}"
    
    logger.info(f"Starting ensemble experiment: {args.experiment_name}")
    
    # Create experiment tracker
    experiment = create_experiment(name=args.experiment_name)
    
    # Load configuration
    if Path(args.ensemble_config).exists():
        config = load_ensemble_config(Path(args.ensemble_config))
    else:
        config = {}
    
    # Override with command line arguments
    config.update({
        "base_model_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "seed": args.seed
    })
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load tokenizer (use first model's tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.base_models[0])
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_ag_news_datasets(tokenizer=tokenizer)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_train_val_test_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=args.batch_size
    )
    
    # Initialize ensemble trainer
    trainer = EnsembleTrainer(
        base_models=args.base_models,
        ensemble_type=args.ensemble_type,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device,
        experiment_name=args.experiment_name
    )
    
    # Train ensemble
    results = trainer.train()
    
    # Save ensemble
    trainer.save_ensemble()
    
    # Save results
    output_dir = Path(MODELS_DIR) / args.experiment_name
    ensure_dir(output_dir)
    safe_save(results, output_dir / "ensemble_results.json")
    
    logger.info(f"Ensemble training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
