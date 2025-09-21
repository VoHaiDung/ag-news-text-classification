#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Hyperparameter Search for AG News Classification
=================================================

Implements hyperparameter optimization using various strategies,
following best practices from:
- Bergstra & Bengio (2012): "Random Search for Hyper-Parameter Optimization"
- Akiba et al. (2019): "Optuna: A Next-generation Hyperparameter Optimization Framework"

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
import json
import time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
from optuna.trial import Trial
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.model_selection import StratifiedKFold

from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from quickstart.train_simple import AGNewsDataset, SimpleTrainer
from configs.constants import AG_NEWS_CLASSES

logger = setup_logging(__name__)

class HyperparameterOptimizer:
    """
    Hyperparameter optimizer for AG News models.
    
    Implements optimization strategies from:
    - Snoek et al. (2012): "Practical Bayesian Optimization of Machine Learning Algorithms"
    - Li et al. (2017): "Hyperband: A Novel Bandit-Based Approach"
    """
    
    def __init__(
        self,
        model_name: str,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        optimization_metric: str = "f1_macro",
        n_trials: int = 20,
        direction: str = "maximize",
        sampler: str = "tpe",
        pruner: str = "median",
        seed: int = 42
    ):
        """
        Initialize optimizer.
        
        Args:
            model_name: Model to optimize
            train_data: Training data
            val_data: Validation data
            optimization_metric: Metric to optimize
            n_trials: Number of trials
            direction: Optimization direction
            sampler: Sampling strategy
            pruner: Pruning strategy
            seed: Random seed
        """
        self.model_name = model_name
        self.train_data = train_data
        self.val_data = val_data
        self.optimization_metric = optimization_metric
        self.n_trials = n_trials
        self.direction = direction
        self.seed = seed
        
        # Setup sampler
        if sampler == "tpe":
            self.sampler = optuna.samplers.TPESampler(seed=seed)
        elif sampler == "random":
            self.sampler = optuna.samplers.RandomSampler(seed=seed)
        elif sampler == "grid":
            self.sampler = optuna.samplers.GridSampler()
        else:
            self.sampler = optuna.samplers.TPESampler(seed=seed)
        
        # Setup pruner
        if pruner == "median":
            self.pruner = optuna.pruners.MedianPruner()
        elif pruner == "hyperband":
            self.pruner = optuna.pruners.HyperbandPruner()
        else:
            self.pruner = None
        
        # Results storage
        self.best_params = None
        self.best_score = None
        self.trial_history = []
    
    def objective(self, trial: Trial) -> float:
        """
        Objective function for optimization.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Objective value
        """
        # Sample hyperparameters
        params = self._sample_hyperparameters(trial)
        
        # Log trial
        logger.info(f"Trial {trial.number}: {params}")
        
        # Train model with sampled parameters
        score = self._train_and_evaluate(params, trial)
        
        # Store trial info
        self.trial_history.append({
            "trial": trial.number,
            "params": params,
            "score": score,
            "duration": trial.duration.total_seconds() if trial.duration else 0
        })
        
        return score
    
    def _sample_hyperparameters(self, trial: Trial) -> Dict[str, Any]:
        """Sample hyperparameters for trial."""
        params = {
            # Learning rate (log scale)
            "learning_rate": trial.suggest_loguniform("learning_rate", 1e-6, 1e-3),
            
            # Batch size (categorical)
            "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32, 64]),
            
            # Number of epochs
            "num_epochs": trial.suggest_int("num_epochs", 3, 15),
            
            # Warmup ratio
            "warmup_ratio": trial.suggest_uniform("warmup_ratio", 0.0, 0.2),
            
            # Weight decay
            "weight_decay": trial.suggest_loguniform("weight_decay", 1e-5, 0.1),
            
            # Dropout rate
            "dropout_rate": trial.suggest_uniform("dropout_rate", 0.0, 0.5),
            
            # Gradient accumulation
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [1, 2, 4, 8]
            ),
            
            # Label smoothing
            "label_smoothing": trial.suggest_uniform("label_smoothing", 0.0, 0.2),
            
            # Max gradient norm
            "max_grad_norm": trial.suggest_categorical("max_grad_norm", [0.5, 1.0, 2.0]),
            
            # Learning rate scheduler
            "scheduler": trial.suggest_categorical(
                "scheduler", ["linear", "cosine", "polynomial"]
            ),
            
            # Mixed precision
            "fp16": trial.suggest_categorical("fp16", [True, False])
        }
        
        return params
    
    def _train_and_evaluate(
        self,
        params: Dict[str, Any],
        trial: Optional[Trial] = None
    ) -> float:
        """
        Train model and evaluate with given parameters.
        
        Args:
            params: Hyperparameters
            trial: Optuna trial for pruning
            
        Returns:
            Evaluation score
        """
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(AG_NEWS_CLASSES)
        )
        
        # Apply dropout rate
        if hasattr(model.config, 'hidden_dropout_prob'):
            model.config.hidden_dropout_prob = params["dropout_rate"]
        if hasattr(model.config, 'attention_probs_dropout_prob'):
            model.config.attention_probs_dropout_prob = params["dropout_rate"]
        
        # Create datasets
        train_dataset = AGNewsDataset(
            self.train_data["text"].values,
            self.train_data["label"].values,
            tokenizer,
            max_length=256
        )
        
        val_dataset = AGNewsDataset(
            self.val_data["text"].values,
            self.val_data["label"].values,
            tokenizer,
            max_length=256
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"./optuna_trial_{trial.number if trial else 0}",
            num_train_epochs=params["num_epochs"],
            per_device_train_batch_size=params["batch_size"],
            per_device_eval_batch_size=params["batch_size"] * 2,
            gradient_accumulation_steps=params["gradient_accumulation_steps"],
            learning_rate=params["learning_rate"],
            warmup_ratio=params["warmup_ratio"],
            weight_decay=params["weight_decay"],
            label_smoothing_factor=params["label_smoothing"],
            max_grad_norm=params["max_grad_norm"],
            lr_scheduler_type=params["scheduler"],
            fp16=params["fp16"],
            evaluation_strategy="epoch",
            save_strategy="no",
            logging_strategy="epoch",
            report_to="none",
            seed=self.seed,
            metric_for_best_model=self.optimization_metric,
            greater_is_better=self.direction == "maximize"
        )
        
        # Define compute metrics
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            from sklearn.metrics import accuracy_score, f1_score
            
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1_macro": f1_score(labels, predictions, average="macro")
            }
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer
        )
        
        # Train with pruning callback if trial provided
        if trial:
            class OptunaCallback:
                def __init__(self, trial, metric):
                    self.trial = trial
                    self.metric = metric
                
                def on_evaluate(self, args, state, control, metrics, **kwargs):
                    self.trial.report(metrics[self.metric], state.epoch)
                    
                    if self.trial.should_prune():
                        raise optuna.TrialPruned()
            
            callback = OptunaCallback(trial, self.optimization_metric)
            trainer.add_callback(callback)
        
        # Train
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        # Clean up
        del model
        del tokenizer
        torch.cuda.empty_cache()
        
        return eval_results[self.optimization_metric]
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Returns:
            Best parameters and results
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        # Create study
        study = optuna.create_study(
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        # Optimize
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            show_progress_bar=True
        )
        
        # Get best results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Best score: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "study": study,
            "trial_history": self.trial_history
        }
    
    def grid_search(
        self,
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Perform grid search optimization.
        
        Args:
            param_grid: Parameter grid
            
        Returns:
            Best parameters and results
        """
        from itertools import product
        
        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        best_score = None
        best_params = None
        
        for values in product(*param_values):
            params = dict(zip(param_names, values))
            
            logger.info(f"Evaluating: {params}")
            
            score = self._train_and_evaluate(params)
            
            if best_score is None or score > best_score:
                best_score = score
                best_params = params
                logger.info(f"New best score: {best_score:.4f}")
        
        self.best_params = best_params
        self.best_score = best_score
        
        return {
            "best_params": best_params,
            "best_score": best_score
        }
    
    def save_results(self, output_path: Path):
        """Save optimization results."""
        results = {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "optimization_metric": self.optimization_metric,
            "n_trials": self.n_trials,
            "trial_history": self.trial_history
        }
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_path}")

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for AG News models"
    )
    
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        help="Model to optimize"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Data directory"
    )
    
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of optimization trials"
    )
    
    parser.add_argument(
        "--optimization-metric",
        default="f1_macro",
        choices=["accuracy", "f1_macro"],
        help="Metric to optimize"
    )
    
    parser.add_argument(
        "--sampler",
        default="tpe",
        choices=["tpe", "random", "grid"],
        help="Sampling strategy"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("hyperparameter_search_results.json"),
        help="Output path for results"
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
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    train_data = pd.read_csv(args.data_dir / "train.csv")
    val_data = pd.read_csv(args.data_dir / "validation.csv")
    
    # Subsample for faster optimization
    train_data = train_data.sample(n=min(5000, len(train_data)), random_state=args.seed)
    val_data = val_data.sample(n=min(1000, len(val_data)), random_state=args.seed)
    
    logger.info(f"Using {len(train_data)} train and {len(val_data)} validation samples")
    
    # Create optimizer
    optimizer = HyperparameterOptimizer(
        model_name=args.model_name,
        train_data=train_data,
        val_data=val_data,
        optimization_metric=args.optimization_metric,
        n_trials=args.n_trials,
        sampler=args.sampler,
        seed=args.seed
    )
    
    # Run optimization
    start_time = time.time()
    results = optimizer.optimize()
    duration = time.time() - start_time
    
    # Save results
    optimizer.save_results(args.output_path)
    
    # Print summary
    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 60)
    print(f"Best {args.optimization_metric}: {results['best_score']:.4f}")
    print(f"Duration: {duration:.1f} seconds")
    print("\nBest Parameters:")
    for param, value in results["best_params"].items():
        print(f"  {param}: {value}")
    print("=" * 60)

if __name__ == "__main__":
    main()
