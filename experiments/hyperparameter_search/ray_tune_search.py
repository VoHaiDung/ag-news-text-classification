"""
Ray Tune Hyperparameter Search for AG News Text Classification
================================================================================
This module implements distributed hyperparameter optimization using Ray Tune,
providing scalable and efficient search with advanced scheduling algorithms.

Ray Tune offers state-of-the-art hyperparameter tuning with support for
distributed execution, early stopping, and population-based training.

References:
    - Liaw, R., et al. (2018). Tune: A Research Platform for Distributed Model Selection and Training
    - Li, L., et al. (2020). A System for Massively Parallel Hyperparameter Tuning

Author: Võ Hải Dũng
License: MIT
"""

import logging
import os
from typing import Dict, Any, List, Optional, Callable, Union
from pathlib import Path
import json
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import (
    ASHAScheduler,
    PopulationBasedTraining,
    HyperBandScheduler,
    MedianStoppingRule,
    FIFOScheduler
)
from ray.tune.suggest import (
    TuneBOHB,
    TuneOptuna,
    TuneSkOpt,
    TuneHyperOpt
)
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.integration.pytorch_lightning import TuneReportCallback

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class RayTuneSearch:
    """
    Distributed hyperparameter search using Ray Tune.
    
    This class provides:
    - Multiple search algorithms (BOHB, Optuna, BayesOpt, etc.)
    - Advanced scheduling (ASHA, PBT, HyperBand)
    - Distributed execution support
    - Resource management
    - Checkpoint and resume capabilities
    """
    
    def __init__(
        self,
        model_name: str,
        search_space: Dict[str, Any],
        metric: str = "accuracy",
        mode: str = "max",
        num_samples: int = 100,
        scheduler_type: str = "asha",
        search_algorithm: str = "optuna",
        resources_per_trial: Optional[Dict[str, Union[int, float]]] = None,
        local_dir: str = "./ray_results",
        name: Optional[str] = None,
        resume: bool = False,
        seed: int = 42
    ):
        """
        Initialize Ray Tune hyperparameter search.
        
        Args:
            model_name: Name of model to optimize
            search_space: Hyperparameter search space
            metric: Metric to optimize
            mode: Optimization mode ("max" or "min")
            num_samples: Number of trials to run
            scheduler_type: Type of scheduler to use
            search_algorithm: Search algorithm to use
            resources_per_trial: Resources allocated per trial
            local_dir: Directory for saving results
            name: Experiment name
            resume: Whether to resume from checkpoint
            seed: Random seed
        """
        self.model_name = model_name
        self.search_space = self._convert_search_space(search_space)
        self.metric = metric
        self.mode = mode
        self.num_samples = num_samples
        self.scheduler_type = scheduler_type
        self.search_algorithm = search_algorithm
        self.resources_per_trial = resources_per_trial or {"cpu": 2, "gpu": 0.5}
        self.local_dir = Path(local_dir)
        self.name = name or f"ray_tune_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.resume = resume
        self.seed = seed
        
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize search algorithm
        self.search_alg = self._create_search_algorithm()
        
        # Results storage
        self.results = None
        self.best_config = None
        self.best_checkpoint = None
        
        logger.info(f"Initialized Ray Tune search for {model_name}")
    
    def _convert_search_space(self, search_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert search space to Ray Tune format.
        
        Args:
            search_space: Original search space
            
        Returns:
            Ray Tune compatible search space
        """
        tune_space = {}
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "uniform")
                
                if param_type == "uniform":
                    tune_space[param_name] = tune.uniform(
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_type == "loguniform":
                    tune_space[param_name] = tune.loguniform(
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_type == "choice":
                    tune_space[param_name] = tune.choice(param_config["choices"])
                elif param_type == "randint":
                    tune_space[param_name] = tune.randint(
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_type == "quniform":
                    tune_space[param_name] = tune.quniform(
                        param_config["low"],
                        param_config["high"],
                        param_config.get("q", 1)
                    )
                elif param_type == "grid":
                    tune_space[param_name] = tune.grid_search(param_config["values"])
                else:
                    tune_space[param_name] = param_config
            else:
                # Direct value or list
                tune_space[param_name] = param_config
        
        return tune_space
    
    def _create_scheduler(self):
        """
        Create scheduler based on type.
        
        Returns:
            Ray Tune scheduler
        """
        if self.scheduler_type == "asha":
            return ASHAScheduler(
                metric=self.metric,
                mode=self.mode,
                max_t=100,
                grace_period=10,
                reduction_factor=3,
                brackets=1
            )
        
        elif self.scheduler_type == "pbt":
            return PopulationBasedTraining(
                metric=self.metric,
                mode=self.mode,
                perturbation_interval=4,
                hyperparam_mutations={
                    "learning_rate": tune.loguniform(1e-5, 1e-2),
                    "batch_size": [16, 32, 64],
                }
            )
        
        elif self.scheduler_type == "hyperband":
            return HyperBandScheduler(
                metric=self.metric,
                mode=self.mode,
                max_t=100
            )
        
        elif self.scheduler_type == "median":
            return MedianStoppingRule(
                metric=self.metric,
                mode=self.mode,
                grace_period=10,
                min_samples_required=3
            )
        
        else:
            return FIFOScheduler()
    
    def _create_search_algorithm(self):
        """
        Create search algorithm.
        
        Returns:
            Ray Tune search algorithm
        """
        if self.search_algorithm == "optuna":
            from optuna.samplers import TPESampler
            
            return TuneOptuna(
                sampler=TPESampler(seed=self.seed),
                metric=self.metric,
                mode=self.mode
            )
        
        elif self.search_algorithm == "bohb":
            return TuneBOHB(
                metric=self.metric,
                mode=self.mode,
                seed=self.seed
            )
        
        elif self.search_algorithm == "skopt":
            return TuneSkOpt(
                metric=self.metric,
                mode=self.mode
            )
        
        elif self.search_algorithm == "hyperopt":
            return TuneHyperOpt(
                metric=self.metric,
                mode=self.mode,
                n_initial_points=20
            )
        
        elif self.search_algorithm == "bayesopt":
            return BayesOptSearch(
                metric=self.metric,
                mode=self.mode,
                random_state=self.seed
            )
        
        else:
            return None  # Use default algorithm
    
    def objective_function(self, config: Dict[str, Any]):
        """
        Objective function for Ray Tune.
        
        Args:
            config: Hyperparameter configuration
        """
        # Set seed for reproducibility
        set_seed(self.seed)
        
        # Load data
        dataset = AGNewsDataset()
        train_data, val_data = dataset.load_train_val_split()
        
        # Create model with config
        model = self.factory.create_model(
            self.model_name,
            **config
        )
        
        # Create trainer
        trainer_config = {
            "learning_rate": config.get("learning_rate", 1e-4),
            "batch_size": config.get("batch_size", 32),
            "num_epochs": config.get("num_epochs", 10),
            "optimizer": config.get("optimizer", "adamw"),
            "weight_decay": config.get("weight_decay", 0.01)
        }
        
        trainer = BaseTrainer(
            model=model,
            config=trainer_config,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Training loop with checkpointing
        for epoch in range(trainer_config["num_epochs"]):
            # Train epoch
            train_metrics = trainer.train_epoch(train_data)
            
            # Validate
            val_metrics = trainer.validate(val_data)
            
            # Report to Ray Tune
            metrics_to_report = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                self.metric: val_metrics.get(self.metric, val_metrics.get("accuracy", 0)),
                "accuracy": val_metrics.get("accuracy", 0),
                "f1_score": val_metrics.get("f1_score", 0)
            }
            
            # Checkpoint
            checkpoint_dir = tune.get_trial_dir()
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_{epoch}.pt"
            
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "metrics": metrics_to_report,
                "config": config
            }, checkpoint_path)
            
            # Report metrics and checkpoint
            tune.report(
                **metrics_to_report,
                checkpoint_dir=checkpoint_dir
            )
    
    def run(self) -> Dict[str, Any]:
        """
        Run hyperparameter search.
        
        Returns:
            Search results
        """
        logger.info(f"Starting Ray Tune search with {self.num_samples} trials")
        
        # Configure reporter
        reporter = CLIReporter(
            metric_columns=[
                self.metric,
                "accuracy",
                "train_loss",
                "val_loss"
            ],
            parameter_columns=list(self.search_space.keys())[:5]  # Show top 5 params
        )
        
        # Run optimization
        analysis = tune.run(
            self.objective_function,
            name=self.name,
            config=self.search_space,
            metric=self.metric,
            mode=self.mode,
            num_samples=self.num_samples,
            scheduler=self.scheduler,
            search_alg=self.search_alg,
            resources_per_trial=self.resources_per_trial,
            local_dir=str(self.local_dir),
            progress_reporter=reporter,
            resume=self.resume,
            verbose=1,
            checkpoint_freq=5,
            checkpoint_at_end=True,
            max_concurrent_trials=4,
            raise_on_failed_trial=False
        )
        
        # Get best result
        self.best_config = analysis.get_best_config(
            metric=self.metric,
            mode=self.mode
        )
        
        best_trial = analysis.get_best_trial(
            metric=self.metric,
            mode=self.mode
        )
        
        self.best_checkpoint = analysis.get_best_checkpoint(
            best_trial,
            metric=self.metric,
            mode=self.mode
        )
        
        # Compile results
        self.results = {
            "best_config": self.best_config,
            "best_score": best_trial.last_result[self.metric],
            "best_trial_id": best_trial.trial_id,
            "best_checkpoint": str(self.best_checkpoint),
            "all_trials": self._extract_trial_results(analysis),
            "statistics": self._calculate_statistics(analysis),
            "search_info": {
                "num_samples": self.num_samples,
                "scheduler": self.scheduler_type,
                "algorithm": self.search_algorithm,
                "metric": self.metric,
                "mode": self.mode
            }
        }
        
        logger.info(f"Best {self.metric}: {self.results['best_score']:.4f}")
        logger.info(f"Best config: {self.best_config}")
        
        return self.results
    
    def _extract_trial_results(self, analysis) -> List[Dict[str, Any]]:
        """
        Extract results from all trials.
        
        Args:
            analysis: Ray Tune ExperimentAnalysis
            
        Returns:
            List of trial results
        """
        trials_data = []
        
        for trial in analysis.trials:
            trial_data = {
                "trial_id": trial.trial_id,
                "config": trial.config,
                "metric_score": trial.last_result.get(self.metric),
                "accuracy": trial.last_result.get("accuracy"),
                "status": trial.status,
                "iterations": trial.last_result.get("training_iteration", 0),
                "time_total_s": trial.last_result.get("time_total_s", 0),
                "timestamp": trial.last_result.get("timestamp", 0)
            }
            trials_data.append(trial_data)
        
        # Sort by metric score
        trials_data = sorted(
            trials_data,
            key=lambda x: x["metric_score"] if x["metric_score"] is not None else -float('inf'),
            reverse=(self.mode == "max")
        )
        
        return trials_data
    
    def _calculate_statistics(self, analysis) -> Dict[str, Any]:
        """
        Calculate statistics from search results.
        
        Args:
            analysis: Ray Tune ExperimentAnalysis
            
        Returns:
            Statistics dictionary
        """
        # Get all metric values
        metric_values = []
        for trial in analysis.trials:
            value = trial.last_result.get(self.metric)
            if value is not None:
                metric_values.append(value)
        
        if not metric_values:
            return {}
        
        return {
            "mean": float(np.mean(metric_values)),
            "std": float(np.std(metric_values)),
            "min": float(np.min(metric_values)),
            "max": float(np.max(metric_values)),
            "median": float(np.median(metric_values)),
            "q25": float(np.percentile(metric_values, 25)),
            "q75": float(np.percentile(metric_values, 75)),
            "num_completed": len(metric_values),
            "num_failed": len(analysis.trials) - len(metric_values)
        }
    
    def get_best_model(self):
        """
        Load and return the best model.
        
        Returns:
            Best model instance
        """
        if self.best_checkpoint is None:
            raise ValueError("No best checkpoint found. Run search first.")
        
        # Load checkpoint
        checkpoint_data = torch.load(self.best_checkpoint)
        
        # Create model with best config
        model = self.factory.create_model(
            self.model_name,
            **self.best_config
        )
        
        # Load state dict
        model.load_state_dict(checkpoint_data["model_state_dict"])
        
        return model
    
    def parallel_coordinate_plot(self):
        """Create parallel coordinate plot of hyperparameters."""
        if self.results is None:
            raise ValueError("No results available. Run search first.")
        
        import plotly.graph_objects as go
        
        # Prepare data
        trials = self.results["all_trials"][:50]  # Top 50 trials
        
        # Extract parameter values
        param_names = list(self.search_space.keys())
        data = []
        
        for trial in trials:
            if trial["metric_score"] is not None:
                values = [trial["config"].get(param, 0) for param in param_names]
                values.append(trial["metric_score"])
                data.append(values)
        
        df = pd.DataFrame(data, columns=param_names + [self.metric])
        
        # Create plot
        fig = go.Figure(data=go.Parcoords(
            line=dict(
                color=df[self.metric],
                colorscale='Viridis',
                showscale=True
            ),
            dimensions=[
                dict(
                    label=col,
                    values=df[col]
                ) for col in df.columns
            ]
        ))
        
        fig.update_layout(
            title=f"Hyperparameter Parallel Coordinates - {self.model_name}",
            height=600
        )
        
        return fig
    
    def get_importance_analysis(self) -> Dict[str, float]:
        """
        Analyze hyperparameter importance.
        
        Returns:
            Importance scores for each hyperparameter
        """
        if self.results is None:
            raise ValueError("No results available. Run search first.")
        
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import LabelEncoder
        
        # Prepare data
        trials = self.results["all_trials"]
        
        # Extract features and target
        X = []
        y = []
        
        param_names = list(self.search_space.keys())
        
        for trial in trials:
            if trial["metric_score"] is not None:
                features = []
                for param in param_names:
                    value = trial["config"].get(param)
                    
                    # Handle categorical parameters
                    if isinstance(value, str):
                        # Simple encoding for categorical
                        value = hash(value) % 1000
                    elif value is None:
                        value = 0
                    
                    features.append(value)
                
                X.append(features)
                y.append(trial["metric_score"])
        
        if not X:
            return {}
        
        X = np.array(X)
        y = np.array(y)
        
        # Train Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=self.seed)
        rf.fit(X, y)
        
        # Get feature importance
        importance_scores = {
            param: float(score)
            for param, score in zip(param_names, rf.feature_importances_)
        }
        
        # Sort by importance
        importance_scores = dict(
            sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_scores
    
    def save_results(self, filepath: str):
        """
        Save search results.
        
        Args:
            filepath: Path to save results
        """
        if self.results is None:
            raise ValueError("No results to save. Run search first.")
        
        # Convert to serializable format
        serializable_results = self._make_serializable(self.results)
        
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
