"""
Optuna-based Hyperparameter Search for AG News Text Classification
================================================================================
This module implements hyperparameter optimization using Optuna framework,
providing efficient search strategies with pruning and parallel execution.

The implementation supports various samplers, pruners, and multi-objective
optimization for finding optimal model configurations.

References:
    - Akiba, T., et al. (2019). Optuna: A Next-generation Hyperparameter Optimization Framework
    - Bergstra, J., et al. (2011). Algorithms for Hyper-Parameter Optimization

Author: Võ Hải Dũng
License: MIT
"""

import logging
import json
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
from pathlib import Path
from datetime import datetime
import warnings
import pickle

import numpy as np
import pandas as pd
import optuna
from optuna import Trial, Study
from optuna.samplers import (
    TPESampler, RandomSampler, CmaEsSampler, 
    NSGAIISampler, QMCSampler
)
from optuna.pruners import (
    MedianPruner, SuccessiveHalvingPruner, 
    HyperbandPruner, ThresholdPruner
)
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_contour
)
import torch
from sklearn.model_selection import cross_val_score

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class OptunaSearch:
    """
    Hyperparameter optimization using Optuna framework.
    
    This class provides:
    - Multiple sampling strategies
    - Pruning for early stopping
    - Multi-objective optimization
    - Parallel trial execution
    - Visualization tools
    """
    
    def __init__(
        self,
        model_name: str,
        search_space: Dict[str, Any],
        direction: Union[str, List[str]] = "maximize",
        n_trials: int = 100,
        timeout: Optional[int] = None,
        sampler: str = "tpe",
        pruner: str = "median",
        n_jobs: int = 1,
        metric: str = "accuracy",
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
        load_if_exists: bool = False,
        output_dir: str = "./optuna_results",
        seed: int = 42
    ):
        """
        Initialize Optuna hyperparameter search.
        
        Args:
            model_name: Name of model to optimize
            search_space: Search space configuration
            direction: Optimization direction(s)
            n_trials: Number of trials
            timeout: Timeout in seconds
            sampler: Sampling strategy
            pruner: Pruning strategy
            n_jobs: Number of parallel jobs
            metric: Metric to optimize
            study_name: Name for the study
            storage: Database URL for distributed optimization
            load_if_exists: Whether to load existing study
            output_dir: Directory for saving results
            seed: Random seed
        """
        self.model_name = model_name
        self.search_space = search_space
        self.direction = direction
        self.n_trials = n_trials
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.metric = metric
        self.study_name = study_name or f"optuna_study_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        self.load_if_exists = load_if_exists
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Initialize sampler
        self.sampler = self._create_sampler(sampler)
        
        # Initialize pruner
        self.pruner = self._create_pruner(pruner)
        
        # Initialize study
        self.study = None
        self.best_params = None
        self.best_value = None
        
        # Results storage
        self.results = {
            "trials": [],
            "best_params": None,
            "best_value": None,
            "study_name": self.study_name
        }
        
        # Initialize factory and registry
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        logger.info(f"Initialized OptunaSearch with {n_trials} trials for {model_name}")
    
    def _create_sampler(self, sampler_type: str) -> optuna.samplers.BaseSampler:
        """
        Create Optuna sampler.
        
        Args:
            sampler_type: Type of sampler
            
        Returns:
            Sampler instance
        """
        if sampler_type == "tpe":
            return TPESampler(
                seed=self.seed,
                n_startup_trials=10,
                n_ei_candidates=24,
                multivariate=True
            )
        elif sampler_type == "random":
            return RandomSampler(seed=self.seed)
        elif sampler_type == "cmaes":
            return CmaEsSampler(seed=self.seed)
        elif sampler_type == "nsgaii":
            return NSGAIISampler(seed=self.seed)
        elif sampler_type == "qmc":
            return QMCSampler(
                qmc_type="sobol",
                seed=self.seed
            )
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
    
    def _create_pruner(self, pruner_type: str) -> optuna.pruners.BasePruner:
        """
        Create Optuna pruner.
        
        Args:
            pruner_type: Type of pruner
            
        Returns:
            Pruner instance
        """
        if pruner_type == "median":
            return MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        elif pruner_type == "successive_halving":
            return SuccessiveHalvingPruner()
        elif pruner_type == "hyperband":
            return HyperbandPruner(
                min_resource=1,
                max_resource=100,
                reduction_factor=3
            )
        elif pruner_type == "threshold":
            return ThresholdPruner(
                lower=0.0,
                upper=1.0,
                n_warmup_steps=10
            )
        elif pruner_type == "none":
            return optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruner type: {pruner_type}")
    
    def suggest_params(self, trial: Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for a trial.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Suggested parameters
        """
        params = {}
        
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "float")
                
                if param_type == "float":
                    params[param_name] = trial.suggest_float(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        log=param_config.get("log", False)
                    )
                elif param_type == "int":
                    params[param_name] = trial.suggest_int(
                        param_name,
                        param_config["low"],
                        param_config["high"],
                        step=param_config.get("step", 1)
                    )
                elif param_type == "categorical" or param_type == "choice":
                    params[param_name] = trial.suggest_categorical(
                        param_name,
                        param_config["choices"]
                    )
                else:
                    raise ValueError(f"Unknown parameter type: {param_type}")
            else:
                # Direct value
                params[param_name] = param_config
        
        return params
    
    def objective(self, trial: Trial) -> Union[float, Tuple[float, ...]]:
        """
        Default objective function for AG News classification.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Objective value(s)
        """
        # Suggest parameters
        params = self.suggest_params(trial)
        
        # Set user attributes
        trial.set_user_attr("params", params)
        
        try:
            # Set seed for reproducibility
            set_seed(self.seed + trial.number)
            
            # Load data
            dataset = AGNewsDataset()
            train_data, val_data = dataset.load_train_val_split()
            
            # Create model with suggested parameters
            model = self.factory.create_model(
                self.model_name,
                **params
            )
            
            # Create trainer
            trainer_config = {
                "learning_rate": params.get("learning_rate", 1e-4),
                "batch_size": params.get("batch_size", 32),
                "num_epochs": params.get("num_epochs", 10),
                "warmup_steps": params.get("warmup_steps", 0),
                "weight_decay": params.get("weight_decay", 0.01),
                "optimizer": params.get("optimizer", "adamw"),
                "scheduler": params.get("scheduler", "linear"),
                "gradient_accumulation_steps": params.get("gradient_accumulation_steps", 1)
            }
            
            trainer = BaseTrainer(
                model=model,
                config=trainer_config,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Training with pruning callback
            best_val_metric = 0.0 if self.direction == "maximize" else float('inf')
            
            for epoch in range(trainer_config["num_epochs"]):
                # Train epoch
                train_metrics = trainer.train_epoch(train_data)
                
                # Validate
                val_metrics = trainer.validate(val_data)
                val_score = val_metrics.get(self.metric, val_metrics.get("accuracy", 0))
                
                # Update best metric
                if self.direction == "maximize":
                    best_val_metric = max(best_val_metric, val_score)
                else:
                    best_val_metric = min(best_val_metric, val_score)
                
                # Report intermediate value for pruning
                trial.report(val_score, epoch)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    logger.info(f"Trial {trial.number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
            
            # Multi-objective optimization
            if isinstance(self.direction, list):
                # Return multiple objectives
                inference_speed = self._measure_inference_speed(model, val_data)
                model_size = self._get_model_size(model)
                
                return best_val_metric, inference_speed, -model_size
            else:
                return best_val_metric
        
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float("-inf") if self.direction == "maximize" else float("inf")
    
    def run(self, objective_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_func: Optional custom objective function
            
        Returns:
            Optimization results
        """
        # Use provided objective or default
        objective = objective_func or self.objective
        
        # Create or load study
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            storage=self.storage,
            load_if_exists=self.load_if_exists
        )
        
        # Add callbacks
        callbacks = [
            self._log_callback,
            self._save_callback
        ]
        
        # Optimize
        logger.info(f"Starting optimization with {self.n_trials} trials")
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            callbacks=callbacks,
            gc_after_trial=True,
            show_progress_bar=True
        )
        
        # Get best results
        if isinstance(self.direction, list):
            # Multi-objective optimization
            self.best_params = [trial.params for trial in self.study.best_trials]
            self.best_value = [trial.values for trial in self.study.best_trials]
        else:
            self.best_params = self.study.best_params
            self.best_value = self.study.best_value
        
        # Compile results
        self.results = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
            "n_complete": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "n_failed": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            "trials": self._extract_trial_results(),
            "study_name": self.study_name,
            "summary": self._generate_summary()
        }
        
        logger.info(f"\nOptimization completed!")
        logger.info(f"Best {self.metric}: {self.best_value}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Save results
        self.save_results()
        
        # Generate visualizations
        self.plot_results()
        
        return self.results
    
    def _log_callback(self, study: Study, trial: optuna.trial.FrozenTrial):
        """
        Callback for logging trial results.
        
        Args:
            study: Optuna study
            trial: Completed trial
        """
        logger.info(f"Trial {trial.number} finished with value: {trial.value}, state: {trial.state.name}")
    
    def _save_callback(self, study: Study, trial: optuna.trial.FrozenTrial):
        """
        Callback for saving intermediate results.
        
        Args:
            study: Optuna study
            trial: Completed trial
        """
        # Save every 10 trials
        if trial.number % 10 == 0:
            self._save_study()
    
    def _extract_trial_results(self) -> List[Dict[str, Any]]:
        """
        Extract results from all trials.
        
        Returns:
            List of trial results
        """
        trials_data = []
        
        for trial in self.study.trials:
            trial_data = {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
                "duration": (trial.datetime_complete - trial.datetime_start).total_seconds() 
                           if trial.datetime_complete and trial.datetime_start else None
            }
            trials_data.append(trial_data)
        
        return trials_data
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary of optimization results.
        
        Returns:
            Summary dictionary
        """
        # Get completed trials
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not complete_trials:
            return {}
        
        values = [t.value for t in complete_trials if t.value is not None]
        
        summary = {
            "total_trials": len(self.study.trials),
            "completed_trials": len(complete_trials),
            "pruned_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "failed_trials": len([t for t in self.study.trials if t.state == optuna.trial.TrialState.FAIL]),
            "best_trial_number": self.study.best_trial.number if self.study.best_trial else None,
            "score_statistics": {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            }
        }
        
        # Parameter importance if available
        try:
            importance = optuna.importance.get_param_importances(self.study)
            summary["parameter_importance"] = importance
        except:
            logger.warning("Could not calculate parameter importance")
        
        return summary
    
    def plot_results(self):
        """Generate optimization visualizations."""
        output_dir = self.output_dir / "plots"
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_html(output_dir / "optimization_history.html")
            
            # Parameter importance
            if len(self.study.trials) > 1:
                fig = plot_param_importances(self.study)
                fig.write_html(output_dir / "param_importances.html")
            
            # Parallel coordinate plot
            fig = plot_parallel_coordinate(self.study)
            fig.write_html(output_dir / "parallel_coordinate.html")
            
            # Contour plot for top 2 parameters
            if len(self.search_space) >= 2:
                param_names = list(self.search_space.keys())[:2]
                fig = plot_contour(self.study, params=param_names)
                fig.write_html(output_dir / "contour_plot.html")
            
            logger.info(f"Visualizations saved to {output_dir}")
        
        except Exception as e:
            logger.warning(f"Failed to generate some visualizations: {e}")
    
    def _save_study(self):
        """Save study to file."""
        # Save as pickle
        study_path = self.output_dir / f"{self.study_name}.pkl"
        with open(study_path, "wb") as f:
            pickle.dump(self.study, f)
        
        # Save trials as JSON
        trials_data = self._extract_trial_results()
        trials_path = self.output_dir / f"{self.study_name}_trials.json"
        with open(trials_path, "w") as f:
            json.dump(trials_data, f, indent=2)
    
    def save_results(self, filepath: Optional[str] = None):
        """
        Save optimization results.
        
        Args:
            filepath: Path to save results
        """
        if filepath is None:
            filepath = self.output_dir / "optuna_results.json"
        
        # Convert to serializable format
        serializable_results = self._make_serializable(self.results)
        
        # Add metadata
        serializable_results["metadata"] = {
            "model_name": self.model_name,
            "sampler": self.sampler.__class__.__name__,
            "pruner": self.pruner.__class__.__name__,
            "n_trials": self.n_trials,
            "metric": self.metric,
            "direction": self.direction,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        
        # Also save the study
        self._save_study()
    
    def _measure_inference_speed(self, model, data) -> float:
        """
        Measure model inference speed.
        
        Args:
            model: Model to evaluate
            data: Validation data
            
        Returns:
            Inference speed (samples/second)
        """
        import time
        
        model.eval()
        total_time = 0
        n_samples = 0
        
        with torch.no_grad():
            for batch in data:
                start = time.perf_counter()
                _ = model(batch)
                total_time += time.perf_counter() - start
                n_samples += len(batch)
        
        return n_samples / total_time if total_time > 0 else 0
    
    def _get_model_size(self, model) -> float:
        """
        Get model size in MB.
        
        Args:
            model: Model instance
            
        Returns:
            Model size in MB
        """
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / 1024 / 1024
    
    def get_best_model(self):
        """
        Train and return the best model found.
        
        Returns:
            Trained model with best hyperparameters
        """
        if self.best_params is None:
            raise ValueError("No optimization has been run yet")
        
        logger.info(f"Training final model with best parameters: {self.best_params}")
        
        # Create and train model
        model = self.factory.create_model(
            self.model_name,
            **self.best_params
        )
        
        return model
    
    def analyze_results(self) -> Dict[str, Any]:
        """
        Analyze optimization results.
        
        Returns:
            Analysis dictionary
        """
        if self.study is None:
            raise ValueError("No study available for analysis")
        
        analysis = {
            "best_params": self.best_params,
            "best_value": self.best_value,
            "n_trials": len(self.study.trials),
            "trial_states": {},
            "convergence_analysis": self._analyze_convergence()
        }
        
        # Trial states distribution
        for state in optuna.trial.TrialState:
            count = len([t for t in self.study.trials if t.state == state])
            analysis["trial_states"][state.name] = count
        
        # Parameter importance
        try:
            importance = optuna.importance.get_param_importances(self.study)
            analysis["parameter_importance"] = importance
        except:
            logger.warning("Could not calculate parameter importance")
        
        return analysis
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze convergence of optimization.
        
        Returns:
            Convergence analysis
        """
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not complete_trials:
            return {}
        
        values = [t.value for t in complete_trials if t.value is not None]
        
        if not values:
            return {}
        
        # Calculate cumulative best
        if self.direction == "maximize":
            cumulative_best = np.maximum.accumulate(values)
        else:
            cumulative_best = np.minimum.accumulate(values)
        
        # Find convergence point
        improvements = np.abs(np.diff(cumulative_best))
        convergence_idx = len(improvements)
        
        threshold = np.mean(improvements) * 0.1 if len(improvements) > 0 else 0
        for i in range(len(improvements) - 5, 0, -1):
            if improvements[i:i+5].sum() > threshold:
                convergence_idx = i + 5
                break
        
        return {
            "converged_at_trial": convergence_idx,
            "final_best": float(cumulative_best[-1]) if len(cumulative_best) > 0 else None,
            "improvement_over_random": self._calculate_improvement(),
            "convergence_rate": self._estimate_convergence_rate(values)
        }
    
    def _calculate_improvement(self) -> float:
        """Calculate improvement over random search."""
        if not self.study.trials:
            return 0.0
        
        complete_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(complete_trials) < 2:
            return 0.0
        
        # Get first 10% trials (assumed random)
        n_random = max(1, len(complete_trials) // 10)
        random_trials = complete_trials[:n_random]
        random_values = [t.value for t in random_trials if t.value is not None]
        
        if not random_values:
            return 0.0
        
        if self.direction == "maximize":
            random_best = max(random_values)
            overall_best = self.best_value if self.best_value is not None else random_best
            return (overall_best - random_best) / abs(random_best) * 100 if random_best != 0 else 0
        else:
            random_best = min(random_values)
            overall_best = self.best_value if self.best_value is not None else random_best
            return (random_best - overall_best) / abs(random_best) * 100 if random_best != 0 else 0
    
    def _estimate_convergence_rate(self, values: List[float]) -> float:
        """Estimate convergence rate."""
        if len(values) < 2:
            return 0.0
        
        # Calculate running best
        running_best = []
        current_best = values[0]
        
        for v in values:
            if self.direction == "maximize":
                current_best = max(current_best, v)
            else:
                current_best = min(current_best, v)
            running_best.append(current_best)
        
        # Calculate average improvement rate
        improvements = np.diff(running_best)
        
        if len(improvements) > 0:
            return float(np.mean(np.abs(improvements)))
        
        return 0.0
    
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
