"""
Main Experiment Runner for AG News Text Classification
================================================================================
This module orchestrates the execution of experiments, managing configuration,
resource allocation, and result collection following experimental best practices.

The runner implements automated experiment management with support for
hyperparameter optimization, distributed execution, and comprehensive logging.

References:
    - Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems
    - Bergstra, J., et al. (2011). Algorithms for Hyper-Parameter Optimization

Author: Võ Hải Dũng
License: MIT
"""

import os
import json
import time
import logging
import traceback
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import wandb
import mlflow
import optuna
from omegaconf import OmegaConf, DictConfig

from src.core.registry import Registry
from src.core.factory import Factory
from src.utils.reproducibility import set_seed, ensure_reproducibility
from src.utils.experiment_tracking import ExperimentTracker
from src.utils.logging_config import setup_logging
from src.utils.memory_utils import get_memory_usage, clear_memory
from src.utils.profiling_utils import profile_function
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for experiment execution."""
    
    name: str
    description: str
    model_config: Dict[str, Any]
    data_config: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_config: Dict[str, Any]
    
    # Execution settings
    num_runs: int = 5
    seed: int = 42
    use_gpu: bool = True
    device_ids: List[int] = field(default_factory=lambda: [0])
    
    # Tracking settings
    enable_tracking: bool = True
    tracking_backend: str = "wandb"
    project_name: str = "ag-news-experiments"
    
    # Resource management
    max_workers: int = 4
    memory_limit: int = 32768  # MB
    timeout: int = 3600  # seconds
    
    # Output settings
    save_results: bool = True
    results_dir: str = "outputs/results/experiments"
    save_checkpoints: bool = True
    checkpoint_dir: str = "outputs/models/checkpoints"
    
    # Debugging
    debug_mode: bool = False
    profile_execution: bool = False
    verbose: int = 1


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    
    experiment_id: str
    config: ExperimentConfig
    metrics: Dict[str, float]
    
    # Timing information
    start_time: datetime
    end_time: datetime
    duration: float
    
    # Resource usage
    peak_memory: float
    avg_memory: float
    gpu_memory: Optional[float] = None
    
    # Additional metadata
    model_params: int = 0
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    
    # Error handling
    status: str = "completed"
    error_message: Optional[str] = None
    traceback: Optional[str] = None
    
    # Statistical information
    metrics_per_run: Optional[List[Dict[str, float]]] = None
    std_metrics: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None


class ExperimentRunner:
    """
    Main class for running experiments with comprehensive management.
    
    This class provides:
    - Automated experiment execution
    - Resource management and monitoring
    - Result tracking and aggregation
    - Statistical analysis
    - Error handling and recovery
    """
    
    def __init__(self, base_config_path: Optional[str] = None):
        """
        Initialize experiment runner.
        
        Args:
            base_config_path: Path to base configuration file
        """
        self.base_config_path = base_config_path
        self.registry = Registry()
        self.factory = Factory()
        self.tracker = None
        self.results = []
        
        # Setup logging
        setup_logging(level=logging.INFO)
        
        # Load base configuration if provided
        if base_config_path:
            self.base_config = self._load_config(base_config_path)
        else:
            self.base_config = {}
    
    def run_experiment(
        self,
        config: Union[ExperimentConfig, Dict[str, Any], str],
        override_params: Optional[Dict[str, Any]] = None
    ) -> ExperimentResult:
        """
        Run a single experiment with given configuration.
        
        Args:
            config: Experiment configuration (object, dict, or path)
            override_params: Parameters to override in config
            
        Returns:
            ExperimentResult containing metrics and metadata
        """
        # Prepare configuration
        config = self._prepare_config(config, override_params)
        
        # Initialize tracking
        if config.enable_tracking:
            self.tracker = self._initialize_tracking(config)
        
        # Set up experiment
        experiment_id = self._generate_experiment_id(config)
        logger.info(f"Starting experiment: {experiment_id}")
        logger.info(f"Description: {config.description}")
        
        # Initialize result container
        result = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            metrics={},
            start_time=datetime.now(),
            end_time=datetime.now(),
            duration=0.0,
            peak_memory=0.0,
            avg_memory=0.0
        )
        
        try:
            # Run experiment with resource monitoring
            if config.profile_execution:
                metrics = self._run_with_profiling(config)
            else:
                metrics = self._run_experiment_impl(config)
            
            # Update result
            result.metrics = metrics["aggregated"]
            result.metrics_per_run = metrics.get("per_run", None)
            result.std_metrics = metrics.get("std", None)
            result.confidence_intervals = metrics.get("ci", None)
            result.status = "completed"
            
            # Log results
            self._log_results(result)
            
        except Exception as e:
            # Handle errors
            result.status = "failed"
            result.error_message = str(e)
            result.traceback = traceback.format_exc()
            logger.error(f"Experiment failed: {e}")
            logger.debug(result.traceback)
        
        finally:
            # Record timing and resource usage
            result.end_time = datetime.now()
            result.duration = (result.end_time - result.start_time).total_seconds()
            result.peak_memory = get_memory_usage()["peak"]
            result.avg_memory = get_memory_usage()["average"]
            
            # Clean up resources
            clear_memory()
            
            # Save results
            if config.save_results:
                self._save_results(result)
            
            # Finalize tracking
            if self.tracker:
                self.tracker.finish()
        
        return result
    
    def run_experiments(
        self,
        configs: List[Union[ExperimentConfig, Dict[str, Any], str]],
        parallel: bool = False,
        max_workers: Optional[int] = None
    ) -> List[ExperimentResult]:
        """
        Run multiple experiments sequentially or in parallel.
        
        Args:
            configs: List of experiment configurations
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            List of experiment results
        """
        results = []
        
        if parallel:
            max_workers = max_workers or mp.cpu_count()
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self.run_experiment, config)
                    for config in configs
                ]
                
                for future in tqdm(futures, desc="Running experiments"):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Experiment failed: {e}")
        else:
            for config in tqdm(configs, desc="Running experiments"):
                result = self.run_experiment(config)
                results.append(result)
        
        return results
    
    def run_grid_search(
        self,
        base_config: Union[ExperimentConfig, Dict[str, Any]],
        param_grid: Dict[str, List[Any]],
        metric: str = "accuracy",
        mode: str = "max"
    ) -> Tuple[ExperimentResult, List[ExperimentResult]]:
        """
        Run grid search over hyperparameters.
        
        Args:
            base_config: Base experiment configuration
            param_grid: Dictionary of parameters to search
            metric: Metric to optimize
            mode: Optimization mode ("max" or "min")
            
        Returns:
            Best result and all results
        """
        # Generate configurations
        configs = self._generate_grid_configs(base_config, param_grid)
        
        # Run experiments
        results = self.run_experiments(configs)
        
        # Find best result
        best_result = self._find_best_result(results, metric, mode)
        
        return best_result, results
    
    def run_bayesian_optimization(
        self,
        base_config: Union[ExperimentConfig, Dict[str, Any]],
        search_space: Dict[str, Any],
        n_trials: int = 100,
        metric: str = "accuracy",
        mode: str = "max"
    ) -> Tuple[ExperimentResult, optuna.Study]:
        """
        Run Bayesian optimization for hyperparameter tuning.
        
        Args:
            base_config: Base experiment configuration
            search_space: Search space definition for Optuna
            n_trials: Number of optimization trials
            metric: Metric to optimize
            mode: Optimization mode ("max" or "min")
            
        Returns:
            Best result and Optuna study object
        """
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize" if mode == "max" else "minimize",
            study_name=f"optuna_{base_config.name}",
            load_if_exists=True
        )
        
        # Define objective function
        def objective(trial):
            # Sample hyperparameters
            params = self._sample_params_from_trial(trial, search_space)
            
            # Run experiment
            result = self.run_experiment(base_config, override_params=params)
            
            # Return metric value
            return result.metrics.get(metric, float("-inf"))
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        # Get best result
        best_params = study.best_params
        best_result = self.run_experiment(base_config, override_params=best_params)
        
        return best_result, study
    
    def _run_experiment_impl(self, config: ExperimentConfig) -> Dict[str, Any]:
        """
        Actual implementation of experiment execution.
        
        Args:
            config: Experiment configuration
            
        Returns:
            Dictionary of metrics
        """
        all_metrics = []
        
        for run_idx in range(config.num_runs):
            # Set seed for reproducibility
            run_seed = config.seed + run_idx
            set_seed(run_seed)
            ensure_reproducibility()
            
            logger.info(f"Run {run_idx + 1}/{config.num_runs} (seed={run_seed})")
            
            # Load data
            data = self._load_data(config.data_config)
            
            # Create model
            model = self._create_model(config.model_config)
            
            # Create trainer
            trainer = self._create_trainer(config.training_config, model)
            
            # Train model
            trainer.train(data["train"], data["validation"])
            
            # Evaluate model
            metrics = self._evaluate_model(
                model,
                data["test"],
                config.evaluation_config
            )
            
            all_metrics.append(metrics)
            
            # Track metrics
            if self.tracker:
                self.tracker.log_metrics(metrics, step=run_idx)
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(all_metrics)
        
        return {
            "aggregated": aggregated_metrics["mean"],
            "std": aggregated_metrics["std"],
            "per_run": all_metrics,
            "ci": aggregated_metrics["confidence_intervals"]
        }
    
    def _aggregate_metrics(
        self,
        metrics_list: List[Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Aggregate metrics across multiple runs.
        
        Args:
            metrics_list: List of metrics from each run
            
        Returns:
            Aggregated statistics
        """
        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(metrics_list)
        
        # Calculate statistics
        mean_metrics = df.mean().to_dict()
        std_metrics = df.std().to_dict()
        
        # Calculate confidence intervals
        confidence_intervals = {}
        for metric in df.columns:
            values = df[metric].values
            ci_lower, ci_upper = self._bootstrap_confidence_interval(values)
            confidence_intervals[metric] = (ci_lower, ci_upper)
        
        return {
            "mean": mean_metrics,
            "std": std_metrics,
            "confidence_intervals": confidence_intervals
        }
    
    def _bootstrap_confidence_interval(
        self,
        values: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval.
        
        Args:
            values: Array of values
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Lower and upper bounds of confidence interval
        """
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = (1 - confidence) / 2
        lower = np.percentile(bootstrap_means, alpha * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
        
        return lower, upper
    
    def _prepare_config(
        self,
        config: Union[ExperimentConfig, Dict[str, Any], str],
        override_params: Optional[Dict[str, Any]] = None
    ) -> ExperimentConfig:
        """
        Prepare and validate experiment configuration.
        
        Args:
            config: Input configuration
            override_params: Parameters to override
            
        Returns:
            Validated ExperimentConfig object
        """
        # Handle different config types
        if isinstance(config, str):
            config = self._load_config(config)
        
        if isinstance(config, dict):
            # Merge with base config
            merged_config = {**self.base_config, **config}
            
            # Apply overrides
            if override_params:
                merged_config = self._deep_update(merged_config, override_params)
            
            # Create ExperimentConfig object
            config = ExperimentConfig(**merged_config)
        
        return config
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        path = Path(config_path)
        
        if path.suffix == ".yaml" or path.suffix == ".yml":
            with open(path, "r") as f:
                return yaml.safe_load(f)
        elif path.suffix == ".json":
            with open(path, "r") as f:
                return json.load(f)
        else:
            # Use OmegaConf for advanced config loading
            return OmegaConf.to_container(OmegaConf.load(path))
    
    def _deep_update(self, base: Dict, update: Dict) -> Dict:
        """Deep update dictionary."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = self._deep_update(base[key], value)
            else:
                base[key] = value
        return base
    
    def _generate_experiment_id(self, config: ExperimentConfig) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_slug = config.name.lower().replace(" ", "_")
        return f"{name_slug}_{timestamp}"
    
    def _initialize_tracking(self, config: ExperimentConfig) -> ExperimentTracker:
        """Initialize experiment tracking."""
        return ExperimentTracker(
            backend=config.tracking_backend,
            project=config.project_name,
            name=config.name,
            config=asdict(config)
        )
    
    def _save_results(self, result: ExperimentResult):
        """Save experiment results to disk."""
        results_dir = Path(result.config.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        result_path = results_dir / f"{result.experiment_id}.json"
        
        # Convert to serializable format
        result_dict = asdict(result)
        result_dict["start_time"] = result.start_time.isoformat()
        result_dict["end_time"] = result.end_time.isoformat()
        
        with open(result_path, "w") as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Results saved to {result_path}")
    
    def _log_results(self, result: ExperimentResult):
        """Log experiment results."""
        logger.info(f"Experiment completed: {result.experiment_id}")
        logger.info(f"Duration: {result.duration:.2f} seconds")
        logger.info(f"Peak memory: {result.peak_memory:.2f} MB")
        logger.info("Metrics:")
        for metric, value in result.metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
