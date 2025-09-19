"""
Experiment tracking utilities for AG News Text Classification Framework.

Provides tools for tracking experiments, hyperparameters, metrics, and artifacts
to ensure reproducible and comparable research results.

References:
    - Biewald, L. (2020). "Experiment Tracking with Weights and Biases". arXiv.
    - MLflow Documentation: https://mlflow.org/docs/latest/index.html

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List, Union, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
import hashlib
import shutil
import warnings

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Container for experiment configuration."""
    
    name: str
    project: str = "ag-news-classification"
    tags: List[str] = None
    description: str = ""
    hyperparameters: Dict[str, Any] = None
    config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.hyperparameters is None:
            self.hyperparameters = {}
        if self.config is None:
            self.config = {}


class ExperimentTracker:
    """
    Comprehensive experiment tracker for ML research.
    
    Tracks hyperparameters, metrics, artifacts, and models across experiments
    following best practices from MLOps literature.
    """
    
    def __init__(
        self,
        experiment_name: str,
        project_name: str = "ag-news-classification",
        base_dir: Union[str, Path] = "outputs/experiments",
        use_mlflow: bool = False,
        use_wandb: bool = False,
        use_tensorboard: bool = True,
    ):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_name: Name of the experiment
            project_name: Project name
            base_dir: Base directory for experiment data
            use_mlflow: Whether to use MLflow
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
        """
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.base_dir = Path(base_dir)
        
        # Create experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{experiment_name}_{timestamp}"
        
        # Setup experiment directory
        self.experiment_dir = self.base_dir / self.experiment_id
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.metrics_dir = self.experiment_dir / "metrics"
        self.artifacts_dir = self.experiment_dir / "artifacts"
        self.models_dir = self.experiment_dir / "models"
        self.logs_dir = self.experiment_dir / "logs"
        
        for dir_path in [self.metrics_dir, self.artifacts_dir, self.models_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize tracking backends
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        self.mlflow_run = None
        self.wandb_run = None
        self.tensorboard_writer = None
        
        self._init_backends()
        
        # Tracking data
        self.hyperparameters = {}
        self.metrics_history = []
        self.artifacts = []
        self.models = []
        
        logger.info(f"Experiment tracker initialized: {self.experiment_id}")
    
    def _init_backends(self):
        """Initialize tracking backends."""
        # MLflow
        if self.use_mlflow:
            try:
                import mlflow
                mlflow.set_experiment(self.project_name)
                self.mlflow_run = mlflow.start_run(run_name=self.experiment_name)
                logger.info("MLflow tracking enabled")
            except ImportError:
                logger.warning("MLflow not installed, disabling MLflow tracking")
                self.use_mlflow = False
        
        # Weights & Biases
        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    dir=str(self.experiment_dir),
                )
                logger.info("Weights & Biases tracking enabled")
            except ImportError:
                logger.warning("wandb not installed, disabling W&B tracking")
                self.use_wandb = False
        
        # TensorBoard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tensorboard_writer = SummaryWriter(
                    log_dir=str(self.logs_dir / "tensorboard")
                )
                logger.info("TensorBoard tracking enabled")
            except ImportError:
                logger.warning("TensorBoard not available, disabling TensorBoard tracking")
                self.use_tensorboard = False
    
    def log_hyperparameters(self, hyperparameters: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hyperparameters: Dictionary of hyperparameters
        """
        self.hyperparameters.update(hyperparameters)
        
        # Save to file
        hparams_file = self.experiment_dir / "hyperparameters.json"
        with open(hparams_file, "w") as f:
            json.dump(self.hyperparameters, f, indent=2, default=str)
        
        # Log to backends
        if self.use_mlflow:
            import mlflow
            for key, value in hyperparameters.items():
                mlflow.log_param(key, value)
        
        if self.use_wandb:
            import wandb
            wandb.config.update(hyperparameters)
        
        if self.use_tensorboard and self.tensorboard_writer:
            # TensorBoard hyperparameter logging
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer.add_hparams(
                hyperparameters,
                {"dummy": 0}  # TensorBoard requires at least one metric
            )
        
        logger.debug(f"Logged {len(hyperparameters)} hyperparameters")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ):
        """
        Log metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Training step
            epoch: Training epoch
        """
        # Add metadata
        log_entry = {
            "metrics": metrics,
            "step": step,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
        }
        
        self.metrics_history.append(log_entry)
        
        # Save to file
        metrics_file = self.metrics_dir / f"metrics_step_{step}.json"
        with open(metrics_file, "w") as f:
            json.dump(log_entry, f, indent=2)
        
        # Log to backends
        if self.use_mlflow:
            import mlflow
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step)
        
        if self.use_wandb:
            import wandb
            wandb.log(metrics, step=step)
        
        if self.use_tensorboard and self.tensorboard_writer:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step or 0)
        
        # Log to console
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        logger.info(f"[Step {step}] {metric_str}")
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: str = "file",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log artifact.
        
        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact
            metadata: Optional metadata
        """
        artifact_path = Path(artifact_path)
        
        if not artifact_path.exists():
            logger.warning(f"Artifact not found: {artifact_path}")
            return
        
        # Copy to artifacts directory
        dest_path = self.artifacts_dir / artifact_path.name
        if artifact_path.is_dir():
            shutil.copytree(artifact_path, dest_path, dirs_exist_ok=True)
        else:
            shutil.copy2(artifact_path, dest_path)
        
        # Track artifact
        artifact_info = {
            "name": artifact_path.name,
            "type": artifact_type,
            "path": str(dest_path),
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.artifacts.append(artifact_info)
        
        # Log to backends
        if self.use_mlflow:
            import mlflow
            mlflow.log_artifact(str(artifact_path))
        
        if self.use_wandb:
            import wandb
            wandb.save(str(artifact_path))
        
        logger.debug(f"Logged artifact: {artifact_path.name}")
    
    def log_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
    ):
        """
        Log model checkpoint.
        
        Args:
            model: Model to save
            model_name: Model name
            epoch: Training epoch
            metrics: Associated metrics
        """
        # Create model filename
        if epoch is not None:
            filename = f"{model_name}_epoch_{epoch}.pt"
        else:
            filename = f"{model_name}.pt"
        
        model_path = self.models_dir / filename
        
        # Save model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        
        torch.save(checkpoint, model_path)
        
        # Track model
        model_info = {
            "name": model_name,
            "path": str(model_path),
            "epoch": epoch,
            "metrics": metrics,
            "timestamp": checkpoint["timestamp"],
        }
        self.models.append(model_info)
        
        # Log to backends
        if self.use_mlflow:
            import mlflow
            mlflow.pytorch.log_model(model, model_name)
        
        if self.use_wandb:
            import wandb
            wandb.save(str(model_path))
        
        logger.info(f"Logged model: {filename}")
    
    def log_figure(
        self,
        figure: plt.Figure,
        name: str,
        step: Optional[int] = None,
    ):
        """
        Log matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            name: Figure name
            step: Training step
        """
        # Save figure
        figure_path = self.artifacts_dir / f"{name}.png"
        figure.savefig(figure_path, dpi=150, bbox_inches="tight")
        
        # Log as artifact
        self.log_artifact(figure_path, artifact_type="figure")
        
        # Log to TensorBoard
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_figure(name, figure, step or 0)
        
        plt.close(figure)
        logger.debug(f"Logged figure: {name}")
    
    def create_summary(self) -> Dict[str, Any]:
        """
        Create experiment summary.
        
        Returns:
            Summary dictionary
        """
        # Get best metrics
        best_metrics = {}
        if self.metrics_history:
            all_metrics = {}
            for entry in self.metrics_history:
                for key, value in entry["metrics"].items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)
            
            for key, values in all_metrics.items():
                best_metrics[key] = {
                    "best": max(values) if "loss" not in key else min(values),
                    "last": values[-1],
                    "mean": np.mean(values),
                    "std": np.std(values),
                }
        
        summary = {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "project_name": self.project_name,
            "hyperparameters": self.hyperparameters,
            "num_metrics_logged": len(self.metrics_history),
            "num_artifacts": len(self.artifacts),
            "num_models": len(self.models),
            "best_metrics": best_metrics,
            "duration": self._get_duration(),
        }
        
        # Save summary
        summary_file = self.experiment_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        
        return summary
    
    def _get_duration(self) -> str:
        """Get experiment duration."""
        if not self.metrics_history:
            return "0:00:00"
        
        start_time = datetime.fromisoformat(self.metrics_history[0]["timestamp"])
        end_time = datetime.fromisoformat(self.metrics_history[-1]["timestamp"])
        duration = end_time - start_time
        
        return str(duration)
    
    def compare_with(self, other_experiment_id: str) -> pd.DataFrame:
        """
        Compare with another experiment.
        
        Args:
            other_experiment_id: ID of experiment to compare with
            
        Returns:
            Comparison DataFrame
        """
        other_dir = self.base_dir / other_experiment_id
        
        if not other_dir.exists():
            logger.error(f"Experiment not found: {other_experiment_id}")
            return pd.DataFrame()
        
        # Load other experiment summary
        other_summary_file = other_dir / "summary.json"
        if not other_summary_file.exists():
            logger.error(f"Summary not found for: {other_experiment_id}")
            return pd.DataFrame()
        
        with open(other_summary_file) as f:
            other_summary = json.load(f)
        
        # Create comparison
        this_summary = self.create_summary()
        
        comparison_data = []
        
        # Compare hyperparameters
        for key in set(list(this_summary["hyperparameters"].keys()) + 
                      list(other_summary["hyperparameters"].keys())):
            comparison_data.append({
                "Type": "Hyperparameter",
                "Name": key,
                "This Experiment": this_summary["hyperparameters"].get(key, "N/A"),
                "Other Experiment": other_summary["hyperparameters"].get(key, "N/A"),
            })
        
        # Compare metrics
        for key in set(list(this_summary["best_metrics"].keys()) + 
                      list(other_summary["best_metrics"].keys())):
            this_best = this_summary["best_metrics"].get(key, {}).get("best", "N/A")
            other_best = other_summary["best_metrics"].get(key, {}).get("best", "N/A")
            
            comparison_data.append({
                "Type": "Metric",
                "Name": key,
                "This Experiment": this_best,
                "Other Experiment": other_best,
            })
        
        return pd.DataFrame(comparison_data)
    
    def close(self):
        """Close experiment tracking."""
        # Create final summary
        summary = self.create_summary()
        
        # Close backends
        if self.use_mlflow:
            import mlflow
            mlflow.end_run()
        
        if self.use_wandb:
            import wandb
            wandb.finish()
        
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        logger.info(f"Experiment closed: {self.experiment_id}")
        logger.info(f"Results saved to: {self.experiment_dir}")


# Global experiment tracker instance
_experiment_tracker: Optional[ExperimentTracker] = None


def create_experiment(
    name: str,
    project: str = "ag-news-classification",
    **kwargs
) -> ExperimentTracker:
    """
    Create new experiment.
    
    Args:
        name: Experiment name
        project: Project name
        **kwargs: Additional arguments for ExperimentTracker
        
    Returns:
        Experiment tracker instance
    """
    global _experiment_tracker
    
    _experiment_tracker = ExperimentTracker(
        experiment_name=name,
        project_name=project,
        **kwargs
    )
    
    return _experiment_tracker


def get_experiment_tracker() -> Optional[ExperimentTracker]:
    """Get current experiment tracker."""
    return _experiment_tracker


def log_hyperparameters(hyperparameters: Dict[str, Any]):
    """Log hyperparameters to current experiment."""
    if _experiment_tracker:
        _experiment_tracker.log_hyperparameters(hyperparameters)


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    epoch: Optional[int] = None,
):
    """Log metrics to current experiment."""
    if _experiment_tracker:
        _experiment_tracker.log_metrics(metrics, step, epoch)


def log_artifacts(
    artifact_path: Union[str, Path],
    artifact_type: str = "file",
    metadata: Optional[Dict[str, Any]] = None,
):
    """Log artifact to current experiment."""
    if _experiment_tracker:
        _experiment_tracker.log_artifact(artifact_path, artifact_type, metadata)


def log_model(
    model: torch.nn.Module,
    model_name: str,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
):
    """Log model to current experiment."""
    if _experiment_tracker:
        _experiment_tracker.log_model(model, model_name, epoch, metrics)


def get_experiment_id() -> Optional[str]:
    """Get current experiment ID."""
    if _experiment_tracker:
        return _experiment_tracker.experiment_id
    return None


def compare_experiments(
    experiment_ids: List[str],
    base_dir: Union[str, Path] = "outputs/experiments",
) -> pd.DataFrame:
    """
    Compare multiple experiments.
    
    Args:
        experiment_ids: List of experiment IDs
        base_dir: Base directory for experiments
        
    Returns:
        Comparison DataFrame
    """
    base_dir = Path(base_dir)
    
    comparison_data = []
    
    for exp_id in experiment_ids:
        exp_dir = base_dir / exp_id
        summary_file = exp_dir / "summary.json"
        
        if not summary_file.exists():
            logger.warning(f"Summary not found for: {exp_id}")
            continue
        
        with open(summary_file) as f:
            summary = json.load(f)
        
        # Extract key metrics
        row = {
            "Experiment ID": exp_id,
            "Name": summary.get("experiment_name", "N/A"),
        }
        
        # Add hyperparameters
        for key, value in summary.get("hyperparameters", {}).items():
            row[f"HP_{key}"] = value
        
        # Add best metrics
        for key, value in summary.get("best_metrics", {}).items():
            row[f"Metric_{key}"] = value.get("best", "N/A")
        
        comparison_data.append(row)
    
    return pd.DataFrame(comparison_data)


# Export public API
__all__ = [
    "ExperimentTracker",
    "ExperimentConfig",
    "create_experiment",
    "get_experiment_tracker",
    "log_hyperparameters",
    "log_metrics",
    "log_artifacts",
    "log_model",
    "get_experiment_id",
    "compare_experiments",
]
