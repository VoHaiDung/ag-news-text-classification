"""
Experiment Tracker for AG News Text Classification
================================================================================
This module provides comprehensive experiment tracking functionality with
support for multiple backends and advanced features for managing ML experiments.

The tracker implements best practices for experiment management including
versioning, artifact storage, and metric tracking.

References:
    - Zaharia, M., et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow
    - Biewald, L. (2020). Experiment Tracking with Weights and Biases

Author: Võ Hải Dũng
License: MIT
"""

import os
import json
import logging
import hashlib
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import pickle

import numpy as np
import pandas as pd
import wandb
import mlflow
from tensorboardX import SummaryWriter
import neptune.new as neptune

from src.utils.io_utils import save_json, load_json

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    Unified experiment tracking interface supporting multiple backends.
    
    This class provides:
    - Multi-backend support (W&B, MLflow, TensorBoard, Neptune)
    - Automatic artifact management
    - Metric tracking and visualization
    - Hyperparameter logging
    - Model checkpointing
    """
    
    def __init__(
        self,
        backend: str = "wandb",
        project: str = "ag-news-classification",
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        resume: bool = False,
        offline: bool = False
    ):
        """
        Initialize experiment tracker.
        
        Args:
            backend: Tracking backend ("wandb", "mlflow", "tensorboard", "neptune")
            project: Project name
            name: Experiment name
            config: Configuration dictionary
            tags: List of tags for the experiment
            resume: Whether to resume from previous run
            offline: Whether to run in offline mode
        """
        self.backend = backend
        self.project = project
        self.name = name or self._generate_experiment_name()
        self.config = config or {}
        self.tags = tags or []
        self.resume = resume
        self.offline = offline
        
        self.run = None
        self.run_id = None
        self.artifacts_dir = Path(f"outputs/tracking/{self.name}")
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backend
        self._initialize_backend()
        
        # Track initialization
        self.log_config(self.config)
        
        logger.info(f"Initialized {backend} tracker for experiment: {self.name}")
    
    def _initialize_backend(self):
        """Initialize the selected tracking backend."""
        if self.backend == "wandb":
            self._initialize_wandb()
        elif self.backend == "mlflow":
            self._initialize_mlflow()
        elif self.backend == "tensorboard":
            self._initialize_tensorboard()
        elif self.backend == "neptune":
            self._initialize_neptune()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")
    
    def _initialize_wandb(self):
        """Initialize Weights & Biases tracking."""
        wandb_config = {
            "project": self.project,
            "name": self.name,
            "config": self.config,
            "tags": self.tags,
            "resume": "allow" if self.resume else False,
            "mode": "offline" if self.offline else "online"
        }
        
        self.run = wandb.init(**wandb_config)
        self.run_id = self.run.id
    
    def _initialize_mlflow(self):
        """Initialize MLflow tracking."""
        mlflow.set_experiment(self.project)
        
        if self.resume and self.run_id:
            self.run = mlflow.start_run(run_id=self.run_id)
        else:
            self.run = mlflow.start_run(run_name=self.name, tags={"tags": ",".join(self.tags)})
            self.run_id = self.run.info.run_id
        
        # Log parameters
        for key, value in self.config.items():
            if isinstance(value, (str, int, float, bool)):
                mlflow.log_param(key, value)
    
    def _initialize_tensorboard(self):
        """Initialize TensorBoard tracking."""
        log_dir = self.artifacts_dir / "tensorboard"
        self.run = SummaryWriter(log_dir=str(log_dir))
        self.run_id = self.name
    
    def _initialize_neptune(self):
        """Initialize Neptune tracking."""
        self.run = neptune.init_run(
            project=self.project,
            name=self.name,
            tags=self.tags
        )
        self.run_id = self.run["sys/id"].fetch()
        
        # Log parameters
        self.run["parameters"] = self.config
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int]],
        step: Optional[int] = None,
        epoch: Optional[int] = None
    ):
        """
        Log metrics to tracking backend.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/iteration
            epoch: Current epoch
        """
        # Add prefix for organization
        if epoch is not None:
            metrics = {f"epoch/{k}": v for k, v in metrics.items()}
        
        if self.backend == "wandb":
            wandb.log(metrics, step=step)
        
        elif self.backend == "mlflow":
            for key, value in metrics.items():
                mlflow.log_metric(key, value, step=step or 0)
        
        elif self.backend == "tensorboard":
            for key, value in metrics.items():
                self.run.add_scalar(key, value, global_step=step or 0)
        
        elif self.backend == "neptune":
            for key, value in metrics.items():
                self.run[f"metrics/{key}"].log(value, step=step)
        
        # Also save locally
        self._save_metrics_locally(metrics, step, epoch)
    
    def log_config(self, config: Dict[str, Any]):
        """
        Log configuration/hyperparameters.
        
        Args:
            config: Configuration dictionary
        """
        if self.backend == "wandb":
            wandb.config.update(config)
        
        elif self.backend == "mlflow":
            for key, value in self._flatten_dict(config).items():
                if isinstance(value, (str, int, float, bool)):
                    mlflow.log_param(key, value)
        
        elif self.backend == "tensorboard":
            # TensorBoard doesn't have native config logging
            config_str = json.dumps(config, indent=2)
            self.run.add_text("config", f"```json\n{config_str}\n```", global_step=0)
        
        elif self.backend == "neptune":
            self.run["config"] = config
        
        # Save config locally
        save_json(config, self.artifacts_dir / "config.json")
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        artifact_type: str = "model",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an artifact (model, dataset, etc.).
        
        Args:
            artifact_path: Path to artifact file
            artifact_type: Type of artifact
            metadata: Additional metadata
        """
        artifact_path = Path(artifact_path)
        
        if not artifact_path.exists():
            logger.warning(f"Artifact not found: {artifact_path}")
            return
        
        if self.backend == "wandb":
            artifact = wandb.Artifact(
                name=f"{self.name}_{artifact_type}",
                type=artifact_type,
                metadata=metadata
            )
            artifact.add_file(str(artifact_path))
            wandb.log_artifact(artifact)
        
        elif self.backend == "mlflow":
            mlflow.log_artifact(str(artifact_path))
            if metadata:
                mlflow.log_dict(metadata, f"{artifact_path.stem}_metadata.json")
        
        elif self.backend == "tensorboard":
            # TensorBoard doesn't support artifact logging directly
            # Save reference in text
            self.run.add_text(
                f"artifact/{artifact_type}",
                f"Path: {artifact_path}\nMetadata: {metadata}",
                global_step=0
            )
        
        elif self.backend == "neptune":
            self.run[f"artifacts/{artifact_type}"].upload(str(artifact_path))
            if metadata:
                self.run[f"artifacts/{artifact_type}/metadata"] = metadata
    
    def log_model(
        self,
        model,
        model_name: str,
        framework: str = "pytorch",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a trained model.
        
        Args:
            model: Model object
            model_name: Name for the model
            framework: ML framework ("pytorch", "tensorflow", "sklearn")
            metadata: Additional metadata
        """
        model_path = self.artifacts_dir / f"{model_name}.pkl"
        
        # Save model based on framework
        if framework == "pytorch":
            import torch
            torch.save(model.state_dict(), model_path)
        elif framework == "tensorflow":
            model.save(model_path)
        elif framework == "sklearn":
            import joblib
            joblib.dump(model, model_path)
        else:
            # Generic pickle
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        
        # Log as artifact
        self.log_artifact(model_path, "model", metadata)
        
        # Framework-specific logging
        if self.backend == "mlflow" and framework == "pytorch":
            mlflow.pytorch.log_model(model, model_name)
        elif self.backend == "mlflow" and framework == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
    
    def log_dataset(
        self,
        dataset: Union[pd.DataFrame, np.ndarray, Dict[str, Any]],
        name: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log a dataset.
        
        Args:
            dataset: Dataset object
            name: Name for the dataset
            metadata: Additional metadata
        """
        dataset_path = self.artifacts_dir / f"{name}_dataset.pkl"
        
        # Save dataset
        if isinstance(dataset, pd.DataFrame):
            dataset.to_pickle(dataset_path)
        elif isinstance(dataset, np.ndarray):
            np.save(dataset_path, dataset)
        else:
            with open(dataset_path, "wb") as f:
                pickle.dump(dataset, f)
        
        # Log as artifact
        self.log_artifact(dataset_path, "dataset", metadata)
        
        # Log dataset statistics
        if isinstance(dataset, pd.DataFrame):
            stats = {
                "shape": dataset.shape,
                "columns": list(dataset.columns),
                "dtypes": dataset.dtypes.to_dict()
            }
            self.log_metrics({f"dataset/{name}/{k}": str(v) for k, v in stats.items()})
    
    def log_figure(
        self,
        figure,
        name: str,
        step: Optional[int] = None
    ):
        """
        Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure object
            name: Name for the figure
            step: Current step
        """
        figure_path = self.artifacts_dir / f"{name}.png"
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
        
        if self.backend == "wandb":
            wandb.log({name: wandb.Image(str(figure_path))}, step=step)
        
        elif self.backend == "mlflow":
            mlflow.log_artifact(str(figure_path))
        
        elif self.backend == "tensorboard":
            self.run.add_figure(name, figure, global_step=step or 0)
        
        elif self.backend == "neptune":
            self.run[f"figures/{name}"].upload(str(figure_path))
    
    def log_table(
        self,
        table: pd.DataFrame,
        name: str,
        step: Optional[int] = None
    ):
        """
        Log a pandas DataFrame as a table.
        
        Args:
            table: Pandas DataFrame
            name: Name for the table
            step: Current step
        """
        table_path = self.artifacts_dir / f"{name}.csv"
        table.to_csv(table_path, index=False)
        
        if self.backend == "wandb":
            wandb.log({name: wandb.Table(dataframe=table)}, step=step)
        
        elif self.backend == "mlflow":
            mlflow.log_artifact(str(table_path))
        
        elif self.backend == "tensorboard":
            # Convert to markdown for TensorBoard
            table_md = table.to_markdown()
            self.run.add_text(name, table_md, global_step=step or 0)
        
        elif self.backend == "neptune":
            self.run[f"tables/{name}"].upload(str(table_path))
    
    def log_text(
        self,
        text: str,
        name: str,
        step: Optional[int] = None
    ):
        """
        Log text data.
        
        Args:
            text: Text to log
            name: Name for the text
            step: Current step
        """
        if self.backend == "wandb":
            wandb.log({name: wandb.Html(text)}, step=step)
        
        elif self.backend == "mlflow":
            mlflow.log_text(text, f"{name}.txt")
        
        elif self.backend == "tensorboard":
            self.run.add_text(name, text, global_step=step or 0)
        
        elif self.backend == "neptune":
            self.run[f"text/{name}"] = text
    
    def log_code(self, code_dir: Optional[str] = None):
        """
        Log source code.
        
        Args:
            code_dir: Directory containing code (defaults to current directory)
        """
        code_dir = code_dir or "."
        
        if self.backend == "wandb":
            wandb.run.log_code(code_dir)
        
        elif self.backend == "mlflow":
            # MLflow doesn't have direct code logging
            # Create archive and log as artifact
            import shutil
            archive_path = self.artifacts_dir / "code.zip"
            shutil.make_archive(archive_path.stem, "zip", code_dir)
            mlflow.log_artifact(str(archive_path))
        
        elif self.backend == "neptune":
            self.run["source_code"].upload_files(code_dir)
    
    def set_tags(self, tags: List[str]):
        """
        Set or update experiment tags.
        
        Args:
            tags: List of tags
        """
        if self.backend == "wandb":
            wandb.run.tags = tags
        
        elif self.backend == "mlflow":
            for tag in tags:
                mlflow.set_tag("tag", tag)
        
        elif self.backend == "neptune":
            self.run["sys/tags"].add(tags)
    
    def finish(self):
        """Finish and close the experiment run."""
        if self.backend == "wandb":
            wandb.finish()
        
        elif self.backend == "mlflow":
            mlflow.end_run()
        
        elif self.backend == "tensorboard":
            self.run.close()
        
        elif self.backend == "neptune":
            self.run.stop()
        
        logger.info(f"Finished tracking experiment: {self.name}")
    
    def _generate_experiment_name(self) -> str:
        """Generate unique experiment name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"experiment_{timestamp}"
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = "."
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _save_metrics_locally(
        self,
        metrics: Dict[str, Any],
        step: Optional[int],
        epoch: Optional[int]
    ):
        """Save metrics to local file for backup."""
        metrics_file = self.artifacts_dir / "metrics.jsonl"
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "epoch": epoch,
            "metrics": metrics
        }
        
        with open(metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    
    def get_metric_history(self, metric_name: str) -> List[Tuple[int, float]]:
        """
        Get history of a specific metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            List of (step, value) tuples
        """
        history = []
        
        if self.backend == "wandb":
            history_data = wandb.run.history(keys=[metric_name])
            history = [(i, row[metric_name]) for i, row in history_data.iterrows()]
        
        elif self.backend == "mlflow":
            client = mlflow.tracking.MlflowClient()
            metric_history = client.get_metric_history(self.run_id, metric_name)
            history = [(m.step, m.value) for m in metric_history]
        
        elif self.backend == "neptune":
            metric_series = self.run[f"metrics/{metric_name}"].fetch_values()
            history = [(v["step"], v["value"]) for v in metric_series]
        
        return history
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: List of metrics to compare
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for exp_id in experiment_ids:
            exp_data = {"experiment_id": exp_id}
            
            if self.backend == "wandb":
                api = wandb.Api()
                run = api.run(f"{self.project}/{exp_id}")
                for metric in metrics:
                    exp_data[metric] = run.summary.get(metric)
            
            elif self.backend == "mlflow":
                client = mlflow.tracking.MlflowClient()
                run = client.get_run(exp_id)
                for metric in metrics:
                    exp_data[metric] = run.data.metrics.get(metric)
            
            comparison_data.append(exp_data)
        
        return pd.DataFrame(comparison_data)
