"""
Weights & Biases Logger Callback for Experiment Tracking
========================================================

This module implements W&B logging for comprehensive experiment tracking.
Based on W&B best practices and integration patterns from:
- Biewald (2020): Experiment Tracking with Weights and Biases
- MLOps community best practices

The logger provides:
1. Automatic metric logging and visualization
2. Hyperparameter tracking and sweeps
3. Model checkpointing and versioning
4. Artifact management
5. System metrics monitoring
6. Collaborative experiment tracking

Author: Võ Hải Dũng
License: MIT
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger(__name__)


class WandBLogger:
    """
    Weights & Biases logging callback for experiment tracking.
    
    Provides comprehensive experiment tracking with:
    - Automatic metric logging
    - Hyperparameter tracking
    - Model versioning
    - Artifact management
    - System resource monitoring
    - Collaborative features
    
    Implementation follows W&B best practices for ML experiments.
    """
    
    def __init__(
        self,
        project: str = "ag-news-classification",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        config: Optional[Dict[str, Any]] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = "train",
        mode: str = "online",
        dir: Optional[Union[str, Path]] = None,
        resume: Union[bool, str] = False,
        reinit: bool = False,
        log_model: bool = True,
        log_code: bool = True,
        log_gradients: bool = False,
        log_freq: int = 100,
        log_graph: bool = True,
        save_checkpoints: bool = True,
        checkpoint_freq: int = 1,
        monitor_metric: str = "val_loss",
        monitor_mode: str = "min"
    ):
        """
        Initialize W&B logger.
        
        Args:
            project: W&B project name
            entity: W&B entity (team/username)
            name: Run name
            tags: List of tags for the run
            config: Configuration dictionary
            group: Group name for organizing runs
            job_type: Type of job (train/eval/sweep)
            mode: Logging mode (online/offline/disabled)
            dir: Directory to save W&B files
            resume: Resume from previous run
            reinit: Allow reinitializing run
            log_model: Whether to log model artifacts
            log_code: Whether to log code
            log_gradients: Whether to log gradients
            log_freq: Logging frequency (in steps)
            log_graph: Whether to log model graph
            save_checkpoints: Whether to save checkpoints
            checkpoint_freq: Checkpoint frequency (in epochs)
            monitor_metric: Metric to monitor for best model
            monitor_mode: Monitoring mode (min/max)
        """
        if not WANDB_AVAILABLE:
            logger.warning("W&B not installed. Install with: pip install wandb")
            self.disabled = True
            return
        
        self.disabled = False
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.config = config or {}
        self.group = group
        self.job_type = job_type
        self.mode = mode
        self.dir = Path(dir) if dir else Path("outputs/wandb")
        self.resume = resume
        self.reinit = reinit
        
        # Logging configuration
        self.log_model = log_model
        self.log_code = log_code
        self.log_gradients = log_gradients
        self.log_freq = log_freq
        self.log_graph = log_graph
        
        # Checkpoint configuration
        self.save_checkpoints = save_checkpoints
        self.checkpoint_freq = checkpoint_freq
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        
        # State tracking
        self.run = None
        self.global_step = 0
        self.epoch = 0
        self.best_metric = None
        self.graph_logged = False
        
        # Create directory
        self.dir.mkdir(parents=True, exist_ok=True)
    
    def on_train_begin(
        self,
        model: Optional[torch.nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Called at the beginning of training.
        
        Initializes W&B run and logs configuration.
        
        Args:
            model: Model being trained
            config: Training configuration
        """
        if self.disabled:
            return
        
        # Update config if provided
        if config:
            self.config.update(self._extract_config(config))
        
        # Initialize W&B run
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            tags=self.tags,
            config=self.config,
            group=self.group,
            job_type=self.job_type,
            mode=self.mode,
            dir=str(self.dir),
            resume=self.resume,
            reinit=self.reinit
        )
        
        # Log code if requested
        if self.log_code:
            wandb.run.log_code("src")
        
        # Watch model for gradient logging
        if model and self.log_gradients:
            wandb.watch(
                model,
                log="all" if self.log_gradients else "gradients",
                log_freq=self.log_freq
            )
        
        logger.info(f"W&B run initialized: {wandb.run.name} ({wandb.run.id})")
    
    def on_epoch_begin(self, epoch: int, **kwargs):
        """
        Called at the beginning of an epoch.
        
        Args:
            epoch: Current epoch number
        """
        if self.disabled:
            return
        
        self.epoch = epoch
        wandb.log({"epoch": epoch}, step=self.global_step)
    
    def on_batch_end(
        self,
        batch: int,
        loss: float,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        inputs: Optional[torch.Tensor] = None,
        outputs: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Called at the end of a batch.
        
        Args:
            batch: Batch index
            loss: Batch loss value
            model: Model being trained
            optimizer: Optimizer being used
            inputs: Input batch
            outputs: Model outputs
        """
        if self.disabled:
            return
        
        self.global_step += 1
        
        # Log at specified frequency
        if self.global_step % self.log_freq == 0:
            metrics = {"train/batch_loss": loss}
            
            # Log learning rate
            if optimizer:
                for i, param_group in enumerate(optimizer.param_groups):
                    lr = param_group.get('lr', 0)
                    metrics[f"train/lr_group_{i}"] = lr
            
            # Log system metrics
            metrics.update(self._get_system_metrics())
            
            wandb.log(metrics, step=self.global_step)
            
            # Log model graph (once)
            if self.log_graph and not self.graph_logged and model and inputs is not None:
                try:
                    # Create dummy forward pass for graph
                    import torch.onnx
                    import tempfile
                    
                    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=True) as f:
                        torch.onnx.export(
                            model,
                            inputs,
                            f.name,
                            input_names=['input'],
                            output_names=['output'],
                            dynamic_axes={'input': {0: 'batch_size'}}
                        )
                        wandb.save(f.name)
                    
                    self.graph_logged = True
                    logger.debug("Model graph logged to W&B")
                except Exception as e:
                    logger.warning(f"Failed to log model graph: {e}")
    
    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
        **kwargs
    ):
        """
        Called at the end of an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
            model: Model being trained
        """
        if self.disabled:
            return
        
        # Prepare metrics with proper naming
        wandb_metrics = {}
        for key, value in metrics.items():
            if "train" in key.lower():
                wandb_metrics[f"train/{key}"] = value
            elif "val" in key.lower() or "valid" in key.lower():
                wandb_metrics[f"val/{key}"] = value
            else:
                wandb_metrics[key] = value
        
        # Log metrics
        wandb.log(wandb_metrics, step=self.global_step)
        
        # Check for best model
        if self.monitor_metric in metrics:
            current_metric = metrics[self.monitor_metric]
            is_best = False
            
            if self.best_metric is None:
                is_best = True
            elif self.monitor_mode == "min" and current_metric < self.best_metric:
                is_best = True
            elif self.monitor_mode == "max" and current_metric > self.best_metric:
                is_best = True
            
            if is_best:
                self.best_metric = current_metric
                wandb_metrics[f"best/{self.monitor_metric}"] = current_metric
                
                # Save best model
                if self.save_checkpoints and model:
                    self._save_checkpoint(model, epoch, is_best=True)
        
        # Save regular checkpoint
        if self.save_checkpoints and model and epoch % self.checkpoint_freq == 0:
            self._save_checkpoint(model, epoch, is_best=False)
        
        # Log learning rate schedule plot
        if epoch % 10 == 0:
            self._log_lr_schedule(epoch)
        
        # Create summary metrics
        wandb.run.summary.update({
            "best_epoch": epoch if self.best_metric == metrics.get(self.monitor_metric) else wandb.run.summary.get("best_epoch", 0),
            f"best_{self.monitor_metric}": self.best_metric
        })
    
    def on_train_end(
        self,
        model: Optional[torch.nn.Module] = None,
        **kwargs
    ):
        """
        Called at the end of training.
        
        Finalizes logging and saves artifacts.
        
        Args:
            model: Final trained model
        """
        if self.disabled:
            return
        
        # Save final model
        if self.log_model and model:
            self._save_model_artifact(model, "final")
        
        # Log final summary
        wandb.run.summary.update({
            "training_completed": True,
            "total_steps": self.global_step,
            "total_epochs": self.epoch
        })
        
        # Finish run
        wandb.finish()
        logger.info(f"W&B run completed: {wandb.run.name}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log custom metrics.
        
        Args:
            metrics: Dictionary of metrics
            step: Step number (uses internal counter if None)
        """
        if self.disabled:
            return
        
        step = step if step is not None else self.global_step
        wandb.log(metrics, step=step)
    
    def log_table(self, key: str, columns: List[str], data: List[List[Any]]):
        """
        Log a table of data.
        
        Args:
            key: Table key
            columns: Column names
            data: Table data
        """
        if self.disabled:
            return
        
        table = wandb.Table(columns=columns, data=data)
        wandb.log({key: table})
    
    def log_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: List[str],
        title: str = "Confusion Matrix"
    ):
        """
        Log confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: List of class names
            title: Title for the plot
        """
        if self.disabled:
            return
        
        wandb.log({
            title: wandb.plot.confusion_matrix(
                probs=None,
                y_true=y_true,
                preds=y_pred,
                class_names=class_names
            )
        })
    
    def log_roc_curve(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        labels: Optional[List[str]] = None
    ):
        """
        Log ROC curve.
        
        Args:
            y_true: True labels
            y_scores: Prediction scores
            labels: Class labels
        """
        if self.disabled:
            return
        
        wandb.log({
            "roc_curve": wandb.plot.roc_curve(
                y_true,
                y_scores,
                labels=labels
            )
        })
    
    def log_artifact(
        self,
        artifact_path: Union[str, Path],
        name: str,
        type: str = "dataset",
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log an artifact.
        
        Args:
            artifact_path: Path to artifact
            name: Artifact name
            type: Artifact type
            metadata: Artifact metadata
        """
        if self.disabled:
            return
        
        artifact = wandb.Artifact(name, type=type, metadata=metadata)
        artifact.add_file(str(artifact_path))
        wandb.log_artifact(artifact)
    
    def _extract_config(self, config: Any) -> Dict[str, Any]:
        """
        Extract configuration dictionary from config object.
        
        Args:
            config: Configuration object
            
        Returns:
            Configuration dictionary
        """
        if isinstance(config, dict):
            return config
        elif hasattr(config, '__dict__'):
            return {k: v for k, v in config.__dict__.items() 
                   if not k.startswith('_') and isinstance(v, (int, float, str, bool, list, dict))}
        else:
            return {}
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """
        Get system resource metrics.
        
        Returns:
            Dictionary of system metrics
        """
        metrics = {}
        
        try:
            import psutil
            
            # CPU metrics
            metrics["system/cpu_percent"] = psutil.cpu_percent(interval=0.1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics["system/memory_percent"] = memory.percent
            metrics["system/memory_used_gb"] = memory.used / (1024**3)
            
            # GPU metrics if available
            if torch.cuda.is_available():
                metrics["system/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / (1024**3)
                metrics["system/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved() / (1024**3)
                
                # GPU utilization (if nvidia-ml-py is available)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    metrics["system/gpu_utilization"] = util.gpu
                    metrics["system/gpu_memory_utilization"] = util.memory
                except:
                    pass
                    
        except ImportError:
            pass
        
        return metrics
    
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            epoch: Current epoch
            is_best: Whether this is the best model
        """
        checkpoint_name = f"checkpoint_epoch_{epoch}"
        if is_best:
            checkpoint_name = "best_model"
        
        # Save locally first
        checkpoint_path = self.dir / f"{checkpoint_name}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_metric': self.best_metric,
            'global_step': self.global_step
        }, checkpoint_path)
        
        # Log to W&B
        wandb.save(str(checkpoint_path))
        
        if is_best:
            logger.info(f"Best model saved at epoch {epoch} ({self.monitor_metric}: {self.best_metric:.4f})")
    
    def _save_model_artifact(
        self,
        model: torch.nn.Module,
        artifact_name: str
    ):
        """
        Save model as W&B artifact.
        
        Args:
            model: Model to save
            artifact_name: Name for the artifact
        """
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pt"
            torch.save(model.state_dict(), model_path)
            
            artifact = wandb.Artifact(
                f"{self.project}-{artifact_name}",
                type="model",
                metadata={
                    "epoch": self.epoch,
                    "best_metric": self.best_metric,
                    "global_step": self.global_step
                }
            )
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
    
    def _log_lr_schedule(self, epoch: int):
        """
        Log learning rate schedule visualization.
        
        Args:
            epoch: Current epoch
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create dummy LR schedule plot
            epochs = list(range(epoch + 1))
            lrs = [self.config.get('learning_rate', 2e-5) * (0.95 ** e) for e in epochs]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs, lrs, 'b-', label='Learning Rate')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            wandb.log({"lr_schedule": wandb.Image(fig)})
            plt.close(fig)
            
        except ImportError:
            pass
