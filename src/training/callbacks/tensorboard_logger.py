"""
TensorBoard Logger Callback for Training Monitoring
====================================================

This module implements TensorBoard logging for training metrics visualization.
Based on TensorBoard best practices and integration patterns from:
- Abadi et al. (2016): TensorFlow's visualization toolkit
- PyTorch Lightning's TensorBoard integration

The logger tracks:
1. Training metrics (loss, accuracy, learning rate)
2. Validation metrics and comparisons
3. Model architecture and parameter distributions
4. Gradient flow and optimization dynamics
5. Custom scalar/histogram/image logging

Author: Võ Hải Dũng
License: MIT
"""

import os
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

logger = logging.getLogger(__name__)


class TensorBoardLogger:
    """
    TensorBoard logging callback for training visualization.
    
    Provides comprehensive logging of training dynamics with support for:
    - Scalar metrics tracking
    - Histogram visualization
    - Model graph visualization
    - Gradient flow analysis
    - Learning rate scheduling visualization
    - Custom metric logging
    
    Implementation follows TensorBoard best practices for deep learning experiments.
    """
    
    def __init__(
        self,
        log_dir: Optional[Union[str, Path]] = None,
        comment: str = "",
        flush_secs: int = 120,
        max_queue: int = 10,
        filename_suffix: str = "",
        write_graph: bool = True,
        write_images: bool = False,
        write_histograms: bool = True,
        log_gradient_flow: bool = True,
        log_learning_rate: bool = True,
        log_weight_histograms: bool = False,
        update_freq: int = 10,
        profile_batch: Optional[int] = None
    ):
        """
        Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for saving logs (default: outputs/logs/tensorboard)
            comment: Comment to append to log directory name
            flush_secs: Seconds between flush operations
            max_queue: Maximum queue size for pending events
            filename_suffix: Suffix for event file names
            write_graph: Whether to write model graph
            write_images: Whether to write image summaries
            write_histograms: Whether to write histogram summaries
            log_gradient_flow: Whether to log gradient flow
            log_learning_rate: Whether to log learning rate
            log_weight_histograms: Whether to log weight histograms
            update_freq: Frequency of updates (in batches)
            profile_batch: Batch to profile (for performance analysis)
        """
        # Setup log directory
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = Path("outputs/logs/tensorboard") / f"run_{timestamp}"
            if comment:
                log_dir = log_dir.parent / f"{log_dir.name}_{comment}"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SummaryWriter
        self.writer = SummaryWriter(
            log_dir=str(self.log_dir),
            comment=comment,
            flush_secs=flush_secs,
            max_queue=max_queue,
            filename_suffix=filename_suffix
        )
        
        # Configuration
        self.write_graph = write_graph
        self.write_images = write_images
        self.write_histograms = write_histograms
        self.log_gradient_flow = log_gradient_flow
        self.log_learning_rate = log_learning_rate
        self.log_weight_histograms = log_weight_histograms
        self.update_freq = update_freq
        self.profile_batch = profile_batch
        
        # State tracking
        self.global_step = 0
        self.epoch = 0
        self.graph_written = False
        self.best_metrics = {}
        
        logger.info(f"TensorBoard logger initialized. Logs: {self.log_dir}")
    
    def on_train_begin(self, **kwargs):
        """
        Called at the beginning of training.
        
        Logs training configuration and hyperparameters.
        """
        # Log hyperparameters
        if "config" in kwargs:
            config = kwargs["config"]
            hparams = self._extract_hparams(config)
            self.writer.add_hparams(hparams, {})
            logger.debug("Logged hyperparameters to TensorBoard")
        
        # Log training info
        self.writer.add_text(
            "Training/Info",
            f"Training started at {datetime.now().isoformat()}",
            global_step=0
        )
    
    def on_epoch_begin(self, epoch: int, **kwargs):
        """
        Called at the beginning of an epoch.
        
        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch
        self.writer.add_scalar("Training/Epoch", epoch, self.global_step)
    
    def on_batch_end(
        self,
        batch: int,
        loss: float,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        inputs: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Called at the end of a batch.
        
        Args:
            batch: Batch index
            loss: Batch loss value
            model: Model being trained
            optimizer: Optimizer being used
            inputs: Input batch (for graph visualization)
        """
        self.global_step += 1
        
        # Log loss
        self.writer.add_scalar("Training/BatchLoss", loss, self.global_step)
        
        # Log at specified frequency
        if self.global_step % self.update_freq == 0:
            
            # Log learning rate
            if self.log_learning_rate and optimizer is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    lr = param_group.get('lr', 0)
                    self.writer.add_scalar(
                        f"Training/LearningRate/Group{i}",
                        lr,
                        self.global_step
                    )
            
            # Log gradient flow
            if self.log_gradient_flow and model is not None:
                self._log_gradient_flow(model)
            
            # Log weight histograms
            if self.log_weight_histograms and model is not None:
                self._log_weight_histograms(model)
            
            # Write model graph (once)
            if self.write_graph and not self.graph_written and model is not None and inputs is not None:
                try:
                    self.writer.add_graph(model, inputs)
                    self.graph_written = True
                    logger.debug("Model graph written to TensorBoard")
                except Exception as e:
                    logger.warning(f"Failed to write model graph: {e}")
        
        # Profile specific batch
        if self.profile_batch and batch == self.profile_batch:
            self._profile_batch(model, inputs)
    
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
        # Log epoch metrics
        for metric_name, metric_value in metrics.items():
            # Separate train/val metrics
            if "train" in metric_name.lower():
                self.writer.add_scalar(
                    f"Train/{metric_name}",
                    metric_value,
                    epoch
                )
            elif "val" in metric_name.lower() or "valid" in metric_name.lower():
                self.writer.add_scalar(
                    f"Validation/{metric_name}",
                    metric_value,
                    epoch
                )
            else:
                self.writer.add_scalar(
                    f"Metrics/{metric_name}",
                    metric_value,
                    epoch
                )
        
        # Track best metrics
        for key, value in metrics.items():
            if "loss" in key.lower():
                # Lower is better for loss
                if key not in self.best_metrics or value < self.best_metrics[key]:
                    self.best_metrics[key] = value
                    self.writer.add_scalar(f"Best/{key}", value, epoch)
            else:
                # Higher is better for accuracy/f1
                if key not in self.best_metrics or value > self.best_metrics[key]:
                    self.best_metrics[key] = value
                    self.writer.add_scalar(f"Best/{key}", value, epoch)
        
        # Log parameter statistics
        if self.write_histograms and model is not None:
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    self.writer.add_histogram(
                        f"Parameters/{name}",
                        param.data.cpu().numpy(),
                        epoch
                    )
                    self.writer.add_histogram(
                        f"Gradients/{name}",
                        param.grad.data.cpu().numpy(),
                        epoch
                    )
        
        # Flush to ensure data is written
        self.writer.flush()
    
    def on_train_end(self, **kwargs):
        """
        Called at the end of training.
        
        Logs final summary and closes writer.
        """
        # Log final summary
        summary_text = "Training completed\n\n"
        summary_text += "Best metrics:\n"
        for key, value in self.best_metrics.items():
            summary_text += f"- {key}: {value:.4f}\n"
        
        self.writer.add_text("Training/Summary", summary_text, self.global_step)
        
        # Close writer
        self.writer.close()
        logger.info(f"TensorBoard logging completed. Logs saved to: {self.log_dir}")
    
    def log_scalar(self, tag: str, value: float, step: Optional[int] = None):
        """
        Log a scalar value.
        
        Args:
            tag: Metric tag
            value: Metric value
            step: Global step (uses internal counter if None)
        """
        step = step if step is not None else self.global_step
        self.writer.add_scalar(tag, value, step)
    
    def log_histogram(self, tag: str, values: np.ndarray, step: Optional[int] = None):
        """
        Log a histogram.
        
        Args:
            tag: Histogram tag
            values: Values to create histogram from
            step: Global step
        """
        step = step if step is not None else self.global_step
        self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: Optional[int] = None):
        """
        Log an image.
        
        Args:
            tag: Image tag
            image: Image tensor (C, H, W)
            step: Global step
        """
        if self.write_images:
            step = step if step is not None else self.global_step
            self.writer.add_image(tag, image, step)
    
    def log_text(self, tag: str, text: str, step: Optional[int] = None):
        """
        Log text.
        
        Args:
            tag: Text tag
            text: Text content
            step: Global step
        """
        step = step if step is not None else self.global_step
        self.writer.add_text(tag, text, step)
    
    def log_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        step: Optional[int] = None
    ):
        """
        Log confusion matrix as an image.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            step: Global step
        """
        if self.write_images:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 10))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            ax.set(
                xticks=np.arange(cm.shape[1]),
                yticks=np.arange(cm.shape[0]),
                xticklabels=class_names,
                yticklabels=class_names,
                ylabel='True label',
                xlabel='Predicted label'
            )
            
            # Rotate the tick labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            
            fig.tight_layout()
            
            # Convert to tensor
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            image = torch.from_numpy(image).permute(2, 0, 1)
            
            step = step if step is not None else self.global_step
            self.writer.add_image("ConfusionMatrix", image, step)
            
            plt.close(fig)
    
    def _extract_hparams(self, config: Any) -> Dict[str, Any]:
        """
        Extract hyperparameters from config object.
        
        Args:
            config: Configuration object
            
        Returns:
            Dictionary of hyperparameters
        """
        hparams = {}
        
        # Extract from config attributes
        if hasattr(config, '__dict__'):
            for key, value in config.__dict__.items():
                if isinstance(value, (int, float, str, bool)):
                    hparams[key] = value
        
        # Extract from dictionary
        elif isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (int, float, str, bool)):
                    hparams[key] = value
        
        return hparams
    
    def _log_gradient_flow(self, model: torch.nn.Module):
        """
        Log gradient flow through the network.
        
        Helps identify vanishing/exploding gradient problems.
        
        Args:
            model: Model to analyze
        """
        ave_grads = []
        max_grads = []
        layers = []
        
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())
        
        if layers:
            # Log average gradients
            for layer, avg_grad in zip(layers, ave_grads):
                self.writer.add_scalar(
                    f"GradientFlow/Average/{layer}",
                    avg_grad,
                    self.global_step
                )
            
            # Log max gradients
            for layer, max_grad in zip(layers, max_grads):
                self.writer.add_scalar(
                    f"GradientFlow/Max/{layer}",
                    max_grad,
                    self.global_step
                )
            
            # Log overall statistics
            self.writer.add_scalar(
                "GradientFlow/GlobalAverage",
                np.mean(ave_grads),
                self.global_step
            )
            self.writer.add_scalar(
                "GradientFlow/GlobalMax",
                np.max(max_grads),
                self.global_step
            )
    
    def _log_weight_histograms(self, model: torch.nn.Module):
        """
        Log weight histograms for model parameters.
        
        Args:
            model: Model to analyze
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.writer.add_histogram(
                    f"Weights/{name}",
                    param.data.cpu().numpy(),
                    self.global_step
                )
    
    def _profile_batch(self, model: torch.nn.Module, inputs: torch.Tensor):
        """
        Profile a batch for performance analysis.
        
        Args:
            model: Model to profile
            inputs: Input batch
        """
        try:
            from torch.profiler import profile, ProfilerActivity
            
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                with torch.no_grad():
                    _ = model(inputs)
            
            # Log profiling results
            self.writer.add_text(
                "Profiling/Summary",
                prof.key_averages().table(sort_by="cuda_time_total", row_limit=20),
                self.global_step
            )
            
            logger.info("Batch profiling completed")
            
        except ImportError:
            logger.warning("torch.profiler not available, skipping profiling")
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
    
    def flush(self):
        """Force flush pending events."""
        self.writer.flush()
    
    def close(self):
        """Close the writer."""
        self.writer.close()
