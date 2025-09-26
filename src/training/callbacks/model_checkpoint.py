"""
Model Checkpoint Callback Implementation
=========================================

This module implements checkpoint callbacks for saving model states
during training with various strategies and monitoring.

Features:
- Best model tracking
- Periodic checkpointing
- Multi-metric monitoring
- Checkpoint management

References:
- PyTorch Lightning Callbacks
- Keras ModelCheckpoint
- HuggingFace Trainer Callbacks

Author: Võ Hải Dũng
License: MIT
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from datetime import datetime
import logging

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""
    
    save_dir: str = "./checkpoints"
    filename_prefix: str = "model"
    monitor: str = "val_loss"
    mode: str = "min"  # min or max
    save_top_k: int = 3
    save_last: bool = True
    save_every_n_epochs: Optional[int] = None
    save_every_n_steps: Optional[int] = None
    save_on_train_epoch_end: bool = True
    save_weights_only: bool = False
    auto_insert_metric_name: bool = True
    verbose: bool = True


class ModelCheckpoint:
    """
    Model checkpoint callback for saving training states.
    
    Monitors metrics and saves best models with checkpoint management.
    """
    
    def __init__(self, config: Optional[CheckpointConfig] = None):
        """
        Initialize model checkpoint callback.
        
        Args:
            config: Checkpoint configuration
        """
        self.config = config or CheckpointConfig()
        
        # Create save directory
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking
        self.best_score = float('inf') if self.config.mode == 'min' else float('-inf')
        self.best_checkpoints = []
        self.last_checkpoint = None
        self.checkpoint_count = 0
        
        logger.info(f"ModelCheckpoint initialized: monitoring {self.config.monitor}")
    
    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
        metrics: Dict[str, float],
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None
    ):
        """
        Called at the end of each epoch.
        
        Args:
            epoch: Current epoch
            model: Model to save
            metrics: Current metrics
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
        """
        # Check if should save on this epoch
        should_save = False
        
        if self.config.save_every_n_epochs:
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                should_save = True
        
        # Check monitored metric
        if self.config.monitor in metrics:
            current_score = metrics[self.config.monitor]
            
            if self._is_better(current_score):
                self.best_score = current_score
                should_save = True
                
                # Save as best model
                self._save_checkpoint(
                    model=model,
                    epoch=epoch,
                    metrics=metrics,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    is_best=True
                )
        
        # Save last checkpoint if configured
        if self.config.save_last:
            self._save_last_checkpoint(
                model=model,
                epoch=epoch,
                metrics=metrics,
                optimizer=optimizer,
                scheduler=scheduler
            )
        
        # Manage checkpoint files
        self._manage_checkpoints()
    
    def on_train_step_end(
        self,
        step: int,
        model: nn.Module,
        metrics: Dict[str, float]
    ):
        """
        Called at the end of each training step.
        
        Args:
            step: Current global step
            model: Model to save
            metrics: Current metrics
        """
        if self.config.save_every_n_steps:
            if (step + 1) % self.config.save_every_n_steps == 0:
                self._save_checkpoint(
                    model=model,
                    step=step,
                    metrics=metrics,
                    is_step_checkpoint=True
                )
    
    def _is_better(self, current: float) -> bool:
        """Check if current score is better than best."""
        if self.config.mode == 'min':
            return current < self.best_score
        return current > self.best_score
    
    def _save_checkpoint(
        self,
        model: nn.Module,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None,
        is_best: bool = False,
        is_step_checkpoint: bool = False
    ):
        """Save model checkpoint."""
        # Generate filename
        filename = self._generate_filename(
            epoch=epoch,
            step=step,
            metrics=metrics,
            is_best=is_best,
            is_step_checkpoint=is_step_checkpoint
        )
        
        filepath = self.save_dir / filename
        
        # Prepare checkpoint data
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict() if not self.config.save_weights_only else None,
            'metrics': metrics,
            'best_score': self.best_score,
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat()
        }
        
        if not self.config.save_weights_only:
            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        
        if self.config.verbose:
            logger.info(f"Saved checkpoint: {filepath}")
        
        # Track best checkpoints
        if is_best:
            self.best_checkpoints.append({
                'filepath': filepath,
                'score': metrics.get(self.config.monitor) if metrics else None,
                'epoch': epoch,
                'step': step
            })
        
        self.checkpoint_count += 1
        
        return filepath
    
    def _save_last_checkpoint(
        self,
        model: nn.Module,
        epoch: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        optimizer: Optional[Any] = None,
        scheduler: Optional[Any] = None
    ):
        """Save last checkpoint."""
        filename = f"{self.config.filename_prefix}_last.pt"
        filepath = self.save_dir / filename
        
        # Remove previous last checkpoint
        if self.last_checkpoint and self.last_checkpoint.exists():
            self.last_checkpoint.unlink()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        if not self.config.save_weights_only:
            if optimizer:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        self.last_checkpoint = filepath
    
    def _generate_filename(
        self,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False,
        is_step_checkpoint: bool = False
    ) -> str:
        """Generate checkpoint filename."""
        parts = [self.config.filename_prefix]
        
        if is_best:
            parts.append("best")
        elif is_step_checkpoint:
            parts.append(f"step_{step}")
        else:
            if epoch is not None:
                parts.append(f"epoch_{epoch}")
        
        if self.config.auto_insert_metric_name and metrics:
            if self.config.monitor in metrics:
                value = metrics[self.config.monitor]
                parts.append(f"{self.config.monitor}_{value:.4f}")
        
        return "_".join(parts) + ".pt"
    
    def _manage_checkpoints(self):
        """Manage checkpoint files to keep only top-k."""
        if self.config.save_top_k <= 0:
            return
        
        # Sort best checkpoints by score
        if self.best_checkpoints:
            reverse = self.config.mode == 'max'
            self.best_checkpoints.sort(
                key=lambda x: x['score'] if x['score'] is not None else float('-inf'),
                reverse=reverse
            )
            
            # Keep only top-k
            while len(self.best_checkpoints) > self.config.save_top_k:
                # Remove worst checkpoint
                to_remove = self.best_checkpoints.pop()
                filepath = to_remove['filepath']
                
                if filepath.exists():
                    filepath.unlink()
                    if self.config.verbose:
                        logger.info(f"Removed checkpoint: {filepath}")
    
    def load_best_checkpoint(self, model: nn.Module) -> nn.Module:
        """
        Load best checkpoint into model.
        
        Args:
            model: Model to load weights into
            
        Returns:
            Model with loaded weights
        """
        if not self.best_checkpoints:
            logger.warning("No best checkpoint found")
            return model
        
        best = self.best_checkpoints[0]
        checkpoint = torch.load(best['filepath'])
        
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best checkpoint from {best['filepath']}")
        
        return model
