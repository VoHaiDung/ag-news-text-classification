"""
Early Stopping Callback for Training
=====================================

Implementation of early stopping with various strategies to prevent overfitting,
based on:
- Prechelt (1998): "Early Stopping - But When?"
- Yao et al. (2007): "On Early Stopping in Gradient Descent Learning"
- Dodge et al. (2020): "Fine-Tuning Pretrained Language Models"

Provides multiple stopping criteria and patience strategies for robust training.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, List, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping"""
    
    # Main parameters
    patience: int = 5  # Number of epochs to wait
    min_delta: float = 0.0001  # Minimum change to qualify as improvement
    mode: str = 'min'  # 'min' or 'max'
    
    # Advanced patience strategies
    patience_strategy: str = 'constant'  # 'constant', 'linear', 'exponential'
    initial_patience: int = 3
    max_patience: int = 10
    patience_factor: float = 1.5
    
    # Metric tracking
    monitor: str = 'val_loss'  # Metric to monitor
    baseline: Optional[float] = None  # Baseline value
    restore_best_weights: bool = True
    
    # Warmup
    warmup_epochs: int = 0  # Don't stop during warmup
    min_epochs: int = 5  # Minimum epochs before stopping
    
    # Generalization loss
    use_generalization_loss: bool = False
    generalization_threshold: float = 0.1
    
    # Moving average
    use_moving_average: bool = False
    ma_window: int = 3
    
    # Divergence detection
    detect_divergence: bool = True
    divergence_threshold: float = 5.0
    
    # Checkpointing
    save_checkpoint: bool = True
    checkpoint_dir: Path = Path("checkpoints")
    keep_best_only: bool = True


class EarlyStopping:
    """
    Early stopping handler with advanced strategies.
    
    Monitors training metrics and stops training when no improvement
    is observed, with support for various patience strategies and
    stopping criteria.
    """
    
    def __init__(self, config: Optional[EarlyStoppingConfig] = None):
        """
        Initialize early stopping.
        
        Args:
            config: Early stopping configuration
        """
        self.config = config or EarlyStoppingConfig()
        
        # Initialize state
        self.best_score = None
        self.best_epoch = 0
        self.counter = 0
        self.current_patience = self.config.patience
        self.stopped_epoch = 0
        self.early_stop = False
        
        # History tracking
        self.score_history = []
        self.patience_history = []
        
        # Best model state
        self.best_model_state = None
        
        # Moving average buffer
        if self.config.use_moving_average:
            self.ma_buffer = []
        
        # Determine improvement direction
        self.monitor_op = np.less if self.config.mode == 'min' else np.greater
        self.best_score = np.inf if self.config.mode == 'min' else -np.inf
        
        logger.info(
            f"Initialized EarlyStopping: "
            f"monitor={config.monitor}, patience={config.patience}, "
            f"mode={config.mode}"
        )
    
    def __call__(
        self,
        score: float,
        model: Optional[torch.nn.Module] = None,
        epoch: int = 0
    ) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            model: Model to save state
            epoch: Current epoch
            
        Returns:
            True if training should stop
        """
        # Skip during warmup
        if epoch < self.config.warmup_epochs:
            logger.debug(f"Warmup epoch {epoch}, skipping early stopping")
            return False
        
        # Check minimum epochs
        if epoch < self.config.min_epochs:
            logger.debug(f"Epoch {epoch} < min_epochs {self.config.min_epochs}")
            return False
        
        # Apply moving average if configured
        if self.config.use_moving_average:
            score = self._apply_moving_average(score)
        
        # Check for divergence
        if self.config.detect_divergence and self._is_diverging(score):
            logger.warning(f"Divergence detected at epoch {epoch}")
            self.early_stop = True
            self.stopped_epoch = epoch
            return True
        
        # Track history
        self.score_history.append(score)
        
        # Check improvement
        is_improvement = self._check_improvement(score)
        
        if is_improvement:
            self._on_improvement(score, model, epoch)
        else:
            self._on_no_improvement(epoch)
        
        return self.early_stop
    
    def _check_improvement(self, score: float) -> bool:
        """Check if score improved"""
        if self.config.baseline is not None:
            # Compare with baseline first
            if not self.monitor_op(score, self.config.baseline):
                return False
        
        # Check against best score
        if self.best_score is None:
            return True
        
        # Apply minimum delta
        if self.config.mode == 'min':
            return score < self.best_score - self.config.min_delta
        else:
            return score > self.best_score + self.config.min_delta
    
    def _on_improvement(
        self,
        score: float,
        model: Optional[torch.nn.Module],
        epoch: int
    ):
        """Handle improvement in score"""
        logger.info(
            f"Improvement detected: {self.config.monitor} "
            f"{self.best_score:.6f} -> {score:.6f} at epoch {epoch}"
        )
        
        self.best_score = score
        self.best_epoch = epoch
        self.counter = 0
        
        # Save best model state
        if model is not None and self.config.restore_best_weights:
            self.best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        
        # Save checkpoint
        if self.config.save_checkpoint and model is not None:
            self._save_checkpoint(model, epoch, score)
        
        # Update patience if using adaptive strategy
        self._update_patience(improved=True)
    
    def _on_no_improvement(self, epoch: int):
        """Handle no improvement in score"""
        self.counter += 1
        logger.info(
            f"No improvement for {self.counter} epochs "
            f"(patience: {self.current_patience})"
        )
        
        if self.counter >= self.current_patience:
            self.early_stop = True
            self.stopped_epoch = epoch
            logger.info(f"Early stopping triggered at epoch {epoch}")
        
        # Update patience if using adaptive strategy
        self._update_patience(improved=False)
    
    def _update_patience(self, improved: bool):
        """Update patience based on strategy"""
        if self.config.patience_strategy == 'constant':
            return
        
        elif self.config.patience_strategy == 'linear':
            if not improved:
                # Increase patience linearly
                self.current_patience = min(
                    self.current_patience + 1,
                    self.config.max_patience
                )
            else:
                # Reset to initial
                self.current_patience = self.config.initial_patience
                
        elif self.config.patience_strategy == 'exponential':
            if not improved:
                # Increase patience exponentially
                self.current_patience = min(
                    int(self.current_patience * self.config.patience_factor),
                    self.config.max_patience
                )
            else:
                # Reset to initial
                self.current_patience = self.config.initial_patience
        
        self.patience_history.append(self.current_patience)
    
    def _apply_moving_average(self, score: float) -> float:
        """Apply moving average to score"""
        self.ma_buffer.append(score)
        
        if len(self.ma_buffer) > self.config.ma_window:
            self.ma_buffer.pop(0)
        
        return np.mean(self.ma_buffer)
    
    def _is_diverging(self, score: float) -> bool:
        """Check if training is diverging"""
        if not self.config.detect_divergence:
            return False
        
        if len(self.score_history) < 2:
            return False
        
        # Check if score is much worse than initial
        initial_score = self.score_history[0]
        
        if self.config.mode == 'min':
            return score > initial_score * self.config.divergence_threshold
        else:
            return score < initial_score / self.config.divergence_threshold
    
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        epoch: int,
        score: float
    ):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
            'monitor': self.config.monitor
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Remove old checkpoints if keeping best only
        if self.config.keep_best_only:
            for old_checkpoint in checkpoint_dir.glob("best_model_epoch_*.pt"):
                if old_checkpoint != checkpoint_path:
                    old_checkpoint.unlink()
    
    def restore_best_weights(self, model: torch.nn.Module):
        """Restore best model weights"""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
            logger.info(f"Restored best weights from epoch {self.best_epoch}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get early stopping statistics"""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'stopped_epoch': self.stopped_epoch,
            'total_patience_used': self.counter,
            'score_history': self.score_history[-10:],  # Last 10 scores
            'early_stopped': self.early_stop
        }
    
    def reset(self):
        """Reset early stopping state"""
        self.best_score = np.inf if self.config.mode == 'min' else -np.inf
        self.best_epoch = 0
        self.counter = 0
        self.current_patience = self.config.patience
        self.stopped_epoch = 0
        self.early_stop = False
        self.score_history = []
        self.patience_history = []
        self.best_model_state = None
        
        if self.config.use_moving_average:
            self.ma_buffer = []
        
        logger.info("Reset early stopping state")


class GeneralizationLoss:
    """
    Track generalization loss for early stopping.
    
    Based on Prechelt (1998), stops when generalization loss exceeds threshold.
    GL = 100 * (val_loss / min_val_loss - 1)
    """
    
    def __init__(self, threshold: float = 5.0):
        """
        Initialize generalization loss tracker.
        
        Args:
            threshold: GL threshold for stopping
        """
        self.threshold = threshold
        self.min_val_loss = np.inf
        self.gl_history = []
        
    def update(self, val_loss: float) -> float:
        """
        Update and return generalization loss.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            Current generalization loss
        """
        self.min_val_loss = min(self.min_val_loss, val_loss)
        
        if self.min_val_loss > 0:
            gl = 100 * (val_loss / self.min_val_loss - 1)
        else:
            gl = 0
        
        self.gl_history.append(gl)
        return gl
    
    def should_stop(self, val_loss: float) -> bool:
        """Check if should stop based on GL"""
        gl = self.update(val_loss)
        return gl > self.threshold


# Export classes
__all__ = [
    'EarlyStoppingConfig',
    'EarlyStopping',
    'GeneralizationLoss'
]
