"""
Learning Rate Monitor Callback for Training
===========================================

This module implements learning rate monitoring during training.
Based on learning rate scheduling best practices from:
- Loshchilov & Hutter (2016): SGDR - Stochastic Gradient Descent with Warm Restarts
- Smith (2017): Cyclical Learning Rates for Training Neural Networks

The monitor tracks:
1. Learning rate changes across parameter groups
2. Effective learning rate with momentum
3. Learning rate schedule visualization
4. Optimal LR finding

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import torch
from collections import deque
import warnings

logger = logging.getLogger(__name__)


class LearningRateMonitor:
    """
    Learning rate monitoring callback.
    
    Tracks and visualizes learning rate changes during training,
    providing insights into optimization dynamics.
    
    Features:
    - Multi-parameter group tracking
    - Effective LR calculation
    - Schedule visualization
    - LR range test support
    - Automatic LR finding
    """
    
    def __init__(
        self,
        log_momentum: bool = False,
        log_weight_decay: bool = False,
        moving_average_window: int = 100,
        log_frequency: int = 1,
        track_best_lr: bool = True,
        visualize: bool = False,
        save_history: bool = True
    ):
        """
        Initialize learning rate monitor.
        
        Args:
            log_momentum: Whether to log momentum values
            log_weight_decay: Whether to log weight decay
            moving_average_window: Window size for moving averages
            log_frequency: Frequency of logging (in batches)
            track_best_lr: Whether to track best LR based on loss
            visualize: Whether to create visualizations
            save_history: Whether to save LR history
        """
        self.log_momentum = log_momentum
        self.log_weight_decay = log_weight_decay
        self.moving_average_window = moving_average_window
        self.log_frequency = log_frequency
        self.track_best_lr = track_best_lr
        self.visualize = visualize
        self.save_history = save_history
        
        # State tracking
        self.lr_history = []
        self.loss_history = []
        self.step = 0
        self.epoch = 0
        self.best_lr = None
        self.best_loss = float('inf')
        
        # Moving average for smoothing
        self.loss_window = deque(maxlen=moving_average_window)
        
        # Parameter group tracking
        self.param_groups_history = {}
        
        logger.info("Learning rate monitor initialized")
    
    def on_train_begin(
        self,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        """
        Called at the beginning of training.
        
        Args:
            optimizer: Optimizer being used
        """
        if optimizer:
            self._log_initial_state(optimizer)
    
    def on_batch_begin(
        self,
        batch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        """
        Called at the beginning of a batch.
        
        Args:
            batch: Batch index
            optimizer: Optimizer being used
        """
        if optimizer and batch % self.log_frequency == 0:
            self._record_lr_state(optimizer)
    
    def on_batch_end(
        self,
        batch: int,
        loss: float,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        """
        Called at the end of a batch.
        
        Args:
            batch: Batch index
            loss: Batch loss
            optimizer: Optimizer being used
        """
        self.step += 1
        
        # Record loss
        self.loss_window.append(loss)
        if self.save_history:
            self.loss_history.append(loss)
        
        # Track best LR
        if self.track_best_lr and optimizer:
            self._update_best_lr(optimizer, loss)
        
        # Log current state
        if self.step % self.log_frequency == 0 and optimizer:
            self._log_current_state(optimizer, loss)
    
    def on_epoch_end(
        self,
        epoch: int,
        metrics: Dict[str, float],
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs
    ):
        """
        Called at the end of an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Epoch metrics
            optimizer: Optimizer being used
        """
        self.epoch = epoch
        
        if optimizer:
            # Log epoch summary
            self._log_epoch_summary(optimizer, metrics)
            
            # Visualize if requested
            if self.visualize:
                self._create_visualization()
    
    def on_train_end(self, **kwargs):
        """
        Called at the end of training.
        
        Generates final report and visualizations.
        """
        # Generate final report
        report = self._generate_report()
        logger.info(f"Learning rate monitoring complete:\n{report}")
        
        # Save history if requested
        if self.save_history:
            self._save_lr_history()
    
    def get_current_lr(
        self,
        optimizer: torch.optim.Optimizer,
        param_group_idx: int = 0
    ) -> float:
        """
        Get current learning rate for a parameter group.
        
        Args:
            optimizer: Optimizer
            param_group_idx: Parameter group index
            
        Returns:
            Current learning rate
        """
        return optimizer.param_groups[param_group_idx]['lr']
    
    def get_effective_lr(
        self,
        optimizer: torch.optim.Optimizer,
        param_group_idx: int = 0
    ) -> float:
        """
        Calculate effective learning rate considering momentum.
        
        The effective LR approximates the actual step size when using momentum.
        
        Args:
            optimizer: Optimizer
            param_group_idx: Parameter group index
            
        Returns:
            Effective learning rate
        """
        param_group = optimizer.param_groups[param_group_idx]
        lr = param_group['lr']
        
        # Calculate effective LR based on optimizer type
        if isinstance(optimizer, torch.optim.SGD):
            momentum = param_group.get('momentum', 0)
            if momentum > 0:
                # Effective LR with momentum (approximation)
                effective_lr = lr / (1 - momentum)
            else:
                effective_lr = lr
                
        elif isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            # Adam's effective LR is more complex due to adaptive moments
            # This is a simplified approximation
            beta1 = param_group.get('betas', (0.9, 0.999))[0]
            effective_lr = lr / (1 - beta1)
            
        else:
            effective_lr = lr
        
        return effective_lr
    
    def suggest_lr_range(
        self,
        loss_history: Optional[List[float]] = None,
        smoothing: float = 0.05
    ) -> Tuple[float, float]:
        """
        Suggest learning rate range based on loss history.
        
        Implements the LR range test methodology from Leslie Smith.
        
        Args:
            loss_history: Loss values (uses internal history if None)
            smoothing: Smoothing factor for loss
            
        Returns:
            Tuple of (min_lr, max_lr) suggestions
        """
        if loss_history is None:
            loss_history = self.loss_history
        
        if len(loss_history) < 10:
            logger.warning("Insufficient loss history for LR range suggestion")
            return (1e-5, 1e-2)
        
        # Smooth the loss
        smoothed_loss = []
        for i, loss in enumerate(loss_history):
            if i == 0:
                smoothed_loss.append(loss)
            else:
                smoothed_loss.append(
                    smoothing * loss + (1 - smoothing) * smoothed_loss[-1]
                )
        
        # Find steepest descent
        gradients = np.gradient(smoothed_loss)
        min_gradient_idx = np.argmin(gradients)
        
        # Find where loss starts increasing
        increasing_idx = len(smoothed_loss) - 1
        for i in range(min_gradient_idx, len(smoothed_loss) - 1):
            if smoothed_loss[i] < smoothed_loss[i + 1]:
                increasing_idx = i
                break
        
        # Map indices to LR values
        if self.lr_history:
            min_lr = self.lr_history[min_gradient_idx]
            max_lr = self.lr_history[min(increasing_idx, len(self.lr_history) - 1)]
        else:
            # Default range
            min_lr = 1e-5
            max_lr = 1e-2
        
        return (min_lr, max_lr)
    
    def _log_initial_state(self, optimizer: torch.optim.Optimizer):
        """
        Log initial optimizer state.
        
        Args:
            optimizer: Optimizer
        """
        logger.info("Initial optimizer state:")
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            logger.info(f"  Group {i}: LR={lr:.2e}")
            
            if self.log_momentum:
                if 'momentum' in param_group:
                    logger.info(f"    Momentum: {param_group['momentum']}")
                elif 'betas' in param_group:
                    logger.info(f"    Betas: {param_group['betas']}")
            
            if self.log_weight_decay:
                wd = param_group.get('weight_decay', 0)
                logger.info(f"    Weight decay: {wd}")
    
    def _record_lr_state(self, optimizer: torch.optim.Optimizer):
        """
        Record current LR state.
        
        Args:
            optimizer: Optimizer
        """
        if self.save_history:
            current_lrs = [pg['lr'] for pg in optimizer.param_groups]
            self.lr_history.append(current_lrs[0] if len(current_lrs) == 1 else current_lrs)
            
            # Record per-group history
            for i, pg in enumerate(optimizer.param_groups):
                if i not in self.param_groups_history:
                    self.param_groups_history[i] = {
                        'lr': [],
                        'momentum': [],
                        'weight_decay': []
                    }
                
                self.param_groups_history[i]['lr'].append(pg['lr'])
                
                if self.log_momentum:
                    if 'momentum' in pg:
                        self.param_groups_history[i]['momentum'].append(pg['momentum'])
                    elif 'betas' in pg:
                        self.param_groups_history[i]['momentum'].append(pg['betas'][0])
                
                if self.log_weight_decay:
                    self.param_groups_history[i]['weight_decay'].append(
                        pg.get('weight_decay', 0)
                    )
    
    def _update_best_lr(self, optimizer: torch.optim.Optimizer, loss: float):
        """
        Update best LR based on loss.
        
        Args:
            optimizer: Optimizer
            loss: Current loss
        """
        # Use moving average for stability
        avg_loss = np.mean(self.loss_window) if len(self.loss_window) > 0 else loss
        
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self.best_lr = self.get_current_lr(optimizer)
    
    def _log_current_state(self, optimizer: torch.optim.Optimizer, loss: float):
        """
        Log current learning rate state.
        
        Args:
            optimizer: Optimizer
            loss: Current loss
        """
        # Get current LRs
        current_lrs = []
        effective_lrs = []
        
        for i, pg in enumerate(optimizer.param_groups):
            lr = pg['lr']
            eff_lr = self.get_effective_lr(optimizer, i)
            current_lrs.append(lr)
            effective_lrs.append(eff_lr)
        
        # Log summary
        if len(current_lrs) == 1:
            logger.debug(
                f"Step {self.step}: LR={current_lrs[0]:.2e}, "
                f"Effective LR={effective_lrs[0]:.2e}, Loss={loss:.4f}"
            )
        else:
            logger.debug(f"Step {self.step}: Loss={loss:.4f}")
            for i, (lr, eff_lr) in enumerate(zip(current_lrs, effective_lrs)):
                logger.debug(f"  Group {i}: LR={lr:.2e}, Effective={eff_lr:.2e}")
    
    def _log_epoch_summary(
        self,
        optimizer: torch.optim.Optimizer,
        metrics: Dict[str, float]
    ):
        """
        Log epoch summary.
        
        Args:
            optimizer: Optimizer
            metrics: Epoch metrics
        """
        logger.info(f"Epoch {self.epoch} LR Summary:")
        
        for i, pg in enumerate(optimizer.param_groups):
            lr = pg['lr']
            eff_lr = self.get_effective_lr(optimizer, i)
            logger.info(f"  Group {i}: LR={lr:.2e}, Effective={eff_lr:.2e}")
        
        if self.best_lr is not None:
            logger.info(f"  Best LR so far: {self.best_lr:.2e} (loss={self.best_loss:.4f})")
        
        # Suggest LR adjustment if loss is not improving
        if 'val_loss' in metrics and len(self.loss_history) > 100:
            recent_losses = self.loss_history[-100:]
            if np.std(recent_losses) < 0.001 * np.mean(recent_losses):
                logger.warning("Loss plateau detected. Consider adjusting learning rate.")
    
    def _generate_report(self) -> str:
        """
        Generate final LR monitoring report.
        
        Returns:
            Report string
        """
        report_lines = [
            "=" * 50,
            "Learning Rate Monitoring Report",
            "=" * 50,
            f"Total steps: {self.step}",
            f"Total epochs: {self.epoch}",
        ]
        
        if self.best_lr is not None:
            report_lines.append(f"Best LR: {self.best_lr:.2e} (loss={self.best_loss:.4f})")
        
        if self.lr_history:
            report_lines.append(f"LR range: [{min(self.lr_history):.2e}, {max(self.lr_history):.2e}]")
            
            # Suggest optimal range
            try:
                min_lr, max_lr = self.suggest_lr_range()
                report_lines.append(f"Suggested LR range: [{min_lr:.2e}, {max_lr:.2e}]")
            except Exception as e:
                logger.debug(f"Could not suggest LR range: {e}")
        
        return "\n".join(report_lines)
    
    def _create_visualization(self):
        """Create and save LR schedule visualization."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot LR schedule
            if self.lr_history:
                axes[0].plot(self.lr_history, 'b-', alpha=0.7)
                axes[0].set_xlabel('Step')
                axes[0].set_ylabel('Learning Rate')
                axes[0].set_title('Learning Rate Schedule')
                axes[0].set_yscale('log')
                axes[0].grid(True, alpha=0.3)
            
            # Plot loss vs LR
            if self.lr_history and self.loss_history:
                axes[1].plot(self.lr_history[:len(self.loss_history)], 
                           self.loss_history, 'r.', alpha=0.5)
                axes[1].set_xlabel('Learning Rate')
                axes[1].set_ylabel('Loss')
                axes[1].set_title('Loss vs Learning Rate')
                axes[1].set_xscale('log')
                axes[1].grid(True, alpha=0.3)
                
                # Mark best LR
                if self.best_lr is not None:
                    axes[1].axvline(x=self.best_lr, color='g', linestyle='--', 
                                  label=f'Best LR: {self.best_lr:.2e}')
                    axes[1].legend()
            
            plt.tight_layout()
            plt.savefig('outputs/lr_monitoring.png', dpi=100)
            plt.close()
            
            logger.info("LR visualization saved to outputs/lr_monitoring.png")
            
        except ImportError:
            logger.debug("Matplotlib not available for visualization")
    
    def _save_lr_history(self):
        """Save LR history to file."""
        import json
        from pathlib import Path
        
        output_dir = Path("outputs/training")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        history_data = {
            'lr_history': self.lr_history,
            'loss_history': self.loss_history[:len(self.lr_history)],
            'best_lr': self.best_lr,
            'best_loss': self.best_loss,
            'param_groups': self.param_groups_history
        }
        
        with open(output_dir / 'lr_history.json', 'w') as f:
            json.dump(history_data, f, indent=2, default=str)
        
        logger.info(f"LR history saved to {output_dir / 'lr_history.json'}")
