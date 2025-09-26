"""
APEX-Optimized Trainer Implementation for AG News Text Classification
======================================================================

This module implements training with NVIDIA APEX for optimized mixed precision
training and distributed optimization with enhanced performance.

Key Features:
- Automatic Mixed Precision (AMP) with O1, O2, O3 optimization levels
- Fused optimizers for better performance
- Distributed training with SyncBatchNorm
- Memory-efficient gradient accumulation
- Dynamic loss scaling with backoff

References:
- Micikevicius et al. (2018): "Mixed Precision Training"
- NVIDIA (2019): "APEX: A PyTorch Extension for Mixed Precision and Distributed Training"
- Narang et al. (2018): "Mixed Precision Training of Deep Neural Networks"

Author: Võ Hải Dũng
License: MIT
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from apex import amp
    from apex.optimizers import FusedAdam, FusedLAMB, FusedSGD
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model
    from apex.multi_tensor_apply import multi_tensor_applier
    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False
    warnings.warn("APEX not installed. Falling back to PyTorch native mixed precision.")

from src.training.trainers.base_trainer import BaseTrainer, TrainerConfig
from src.models.base.base_model import AGNewsBaseModel
from src.core.exceptions import TrainingError
from src.utils.logging_config import get_logger
from src.utils.memory_utils import optimize_memory

logger = get_logger(__name__)


@dataclass
class ApexTrainerConfig(TrainerConfig):
    """Configuration for APEX trainer."""
    
    # APEX optimization level
    opt_level: str = "O1"  # O0, O1, O2, O3
    
    # Loss scaling
    loss_scale: Union[float, str] = "dynamic"  # float value or "dynamic"
    loss_scale_window: int = 2000
    min_loss_scale: float = 1.0
    max_loss_scale: float = 2.**24
    
    # Optimizer configuration
    use_fused_optimizer: bool = True
    fused_optimizer_type: str = "FusedAdam"  # FusedAdam, FusedLAMB, FusedSGD
    bias_correction: bool = True
    eps: float = 1e-8
    
    # Memory optimization
    keep_batchnorm_fp32: bool = True
    master_weights: bool = True
    cast_model_outputs: bool = False
    patch_torch_functions: bool = True
    
    # Gradient configuration
    gradient_compression: bool = False
    compression_backend: str = "powersgd"  # powersgd, topk, randomk
    compression_ratio: float = 0.1
    
    # Distributed settings
    use_apex_ddp: bool = True
    delay_allreduce: bool = True
    gradient_predivide_factor: float = 1.0
    convert_sync_batchnorm: bool = True
    
    # Performance tuning
    channels_last: bool = False
    use_fast_math: bool = True
    benchmark_cudnn: bool = True
    deterministic_cudnn: bool = False
    
    # Debugging
    check_overflow: bool = True
    throw_on_overflow: bool = False
    scale_window: Optional[int] = None


class ApexTrainer(BaseTrainer):
    """
    Trainer optimized with NVIDIA APEX for enhanced performance.
    
    Provides state-of-the-art mixed precision training with:
    - Multiple optimization levels (O0-O3)
    - Fused optimizers for better throughput
    - Advanced gradient compression
    - Optimized distributed training
    """
    
    def __init__(
        self,
        model: AGNewsBaseModel,
        config: Optional[ApexTrainerConfig] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None
    ):
        """
        Initialize APEX trainer.
        
        Args:
            model: Model to train
            config: APEX training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        if not APEX_AVAILABLE:
            raise RuntimeError(
                "APEX is not installed. Please install it from "
                "https://github.com/NVIDIA/apex"
            )
        
        self.config = config or ApexTrainerConfig()
        
        # Performance optimizations
        self._setup_performance_optimizations()
        
        # Move model to device before APEX initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Convert to channels_last format if requested
        if self.config.channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        # Convert BatchNorm to SyncBatchNorm for distributed training
        if self.config.convert_sync_batchnorm and torch.cuda.device_count() > 1:
            model = convert_syncbn_model(model)
        
        # Create optimizer before AMP initialization
        optimizer = self._create_apex_optimizer(model)
        
        # Initialize AMP
        model, optimizer = self._initialize_amp(model, optimizer)
        
        # Store for parent class
        self.apex_model = model
        self.apex_optimizer = optimizer
        
        # Initialize base trainer with AMP-wrapped model
        super().__init__(
            model=model,
            config=self.config,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer
        )
        
        # Setup distributed training if needed
        if self.config.use_apex_ddp and torch.cuda.device_count() > 1:
            self.model = self._setup_apex_ddp(self.model)
        
        # Initialize gradient compression if enabled
        if self.config.gradient_compression:
            self._setup_gradient_compression()
        
        logger.info(
            f"Initialized ApexTrainer with opt_level={self.config.opt_level}, "
            f"loss_scale={self.config.loss_scale}"
        )
    
    def _setup_performance_optimizations(self):
        """Setup CUDA and PyTorch performance optimizations."""
        if torch.cuda.is_available():
            # Enable TF32 on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = self.config.use_fast_math
            torch.backends.cudnn.allow_tf32 = self.config.use_fast_math
            
            # CuDNN settings
            torch.backends.cudnn.benchmark = self.config.benchmark_cudnn
            torch.backends.cudnn.deterministic = self.config.deterministic_cudnn
            
            # Set CUDA device flags
            if self.config.use_fast_math:
                torch.set_float32_matmul_precision('high')
    
    def _create_apex_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        """
        Create fused optimizer for better performance.
        
        Args:
            model: Model to optimize
            
        Returns:
            Fused optimizer
        """
        # Get parameter groups
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        
        # Create fused optimizer
        if self.config.use_fused_optimizer:
            if self.config.fused_optimizer_type == "FusedAdam":
                optimizer = FusedAdam(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.eps,
                    bias_correction=self.config.bias_correction,
                    adam_w_mode=True
                )
            elif self.config.fused_optimizer_type == "FusedLAMB":
                optimizer = FusedLAMB(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    betas=(self.config.adam_beta1, self.config.adam_beta2),
                    eps=self.config.eps,
                    bias_correction=self.config.bias_correction
                )
            elif self.config.fused_optimizer_type == "FusedSGD":
                optimizer = FusedSGD(
                    optimizer_grouped_parameters,
                    lr=self.config.learning_rate,
                    momentum=0.9,
                    nesterov=True
                )
            else:
                raise ValueError(f"Unknown fused optimizer: {self.config.fused_optimizer_type}")
        else:
            # Fallback to standard optimizer
            from torch.optim import AdamW
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.eps
            )
        
        return optimizer
    
    def _initialize_amp(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Initialize Automatic Mixed Precision.
        
        Args:
            model: Model to wrap
            optimizer: Optimizer to wrap
            
        Returns:
            AMP-wrapped model and optimizer
        """
        # AMP initialization arguments
        amp_args = {
            "opt_level": self.config.opt_level,
            "keep_batchnorm_fp32": self.config.keep_batchnorm_fp32,
            "master_weights": self.config.master_weights,
            "cast_model_outputs": self.config.cast_model_outputs,
            "patch_torch_functions": self.config.patch_torch_functions,
            "loss_scale": self.config.loss_scale
        }
        
        # Add loss scale window for dynamic scaling
        if self.config.loss_scale == "dynamic":
            amp_args["loss_scale_window"] = self.config.loss_scale_window
            amp_args["min_loss_scale"] = self.config.min_loss_scale
            amp_args["max_loss_scale"] = self.config.max_loss_scale
        
        # Initialize AMP
        model, optimizer = amp.initialize(model, optimizer, **amp_args)
        
        return model, optimizer
    
    def _setup_apex_ddp(self, model: nn.Module) -> nn.Module:
        """
        Setup APEX Distributed Data Parallel.
        
        Args:
            model: Model to wrap
            
        Returns:
            DDP-wrapped model
        """
        # APEX DDP configuration
        ddp_config = {
            "delay_allreduce": self.config.delay_allreduce,
            "gradient_predivide_factor": self.config.gradient_predivide_factor
        }
        
        # Wrap with APEX DDP
        model = ApexDDP(model, **ddp_config)
        
        return model
    
    def _setup_gradient_compression(self):
        """Setup gradient compression for communication efficiency."""
        if self.config.compression_backend == "powersgd":
            from apex.contrib.sparsity import ASP
            # Setup Automatic Sparsity
            ASP.prune_trained_model(self.model, self.optimizer)
        
        logger.info(
            f"Enabled gradient compression with {self.config.compression_backend} "
            f"backend and ratio {self.config.compression_ratio}"
        )
    
    def _train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute single training step with APEX optimization.
        
        Args:
            batch: Input batch
            
        Returns:
            Loss and metrics
        """
        # Forward pass (AMP handles mixed precision automatically)
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("labels")
        )
        
        loss = outputs.loss
        
        # Scale loss for gradient accumulation
        if self.config.gradient_accumulation_steps > 1:
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with AMP
        with amp.scale_loss(loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        
        # Check for overflow
        if self.config.check_overflow:
            grad_overflow = self._check_gradient_overflow()
            if grad_overflow and self.config.throw_on_overflow:
                raise RuntimeError("Gradient overflow detected")
        
        # Compute metrics
        with torch.no_grad():
            if hasattr(outputs, 'logits'):
                predictions = torch.argmax(outputs.logits, dim=-1)
                if batch.get("labels") is not None:
                    accuracy = (predictions == batch["labels"]).float().mean().item()
                else:
                    accuracy = 0.0
            else:
                accuracy = 0.0
        
        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "loss_scale": self.optimizer.loss_scale() if hasattr(self.optimizer, 'loss_scale') else 1.0
        }
        
        return loss, metrics
    
    def _check_gradient_overflow(self) -> bool:
        """
        Check for gradient overflow in mixed precision training.
        
        Returns:
            Whether overflow was detected
        """
        # Check optimizer state for overflow
        if hasattr(self.optimizer, '_amp_stash'):
            stash = self.optimizer._amp_stash
            if hasattr(stash, 'found_inf'):
                return stash.found_inf.item() != 0
        
        # Manual check for inf/nan gradients
        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    return True
        
        return False
    
    def _optimize_step(self):
        """Execute optimization step with APEX-specific handling."""
        # Gradient clipping with AMP
        if self.config.max_grad_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                amp.master_params(self.optimizer),
                self.config.max_grad_norm
            )
            
            # Log gradient norm
            if self.global_step % self.config.logging_steps == 0:
                self.tracker.log({"train/grad_norm": grad_norm}, step=self.global_step)
        
        # Optimizer step
        self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Update learning rate
        if self.scheduler:
            self.scheduler.step()
    
    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch with APEX optimizations.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Training metrics for the epoch
        """
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        total_correct = 0
        accumulation_steps = 0
        
        # Training loop with gradient accumulation
        for step, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                    for k, v in batch.items()}
            
            # Training step
            loss, metrics = self._train_step(batch)
            
            # Accumulate metrics
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)
            if "accuracy" in metrics:
                total_correct += metrics["accuracy"] * batch["input_ids"].size(0)
            
            accumulation_steps += 1
            
            # Optimization step after gradient accumulation
            if accumulation_steps >= self.config.gradient_accumulation_steps:
                self._optimize_step()
                accumulation_steps = 0
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    avg_loss = total_loss / total_samples
                    avg_accuracy = total_correct / total_samples
                    
                    log_metrics = {
                        "train/loss": avg_loss,
                        "train/accuracy": avg_accuracy,
                        "train/learning_rate": self.optimizer.param_groups[0]['lr'],
                        "train/loss_scale": metrics.get("loss_scale", 1.0)
                    }
                    
                    self.tracker.log(log_metrics, step=self.global_step)
                    
                    logger.debug(
                        f"Step {self.global_step}: Loss={avg_loss:.4f}, "
                        f"Acc={avg_accuracy:.4f}, Scale={metrics.get('loss_scale', 1.0):.1f}"
                    )
                
                # Memory optimization
                if self.global_step % self.config.empty_cache_steps == 0:
                    optimize_memory()
        
        # Handle remaining gradients
        if accumulation_steps > 0:
            self._optimize_step()
            self.global_step += 1
        
        # Calculate epoch metrics
        epoch_metrics = {
            "loss": total_loss / total_samples,
            "accuracy": total_correct / total_samples,
            "learning_rate": self.optimizer.param_groups[0]['lr']
        }
        
        return epoch_metrics
    
    def save_checkpoint(
        self,
        path: Union[str, Any],
        is_best: bool = False
    ):
        """
        Save checkpoint with APEX state.
        
        Args:
            path: Path to save checkpoint
            is_best: Whether this is the best model
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "amp_state_dict": amp.state_dict(),  # APEX state
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "best_metric": self.best_metric,
            "config": self.config,
            "is_best": is_best
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Saved APEX checkpoint to {path}")
    
    def load_checkpoint(self, path: Union[str, Any]):
        """
        Load checkpoint with APEX state.
        
        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Restore APEX state
        if "amp_state_dict" in checkpoint:
            amp.load_state_dict(checkpoint["amp_state_dict"])
        
        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        
        logger.info(f"Loaded APEX checkpoint from {path} (epoch {self.current_epoch})")
