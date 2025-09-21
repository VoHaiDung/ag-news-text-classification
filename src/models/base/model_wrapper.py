"""
Model Wrapper for Enhanced Functionality
=========================================

This module implements model wrapper classes that provide additional functionality
around base models including optimization, monitoring, and deployment features.

The wrapper pattern follows principles from:
- Gamma et al. (1994): "Design Patterns" - Decorator/Wrapper Pattern
- Fowler (2002): "Patterns of Enterprise Application Architecture"
- Howard & Ruder (2018): "Universal Language Model Fine-tuning"

Key Features:
1. Mixed precision training support
2. Gradient checkpointing for memory efficiency
3. Model optimization (quantization, pruning)
4. Distributed training support
5. Model versioning and metadata tracking
6. Performance profiling and monitoring

Mathematical Foundation:
The wrapper maintains mathematical equivalence while adding operational features:
f_wrapped(x) = f_original(x) + additional_operations

Author: Võ Hải Dũng
License: MIT
"""

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
from contextlib import contextmanager
import json
import hashlib
from datetime import datetime
from collections import OrderedDict, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel

from src.core.types import (
    ModelOutput,
    BatchData,
    ModelConfig,
    PathLike,
    ModelInfo
)
from src.core.exceptions import (
    ModelError,
    ModelInitializationError,
    ModelLoadError,
    ModelSaveError,
    OptimizationError
)
from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.utils.logging_config import get_logger
from src.utils.memory_utils import get_memory_info, optimize_memory
from src.utils.profiling_utils import profile_model

logger = get_logger(__name__)


@dataclass
class WrapperConfig:
    """
    Configuration for model wrapper.
    
    Attributes:
        enable_mixed_precision: Enable automatic mixed precision training
        enable_gradient_checkpointing: Enable gradient checkpointing
        enable_profiling: Enable performance profiling
        enable_monitoring: Enable resource monitoring
        optimization_level: Optimization level (O0, O1, O2, O3)
        distributed_backend: Backend for distributed training
        model_parallel: Enable model parallelism
        checkpoint_activations: Checkpoint intermediate activations
        memory_efficient_attention: Use memory-efficient attention
        compile_model: Compile model with torch.compile (PyTorch 2.0+)
        quantization_config: Quantization configuration
        pruning_config: Pruning configuration
    """
    enable_mixed_precision: bool = False
    enable_gradient_checkpointing: bool = False
    enable_profiling: bool = False
    enable_monitoring: bool = False
    optimization_level: str = "O1"
    distributed_backend: Optional[str] = None
    model_parallel: bool = False
    checkpoint_activations: bool = False
    memory_efficient_attention: bool = False
    compile_model: bool = False
    quantization_config: Optional[Dict[str, Any]] = None
    pruning_config: Optional[Dict[str, Any]] = None
    metadata_tracking: bool = True
    save_optimizer_state: bool = True
    save_training_state: bool = True


class ModelWrapper(nn.Module):
    """
    Wrapper class that adds enhanced functionality to base models.
    
    This class implements the Decorator pattern to add features without
    modifying the original model implementation. It provides:
    
    1. Automatic mixed precision training
    2. Gradient checkpointing for memory efficiency
    3. Model optimization techniques
    4. Distributed training support
    5. Performance monitoring and profiling
    
    The wrapper maintains full compatibility with the wrapped model's
    interface while adding additional capabilities.
    """
    
    def __init__(
        self,
        model: AGNewsBaseModel,
        config: Optional[WrapperConfig] = None
    ):
        """
        Initialize model wrapper.
        
        Args:
            model: Base model to wrap
            config: Wrapper configuration
        """
        super().__init__()
        
        self.wrapped_model = model
        self.config = config or WrapperConfig()
        
        # Initialize components
        self._init_optimization()
        self._init_monitoring()
        self._init_metadata()
        
        # Training state
        self.training_state = {
            "global_step": 0,
            "epoch": 0,
            "best_metric": float("-inf"),
            "training_time": 0.0,
            "inference_calls": 0
        }
        
        logger.info(f"Initialized ModelWrapper for {model.__class__.__name__}")
    
    def _init_optimization(self):
        """Initialize optimization components."""
        # Mixed precision
        if self.config.enable_mixed_precision:
            self.scaler = GradScaler()
            logger.info("Enabled mixed precision training")
        else:
            self.scaler = None
        
        # Gradient checkpointing
        if self.config.enable_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Model compilation (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, "compile"):
            self.wrapped_model = torch.compile(self.wrapped_model)
            logger.info("Compiled model with torch.compile")
    
    def _init_monitoring(self):
        """Initialize monitoring components."""
        self.performance_metrics = defaultdict(list)
        self.resource_usage = defaultdict(list)
        self.gradient_stats = defaultdict(list)
        
        if self.config.enable_monitoring:
            self.monitor_interval = 100  # Log every N steps
            logger.info("Enabled resource monitoring")
    
    def _init_metadata(self):
        """Initialize metadata tracking."""
        self.metadata = {
            "creation_time": datetime.now().isoformat(),
            "model_class": self.wrapped_model.__class__.__name__,
            "wrapper_version": "1.0.0",
            "configuration": self.config.__dict__ if hasattr(self.config, "__dict__") else {},
            "model_hash": self._compute_model_hash(),
            "parameter_count": self._count_parameters()
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass with enhanced functionality.
        
        Implements the forward pass with additional features like mixed
        precision, profiling, and monitoring.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with additional metadata
        """
        # Start timing
        start_time = time.perf_counter()
        
        # Context managers for optimization
        contexts = []
        
        # Mixed precision context
        if self.config.enable_mixed_precision and self.training:
            contexts.append(autocast())
        
        # Profiling context
        if self.config.enable_profiling:
            contexts.append(self._profiling_context())
        
        # Apply contexts and forward
        with self._nested_contexts(contexts):
            outputs = self.wrapped_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        
        # Monitor performance
        if self.config.enable_monitoring:
            self._monitor_forward_pass(start_time, outputs)
        
        # Update training state
        if self.training:
            self.training_state["global_step"] += 1
        else:
            self.training_state["inference_calls"] += 1
        
        return outputs
    
    @contextmanager
    def _nested_contexts(self, contexts):
        """Apply nested context managers."""
        if not contexts:
            yield
            return
        
        with contexts[0]:
            with self._nested_contexts(contexts[1:]):
                yield
    
    @contextmanager
    def _profiling_context(self):
        """Context manager for profiling."""
        if hasattr(torch.profiler, "profile"):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True
            ) as prof:
                yield
                # Log profiling results
                if self.training_state["global_step"] % 100 == 0:
                    logger.debug(prof.key_averages().table(sort_by="cuda_time_total"))
        else:
            yield
    
    def _monitor_forward_pass(self, start_time: float, outputs: ModelOutputs):
        """Monitor forward pass performance."""
        # Timing
        forward_time = time.perf_counter() - start_time
        self.performance_metrics["forward_time"].append(forward_time)
        
        # Memory usage
        if torch.cuda.is_available():
            memory_info = get_memory_info()
            self.resource_usage["gpu_memory_mb"].append(
                memory_info["allocated"] / 1024 / 1024
            )
        
        # Loss tracking
        if outputs.loss is not None:
            self.performance_metrics["loss"].append(outputs.loss.item())
        
        # Log periodically
        if self.training_state["global_step"] % self.monitor_interval == 0:
            self._log_monitoring_stats()
    
    def _log_monitoring_stats(self):
        """Log monitoring statistics."""
        stats = {
            "step": self.training_state["global_step"],
            "avg_forward_time": sum(self.performance_metrics["forward_time"][-100:]) / 
                               min(100, len(self.performance_metrics["forward_time"])),
        }
        
        if self.performance_metrics["loss"]:
            stats["avg_loss"] = sum(self.performance_metrics["loss"][-100:]) / \
                               min(100, len(self.performance_metrics["loss"]))
        
        if self.resource_usage["gpu_memory_mb"]:
            stats["avg_gpu_memory_mb"] = sum(self.resource_usage["gpu_memory_mb"][-100:]) / \
                                        min(100, len(self.resource_usage["gpu_memory_mb"]))
        
        logger.debug(f"Performance stats: {stats}")
    
    def optimize_for_inference(
        self,
        quantization: Optional[str] = "dynamic",
        optimize_graph: bool = True,
        convert_to_torchscript: bool = False
    ) -> "ModelWrapper":
        """
        Optimize model for inference.
        
        Applies various optimization techniques to improve inference performance
        following best practices from:
        - Jacob et al. (2018): "Quantization and Training of Neural Networks"
        - ONNX Runtime optimization guidelines
        
        Args:
            quantization: Quantization type ("dynamic", "static", "qat", None)
            optimize_graph: Apply graph optimizations
            convert_to_torchscript: Convert to TorchScript
            
        Returns:
            Optimized model wrapper
            
        Raises:
            OptimizationError: If optimization fails
        """
        try:
            # Ensure eval mode
            self.eval()
            
            # Apply quantization
            if quantization:
                self._apply_quantization(quantization)
            
            # Graph optimization
            if optimize_graph and hasattr(torch, "jit"):
                self._optimize_graph()
            
            # TorchScript conversion
            if convert_to_torchscript:
                self.wrapped_model = self._convert_to_torchscript()
            
            # Update metadata
            self.metadata["optimizations"] = {
                "quantization": quantization,
                "graph_optimized": optimize_graph,
                "torchscript": convert_to_torchscript,
                "timestamp": datetime.now().isoformat()
            }
            
            logger.info("Model optimized for inference")
            return self
            
        except Exception as e:
            raise OptimizationError(f"Failed to optimize model: {e}")
    
    def _apply_quantization(self, quantization_type: str):
        """Apply quantization to model."""
        if quantization_type == "dynamic":
            # Dynamic quantization
            self.wrapped_model = torch.quantization.quantize_dynamic(
                self.wrapped_model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Applied dynamic quantization")
            
        elif quantization_type == "static":
            # Static quantization requires calibration
            logger.warning("Static quantization requires calibration data")
            
        elif quantization_type == "qat":
            # Quantization-aware training
            logger.warning("QAT requires training with fake quantization")
    
    def _optimize_graph(self):
        """Apply graph-level optimizations."""
        # Fuse operations
        if hasattr(torch.nn.utils, "fusion"):
            torch.nn.utils.fusion.fuse_conv_bn_eval(self.wrapped_model)
        
        # Remove unnecessary operations
        self.wrapped_model.eval()
        
        logger.info("Applied graph optimizations")
    
    def _convert_to_torchscript(self) -> torch.jit.ScriptModule:
        """Convert model to TorchScript."""
        # Create example input
        example_input = torch.randint(0, 1000, (1, 512))
        example_mask = torch.ones(1, 512)
        
        # Trace model
        traced = torch.jit.trace(
            self.wrapped_model,
            (example_input, example_mask)
        )
        
        logger.info("Converted to TorchScript")
        return traced
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.wrapped_model, "gradient_checkpointing_enable"):
            self.wrapped_model.gradient_checkpointing_enable()
            logger.info("Enabled gradient checkpointing")
        else:
            logger.warning("Model does not support gradient checkpointing")
    
    def apply_pruning(
        self,
        pruning_method: str = "magnitude",
        sparsity: float = 0.3
    ):
        """
        Apply pruning to reduce model size.
        
        Implements structured and unstructured pruning following:
        - Han et al. (2015): "Deep Compression"
        - Louizos et al. (2018): "Learning Sparse Neural Networks"
        
        Args:
            pruning_method: Pruning method ("magnitude", "random", "structured")
            sparsity: Target sparsity level
        """
        import torch.nn.utils.prune as prune
        
        if pruning_method == "magnitude":
            # Magnitude-based pruning
            for name, module in self.wrapped_model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prune.l1_unstructured(
                        module,
                        name="weight",
                        amount=sparsity
                    )
        
        elif pruning_method == "structured":
            # Structured pruning (remove entire channels/neurons)
            for name, module in self.wrapped_model.named_modules():
                if isinstance(module, nn.Linear):
                    prune.ln_structured(
                        module,
                        name="weight",
                        amount=sparsity,
                        n=2,
                        dim=0
                    )
        
        # Update metadata
        self.metadata["pruning"] = {
            "method": pruning_method,
            "sparsity": sparsity,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Applied {pruning_method} pruning with {sparsity:.1%} sparsity")
    
    def profile_performance(
        self,
        input_shape: Tuple[int, ...] = (1, 512),
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Profile model performance.
        
        Args:
            input_shape: Shape of input tensor
            num_iterations: Number of profiling iterations
            warmup_iterations: Number of warmup iterations
            
        Returns:
            Performance metrics dictionary
        """
        self.eval()
        device = next(self.parameters()).device
        
        # Create dummy input
        dummy_input = torch.randint(0, 1000, input_shape).to(device)
        dummy_mask = torch.ones(input_shape).to(device)
        
        # Warmup
        for _ in range(warmup_iterations):
            with torch.no_grad():
                _ = self(dummy_input, dummy_mask)
        
        # Profile
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        forward_times = []
        memory_usage = []
        
        for _ in range(num_iterations):
            start = time.perf_counter()
            
            with torch.no_grad():
                _ = self(dummy_input, dummy_mask)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            forward_times.append(time.perf_counter() - start)
            
            if torch.cuda.is_available():
                memory_usage.append(torch.cuda.memory_allocated() / 1024 / 1024)
        
        # Compute statistics
        import numpy as np
        
        metrics = {
            "mean_latency_ms": np.mean(forward_times) * 1000,
            "std_latency_ms": np.std(forward_times) * 1000,
            "p50_latency_ms": np.percentile(forward_times, 50) * 1000,
            "p95_latency_ms": np.percentile(forward_times, 95) * 1000,
            "p99_latency_ms": np.percentile(forward_times, 99) * 1000,
            "throughput_samples_per_sec": 1 / np.mean(forward_times),
        }
        
        if memory_usage:
            metrics["mean_memory_mb"] = np.mean(memory_usage)
            metrics["peak_memory_mb"] = np.max(memory_usage)
        
        # Model complexity
        metrics["parameters_total"] = self._count_parameters()["total"]
        metrics["parameters_trainable"] = self._count_parameters()["trainable"]
        
        return metrics
    
    def save_checkpoint(
        self,
        checkpoint_path: PathLike,
        save_optimizer: Optional[Any] = None,
        save_scheduler: Optional[Any] = None,
        **kwargs
    ):
        """
        Save complete checkpoint with all components.
        
        Args:
            checkpoint_path: Path to save checkpoint
            save_optimizer: Optimizer state to save
            save_scheduler: Scheduler state to save
            **kwargs: Additional items to save
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            "model_state_dict": self.wrapped_model.state_dict(),
            "wrapper_config": self.config,
            "training_state": self.training_state,
            "metadata": self.metadata,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add optimizer state
        if save_optimizer and self.config.save_optimizer_state:
            checkpoint["optimizer_state_dict"] = save_optimizer.state_dict()
        
        # Add scheduler state
        if save_scheduler:
            checkpoint["scheduler_state_dict"] = save_scheduler.state_dict()
        
        # Add additional items
        checkpoint.update(kwargs)
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        
        # Save metadata separately
        metadata_path = checkpoint_path.parent / "checkpoint_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(
                {
                    "checkpoint_path": str(checkpoint_path),
                    "timestamp": checkpoint["timestamp"],
                    "training_state": self.training_state,
                    "metadata": self.metadata
                },
                f,
                indent=2
            )
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: PathLike,
        load_optimizer: Optional[Any] = None,
        load_scheduler: Optional[Any] = None,
        strict: bool = True
    ):
        """
        Load checkpoint and restore state.
        
        Args:
            checkpoint_path: Path to checkpoint
            load_optimizer: Optimizer to load state into
            load_scheduler: Scheduler to load state into
            strict: Strict state dict loading
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise ModelLoadError(f"Checkpoint not found: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load model state
        self.wrapped_model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=strict
        )
        
        # Restore training state
        if "training_state" in checkpoint:
            self.training_state.update(checkpoint["training_state"])
        
        # Restore metadata
        if "metadata" in checkpoint:
            self.metadata.update(checkpoint["metadata"])
        
        # Load optimizer state
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            load_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if load_scheduler and "scheduler_state_dict" in checkpoint:
            load_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    def _compute_model_hash(self) -> str:
        """Compute hash of model parameters for versioning."""
        hasher = hashlib.sha256()
        
        for param in self.wrapped_model.parameters():
            hasher.update(param.data.cpu().numpy().tobytes())
        
        return hasher.hexdigest()[:16]
    
    def _count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total = sum(p.numel() for p in self.wrapped_model.parameters())
        trainable = sum(p.numel() for p in self.wrapped_model.parameters() 
                       if p.requires_grad)
        
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable
        }
    
    def get_model_info(self) -> ModelInfo:
        """
        Get comprehensive model information.
        
        Returns:
            Model information dictionary
        """
        param_count = self._count_parameters()
        
        return {
            "name": self.wrapped_model.__class__.__name__,
            "type": "wrapped_model",
            "num_parameters": param_count["total"],
            "architecture": str(self.wrapped_model.__class__),
            "pretrained": True,
            "metadata": {
                **self.metadata,
                "trainable_parameters": param_count["trainable"],
                "frozen_parameters": param_count["frozen"],
                "training_steps": self.training_state["global_step"],
                "inference_calls": self.training_state["inference_calls"],
                "wrapper_config": self.config.__dict__ if hasattr(self.config, "__dict__") else {}
            }
        }
    
    def to(self, device: Union[str, torch.device]) -> "ModelWrapper":
        """
        Move model to device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.wrapped_model = self.wrapped_model.to(device)
        return self
    
    def train(self, mode: bool = True) -> "ModelWrapper":
        """Set training mode."""
        self.wrapped_model.train(mode)
        return super().train(mode)
    
    def eval(self) -> "ModelWrapper":
        """Set evaluation mode."""
        self.wrapped_model.eval()
        return super().eval()
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.wrapped_model.parameters()).device


class DistributedModelWrapper(ModelWrapper):
    """
    Extended wrapper for distributed training.
    
    Provides additional functionality for distributed training scenarios
    including data parallel and model parallel strategies.
    """
    
    def __init__(
        self,
        model: AGNewsBaseModel,
        config: Optional[WrapperConfig] = None,
        world_size: int = 1,
        rank: int = 0,
        backend: str = "nccl"
    ):
        """
        Initialize distributed wrapper.
        
        Args:
            model: Base model to wrap
            config: Wrapper configuration
            world_size: Number of processes
            rank: Process rank
            backend: Distribution backend
        """
        super().__init__(model, config)
        
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        
        if world_size > 1:
            self._setup_distributed()
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if self.config.model_parallel:
            # Model parallel setup
            logger.info("Setting up model parallel training")
            # Implementation depends on specific requirements
        else:
            # Data parallel setup
            self.wrapped_model = DistributedDataParallel(
                self.wrapped_model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True
            )
            logger.info(f"Setup DDP on rank {self.rank}/{self.world_size}")
    
    def all_reduce_gradients(self):
        """Synchronize gradients across processes."""
        if self.world_size > 1:
            for param in self.wrapped_model.parameters():
                if param.grad is not None:
                    torch.distributed.all_reduce(param.grad.data)
                    param.grad.data /= self.world_size
    
    def broadcast_parameters(self):
        """Broadcast parameters from rank 0 to all processes."""
        if self.world_size > 1:
            for param in self.wrapped_model.parameters():
                torch.distributed.broadcast(param.data, src=0)


# Export public API
__all__ = [
    "WrapperConfig",
    "ModelWrapper",
    "DistributedModelWrapper"
]
