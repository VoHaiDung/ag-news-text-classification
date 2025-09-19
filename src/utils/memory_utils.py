"""
Memory management utilities for AG News Text Classification Framework.

Provides utilities for memory optimization, monitoring, and profiling
to enable efficient training and inference on resource-constrained systems.

References:
    - Rajbhandari, S., et al. (2020). "ZeRO: Memory optimizations toward 
      training trillion parameter models". SC20.
    - Chen, T., et al. (2016). "Training deep nets with sublinear memory cost".
      arXiv preprint arXiv:1604.06174.

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
"""

import gc
import logging
import psutil
import tracemalloc
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass
import warnings

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Container for memory statistics."""
    
    cpu_memory_used: float  # MB
    cpu_memory_available: float  # MB
    cpu_memory_percent: float  # Percentage
    gpu_memory_used: Optional[float] = None  # MB
    gpu_memory_available: Optional[float] = None  # MB
    gpu_memory_reserved: Optional[float] = None  # MB
    gpu_memory_percent: Optional[float] = None  # Percentage


class MemoryMonitor:
    """
    Monitor memory usage during training and inference.
    
    Tracks both CPU and GPU memory consumption with detailed statistics.
    """
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        log_interval: int = 100,
        track_peak: bool = True,
    ):
        """
        Initialize memory monitor.
        
        Args:
            device: Device to monitor (None for auto-detect)
            log_interval: Logging interval in steps
            track_peak: Whether to track peak memory usage
        """
        self.device = device or (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.log_interval = log_interval
        self.track_peak = track_peak
        
        self.step_count = 0
        self.memory_history = []
        self.peak_memory = {
            "cpu": 0.0,
            "gpu": 0.0,
        }
        
        # Start CPU memory tracking
        tracemalloc.start()
        
        # Reset GPU memory statistics
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory statistics.
        
        Returns:
            Memory statistics
        """
        # CPU memory
        cpu_info = psutil.virtual_memory()
        cpu_memory_used = (cpu_info.total - cpu_info.available) / (1024 ** 2)  # MB
        cpu_memory_available = cpu_info.available / (1024 ** 2)  # MB
        cpu_memory_percent = cpu_info.percent
        
        stats = MemoryStats(
            cpu_memory_used=cpu_memory_used,
            cpu_memory_available=cpu_memory_available,
            cpu_memory_percent=cpu_memory_percent,
        )
        
        # GPU memory
        if self.device.type == "cuda":
            gpu_memory_used = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
            gpu_memory_reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)  # MB
            
            gpu_props = torch.cuda.get_device_properties(self.device)
            gpu_memory_total = gpu_props.total_memory / (1024 ** 2)  # MB
            gpu_memory_available = gpu_memory_total - gpu_memory_used
            gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
            
            stats.gpu_memory_used = gpu_memory_used
            stats.gpu_memory_available = gpu_memory_available
            stats.gpu_memory_reserved = gpu_memory_reserved
            stats.gpu_memory_percent = gpu_memory_percent
            
            # Update peak memory
            if self.track_peak:
                self.peak_memory["gpu"] = max(
                    self.peak_memory["gpu"],
                    gpu_memory_used
                )
        
        # Update peak CPU memory
        if self.track_peak:
            self.peak_memory["cpu"] = max(
                self.peak_memory["cpu"],
                cpu_memory_used
            )
        
        return stats
    
    def log_memory(self, prefix: str = "", force: bool = False):
        """
        Log memory usage.
        
        Args:
            prefix: Prefix for log message
            force: Force logging regardless of interval
        """
        self.step_count += 1
        
        if not force and self.step_count % self.log_interval != 0:
            return
        
        stats = self.get_memory_stats()
        self.memory_history.append(stats)
        
        # Log CPU memory
        logger.info(
            f"{prefix}CPU Memory: {stats.cpu_memory_used:.1f}MB used "
            f"({stats.cpu_memory_percent:.1f}%), "
            f"{stats.cpu_memory_available:.1f}MB available"
        )
        
        # Log GPU memory
        if stats.gpu_memory_used is not None:
            logger.info(
                f"{prefix}GPU Memory: {stats.gpu_memory_used:.1f}MB used "
                f"({stats.gpu_memory_percent:.1f}%), "
                f"{stats.gpu_memory_available:.1f}MB available, "
                f"{stats.gpu_memory_reserved:.1f}MB reserved"
            )
    
    def get_peak_memory(self) -> Dict[str, float]:
        """
        Get peak memory usage.
        
        Returns:
            Dictionary with peak memory usage
        """
        return self.peak_memory.copy()
    
    def reset(self):
        """Reset memory monitoring."""
        self.step_count = 0
        self.memory_history.clear()
        self.peak_memory = {"cpu": 0.0, "gpu": 0.0}
        
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get memory usage summary.
        
        Returns:
            Summary dictionary
        """
        if not self.memory_history:
            return {}
        
        cpu_usage = [s.cpu_memory_used for s in self.memory_history]
        gpu_usage = [
            s.gpu_memory_used for s in self.memory_history
            if s.gpu_memory_used is not None
        ]
        
        summary = {
            "cpu": {
                "mean_mb": np.mean(cpu_usage),
                "max_mb": np.max(cpu_usage),
                "min_mb": np.min(cpu_usage),
                "std_mb": np.std(cpu_usage),
                "peak_mb": self.peak_memory["cpu"],
            }
        }
        
        if gpu_usage:
            summary["gpu"] = {
                "mean_mb": np.mean(gpu_usage),
                "max_mb": np.max(gpu_usage),
                "min_mb": np.min(gpu_usage),
                "std_mb": np.std(gpu_usage),
                "peak_mb": self.peak_memory["gpu"],
            }
        
        return summary


def get_memory_usage(device: Optional[torch.device] = None) -> MemoryStats:
    """
    Get current memory usage.
    
    Args:
        device: Device to check (None for auto-detect)
        
    Returns:
        Memory statistics
    """
    monitor = MemoryMonitor(device=device)
    return monitor.get_memory_stats()


def log_memory_usage(
    prefix: str = "",
    device: Optional[torch.device] = None
):
    """
    Log current memory usage.
    
    Args:
        prefix: Prefix for log message
        device: Device to check
    """
    stats = get_memory_usage(device)
    
    logger.info(
        f"{prefix}CPU: {stats.cpu_memory_used:.1f}MB "
        f"({stats.cpu_memory_percent:.1f}%)"
    )
    
    if stats.gpu_memory_used is not None:
        logger.info(
            f"{prefix}GPU: {stats.gpu_memory_used:.1f}MB "
            f"({stats.gpu_memory_percent:.1f}%)"
        )


def optimize_memory():
    """Optimize memory by clearing caches and running garbage collection."""
    # Python garbage collection
    gc.collect()
    
    # PyTorch cache clearing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    logger.debug("Memory optimized: caches cleared and garbage collected")


def clear_memory(aggressive: bool = False):
    """
    Clear memory with optional aggressive cleaning.
    
    Args:
        aggressive: Whether to use aggressive cleaning
    """
    # Standard cleanup
    optimize_memory()
    
    if aggressive:
        # Clear all matplotlib figures
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except ImportError:
            pass
        
        # Clear IPython output
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython is not None:
                ipython.magic("reset -f")
        except:
            pass
        
        # Multiple garbage collection passes
        for _ in range(3):
            gc.collect()
        
        logger.debug("Aggressive memory clearing completed")


def estimate_model_memory(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    dtype: torch.dtype = torch.float32,
    include_gradients: bool = True,
    include_optimizer: bool = True,
) -> Dict[str, float]:
    """
    Estimate memory requirements for model.
    
    Args:
        model: Model to estimate
        input_shape: Input tensor shape (without batch dimension)
        batch_size: Batch size
        dtype: Data type
        include_gradients: Include gradient memory
        include_optimizer: Include optimizer state memory
        
    Returns:
        Memory estimates in MB
    """
    # Calculate parameter memory
    param_memory = 0
    for param in model.parameters():
        param_memory += param.numel() * param.element_size()
    
    param_memory_mb = param_memory / (1024 ** 2)
    
    # Calculate activation memory (rough estimate)
    # This is a simplified estimation
    activation_memory = 0
    
    # Estimate based on input size
    input_memory = batch_size * np.prod(input_shape) * dtype.itemsize
    activation_memory += input_memory
    
    # Add intermediate activations (rough multiplier)
    activation_memory *= 3  # Conservative estimate
    activation_memory_mb = activation_memory / (1024 ** 2)
    
    # Total memory
    total_memory_mb = param_memory_mb + activation_memory_mb
    
    # Add gradient memory
    if include_gradients:
        gradient_memory_mb = param_memory_mb
        total_memory_mb += gradient_memory_mb
    else:
        gradient_memory_mb = 0
    
    # Add optimizer state memory (Adam uses 2x parameters)
    if include_optimizer:
        optimizer_memory_mb = param_memory_mb * 2
        total_memory_mb += optimizer_memory_mb
    else:
        optimizer_memory_mb = 0
    
    return {
        "parameters_mb": param_memory_mb,
        "activations_mb": activation_memory_mb,
        "gradients_mb": gradient_memory_mb,
        "optimizer_mb": optimizer_memory_mb,
        "total_mb": total_memory_mb,
    }


@contextmanager
def profile_memory_usage(
    tag: str = "operation",
    device: Optional[torch.device] = None,
    log_result: bool = True,
):
    """
    Context manager for profiling memory usage.
    
    Args:
        tag: Tag for the operation
        device: Device to profile
        log_result: Whether to log results
        
    Yields:
        Memory statistics before operation
    """
    monitor = MemoryMonitor(device=device)
    
    # Get initial memory
    stats_before = monitor.get_memory_stats()
    
    # Clear caches for accurate measurement
    optimize_memory()
    
    try:
        yield stats_before
    finally:
        # Get final memory
        stats_after = monitor.get_memory_stats()
        
        # Calculate differences
        cpu_diff = stats_after.cpu_memory_used - stats_before.cpu_memory_used
        
        if log_result:
            logger.info(
                f"[{tag}] CPU memory change: {cpu_diff:+.1f}MB "
                f"({stats_before.cpu_memory_used:.1f} -> "
                f"{stats_after.cpu_memory_used:.1f})"
            )
        
        if stats_after.gpu_memory_used is not None:
            gpu_diff = stats_after.gpu_memory_used - stats_before.gpu_memory_used
            
            if log_result:
                logger.info(
                    f"[{tag}] GPU memory change: {gpu_diff:+.1f}MB "
                    f"({stats_before.gpu_memory_used:.1f} -> "
                    f"{stats_after.gpu_memory_used:.1f})"
                )


def enable_memory_efficient_mode():
    """Enable memory-efficient settings for PyTorch."""
    # Enable cudNN autotuner
    torch.backends.cudnn.benchmark = True
    
    # Enable TF32 for Ampere GPUs
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Set memory fraction
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Enable gradient checkpointing hint
    logger.info(
        "Memory efficient mode enabled. Consider using gradient checkpointing "
        "for large models."
    )


def get_gpu_memory_info(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get detailed GPU memory information.
    
    Args:
        device: GPU device
        
    Returns:
        Dictionary with memory information in MB
    """
    if not torch.cuda.is_available():
        return {}
    
    if device is None:
        device = torch.device("cuda")
    
    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
        "reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
        "free_mb": (
            torch.cuda.get_device_properties(device).total_memory -
            torch.cuda.memory_allocated(device)
        ) / (1024 ** 2),
        "total_mb": torch.cuda.get_device_properties(device).total_memory / (1024 ** 2),
    }


def memory_efficient_decorator(func: Callable) -> Callable:
    """
    Decorator for memory-efficient function execution.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Clear memory before execution
        optimize_memory()
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Clear memory after execution
            optimize_memory()
        
        return result
    
    return wrapper


class MemoryOptimizer:
    """
    Memory optimizer for training and inference.
    
    Implements various memory optimization techniques including
    gradient accumulation, mixed precision, and gradient checkpointing.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        enable_mixed_precision: bool = True,
        enable_gradient_checkpointing: bool = False,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Initialize memory optimizer.
        
        Args:
            model: Model to optimize
            enable_mixed_precision: Use mixed precision training
            enable_gradient_checkpointing: Use gradient checkpointing
            gradient_accumulation_steps: Gradient accumulation steps
        """
        self.model = model
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Setup mixed precision
        if self.enable_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        
        # Setup gradient checkpointing
        if self.enable_gradient_checkpointing:
            self._enable_gradient_checkpointing()
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for supported models."""
        # Check if model supports gradient checkpointing
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
        else:
            logger.warning(
                "Model does not support gradient checkpointing. "
                "Consider implementing checkpoint() in forward pass."
            )
    
    def optimize_batch_size(
        self,
        initial_batch_size: int,
        max_batch_size: int,
        input_shape: Tuple[int, ...],
    ) -> int:
        """
        Find optimal batch size for available memory.
        
        Args:
            initial_batch_size: Starting batch size
            max_batch_size: Maximum batch size to try
            input_shape: Input tensor shape
            
        Returns:
            Optimal batch size
        """
        batch_size = initial_batch_size
        optimal_batch_size = initial_batch_size
        
        while batch_size <= max_batch_size:
            try:
                # Create dummy batch
                dummy_input = torch.randn(
                    batch_size,
                    *input_shape,
                    device=next(self.model.parameters()).device
                )
                
                # Try forward pass
                with torch.no_grad():
                    _ = self.model(dummy_input)
                
                # If successful, update optimal batch size
                optimal_batch_size = batch_size
                batch_size *= 2
                
                # Clear memory
                del dummy_input
                optimize_memory()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.debug(f"Batch size {batch_size} too large")
                    break
                else:
                    raise
        
        logger.info(f"Optimal batch size: {optimal_batch_size}")
        return optimal_batch_size


# Export public API
__all__ = [
    "MemoryMonitor",
    "MemoryStats",
    "MemoryOptimizer",
    "get_memory_usage",
    "log_memory_usage",
    "optimize_memory",
    "clear_memory",
    "estimate_model_memory",
    "profile_memory_usage",
    "enable_memory_efficient_mode",
    "get_gpu_memory_info",
    "memory_efficient_decorator",
]
