"""
Performance profiling utilities for AG News Text Classification Framework.

Provides tools for profiling model training, inference, and optimization
to identify performance bottlenecks and optimization opportunities.

References:
    - Paszke, A., et al. (2019). "PyTorch: An imperative style, high-performance 
      deep learning library". NeurIPS.
    - NVIDIA. (2021). "Deep Learning Profiler Developer Guide".

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
"""

import time
import logging
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from functools import wraps
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import json
import warnings

import torch
import torch.profiler as profiler
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ProfileResult:
    """Container for profiling results."""
    
    name: str
    duration_ms: float
    cpu_time_ms: float
    cuda_time_ms: Optional[float] = None
    memory_mb: Optional[float] = None
    flops: Optional[int] = None
    num_calls: int = 1
    children: List["ProfileResult"] = field(default_factory=list)


class Timer:
    """
    High-precision timer for performance measurement.
    
    Supports both CPU and CUDA timing with synchronization.
    """
    
    def __init__(
        self,
        name: str = "Timer",
        cuda_sync: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize timer.
        
        Args:
            name: Timer name
            cuda_sync: Whether to synchronize CUDA
            verbose: Whether to print results
        """
        self.name = name
        self.cuda_sync = cuda_sync
        self.verbose = verbose
        
        self.start_time = None
        self.elapsed_time = None
        self.history = []
    
    def start(self):
        """Start timing."""
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.start_time = time.perf_counter()
        return self
    
    def stop(self) -> float:
        """
        Stop timing and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            raise RuntimeError("Timer not started")
        
        if self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        
        self.elapsed_time = time.perf_counter() - self.start_time
        self.history.append(self.elapsed_time)
        
        if self.verbose:
            logger.info(f"[{self.name}] Elapsed: {self.elapsed_time*1000:.2f}ms")
        
        self.start_time = None
        return self.elapsed_time
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get timing statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.history:
            return {}
        
        times_ms = [t * 1000 for t in self.history]
        
        return {
            "mean_ms": np.mean(times_ms),
            "std_ms": np.std(times_ms),
            "min_ms": np.min(times_ms),
            "max_ms": np.max(times_ms),
            "median_ms": np.median(times_ms),
            "total_ms": np.sum(times_ms),
            "count": len(times_ms),
        }
    
    def reset(self):
        """Reset timer history."""
        self.history.clear()
        self.start_time = None
        self.elapsed_time = None


class Profiler:
    """
    Comprehensive profiler for deep learning models.
    
    Profiles execution time, memory usage, and computational operations.
    """
    
    def __init__(
        self,
        enabled: bool = True,
        use_cuda: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = False,
        with_modules: bool = True,
    ):
        """
        Initialize profiler.
        
        Args:
            enabled: Whether profiling is enabled
            use_cuda: Profile CUDA operations
            profile_memory: Profile memory usage
            with_stack: Record stack traces
            with_flops: Calculate FLOPs
            with_modules: Profile module hierarchy
        """
        self.enabled = enabled
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        
        self.profiler = None
        self.results = []
    
    def start(self):
        """Start profiling."""
        if not self.enabled:
            return
        
        activities = [profiler.ProfilerActivity.CPU]
        if self.use_cuda:
            activities.append(profiler.ProfilerActivity.CUDA)
        
        self.profiler = profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules,
        )
        
        self.profiler.__enter__()
        logger.debug("Profiling started")
    
    def stop(self):
        """Stop profiling and collect results."""
        if not self.enabled or self.profiler is None:
            return
        
        self.profiler.__exit__(None, None, None)
        
        # Process results
        self._process_results()
        
        logger.debug("Profiling stopped")
    
    def _process_results(self):
        """Process profiling results."""
        if self.profiler is None:
            return
        
        # Get key averages
        key_averages = self.profiler.key_averages()
        
        for event in key_averages:
            result = ProfileResult(
                name=event.key,
                duration_ms=event.cpu_time_total / 1000,  # Convert to ms
                cpu_time_ms=event.cpu_time_total / 1000,
                cuda_time_ms=event.cuda_time_total / 1000 if self.use_cuda else None,
                memory_mb=(
                    event.cpu_memory_usage / (1024 ** 2)
                    if self.profile_memory and event.cpu_memory_usage > 0
                    else None
                ),
                flops=event.flops if self.with_flops else None,
                num_calls=event.count,
            )
            self.results.append(result)
    
    def print_summary(self, top_k: int = 10):
        """
        Print profiling summary.
        
        Args:
            top_k: Number of top operations to show
        """
        if not self.results:
            logger.warning("No profiling results available")
            return
        
        # Sort by total time
        sorted_results = sorted(
            self.results,
            key=lambda x: x.duration_ms,
            reverse=True
        )[:top_k]
        
        logger.info("=" * 80)
        logger.info("Profiling Summary (Top {} operations)".format(top_k))
        logger.info("=" * 80)
        
        for i, result in enumerate(sorted_results, 1):
            logger.info(
                f"{i}. {result.name}: "
                f"CPU={result.cpu_time_ms:.2f}ms"
                + (f", CUDA={result.cuda_time_ms:.2f}ms" if result.cuda_time_ms else "")
                + (f", Memory={result.memory_mb:.2f}MB" if result.memory_mb else "")
                + f", Calls={result.num_calls}"
            )
        
        logger.info("=" * 80)
    
    def export_chrome_trace(self, filepath: Union[str, Path]):
        """
        Export profiling results for Chrome tracing.
        
        Args:
            filepath: Output file path
        """
        if self.profiler is None:
            logger.warning("No profiling data to export")
            return
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        self.profiler.export_chrome_trace(str(filepath))
        logger.info(f"Chrome trace exported to {filepath}")
    
    def export_results(self, filepath: Union[str, Path]):
        """
        Export profiling results to JSON.
        
        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        results_dict = []
        for result in self.results:
            results_dict.append({
                "name": result.name,
                "duration_ms": result.duration_ms,
                "cpu_time_ms": result.cpu_time_ms,
                "cuda_time_ms": result.cuda_time_ms,
                "memory_mb": result.memory_mb,
                "flops": result.flops,
                "num_calls": result.num_calls,
            })
        
        with open(filepath, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results exported to {filepath}")
    
    def get_summary_df(self) -> pd.DataFrame:
        """
        Get profiling summary as DataFrame.
        
        Returns:
            DataFrame with profiling results
        """
        if not self.results:
            return pd.DataFrame()
        
        data = []
        for result in self.results:
            data.append({
                "Operation": result.name,
                "CPU Time (ms)": result.cpu_time_ms,
                "CUDA Time (ms)": result.cuda_time_ms,
                "Memory (MB)": result.memory_mb,
                "FLOPs": result.flops,
                "Calls": result.num_calls,
            })
        
        return pd.DataFrame(data)


def profile_function(
    func: Optional[Callable] = None,
    name: Optional[str] = None,
    profile_memory: bool = True,
    print_summary: bool = True,
):
    """
    Decorator for profiling functions.
    
    Args:
        func: Function to profile
        name: Profile name
        profile_memory: Whether to profile memory
        print_summary: Whether to print summary
        
    Returns:
        Decorated function or decorator
    """
    def decorator(f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            prof_name = name or f.__name__
            
            with Timer(prof_name) as timer:
                # Profile with PyTorch profiler
                prof = Profiler(
                    profile_memory=profile_memory,
                    with_modules=False,
                )
                
                prof.start()
                result = f(*args, **kwargs)
                prof.stop()
                
                if print_summary:
                    prof.print_summary(top_k=5)
            
            return result
        
        return wrapper
    
    if func is None:
        return decorator
    else:
        return decorator(func)


def profile_model(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    batch_size: int = 1,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Profile model performance.
    
    Args:
        model: Model to profile
        input_shape: Input tensor shape
        batch_size: Batch size
        num_iterations: Number of iterations
        warmup_iterations: Warmup iterations
        device: Device to use
        
    Returns:
        Profiling results
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape, device=device)
    
    # Warmup
    logger.info(f"Running {warmup_iterations} warmup iterations...")
    for _ in range(warmup_iterations):
        with torch.no_grad():
            _ = model(dummy_input)
    
    # Profile
    logger.info(f"Profiling {num_iterations} iterations...")
    
    prof = Profiler(
        profile_memory=True,
        with_flops=True,
        with_modules=True,
    )
    
    timer = Timer("Model Forward", cuda_sync=True, verbose=False)
    
    prof.start()
    
    for _ in range(num_iterations):
        with torch.no_grad():
            timer.start()
            _ = model(dummy_input)
            timer.stop()
    
    prof.stop()
    
    # Get statistics
    timing_stats = timer.get_statistics()
    
    # Calculate throughput
    throughput = batch_size * 1000 / timing_stats["mean_ms"]  # samples/sec
    
    results = {
        "timing": timing_stats,
        "throughput_samples_per_sec": throughput,
        "profiler_results": prof.results[:10],  # Top 10 operations
    }
    
    # Print summary
    logger.info("=" * 80)
    logger.info("Model Profiling Results")
    logger.info("=" * 80)
    logger.info(f"Mean latency: {timing_stats['mean_ms']:.2f}ms")
    logger.info(f"Throughput: {throughput:.2f} samples/sec")
    logger.info("=" * 80)
    
    prof.print_summary(top_k=10)
    
    return results


def benchmark_function(
    func: Callable,
    *args,
    num_iterations: int = 100,
    warmup_iterations: int = 10,
    return_results: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Benchmark function performance.
    
    Args:
        func: Function to benchmark
        *args: Function arguments
        num_iterations: Number of iterations
        warmup_iterations: Warmup iterations
        return_results: Whether to return function results
        **kwargs: Function keyword arguments
        
    Returns:
        Benchmark results
    """
    # Warmup
    for _ in range(warmup_iterations):
        _ = func(*args, **kwargs)
    
    # Benchmark
    timer = Timer(func.__name__, cuda_sync=True, verbose=False)
    results = []
    
    for _ in range(num_iterations):
        timer.start()
        result = func(*args, **kwargs)
        timer.stop()
        
        if return_results:
            results.append(result)
    
    stats = timer.get_statistics()
    
    logger.info(
        f"[{func.__name__}] "
        f"Mean: {stats['mean_ms']:.2f}ms, "
        f"Std: {stats['std_ms']:.2f}ms, "
        f"Min: {stats['min_ms']:.2f}ms, "
        f"Max: {stats['max_ms']:.2f}ms"
    )
    
    benchmark_result = {
        "function": func.__name__,
        "statistics": stats,
        "num_iterations": num_iterations,
    }
    
    if return_results:
        benchmark_result["results"] = results
    
    return benchmark_result


@contextmanager
def profile_context(
    name: str = "ProfiledOperation",
    enabled: bool = True,
    print_summary: bool = True,
):
    """
    Context manager for profiling code blocks.
    
    Args:
        name: Profile name
        enabled: Whether profiling is enabled
        print_summary: Whether to print summary
        
    Yields:
        Profiler instance
    """
    if not enabled:
        yield None
        return
    
    prof = Profiler(
        profile_memory=True,
        with_modules=False,
    )
    
    timer = Timer(name)
    
    timer.start()
    prof.start()
    
    try:
        yield prof
    finally:
        prof.stop()
        timer.stop()
        
        if print_summary:
            prof.print_summary(top_k=5)


def log_profiling_results(
    results: Union[Dict[str, Any], List[ProfileResult]],
    prefix: str = "",
):
    """
    Log profiling results.
    
    Args:
        results: Profiling results
        prefix: Log message prefix
    """
    if isinstance(results, dict):
        # Log dictionary results
        for key, value in results.items():
            if isinstance(value, dict):
                log_profiling_results(value, prefix=f"{prefix}{key}.")
            else:
                logger.info(f"{prefix}{key}: {value}")
    
    elif isinstance(results, list):
        # Log list of ProfileResult
        for i, result in enumerate(results[:5]):  # Top 5
            logger.info(
                f"{prefix}[{i}] {result.name}: "
                f"{result.duration_ms:.2f}ms ({result.num_calls} calls)"
            )


def create_profiling_report(
    model: torch.nn.Module,
    test_inputs: List[torch.Tensor],
    output_dir: Union[str, Path],
) -> Dict[str, Any]:
    """
    Create comprehensive profiling report.
    
    Args:
        model: Model to profile
        test_inputs: List of test input tensors
        output_dir: Output directory for reports
        
    Returns:
        Profiling report dictionary
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report = {
        "model_name": model.__class__.__name__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(next(model.parameters()).device),
    }
    
    # Profile each input
    for i, input_tensor in enumerate(test_inputs):
        logger.info(f"Profiling input {i+1}/{len(test_inputs)}")
        
        result = profile_model(
            model,
            input_shape=input_tensor.shape[1:],
            batch_size=input_tensor.shape[0],
            num_iterations=50,
            warmup_iterations=5,
        )
        
        report[f"input_{i}"] = result
    
    # Save report
    report_path = output_dir / "profiling_report.json"
    with open(report_path, "w") as f:
        json.dump(
            report,
            f,
            indent=2,
            default=lambda x: str(x) if isinstance(x, ProfileResult) else x
        )
    
    logger.info(f"Profiling report saved to {report_path}")
    
    return report


# Export public API
__all__ = [
    "Timer",
    "Profiler",
    "ProfileResult",
    "profile_function",
    "profile_model",
    "benchmark_function",
    "profile_context",
    "log_profiling_results",
    "create_profiling_report",
]
