"""
Memory Benchmark for AG News Text Classification
================================================================================
This module implements comprehensive memory usage benchmarking for models,
measuring RAM consumption, GPU memory usage, and memory efficiency.

The benchmark evaluates:
- Peak memory usage during inference and training
- Memory scaling with batch size
- Memory-accuracy tradeoffs
- Memory optimization effectiveness

References:
    - Strubell, E., et al. (2019). Energy and Policy Considerations for Deep Learning
    - Schwartz, R., et al. (2020). Green AI

Author: Võ Hải Dũng
License: MIT
"""

import logging
import gc
import tracemalloc
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
import psutil
import os

import numpy as np
import torch
from tqdm import tqdm

from src.core.registry import Registry
from src.core.factory import Factory
from src.utils.memory_utils import get_memory_usage, track_memory
from src.utils.reproducibility import set_seed

logger = logging.getLogger(__name__)


@dataclass
class MemoryMetrics:
    """Container for memory measurement metrics."""
    
    peak_memory_mb: float
    average_memory_mb: float
    baseline_memory_mb: float
    incremental_memory_mb: float
    gpu_memory_mb: Optional[float]
    gpu_memory_cached_mb: Optional[float]
    model_size_mb: float
    batch_size: int
    sequence_length: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "average_memory_mb": self.average_memory_mb,
            "baseline_memory_mb": self.baseline_memory_mb,
            "incremental_memory_mb": self.incremental_memory_mb,
            "gpu_memory_mb": self.gpu_memory_mb,
            "gpu_memory_cached_mb": self.gpu_memory_cached_mb,
            "model_size_mb": self.model_size_mb,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "memory_efficiency": self.incremental_memory_mb / self.model_size_mb
        }


class MemoryBenchmark:
    """
    Comprehensive memory benchmarking for text classification models.
    
    This class measures:
    - CPU memory usage
    - GPU memory usage
    - Memory scaling behavior
    - Memory optimization effectiveness
    """
    
    def __init__(
        self,
        models: List[str],
        batch_sizes: List[int] = [1, 8, 16, 32, 64],
        sequence_lengths: List[int] = [128, 256, 512],
        device: str = "cuda",
        num_iterations: int = 10,
        track_gradients: bool = True,
        seed: int = 42
    ):
        """
        Initialize memory benchmark.
        
        Args:
            models: List of model names to benchmark
            batch_sizes: Batch sizes to test
            sequence_lengths: Sequence lengths to test
            device: Device to run on
            num_iterations: Number of iterations for averaging
            track_gradients: Whether to track gradient memory
            seed: Random seed
        """
        self.models = models
        self.batch_sizes = batch_sizes
        self.sequence_lengths = sequence_lengths
        self.device = device
        self.num_iterations = num_iterations
        self.track_gradients = track_gradients
        self.seed = seed
        
        self.registry = Registry()
        self.factory = Factory()
        self.results = {}
        
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run complete memory benchmark.
        
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Starting memory benchmark")
        logger.info(f"Models: {self.models}")
        logger.info(f"Device: {self.device}")
        
        # Get baseline memory
        baseline_memory = self._get_baseline_memory()
        logger.info(f"Baseline memory: {baseline_memory:.2f} MB")
        
        # Benchmark each model
        for model_name in self.models:
            logger.info(f"\nBenchmarking model: {model_name}")
            self.results[model_name] = self._benchmark_model(
                model_name,
                baseline_memory
            )
        
        # Generate comparison report
        comparison = self._generate_comparison()
        
        # Calculate memory efficiency
        efficiency = self._calculate_efficiency()
        
        return {
            "model_results": self.results,
            "comparison": comparison,
            "efficiency": efficiency,
            "system_info": self._get_system_info()
        }
    
    def _benchmark_model(
        self,
        model_name: str,
        baseline_memory: float
    ) -> Dict[str, Any]:
        """
        Benchmark memory usage for a single model.
        
        Args:
            model_name: Name of model to benchmark
            baseline_memory: Baseline memory usage
            
        Returns:
            Memory benchmark results
        """
        # Load model
        model = self._load_model(model_name)
        model_size = self._get_model_size(model)
        
        logger.info(f"Model size: {model_size:.2f} MB")
        
        # Prepare results storage
        inference_results = {}
        training_results = {}
        
        # Benchmark inference memory
        for batch_size in self.batch_sizes:
            for seq_len in self.sequence_lengths:
                key = f"batch_{batch_size}_seq_{seq_len}"
                
                logger.info(f"Testing inference: batch_size={batch_size}, seq_len={seq_len}")
                
                try:
                    metrics = self._benchmark_inference_memory(
                        model,
                        batch_size,
                        seq_len,
                        baseline_memory,
                        model_size
                    )
                    inference_results[key] = metrics.to_dict()
                except Exception as e:
                    logger.warning(f"Failed to benchmark {key}: {e}")
                    inference_results[key] = {"error": str(e)}
                
                # Clear memory
                self._clear_memory()
        
        # Benchmark training memory if requested
        if self.track_gradients:
            for batch_size in self.batch_sizes[:3]:  # Limit for training
                key = f"batch_{batch_size}"
                
                logger.info(f"Testing training: batch_size={batch_size}")
                
                try:
                    metrics = self._benchmark_training_memory(
                        model,
                        batch_size,
                        baseline_memory,
                        model_size
                    )
                    training_results[key] = metrics
                except Exception as e:
                    logger.warning(f"Failed to benchmark training {key}: {e}")
                    training_results[key] = {"error": str(e)}
                
                # Clear memory
                self._clear_memory()
        
        return {
            "inference": inference_results,
            "training": training_results,
            "model_info": {
                "size_mb": model_size,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                )
            }
        }
    
    def _benchmark_inference_memory(
        self,
        model: Any,
        batch_size: int,
        sequence_length: int,
        baseline_memory: float,
        model_size: float
    ) -> MemoryMetrics:
        """
        Benchmark inference memory usage.
        
        Args:
            model: Model to benchmark
            batch_size: Batch size
            sequence_length: Sequence length
            baseline_memory: Baseline memory
            model_size: Model size in MB
            
        Returns:
            Memory metrics
        """
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        # Start memory tracking
        if self.device == "cpu":
            tracemalloc.start()
        
        memory_readings = []
        gpu_memory_readings = []
        
        with torch.no_grad():
            for _ in range(self.num_iterations):
                # Clear memory
                self._clear_memory()
                
                # Create input
                input_ids = torch.randint(
                    0, 30000,
                    (batch_size, sequence_length),
                    dtype=torch.long
                ).to(self.device)
                
                # Get memory before inference
                if self.device == "cpu":
                    mem_before = self._get_cpu_memory()
                else:
                    mem_before = self._get_gpu_memory()
                
                # Run inference
                _ = model(input_ids)
                
                # Get memory after inference
                if self.device == "cpu":
                    mem_after = self._get_cpu_memory()
                    memory_readings.append(mem_after - mem_before)
                else:
                    mem_after = self._get_gpu_memory()
                    gpu_memory_readings.append(mem_after - mem_before)
        
        # Stop memory tracking
        if self.device == "cpu":
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_memory = peak / (1024 * 1024)  # Convert to MB
        else:
            peak_memory = max(gpu_memory_readings) if gpu_memory_readings else 0
        
        # Calculate metrics
        if self.device == "cpu":
            avg_memory = np.mean(memory_readings) if memory_readings else 0
            gpu_memory = None
            gpu_cached = None
        else:
            avg_memory = np.mean(gpu_memory_readings) if gpu_memory_readings else 0
            gpu_memory = avg_memory
            gpu_cached = torch.cuda.memory_reserved() / (1024 * 1024) if self.device == "cuda" else None
        
        metrics = MemoryMetrics(
            peak_memory_mb=peak_memory,
            average_memory_mb=avg_memory,
            baseline_memory_mb=baseline_memory,
            incremental_memory_mb=avg_memory - baseline_memory,
            gpu_memory_mb=gpu_memory,
            gpu_memory_cached_mb=gpu_cached,
            model_size_mb=model_size,
            batch_size=batch_size,
            sequence_length=sequence_length
        )
        
        return metrics
    
    def _benchmark_training_memory(
        self,
        model: Any,
        batch_size: int,
        baseline_memory: float,
        model_size: float
    ) -> Dict[str, Any]:
        """
        Benchmark training memory usage.
        
        Args:
            model: Model to benchmark
            batch_size: Batch size
            baseline_memory: Baseline memory
            model_size: Model size
            
        Returns:
            Training memory metrics
        """
        # Move model to device
        model = model.to(self.device)
        model.train()
        
        # Setup training components
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        
        sequence_length = 256  # Fixed for training
        
        # Track memory
        memory_stages = {
            "forward": [],
            "backward": [],
            "optimizer": [],
            "total": []
        }
        
        for _ in range(self.num_iterations):
            # Clear memory and gradients
            self._clear_memory()
            optimizer.zero_grad()
            
            # Create input
            input_ids = torch.randint(
                0, 30000,
                (batch_size, sequence_length),
                dtype=torch.long
            ).to(self.device)
            labels = torch.randint(0, 4, (batch_size,)).to(self.device)
            
            # Measure forward pass
            mem_start = self._get_current_memory()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            mem_forward = self._get_current_memory() - mem_start
            
            # Measure backward pass
            mem_before_backward = self._get_current_memory()
            loss.backward()
            mem_backward = self._get_current_memory() - mem_before_backward
            
            # Measure optimizer step
            mem_before_step = self._get_current_memory()
            optimizer.step()
            mem_optimizer = self._get_current_memory() - mem_before_step
            
            # Total memory
            mem_total = self._get_current_memory() - mem_start
            
            memory_stages["forward"].append(mem_forward)
            memory_stages["backward"].append(mem_backward)
            memory_stages["optimizer"].append(mem_optimizer)
            memory_stages["total"].append(mem_total)
        
        return {
            "forward_mb": np.mean(memory_stages["forward"]),
            "backward_mb": np.mean(memory_stages["backward"]),
            "optimizer_mb": np.mean(memory_stages["optimizer"]),
            "total_mb": np.mean(memory_stages["total"]),
            "gradient_memory_mb": np.mean(memory_stages["backward"]),
            "activation_memory_mb": np.mean(memory_stages["forward"]),
            "peak_memory_mb": np.max(memory_stages["total"])
        }
    
    def _calculate_efficiency(self) -> Dict[str, Any]:
        """
        Calculate memory efficiency metrics.
        
        Returns:
            Efficiency metrics
        """
        efficiency_metrics = {}
        
        for model_name in self.models:
            if model_name not in self.results:
                continue
            
            model_results = self.results[model_name]
            inference_results = model_results["inference"]
            model_size = model_results["model_info"]["size_mb"]
            
            # Calculate memory scaling efficiency
            memory_per_sample = []
            
            for key, metrics in inference_results.items():
                if "error" not in metrics:
                    batch_size = metrics["batch_size"]
                    incremental_mem = metrics["incremental_memory_mb"]
                    memory_per_sample.append(incremental_mem / batch_size)
            
            efficiency_metrics[model_name] = {
                "model_size_mb": model_size,
                "avg_memory_per_sample_mb": np.mean(memory_per_sample) if memory_per_sample else 0,
                "memory_overhead_ratio": np.mean(memory_per_sample) / model_size if memory_per_sample and model_size > 0 else 0,
                "batch_scaling_efficiency": self._calculate_batch_scaling(inference_results)
            }
        
        return efficiency_metrics
    
    def _calculate_batch_scaling(
        self,
        inference_results: Dict[str, Any]
    ) -> float:
        """
        Calculate batch scaling efficiency.
        
        Args:
            inference_results: Inference memory results
            
        Returns:
            Batch scaling efficiency score
        """
        # Group by sequence length
        seq_len = 256  # Use fixed sequence length
        memories = []
        batch_sizes_used = []
        
        for batch_size in self.batch_sizes:
            key = f"batch_{batch_size}_seq_{seq_len}"
            if key in inference_results and "error" not in inference_results[key]:
                memories.append(inference_results[key]["incremental_memory_mb"])
                batch_sizes_used.append(batch_size)
        
        if len(memories) < 2:
            return 0.0
        
        # Calculate scaling efficiency
        # Ideal: memory increases linearly with batch size
        # Calculate correlation between batch size and memory
        correlation = np.corrcoef(batch_sizes_used, memories)[0, 1]
        
        # Calculate linearity score
        x = np.array(batch_sizes_used).reshape(-1, 1)
        y = np.array(memories)
        
        # Fit linear regression
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(x, y)
        r2_score = reg.score(x, y)
        
        return r2_score
    
    def _generate_comparison(self) -> Dict[str, Any]:
        """
        Generate model comparison report.
        
        Returns:
            Comparison metrics
        """
        comparison = {
            "most_efficient": None,
            "least_memory": None,
            "best_scaling": None,
            "rankings": {}
        }
        
        if not self.results:
            return comparison
        
        # Collect metrics for comparison
        model_metrics = {}
        
        for model_name in self.models:
            if model_name in self.results:
                # Get standard configuration metrics
                standard_key = "batch_32_seq_256"
                inference_results = self.results[model_name]["inference"]
                
                if standard_key in inference_results and "error" not in inference_results[standard_key]:
                    model_metrics[model_name] = inference_results[standard_key]
        
        if not model_metrics:
            return comparison
        
        # Find best performers
        comparison["least_memory"] = min(
            model_metrics.items(),
            key=lambda x: x[1]["incremental_memory_mb"]
        )[0]
        
        comparison["most_efficient"] = min(
            model_metrics.items(),
            key=lambda x: x[1]["memory_efficiency"]
        )[0]
        
        # Create rankings
        comparison["rankings"]["by_memory"] = sorted(
            model_metrics.keys(),
            key=lambda x: model_metrics[x]["incremental_memory_mb"]
        )
        
        comparison["rankings"]["by_efficiency"] = sorted(
            model_metrics.keys(),
            key=lambda x: model_metrics[x]["memory_efficiency"]
        )
        
        return comparison
    
    def _get_baseline_memory(self) -> float:
        """Get baseline memory usage."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    def _get_current_memory(self) -> float:
        """Get current memory usage."""
        if self.device == "cuda":
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 * 1024)
        else:
            return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    def _get_cpu_memory(self) -> float:
        """Get CPU memory usage in MB."""
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    
    def _get_gpu_memory(self) -> float:
        """Get GPU memory usage in MB."""
        if self.device == "cuda":
            torch.cuda.synchronize()
            return torch.cuda.memory_allocated() / (1024 * 1024)
        return 0.0
    
    def _get_model_size(self, model: Any) -> float:
        """
        Get model size in MB.
        
        Args:
            model: Model instance
            
        Returns:
            Model size in MB
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        return (param_size + buffer_size) / (1024 * 1024)
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            "cpu_memory_gb": psutil.virtual_memory().total / (1024**3),
            "available_memory_gb": psutil.virtual_memory().available / (1024**3)
        }
        
        if self.device == "cuda":
            info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            info["gpu_name"] = torch.cuda.get_device_name(0)
        
        return info
    
    def _load_model(self, model_name: str) -> Any:
        """Load model for benchmarking."""
        return self.factory.create_model(model_name)
    
    def _clear_memory(self):
        """Clear memory caches."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def save_results(self, filepath: str):
        """
        Save benchmark results to file.
        
        Args:
            filepath: Path to save results
        """
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
