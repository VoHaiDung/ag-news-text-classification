"""
Speed Benchmark for AG News Text Classification
================================================================================
This module implements comprehensive speed and latency benchmarking for model
inference and training, measuring throughput, latency, and scalability metrics.

The benchmark evaluates:
- Inference speed (single sample and batch)
- Training throughput
- Memory-speed tradeoffs
- Hardware utilization efficiency

References:
    - Coleman, C., et al. (2019). Analysis of DAWNBench
    - Reddi, V. J., et al. (2020). MLPerf Inference Benchmark

Author: Võ Hải Dũng
License: MIT
"""

import logging
import time
import gc
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
import psutil
from tqdm import tqdm

from src.core.registry import Registry
from src.core.factory import Factory
from src.utils.reproducibility import set_seed
from src.utils.profiling_utils import profile_memory, profile_time
from src.data.datasets.ag_news import AGNewsDataset
from src.data.loaders.dataloader import create_dataloader

logger = logging.getLogger(__name__)


@dataclass
class SpeedMetrics:
    """Container for speed measurement metrics."""
    
    samples_per_second: float
    latency_mean: float
    latency_std: float
    latency_p50: float
    latency_p95: float
    latency_p99: float
    total_time: float
    num_samples: int
    batch_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "samples_per_second": self.samples_per_second,
            "latency_mean_ms": self.latency_mean * 1000,
            "latency_std_ms": self.latency_std * 1000,
            "latency_p50_ms": self.latency_p50 * 1000,
            "latency_p95_ms": self.latency_p95 * 1000,
            "latency_p99_ms": self.latency_p99 * 1000,
            "total_time_s": self.total_time,
            "num_samples": self.num_samples,
            "batch_size": self.batch_size
        }


class SpeedBenchmark:
    """
    Comprehensive speed benchmarking for text classification models.
    
    This class measures:
    - Inference latency and throughput
    - Training speed
    - Batch processing efficiency
    - Hardware utilization
    """
    
    def __init__(
        self,
        models: List[str],
        batch_sizes: List[int] = [1, 8, 16, 32, 64, 128],
        sequence_lengths: List[int] = [128, 256, 512],
        num_warmup: int = 10,
        num_iterations: int = 100,
        device: str = "cuda",
        seed: int = 42
    ):
        """
        Initialize speed benchmark.
        
        Args:
            models: List of model names to benchmark
            batch_sizes: Batch sizes to test
            sequence_lengths: Sequence lengths to test
            num_warmup: Number of warmup iterations
            num_iterations: Number of benchmark iterations
            device: Device to run on (cuda/cpu)
            seed: Random seed
        """
        self.models = models
        self.batch_sizes = batch_sizes
        self.sequence_lengths = sequence_lengths
        self.num_warmup = num_warmup
        self.num_iterations = num_iterations
        self.device = device
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
        Run complete speed benchmark.
        
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Starting speed benchmark")
        logger.info(f"Models: {self.models}")
        logger.info(f"Batch sizes: {self.batch_sizes}")
        logger.info(f"Device: {self.device}")
        
        # Benchmark each model
        for model_name in self.models:
            logger.info(f"\nBenchmarking model: {model_name}")
            self.results[model_name] = self._benchmark_model(model_name)
        
        # Generate comparison report
        comparison = self._generate_comparison()
        
        # Calculate efficiency metrics
        efficiency = self._calculate_efficiency()
        
        return {
            "model_results": self.results,
            "comparison": comparison,
            "efficiency": efficiency,
            "hardware_info": self._get_hardware_info()
        }
    
    def _benchmark_model(self, model_name: str) -> Dict[str, Any]:
        """
        Benchmark a single model.
        
        Args:
            model_name: Name of model to benchmark
            
        Returns:
            Benchmark results for the model
        """
        # Load model
        model = self._load_model(model_name)
        
        # Prepare results storage
        inference_results = {}
        training_results = {}
        
        # Benchmark inference
        for batch_size in self.batch_sizes:
            for seq_len in self.sequence_lengths:
                key = f"batch_{batch_size}_seq_{seq_len}"
                
                logger.info(f"Testing batch_size={batch_size}, seq_len={seq_len}")
                
                # Run inference benchmark
                inference_metrics = self._benchmark_inference(
                    model, batch_size, seq_len
                )
                inference_results[key] = inference_metrics.to_dict()
                
                # Clear cache
                self._clear_cache()
        
        # Benchmark training (if applicable)
        if self._supports_training(model):
            for batch_size in self.batch_sizes[:3]:  # Limit training benchmark
                key = f"batch_{batch_size}"
                
                logger.info(f"Testing training with batch_size={batch_size}")
                
                training_metrics = self._benchmark_training(
                    model, batch_size
                )
                training_results[key] = training_metrics
        
        return {
            "inference": inference_results,
            "training": training_results,
            "model_info": self._get_model_info(model)
        }
    
    def _benchmark_inference(
        self,
        model: Any,
        batch_size: int,
        sequence_length: int
    ) -> SpeedMetrics:
        """
        Benchmark model inference speed.
        
        Args:
            model: Model to benchmark
            batch_size: Batch size
            sequence_length: Sequence length
            
        Returns:
            Speed metrics
        """
        # Create dummy input
        dummy_input = self._create_dummy_input(batch_size, sequence_length)
        
        # Move model to device
        model = model.to(self.device)
        model.eval()
        
        # Warmup
        logger.info(f"Running {self.num_warmup} warmup iterations")
        with torch.no_grad():
            for _ in range(self.num_warmup):
                _ = model(dummy_input)
                if self.device == "cuda":
                    torch.cuda.synchronize()
        
        # Benchmark
        latencies = []
        logger.info(f"Running {self.num_iterations} benchmark iterations")
        
        with torch.no_grad():
            for _ in tqdm(range(self.num_iterations), desc="Benchmarking"):
                # Measure single iteration
                start_time = time.perf_counter()
                
                _ = model(dummy_input)
                
                if self.device == "cuda":
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                latency = end_time - start_time
                latencies.append(latency)
        
        # Calculate metrics
        latencies = np.array(latencies)
        total_time = np.sum(latencies)
        num_samples = batch_size * self.num_iterations
        
        metrics = SpeedMetrics(
            samples_per_second=num_samples / total_time,
            latency_mean=np.mean(latencies),
            latency_std=np.std(latencies),
            latency_p50=np.percentile(latencies, 50),
            latency_p95=np.percentile(latencies, 95),
            latency_p99=np.percentile(latencies, 99),
            total_time=total_time,
            num_samples=num_samples,
            batch_size=batch_size
        )
        
        return metrics
    
    def _benchmark_training(
        self,
        model: Any,
        batch_size: int
    ) -> Dict[str, Any]:
        """
        Benchmark model training speed.
        
        Args:
            model: Model to benchmark
            batch_size: Batch size
            
        Returns:
            Training speed metrics
        """
        # Create dummy data
        dummy_input = self._create_dummy_input(batch_size, 256)
        dummy_labels = torch.randint(0, 4, (batch_size,)).to(self.device)
        
        # Setup training
        model = model.to(self.device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Warmup
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(dummy_input)
            loss = criterion(outputs, dummy_labels)
            loss.backward()
            optimizer.step()
            if self.device == "cuda":
                torch.cuda.synchronize()
        
        # Benchmark
        forward_times = []
        backward_times = []
        step_times = []
        
        for _ in tqdm(range(self.num_iterations), desc="Training benchmark"):
            # Forward pass
            optimizer.zero_grad()
            
            start = time.perf_counter()
            outputs = model(dummy_input)
            loss = criterion(outputs, dummy_labels)
            if self.device == "cuda":
                torch.cuda.synchronize()
            forward_time = time.perf_counter() - start
            
            # Backward pass
            start = time.perf_counter()
            loss.backward()
            if self.device == "cuda":
                torch.cuda.synchronize()
            backward_time = time.perf_counter() - start
            
            # Optimizer step
            start = time.perf_counter()
            optimizer.step()
            if self.device == "cuda":
                torch.cuda.synchronize()
            step_time = time.perf_counter() - start
            
            forward_times.append(forward_time)
            backward_times.append(backward_time)
            step_times.append(step_time)
        
        return {
            "forward_time_ms": np.mean(forward_times) * 1000,
            "backward_time_ms": np.mean(backward_times) * 1000,
            "step_time_ms": np.mean(step_times) * 1000,
            "total_time_ms": (np.mean(forward_times) + 
                            np.mean(backward_times) + 
                            np.mean(step_times)) * 1000,
            "samples_per_second": batch_size / (np.mean(forward_times) + 
                                               np.mean(backward_times) + 
                                               np.mean(step_times))
        }
    
    def _create_dummy_input(
        self,
        batch_size: int,
        sequence_length: int
    ) -> torch.Tensor:
        """
        Create dummy input tensor.
        
        Args:
            batch_size: Batch size
            sequence_length: Sequence length
            
        Returns:
            Dummy input tensor
        """
        # Create random token IDs
        input_ids = torch.randint(
            0, 30000,  # Typical vocab size
            (batch_size, sequence_length),
            dtype=torch.long
        ).to(self.device)
        
        return input_ids
    
    def _calculate_efficiency(self) -> Dict[str, Any]:
        """
        Calculate efficiency metrics across models.
        
        Returns:
            Efficiency comparison metrics
        """
        efficiency_metrics = {}
        
        for model_name in self.models:
            if model_name not in self.results:
                continue
            
            inference_results = self.results[model_name]["inference"]
            
            # Calculate average metrics
            throughputs = []
            latencies = []
            
            for key, metrics in inference_results.items():
                throughputs.append(metrics["samples_per_second"])
                latencies.append(metrics["latency_mean_ms"])
            
            efficiency_metrics[model_name] = {
                "avg_throughput": np.mean(throughputs),
                "avg_latency_ms": np.mean(latencies),
                "throughput_std": np.std(throughputs),
                "latency_std_ms": np.std(latencies),
                "scaling_efficiency": self._calculate_scaling_efficiency(
                    inference_results
                )
            }
        
        return efficiency_metrics
    
    def _calculate_scaling_efficiency(
        self,
        inference_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate batch scaling efficiency.
        
        Args:
            inference_results: Inference benchmark results
            
        Returns:
            Scaling efficiency metrics
        """
        # Group by sequence length
        scaling_by_seq = {}
        
        for seq_len in self.sequence_lengths:
            throughputs = []
            
            for batch_size in self.batch_sizes:
                key = f"batch_{batch_size}_seq_{seq_len}"
                if key in inference_results:
                    throughputs.append(
                        inference_results[key]["samples_per_second"]
                    )
            
            if len(throughputs) > 1:
                # Calculate scaling efficiency
                # Perfect scaling: throughput increases linearly with batch size
                base_throughput = throughputs[0]
                scaling_factors = [t / base_throughput for t in throughputs]
                ideal_scaling = [self.batch_sizes[i] / self.batch_sizes[0] 
                               for i in range(len(throughputs))]
                
                efficiency = np.mean([
                    min(actual / ideal, 1.0)
                    for actual, ideal in zip(scaling_factors, ideal_scaling)
                ])
                
                scaling_by_seq[f"seq_{seq_len}"] = efficiency
        
        return scaling_by_seq
    
    def _generate_comparison(self) -> Dict[str, Any]:
        """
        Generate model comparison report.
        
        Returns:
            Comparison metrics
        """
        comparison = {
            "fastest_inference": None,
            "highest_throughput": None,
            "most_efficient": None,
            "rankings": {}
        }
        
        # Collect metrics for comparison
        model_metrics = {}
        
        for model_name in self.models:
            if model_name not in self.results:
                continue
            
            inference_results = self.results[model_name]["inference"]
            
            # Get metrics for standard config (batch_size=32, seq_len=256)
            standard_key = "batch_32_seq_256"
            if standard_key in inference_results:
                model_metrics[model_name] = inference_results[standard_key]
        
        if not model_metrics:
            return comparison
        
        # Find best performers
        comparison["fastest_inference"] = min(
            model_metrics.items(),
            key=lambda x: x[1]["latency_mean_ms"]
        )[0]
        
        comparison["highest_throughput"] = max(
            model_metrics.items(),
            key=lambda x: x[1]["samples_per_second"]
        )[0]
        
        # Create rankings
        comparison["rankings"]["by_latency"] = sorted(
            model_metrics.keys(),
            key=lambda x: model_metrics[x]["latency_mean_ms"]
        )
        
        comparison["rankings"]["by_throughput"] = sorted(
            model_metrics.keys(),
            key=lambda x: model_metrics[x]["samples_per_second"],
            reverse=True
        )
        
        return comparison
    
    def _get_hardware_info(self) -> Dict[str, Any]:
        """
        Get hardware information.
        
        Returns:
            Hardware specifications
        """
        info = {
            "cpu": {
                "count": psutil.cpu_count(logical=False),
                "count_logical": psutil.cpu_count(logical=True),
                "freq_mhz": psutil.cpu_freq().current if psutil.cpu_freq() else None
            },
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3)
            }
        }
        
        if self.device == "cuda":
            info["gpu"] = {
                "name": torch.cuda.get_device_name(0),
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                "capability": f"{torch.cuda.get_device_capability()[0]}.{torch.cuda.get_device_capability()[1]}"
            }
        
        return info
    
    def _get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get model information.
        
        Args:
            model: Model instance
            
        Returns:
            Model specifications
        """
        info = {
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
        }
        
        # Calculate model size
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        info["size_mb"] = (param_size + buffer_size) / (1024**2)
        
        return info
    
    def _load_model(self, model_name: str) -> Any:
        """Load model for benchmarking."""
        return self.factory.create_model(model_name)
    
    def _supports_training(self, model: Any) -> bool:
        """Check if model supports training."""
        return hasattr(model, "train") and hasattr(model, "parameters")
    
    def _clear_cache(self):
        """Clear GPU/CPU cache."""
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
    
    def visualize_results(self, save_path: Optional[str] = None):
        """
        Visualize benchmark results.
        
        Args:
            save_path: Path to save visualization
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Throughput comparison
        self._plot_throughput_comparison(axes[0, 0])
        
        # Plot 2: Latency comparison
        self._plot_latency_comparison(axes[0, 1])
        
        # Plot 3: Batch scaling
        self._plot_batch_scaling(axes[1, 0])
        
        # Plot 4: Sequence length impact
        self._plot_sequence_impact(axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def _plot_throughput_comparison(self, ax):
        """Plot throughput comparison across models."""
        model_names = []
        throughputs = []
        
        for model_name in self.models:
            if model_name in self.results:
                # Get throughput for standard config
                standard_key = "batch_32_seq_256"
                inference_results = self.results[model_name]["inference"]
                
                if standard_key in inference_results:
                    model_names.append(model_name)
                    throughputs.append(
                        inference_results[standard_key]["samples_per_second"]
                    )
        
        ax.bar(model_names, throughputs)
        ax.set_xlabel("Model")
        ax.set_ylabel("Samples/Second")
        ax.set_title("Throughput Comparison (Batch=32, Seq=256)")
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_latency_comparison(self, ax):
        """Plot latency comparison with percentiles."""
        data = []
        labels = []
        
        for model_name in self.models[:5]:  # Limit to 5 models
            if model_name in self.results:
                standard_key = "batch_1_seq_256"
                inference_results = self.results[model_name]["inference"]
                
                if standard_key in inference_results:
                    metrics = inference_results[standard_key]
                    data.append([
                        metrics["latency_p50_ms"],
                        metrics["latency_p95_ms"],
                        metrics["latency_p99_ms"]
                    ])
                    labels.append(model_name)
        
        if data:
            x = np.arange(len(labels))
            width = 0.25
            
            p50 = [d[0] for d in data]
            p95 = [d[1] for d in data]
            p99 = [d[2] for d in data]
            
            ax.bar(x - width, p50, width, label="P50")
            ax.bar(x, p95, width, label="P95")
            ax.bar(x + width, p99, width, label="P99")
            
            ax.set_xlabel("Model")
            ax.set_ylabel("Latency (ms)")
            ax.set_title("Latency Percentiles (Single Sample)")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.legend()
    
    def _plot_batch_scaling(self, ax):
        """Plot batch size scaling efficiency."""
        for model_name in self.models[:3]:  # Limit to 3 models
            if model_name not in self.results:
                continue
            
            throughputs = []
            batch_sizes_used = []
            
            for batch_size in self.batch_sizes:
                key = f"batch_{batch_size}_seq_256"
                inference_results = self.results[model_name]["inference"]
                
                if key in inference_results:
                    throughputs.append(
                        inference_results[key]["samples_per_second"]
                    )
                    batch_sizes_used.append(batch_size)
            
            if throughputs:
                ax.plot(batch_sizes_used, throughputs, marker='o', label=model_name)
        
        ax.set_xlabel("Batch Size")
        ax.set_ylabel("Throughput (samples/s)")
        ax.set_title("Batch Size Scaling")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_sequence_impact(self, ax):
        """Plot sequence length impact on performance."""
        for model_name in self.models[:3]:  # Limit to 3 models
            if model_name not in self.results:
                continue
            
            latencies = []
            seq_lengths_used = []
            
            for seq_len in self.sequence_lengths:
                key = f"batch_1_seq_{seq_len}"
                inference_results = self.results[model_name]["inference"]
                
                if key in inference_results:
                    latencies.append(
                        inference_results[key]["latency_mean_ms"]
                    )
                    seq_lengths_used.append(seq_len)
            
            if latencies:
                ax.plot(seq_lengths_used, latencies, marker='o', label=model_name)
        
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Sequence Length Impact")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def save_results(self, filepath: str):
        """
        Save benchmark results to file.
        
        Args:
            filepath: Path to save results
        """
        with open(filepath, "w") as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
