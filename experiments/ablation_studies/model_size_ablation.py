"""
Model Size Ablation Study for AG News Text Classification
================================================================================
This module performs ablation studies on model architecture dimensions including
hidden sizes, number of layers, attention heads, and parameter efficiency.

Model size ablation helps identify optimal model capacity for the task and
understand the trade-offs between model size, performance, and efficiency.

References:
    - Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models
    - Tay, Y., et al. (2022). Scale Efficiently: Insights from Pre-training

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import json
from datetime import datetime
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.utils.memory_utils import get_memory_usage, clear_memory
from src.utils.profiling_utils import profile_model
from src.data.datasets.ag_news import AGNewsDataset
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class ModelSizeAblation:
    """
    Performs model size ablation studies for text classification.
    
    Analyzes the impact of:
    - Model depth (number of layers)
    - Model width (hidden dimensions)
    - Attention heads
    - Feed-forward dimensions
    - Parameter efficiency techniques
    """
    
    def __init__(
        self,
        base_model: str = "bert",
        size_configurations: Optional[List[Dict[str, Any]]] = None,
        dataset_name: str = "ag_news",
        num_trials: int = 3,
        device: str = "cuda",
        output_dir: str = "./ablation_results/model_size",
        seed: int = 42,
        profile_models: bool = True
    ):
        """
        Initialize model size ablation study.
        
        Args:
            base_model: Base model architecture
            size_configurations: List of model size configurations to test
            dataset_name: Dataset name
            num_trials: Number of trials per configuration
            device: Device to use
            output_dir: Output directory
            seed: Random seed
            profile_models: Whether to profile model efficiency
        """
        self.base_model = base_model
        self.size_configurations = size_configurations or self._get_default_configurations()
        self.dataset_name = dataset_name
        self.num_trials = num_trials
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.profile_models = profile_models
        
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        self.results = {
            "configurations": {},
            "scaling_analysis": {},
            "efficiency_metrics": {},
            "optimal_configurations": {},
            "summary": {}
        }
        
        set_seed(seed)
        logger.info(f"Initialized Model Size Ablation for {base_model}")
    
    def _get_default_configurations(self) -> List[Dict[str, Any]]:
        """Get default model size configurations."""
        return [
            # Tiny models
            {
                "name": "tiny",
                "hidden_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "intermediate_size": 512,
                "max_position_embeddings": 128
            },
            # Mini models
            {
                "name": "mini",
                "hidden_size": 256,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "intermediate_size": 1024,
                "max_position_embeddings": 256
            },
            # Small models
            {
                "name": "small",
                "hidden_size": 384,
                "num_hidden_layers": 6,
                "num_attention_heads": 6,
                "intermediate_size": 1536,
                "max_position_embeddings": 256
            },
            # Base models
            {
                "name": "base",
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "max_position_embeddings": 512
            },
            # Large models
            {
                "name": "large",
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "max_position_embeddings": 512
            }
        ]
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """
        Run complete model size ablation study.
        
        Returns:
            Ablation study results
        """
        logger.info("Starting model size ablation study")
        
        # Load dataset
        dataset = self._load_dataset()
        
        # Test each configuration
        for config in self.size_configurations:
            logger.info(f"\nTesting configuration: {config['name']}")
            
            config_results = self._test_configuration(config, dataset)
            self.results["configurations"][config["name"]] = config_results
            
            logger.info(
                f"Config: {config['name']} | "
                f"Params: {config_results['num_parameters']/1e6:.1f}M | "
                f"Accuracy: {config_results['mean_accuracy']:.4f} | "
                f"Speed: {config_results['inference_speed']:.1f} samples/sec"
            )
        
        # Analyze scaling laws
        self.results["scaling_analysis"] = self._analyze_scaling_laws()
        
        # Find optimal configurations
        self.results["optimal_configurations"] = self._find_optimal_configurations()
        
        # Generate summary
        self.results["summary"] = self._generate_summary()
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return self.results
    
    def _test_configuration(
        self,
        config: Dict[str, Any],
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test a specific model configuration.
        
        Args:
            config: Model configuration
            dataset: Dataset dictionary
            
        Returns:
            Configuration test results
        """
        results = {
            "config": config,
            "trials": [],
            "mean_accuracy": 0,
            "std_accuracy": 0,
            "mean_f1": 0,
            "num_parameters": 0,
            "memory_usage": 0,
            "training_time": 0,
            "inference_speed": 0,
            "flops": 0
        }
        
        accuracies = []
        f1_scores = []
        training_times = []
        
        for trial in range(self.num_trials):
            logger.info(f"Trial {trial + 1}/{self.num_trials}")
            
            # Set seed
            set_seed(self.seed + trial)
            
            # Create model with configuration
            model = self._create_model_with_config(config)
            
            # Count parameters
            if trial == 0:
                results["num_parameters"] = self._count_parameters(model)
                
                # Profile model if requested
                if self.profile_models:
                    profile_results = self._profile_model(model, dataset)
                    results.update(profile_results)
            
            # Train model
            trainer = BaseTrainer(
                model=model,
                config=self._get_training_config(config),
                device=self.device
            )
            
            # Track training time
            start_time = time.time()
            
            trainer.train(
                dataset["train"]["texts"][:5000],  # Use subset for efficiency
                dataset["train"]["labels"][:5000],
                dataset["val"]["texts"][:1000],
                dataset["val"]["labels"][:1000]
            )
            
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            # Evaluate
            test_metrics = trainer.evaluate(
                dataset["test"]["texts"][:1000],
                dataset["test"]["labels"][:1000]
            )
            
            accuracies.append(test_metrics["accuracy"])
            f1_scores.append(test_metrics["f1_weighted"])
            
            results["trials"].append({
                "accuracy": test_metrics["accuracy"],
                "f1": test_metrics["f1_weighted"],
                "training_time": training_time
            })
            
            # Clear memory after each trial
            clear_memory()
        
        results["mean_accuracy"] = np.mean(accuracies)
        results["std_accuracy"] = np.std(accuracies)
        results["mean_f1"] = np.mean(f1_scores)
        results["training_time"] = np.mean(training_times)
        
        return results
    
    def _create_model_with_config(self, config: Dict[str, Any]):
        """Create model with specific size configuration."""
        model_config = {
            "num_labels": 4,  # AG News has 4 classes
            **config
        }
        
        # Create model based on base architecture
        if self.base_model == "bert":
            from transformers import BertConfig, BertForSequenceClassification
            bert_config = BertConfig(**model_config)
            model = BertForSequenceClassification(bert_config)
        else:
            # Fallback to factory
            model = self.factory.create_model(
                self.base_model,
                **model_config
            )
        
        return model
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total and trainable parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return total_params
    
    def _profile_model(
        self,
        model: nn.Module,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Profile model efficiency metrics.
        
        Args:
            model: Model to profile
            dataset: Dataset for profiling
            
        Returns:
            Profiling results
        """
        profile_results = {
            "memory_usage": 0,
            "inference_speed": 0,
            "flops": 0,
            "latency_p50": 0,
            "latency_p95": 0,
            "latency_p99": 0
        }
        
        model.eval()
        model = model.to(self.device)
        
        # Create sample batch
        batch_size = 32
        sample_texts = dataset["test"]["texts"][:batch_size]
        
        # Measure memory usage
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        initial_memory = get_memory_usage(self.device)
        
        # Tokenize (simplified)
        max_length = 128
        input_ids = torch.randint(0, 1000, (batch_size, max_length)).to(self.device)
        attention_mask = torch.ones((batch_size, max_length)).to(self.device)
        
        # Warm up
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids, attention_mask=attention_mask)
        
        # Measure inference speed
        latencies = []
        num_iterations = 100
        
        for _ in range(num_iterations):
            start = time.time()
            with torch.no_grad():
                _ = model(input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            latencies.append(time.time() - start)
        
        # Calculate metrics
        profile_results["memory_usage"] = get_memory_usage(self.device) - initial_memory
        profile_results["inference_speed"] = batch_size / np.mean(latencies)
        profile_results["latency_p50"] = np.percentile(latencies, 50) * 1000  # ms
        profile_results["latency_p95"] = np.percentile(latencies, 95) * 1000
        profile_results["latency_p99"] = np.percentile(latencies, 99) * 1000
        
        # Estimate FLOPs (simplified)
        hidden_size = model.config.hidden_size if hasattr(model, 'config') else 768
        num_layers = model.config.num_hidden_layers if hasattr(model, 'config') else 12
        profile_results["flops"] = self._estimate_flops(
            batch_size, max_length, hidden_size, num_layers
        )
        
        return profile_results
    
    def _estimate_flops(
        self,
        batch_size: int,
        seq_length: int,
        hidden_size: int,
        num_layers: int
    ) -> float:
        """
        Estimate FLOPs for transformer model.
        
        Simplified estimation based on:
        - Self-attention: 4 * batch * seq^2 * hidden
        - FFN: 8 * batch * seq * hidden^2
        """
        attention_flops = 4 * batch_size * seq_length * seq_length * hidden_size
        ffn_flops = 8 * batch_size * seq_length * hidden_size * hidden_size
        total_flops = num_layers * (attention_flops + ffn_flops)
        
        return total_flops
    
    def _analyze_scaling_laws(self) -> Dict[str, Any]:
        """
        Analyze scaling laws from results.
        
        Returns:
            Scaling analysis results
        """
        analysis = {
            "parameter_scaling": {},
            "compute_scaling": {},
            "efficiency_frontier": [],
            "optimal_scale": None
        }
        
        # Extract data for analysis
        params = []
        accuracies = []
        speeds = []
        memories = []
        
        for config_name, results in self.results["configurations"].items():
            params.append(results["num_parameters"])
            accuracies.append(results["mean_accuracy"])
            speeds.append(results["inference_speed"])
            memories.append(results["memory_usage"])
        
        if len(params) > 1:
            # Fit power law: accuracy = a * params^b
            log_params = np.log(params)
            coeffs = np.polyfit(log_params, accuracies, 1)
            
            analysis["parameter_scaling"] = {
                "scaling_exponent": float(coeffs[0]),
                "scaling_constant": float(coeffs[1]),
                "r_squared": float(np.corrcoef(log_params, accuracies)[0, 1] ** 2)
            }
            
            # Compute efficiency (accuracy per parameter)
            efficiencies = [acc / (param / 1e6) for acc, param in zip(accuracies, params)]
            best_efficiency_idx = np.argmax(efficiencies)
            
            analysis["optimal_scale"] = {
                "config": list(self.results["configurations"].keys())[best_efficiency_idx],
                "parameters": params[best_efficiency_idx],
                "accuracy": accuracies[best_efficiency_idx],
                "efficiency": efficiencies[best_efficiency_idx]
            }
            
            # Pareto frontier for accuracy vs speed
            frontier_indices = self._compute_pareto_frontier(accuracies, speeds)
            analysis["efficiency_frontier"] = [
                list(self.results["configurations"].keys())[i]
                for i in frontier_indices
            ]
        
        return analysis
    
    def _compute_pareto_frontier(
        self,
        objectives1: List[float],
        objectives2: List[float]
    ) -> List[int]:
        """
        Compute Pareto frontier for two objectives.
        
        Args:
            objectives1: First objective values (maximize)
            objectives2: Second objective values (maximize)
            
        Returns:
            Indices of Pareto optimal points
        """
        n_points = len(objectives1)
        pareto_indices = []
        
        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if (objectives1[j] >= objectives1[i] and 
                        objectives2[j] >= objectives2[i] and
                        (objectives1[j] > objectives1[i] or 
                         objectives2[j] > objectives2[i])):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def run_depth_ablation(self) -> Dict[str, Any]:
        """
        Specific ablation for model depth (number of layers).
        
        Returns:
            Depth ablation results
        """
        logger.info("Running depth-specific ablation")
        
        depths = [1, 2, 4, 6, 8, 12, 16, 24]
        base_config = {
            "hidden_size": 768,
            "num_attention_heads": 12,
            "intermediate_size": 3072
        }
        
        depth_results = {}
        dataset = self._load_dataset()
        
        for depth in depths:
            logger.info(f"Testing depth: {depth} layers")
            
            config = {
                **base_config,
                "num_hidden_layers": depth,
                "name": f"depth_{depth}"
            }
            
            results = self._test_configuration(config, dataset)
            depth_results[f"depth_{depth}"] = results
        
        # Analyze depth scaling
        analysis = self._analyze_depth_scaling(depth_results)
        
        return {
            "results": depth_results,
            "analysis": analysis
        }
    
    def run_width_ablation(self) -> Dict[str, Any]:
        """
        Specific ablation for model width (hidden dimensions).
        
        Returns:
            Width ablation results
        """
        logger.info("Running width-specific ablation")
        
        widths = [128, 256, 384, 512, 768, 1024, 1280]
        base_config = {
            "num_hidden_layers": 6,
            "num_attention_heads": 8
        }
        
        width_results = {}
        dataset = self._load_dataset()
        
        for width in widths:
            logger.info(f"Testing width: {width} dimensions")
            
            config = {
                **base_config,
                "hidden_size": width,
                "intermediate_size": width * 4,
                "name": f"width_{width}"
            }
            
            results = self._test_configuration(config, dataset)
            width_results[f"width_{width}"] = results
        
        # Analyze width scaling
        analysis = self._analyze_width_scaling(width_results)
        
        return {
            "results": width_results,
            "analysis": analysis
        }
    
    def _analyze_depth_scaling(self, depth_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze depth scaling patterns."""
        depths = []
        accuracies = []
        
        for key, result in depth_results.items():
            depth = int(key.split("_")[1])
            depths.append(depth)
            accuracies.append(result["mean_accuracy"])
        
        # Find optimal depth
        optimal_idx = np.argmax(accuracies)
        
        # Check for saturation
        saturation_point = None
        for i in range(1, len(accuracies)):
            if i > 0 and accuracies[i] - accuracies[i-1] < 0.001:
                saturation_point = depths[i-1]
                break
        
        return {
            "optimal_depth": depths[optimal_idx],
            "optimal_accuracy": accuracies[optimal_idx],
            "saturation_depth": saturation_point,
            "depth_importance": max(accuracies) - min(accuracies)
        }
    
    def _analyze_width_scaling(self, width_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze width scaling patterns."""
        widths = []
        accuracies = []
        
        for key, result in width_results.items():
            width = int(key.split("_")[1])
            widths.append(width)
            accuracies.append(result["mean_accuracy"])
        
        # Find optimal width
        optimal_idx = np.argmax(accuracies)
        
        # Calculate width efficiency
        efficiencies = [acc / (w / 768) for acc, w in zip(accuracies, widths)]
        
        return {
            "optimal_width": widths[optimal_idx],
            "optimal_accuracy": accuracies[optimal_idx],
            "width_efficiency": max(efficiencies),
            "width_importance": max(accuracies) - min(accuracies)
        }
    
    def _find_optimal_configurations(self) -> Dict[str, Any]:
        """
        Find optimal configurations for different constraints.
        
        Returns:
            Optimal configurations for various scenarios
        """
        configs = list(self.results["configurations"].values())
        
        # Sort by different metrics
        by_accuracy = max(configs, key=lambda x: x["mean_accuracy"])
        by_speed = max(configs, key=lambda x: x["inference_speed"])
        by_efficiency = max(configs, key=lambda x: x["mean_accuracy"] / (x["num_parameters"] / 1e6))
        
        # Find best under constraints
        small_models = [c for c in configs if c["num_parameters"] < 10e6]
        best_small = max(small_models, key=lambda x: x["mean_accuracy"]) if small_models else None
        
        fast_models = [c for c in configs if c["inference_speed"] > 100]
        best_fast = max(fast_models, key=lambda x: x["mean_accuracy"]) if fast_models else None
        
        return {
            "best_accuracy": {
                "config": by_accuracy["config"]["name"],
                "accuracy": by_accuracy["mean_accuracy"],
                "parameters": by_accuracy["num_parameters"]
            },
            "best_speed": {
                "config": by_speed["config"]["name"],
                "speed": by_speed["inference_speed"],
                "accuracy": by_speed["mean_accuracy"]
            },
            "best_efficiency": {
                "config": by_efficiency["config"]["name"],
                "efficiency": by_efficiency["mean_accuracy"] / (by_efficiency["num_parameters"] / 1e6),
                "accuracy": by_efficiency["mean_accuracy"]
            },
            "best_small": {
                "config": best_small["config"]["name"] if best_small else None,
                "accuracy": best_small["mean_accuracy"] if best_small else None,
                "parameters": best_small["num_parameters"] if best_small else None
            } if best_small else None,
            "best_fast": {
                "config": best_fast["config"]["name"] if best_fast else None,
                "accuracy": best_fast["mean_accuracy"] if best_fast else None,
                "speed": best_fast["inference_speed"] if best_fast else None
            } if best_fast else None
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive summary."""
        configs = list(self.results["configurations"].values())
        
        return {
            "num_configurations_tested": len(configs),
            "parameter_range": {
                "min": min(c["num_parameters"] for c in configs),
                "max": max(c["num_parameters"] for c in configs)
            },
            "accuracy_range": {
                "min": min(c["mean_accuracy"] for c in configs),
                "max": max(c["mean_accuracy"] for c in configs)
            },
            "speed_range": {
                "min": min(c["inference_speed"] for c in configs),
                "max": max(c["inference_speed"] for c in configs)
            },
            "scaling_coefficient": self.results["scaling_analysis"].get(
                "parameter_scaling", {}
            ).get("scaling_exponent", None),
            "optimal_scale": self.results["scaling_analysis"].get("optimal_scale", None),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        optimal = self.results["optimal_configurations"]
        
        # Accuracy vs efficiency trade-off
        if optimal.get("best_efficiency"):
            recommendations.append(
                f"For best efficiency, use {optimal['best_efficiency']['config']} "
                f"(efficiency: {optimal['best_efficiency']['efficiency']:.2f})"
            )
        
        # Speed constraints
        if optimal.get("best_fast"):
            recommendations.append(
                f"For real-time inference, use {optimal['best_fast']['config']} "
                f"({optimal['best_fast']['speed']:.1f} samples/sec)"
            )
        
        # Resource constraints
        if optimal.get("best_small"):
            recommendations.append(
                f"For edge deployment, use {optimal['best_small']['config']} "
                f"({optimal['best_small']['parameters']/1e6:.1f}M parameters)"
            )
        
        # Scaling insights
        if self.results["scaling_analysis"].get("parameter_scaling"):
            exp = self.results["scaling_analysis"]["parameter_scaling"]["scaling_exponent"]
            if exp < 0.5:
                recommendations.append(
                    "Diminishing returns observed - larger models may not be worth the cost"
                )
            else:
                recommendations.append(
                    "Strong scaling observed - consider larger models if resources permit"
                )
        
        return recommendations
    
    def _generate_visualizations(self):
        """Generate visualizations for model size ablation."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Extract data
        configs = list(self.results["configurations"].values())
        names = [c["config"]["name"] for c in configs]
        params = [c["num_parameters"] / 1e6 for c in configs]
        accuracies = [c["mean_accuracy"] for c in configs]
        speeds = [c["inference_speed"] for c in configs]
        memories = [c["memory_usage"] / 1e6 if c["memory_usage"] else 0 for c in configs]
        
        # 1. Accuracy vs Parameters
        ax = axes[0, 0]
        ax.scatter(params, accuracies, s=100, alpha=0.6)
        for i, name in enumerate(names):
            ax.annotate(name, (params[i], accuracies[i]), fontsize=8)
        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Model Size')
        ax.set_xscale('log')
        
        # 2. Speed vs Parameters
        ax = axes[0, 1]
        ax.scatter(params, speeds, s=100, alpha=0.6, color='orange')
        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('Inference Speed (samples/sec)')
        ax.set_title('Speed vs Model Size')
        ax.set_xscale('log')
        
        # 3. Memory vs Parameters
        ax = axes[0, 2]
        ax.scatter(params, memories, s=100, alpha=0.6, color='red')
        ax.set_xlabel('Parameters (M)')
        ax.set_ylabel('Memory Usage (MB)')
        ax.set_title('Memory vs Model Size')
        ax.set_xscale('log')
        
        # 4. Efficiency plot
        ax = axes[1, 0]
        efficiencies = [acc / param for acc, param in zip(accuracies, params)]
        bars = ax.bar(names, efficiencies, color='green', alpha=0.6)
        ax.set_xlabel('Model Configuration')
        ax.set_ylabel('Efficiency (Accuracy/M params)')
        ax.set_title('Model Efficiency')
        ax.tick_params(axis='x', rotation=45)
        
        # 5. Pareto frontier
        ax = axes[1, 1]
        ax.scatter(speeds, accuracies, s=100, alpha=0.6)
        
        # Highlight Pareto optimal points
        if self.results["scaling_analysis"].get("efficiency_frontier"):
            frontier_names = self.results["scaling_analysis"]["efficiency_frontier"]
            frontier_indices = [names.index(n) for n in frontier_names if n in names]
            frontier_speeds = [speeds[i] for i in frontier_indices]
            frontier_accs = [accuracies[i] for i in frontier_indices]
            ax.scatter(frontier_speeds, frontier_accs, s=200, alpha=0.8, 
                      color='red', marker='*', label='Pareto Optimal')
        
        ax.set_xlabel('Inference Speed (samples/sec)')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy-Speed Trade-off')
        ax.legend()
        
        # 6. Scaling law fit
        ax = axes[1, 2]
        if len(params) > 1:
            # Fit and plot scaling law
            log_params = np.log(params)
            coeffs = np.polyfit(log_params, accuracies, 1)
            fit_line = np.poly1d(coeffs)
            
            sorted_indices = np.argsort(params)
            ax.plot(np.array(params)[sorted_indices], 
                   fit_line(np.log(np.array(params)[sorted_indices])),
                   'r--', alpha=0.5, label='Power Law Fit')
            ax.scatter(params, accuracies, s=100, alpha=0.6)
            ax.set_xlabel('Parameters (M)')
            ax.set_ylabel('Accuracy')
            ax.set_title(f'Scaling Law (exponent: {coeffs[0]:.3f})')
            ax.set_xscale('log')
            ax.legend()
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "model_size_ablation_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {plot_path}")
        plt.show()
    
    def _get_training_config(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """Get training configuration adjusted for model size."""
        # Adjust batch size and learning rate based on model size
        hidden_size = model_config.get("hidden_size", 768)
        
        if hidden_size <= 256:
            batch_size = 64
            learning_rate = 5e-5
        elif hidden_size <= 512:
            batch_size = 32
            learning_rate = 3e-5
        else:
            batch_size = 16
            learning_rate = 2e-5
        
        return {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": 2,  # Fewer epochs for efficiency
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "max_length": model_config.get("max_position_embeddings", 256)
        }
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset for ablation study."""
        dataset = AGNewsDataset()
        return dataset.load_splits()
    
    def _save_results(self):
        """Save ablation results."""
        results_path = self.output_dir / "model_size_ablation_results.json"
        
        # Convert numpy types for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
        
        # Save summary as CSV
        df_data = []
        for name, result in self.results["configurations"].items():
            df_data.append({
                "configuration": name,
                "parameters": result["num_parameters"],
                "accuracy": result["mean_accuracy"],
                "std_accuracy": result["std_accuracy"],
                "f1_score": result["mean_f1"],
                "inference_speed": result["inference_speed"],
                "memory_usage": result["memory_usage"],
                "training_time": result["training_time"]
            })
        
        df = pd.DataFrame(df_data)
        csv_path = self.output_dir / "model_size_summary.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved summary to {csv_path}")
    
    def _make_serializable(self, obj):
        """Make object JSON serializable."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj


def run_model_size_ablation():
    """Run model size ablation study."""
    logger.info("Starting model size ablation study")
    
    ablation = ModelSizeAblation(
        base_model="bert",
        num_trials=2,
        profile_models=True
    )
    
    # Run main ablation
    results = ablation.run_ablation_study()
    
    # Run specific ablations
    depth_results = ablation.run_depth_ablation()
    width_results = ablation.run_width_ablation()
    
    logger.info(f"Main results: {results['summary']}")
    logger.info(f"Depth analysis: {depth_results['analysis']}")
    logger.info(f"Width analysis: {width_results['analysis']}")
    
    return results


if __name__ == "__main__":
    run_model_size_ablation()
