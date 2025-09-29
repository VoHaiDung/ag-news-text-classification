"""
HyperBand Optimization for AG News Text Classification
================================================================================
This module implements the HyperBand algorithm for efficient hyperparameter
optimization with adaptive resource allocation and early stopping.

HyperBand extends Successive Halving by running multiple brackets with different
resource allocations, providing theoretical guarantees on performance.

References:
    - Li, L., et al. (2017). Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization
    - Jamieson, K., & Talwalkar, A. (2016). Non-stochastic Best Arm Identification and Hyperparameter Optimization

Author: Võ Hải Dũng
License: MIT
"""

import logging
import math
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
import json
from datetime import datetime
import numpy as np
import pandas as pd
from collections import defaultdict

import torch
from sklearn.model_selection import ParameterSampler

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class HyperBand:
    """
    HyperBand algorithm for hyperparameter optimization.
    
    This class implements:
    - Successive halving with multiple brackets
    - Adaptive resource allocation
    - Early stopping based on performance
    - Efficient exploration-exploitation trade-off
    """
    
    def __init__(
        self,
        model_name: str,
        search_space: Dict[str, Any],
        max_iter: int = 81,  # Maximum iterations per configuration
        eta: int = 3,  # Downsampling rate
        metric: str = "accuracy",
        mode: str = "max",
        resource_name: str = "num_epochs",
        min_resource: int = 1,
        output_dir: str = "./hyperband_results",
        seed: int = 42
    ):
        """
        Initialize HyperBand optimizer.
        
        Args:
            model_name: Name of model to optimize
            search_space: Hyperparameter search space
            max_iter: Maximum resource allocation (R)
            eta: Downsampling rate
            metric: Metric to optimize
            mode: Optimization mode ("max" or "min")
            resource_name: Name of resource to allocate
            min_resource: Minimum resource allocation
            output_dir: Directory for saving results
            seed: Random seed
        """
        self.model_name = model_name
        self.search_space = search_space
        self.max_iter = max_iter
        self.eta = eta
        self.metric = metric
        self.mode = mode
        self.resource_name = resource_name
        self.min_resource = min_resource
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        # Calculate number of brackets
        self.s_max = int(math.log(max_iter / min_resource) / math.log(eta))
        self.B = (self.s_max + 1) * max_iter
        
        # Results storage
        self.results = {
            "brackets": [],
            "best_config": None,
            "best_score": None,
            "all_evaluations": []
        }
        
        logger.info(f"Initialized HyperBand with s_max={self.s_max}, B={self.B}")
    
    def run(self) -> Dict[str, Any]:
        """
        Run HyperBand optimization.
        
        Returns:
            Optimization results
        """
        logger.info(f"Starting HyperBand optimization for {self.model_name}")
        
        best_score = -float('inf') if self.mode == "max" else float('inf')
        best_config = None
        
        # Run each bracket
        for s in reversed(range(self.s_max + 1)):
            # Calculate parameters for this bracket
            n = int(math.ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))
            r = self.max_iter * self.eta ** (-s)
            
            logger.info(f"\nBracket {self.s_max - s}/{self.s_max}: n={n}, r={r:.2f}")
            
            # Run successive halving for this bracket
            bracket_results = self._successive_halving(n, r, s)
            
            # Update best configuration
            if bracket_results["best_score"] is not None:
                is_better = (
                    (self.mode == "max" and bracket_results["best_score"] > best_score) or
                    (self.mode == "min" and bracket_results["best_score"] < best_score)
                )
                
                if is_better:
                    best_score = bracket_results["best_score"]
                    best_config = bracket_results["best_config"]
            
            # Store bracket results
            self.results["brackets"].append({
                "bracket_id": self.s_max - s,
                "s": s,
                "n": n,
                "r": r,
                "results": bracket_results
            })
        
        # Compile final results
        self.results["best_config"] = best_config
        self.results["best_score"] = best_score
        self.results["summary"] = self._generate_summary()
        
        logger.info(f"\nHyperBand completed!")
        logger.info(f"Best {self.metric}: {best_score:.4f}")
        logger.info(f"Best config: {best_config}")
        
        return self.results
    
    def _successive_halving(
        self,
        n: int,
        r: float,
        s: int
    ) -> Dict[str, Any]:
        """
        Run successive halving for one bracket.
        
        Args:
            n: Initial number of configurations
            r: Initial resource allocation
            s: Bracket index
            
        Returns:
            Bracket results
        """
        # Sample n configurations
        configurations = self._sample_configurations(n)
        
        # Track results for this bracket
        bracket_results = {
            "configurations": [],
            "best_config": None,
            "best_score": None,
            "rounds": []
        }
        
        # Successive halving rounds
        for i in range(s + 1):
            # Calculate resource allocation for this round
            n_i = int(n * self.eta ** (-i))
            r_i = int(r * self.eta ** i)
            
            logger.info(f"  Round {i}: n={n_i}, r={r_i}")
            
            # Evaluate configurations with r_i resources
            round_results = []
            
            for j, config in enumerate(configurations[:n_i]):
                logger.info(f"    Evaluating config {j+1}/{n_i}")
                
                # Set resource in config
                config[self.resource_name] = r_i
                
                # Evaluate configuration
                score = self._evaluate_configuration(config)
                
                round_results.append({
                    "config": config,
                    "score": score,
                    "resource": r_i
                })
                
                # Track in all evaluations
                self.results["all_evaluations"].append({
                    "bracket": self.s_max - s,
                    "round": i,
                    "config": config,
                    "score": score,
                    "resource": r_i
                })
            
            # Sort configurations by score
            round_results = sorted(
                round_results,
                key=lambda x: x["score"],
                reverse=(self.mode == "max")
            )
            
            # Keep top configurations for next round
            keep_count = int(n_i / self.eta)
            configurations = [r["config"] for r in round_results[:keep_count]]
            
            # Store round results
            bracket_results["rounds"].append({
                "round": i,
                "n": n_i,
                "r": r_i,
                "results": round_results,
                "kept": keep_count
            })
            
            # Update best in bracket
            if round_results:
                best_in_round = round_results[0]
                
                if bracket_results["best_score"] is None:
                    bracket_results["best_score"] = best_in_round["score"]
                    bracket_results["best_config"] = best_in_round["config"]
                else:
                    is_better = (
                        (self.mode == "max" and best_in_round["score"] > bracket_results["best_score"]) or
                        (self.mode == "min" and best_in_round["score"] < bracket_results["best_score"])
                    )
                    
                    if is_better:
                        bracket_results["best_score"] = best_in_round["score"]
                        bracket_results["best_config"] = best_in_round["config"]
        
        return bracket_results
    
    def _sample_configurations(self, n: int) -> List[Dict[str, Any]]:
        """
        Sample n configurations from search space.
        
        Args:
            n: Number of configurations to sample
            
        Returns:
            List of configurations
        """
        # Convert search space to sklearn format
        param_distributions = {}
        
        for param_name, param_config in self.search_space.items():
            if isinstance(param_config, dict):
                param_type = param_config.get("type", "uniform")
                
                if param_type == "uniform":
                    from scipy.stats import uniform
                    param_distributions[param_name] = uniform(
                        param_config["low"],
                        param_config["high"] - param_config["low"]
                    )
                elif param_type == "loguniform":
                    from scipy.stats import loguniform
                    param_distributions[param_name] = loguniform(
                        param_config["low"],
                        param_config["high"]
                    )
                elif param_type == "choice":
                    param_distributions[param_name] = param_config["choices"]
                elif param_type == "randint":
                    from scipy.stats import randint
                    param_distributions[param_name] = randint(
                        param_config["low"],
                        param_config["high"]
                    )
            else:
                param_distributions[param_name] = [param_config]
        
        # Sample configurations
        sampler = ParameterSampler(
            param_distributions,
            n_iter=n,
            random_state=self.seed
        )
        
        return list(sampler)
    
    def _evaluate_configuration(
        self,
        config: Dict[str, Any]
    ) -> float:
        """
        Evaluate a configuration.
        
        Args:
            config: Hyperparameter configuration
            
        Returns:
            Metric score
        """
        try:
            # Set seed for reproducibility
            set_seed(self.seed)
            
            # Load data
            dataset = AGNewsDataset()
            train_data, val_data = dataset.load_train_val_split()
            
            # Create model
            model_config = {k: v for k, v in config.items() if k != self.resource_name}
            model = self.factory.create_model(
                self.model_name,
                **model_config
            )
            
            # Create trainer
            trainer_config = {
                "learning_rate": config.get("learning_rate", 1e-4),
                "batch_size": config.get("batch_size", 32),
                "num_epochs": config.get(self.resource_name, 1),
                "optimizer": config.get("optimizer", "adamw"),
                "weight_decay": config.get("weight_decay", 0.01)
            }
            
            trainer = BaseTrainer(
                model=model,
                config=trainer_config,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Train model
            for epoch in range(trainer_config["num_epochs"]):
                trainer.train_epoch(train_data)
            
            # Evaluate
            val_metrics = trainer.validate(val_data)
            
            # Get metric score
            score = val_metrics.get(self.metric, val_metrics.get("accuracy", 0))
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Error evaluating configuration: {e}")
            return 0.0 if self.mode == "max" else float('inf')
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary of HyperBand results.
        
        Returns:
            Summary dictionary
        """
        # Calculate statistics
        all_scores = [
            eval_result["score"]
            for eval_result in self.results["all_evaluations"]
            if eval_result["score"] is not None
        ]
        
        if not all_scores:
            return {}
        
        # Resource usage analysis
        resource_usage = defaultdict(list)
        for eval_result in self.results["all_evaluations"]:
            resource_usage[eval_result["resource"]].append(eval_result["score"])
        
        resource_efficiency = {}
        for resource, scores in resource_usage.items():
            resource_efficiency[resource] = {
                "mean_score": float(np.mean(scores)),
                "std_score": float(np.std(scores)),
                "num_evaluations": len(scores)
            }
        
        # Bracket performance
        bracket_performance = []
        for bracket in self.results["brackets"]:
            bracket_performance.append({
                "bracket_id": bracket["bracket_id"],
                "best_score": bracket["results"]["best_score"],
                "num_configs": bracket["n"],
                "initial_resource": bracket["r"]
            })
        
        return {
            "total_evaluations": len(self.results["all_evaluations"]),
            "num_brackets": len(self.results["brackets"]),
            "score_statistics": {
                "mean": float(np.mean(all_scores)),
                "std": float(np.std(all_scores)),
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores)),
                "median": float(np.median(all_scores))
            },
            "resource_efficiency": resource_efficiency,
            "bracket_performance": bracket_performance,
            "total_resource_used": sum(
                eval_result["resource"]
                for eval_result in self.results["all_evaluations"]
            )
        }
    
    def plot_results(self):
        """Plot HyperBand optimization results."""
        import matplotlib.pyplot as plt
        
        if not self.results["all_evaluations"]:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Score progression
        ax = axes[0, 0]
        scores = [r["score"] for r in self.results["all_evaluations"]]
        iterations = range(len(scores))
        
        ax.plot(iterations, scores, 'b-', alpha=0.5, label="All evaluations")
        
        # Mark best scores per bracket
        bracket_ends = []
        current_idx = 0
        for bracket in self.results["brackets"]:
            num_evals = sum(
                len(round_data["results"])
                for round_data in bracket["results"]["rounds"]
            )
            bracket_ends.append(current_idx + num_evals)
            current_idx += num_evals
        
        for i, end_idx in enumerate(bracket_ends):
            ax.axvline(x=end_idx, color='r', linestyle='--', alpha=0.3)
            ax.text(end_idx, ax.get_ylim()[1], f"B{i}", ha='center')
        
        ax.set_xlabel("Evaluation")
        ax.set_ylabel(self.metric.capitalize())
        ax.set_title("Score Progression")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Resource vs Score
        ax = axes[0, 1]
        resources = [r["resource"] for r in self.results["all_evaluations"]]
        
        ax.scatter(resources, scores, alpha=0.6)
        ax.set_xlabel(f"Resource ({self.resource_name})")
        ax.set_ylabel(self.metric.capitalize())
        ax.set_title("Resource Allocation vs Performance")
        ax.grid(True, alpha=0.3)
        
        # 3. Bracket comparison
        ax = axes[1, 0]
        bracket_ids = []
        bracket_scores = []
        
        for bracket in self.results["brackets"]:
            if bracket["results"]["best_score"] is not None:
                bracket_ids.append(bracket["bracket_id"])
                bracket_scores.append(bracket["results"]["best_score"])
        
        ax.bar(bracket_ids, bracket_scores)
        ax.set_xlabel("Bracket ID")
        ax.set_ylabel(f"Best {self.metric.capitalize()}")
        ax.set_title("Bracket Performance Comparison")
        ax.grid(True, alpha=0.3)
        
        # 4. Resource efficiency
        ax = axes[1, 1]
        resource_levels = sorted(set(resources))
        mean_scores = []
        
        for r in resource_levels:
            level_scores = [
                s for s, res in zip(scores, resources) if res == r
            ]
            if level_scores:
                mean_scores.append(np.mean(level_scores))
            else:
                mean_scores.append(0)
        
        ax.plot(resource_levels, mean_scores, 'o-')
        ax.set_xlabel(f"Resource ({self.resource_name})")
        ax.set_ylabel(f"Mean {self.metric.capitalize()}")
        ax.set_title("Resource Efficiency")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "hyperband_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {plot_path}")
        
        plt.show()
    
    def save_results(self, filepath: Optional[str] = None):
        """
        Save HyperBand results.
        
        Args:
            filepath: Path to save results
        """
        if filepath is None:
            filepath = self.output_dir / "hyperband_results.json"
        
        # Convert to serializable format
        serializable_results = self._make_serializable(self.results)
        
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
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
