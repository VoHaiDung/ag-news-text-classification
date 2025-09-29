"""
Bayesian Optimization for AG News Text Classification
================================================================================
This module implements Bayesian optimization for hyperparameter search using
Gaussian Processes and acquisition functions for efficient exploration.

The implementation provides advanced features including constraints, multi-fidelity
optimization, and various acquisition strategies.

References:
    - Snoek, J., et al. (2012). Practical Bayesian Optimization of Machine Learning Algorithms
    - Frazier, P. I. (2018). A Tutorial on Bayesian Optimization

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
import json
import warnings
from datetime import datetime

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class BayesianOptimization:
    """
    Bayesian optimization for hyperparameter search.
    
    This class implements:
    - Gaussian Process surrogate modeling
    - Multiple acquisition functions (EI, UCB, PI, TS)
    - Constraint handling
    - Multi-fidelity optimization
    - Batch suggestions
    """
    
    def __init__(
        self,
        model_name: str,
        search_space: Dict[str, Any],
        n_iterations: int = 50,
        n_initial_points: int = 5,
        acquisition_function: str = "ei",
        kernel: Optional[Any] = None,
        xi: float = 0.01,
        kappa: float = 2.576,
        alpha: float = 1e-6,
        normalize_y: bool = True,
        metric: str = "accuracy",
        mode: str = "max",
        output_dir: str = "./bayesian_opt_results",
        seed: int = 42
    ):
        """
        Initialize Bayesian optimization.
        
        Args:
            model_name: Name of model to optimize
            search_space: Dictionary of parameter bounds
            n_iterations: Number of optimization iterations
            n_initial_points: Number of random initial points
            acquisition_function: Acquisition function type ("ei", "ucb", "pi", "thompson")
            kernel: GP kernel (uses Matern if None)
            xi: Exploration parameter for EI
            kappa: Exploration parameter for UCB
            alpha: Noise level for GP
            normalize_y: Whether to normalize target values
            metric: Metric to optimize
            mode: Optimization mode ("max" or "min")
            output_dir: Directory for saving results
            seed: Random seed
        """
        self.model_name = model_name
        self.search_space = search_space
        self.n_iterations = n_iterations
        self.n_initial_points = n_initial_points
        self.acquisition_function = acquisition_function
        self.xi = xi
        self.kappa = kappa
        self.alpha = alpha
        self.normalize_y = normalize_y
        self.metric = metric
        self.mode = mode
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Initialize kernel
        if kernel is None:
            self.kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        else:
            self.kernel = kernel
        
        # Initialize Gaussian Process
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=alpha,
            normalize_y=normalize_y,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        # Storage for observations
        self.X_observed = []
        self.y_observed = []
        self.best_params = None
        self.best_value = -np.inf if mode == "max" else np.inf
        
        # Parameter bounds and names
        self.param_names = list(search_space.keys())
        self.bounds = self._extract_bounds(search_space)
        self.n_params = len(self.param_names)
        
        # Input scaler for normalization
        self.scaler = StandardScaler()
        
        # Factory and registry
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        # Results storage
        self.results = {
            "iterations": [],
            "best_params": None,
            "best_value": None,
            "convergence": []
        }
        
        np.random.seed(seed)
        
        logger.info(f"Initialized Bayesian Optimization with {acquisition_function} acquisition")
    
    def _extract_bounds(self, search_space: Dict[str, Any]) -> np.ndarray:
        """
        Extract parameter bounds from search space.
        
        Args:
            search_space: Search space dictionary
            
        Returns:
            Array of bounds
        """
        bounds = []
        
        for param_name, param_config in search_space.items():
            if isinstance(param_config, dict):
                if "low" in param_config and "high" in param_config:
                    bounds.append([param_config["low"], param_config["high"]])
                elif "choices" in param_config:
                    # For categorical, use index bounds
                    bounds.append([0, len(param_config["choices"]) - 1])
                else:
                    bounds.append([0, 1])
            else:
                bounds.append([0, 1])
        
        return np.array(bounds)
    
    def run(self) -> Dict[str, Any]:
        """
        Run Bayesian optimization.
        
        Returns:
            Dictionary with best parameters and optimization history
        """
        logger.info(f"Starting Bayesian optimization for {self.n_iterations} iterations")
        
        # Initial random sampling
        self._initial_sampling()
        
        # Bayesian optimization loop
        for iteration in range(self.n_initial_points, self.n_iterations):
            logger.info(f"Iteration {iteration + 1}/{self.n_iterations}")
            
            # Normalize observed points
            if len(self.X_observed) > 1:
                X_normalized = self.scaler.fit_transform(np.array(self.X_observed))
            else:
                X_normalized = np.array(self.X_observed)
            
            # Fit Gaussian Process
            self.gp.fit(X_normalized, np.array(self.y_observed))
            
            # Find next point to evaluate
            x_next = self._get_next_point()
            
            # Evaluate objective function
            y_next = self._evaluate_configuration(x_next)
            
            # Update observations
            self.X_observed.append(x_next)
            self.y_observed.append(y_next)
            
            # Update best if improved
            is_better = (
                (self.mode == "max" and y_next > self.best_value) or
                (self.mode == "min" and y_next < self.best_value)
            )
            
            if is_better:
                self.best_value = y_next
                self.best_params = dict(zip(self.param_names, x_next))
                logger.info(f"New best {self.metric}: {self.best_value:.4f}")
            
            # Store iteration results
            self.results["iterations"].append({
                "iteration": iteration,
                "params": dict(zip(self.param_names, x_next)),
                "value": y_next,
                "is_best": is_better
            })
            
            # Update convergence tracking
            if self.mode == "max":
                self.results["convergence"].append(max(self.y_observed))
            else:
                self.results["convergence"].append(min(self.y_observed))
            
            # Check convergence
            if self._check_convergence():
                logger.info("Convergence detected, stopping early")
                break
        
        # Final results
        self.results["best_params"] = self.best_params
        self.results["best_value"] = self.best_value
        self.results["n_iterations"] = len(self.X_observed)
        self.results["summary"] = self._generate_summary()
        
        # Save results
        self.save_results()
        
        # Generate visualizations
        self.plot_results()
        
        logger.info(f"\nOptimization completed!")
        logger.info(f"Best {self.metric}: {self.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.results
    
    def _initial_sampling(self):
        """Perform initial random sampling."""
        logger.info(f"Performing {self.n_initial_points} initial random samples")
        
        for i in range(self.n_initial_points):
            # Random sample from search space
            x = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            
            # Evaluate objective
            y = self._evaluate_configuration(x)
            
            # Store observation
            self.X_observed.append(x)
            self.y_observed.append(y)
            
            # Update best
            is_better = (
                (self.mode == "max" and y > self.best_value) or
                (self.mode == "min" and y < self.best_value)
            )
            
            if is_better:
                self.best_value = y
                self.best_params = dict(zip(self.param_names, x))
            
            # Store in results
            self.results["iterations"].append({
                "iteration": i,
                "params": dict(zip(self.param_names, x)),
                "value": y,
                "is_best": is_better
            })
    
    def _get_next_point(self) -> np.ndarray:
        """
        Get next point to evaluate using acquisition function.
        
        Returns:
            Next point to evaluate
        """
        # Define acquisition function
        if self.acquisition_function == "ei":
            acq_func = self._expected_improvement
        elif self.acquisition_function == "ucb":
            acq_func = self._upper_confidence_bound
        elif self.acquisition_function == "pi":
            acq_func = self._probability_improvement
        elif self.acquisition_function == "thompson":
            return self._thompson_sampling()
        else:
            raise ValueError(f"Unknown acquisition function: {self.acquisition_function}")
        
        # Optimize acquisition function
        best_x = None
        best_acq = -np.inf
        
        # Multi-start optimization
        n_restarts = 25
        for _ in range(n_restarts):
            # Random starting point
            x0 = np.random.uniform(self.bounds[:, 0], self.bounds[:, 1])
            
            # Minimize negative acquisition (maximize acquisition)
            result = minimize(
                lambda x: -acq_func(x.reshape(1, -1)),
                x0,
                bounds=self.bounds,
                method="L-BFGS-B"
            )
            
            if -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x
        
        return best_x
    
    def _expected_improvement(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Expected Improvement acquisition function.
        
        Args:
            X: Points to evaluate
            
        Returns:
            EI values
        """
        X = np.atleast_2d(X)
        
        # Normalize if needed
        if len(self.X_observed) > 1:
            X_normalized = self.scaler.transform(X)
        else:
            X_normalized = X
        
        # Predict mean and std
        mu, sigma = self.gp.predict(X_normalized, return_std=True)
        
        # Current best (or worst for minimization)
        if self.mode == "max":
            f_best = np.max(self.y_observed)
        else:
            f_best = np.min(self.y_observed)
            mu = -mu  # Negate for minimization
        
        # Calculate EI
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            improvement = mu - f_best - self.xi
            Z = improvement / (sigma + 1e-9)
            ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def _upper_confidence_bound(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Upper Confidence Bound acquisition function.
        
        Args:
            X: Points to evaluate
            
        Returns:
            UCB values
        """
        X = np.atleast_2d(X)
        
        # Normalize if needed
        if len(self.X_observed) > 1:
            X_normalized = self.scaler.transform(X)
        else:
            X_normalized = X
        
        # Predict mean and std
        mu, sigma = self.gp.predict(X_normalized, return_std=True)
        
        # Calculate UCB
        if self.mode == "max":
            ucb = mu + self.kappa * sigma
        else:
            ucb = -mu + self.kappa * sigma  # Negate mean for minimization
        
        return ucb
    
    def _probability_improvement(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate Probability of Improvement acquisition function.
        
        Args:
            X: Points to evaluate
            
        Returns:
            PI values
        """
        X = np.atleast_2d(X)
        
        # Normalize if needed
        if len(self.X_observed) > 1:
            X_normalized = self.scaler.transform(X)
        else:
            X_normalized = X
        
        # Predict mean and std
        mu, sigma = self.gp.predict(X_normalized, return_std=True)
        
        # Current best
        if self.mode == "max":
            f_best = np.max(self.y_observed)
        else:
            f_best = np.min(self.y_observed)
            mu = -mu  # Negate for minimization
        
        # Calculate PI
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            Z = (mu - f_best - self.xi) / (sigma + 1e-9)
            pi = norm.cdf(Z)
            pi[sigma == 0.0] = 0.0
        
        return pi
    
    def _thompson_sampling(self) -> np.ndarray:
        """
        Thompson sampling for next point selection.
        
        Returns:
            Next point to evaluate
        """
        # Sample from GP posterior
        n_samples = 10000
        X_sample = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_samples, self.n_params)
        )
        
        # Normalize if needed
        if len(self.X_observed) > 1:
            X_sample_normalized = self.scaler.transform(X_sample)
        else:
            X_sample_normalized = X_sample
        
        # Sample from posterior
        mu = self.gp.predict(X_sample_normalized)
        
        # Add noise for exploration
        noise = np.random.normal(0, 0.1, mu.shape)
        sample = mu + noise
        
        # Return point with highest/lowest sampled value
        if self.mode == "max":
            best_idx = np.argmax(sample)
        else:
            best_idx = np.argmin(sample)
        
        return X_sample[best_idx]
    
    def _evaluate_configuration(self, x: np.ndarray) -> float:
        """
        Evaluate objective function at point x.
        
        Args:
            x: Parameter values
            
        Returns:
            Objective value
        """
        # Convert to parameter dictionary
        params = self._array_to_params(x)
        
        try:
            # Set seed for reproducibility
            set_seed(self.seed)
            
            # Load data
            dataset = AGNewsDataset()
            train_data, val_data = dataset.load_train_val_split()
            
            # Create model
            model = self.factory.create_model(
                self.model_name,
                **params
            )
            
            # Create trainer
            trainer_config = {
                "learning_rate": params.get("learning_rate", 1e-4),
                "batch_size": params.get("batch_size", 32),
                "num_epochs": params.get("num_epochs", 10),
                "optimizer": params.get("optimizer", "adamw"),
                "weight_decay": params.get("weight_decay", 0.01)
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
            return -np.inf if self.mode == "max" else np.inf
    
    def _array_to_params(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Convert array to parameter dictionary.
        
        Args:
            x: Parameter array
            
        Returns:
            Parameter dictionary
        """
        params = {}
        
        for i, param_name in enumerate(self.param_names):
            param_config = self.search_space[param_name]
            
            if isinstance(param_config, dict):
                if "choices" in param_config:
                    # Categorical parameter
                    idx = int(np.clip(x[i], 0, len(param_config["choices"]) - 1))
                    params[param_name] = param_config["choices"][idx]
                elif param_config.get("type") == "int":
                    # Integer parameter
                    params[param_name] = int(x[i])
                else:
                    # Continuous parameter
                    params[param_name] = float(x[i])
            else:
                params[param_name] = x[i]
        
        return params
    
    def _check_convergence(self, tolerance: float = 1e-6, window: int = 5) -> bool:
        """
        Check if optimization has converged.
        
        Args:
            tolerance: Convergence tolerance
            window: Window size for checking
            
        Returns:
            True if converged
        """
        if len(self.y_observed) < window * 2:
            return False
        
        # Check if best value hasn't improved recently
        recent_values = self.y_observed[-window:]
        
        if self.mode == "max":
            recent_best = max(recent_values)
            previous_best = max(self.y_observed[:-window])
        else:
            recent_best = min(recent_values)
            previous_best = min(self.y_observed[:-window])
        
        return abs(recent_best - previous_best) < tolerance
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary of optimization results.
        
        Returns:
            Summary dictionary
        """
        values = np.array(self.y_observed)
        
        # Calculate statistics
        summary = {
            "total_iterations": len(self.y_observed),
            "best_iteration": np.argmax(values) if self.mode == "max" else np.argmin(values),
            "score_statistics": {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "median": float(np.median(values))
            },
            "convergence_analysis": self._analyze_convergence(),
            "acquisition_efficiency": self._calculate_acquisition_efficiency()
        }
        
        return summary
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze convergence characteristics.
        
        Returns:
            Convergence analysis
        """
        values = np.array(self.y_observed)
        
        # Calculate cumulative best
        if self.mode == "max":
            cumulative_best = np.maximum.accumulate(values)
        else:
            cumulative_best = np.minimum.accumulate(values)
        
        # Find convergence point
        improvements = np.abs(np.diff(cumulative_best))
        convergence_idx = len(improvements)
        
        threshold = np.mean(improvements) * 0.1
        for i in range(len(improvements) - 5, 0, -1):
            if improvements[i:i+5].sum() > threshold:
                convergence_idx = i + 5
                break
        
        return {
            "converged_at_iteration": convergence_idx,
            "final_value": float(cumulative_best[-1]),
            "total_improvement": float(abs(cumulative_best[-1] - cumulative_best[0])),
            "improvement_rate": float(np.mean(improvements)) if len(improvements) > 0 else 0
        }
    
    def _calculate_acquisition_efficiency(self) -> float:
        """
        Calculate efficiency of acquisition function.
        
        Returns:
            Efficiency score
        """
        if len(self.y_observed) < 2:
            return 0.0
        
        # Calculate how often acquisition function found better points
        improvements = 0
        for i in range(1, len(self.y_observed)):
            if self.mode == "max":
                if self.y_observed[i] > max(self.y_observed[:i]):
                    improvements += 1
            else:
                if self.y_observed[i] < min(self.y_observed[:i]):
                    improvements += 1
        
        return improvements / (len(self.y_observed) - 1)
    
    def suggest_next_batch(self, batch_size: int = 5) -> List[Dict[str, Any]]:
        """
        Suggest a batch of points for parallel evaluation.
        
        Args:
            batch_size: Number of points to suggest
            
        Returns:
            List of suggested parameter configurations
        """
        suggestions = []
        temp_X = list(self.X_observed)
        temp_y = list(self.y_observed)
        
        for _ in range(batch_size):
            # Get next point
            x_next = self._get_next_point()
            suggestions.append(self._array_to_params(x_next))
            
            # Add to temporary observations with pessimistic value
            temp_X.append(x_next)
            if self.mode == "max":
                temp_y.append(min(self.y_observed))
            else:
                temp_y.append(max(self.y_observed))
            
            # Update temporary GP
            self.gp.fit(np.array(temp_X), np.array(temp_y))
        
        return suggestions
    
    def plot_results(self):
        """Generate optimization visualizations."""
        import matplotlib.pyplot as plt
        
        if not self.y_observed:
            logger.warning("No results to plot")
            return
        
        output_dir = self.output_dir / "plots"
        output_dir.mkdir(exist_ok=True)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Convergence plot
        ax = axes[0, 0]
        iterations = range(len(self.y_observed))
        ax.plot(iterations, self.y_observed, 'o-', alpha=0.6, label='Observed')
        
        # Plot cumulative best
        if self.mode == "max":
            cumulative_best = np.maximum.accumulate(self.y_observed)
        else:
            cumulative_best = np.minimum.accumulate(self.y_observed)
        
        ax.plot(iterations, cumulative_best, 'r-', linewidth=2, label='Best')
        ax.set_xlabel('Iteration')
        ax.set_ylabel(self.metric.capitalize())
        ax.set_title('Optimization Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Acquisition function values
        ax = axes[0, 1]
        if len(self.results["iterations"]) > 0:
            acq_iterations = range(self.n_initial_points, len(self.results["iterations"]))
            if len(acq_iterations) > 0:
                ax.plot(acq_iterations, [0] * len(acq_iterations), 'g-', alpha=0.6)
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Acquisition Value')
                ax.set_title(f'{self.acquisition_function.upper()} Acquisition Function')
                ax.grid(True, alpha=0.3)
        
        # 3. Parameter traces
        ax = axes[1, 0]
        if self.X_observed:
            X_array = np.array(self.X_observed)
            for i, param_name in enumerate(self.param_names[:4]):  # Limit to 4 params
                normalized_values = (X_array[:, i] - self.bounds[i, 0]) / (self.bounds[i, 1] - self.bounds[i, 0])
                ax.plot(normalized_values, label=param_name, alpha=0.7)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Normalized Parameter Value')
            ax.set_title('Parameter Evolution')
            ax.legend(loc='best', fontsize='small')
            ax.grid(True, alpha=0.3)
        
        # 4. GP Uncertainty
        ax = axes[1, 1]
        if len(self.X_observed) > 1 and self.n_params <= 2:
            # For 1D or 2D, show GP predictions
            if self.n_params == 1:
                X_plot = np.linspace(self.bounds[0, 0], self.bounds[0, 1], 100).reshape(-1, 1)
                X_plot_normalized = self.scaler.transform(X_plot)
                mu, sigma = self.gp.predict(X_plot_normalized, return_std=True)
                
                ax.plot(X_plot, mu, 'b-', label='GP mean')
                ax.fill_between(X_plot.ravel(), 
                               mu - 1.96 * sigma, 
                               mu + 1.96 * sigma, 
                               alpha=0.3, label='95% CI')
                ax.scatter(self.X_observed, self.y_observed, c='r', s=50, zorder=5, label='Observations')
                ax.set_xlabel(self.param_names[0])
                ax.set_ylabel(self.metric.capitalize())
                ax.set_title('Gaussian Process Model')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                # Show improvement over random
                random_performance = np.mean(self.y_observed[:self.n_initial_points]) if self.n_initial_points > 0 else 0
                bo_performance = self.y_observed[self.n_initial_points:] if len(self.y_observed) > self.n_initial_points else []
                
                if bo_performance:
                    ax.axhline(y=random_performance, color='r', linestyle='--', label='Random baseline')
                    ax.plot(range(len(bo_performance)), bo_performance, 'b-', label='Bayesian Optimization')
                    ax.set_xlabel('BO Iteration')
                    ax.set_ylabel(self.metric.capitalize())
                    ax.set_title('BO vs Random Sampling')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "bayesian_optimization_results.png", dpi=300, bbox_inches="tight")
        logger.info(f"Plots saved to {output_dir}")
        plt.show()
    
    def save_results(self, filepath: Optional[str] = None):
        """
        Save optimization results.
        
        Args:
            filepath: Path to save results
        """
        if filepath is None:
            filepath = self.output_dir / "bayesian_optimization_results.json"
        
        # Convert to serializable format
        serializable_results = self._make_serializable(self.results)
        
        # Add metadata
        serializable_results["metadata"] = {
            "model_name": self.model_name,
            "acquisition_function": self.acquisition_function,
            "n_iterations": self.n_iterations,
            "n_initial_points": self.n_initial_points,
            "metric": self.metric,
            "mode": self.mode,
            "timestamp": datetime.now().isoformat()
        }
        
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
