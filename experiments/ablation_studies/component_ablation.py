"""
Component Ablation Study for AG News Text Classification
================================================================================
This module performs systematic ablation studies to analyze the contribution of
different components in the text classification pipeline.

Component ablation helps identify critical components and their relative importance
by systematically removing or modifying them and measuring performance impact.

References:
    - Meyes, R., et al. (2019). Ablation Studies in Artificial Neural Networks
    - Lipton, Z. C., & Steinhardt, J. (2019). Troubleshooting Deep Neural Networks

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import json
from datetime import datetime
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.core.factory import Factory
from src.core.registry import Registry
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


class ComponentAblation:
    """
    Performs component ablation studies for text classification models.
    
    This class systematically evaluates the impact of removing or modifying
    individual components to understand their contribution to overall performance.
    """
    
    def __init__(
        self,
        base_model_name: str = "bert-base",
        base_config: Optional[Dict[str, Any]] = None,
        components_to_ablate: Optional[List[str]] = None,
        dataset_name: str = "ag_news",
        num_trials: int = 3,
        device: str = "cuda",
        output_dir: str = "./ablation_results/component",
        seed: int = 42
    ):
        """
        Initialize component ablation study.
        
        Args:
            base_model_name: Name of base model
            base_config: Base configuration
            components_to_ablate: List of components to ablate
            dataset_name: Dataset name
            num_trials: Number of trials per configuration
            device: Device to use
            output_dir: Output directory
            seed: Random seed
        """
        self.base_model_name = base_model_name
        self.base_config = base_config or self._get_default_config()
        self.components_to_ablate = components_to_ablate or self._get_default_components()
        self.dataset_name = dataset_name
        self.num_trials = num_trials
        self.device = device if torch.cuda.is_available() else "cpu"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        
        # Initialize components
        self.factory = Factory()
        self.registry = Registry()
        self.metrics_calculator = ClassificationMetrics()
        
        # Results storage
        self.results = {
            "baseline": {},
            "ablations": {},
            "summary": {},
            "importance_scores": {}
        }
        
        set_seed(seed)
        logger.info(f"Initialized Component Ablation for {base_model_name}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "embedding_dim": 768,
            "hidden_dim": 768,
            "num_layers": 12,
            "num_heads": 12,
            "dropout": 0.1,
            "max_length": 256,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "num_epochs": 3,
            "use_attention": True,
            "use_layer_norm": True,
            "use_residual": True,
            "use_position_embeddings": True,
            "pooling_strategy": "cls",
            "classifier_layers": 2,
            "activation": "gelu",
            "optimizer": "adamw",
            "scheduler": "linear",
            "warmup_ratio": 0.1,
            "weight_decay": 0.01,
            "gradient_clipping": 1.0,
            "label_smoothing": 0.0,
            "mixup_alpha": 0.0,
            "use_data_augmentation": True,
            "use_ensemble": False
        }
    
    def _get_default_components(self) -> List[str]:
        """Get default components to ablate."""
        return [
            "attention",
            "layer_norm",
            "residual_connections",
            "position_embeddings",
            "dropout",
            "warmup",
            "weight_decay",
            "gradient_clipping",
            "data_augmentation",
            "multi_layer_classifier",
            "advanced_pooling",
            "scheduler",
            "label_smoothing",
            "mixup"
        ]
    
    def run_ablation_study(self) -> Dict[str, Any]:
        """
        Run complete component ablation study.
        
        Returns:
            Ablation study results
        """
        logger.info("Starting component ablation study")
        
        # Load data
        dataset = self._load_dataset()
        
        # Run baseline
        logger.info("Training baseline model")
        self.results["baseline"] = self._train_baseline(dataset)
        baseline_score = self.results["baseline"]["mean_accuracy"]
        logger.info(f"Baseline accuracy: {baseline_score:.4f}")
        
        # Run ablations
        for component in self.components_to_ablate:
            logger.info(f"\nAblating component: {component}")
            
            ablation_results = self._ablate_component(component, dataset)
            self.results["ablations"][component] = ablation_results
            
            # Calculate importance score
            importance = self._calculate_importance(
                baseline_score,
                ablation_results["mean_accuracy"]
            )
            self.results["importance_scores"][component] = importance
            
            logger.info(
                f"Component: {component} | "
                f"Ablated accuracy: {ablation_results['mean_accuracy']:.4f} | "
                f"Importance: {importance:.4f}"
            )
        
        # Generate summary
        self.results["summary"] = self._generate_summary()
        
        # Save results
        self._save_results()
        
        # Generate visualizations
        self._generate_visualizations()
        
        return self.results
    
    def _train_baseline(self, dataset: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train baseline model with all components.
        
        Args:
            dataset: Dataset dictionary
            
        Returns:
            Baseline results
        """
        results = {
            "trials": [],
            "mean_accuracy": 0,
            "std_accuracy": 0,
            "mean_f1": 0,
            "training_time": 0
        }
        
        accuracies = []
        f1_scores = []
        
        for trial in range(self.num_trials):
            logger.info(f"Baseline trial {trial + 1}/{self.num_trials}")
            
            # Set seed for reproducibility
            set_seed(self.seed + trial)
            
            # Create model
            model = self._create_model(self.base_config)
            
            # Train model
            trainer = BaseTrainer(
                model=model,
                config=self.base_config,
                device=self.device
            )
            
            # Training
            train_metrics = trainer.train(
                dataset["train"]["texts"],
                dataset["train"]["labels"],
                dataset["val"]["texts"],
                dataset["val"]["labels"]
            )
            
            # Evaluation
            test_metrics = trainer.evaluate(
                dataset["test"]["texts"],
                dataset["test"]["labels"]
            )
            
            accuracies.append(test_metrics["accuracy"])
            f1_scores.append(test_metrics["f1_weighted"])
            
            results["trials"].append({
                "accuracy": test_metrics["accuracy"],
                "f1": test_metrics["f1_weighted"],
                "metrics": test_metrics
            })
        
        results["mean_accuracy"] = np.mean(accuracies)
        results["std_accuracy"] = np.std(accuracies)
        results["mean_f1"] = np.mean(f1_scores)
        
        return results
    
    def _ablate_component(
        self,
        component: str,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train model with specific component ablated.
        
        Args:
            component: Component to ablate
            dataset: Dataset dictionary
            
        Returns:
            Ablation results
        """
        # Create ablated configuration
        ablated_config = self._create_ablated_config(component)
        
        results = {
            "component": component,
            "trials": [],
            "mean_accuracy": 0,
            "std_accuracy": 0,
            "mean_f1": 0,
            "performance_drop": 0
        }
        
        accuracies = []
        f1_scores = []
        
        for trial in range(self.num_trials):
            logger.info(f"Ablation trial {trial + 1}/{self.num_trials}")
            
            # Set seed
            set_seed(self.seed + trial)
            
            # Create model with ablated component
            model = self._create_model(ablated_config)
            
            # Train model
            trainer = BaseTrainer(
                model=model,
                config=ablated_config,
                device=self.device
            )
            
            # Training
            train_metrics = trainer.train(
                dataset["train"]["texts"],
                dataset["train"]["labels"],
                dataset["val"]["texts"],
                dataset["val"]["labels"]
            )
            
            # Evaluation
            test_metrics = trainer.evaluate(
                dataset["test"]["texts"],
                dataset["test"]["labels"]
            )
            
            accuracies.append(test_metrics["accuracy"])
            f1_scores.append(test_metrics["f1_weighted"])
            
            results["trials"].append({
                "accuracy": test_metrics["accuracy"],
                "f1": test_metrics["f1_weighted"],
                "metrics": test_metrics
            })
        
        results["mean_accuracy"] = np.mean(accuracies)
        results["std_accuracy"] = np.std(accuracies)
        results["mean_f1"] = np.mean(f1_scores)
        results["performance_drop"] = self.results["baseline"]["mean_accuracy"] - results["mean_accuracy"]
        
        return results
    
    def _create_ablated_config(self, component: str) -> Dict[str, Any]:
        """
        Create configuration with component ablated.
        
        Args:
            component: Component to ablate
            
        Returns:
            Ablated configuration
        """
        config = self.base_config.copy()
        
        # Define ablation strategies for each component
        ablation_strategies = {
            "attention": {"use_attention": False},
            "layer_norm": {"use_layer_norm": False},
            "residual_connections": {"use_residual": False},
            "position_embeddings": {"use_position_embeddings": False},
            "dropout": {"dropout": 0.0},
            "warmup": {"warmup_ratio": 0.0},
            "weight_decay": {"weight_decay": 0.0},
            "gradient_clipping": {"gradient_clipping": None},
            "data_augmentation": {"use_data_augmentation": False},
            "multi_layer_classifier": {"classifier_layers": 1},
            "advanced_pooling": {"pooling_strategy": "cls"},
            "scheduler": {"scheduler": None},
            "label_smoothing": {"label_smoothing": 0.0},
            "mixup": {"mixup_alpha": 0.0}
        }
        
        if component in ablation_strategies:
            config.update(ablation_strategies[component])
        
        return config
    
    def _calculate_importance(
        self,
        baseline_score: float,
        ablated_score: float
    ) -> float:
        """
        Calculate component importance score.
        
        Args:
            baseline_score: Baseline performance
            ablated_score: Performance with component ablated
            
        Returns:
            Importance score
        """
        # Relative performance drop
        if baseline_score > 0:
            importance = (baseline_score - ablated_score) / baseline_score
        else:
            importance = 0.0
        
        return importance
    
    def run_interaction_analysis(
        self,
        component_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Analyze interactions between component pairs.
        
        Args:
            component_pairs: List of component pairs to analyze
            
        Returns:
            Interaction analysis results
        """
        logger.info("Running component interaction analysis")
        
        if component_pairs is None:
            # Generate all pairs
            component_pairs = list(itertools.combinations(self.components_to_ablate[:5], 2))
        
        interaction_results = {}
        
        for comp1, comp2 in component_pairs:
            logger.info(f"Analyzing interaction: {comp1} x {comp2}")
            
            # Ablate both components
            double_ablated_config = self.base_config.copy()
            double_ablated_config.update(self._create_ablated_config(comp1))
            double_ablated_config.update(self._create_ablated_config(comp2))
            
            # Train and evaluate
            dataset = self._load_dataset()
            model = self._create_model(double_ablated_config)
            
            trainer = BaseTrainer(
                model=model,
                config=double_ablated_config,
                device=self.device
            )
            
            trainer.train(
                dataset["train"]["texts"][:1000],  # Use subset for efficiency
                dataset["train"]["labels"][:1000],
                dataset["val"]["texts"][:200],
                dataset["val"]["labels"][:200]
            )
            
            test_metrics = trainer.evaluate(
                dataset["test"]["texts"][:500],
                dataset["test"]["labels"][:500]
            )
            
            # Calculate interaction effect
            single_effect_1 = self.results["ablations"].get(comp1, {}).get("performance_drop", 0)
            single_effect_2 = self.results["ablations"].get(comp2, {}).get("performance_drop", 0)
            double_effect = self.results["baseline"]["mean_accuracy"] - test_metrics["accuracy"]
            
            interaction_effect = double_effect - (single_effect_1 + single_effect_2)
            
            interaction_results[f"{comp1}_{comp2}"] = {
                "component_1": comp1,
                "component_2": comp2,
                "single_effect_1": single_effect_1,
                "single_effect_2": single_effect_2,
                "double_effect": double_effect,
                "interaction_effect": interaction_effect,
                "synergistic": interaction_effect < 0  # Negative means components help each other
            }
        
        return interaction_results
    
    def _generate_summary(self) -> Dict[str, Any]:
        """
        Generate summary of ablation results.
        
        Returns:
            Summary dictionary
        """
        # Sort components by importance
        sorted_components = sorted(
            self.results["importance_scores"].items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Categorize components
        critical_components = []
        important_components = []
        moderate_components = []
        negligible_components = []
        
        for component, importance in sorted_components:
            if importance > 0.1:
                critical_components.append(component)
            elif importance > 0.05:
                important_components.append(component)
            elif importance > 0.01:
                moderate_components.append(component)
            else:
                negligible_components.append(component)
        
        summary = {
            "baseline_accuracy": self.results["baseline"]["mean_accuracy"],
            "num_components_tested": len(self.components_to_ablate),
            "most_important_component": sorted_components[0] if sorted_components else None,
            "least_important_component": sorted_components[-1] if sorted_components else None,
            "critical_components": critical_components,
            "important_components": important_components,
            "moderate_components": moderate_components,
            "negligible_components": negligible_components,
            "average_importance": np.mean(list(self.results["importance_scores"].values())),
            "total_trials": self.num_trials * (len(self.components_to_ablate) + 1)
        }
        
        return summary
    
    def _generate_visualizations(self):
        """Generate visualizations of ablation results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Component importance bar plot
        ax = axes[0, 0]
        components = list(self.results["importance_scores"].keys())
        importances = list(self.results["importance_scores"].values())
        
        colors = ['red' if imp > 0.05 else 'orange' if imp > 0.01 else 'gray' 
                  for imp in importances]
        
        bars = ax.barh(range(len(components)), importances, color=colors)
        ax.set_yticks(range(len(components)))
        ax.set_yticklabels(components)
        ax.set_xlabel('Importance Score')
        ax.set_title('Component Importance')
        ax.axvline(x=0.05, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
        ax.axvline(x=0.01, color='orange', linestyle='--', alpha=0.5, label='Important threshold')
        ax.legend()
        
        # 2. Performance comparison
        ax = axes[0, 1]
        baseline_acc = self.results["baseline"]["mean_accuracy"]
        
        ablated_accs = [
            self.results["ablations"][comp]["mean_accuracy"]
            for comp in components
        ]
        
        x = range(len(components) + 1)
        accuracies = [baseline_acc] + ablated_accs
        labels = ['Baseline'] + components
        
        ax.bar(x, accuracies, color=['green'] + colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Performance with Component Ablation')
        ax.axhline(y=baseline_acc, color='green', linestyle='--', alpha=0.5)
        
        # 3. Performance drop scatter plot
        ax = axes[1, 0]
        drops = [self.results["ablations"][comp]["performance_drop"] 
                for comp in components]
        stds = [self.results["ablations"][comp]["std_accuracy"] 
               for comp in components]
        
        scatter = ax.scatter(drops, range(len(components)), 
                           s=[std*1000 for std in stds],
                           c=importances, cmap='RdYlGn_r', alpha=0.6)
        
        ax.set_yticks(range(len(components)))
        ax.set_yticklabels(components)
        ax.set_xlabel('Performance Drop')
        ax.set_title('Performance Drop per Component')
        plt.colorbar(scatter, ax=ax, label='Importance')
        
        # 4. Component category pie chart
        ax = axes[1, 1]
        summary = self.results["summary"]
        categories = ['Critical', 'Important', 'Moderate', 'Negligible']
        sizes = [
            len(summary["critical_components"]),
            len(summary["important_components"]),
            len(summary["moderate_components"]),
            len(summary["negligible_components"])
        ]
        colors_pie = ['red', 'orange', 'yellow', 'gray']
        
        wedges, texts, autotexts = ax.pie(sizes, labels=categories, colors=colors_pie,
                                          autopct='%1.1f%%', startangle=90)
        ax.set_title('Component Distribution by Importance')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / "component_ablation_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {plot_path}")
        plt.show()
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset for ablation study."""
        dataset = AGNewsDataset()
        return dataset.load_splits()
    
    def _create_model(self, config: Dict[str, Any]):
        """Create model with given configuration."""
        return self.factory.create_model(
            self.base_model_name,
            **config
        )
    
    def _save_results(self):
        """Save ablation results."""
        results_path = self.output_dir / "component_ablation_results.json"
        
        # Convert numpy types for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
        
        # Also save as CSV for easy analysis
        df_data = []
        for component in self.components_to_ablate:
            if component in self.results["ablations"]:
                ablation = self.results["ablations"][component]
                df_data.append({
                    "component": component,
                    "baseline_acc": self.results["baseline"]["mean_accuracy"],
                    "ablated_acc": ablation["mean_accuracy"],
                    "std_acc": ablation["std_accuracy"],
                    "performance_drop": ablation["performance_drop"],
                    "importance_score": self.results["importance_scores"][component]
                })
        
        df = pd.DataFrame(df_data)
        csv_path = self.output_dir / "component_ablation_summary.csv"
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


def run_component_ablation():
    """Run component ablation study."""
    logger.info("Starting component ablation study")
    
    ablation = ComponentAblation(
        base_model_name="bert-base",
        num_trials=3
    )
    
    results = ablation.run_ablation_study()
    
    # Run interaction analysis for top components
    if results["summary"]["critical_components"]:
        interaction_results = ablation.run_interaction_analysis(
            component_pairs=list(itertools.combinations(
                results["summary"]["critical_components"][:3], 2
            ))
        )
        
        logger.info(f"Interaction analysis: {interaction_results}")
    
    return results


if __name__ == "__main__":
    run_component_ablation()
