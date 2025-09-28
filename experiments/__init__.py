"""
Experiments Module for AG News Text Classification
================================================================================
This module provides comprehensive experimental frameworks for evaluating and
benchmarking text classification models, including baseline comparisons,
ablation studies, and state-of-the-art experiments.

The experimental pipeline implements rigorous evaluation protocols following
best practices in machine learning research, ensuring reproducibility and
statistical validity.

References:
    - Bouthillier, X., et al. (2019). Accounting for Variance in Machine Learning Benchmarks
    - Dodge, J., et al. (2019). Show Your Work: Improved Reporting of Experimental Results
    - Henderson, P., et al. (2018). Deep Reinforcement Learning that Matters

Author: Võ Hải Dũng
License: MIT
"""

from typing import Dict, Any, List

# Version information
__version__ = "1.0.0"

# Experiment configuration defaults
DEFAULT_EXPERIMENT_CONFIG = {
    "seed": 42,
    "num_runs": 5,
    "cross_validation_folds": 5,
    "test_size": 0.2,
    "validation_size": 0.1,
    "metrics": ["accuracy", "f1_macro", "precision_macro", "recall_macro"],
    "save_results": True,
    "results_dir": "outputs/results/experiments",
    "enable_tracking": True,
    "tracking_backend": "wandb"
}

# Statistical testing configuration
STATISTICAL_CONFIG = {
    "significance_level": 0.05,
    "bootstrap_iterations": 1000,
    "confidence_interval": 0.95,
    "effect_size_threshold": 0.2,
    "multiple_comparison_correction": "bonferroni"
}

# Benchmark thresholds
BENCHMARK_THRESHOLDS = {
    "accuracy": {
        "sota": 0.95,
        "good": 0.92,
        "acceptable": 0.90
    },
    "inference_speed": {
        "fast": 10,  # ms per sample
        "medium": 50,
        "slow": 100
    },
    "memory_usage": {
        "efficient": 1024,  # MB
        "moderate": 2048,
        "heavy": 4096
    }
}

# Export main components
__all__ = [
    "__version__",
    "DEFAULT_EXPERIMENT_CONFIG",
    "STATISTICAL_CONFIG",
    "BENCHMARK_THRESHOLDS"
]
