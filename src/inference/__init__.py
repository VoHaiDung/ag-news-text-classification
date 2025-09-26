"""
Inference Module for AG News Text Classification System
================================================================================
This module provides optimized inference capabilities for production deployment,
including model optimization, batching strategies, and serving infrastructure.

The inference module implements best practices for ML model serving including
latency optimization, throughput maximization, and resource efficiency.

References:
    - Crankshaw, D., et al. (2017). Clipper: A Low-Latency Online Prediction Serving System
    - Olston, C., et al. (2017). TensorFlow-Serving: Flexible, High-Performance ML Serving
    - ONNX Runtime Documentation: https://onnxruntime.ai/

Author: Võ Hải Dũng
License: MIT
"""

from typing import List, Dict, Any, Optional

# Version information
__version__ = "1.0.0"

# Core inference components
from .predictors import (
    SinglePredictor,
    BatchPredictor,
    StreamingPredictor,
    EnsemblePredictor
)

from .optimization import (
    ModelOptimizer,
    ONNXConverter,
    TensorRTOptimizer,
    QuantizationOptimizer
)

from .serving import (
    ModelServer,
    BatchServer,
    LoadBalancer
)

# Default configuration
DEFAULT_CONFIG = {
    "batch_size": 32,
    "max_sequence_length": 512,
    "num_threads": 4,
    "use_gpu": True,
    "optimization_level": "O2",
    "dynamic_batching": True,
    "model_cache_size": 5
}

# Export main components
__all__ = [
    # Predictors
    "SinglePredictor",
    "BatchPredictor", 
    "StreamingPredictor",
    "EnsemblePredictor",
    
    # Optimization
    "ModelOptimizer",
    "ONNXConverter",
    "TensorRTOptimizer",
    "QuantizationOptimizer",
    
    # Serving
    "ModelServer",
    "BatchServer",
    "LoadBalancer",
    
    # Configuration
    "DEFAULT_CONFIG",
    "__version__"
]
