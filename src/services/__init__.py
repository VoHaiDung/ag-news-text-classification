"""
Services Module
===============

Implements service layer architecture patterns from:
- Evans (2003): "Domain-Driven Design"
- Fowler (2002): "Patterns of Enterprise Application Architecture"
- Richardson (2018): "Microservices Patterns"

The service layer provides a high-level interface to domain logic,
encapsulating business operations and coordinating between different
components of the system.

Author: Võ Hải Dũng
License: MIT
"""

from typing import Dict, Any

# Service configuration
SERVICE_CONFIG = {
    "prediction": {
        "enabled": True,
        "cache_predictions": True,
        "cache_ttl": 3600,  # 1 hour
        "batch_size": 32,
        "max_concurrent_requests": 100
    },
    "training": {
        "enabled": True,
        "auto_checkpoint": True,
        "checkpoint_interval": 1000,  # steps
        "max_parallel_experiments": 4,
        "distributed_training": False
    },
    "data": {
        "enabled": True,
        "cache_datasets": True,
        "preprocessing_workers": 4,
        "augmentation_enabled": False
    },
    "model_management": {
        "enabled": True,
        "model_registry": "local",  # local/mlflow/wandb
        "version_control": True,
        "auto_optimization": False
    }
}

def get_service_info() -> Dict[str, Any]:
    """
    Get service layer information.
    
    Returns:
        Service metadata dictionary
    """
    return {
        "version": "1.0.0",
        "services": {
            "prediction": "Text classification predictions",
            "training": "Model training orchestration",
            "data": "Data management and processing",
            "model_management": "Model lifecycle management"
        },
        "config": SERVICE_CONFIG
    }

__all__ = [
    "SERVICE_CONFIG",
    "get_service_info"
]
