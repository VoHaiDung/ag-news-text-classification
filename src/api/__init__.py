"""
API Module for AG News Text Classification
================================================================================
This module provides API interfaces for the text classification system,
supporting multiple protocols (REST, gRPC, GraphQL) and service architectures.

The API layer implements production-grade features including authentication,
rate limiting, monitoring, and service orchestration following microservices
patterns and RESTful principles.

References:
    - Fielding, R. T. (2000). Architectural Styles and the Design of Network-based Software Architectures
    - Newman, S. (2015). Building Microservices: Designing Fine-Grained Systems
    - Richardson, C. (2018). Microservices Patterns

Author: Võ Hải Dũng
License: MIT
"""

from typing import Dict, Any

# Version information
__version__ = "1.0.0"
__api_version__ = "v1"

# API configuration defaults
DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "timeout": 30,
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "rate_limit": {
        "requests_per_minute": 60,
        "requests_per_hour": 1000
    }
}

# Export main components
__all__ = [
    "__version__",
    "__api_version__",
    "DEFAULT_CONFIG"
]
