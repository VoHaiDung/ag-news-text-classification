"""
API Module
==========

Provides RESTful, gRPC, and GraphQL API interfaces for the AG News framework,
implementing API design patterns from:
- Fielding (2000): "Architectural Styles and the Design of Network-based Software Architectures"
- Richardson & Ruby (2007): "RESTful Web Services"
- Kleppmann (2017): "Designing Data-Intensive Applications"

Author: Team SOTA AGNews
License: MIT
"""

from typing import Dict, Any

# API Version following Semantic Versioning (Fowler, 2004)
API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"

# API Configuration
API_CONFIG = {
    "rest": {
        "enabled": True,
        "port": 8000,
        "host": "0.0.0.0",
        "workers": 4,
        "timeout": 60,
        "max_request_size": 10 * 1024 * 1024  # 10MB
    },
    "grpc": {
        "enabled": False,
        "port": 50051,
        "max_message_size": 50 * 1024 * 1024  # 50MB
    },
    "graphql": {
        "enabled": False,
        "port": 8001,
        "playground": True
    },
    "security": {
        "enable_auth": True,
        "enable_cors": True,
        "rate_limit": 100,  # requests per minute
        "api_key_header": "X-API-Key"
    },
    "monitoring": {
        "enable_metrics": True,
        "enable_tracing": True,
        "enable_logging": True
    }
}

def get_api_info() -> Dict[str, Any]:
    """
    Get API information.
    
    Returns:
        API metadata dictionary
    """
    return {
        "name": "AG News Classification API",
        "version": API_VERSION,
        "description": "State-of-the-art text classification API for AG News dataset",
        "endpoints": {
            "rest": f"{API_PREFIX}",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

__all__ = [
    "API_VERSION",
    "API_PREFIX",
    "API_CONFIG",
    "get_api_info"
]
