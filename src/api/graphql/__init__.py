"""
GraphQL API Module
================================================================================
This module provides GraphQL API interface for the AG News classification system,
implementing the GraphQL specification with support for queries, mutations, and
subscriptions.

The GraphQL implementation offers:
- Flexible query language for precise data fetching
- Real-time subscriptions for live updates
- Type-safe schema definition
- Efficient data loading with batching

References:
    - GraphQL Specification (June 2018 Edition)
    - Principled GraphQL Best Practices
    - Lee, B. (2015). GraphQL: A data query language

Author: Võ Hải Dũng
License: MIT
"""

from typing import Dict, Any

# Version information
__version__ = "1.0.0"
__graphql_version__ = "16.6.0"

# GraphQL configuration defaults
DEFAULT_CONFIG = {
    "host": "0.0.0.0",
    "port": 8080,
    "path": "/graphql",
    "playground_enabled": True,
    "introspection_enabled": True,
    "max_query_depth": 10,
    "max_query_complexity": 1000,
    "subscription": {
        "enabled": True,
        "keepalive": 30,
        "max_connections": 100
    },
    "batching": {
        "enabled": True,
        "max_batch_size": 10
    },
    "caching": {
        "enabled": True,
        "ttl": 300  # 5 minutes
    }
}

# Schema type definitions
SCHEMA_TYPES = {
    "Query": "Root query type",
    "Mutation": "Root mutation type",
    "Subscription": "Root subscription type",
    "Classification": "Text classification type",
    "Model": "Machine learning model type",
    "Dataset": "Training dataset type",
    "Training": "Training job type",
    "Metrics": "Performance metrics type"
}

# Export configuration
__all__ = [
    "__version__",
    "__graphql_version__",
    "DEFAULT_CONFIG",
    "SCHEMA_TYPES"
]
