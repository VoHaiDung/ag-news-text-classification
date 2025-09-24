"""
REST API Routers Module
================================================================================
This module aggregates all API routers for the AG News classification system.
Implements modular routing architecture following RESTful design principles
and separation of concerns.

The routing system provides:
- Modular endpoint organization
- Automatic API documentation generation
- Consistent URL patterns
- Version management support

References:
    - Masse, M. (2011). REST API Design Rulebook
    - Richardson, L., & Ruby, S. (2007). RESTful Web Services

Author: Võ Hải Dũng
License: MIT
"""

from typing import List, Tuple
from fastapi import APIRouter

# Import all routers
from .classification import router as classification_router
from .training import router as training_router
from .models import router as models_router
from .data import router as data_router
from .health import router as health_router
from .metrics import router as metrics_router
from .admin import router as admin_router

# Version information
__version__ = "1.0.0"

# Router configuration with prefixes and tags
ROUTERS: List[Tuple[APIRouter, dict]] = [
    (
        classification_router,
        {
            "prefix": "/classification",
            "tags": ["Classification"]
        }
    ),
    (
        training_router,
        {
            "prefix": "/training",
            "tags": ["Training"]
        }
    ),
    (
        models_router,
        {
            "prefix": "/models",
            "tags": ["Models"]
        }
    ),
    (
        data_router,
        {
            "prefix": "/data",
            "tags": ["Data"]
        }
    ),
    (
        health_router,
        {
            "prefix": "/health",
            "tags": ["Health"]
        }
    ),
    (
        metrics_router,
        {
            "prefix": "/metrics",
            "tags": ["Metrics"]
        }
    ),
    (
        admin_router,
        {
            "prefix": "/admin",
            "tags": ["Admin"]
        }
    )
]

def get_all_routers() -> List[Tuple[APIRouter, dict]]:
    """
    Get all configured routers with their settings.
    
    Returns:
        List[Tuple[APIRouter, dict]]: List of router instances with configuration
    """
    return ROUTERS

def get_router_by_tag(tag: str) -> APIRouter:
    """
    Get router by its tag name.
    
    Args:
        tag: Router tag name
        
    Returns:
        APIRouter: Router instance or None if not found
    """
    for router, config in ROUTERS:
        if tag in config.get("tags", []):
            return router
    return None

# Export all routers
__all__ = [
    "classification_router",
    "training_router",
    "models_router",
    "data_router",
    "health_router",
    "metrics_router",
    "admin_router",
    "ROUTERS",
    "get_all_routers",
    "get_router_by_tag"
]
