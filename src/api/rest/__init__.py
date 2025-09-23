"""
REST API Module
================================================================================
FastAPI-based REST API implementation for AG News Text Classification.

This module provides RESTful endpoints for classification, training,
model management, and system monitoring following OpenAPI 3.0 specification.

References:
    - Fielding, R. T. (2000). Architectural Styles and REST
    - OpenAPI Initiative (2021). OpenAPI Specification v3.1.0
    - FastAPI Documentation

Author: AG News Development Team
License: MIT
"""

from src.api.rest.app import app, create_app

# API metadata
API_TITLE = "AG News Text Classification REST API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
Production-grade REST API for AG News text classification providing:

## Features
- Single and batch text classification
- Multiple model support (DeBERTa, RoBERTa, XLNet, Ensemble)
- Real-time and asynchronous processing
- Model management and versioning
- Training and fine-tuning capabilities
- Comprehensive metrics and monitoring

## Authentication
Supports JWT tokens and API keys for secure access.

## Rate Limiting
Default limits: 60 requests/minute, 1000 requests/hour
"""

# Export main components
__all__ = [
    "app",
    "create_app",
    "API_TITLE",
    "API_VERSION",
    "API_DESCRIPTION"
]
