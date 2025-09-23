"""
Middleware Components for REST API
================================================================================
Collection of middleware for request/response processing, security,
monitoring, and logging in the FastAPI application.

This module implements the Chain of Responsibility pattern for
request processing pipeline.

References:
    - FastAPI Middleware Documentation
    - ASGI Specification
    - Gamma, E., et al. (1994). Design Patterns: Chain of Responsibility

Author: Võ Hải Dũng
License: MIT
"""

from src.api.rest.middleware.logging_middleware import LoggingMiddleware
from src.api.rest.middleware.metrics_middleware import MetricsMiddleware
from src.api.rest.middleware.security_middleware import SecurityMiddleware

__all__ = [
    "LoggingMiddleware",
    "MetricsMiddleware",
    "SecurityMiddleware"
]
