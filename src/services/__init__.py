"""
Services Module for AG News Text Classification
================================================================================
This module implements the service layer following Domain-Driven Design (DDD)
and microservices architecture patterns, providing business logic and
orchestration for the classification system.

The service layer acts as an intermediary between the API layer and the
domain/data layers, encapsulating business rules and workflows.

References:
    - Evans, E. (2003). Domain-Driven Design: Tackling Complexity in the Heart of Software
    - Newman, S. (2015). Building Microservices: Designing Fine-Grained Systems
    - Richardson, C. (2018). Microservices Patterns

Author: Võ Hải Dũng
License: MIT
"""

from src.services.base_service import (
    BaseService,
    ServiceStatus,
    ServiceConfig,
    ServiceHealth
)
from src.services.service_registry import ServiceRegistry

__all__ = [
    "BaseService",
    "ServiceStatus",
    "ServiceConfig",
    "ServiceHealth",
    "ServiceRegistry"
]

# Service layer version
__version__ = "1.0.0"
