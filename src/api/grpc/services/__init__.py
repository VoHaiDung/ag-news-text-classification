"""
gRPC Services Module
================================================================================
This module contains gRPC service implementations for the AG News classification
system, following Google's API design guidelines and Protocol Buffers best practices.

Service architecture implements:
- Unary and streaming RPC methods
- Error handling and status codes
- Request validation and response formatting
- Service-level interceptors

References:
    - Google API Design Guide
    - gRPC Best Practices
    - Protocol Buffers Style Guide

Author: Võ Hải Dũng
License: MIT
"""

from typing import Dict, Type, Any
import grpc

# Service registry
SERVICE_REGISTRY: Dict[str, Type] = {}

def register_service(name: str, service_class: Type) -> None:
    """
    Register a gRPC service in the registry.
    
    Args:
        name: Service name
        service_class: Service class type
    """
    SERVICE_REGISTRY[name] = service_class

def get_service(name: str) -> Type:
    """
    Get service class from registry.
    
    Args:
        name: Service name
        
    Returns:
        Type: Service class
        
    Raises:
        KeyError: If service not found
    """
    if name not in SERVICE_REGISTRY:
        raise KeyError(f"Service '{name}' not found in registry")
    return SERVICE_REGISTRY[name]

def list_services() -> list:
    """
    List all registered services.
    
    Returns:
        list: Service names
    """
    return list(SERVICE_REGISTRY.keys())

# Base service class
class BaseGRPCService:
    """
    Base class for gRPC services.
    
    Provides common functionality for all gRPC services including
    logging, error handling, and metrics collection.
    """
    
    def __init__(self, service_name: str):
        """
        Initialize base service.
        
        Args:
            service_name: Name of the service
        """
        self.service_name = service_name
        self.metrics = {
            "requests_total": 0,
            "requests_failed": 0,
            "requests_succeeded": 0
        }
    
    def register(self, server: grpc.Server) -> None:
        """
        Register service with gRPC server.
        
        Args:
            server: gRPC server instance
        """
        raise NotImplementedError("Subclasses must implement register method")
    
    def handle_error(self, context: grpc.ServicerContext, error: Exception) -> None:
        """
        Handle service error.
        
        Args:
            context: gRPC context
            error: Exception that occurred
        """
        self.metrics["requests_failed"] += 1
        
        # Set appropriate status code
        if isinstance(error, ValueError):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
        elif isinstance(error, PermissionError):
            context.set_code(grpc.StatusCode.PERMISSION_DENIED)
        elif isinstance(error, FileNotFoundError):
            context.set_code(grpc.StatusCode.NOT_FOUND)
        else:
            context.set_code(grpc.StatusCode.INTERNAL)
        
        context.set_details(str(error))
    
    def validate_request(self, request: Any, required_fields: list) -> bool:
        """
        Validate request has required fields.
        
        Args:
            request: Request message
            required_fields: List of required field names
            
        Returns:
            bool: Validation result
        """
        for field in required_fields:
            if not hasattr(request, field) or not getattr(request, field):
                return False
        return True
    
    def increment_success_metric(self) -> None:
        """Increment successful request counter."""
        self.metrics["requests_total"] += 1
        self.metrics["requests_succeeded"] += 1
    
    def increment_failure_metric(self) -> None:
        """Increment failed request counter."""
        self.metrics["requests_total"] += 1
        self.metrics["requests_failed"] += 1
    
    def get_metrics(self) -> dict:
        """
        Get service metrics.
        
        Returns:
            dict: Service metrics
        """
        return {
            f"{self.service_name}_metrics": self.metrics
        }
    
    async def cleanup(self) -> None:
        """Cleanup service resources."""
        pass

# Export base class and utilities
__all__ = [
    "BaseGRPCService",
    "register_service",
    "get_service",
    "list_services",
    "SERVICE_REGISTRY"
]
