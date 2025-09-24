"""
GraphQL Context Management
================================================================================
This module manages the context for GraphQL operations, providing access to
services, authentication, and request-scoped data.

The context implements:
- Service injection
- Authentication state
- Request tracking
- Data loaders for N+1 prevention

References:
    - GraphQL Context Pattern
    - Dependency Injection Pattern
    - DataLoader Pattern (Facebook)

Author: Võ Hải Dũng
License: MIT
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from strawberry.fastapi import BaseContext
from fastapi import Request, Depends
import asyncio

from .dataloaders import (
    ModelLoader,
    DatasetLoader,
    TrainingJobLoader,
    UserLoader
)
from ...services.core.prediction_service import PredictionService
from ...services.core.training_service import TrainingService
from ...services.core.model_management_service import ModelManagementService
from ...services.core.data_service import DataService

logger = logging.getLogger(__name__)

@dataclass
class User:
    """User information from authentication."""
    id: str
    username: str
    role: str
    permissions: list

class GraphQLContext(BaseContext):
    """
    GraphQL execution context.
    
    Provides access to services, authentication state,
    and request-scoped resources for GraphQL operations.
    """
    
    def __init__(self):
        """Initialize GraphQL context."""
        super().__init__()
        
        # Services
        self.prediction_service = None
        self.training_service = None
        self.model_service = None
        self.data_service = None
        
        # Data loaders
        self.model_loader = None
        self.dataset_loader = None
        self.training_job_loader = None
        self.user_loader = None
        
        # Authentication
        self.user = None
        
        # Request tracking
        self.request_id = None
        
        # Caches
        self._cache = {}
        
    async def initialize(self, request: Request):
        """
        Initialize context with request data.
        
        Args:
            request: FastAPI request object
        """
        # Extract authentication
        self.user = await self._get_user_from_request(request)
        
        # Initialize services
        await self._initialize_services()
        
        # Initialize data loaders
        self._initialize_dataloaders()
        
        # Set request ID
        self.request_id = request.headers.get('x-request-id', 'unknown')
        
        logger.info(f"GraphQL context initialized for request {self.request_id}")
    
    async def _get_user_from_request(self, request: Request) -> Optional[User]:
        """
        Extract user information from request.
        
        Args:
            request: FastAPI request
            
        Returns:
            Optional[User]: User information if authenticated
        """
        # Get authorization header
        auth_header = request.headers.get('authorization', '')
        
        if not auth_header.startswith('Bearer '):
            return None
        
        token = auth_header[7:]
        
        try:
            # Validate token (placeholder implementation)
            # In production, decode JWT and verify
            user_data = {
                'id': 'user123',
                'username': 'test_user',
                'role': 'user',
                'permissions': ['read', 'predict']
            }
            
            return User(**user_data)
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return None
    
    async def _initialize_services(self):
        """Initialize service instances."""
        self.prediction_service = PredictionService()
        self.training_service = TrainingService()
        self.model_service = ModelManagementService()
        self.data_service = DataService()
        
        # Initialize services if needed
        await self.prediction_service.initialize()
        await self.training_service.initialize()
        await self.model_service.initialize()
        await self.data_service.initialize()
    
    def _initialize_dataloaders(self):
        """Initialize DataLoader instances."""
        self.model_loader = ModelLoader(self.model_service)
        self.dataset_loader = DatasetLoader(self.data_service)
        self.training_job_loader = TrainingJobLoader(self.training_service)
        self.user_loader = UserLoader()
    
    async def get_metrics(self, time_range: str = "1h") -> Dict[str, Any]:
        """
        Get system metrics for specified time range.
        
        Args:
            time_range: Time range for metrics
            
        Returns:
            Dict[str, Any]: Metrics data
        """
        metrics = {}
        
        # Collect metrics from services
        if self.prediction_service:
            metrics.update(await self.prediction_service.get_metrics(time_range))
        
        if self.model_service:
            metrics.update(await self.model_service.get_metrics(time_range))
        
        # Add system metrics
        import psutil
        metrics.update({
            'cpu_usage_percent': psutil.cpu_percent(),
            'memory_usage_mb': psutil.virtual_memory().used / 1024 / 1024,
            'gpu_usage_percent': 0  # Placeholder
        })
        
        return metrics
    
    async def clear_all_caches(self):
        """Clear all caches."""
        self._cache.clear()
        
        # Clear service caches
        if self.prediction_service:
            await self.prediction_service.clear_cache()
        
        if self.model_service:
            await self.model_service.clear_cache()
        
        # Clear data loader caches
        if self.model_loader:
            self.model_loader.clear()
        
        if self.dataset_loader:
            self.dataset_loader.clear()
        
        logger.info("All caches cleared")
    
    async def clear_model_cache(self):
        """Clear model-related caches."""
        if self.model_service:
            await self.model_service.clear_cache()
        
        if self.model_loader:
            self.model_loader.clear()
        
        logger.info("Model cache cleared")
    
    async def clear_data_cache(self):
        """Clear data-related caches."""
        if self.data_service:
            await self.data_service.clear_cache()
        
        if self.dataset_loader:
            self.dataset_loader.clear()
        
        logger.info("Data cache cleared")
    
    def get_cache(self, key: str) -> Any:
        """
        Get value from context cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None
        """
        return self._cache.get(key)
    
    def set_cache(self, key: str, value: Any):
        """
        Set value in context cache.
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = value
    
    async def cleanup(self):
        """Cleanup context resources."""
        # Cleanup services
        if self.prediction_service:
            await self.prediction_service.cleanup()
        
        if self.training_service:
            await self.training_service.cleanup()
        
        if self.model_service:
            await self.model_service.cleanup()
        
        if self.data_service:
            await self.data_service.cleanup()
        
        logger.info(f"GraphQL context cleaned up for request {self.request_id}")

# Type alias for Strawberry
Info = strawberry.Info[GraphQLContext, None]

async def get_context(request: Request) -> GraphQLContext:
    """
    Create GraphQL context for request.
    
    Args:
        request: FastAPI request
        
    Returns:
        GraphQLContext: Initialized context
    """
    context = GraphQLContext()
    await context.initialize(request)
    return context

# Export context components
__all__ = [
    "GraphQLContext",
    "User",
    "Info",
    "get_context"
]
