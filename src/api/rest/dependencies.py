"""
API Dependencies
================================================================================
This module defines FastAPI dependency injection functions for common
functionality across endpoints, implementing the Dependency Injection pattern
for clean and testable code.

Provides dependencies for:
- Authentication and authorization
- Request validation and parsing
- Service injection
- Configuration management
- Logging and monitoring

References:
    - Fowler, M. (2004). Inversion of Control Containers and the Dependency Injection pattern
    - FastAPI Documentation: Dependencies
    - Clean Architecture (Robert C. Martin, 2017)

Author: Võ Hải Dũng
License: MIT
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from functools import lru_cache
import uuid
import logging
from fastapi import Depends, HTTPException, status, Header, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext

from ..base.auth import verify_token, get_user_permissions
from ..base.rate_limiter import RateLimiter
from ...services.core.prediction_service import PredictionService
from ...services.core.training_service import TrainingService
from ...services.core.data_service import DataService
from ...services.core.model_management_service import ModelManagementService
from ...core.registry import Registry
from configs.config_loader import ConfigLoader

# Security configuration
SECRET_KEY = "your-secret-key-here"  # Should be loaded from environment
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize components
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = HTTPBearer()
rate_limiter = RateLimiter()
logger = logging.getLogger(__name__)

class DependencyProvider:
    """
    Centralized dependency provider for service injection.
    
    Attributes:
        _instances: Cached service instances
        _config: Configuration loader
    """
    
    def __init__(self):
        """Initialize dependency provider."""
        self._instances = {}
        self._config = None
        
    @lru_cache()
    def get_config(self) -> ConfigLoader:
        """
        Get configuration loader instance.
        
        Returns:
            ConfigLoader: Configuration loader
        """
        if self._config is None:
            self._config = ConfigLoader()
        return self._config
    
    def get_service(self, service_class: type) -> Any:
        """
        Get or create service instance.
        
        Args:
            service_class: Service class type
            
        Returns:
            Any: Service instance
        """
        class_name = service_class.__name__
        if class_name not in self._instances:
            self._instances[class_name] = service_class()
        return self._instances[class_name]

# Initialize provider
dependency_provider = DependencyProvider()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(oauth2_scheme)
) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP authorization credentials
        
    Returns:
        Dict[str, Any]: User information
        
    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        
        # Get user information (placeholder implementation)
        user = {
            "username": username,
            "user_id": payload.get("user_id"),
            "role": payload.get("role", "user"),
            "permissions": payload.get("permissions", [])
        }
        
        return user
    except JWTError:
        raise credentials_exception

async def get_current_admin(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Verify current user has admin privileges.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Dict[str, Any]: Admin user information
        
    Raises:
        HTTPException: If user is not admin
    """
    if current_user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return current_user

def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme)
) -> Optional[Dict[str, Any]]:
    """
    Get optional authenticated user (for public endpoints).
    
    Args:
        credentials: Optional HTTP authorization credentials
        
    Returns:
        Optional[Dict[str, Any]]: User information or None
    """
    if credentials is None:
        return None
    
    try:
        return get_current_user(credentials)
    except HTTPException:
        return None

async def verify_api_key(
    x_api_key: Optional[str] = Header(None)
) -> bool:
    """
    Verify API key for service-to-service authentication.
    
    Args:
        x_api_key: API key from header
        
    Returns:
        bool: Verification result
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
    
    # Verify API key (placeholder implementation)
    valid_api_keys = ["test-api-key"]  # Should be loaded from secure storage
    
    if x_api_key not in valid_api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    return True

async def check_rate_limit(
    request: Request,
    user: Optional[Dict] = Depends(get_optional_user)
) -> None:
    """
    Check rate limit for current request.
    
    Args:
        request: FastAPI request
        user: Optional authenticated user
        
    Raises:
        HTTPException: If rate limit exceeded
    """
    # Get identifier for rate limiting
    if user:
        identifier = f"user:{user['user_id']}"
    else:
        identifier = f"ip:{request.client.host}"
    
    # Check rate limit
    allowed = await rate_limiter.check_limit(
        identifier=identifier,
        limit=100,  # 100 requests
        window=60   # per minute
    )
    
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"}
        )

def get_request_id() -> str:
    """
    Generate unique request ID for tracking.
    
    Returns:
        str: Request ID
    """
    return str(uuid.uuid4())

def get_prediction_service() -> PredictionService:
    """
    Get prediction service instance.
    
    Returns:
        PredictionService: Prediction service
    """
    return dependency_provider.get_service(PredictionService)

def get_training_service() -> TrainingService:
    """
    Get training service instance.
    
    Returns:
        TrainingService: Training service
    """
    return dependency_provider.get_service(TrainingService)

def get_data_service() -> DataService:
    """
    Get data service instance.
    
    Returns:
        DataService: Data service
    """
    return dependency_provider.get_service(DataService)

def get_model_service() -> ModelManagementService:
    """
    Get model management service instance.
    
    Returns:
        ModelManagementService: Model management service
    """
    return dependency_provider.get_service(ModelManagementService)

def get_config_manager() -> ConfigLoader:
    """
    Get configuration manager.
    
    Returns:
        ConfigLoader: Configuration manager
    """
    return dependency_provider.get_config()

def get_service_status() -> Dict[str, Any]:
    """
    Get current service status.
    
    Returns:
        Dict[str, Any]: Service status information
    """
    return {
        "prediction_service": "healthy",
        "training_service": "healthy",
        "data_service": "healthy",
        "model_service": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

async def validate_model_access(
    model_id: str,
    user: Dict = Depends(get_current_user)
) -> bool:
    """
    Validate user access to specific model.
    
    Args:
        model_id: Model identifier
        user: Current user
        
    Returns:
        bool: Access allowed
        
    Raises:
        HTTPException: If access denied
    """
    # Check model access (placeholder implementation)
    user_models = user.get("accessible_models", [])
    
    if model_id not in user_models and user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to model: {model_id}"
        )
    
    return True

async def validate_dataset_access(
    dataset_id: str,
    user: Dict = Depends(get_current_user)
) -> bool:
    """
    Validate user access to specific dataset.
    
    Args:
        dataset_id: Dataset identifier
        user: Current user
        
    Returns:
        bool: Access allowed
        
    Raises:
        HTTPException: If access denied
    """
    # Check dataset access (placeholder implementation)
    user_datasets = user.get("accessible_datasets", [])
    
    if dataset_id not in user_datasets and user.get("role") != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Access denied to dataset: {dataset_id}"
        )
    
    return True

class PaginationDep:
    """Pagination dependency class."""
    
    def __init__(
        self,
        page: int = 1,
        per_page: int = 20,
        max_per_page: int = 100
    ):
        """
        Initialize pagination dependency.
        
        Args:
            page: Page number
            per_page: Items per page
            max_per_page: Maximum items per page
        """
        self.page = max(1, page)
        self.per_page = min(max(1, per_page), max_per_page)
        self.offset = (self.page - 1) * self.per_page
        
    def paginate(self, total: int) -> Dict[str, Any]:
        """
        Calculate pagination metadata.
        
        Args:
            total: Total number of items
            
        Returns:
            Dict[str, Any]: Pagination metadata
        """
        total_pages = (total + self.per_page - 1) // self.per_page
        
        return {
            "page": self.page,
            "per_page": self.per_page,
            "total": total,
            "pages": total_pages,
            "has_next": self.page < total_pages,
            "has_prev": self.page > 1
        }

def get_logger(name: str = __name__) -> logging.Logger:
    """
    Get logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        logging.Logger: Logger instance
    """
    return logging.getLogger(name)

# Export dependencies
__all__ = [
    "get_current_user",
    "get_current_admin",
    "get_optional_user",
    "verify_api_key",
    "check_rate_limit",
    "get_request_id",
    "get_prediction_service",
    "get_training_service",
    "get_data_service",
    "get_model_service",
    "get_config_manager",
    "get_service_status",
    "validate_model_access",
    "validate_dataset_access",
    "PaginationDep",
    "get_logger",
    "dependency_provider"
]
