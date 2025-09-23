"""
Base API Components
================================================================================
Foundation classes and utilities for API implementation providing common
functionality across different API protocols.

This module implements core patterns for authentication, authorization,
rate limiting, and request handling following security best practices
and design patterns.

References:
    - Gamma, E., et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software
    - OWASP (2021). API Security Top 10

Author: Võ Hải Dũng
License: MIT
"""

from src.api.base.base_handler import (
    BaseHandler,
    BatchHandler,
    StreamingHandler,
    APIContext,
    APIResponse,
    ResponseStatus
)
from src.api.base.auth import (
    AuthenticationManager,
    JWTAuthenticator,
    APIKeyAuthenticator,
    RBACAuthorizer,
    AuthToken,
    AuthType,
    Role
)
from src.api.base.rate_limiter import (
    RateLimiter,
    RateLimitManager,
    RateLimitConfig,
    RateLimitStrategy,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    AdaptiveRateLimiter
)
from src.api.base.error_handler import (
    ErrorHandler,
    APIError,
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    NotFoundError,
    ConflictError,
    ServiceUnavailableError
)
from src.api.base.request_validator import (
    RequestValidator,
    ValidationSchema,
    FieldValidator,
    validate_request,
    validate_field
)
from src.api.base.cors_handler import (
    CORSHandler,
    CORSConfig
)

__all__ = [
    # Base handlers
    "BaseHandler",
    "BatchHandler", 
    "StreamingHandler",
    "APIContext",
    "APIResponse",
    "ResponseStatus",
    
    # Authentication
    "AuthenticationManager",
    "JWTAuthenticator",
    "APIKeyAuthenticator",
    "RBACAuthorizer",
    "AuthToken",
    "AuthType",
    "Role",
    
    # Rate limiting
    "RateLimiter",
    "RateLimitManager",
    "RateLimitConfig",
    "RateLimitStrategy",
    "TokenBucketRateLimiter",
    "SlidingWindowRateLimiter",
    "AdaptiveRateLimiter",
    
    # Error handling
    "ErrorHandler",
    "APIError",
    "ValidationError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "NotFoundError",
    "ConflictError",
    "ServiceUnavailableError",
    
    # Request validation
    "RequestValidator",
    "ValidationSchema",
    "FieldValidator",
    "validate_request",
    "validate_field",
    
    # CORS
    "CORSHandler",
    "CORSConfig"
]
