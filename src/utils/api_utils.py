"""
API Utilities for AG News Text Classification System
================================================================================
This module provides utility functions and classes for API operations including
request/response handling, authentication, rate limiting, and API documentation.

The utilities implement best practices for RESTful APIs, gRPC services, and
GraphQL endpoints as outlined in industry standards and academic literature.

References:
    - Fielding, R. T. (2000). Architectural Styles and the Design of Network-based Software Architectures
    - Allamaraju, S. (2010). RESTful Web Services Cookbook
    - Google API Design Guide: https://cloud.google.com/apis/design

Author: Võ Hải Dũng
License: MIT
"""

import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass, asdict
import jwt
import aiohttp
from fastapi import HTTPException, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, ValidationError
import grpc
from prometheus_client import Counter, Histogram, Gauge
import redis
from cryptography.fernet import Fernet
import logging

from ..core.exceptions import AuthenticationError, ValidationError as CustomValidationError

# Configure logging
logger = logging.getLogger(__name__)

# Metrics
api_request_counter = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['method', 'endpoint', 'status']
)
api_request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)
api_rate_limit_exceeded = Counter(
    'api_rate_limit_exceeded_total',
    'Number of rate limit exceeded events',
    ['client_id']
)


@dataclass
class APIResponse:
    """
    Standardized API response structure
    
    Attributes:
        success: Whether the request was successful
        data: Response data
        message: Response message
        error: Error information if any
        metadata: Additional metadata
    """
    success: bool
    data: Optional[Any] = None
    message: Optional[str] = None
    error: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "success": self.success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.data is not None:
            result["data"] = self.data
        if self.message:
            result["message"] = self.message
        if self.error:
            result["error"] = self.error
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result


class RequestValidator:
    """
    Request validation utilities
    
    Provides methods for validating API requests including
    schema validation, content type checking, and input sanitization.
    """
    
    @staticmethod
    def validate_content_type(request: Request, 
                            expected: str = "application/json") -> bool:
        """
        Validate request content type
        
        Args:
            request: FastAPI request object
            expected: Expected content type
            
        Returns:
            Validation result
        """
        content_type = request.headers.get("content-type", "")
        return content_type.startswith(expected)
    
    @staticmethod
    def validate_json_schema(data: Dict[str, Any], 
                           schema: type[BaseModel]) -> Tuple[bool, Optional[str]]:
        """
        Validate JSON data against Pydantic schema
        
        Args:
            data: Data to validate
            schema: Pydantic model schema
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            schema(**data)
            return True, None
        except ValidationError as e:
            return False, str(e)
    
    @staticmethod
    def sanitize_input(text: str, 
                      max_length: int = 10000,
                      allowed_chars: Optional[str] = None) -> str:
        """
        Sanitize text input
        
        Args:
            text: Input text
            max_length: Maximum allowed length
            allowed_chars: Allowed character set
            
        Returns:
            Sanitized text
        """
        # Truncate to max length
        text = text[:max_length]
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char == '\n')
        
        # Apply character whitelist if provided
        if allowed_chars:
            text = ''.join(char for char in text if char in allowed_chars)
        
        return text.strip()
    
    @staticmethod
    def validate_pagination(page: int = 1, 
                          page_size: int = 10,
                          max_page_size: int = 100) -> Tuple[int, int]:
        """
        Validate pagination parameters
        
        Args:
            page: Page number
            page_size: Items per page
            max_page_size: Maximum allowed page size
            
        Returns:
            Validated (page, page_size)
        """
        page = max(1, page)
        page_size = max(1, min(page_size, max_page_size))
        return page, page_size


class RateLimiter:
    """
    Rate limiting implementation using token bucket algorithm
    
    Provides per-client rate limiting with configurable limits
    and time windows.
    """
    
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None,
                 default_limit: int = 100,
                 window_seconds: int = 60):
        """
        Initialize rate limiter
        
        Args:
            redis_client: Redis client for distributed rate limiting
            default_limit: Default request limit
            window_seconds: Time window in seconds
        """
        self.redis_client = redis_client or redis.Redis(
            host='localhost', 
            port=6379, 
            decode_responses=True
        )
        self.default_limit = default_limit
        self.window_seconds = window_seconds
    
    def check_rate_limit(self, 
                        client_id: str,
                        limit: Optional[int] = None) -> Tuple[bool, Dict[str, int]]:
        """
        Check if client has exceeded rate limit
        
        Args:
            client_id: Client identifier
            limit: Custom limit for this client
            
        Returns:
            Tuple of (is_allowed, rate_limit_info)
        """
        limit = limit or self.default_limit
        key = f"rate_limit:{client_id}"
        
        try:
            pipe = self.redis_client.pipeline()
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count requests in window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(uuid.uuid4()): now})
            
            # Set expiry
            pipe.expire(key, self.window_seconds)
            
            results = pipe.execute()
            request_count = results[1]
            
            # Check limit
            is_allowed = request_count < limit
            
            if not is_allowed:
                api_rate_limit_exceeded.labels(client_id=client_id).inc()
            
            return is_allowed, {
                "limit": limit,
                "remaining": max(0, limit - request_count - 1),
                "reset": int(now + self.window_seconds)
            }
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if rate limiting fails
            return True, {"limit": limit, "remaining": limit, "reset": 0}
    
    def get_client_id(self, request: Request) -> str:
        """
        Extract client ID from request
        
        Args:
            request: FastAPI request
            
        Returns:
            Client identifier
        """
        # Try to get from auth token
        auth_header = request.headers.get("authorization")
        if auth_header:
            try:
                token = auth_header.split(" ")[1]
                payload = jwt.decode(token, options={"verify_signature": False})
                if "sub" in payload:
                    return payload["sub"]
            except:
                pass
        
        # Fall back to IP address
        return request.client.host if request.client else "unknown"


class JWTHandler:
    """
    JWT token handling for authentication
    
    Implements JWT-based authentication following RFC 7519 standard.
    """
    
    def __init__(self, 
                 secret_key: str,
                 algorithm: str = "HS256",
                 expiration_minutes: int = 60):
        """
        Initialize JWT handler
        
        Args:
            secret_key: Secret key for signing tokens
            algorithm: Signing algorithm
            expiration_minutes: Token expiration time
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.expiration_minutes = expiration_minutes
    
    def create_token(self, 
                    subject: str,
                    claims: Optional[Dict[str, Any]] = None) -> str:
        """
        Create JWT token
        
        Args:
            subject: Token subject (usually user ID)
            claims: Additional claims
            
        Returns:
            JWT token string
        """
        payload = {
            "sub": subject,
            "iat": datetime.utcnow(),
            "exp": datetime.utcnow() + timedelta(minutes=self.expiration_minutes),
            "jti": str(uuid.uuid4())
        }
        
        if claims:
            payload.update(claims)
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """
        Verify and decode JWT token
        
        Args:
            token: JWT token string
            
        Returns:
            Token payload
            
        Raises:
            AuthenticationError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {e}")
    
    def refresh_token(self, token: str) -> str:
        """
        Refresh JWT token
        
        Args:
            token: Current JWT token
            
        Returns:
            New JWT token
        """
        payload = self.verify_token(token)
        return self.create_token(payload["sub"], {
            k: v for k, v in payload.items() 
            if k not in ["sub", "iat", "exp", "jti"]
        })


class APIKeyManager:
    """
    API key management for authentication
    
    Provides API key generation, validation, and management.
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """
        Initialize API key manager
        
        Args:
            redis_client: Redis client for key storage
        """
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=True
        )
        self.key_prefix = "api_key:"
    
    def generate_api_key(self, 
                        client_id: str,
                        metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate new API key
        
        Args:
            client_id: Client identifier
            metadata: Additional metadata
            
        Returns:
            Generated API key
        """
        # Generate random key
        api_key = f"sk_{uuid.uuid4().hex}"
        
        # Store in Redis
        key_data = {
            "client_id": client_id,
            "created_at": datetime.utcnow().isoformat(),
            "active": True
        }
        
        if metadata:
            key_data.update(metadata)
        
        self.redis_client.hset(
            f"{self.key_prefix}{api_key}",
            mapping=key_data
        )
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Validate API key
        
        Args:
            api_key: API key to validate
            
        Returns:
            Tuple of (is_valid, key_data)
        """
        key_data = self.redis_client.hgetall(f"{self.key_prefix}{api_key}")
        
        if not key_data:
            return False, None
        
        if key_data.get("active") != "True":
            return False, None
        
        # Update last used timestamp
        self.redis_client.hset(
            f"{self.key_prefix}{api_key}",
            "last_used",
            datetime.utcnow().isoformat()
        )
        
        return True, key_data
    
    def revoke_api_key(self, api_key: str) -> bool:
        """
        Revoke API key
        
        Args:
            api_key: API key to revoke
            
        Returns:
            Success status
        """
        return bool(self.redis_client.hset(
            f"{self.key_prefix}{api_key}",
            "active",
            False
        ))


def create_error_response(
    status_code: int,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create standardized error response
    
    Args:
        status_code: HTTP status code
        message: Error message
        details: Additional error details
        
    Returns:
        Error response dictionary
    """
    response = APIResponse(
        success=False,
        message=message,
        error={
            "code": status_code,
            "type": _get_error_type(status_code),
            "timestamp": datetime.utcnow().isoformat()
        }
    )
    
    if details:
        response.error["details"] = details
    
    return response.to_dict()


def _get_error_type(status_code: int) -> str:
    """Get error type from status code"""
    error_types = {
        400: "BAD_REQUEST",
        401: "UNAUTHORIZED",
        403: "FORBIDDEN",
        404: "NOT_FOUND",
        409: "CONFLICT",
        422: "VALIDATION_ERROR",
        429: "TOO_MANY_REQUESTS",
        500: "INTERNAL_ERROR",
        502: "BAD_GATEWAY",
        503: "SERVICE_UNAVAILABLE"
    }
    return error_types.get(status_code, "UNKNOWN_ERROR")


def api_response_wrapper(func: Callable) -> Callable:
    """
    Decorator to wrap API responses in standard format
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            
            # If result is already APIResponse, use it
            if isinstance(result, APIResponse):
                return result.to_dict()
            
            # Otherwise wrap in success response
            return APIResponse(
                success=True,
                data=result
            ).to_dict()
            
        except HTTPException as e:
            return create_error_response(e.status_code, e.detail)
        except Exception as e:
            logger.error(f"API error: {e}", exc_info=True)
            return create_error_response(500, "Internal server error")
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            
            if isinstance(result, APIResponse):
                return result.to_dict()
            
            return APIResponse(
                success=True,
                data=result
            ).to_dict()
            
        except HTTPException as e:
            return create_error_response(e.status_code, e.detail)
        except Exception as e:
            logger.error(f"API error: {e}", exc_info=True)
            return create_error_response(500, "Internal server error")
    
    # Return appropriate wrapper
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def validate_request_signature(
    request_body: bytes,
    signature: str,
    secret: str,
    algorithm: str = "sha256"
) -> bool:
    """
    Validate request signature for webhook security
    
    Args:
        request_body: Request body bytes
        signature: Provided signature
        secret: Shared secret
        algorithm: Hash algorithm
        
    Returns:
        Validation result
    """
    expected_signature = hmac.new(
        secret.encode(),
        request_body,
        getattr(hashlib, algorithm)
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


class ResponseCache:
    """
    Response caching for API endpoints
    
    Provides caching capabilities for API responses to improve
    performance and reduce backend load.
    """
    
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None,
                 default_ttl: int = 300):
        """
        Initialize response cache
        
        Args:
            redis_client: Redis client
            default_ttl: Default TTL in seconds
        """
        self.redis_client = redis_client or redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=False  # Store binary data
        )
        self.default_ttl = default_ttl
    
    def get_cache_key(self, 
                     endpoint: str,
                     params: Dict[str, Any]) -> str:
        """
        Generate cache key
        
        Args:
            endpoint: API endpoint
            params: Request parameters
            
        Returns:
            Cache key
        """
        # Sort params for consistent key generation
        sorted_params = json.dumps(params, sort_keys=True)
        
        # Generate hash
        key_hash = hashlib.md5(
            f"{endpoint}:{sorted_params}".encode()
        ).hexdigest()
        
        return f"api_cache:{key_hash}"
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached response"""
        try:
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set cached response"""
        try:
            self.redis_client.setex(
                key,
                ttl or self.default_ttl,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Cache set error: {e}")
    
    def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        try:
            for key in self.redis_client.scan_iter(match=pattern):
                self.redis_client.delete(key)
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")


def cache_response(ttl: int = 300):
    """
    Decorator for caching API responses
    
    Args:
        ttl: Cache TTL in seconds
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        cache = ResponseCache(default_ttl=ttl)
        
        @wraps(func)
        async def async_wrapper(request: Request, *args, **kwargs):
            # Generate cache key
            cache_key = cache.get_cache_key(
                request.url.path,
                dict(request.query_params)
            )
            
            # Check cache
            cached = cache.get(cache_key)
            if cached:
                return cached
            
            # Call function
            result = await func(request, *args, **kwargs)
            
            # Cache result
            cache.set(cache_key, result, ttl)
            
            return result
        
        return async_wrapper
    
    return decorator


# Export utilities
__all__ = [
    'APIResponse',
    'RequestValidator',
    'RateLimiter',
    'JWTHandler',
    'APIKeyManager',
    'create_error_response',
    'api_response_wrapper',
    'validate_request_signature',
    'ResponseCache',
    'cache_response'
]
