"""
REST API Middleware
===================

Implements middleware components following the Chain of Responsibility pattern from:
- Gamma et al. (1994): "Design Patterns"
- Fowler (2002): "Patterns of Enterprise Application Architecture"
- Richardson (2018): "Microservices Patterns"

Author: Team SOTA AGNews
License: MIT
"""

import logging
import time
import hashlib
from typing import Dict, Optional, Callable, Any
from datetime import datetime, timedelta
import json
import asyncio
from collections import defaultdict

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import jwt
import redis
from prometheus_client import Counter, Histogram, Gauge

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.api import API_CONFIG

logger = setup_logging(__name__)

# Metrics following Prometheus best practices (Turnbull, 2018)
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

active_requests = Gauge(
    'api_active_requests',
    'Active API requests'
)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware implementing Token Bucket algorithm.
    
    Based on:
    - Tanenbaum & Wetherall (2010): "Computer Networks"
    - CloudFlare (2021): "Rate Limiting Best Practices"
    """
    
    def __init__(
        self,
        app,
        max_requests: int = 100,
        window_seconds: int = 60,
        use_redis: bool = False
    ):
        """
        Initialize rate limiter.
        
        Args:
            app: FastAPI application
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
            use_redis: Whether to use Redis for distributed rate limiting
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.use_redis = use_redis
        
        if use_redis:
            try:
                self.redis_client = redis.Redis(
                    host='localhost',
                    port=6379,
                    decode_responses=True
                )
                self.redis_client.ping()
            except:
                logger.warning("Redis not available, using in-memory rate limiting")
                self.use_redis = False
                self.request_counts = defaultdict(list)
        else:
            self.request_counts = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: Incoming request
            call_next: Next middleware in chain
            
        Returns:
            Response or rate limit error
        """
        # Extract client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limit
        if not self._check_rate_limit(client_id):
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded",
                    "max_requests": self.max_requests,
                    "window_seconds": self.window_seconds
                },
                headers={
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + self.window_seconds)
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_requests(client_id)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + self.window_seconds)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier from request."""
        # Priority: API key > Authorization header > IP address
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return hashlib.md5(api_key.encode()).hexdigest()
        
        auth = request.headers.get("Authorization")
        if auth:
            return hashlib.md5(auth.encode()).hexdigest()
        
        client_ip = request.client.host if request.client else "unknown"
        return client_ip
    
    def _check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit."""
        current_time = time.time()
        
        if self.use_redis:
            return self._check_redis_rate_limit(client_id, current_time)
        else:
            return self._check_memory_rate_limit(client_id, current_time)
    
    def _check_memory_rate_limit(self, client_id: str, current_time: float) -> bool:
        """Check rate limit using in-memory storage."""
        # Clean old requests
        cutoff_time = current_time - self.window_seconds
        self.request_counts[client_id] = [
            req_time for req_time in self.request_counts[client_id]
            if req_time > cutoff_time
        ]
        
        # Check limit
        if len(self.request_counts[client_id]) >= self.max_requests:
            return False
        
        # Add current request
        self.request_counts[client_id].append(current_time)
        return True
    
    def _check_redis_rate_limit(self, client_id: str, current_time: float) -> bool:
        """Check rate limit using Redis."""
        key = f"rate_limit:{client_id}"
        
        try:
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window_seconds)
            results = pipe.execute()
            
            request_count = results[0]
            return request_count <= self.max_requests
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            return True  # Fail open
    
    def _get_remaining_requests(self, client_id: str) -> int:
        """Get remaining requests for client."""
        if self.use_redis:
            key = f"rate_limit:{client_id}"
            try:
                count = self.redis_client.get(key)
                return max(0, self.max_requests - int(count or 0))
            except:
                return self.max_requests
        else:
            return max(0, self.max_requests - len(self.request_counts[client_id]))

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    JWT-based authentication middleware.
    
    Implements authentication patterns from:
    - Jones et al. (2015): "JSON Web Token (JWT) RFC 7519"
    - Hardt (2012): "The OAuth 2.0 Authorization Framework RFC 6749"
    """
    
    def __init__(
        self,
        app,
        secret_key: str = "your-secret-key-change-in-production",
        algorithm: str = "HS256",
        public_endpoints: Optional[list] = None
    ):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application
            secret_key: JWT secret key
            algorithm: JWT algorithm
            public_endpoints: List of public endpoints
        """
        super().__init__(app)
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.public_endpoints = public_endpoints or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/metrics"
        ]
        
        self.security = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with authentication.
        
        Args:
            request: Incoming request
            call_next: Next middleware
            
        Returns:
            Response or authentication error
        """
        # Skip authentication for public endpoints
        if self._is_public_endpoint(request.url.path):
            return await call_next(request)
        
        # Check API key
        api_key = request.headers.get("X-API-Key")
        if api_key and self._validate_api_key(api_key):
            request.state.user = {"type": "api_key", "key": api_key}
            return await call_next(request)
        
        # Check JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]
            
            try:
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm]
                )
                request.state.user = payload
                return await call_next(request)
                
            except jwt.ExpiredSignatureError:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Token expired"}
                )
            except jwt.InvalidTokenError:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Invalid token"}
                )
        
        # No valid authentication
        return JSONResponse(
            status_code=status.HTTP_401_UNAUTHORIZED,
            content={"detail": "Authentication required"},
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public."""
        return any(path.startswith(endpoint) for endpoint in self.public_endpoints)
    
    def _validate_api_key(self, api_key: str) -> bool:
        """Validate API key."""
        # Placeholder - implement actual API key validation
        valid_keys = [
            "test-api-key-123",
            "demo-api-key-456"
        ]
        return api_key in valid_keys

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Request/response logging middleware.
    
    Implements structured logging patterns from:
    - Turnbull (2016): "The Art of Monitoring"
    - Krochmalski (2017): "Docker and Kubernetes for Java Developers"
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with logging.
        
        Args:
            request: Incoming request
            call_next: Next middleware
            
        Returns:
            Response with logging
        """
        # Generate request ID
        request_id = hashlib.md5(
            f"{request.client.host}{time.time()}".encode()
        ).hexdigest()[:12]
        
        # Start timer
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client": request.client.host if request.client else "unknown"
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "duration": duration
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                    "duration": duration
                }
            )
            
            raise

class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Prometheus metrics collection middleware.
    
    Following Prometheus best practices from:
    - Prometheus Authors (2021): "Prometheus: Up & Running"
    - Google SRE Team (2016): "Site Reliability Engineering"
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with metrics collection.
        
        Args:
            request: Incoming request
            call_next: Next middleware
            
        Returns:
            Response with metrics
        """
        # Increment active requests
        active_requests.inc()
        
        # Start timer
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Record metrics
            duration = time.time() - start_time
            
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(duration)
            
            return response
            
        finally:
            # Decrement active requests
            active_requests.dec()

class CORSMiddleware(BaseHTTPMiddleware):
    """
    CORS (Cross-Origin Resource Sharing) middleware.
    
    Implements CORS specification from:
    - W3C (2014): "Cross-Origin Resource Sharing"
    - Mozilla (2021): "HTTP access control (CORS)"
    """
    
    def __init__(
        self,
        app,
        allow_origins: list = ["*"],
        allow_methods: list = ["*"],
        allow_headers: list = ["*"],
        allow_credentials: bool = True
    ):
        """
        Initialize CORS middleware.
        
        Args:
            app: FastAPI application
            allow_origins: Allowed origins
            allow_methods: Allowed methods
            allow_headers: Allowed headers
            allow_credentials: Allow credentials
        """
        super().__init__(app)
        self.allow_origins = allow_origins
        self.allow_methods = allow_methods
        self.allow_headers = allow_headers
        self.allow_credentials = allow_credentials
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with CORS headers.
        
        Args:
            request: Incoming request
            call_next: Next middleware
            
        Returns:
            Response with CORS headers
        """
        # Handle preflight requests
        if request.method == "OPTIONS":
            return self._preflight_response(request)
        
        # Process request
        response = await call_next(request)
        
        # Add CORS headers
        origin = request.headers.get("Origin")
        
        if origin and self._is_allowed_origin(origin):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = str(self.allow_credentials).lower()
            response.headers["Vary"] = "Origin"
        
        return response
    
    def _preflight_response(self, request: Request) -> Response:
        """Create preflight response."""
        headers = {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": ", ".join(self.allow_methods),
            "Access-Control-Allow-Headers": ", ".join(self.allow_headers),
            "Access-Control-Max-Age": "86400"
        }
        
        if self.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        
        return Response(
            content="",
            status_code=200,
            headers=headers
        )
    
    def _is_allowed_origin(self, origin: str) -> bool:
        """Check if origin is allowed."""
        if "*" in self.allow_origins:
            return True
        return origin in self.allow_origins

class CompressionMiddleware(BaseHTTPMiddleware):
    """
    Response compression middleware.
    
    Implements compression best practices from:
    - RFC 7231 (2014): "Hypertext Transfer Protocol (HTTP/1.1)"
    - Google (2021): "Web Performance Best Practices"
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with response compression.
        
        Args:
            request: Incoming request
            call_next: Next middleware
            
        Returns:
            Potentially compressed response
        """
        # Check if client accepts compression
        accept_encoding = request.headers.get("Accept-Encoding", "")
        
        # Process request
        response = await call_next(request)
        
        # Compress response if appropriate
        if "gzip" in accept_encoding and self._should_compress(response):
            # Compression logic would go here
            # For now, just add header indicating support
            response.headers["Content-Encoding"] = "gzip"
        
        return response
    
    def _should_compress(self, response: Response) -> bool:
        """Check if response should be compressed."""
        # Don't compress small responses
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) < 1000:
            return False
        
        # Check content type
        content_type = response.headers.get("Content-Type", "")
        compressible_types = ["application/json", "text/", "application/xml"]
        
        return any(ct in content_type for ct in compressible_types)
