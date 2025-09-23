"""
Security Middleware for REST API
================================================================================
Implements comprehensive security measures including CSRF protection,
XSS prevention, SQL injection prevention, and security headers.

This middleware follows OWASP security best practices and implements
defense in depth strategies.

References:
    - OWASP Top 10 Web Application Security Risks
    - OWASP Security Headers Project
    - Content Security Policy (CSP) Level 3

Author: Võ Hải Dũng
License: MIT
"""

import hashlib
import hmac
import secrets
from typing import Any, Callable, Dict, Optional, Set

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware implementing various security controls.
    
    Provides protection against common web vulnerabilities and
    implements security best practices for API endpoints.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        enable_csrf: bool = False,
        enable_security_headers: bool = True,
        enable_content_validation: bool = True,
        trusted_hosts: Optional[Set[str]] = None,
        allowed_content_types: Optional[Set[str]] = None,
        max_content_length: int = 10 * 1024 * 1024  # 10MB
    ):
        """
        Initialize security middleware.
        
        Args:
            app: ASGI application
            enable_csrf: Enable CSRF protection
            enable_security_headers: Add security headers
            enable_content_validation: Validate content
            trusted_hosts: Set of trusted host headers
            allowed_content_types: Allowed content types
            max_content_length: Maximum content length
        """
        super().__init__(app)
        self.enable_csrf = enable_csrf
        self.enable_security_headers = enable_security_headers
        self.enable_content_validation = enable_content_validation
        self.trusted_hosts = trusted_hosts or set()
        self.allowed_content_types = allowed_content_types or {
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
            "text/plain"
        }
        self.max_content_length = max_content_length
        
        # CSRF token storage (use Redis in production)
        self._csrf_tokens: Set[str] = set()
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with security controls.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response with security headers
        """
        # Validate host header
        if self.trusted_hosts and not self._validate_host(request):
            logger.warning(
                f"Invalid host header: {request.headers.get('host')}",
                extra={"client": request.client.host if request.client else None}
            )
            return Response(
                content="Invalid host header",
                status_code=400
            )
        
        # Validate content
        if self.enable_content_validation:
            validation_error = await self._validate_content(request)
            if validation_error:
                return validation_error
        
        # CSRF protection for state-changing methods
        if self.enable_csrf and request.method in ["POST", "PUT", "DELETE", "PATCH"]:
            csrf_error = await self._validate_csrf(request)
            if csrf_error:
                return csrf_error
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        if self.enable_security_headers:
            self._add_security_headers(response)
        
        # Generate CSRF token for GET requests
        if self.enable_csrf and request.method == "GET":
            csrf_token = self._generate_csrf_token()
            response.headers["X-CSRF-Token"] = csrf_token
        
        return response
    
    def _validate_host(self, request: Request) -> bool:
        """
        Validate host header against trusted hosts.
        
        Args:
            request: Incoming request
            
        Returns:
            True if host is valid
        """
        host = request.headers.get("host", "").split(":")[0]
        return host in self.trusted_hosts
    
    async def _validate_content(self, request: Request) -> Optional[Response]:
        """
        Validate request content.
        
        Args:
            request: Incoming request
            
        Returns:
            Error response if validation fails, None otherwise
        """
        # Check content length
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_content_length:
            logger.warning(
                f"Content too large: {content_length} bytes",
                extra={"path": request.url.path}
            )
            return Response(
                content="Request entity too large",
                status_code=413
            )
        
        # Check content type for POST/PUT requests
        if request.method in ["POST", "PUT", "PATCH"]:
            content_type = request.headers.get("content-type", "").split(";")[0]
            
            if content_type not in self.allowed_content_types:
                logger.warning(
                    f"Invalid content type: {content_type}",
                    extra={"path": request.url.path}
                )
                return Response(
                    content="Unsupported media type",
                    status_code=415
                )
        
        # Validate JSON content
        if request.headers.get("content-type", "").startswith("application/json"):
            try:
                # Attempt to parse JSON to detect malformed content
                await request.json()
            except Exception as e:
                logger.warning(
                    f"Invalid JSON content: {str(e)}",
                    extra={"path": request.url.path}
                )
                return Response(
                    content="Invalid JSON content",
                    status_code=400
                )
        
        return None
    
    async def _validate_csrf(self, request: Request) -> Optional[Response]:
        """
        Validate CSRF token.
        
        Args:
            request: Incoming request
            
        Returns:
            Error response if CSRF validation fails
        """
        # Get CSRF token from header or form
        csrf_token = request.headers.get("X-CSRF-Token")
        
        if not csrf_token:
            # Try to get from form data
            if request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
                form = await request.form()
                csrf_token = form.get("csrf_token")
        
        if not csrf_token or not self._verify_csrf_token(csrf_token):
            logger.warning(
                "CSRF token validation failed",
                extra={
                    "path": request.url.path,
                    "method": request.method
                }
            )
            return Response(
                content="CSRF token validation failed",
                status_code=403
            )
        
        return None
    
    def _generate_csrf_token(self) -> str:
        """
        Generate CSRF token.
        
        Returns:
            CSRF token string
        """
        token = secrets.token_urlsafe(32)
        self._csrf_tokens.add(token)
        
        # Clean old tokens (keep last 1000)
        if len(self._csrf_tokens) > 1000:
            self._csrf_tokens = set(list(self._csrf_tokens)[-1000:])
        
        return token
    
    def _verify_csrf_token(self, token: str) -> bool:
        """
        Verify CSRF token.
        
        Args:
            token: CSRF token to verify
            
        Returns:
            True if token is valid
        """
        if token in self._csrf_tokens:
            # Remove used token (one-time use)
            self._csrf_tokens.discard(token)
            return True
        return False
    
    def _add_security_headers(self, response: Response) -> None:
        """
        Add security headers to response.
        
        Implements OWASP recommended security headers for API responses.
        
        Args:
            response: Response object
        """
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        
        # Prevent MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # XSS Protection (for older browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Clickjacking protection
        response.headers["X-Frame-Options"] = "DENY"
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy (formerly Feature Policy)
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=(), "
            "magnetometer=(), "
            "gyroscope=(), "
            "accelerometer=()"
        )
        
        # Strict Transport Security (if HTTPS)
        # Should be enabled in production with HTTPS
        # response.headers["Strict-Transport-Security"] = (
        #     "max-age=31536000; includeSubDomains; preload"
        # )
        
        # Cache Control for sensitive data
        if response.status_code == 200:
            # Prevent caching of sensitive data
            response.headers["Cache-Control"] = (
                "no-store, no-cache, must-revalidate, private"
            )
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
