"""
Logging Middleware for REST API
================================================================================
Implements comprehensive request/response logging with structured logging
support for monitoring, debugging, and audit purposes.

This middleware captures detailed information about API requests and responses
following structured logging best practices and privacy regulations.

References:
    - OWASP Logging Cheat Sheet
    - The Twelve-Factor App: XI. Logs
    - Structured Logging Best Practices

Author: Võ Hải Dũng
License: MIT
"""

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for structured logging of HTTP requests and responses.
    
    Captures request details, response status, processing time, and
    contextual information for comprehensive API monitoring.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        log_request_body: bool = False,
        log_response_body: bool = False,
        max_body_length: int = 1000,
        sensitive_fields: Optional[list] = None,
        exclude_paths: Optional[list] = None
    ):
        """
        Initialize logging middleware.
        
        Args:
            app: ASGI application
            log_request_body: Whether to log request bodies
            log_response_body: Whether to log response bodies
            max_body_length: Maximum body length to log
            sensitive_fields: Fields to redact from logs
            exclude_paths: Paths to exclude from logging
        """
        super().__init__(app)
        self.log_request_body = log_request_body
        self.log_response_body = log_response_body
        self.max_body_length = max_body_length
        self.sensitive_fields = sensitive_fields or [
            "password", "token", "api_key", "secret", "authorization"
        ]
        self.exclude_paths = exclude_paths or ["/health", "/metrics"]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and log details.
        
        Args:
            request: Incoming request
            call_next: Next middleware/handler
            
        Returns:
            Response from the application
        """
        # Skip logging for excluded paths
        if request.url.path in self.exclude_paths:
            return await call_next(request)
        
        # Generate request ID if not present
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Log request
        await self._log_request(request, request_id)
        
        # Process request
        response = None
        error = None
        
        try:
            response = await call_next(request)
            return response
            
        except Exception as e:
            error = e
            raise
            
        finally:
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Log response
            await self._log_response(
                request,
                response,
                request_id,
                processing_time,
                error
            )
    
    async def _log_request(self, request: Request, request_id: str) -> None:
        """
        Log incoming request details.
        
        Args:
            request: Incoming request
            request_id: Unique request identifier
        """
        # Extract request information
        log_data = {
            "event": "api_request",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client": {
                "host": request.client.host if request.client else None,
                "port": request.client.port if request.client else None
            },
            "headers": self._sanitize_headers(dict(request.headers))
        }
        
        # Add request body if configured
        if self.log_request_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = await request.body()
                request._body = body  # Store for later use
                
                if body:
                    body_str = body.decode("utf-8")
                    if len(body_str) <= self.max_body_length:
                        try:
                            body_json = json.loads(body_str)
                            log_data["body"] = self._sanitize_data(body_json)
                        except json.JSONDecodeError:
                            log_data["body"] = body_str[:self.max_body_length]
                    else:
                        log_data["body_truncated"] = True
                        log_data["body_size"] = len(body_str)
            except Exception as e:
                logger.debug(f"Failed to read request body: {str(e)}")
        
        # Log at appropriate level
        logger.info(
            f"Incoming request: {request.method} {request.url.path}",
            extra=log_data
        )
    
    async def _log_response(
        self,
        request: Request,
        response: Optional[Response],
        request_id: str,
        processing_time: float,
        error: Optional[Exception] = None
    ) -> None:
        """
        Log response details.
        
        Args:
            request: Original request
            response: Response object
            request_id: Request identifier
            processing_time: Request processing time
            error: Exception if occurred
        """
        log_data = {
            "event": "api_response",
            "request_id": request_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "method": request.method,
            "path": request.url.path,
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
        if response:
            log_data["status_code"] = response.status_code
            log_data["response_headers"] = self._sanitize_headers(
                dict(response.headers)
            )
            
            # Log response body if configured
            if self.log_response_body and response.status_code >= 400:
                try:
                    if hasattr(response, "body"):
                        body = response.body
                        if isinstance(body, bytes):
                            body_str = body.decode("utf-8")
                            if len(body_str) <= self.max_body_length:
                                try:
                                    log_data["response_body"] = json.loads(body_str)
                                except json.JSONDecodeError:
                                    log_data["response_body"] = body_str[:self.max_body_length]
                except Exception as e:
                    logger.debug(f"Failed to read response body: {str(e)}")
        
        if error:
            log_data["error"] = {
                "type": type(error).__name__,
                "message": str(error)
            }
            logger.error(
                f"Request failed: {request.method} {request.url.path}",
                extra=log_data,
                exc_info=True
            )
        else:
            # Determine log level based on status code
            if response and response.status_code >= 500:
                log_level = logger.error
            elif response and response.status_code >= 400:
                log_level = logger.warning
            else:
                log_level = logger.info
            
            log_level(
                f"Request completed: {request.method} {request.url.path} "
                f"[{response.status_code if response else 'ERROR'}] "
                f"({processing_time:.3f}s)",
                extra=log_data
            )
    
    def _sanitize_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        """
        Sanitize headers by redacting sensitive information.
        
        Args:
            headers: Request/response headers
            
        Returns:
            Sanitized headers dictionary
        """
        sanitized = {}
        
        for key, value in headers.items():
            key_lower = key.lower()
            
            # Redact sensitive headers
            if any(field in key_lower for field in self.sensitive_fields):
                if key_lower == "authorization" and value.startswith("Bearer "):
                    # Show partial token for debugging
                    sanitized[key] = f"Bearer {value[7:12]}...REDACTED"
                else:
                    sanitized[key] = "REDACTED"
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_data(self, data: Any) -> Any:
        """
        Recursively sanitize sensitive data.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(field in key.lower() for field in self.sensitive_fields):
                    sanitized[key] = "REDACTED"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        else:
            return data
