"""
CORS (Cross-Origin Resource Sharing) Handler
================================================================================
Implements CORS handling for API endpoints to enable secure cross-origin
requests from web browsers.

This module provides configurable CORS policies following W3C CORS specification
and OWASP security best practices.

References:
    - W3C (2020). Cross-Origin Resource Sharing
    - OWASP (2021). Cross-Origin Resource Sharing Security Cheat Sheet
    - MDN Web Docs. HTTP access control (CORS)

Author: AG News Development Team
License: MIT
"""

import fnmatch
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class CORSConfig:
    """
    CORS configuration settings.
    
    Implements secure defaults following OWASP recommendations.
    
    Attributes:
        allowed_origins: List of allowed origins or patterns
        allowed_methods: HTTP methods allowed for CORS requests
        allowed_headers: Headers allowed in requests
        exposed_headers: Headers exposed to the browser
        allow_credentials: Whether to allow credentials in CORS requests
        max_age: Preflight cache duration in seconds
        vary_origin: Whether to vary response by origin
    """
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    allowed_methods: List[str] = field(
        default_factory=lambda: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    )
    allowed_headers: List[str] = field(
        default_factory=lambda: ["Content-Type", "Authorization", "X-Request-ID"]
    )
    exposed_headers: List[str] = field(
        default_factory=lambda: ["X-Total-Count", "X-Page-Count"]
    )
    allow_credentials: bool = False
    max_age: int = 3600  # 1 hour
    vary_origin: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        # Security check: credentials with wildcard origin
        if self.allow_credentials and "*" in self.allowed_origins:
            logger.warning(
                "CORS configuration allows credentials with wildcard origin. "
                "This is a security risk and will be disabled."
            )
            self.allow_credentials = False
        
        # Normalize methods to uppercase
        self.allowed_methods = [m.upper() for m in self.allowed_methods]
        
        # Always allow OPTIONS for preflight
        if "OPTIONS" not in self.allowed_methods:
            self.allowed_methods.append("OPTIONS")


class CORSHandler:
    """
    Handler for CORS (Cross-Origin Resource Sharing) policies.
    
    Provides methods to validate origins, generate CORS headers,
    and handle preflight requests.
    """
    
    def __init__(self, config: CORSConfig = None):
        """
        Initialize CORS handler.
        
        Args:
            config: CORS configuration
        """
        self.config = config or CORSConfig()
        self._compiled_patterns = self._compile_origin_patterns()
        
        logger.info(
            f"CORS handler initialized with origins: {self.config.allowed_origins}"
        )
    
    def _compile_origin_patterns(self) -> List[str]:
        """
        Compile origin patterns for efficient matching.
        
        Returns:
            List of compiled patterns
        """
        patterns = []
        for origin in self.config.allowed_origins:
            if "*" in origin and origin != "*":
                # Convert wildcard pattern to regex pattern
                patterns.append(origin)
        return patterns
    
    def is_origin_allowed(self, origin: str) -> bool:
        """
        Check if origin is allowed by CORS policy.
        
        Args:
            origin: Origin header value
            
        Returns:
            True if origin is allowed
        """
        if not origin:
            return False
        
        # Check wildcard
        if "*" in self.config.allowed_origins:
            return True
        
        # Check exact match
        if origin in self.config.allowed_origins:
            return True
        
        # Check patterns
        for pattern in self._compiled_patterns:
            if fnmatch.fnmatch(origin, pattern):
                return True
        
        return False
    
    def get_cors_headers(
        self,
        origin: str,
        request_method: str = None,
        request_headers: List[str] = None
    ) -> Dict[str, str]:
        """
        Generate CORS response headers.
        
        Args:
            origin: Request origin
            request_method: Requested method (for preflight)
            request_headers: Requested headers (for preflight)
            
        Returns:
            Dictionary of CORS headers
        """
        headers = {}
        
        # Check if origin is allowed
        if not self.is_origin_allowed(origin):
            logger.debug(f"Origin not allowed: {origin}")
            return headers
        
        # Access-Control-Allow-Origin
        if "*" in self.config.allowed_origins and not self.config.allow_credentials:
            headers["Access-Control-Allow-Origin"] = "*"
        else:
            headers["Access-Control-Allow-Origin"] = origin
            if self.config.vary_origin:
                headers["Vary"] = "Origin"
        
        # Access-Control-Allow-Credentials
        if self.config.allow_credentials:
            headers["Access-Control-Allow-Credentials"] = "true"
        
        # Preflight response headers
        if request_method:
            headers.update(self._get_preflight_headers(request_method, request_headers))
        
        # Exposed headers
        if self.config.exposed_headers:
            headers["Access-Control-Expose-Headers"] = ", ".join(
                self.config.exposed_headers
            )
        
        return headers
    
    def _get_preflight_headers(
        self,
        request_method: str,
        request_headers: List[str] = None
    ) -> Dict[str, str]:
        """
        Generate preflight response headers.
        
        Args:
            request_method: Requested method
            request_headers: Requested headers
            
        Returns:
            Dictionary of preflight headers
        """
        headers = {}
        
        # Check if method is allowed
        if request_method and request_method.upper() in self.config.allowed_methods:
            headers["Access-Control-Allow-Methods"] = ", ".join(
                self.config.allowed_methods
            )
        
        # Check requested headers
        if request_headers:
            allowed = self._filter_allowed_headers(request_headers)
            if allowed:
                headers["Access-Control-Allow-Headers"] = ", ".join(allowed)
        else:
            # Return all allowed headers
            headers["Access-Control-Allow-Headers"] = ", ".join(
                self.config.allowed_headers
            )
        
        # Max age for preflight cache
        headers["Access-Control-Max-Age"] = str(self.config.max_age)
        
        return headers
    
    def _filter_allowed_headers(self, request_headers: List[str]) -> List[str]:
        """
        Filter requested headers against allowed headers.
        
        Args:
            request_headers: List of requested headers
            
        Returns:
            List of allowed headers
        """
        allowed = []
        
        # Normalize header names
        allowed_lower = [h.lower() for h in self.config.allowed_headers]
        
        for header in request_headers:
            header_lower = header.lower()
            
            # Check exact match or wildcard
            if header_lower in allowed_lower or "*" in allowed_lower:
                allowed.append(header)
            # Check for simple headers (always allowed)
            elif header_lower in ["accept", "accept-language", "content-language"]:
                allowed.append(header)
            # Check for content-type with simple values
            elif header_lower == "content-type":
                allowed.append(header)
        
        return allowed
    
    def handle_preflight(
        self,
        origin: str,
        request_method: str,
        request_headers: str = None
    ) -> tuple[int, Dict[str, str]]:
        """
        Handle CORS preflight request.
        
        Args:
            origin: Origin header
            request_method: Access-Control-Request-Method header
            request_headers: Access-Control-Request-Headers header
            
        Returns:
            Tuple of (status_code, headers)
        """
        # Parse requested headers
        headers_list = None
        if request_headers:
            headers_list = [h.strip() for h in request_headers.split(",")]
        
        # Generate CORS headers
        cors_headers = self.get_cors_headers(origin, request_method, headers_list)
        
        if cors_headers:
            # Successful preflight
            return 204, cors_headers  # No Content
        else:
            # CORS policy violation
            logger.warning(
                f"CORS preflight rejected - Origin: {origin}, "
                f"Method: {request_method}, Headers: {request_headers}"
            )
            return 403, {}  # Forbidden
    
    def add_cors_headers(
        self,
        response_headers: Dict[str, str],
        origin: str
    ) -> Dict[str, str]:
        """
        Add CORS headers to existing response headers.
        
        Args:
            response_headers: Existing response headers
            origin: Request origin
            
        Returns:
            Updated headers dictionary
        """
        cors_headers = self.get_cors_headers(origin)
        response_headers.update(cors_headers)
        return response_headers
    
    def validate_cors_request(
        self,
        origin: str,
        method: str,
        headers: Dict[str, str] = None
    ) -> bool:
        """
        Validate if CORS request is allowed.
        
        Args:
            origin: Request origin
            method: HTTP method
            headers: Request headers
            
        Returns:
            True if request is allowed
        """
        # Check origin
        if not self.is_origin_allowed(origin):
            logger.debug(f"CORS validation failed: origin {origin} not allowed")
            return False
        
        # Check method
        if method.upper() not in self.config.allowed_methods:
            logger.debug(f"CORS validation failed: method {method} not allowed")
            return False
        
        # Check headers if provided
        if headers:
            for header in headers:
                if header.lower() not in [h.lower() for h in self.config.allowed_headers]:
                    # Check if it's a simple header
                    if header.lower() not in ["accept", "accept-language", "content-language", "content-type"]:
                        logger.debug(f"CORS validation failed: header {header} not allowed")
                        return False
        
        return True
    
    def update_config(self, config: CORSConfig) -> None:
        """
        Update CORS configuration.
        
        Args:
            config: New CORS configuration
        """
        self.config = config
        self._compiled_patterns = self._compile_origin_patterns()
        logger.info("CORS configuration updated")
