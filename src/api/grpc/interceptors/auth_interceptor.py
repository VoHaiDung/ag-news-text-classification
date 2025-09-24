"""
Authentication Interceptor
================================================================================
This module implements authentication and authorization interceptor for gRPC
services, following OAuth 2.0 and JWT standards.

Implements security features including:
- JWT token validation
- Role-based access control (RBAC)
- API key authentication
- Method-level authorization

References:
    - RFC 6749: The OAuth 2.0 Authorization Framework
    - RFC 7519: JSON Web Token (JWT)
    - OWASP Authentication Cheat Sheet

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Callable, Any, Optional, Dict, List
import grpc
from jose import jwt, JWTError
import time

from . import BaseInterceptor

logger = logging.getLogger(__name__)

# Security configuration
SECRET_KEY = "your-secret-key-here"  # Should be loaded from environment
ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 30

# Method permissions mapping
METHOD_PERMISSIONS = {
    "/ag_news.ClassificationService/Classify": ["predict", "user"],
    "/ag_news.ClassificationService/ClassifyBatch": ["predict", "user"],
    "/ag_news.TrainingService/TrainModel": ["train", "admin"],
    "/ag_news.ModelManagementService/DeployModel": ["deploy", "admin"],
    "/ag_news.DataService/UploadData": ["data_write", "admin"]
}

class AuthInterceptor(BaseInterceptor):
    """
    Authentication and authorization interceptor.
    
    Validates JWT tokens and enforces role-based access control
    for gRPC method invocations.
    """
    
    def __init__(self):
        """Initialize authentication interceptor."""
        super().__init__("AuthInterceptor")
        self.public_methods = [
            "/grpc.health.v1.Health/Check",
            "/grpc.health.v1.Health/Watch",
            "/ag_news.ClassificationService/GetModelInfo"
        ]
        
    def intercept_unary_unary(
        self,
        request: Any,
        context: grpc.ServicerContext,
        method: Callable,
        handler_call_details: grpc.HandlerCallDetails
    ) -> Any:
        """
        Intercept unary-unary RPC for authentication.
        
        Args:
            request: Request message
            context: gRPC context
            method: Original method
            handler_call_details: Call details
            
        Returns:
            Any: Response message
        """
        try:
            # Check if method requires authentication
            if not self._requires_auth(handler_call_details.method):
                return method(request, context)
            
            # Extract and validate token
            user_info = self._authenticate(context)
            if not user_info:
                context.abort(
                    grpc.StatusCode.UNAUTHENTICATED,
                    "Authentication required"
                )
            
            # Check authorization
            if not self._authorize(user_info, handler_call_details.method):
                context.abort(
                    grpc.StatusCode.PERMISSION_DENIED,
                    "Insufficient permissions"
                )
            
            # Add user info to context
            context.user_info = user_info
            
            # Call original method
            response = method(request, context)
            
            self.metrics["requests_intercepted"] += 1
            return response
            
        except grpc.RpcError:
            # Re-raise gRPC errors
            raise
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            self.metrics["errors_handled"] += 1
            context.abort(
                grpc.StatusCode.INTERNAL,
                "Authentication processing failed"
            )
    
    def _requires_auth(self, method: str) -> bool:
        """
        Check if method requires authentication.
        
        Args:
            method: gRPC method name
            
        Returns:
            bool: True if authentication required
        """
        return method not in self.public_methods
    
    def _authenticate(self, context: grpc.ServicerContext) -> Optional[Dict[str, Any]]:
        """
        Authenticate request using JWT token or API key.
        
        Args:
            context: gRPC context
            
        Returns:
            Optional[Dict[str, Any]]: User information if authenticated
        """
        # Get metadata
        metadata = dict(context.invocation_metadata())
        
        # Check for JWT token
        auth_header = metadata.get('authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            return self._validate_jwt(token)
        
        # Check for API key
        api_key = metadata.get('x-api-key', '')
        if api_key:
            return self._validate_api_key(api_key)
        
        return None
    
    def _validate_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Validate JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Optional[Dict[str, Any]]: User information from token
        """
        try:
            payload = jwt.decode(
                token,
                SECRET_KEY,
                algorithms=[ALGORITHM]
            )
            
            # Check expiration
            exp = payload.get('exp', 0)
            if exp < time.time():
                logger.warning("Token expired")
                return None
            
            # Extract user information
            user_info = {
                'user_id': payload.get('sub'),
                'username': payload.get('username'),
                'role': payload.get('role', 'user'),
                'permissions': payload.get('permissions', []),
                'auth_type': 'jwt'
            }
            
            return user_info
            
        except JWTError as e:
            logger.error(f"JWT validation error: {e}")
            return None
    
    def _validate_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Validate API key.
        
        Args:
            api_key: API key string
            
        Returns:
            Optional[Dict[str, Any]]: Service information
        """
        # Validate API key (placeholder implementation)
        valid_keys = {
            "test-api-key": {
                'service_id': 'test-service',
                'role': 'service',
                'permissions': ['predict', 'data_read'],
                'auth_type': 'api_key'
            }
        }
        
        return valid_keys.get(api_key)
    
    def _authorize(self, user_info: Dict[str, Any], method: str) -> bool:
        """
        Check if user is authorized for method.
        
        Args:
            user_info: User information
            method: gRPC method name
            
        Returns:
            bool: Authorization result
        """
        # Admin has full access
        if user_info.get('role') == 'admin':
            return True
        
        # Get required permissions for method
        required_permissions = METHOD_PERMISSIONS.get(method, [])
        if not required_permissions:
            # No specific permissions required
            return True
        
        # Check if user has any required permission
        user_permissions = user_info.get('permissions', [])
        user_permissions.append(user_info.get('role', ''))
        
        return any(perm in user_permissions for perm in required_permissions)
