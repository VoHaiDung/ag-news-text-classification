"""
Authentication and Authorization Module
================================================================================
Implements comprehensive authentication and authorization mechanisms for API
security following OAuth 2.0, JWT standards, and security best practices.

This module provides multiple authentication strategies including JWT tokens,
API keys, OAuth2, and implements role-based access control (RBAC) with
fine-grained permissions.

References:
    - Hardt, D. (2012). The OAuth 2.0 Authorization Framework. RFC 6749
    - Jones, M., Bradley, J., & Sakimura, N. (2015). JSON Web Token (JWT). RFC 7519
    - OWASP (2021). Authentication Cheat Sheet
    - Seemann, M. (2011). Dependency Injection in .NET

Author: AG News Development Team
License: MIT
"""

import hashlib
import hmac
import logging
import os
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import jwt
from passlib.context import CryptContext

from src.core.exceptions import (
    UnauthorizedError,
    ForbiddenError,
    AuthenticationError,
    AuthorizationError
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Password hashing configuration following OWASP recommendations
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # OWASP recommended minimum for bcrypt
)


class AuthType(Enum):
    """
    Authentication types supported by the system.
    
    Each type represents a different authentication mechanism with
    specific security characteristics and use cases.
    """
    JWT = "jwt"              # JSON Web Tokens for stateless auth
    API_KEY = "api_key"      # API keys for service-to-service
    OAUTH2 = "oauth2"        # OAuth 2.0 for third-party auth
    BASIC = "basic"          # HTTP Basic Authentication
    BEARER = "bearer"        # Bearer token authentication
    SESSION = "session"      # Session-based authentication


class Role(Enum):
    """
    User roles for RBAC system.
    
    Implements hierarchical role structure where higher roles
    inherit permissions from lower roles.
    """
    ADMIN = "admin"          # Full system access
    MODERATOR = "moderator"  # Content moderation capabilities
    USER = "user"            # Standard user access
    SERVICE = "service"      # Service account for API access
    READONLY = "readonly"    # Read-only access
    GUEST = "guest"          # Limited guest access


@dataclass
class AuthToken:
    """
    Authentication token representation.
    
    Encapsulates token information including type, expiration,
    associated user, roles, and permitted scopes.
    
    Attributes:
        token: The token string
        type: Type of authentication
        expires_at: Token expiration timestamp
        user_id: Associated user identifier
        username: User's username
        roles: List of user roles
        scopes: Set of permitted scopes
        metadata: Additional token metadata
        issued_at: Token issuance timestamp
        issuer: Token issuer identifier
        jti: JWT ID for token revocation
    """
    token: str
    type: AuthType
    expires_at: Optional[datetime] = None
    user_id: Optional[str] = None
    username: Optional[str] = None
    roles: List[Role] = field(default_factory=list)
    scopes: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    issuer: Optional[str] = None
    jti: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Generate JTI if not provided
        if self.jti is None:
            self.jti = secrets.token_hex(16)
        
        # Set default expiration if not provided
        if self.expires_at is None and self.type == AuthType.JWT:
            self.expires_at = self.issued_at + timedelta(hours=1)
    
    def is_expired(self) -> bool:
        """
        Check if token has expired.
        
        Returns:
            True if token is expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def has_role(self, role: Role) -> bool:
        """
        Check if token has specific role.
        
        Args:
            role: Role to check
            
        Returns:
            True if token has the role
        """
        return role in self.roles
    
    def has_any_role(self, roles: List[Role]) -> bool:
        """
        Check if token has any of the specified roles.
        
        Args:
            roles: List of roles to check
            
        Returns:
            True if token has any of the roles
        """
        return any(role in self.roles for role in roles)
    
    def has_all_roles(self, roles: List[Role]) -> bool:
        """
        Check if token has all specified roles.
        
        Args:
            roles: List of roles to check
            
        Returns:
            True if token has all the roles
        """
        return all(role in self.roles for role in roles)
    
    def has_scope(self, scope: str) -> bool:
        """
        Check if token has specific scope.
        
        Args:
            scope: Scope to check
            
        Returns:
            True if token has the scope or wildcard scope
        """
        return scope in self.scopes or "*" in self.scopes
    
    def has_any_scope(self, scopes: Set[str]) -> bool:
        """
        Check if token has any of the specified scopes.
        
        Args:
            scopes: Set of scopes to check
            
        Returns:
            True if token has any of the scopes
        """
        return bool(self.scopes.intersection(scopes)) or "*" in self.scopes
    
    def get_time_until_expiry(self) -> Optional[timedelta]:
        """
        Get time remaining until token expiry.
        
        Returns:
            Timedelta until expiry or None if no expiration
        """
        if self.expires_at is None:
            return None
        
        remaining = self.expires_at - datetime.now(timezone.utc)
        return remaining if remaining.total_seconds() > 0 else timedelta(0)


class BaseAuthenticator(ABC):
    """
    Abstract base class for authentication mechanisms.
    
    Implements Strategy pattern allowing different authentication
    methods to be used interchangeably.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize authenticator with configuration.
        
        Args:
            config: Authentication configuration dictionary
        """
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self._initialize()
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize authenticator-specific resources."""
        pass
    
    @abstractmethod
    async def authenticate(self, credentials: Dict[str, Any]) -> AuthToken:
        """
        Authenticate user with provided credentials.
        
        Args:
            credentials: Authentication credentials
            
        Returns:
            AuthToken object with authentication information
            
        Raises:
            AuthenticationError: If authentication fails
        """
        pass
    
    @abstractmethod
    async def validate_token(self, token: str) -> AuthToken:
        """
        Validate authentication token.
        
        Args:
            token: Authentication token string
            
        Returns:
            AuthToken object with token information
            
        Raises:
            AuthenticationError: If token is invalid
        """
        pass
    
    @abstractmethod
    async def revoke_token(self, token: str) -> bool:
        """
        Revoke an authentication token.
        
        Args:
            token: Token to revoke
            
        Returns:
            True if revoked successfully
        """
        pass
    
    async def refresh_token(self, refresh_token: str) -> AuthToken:
        """
        Refresh an authentication token.
        
        Args:
            refresh_token: Refresh token string
            
        Returns:
            New AuthToken object
            
        Raises:
            AuthenticationError: If refresh fails
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support token refresh"
        )


class JWTAuthenticator(BaseAuthenticator):
    """
    JWT-based authentication implementing RFC 7519.
    
    Provides secure stateless authentication using JSON Web Tokens
    with support for access and refresh tokens, token revocation,
    and custom claims.
    """
    
    def _initialize(self) -> None:
        """Initialize JWT configuration and resources."""
        # JWT configuration
        self.secret_key = self.config.get("secret_key") or os.getenv("JWT_SECRET_KEY")
        if not self.secret_key:
            self.secret_key = secrets.token_urlsafe(32)
            self.logger.warning("Using generated JWT secret key - not suitable for production")
        
        self.algorithm = self.config.get("algorithm", "HS256")
        self.access_token_expiry = self.config.get("access_token_expiry_minutes", 15)
        self.refresh_token_expiry = self.config.get("refresh_token_expiry_days", 7)
        self.issuer = self.config.get("issuer", "ag-news-api")
        self.audience = self.config.get("audience", "ag-news-client")
        
        # Token storage for revocation (use Redis in production)
        self._revoked_tokens: Set[str] = set()
        
        # User database (mock - replace with actual database)
        self._users = self._initialize_mock_users()
    
    def _initialize_mock_users(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize mock user database for development.
        
        Returns:
            Dictionary of mock users
        """
        return {
            "admin": {
                "user_id": "1",
                "username": "admin",
                "email": "admin@agnews.com",
                "password_hash": pwd_context.hash("admin123"),
                "roles": [Role.ADMIN],
                "scopes": {"*"},  # All scopes
                "active": True
            },
            "user": {
                "user_id": "2",
                "username": "user",
                "email": "user@agnews.com",
                "password_hash": pwd_context.hash("user123"),
                "roles": [Role.USER],
                "scopes": {"read", "write"},
                "active": True
            },
            "service": {
                "user_id": "3",
                "username": "service",
                "email": "service@agnews.com",
                "password_hash": pwd_context.hash("service123"),
                "roles": [Role.SERVICE],
                "scopes": {"read", "write", "admin"},
                "active": True
            }
        }
    
    async def authenticate(self, credentials: Dict[str, Any]) -> AuthToken:
        """
        Authenticate user and generate JWT tokens.
        
        Args:
            credentials: Dictionary with username and password
            
        Returns:
            AuthToken with access token
            
        Raises:
            AuthenticationError: If authentication fails
        """
        username = credentials.get("username")
        password = credentials.get("password")
        
        if not username or not password:
            raise AuthenticationError("Username and password are required")
        
        # Verify credentials
        user = await self._verify_credentials(username, password)
        
        if not user:
            # Log failed attempt
            self.logger.warning(f"Failed authentication attempt for user: {username}")
            raise AuthenticationError("Invalid username or password")
        
        # Check if user is active
        if not user.get("active", True):
            raise AuthenticationError("User account is disabled")
        
        # Generate access token
        access_token = await self._generate_access_token(user)
        
        # Log successful authentication
        self.logger.info(f"User authenticated successfully: {username}")
        
        return access_token
    
    async def validate_token(self, token: str) -> AuthToken:
        """
        Validate JWT token and extract claims.
        
        Args:
            token: JWT token string
            
        Returns:
            AuthToken with token information
            
        Raises:
            AuthenticationError: If token is invalid
        """
        # Check if token is revoked
        if token in self._revoked_tokens:
            raise AuthenticationError("Token has been revoked")
        
        try:
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer,
                audience=self.audience,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True
                }
            )
            
            # Validate token type
            token_type = payload.get("type", "access")
            if token_type not in ["access", "refresh"]:
                raise AuthenticationError(f"Invalid token type: {token_type}")
            
            # Check JTI for revocation
            jti = payload.get("jti")
            if jti and jti in self._revoked_tokens:
                raise AuthenticationError("Token has been revoked")
            
            # Create AuthToken from payload
            auth_token = AuthToken(
                token=token,
                type=AuthType.JWT,
                expires_at=datetime.fromtimestamp(payload["exp"], tz=timezone.utc),
                user_id=payload["sub"],
                username=payload.get("username"),
                roles=[Role(r) for r in payload.get("roles", [])],
                scopes=set(payload.get("scopes", [])),
                issued_at=datetime.fromtimestamp(payload["iat"], tz=timezone.utc),
                issuer=payload["iss"],
                jti=jti
            )
            
            return auth_token
            
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.InvalidIssuerError:
            raise AuthenticationError("Invalid token issuer")
        except jwt.InvalidAudienceError:
            raise AuthenticationError("Invalid token audience")
        except jwt.InvalidTokenError as e:
            raise AuthenticationError(f"Invalid token: {str(e)}")
        except Exception as e:
            self.logger.error(f"Token validation error: {str(e)}")
            raise AuthenticationError("Token validation failed")
    
    async def refresh_token(self, refresh_token: str) -> AuthToken:
        """
        Generate new access token from refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New AuthToken with access token
            
        Raises:
            AuthenticationError: If refresh fails
        """
        # Validate refresh token
        token_info = await self.validate_token(refresh_token)
        
        # Verify it's a refresh token
        payload = jwt.decode(
            refresh_token,
            self.secret_key,
            algorithms=[self.algorithm],
            options={"verify_exp": False}  # Check manually for better error
        )
        
        if payload.get("type") != "refresh":
            raise AuthenticationError("Token is not a refresh token")
        
        # Check if refresh token is expired
        if token_info.is_expired():
            raise AuthenticationError("Refresh token has expired")
        
        # Get user information
        user = {
            "user_id": token_info.user_id,
            "username": token_info.username,
            "roles": token_info.roles,
            "scopes": token_info.scopes
        }
        
        # Generate new access token
        new_token = await self._generate_access_token(user)
        
        self.logger.info(f"Token refreshed for user: {token_info.username}")
        
        return new_token
    
    async def revoke_token(self, token: str) -> bool:
        """
        Revoke a JWT token.
        
        Args:
            token: Token to revoke
            
        Returns:
            True if revoked successfully
        """
        try:
            # Extract JTI from token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False}
            )
            
            jti = payload.get("jti")
            if jti:
                self._revoked_tokens.add(jti)
                self._revoked_tokens.add(token)  # Also store full token
                
                self.logger.info(f"Token revoked: JTI={jti}")
                return True
            
            # Fallback to revoking full token
            self._revoked_tokens.add(token)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to revoke token: {str(e)}")
            return False
    
    async def _verify_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Verify user credentials against database.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User dictionary if valid, None otherwise
        """
        # Get user from database (mock implementation)
        user = self._users.get(username)
        
        if not user:
            return None
        
        # Verify password
        if not pwd_context.verify(password, user["password_hash"]):
            return None
        
        return user
    
    async def _generate_access_token(self, user: Dict[str, Any]) -> AuthToken:
        """
        Generate JWT access token for user.
        
        Args:
            user: User information dictionary
            
        Returns:
            AuthToken with access token
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(minutes=self.access_token_expiry)
        jti = secrets.token_hex(16)
        
        # Build token payload
        payload = {
            "sub": str(user["user_id"]),
            "username": user.get("username"),
            "email": user.get("email"),
            "roles": [role.value if isinstance(role, Role) else role 
                     for role in user.get("roles", [])],
            "scopes": list(user.get("scopes", [])),
            "type": "access",
            "iat": now,
            "exp": expires_at,
            "iss": self.issuer,
            "aud": self.audience,
            "jti": jti
        }
        
        # Generate token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        # Create AuthToken object
        return AuthToken(
            token=token,
            type=AuthType.JWT,
            expires_at=expires_at,
            user_id=str(user["user_id"]),
            username=user.get("username"),
            roles=user.get("roles", []),
            scopes=user.get("scopes", set()),
            issued_at=now,
            issuer=self.issuer,
            jti=jti
        )
    
    async def _generate_refresh_token(self, user: Dict[str, Any]) -> str:
        """
        Generate JWT refresh token for user.
        
        Args:
            user: User information dictionary
            
        Returns:
            Refresh token string
        """
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(days=self.refresh_token_expiry)
        jti = secrets.token_hex(16)
        
        # Build refresh token payload (minimal claims)
        payload = {
            "sub": str(user["user_id"]),
            "username": user.get("username"),
            "type": "refresh",
            "iat": now,
            "exp": expires_at,
            "iss": self.issuer,
            "aud": self.audience,
            "jti": jti
        }
        
        # Generate token
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        return token
