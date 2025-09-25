"""
API Versioning Implementation
================================================================================
This module implements API versioning strategies for maintaining backward
compatibility while evolving the API, following semantic versioning principles
and REST API versioning best practices.

The implementation supports multiple versioning strategies including URL path
versioning, header versioning, and query parameter versioning.

References:
    - Semantic Versioning 2.0.0. https://semver.org/
    - Tilkov, S. (2011). REST API Design - Resource Modeling
    - Masse, M. (2011). REST API Design Rulebook

Author: Võ Hải Dũng
License: MIT
"""

import re
from typing import Dict, Any, Optional, Callable, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
import logging

from fastapi import Request, HTTPException, Header, Query
from fastapi.routing import APIRouter

logger = logging.getLogger(__name__)


@dataclass
class APIVersion:
    """
    Representation of an API version.
    
    Attributes:
        major: Major version number
        minor: Minor version number
        patch: Patch version number
        release_date: Version release date
        deprecated: Whether version is deprecated
        sunset_date: Date when version will be removed
        changes: List of changes in this version
    """
    major: int
    minor: int
    patch: int
    release_date: datetime
    deprecated: bool = False
    sunset_date: Optional[datetime] = None
    changes: List[str] = None
    
    def __post_init__(self):
        """Initialize changes list if not provided."""
        if self.changes is None:
            self.changes = []
            
    def __str__(self) -> str:
        """String representation of version."""
        return f"v{self.major}.{self.minor}.{self.patch}"
        
    def __lt__(self, other: 'APIVersion') -> bool:
        """Compare versions for ordering."""
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)
        
    def is_compatible_with(self, other: 'APIVersion') -> bool:
        """
        Check if this version is compatible with another version.
        
        Args:
            other: Version to check compatibility with
            
        Returns:
            True if versions are compatible
        """
        # Same major version indicates compatibility
        return self.major == other.major
        
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert version to dictionary representation.
        
        Returns:
            Dictionary with version information
        """
        return {
            'version': str(self),
            'major': self.major,
            'minor': self.minor,
            'patch': self.patch,
            'release_date': self.release_date.isoformat(),
            'deprecated': self.deprecated,
            'sunset_date': self.sunset_date.isoformat() if self.sunset_date else None,
            'changes': self.changes
        }


class VersionManager:
    """
    Manages API versions and version negotiation.
    
    This class handles version registration, negotiation, and compatibility
    checking for the API.
    """
    
    def __init__(self):
        """Initialize version manager."""
        self.versions: Dict[str, APIVersion] = {}
        self.current_version: Optional[APIVersion] = None
        self.default_version: Optional[APIVersion] = None
        self.version_handlers: Dict[str, Dict[str, Callable]] = {}
        
    def register_version(
        self,
        version: APIVersion,
        is_current: bool = False,
        is_default: bool = False
    ) -> None:
        """
        Register a new API version.
        
        Args:
            version: Version to register
            is_current: Whether this is the current version
            is_default: Whether this is the default version
        """
        version_str = str(version)
        self.versions[version_str] = version
        
        if is_current:
            self.current_version = version
        if is_default:
            self.default_version = version
            
        logger.info(f"Registered API version {version_str}")
        
    def get_version(self, version_str: str) -> Optional[APIVersion]:
        """
        Get version by string representation.
        
        Args:
            version_str: Version string (e.g., 'v1.2.3')
            
        Returns:
            APIVersion object or None if not found
        """
        return self.versions.get(version_str)
        
    def parse_version(self, version_str: str) -> Optional[APIVersion]:
        """
        Parse version string to APIVersion object.
        
        Args:
            version_str: Version string to parse
            
        Returns:
            Parsed APIVersion or None if invalid
        """
        # Support multiple formats: v1, v1.2, v1.2.3
        patterns = [
            r'^v?(\d+)$',  # v1 or 1
            r'^v?(\d+)\.(\d+)$',  # v1.2 or 1.2
            r'^v?(\d+)\.(\d+)\.(\d+)$'  # v1.2.3 or 1.2.3
        ]
        
        for pattern in patterns:
            match = re.match(pattern, version_str)
            if match:
                groups = match.groups()
                major = int(groups[0])
                minor = int(groups[1]) if len(groups) > 1 else 0
                patch = int(groups[2]) if len(groups) > 2 else 0
                
                # Find matching registered version
                for v in self.versions.values():
                    if v.major == major and v.minor == minor and v.patch == patch:
                        return v
                        
        return None
        
    def negotiate_version(
        self,
        requested_version: Optional[str],
        accept_header: Optional[str] = None
    ) -> APIVersion:
        """
        Negotiate API version based on request.
        
        Args:
            requested_version: Explicitly requested version
            accept_header: Accept header with version info
            
        Returns:
            Negotiated API version
            
        Raises:
            HTTPException: If version negotiation fails
        """
        # Priority: explicit version > accept header > default
        if requested_version:
            version = self.parse_version(requested_version)
            if version:
                if version.deprecated:
                    logger.warning(f"Using deprecated version {version}")
                return version
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid version: {requested_version}"
                )
                
        # Try to extract version from accept header
        if accept_header:
            version = self._extract_version_from_header(accept_header)
            if version:
                return version
                
        # Return default version
        if self.default_version:
            return self.default_version
            
        # Fallback to current version
        if self.current_version:
            return self.current_version
            
        raise HTTPException(
            status_code=500,
            detail="No API version available"
        )
        
    def _extract_version_from_header(self, accept_header: str) -> Optional[APIVersion]:
        """
        Extract version from Accept header.
        
        Args:
            accept_header: Accept header value
            
        Returns:
            Extracted version or None
        """
        # Format: application/vnd.api+json; version=1.2.3
        pattern = r'version=([^\s;]+)'
        match = re.search(pattern, accept_header)
        if match:
            return self.parse_version(match.group(1))
        return None
        
    def check_deprecation(self, version: APIVersion) -> Optional[Dict[str, Any]]:
        """
        Check if version is deprecated and return deprecation info.
        
        Args:
            version: Version to check
            
        Returns:
            Deprecation information or None
        """
        if version.deprecated:
            info = {
                'deprecated': True,
                'message': f"API version {version} is deprecated"
            }
            if version.sunset_date:
                info['sunset_date'] = version.sunset_date.isoformat()
                info['message'] += f" and will be removed on {version.sunset_date.date()}"
            return info
        return None
        
    def get_available_versions(self) -> List[Dict[str, Any]]:
        """
        Get list of all available API versions.
        
        Returns:
            List of version information dictionaries
        """
        versions = []
        for version in sorted(self.versions.values()):
            version_info = version.to_dict()
            version_info['is_current'] = version == self.current_version
            version_info['is_default'] = version == self.default_version
            versions.append(version_info)
        return versions


class VersionedRouter:
    """
    Router with versioning support.
    
    This class extends FastAPI router to support versioned endpoints
    with automatic version negotiation and compatibility checking.
    """
    
    def __init__(self, version_manager: VersionManager):
        """
        Initialize versioned router.
        
        Args:
            version_manager: Version manager instance
        """
        self.version_manager = version_manager
        self.routers: Dict[str, APIRouter] = {}
        
    def create_router(self, version: APIVersion) -> APIRouter:
        """
        Create a router for specific version.
        
        Args:
            version: API version
            
        Returns:
            FastAPI router for the version
        """
        version_str = str(version)
        if version_str not in self.routers:
            router = APIRouter(prefix=f"/{version_str}")
            self.routers[version_str] = router
        return self.routers[version_str]
        
    def version_route(
        self,
        versions: List[str],
        path: str,
        methods: List[str] = ["GET"]
    ) -> Callable:
        """
        Decorator for versioned routes.
        
        Args:
            versions: List of supported versions
            path: Route path
            methods: HTTP methods
            
        Returns:
            Route decorator
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(request: Request, *args, **kwargs):
                # Get requested version
                version_str = request.path_params.get('version')
                if not version_str:
                    version_str = request.headers.get('API-Version')
                    
                version = self.version_manager.negotiate_version(version_str)
                
                # Check if endpoint supports this version
                if str(version) not in versions:
                    raise HTTPException(
                        status_code=404,
                        detail=f"Endpoint not available in version {version}"
                    )
                    
                # Add version to request state
                request.state.api_version = version
                
                # Check deprecation
                deprecation_info = self.version_manager.check_deprecation(version)
                if deprecation_info:
                    # Add deprecation warning to response headers
                    kwargs['_deprecation_warning'] = deprecation_info
                    
                return await func(request, *args, **kwargs)
                
            # Register route for each version
            for version_str in versions:
                version = self.version_manager.parse_version(version_str)
                if version:
                    router = self.create_router(version)
                    for method in methods:
                        router.add_api_route(
                            path,
                            wrapper,
                            methods=[method],
                            name=f"{func.__name__}_{version_str}"
                        )
                        
            return wrapper
        return decorator


class VersionMiddleware:
    """
    Middleware for API versioning.
    
    This middleware handles version negotiation, adds version headers to
    responses, and enforces version policies.
    """
    
    def __init__(self, app, version_manager: VersionManager):
        """
        Initialize version middleware.
        
        Args:
            app: FastAPI application
            version_manager: Version manager instance
        """
        self.app = app
        self.version_manager = version_manager
        
    async def __call__(self, request: Request, call_next):
        """
        Process request with version handling.
        
        Args:
            request: HTTP request
            call_next: Next middleware or handler
            
        Returns:
            HTTP response with version headers
        """
        # Extract version from request
        version_str = None
        
        # Check URL path
        path_match = re.match(r'^/v(\d+(?:\.\d+)?(?:\.\d+)?)', request.url.path)
        if path_match:
            version_str = f"v{path_match.group(1)}"
            
        # Check header
        if not version_str:
            version_str = request.headers.get('API-Version')
            
        # Check query parameter
        if not version_str:
            version_str = request.query_params.get('version')
            
        # Negotiate version
        try:
            version = self.version_manager.negotiate_version(
                version_str,
                request.headers.get('Accept')
            )
            request.state.api_version = version
        except HTTPException as e:
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=e.status_code,
                content={'detail': e.detail}
            )
            
        # Process request
        response = await call_next(request)
        
        # Add version headers to response
        response.headers['API-Version'] = str(version)
        response.headers['X-API-Version'] = str(version)
        
        # Add deprecation warning if applicable
        deprecation_info = self.version_manager.check_deprecation(version)
        if deprecation_info:
            response.headers['Deprecation'] = 'true'
            if version.sunset_date:
                response.headers['Sunset'] = version.sunset_date.isoformat()
                
        # Add available versions header
        available_versions = [str(v) for v in self.version_manager.versions.values()]
        response.headers['X-Available-Versions'] = ', '.join(available_versions)
        
        return response


# Initialize version manager
version_manager = VersionManager()

# Register API versions
version_manager.register_version(
    APIVersion(
        major=1, minor=0, patch=0,
        release_date=datetime(2024, 1, 1),
        changes=['Initial release']
    ),
    is_default=True
)

version_manager.register_version(
    APIVersion(
        major=1, minor=1, patch=0,
        release_date=datetime(2024, 3, 1),
        changes=['Added batch processing', 'Improved performance']
    )
)

version_manager.register_version(
    APIVersion(
        major=2, minor=0, patch=0,
        release_date=datetime(2024, 6, 1),
        changes=['New API structure', 'Breaking changes in response format'],
    ),
    is_current=True
)
