"""
Cache Service Implementation for AG News Text Classification
================================================================================
This module implements a unified caching service that manages multiple cache
backends and strategies for optimal performance.

The cache service provides:
- Unified interface for multiple cache backends
- Automatic cache key generation
- TTL management
- Cache statistics and monitoring

References:
    - Tanenbaum, A. S., & Van Steen, M. (2017). Distributed Systems
    - High Performance Browser Networking (O'Reilly)

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import hashlib
import json
from typing import Any, Optional, Dict, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import pickle

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class CacheBackend(Enum):
    """
    Available cache backend types.
    
    Types:
        MEMORY: In-memory cache
        REDIS: Redis distributed cache
        HYBRID: Combined memory and Redis
    """
    MEMORY = "memory"
    REDIS = "redis"
    HYBRID = "hybrid"


@dataclass
class CacheEntry:
    """
    Cache entry with metadata.
    
    Attributes:
        key: Cache key
        value: Cached value
        created_at: Creation timestamp
        expires_at: Expiration timestamp
        hit_count: Number of cache hits
        size_bytes: Size of cached value
        tags: Tags for grouping cache entries
    """
    key: str
    value: Any
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    hit_count: int = 0
    size_bytes: int = 0
    tags: List[str] = field(default_factory=list)
    
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() > self.expires_at
    
    def increment_hit(self) -> None:
        """Increment hit counter."""
        self.hit_count += 1


@dataclass
class CacheStats:
    """
    Cache statistics for monitoring.
    
    Attributes:
        total_entries: Total number of cache entries
        total_size_bytes: Total cache size in bytes
        hit_rate: Cache hit rate percentage
        miss_rate: Cache miss rate percentage
        eviction_count: Number of evicted entries
        avg_entry_size: Average entry size in bytes
    """
    total_entries: int = 0
    total_size_bytes: int = 0
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    eviction_count: int = 0
    avg_entry_size: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            "total_entries": self.total_entries,
            "total_size_bytes": self.total_size_bytes,
            "total_size_mb": self.total_size_bytes / (1024 * 1024),
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "eviction_count": self.eviction_count,
            "avg_entry_size_bytes": self.avg_entry_size
        }


class CacheService(BaseService):
    """
    Unified cache service managing multiple cache backends.
    
    This service provides a consistent interface for caching operations
    across different backends with automatic fallback and monitoring.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        backend: CacheBackend = CacheBackend.MEMORY,
        max_size_mb: int = 1024,
        default_ttl_seconds: int = 3600
    ):
        """
        Initialize cache service.
        
        Args:
            config: Service configuration
            backend: Cache backend type
            max_size_mb: Maximum cache size in MB
            default_ttl_seconds: Default TTL for cache entries
        """
        if config is None:
            config = ServiceConfig(name="cache_service")
        super().__init__(config)
        
        self.backend = backend
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.default_ttl = timedelta(seconds=default_ttl_seconds)
        
        # Cache backends
        self._memory_cache: Optional[Any] = None
        self._redis_cache: Optional[Any] = None
        
        # Statistics tracking
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        
        # Cache invalidation callbacks
        self._invalidation_callbacks: List[Callable] = []
        
        self.logger = get_logger("service.cache")
    
    async def _initialize(self) -> None:
        """Initialize cache backends."""
        self.logger.info(f"Initializing cache service with backend: {self.backend.value}")
        
        if self.backend in [CacheBackend.MEMORY, CacheBackend.HYBRID]:
            from src.services.caching.memory_cache import MemoryCache
            self._memory_cache = MemoryCache(
                max_size_bytes=self.max_size_bytes,
                default_ttl=self.default_ttl
            )
            await self._memory_cache.initialize()
        
        if self.backend in [CacheBackend.REDIS, CacheBackend.HYBRID]:
            from src.services.caching.redis_cache import RedisCache
            self._redis_cache = RedisCache(
                default_ttl=self.default_ttl
            )
            await self._redis_cache.initialize()
    
    async def _start(self) -> None:
        """Start cache service."""
        self.logger.info("Cache service started")
    
    async def _stop(self) -> None:
        """Stop cache service."""
        if self._memory_cache:
            await self._memory_cache.cleanup()
        if self._redis_cache:
            await self._redis_cache.cleanup()
        self.logger.info("Cache service stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup cache resources."""
        await self.clear()
    
    async def _check_health(self) -> bool:
        """Check cache service health."""
        try:
            # Test cache operations
            test_key = "__health_check__"
            await self.set(test_key, "healthy", ttl_seconds=10)
            value = await self.get(test_key)
            await self.delete(test_key)
            return value == "healthy"
        except Exception:
            return False
    
    def generate_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            str: Generated cache key
        """
        # Create a unique key from arguments
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        
        # Hash for consistent key length
        key_hash = hashlib.md5(key_str.encode()).hexdigest()
        
        return f"cache:{key_hash}"
    
    async def get(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Any: Cached value or default
        """
        try:
            # Try memory cache first (fastest)
            if self._memory_cache:
                value = await self._memory_cache.get(key)
                if value is not None:
                    self._hit_count += 1
                    self.logger.debug(f"Memory cache hit: {key}")
                    return value
            
            # Try Redis cache (distributed)
            if self._redis_cache:
                value = await self._redis_cache.get(key)
                if value is not None:
                    self._hit_count += 1
                    self.logger.debug(f"Redis cache hit: {key}")
                    
                    # Populate memory cache for faster access
                    if self._memory_cache and self.backend == CacheBackend.HYBRID:
                        await self._memory_cache.set(key, value)
                    
                    return value
            
            self._miss_count += 1
            self.logger.debug(f"Cache miss: {key}")
            return default
            
        except Exception as e:
            self.logger.error(f"Cache get error: {e}")
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            tags: Tags for grouping
            
        Returns:
            bool: True if successful
        """
        try:
            ttl = timedelta(seconds=ttl_seconds) if ttl_seconds else self.default_ttl
            
            # Set in all configured backends
            success = True
            
            if self._memory_cache:
                success = success and await self._memory_cache.set(
                    key, value, ttl, tags
                )
            
            if self._redis_cache:
                success = success and await self._redis_cache.set(
                    key, value, ttl, tags
                )
            
            if success:
                self.logger.debug(f"Cache set: {key}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        try:
            success = True
            
            if self._memory_cache:
                success = success and await self._memory_cache.delete(key)
            
            if self._redis_cache:
                success = success and await self._redis_cache.delete(key)
            
            if success:
                self.logger.debug(f"Cache delete: {key}")
                
                # Trigger invalidation callbacks
                for callback in self._invalidation_callbacks:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(key)
                    else:
                        callback(key)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Cache delete error: {e}")
            return False
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        """
        Delete all entries with specified tags.
        
        Args:
            tags: Tags to match
            
        Returns:
            int: Number of deleted entries
        """
        deleted_count = 0
        
        if self._memory_cache:
            deleted_count += await self._memory_cache.delete_by_tags(tags)
        
        if self._redis_cache:
            deleted_count += await self._redis_cache.delete_by_tags(tags)
        
        self.logger.info(f"Deleted {deleted_count} entries with tags: {tags}")
        return deleted_count
    
    async def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            bool: True if successful
        """
        try:
            success = True
            
            if self._memory_cache:
                success = success and await self._memory_cache.clear()
            
            if self._redis_cache:
                success = success and await self._redis_cache.clear()
            
            self.logger.info("Cache cleared")
            return success
            
        except Exception as e:
            self.logger.error(f"Cache clear error: {e}")
            return False
    
    async def get_stats(self) -> CacheStats:
        """
        Get cache statistics.
        
        Returns:
            CacheStats: Cache statistics
        """
        stats = CacheStats()
        
        # Aggregate stats from all backends
        if self._memory_cache:
            memory_stats = await self._memory_cache.get_stats()
            stats.total_entries += memory_stats.total_entries
            stats.total_size_bytes += memory_stats.total_size_bytes
        
        if self._redis_cache:
            redis_stats = await self._redis_cache.get_stats()
            stats.total_entries += redis_stats.total_entries
            stats.total_size_bytes += redis_stats.total_size_bytes
        
        # Calculate rates
        total_requests = self._hit_count + self._miss_count
        if total_requests > 0:
            stats.hit_rate = (self._hit_count / total_requests) * 100
            stats.miss_rate = (self._miss_count / total_requests) * 100
        
        stats.eviction_count = self._eviction_count
        
        if stats.total_entries > 0:
            stats.avg_entry_size = stats.total_size_bytes / stats.total_entries
        
        return stats
    
    def cached(
        self,
        ttl_seconds: Optional[int] = None,
        key_prefix: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> Callable:
        """
        Decorator for caching function results.
        
        Args:
            ttl_seconds: Cache TTL
            key_prefix: Prefix for cache key
            tags: Tags for cache entry
            
        Returns:
            Callable: Decorated function
            
        Usage:
            @cache_service.cached(ttl_seconds=3600)
            async def expensive_function(param1, param2):
                # Expensive computation
                return result
        """
        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                # Generate cache key
                cache_key = self.generate_key(
                    func.__name__,
                    *args,
                    **kwargs
                )
                
                if key_prefix:
                    cache_key = f"{key_prefix}:{cache_key}"
                
                # Try to get from cache
                cached_value = await self.get(cache_key)
                if cached_value is not None:
                    return cached_value
                
                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                await self.set(cache_key, result, ttl_seconds, tags)
                
                return result
            
            return wrapper
        
        return decorator
    
    def add_invalidation_callback(self, callback: Callable) -> None:
        """
        Add cache invalidation callback.
        
        Args:
            callback: Function to call on cache invalidation
        """
        self._invalidation_callbacks.append(callback)
        self.logger.debug(f"Added invalidation callback: {callback.__name__}")
