"""
Caching Services Module for AG News Text Classification
================================================================================
This module implements multi-level caching strategies for improving performance
and reducing redundant computations.

The caching layer provides:
- In-memory caching
- Distributed caching with Redis
- Cache invalidation strategies
- TTL management

References:
    - Fitzpatrick, B. (2004). Distributed Caching with Memcached
    - Redis Documentation

Author: Võ Hải Dũng
License: MIT
"""

from src.services.caching.cache_service import (
    CacheService,
    CacheEntry,
    CacheStats
)
from src.services.caching.cache_strategies import (
    CacheStrategy,
    LRUStrategy,
    LFUStrategy,
    TTLStrategy
)
from src.services.caching.redis_cache import RedisCache
from src.services.caching.memory_cache import MemoryCache

__all__ = [
    "CacheService",
    "CacheEntry",
    "CacheStats",
    "CacheStrategy",
    "LRUStrategy",
    "LFUStrategy",
    "TTLStrategy",
    "RedisCache",
    "MemoryCache"
]
