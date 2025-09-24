"""
Memory Cache Implementation for AG News Text Classification
================================================================================
This module implements an in-memory cache with LRU eviction and TTL support,
optimized for fast local caching.

The memory cache provides:
- LRU eviction strategy
- TTL-based expiration
- Size-based limits
- Tag-based organization

References:
    - Knuth, D. E. (1997). The Art of Computer Programming, Volume 1
    - Python collections.OrderedDict documentation

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import sys
import pickle
from typing import Any, Optional, Dict, List, Set
from datetime import datetime, timedelta
from collections import OrderedDict
import threading

from src.services.caching.cache_strategies import LRUStrategy
from src.utils.logging_config import get_logger


class MemoryCache:
    """
    In-memory cache implementation with LRU eviction.
    
    This cache provides fast local storage with automatic eviction
    based on LRU strategy and size limits.
    """
    
    def __init__(
        self,
        max_size_bytes: int = 1024 * 1024 * 1024,  # 1GB default
        default_ttl: timedelta = timedelta(hours=1),
        eviction_ratio: float = 0.1
    ):
        """
        Initialize memory cache.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
            default_ttl: Default time-to-live for entries
            eviction_ratio: Fraction of cache to evict when full
        """
        self.max_size_bytes = max_size_bytes
        self.default_ttl = default_ttl
        self.eviction_ratio = eviction_ratio
        
        # Cache storage
        self._cache: OrderedDict = OrderedDict()
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._tags_index: Dict[str, Set[str]] = {}
        
        # Size tracking
        self._current_size = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self._hit_count = 0
        self._miss_count = 0
        self._eviction_count = 0
        
        self.logger = get_logger("cache.memory")
    
    async def initialize(self) -> None:
        """Initialize memory cache."""
        self.logger.info(
            f"Memory cache initialized with max size: "
            f"{self.max_size_bytes / (1024 * 1024):.2f} MB"
        )
    
    async def cleanup(self) -> None:
        """Cleanup memory cache resources."""
        with self._lock:
            self._cache.clear()
            self._metadata.clear()
            self._tags_index.clear()
            self._current_size = 0
        
        self.logger.info("Memory cache cleaned up")
    
    async def get(self, key: str) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None
        """
        with self._lock:
            if key not in self._cache:
                self._miss_count += 1
                return None
            
            # Check expiration
            metadata = self._metadata.get(key, {})
            if metadata.get("expires_at"):
                if datetime.now() > metadata["expires_at"]:
                    # Expired, remove it
                    await self._remove_entry(key)
                    self._miss_count += 1
                    return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            # Update hit count
            self._hit_count += 1
            metadata["hit_count"] = metadata.get("hit_count", 0) + 1
            metadata["last_accessed"] = datetime.now()
            
            return self._cache[key]
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live
            tags: Tags for grouping
            
        Returns:
            bool: True if successful
        """
        try:
            # Calculate size
            size = self._calculate_size(value)
            
            # Check if value is too large
            if size > self.max_size_bytes * 0.5:  # Don't allow single items > 50% of cache
                self.logger.warning(f"Value too large for cache: {size} bytes")
                return False
            
            with self._lock:
                # Check if we need to evict
                if self._current_size + size > self.max_size_bytes:
                    await self._evict_entries(size)
                
                # Remove old entry if exists
                if key in self._cache:
                    await self._remove_entry(key)
                
                # Add new entry
                self._cache[key] = value
                self._cache.move_to_end(key)
                
                # Update metadata
                expires_at = None
                if ttl or self.default_ttl:
                    expires_at = datetime.now() + (ttl or self.default_ttl)
                
                self._metadata[key] = {
                    "size": size,
                    "created_at": datetime.now(),
                    "expires_at": expires_at,
                    "tags": tags or [],
                    "hit_count": 0
                }
                
                # Update tags index
                if tags:
                    for tag in tags:
                        if tag not in self._tags_index:
                            self._tags_index[tag] = set()
                        self._tags_index[tag].add(key)
                
                # Update size
                self._current_size += size
                
                return True
                
        except Exception as e:
            self.logger.error(f"Error setting cache value: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                return True
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
        
        with self._lock:
            keys_to_delete = set()
            
            for tag in tags:
                if tag in self._tags_index:
                    keys_to_delete.update(self._tags_index[tag])
            
            for key in keys_to_delete:
                if await self._remove_entry(key):
                    deleted_count += 1
        
        return deleted_count
    
    async def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            bool: True if successful
        """
        with self._lock:
            self._cache.clear()
            self._metadata.clear()
            self._tags_index.clear()
            self._current_size = 0
            
        self.logger.info("Cache cleared")
        return True
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self._lock:
            total_requests = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "total_entries": len(self._cache),
                "total_size_bytes": self._current_size,
                "max_size_bytes": self.max_size_bytes,
                "usage_percent": (self._current_size / self.max_size_bytes * 100),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "eviction_count": self._eviction_count
            }
    
    async def _remove_entry(self, key: str) -> bool:
        """
        Remove entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if removed
        """
        if key not in self._cache:
            return False
        
        # Remove from cache
        del self._cache[key]
        
        # Update size
        metadata = self._metadata.get(key, {})
        self._current_size -= metadata.get("size", 0)
        
        # Remove from tags index
        for tag in metadata.get("tags", []):
            if tag in self._tags_index:
                self._tags_index[tag].discard(key)
                if not self._tags_index[tag]:
                    del self._tags_index[tag]
        
        # Remove metadata
        if key in self._metadata:
            del self._metadata[key]
        
        return True
    
    async def _evict_entries(self, required_size: int) -> None:
        """
        Evict entries to make room for new data.
        
        Args:
            required_size: Size needed in bytes
        """
        # Calculate how much to evict
        target_size = self.max_size_bytes - required_size
        evict_size = max(
            self._current_size - target_size,
            int(self.max_size_bytes * self.eviction_ratio)
        )
        
        evicted_size = 0
        keys_to_evict = []
        
        # Select keys to evict (LRU order)
        for key in self._cache:
            if evicted_size >= evict_size:
                break
            
            keys_to_evict.append(key)
            metadata = self._metadata.get(key, {})
            evicted_size += metadata.get("size", 0)
        
        # Evict selected keys
        for key in keys_to_evict:
            await self._remove_entry(key)
            self._eviction_count += 1
        
        self.logger.debug(
            f"Evicted {len(keys_to_evict)} entries "
            f"({evicted_size / 1024:.2f} KB)"
        )
    
    def _calculate_size(self, value: Any) -> int:
        """
        Calculate approximate size of value in bytes.
        
        Args:
            value: Value to measure
            
        Returns:
            int: Size in bytes
        """
        try:
            # Try to pickle and measure
            return len(pickle.dumps(value))
        except Exception:
            # Fallback to sys.getsizeof
            return sys.getsizeof(value)
