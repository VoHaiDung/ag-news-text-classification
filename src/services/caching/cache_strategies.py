"""
Cache Strategies Implementation for AG News Text Classification
================================================================================
This module implements various cache eviction and management strategies,
providing configurable policies for cache behavior.

The strategies include:
- LRU (Least Recently Used)
- LFU (Least Frequently Used)
- TTL (Time-To-Live) based
- FIFO (First In First Out)

References:
    - Sleator, D. D., & Tarjan, R. E. (1985). Amortized efficiency of list update and paging rules
    - Podlipnig, S., & Böszörmenyi, L. (2003). A survey of web cache replacement strategies

Author: Võ Hải Dũng
License: MIT
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import OrderedDict, defaultdict
import heapq

from src.utils.logging_config import get_logger


class CacheStrategy(ABC):
    """
    Abstract base class for cache eviction strategies.
    
    This class defines the interface for implementing different
    cache eviction policies.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize cache strategy.
        
        Args:
            max_size: Maximum number of cache entries
        """
        self.max_size = max_size
        self.logger = get_logger(f"cache.strategy.{self.__class__.__name__}")
    
    @abstractmethod
    def should_evict(self, current_size: int) -> bool:
        """
        Determine if eviction is needed.
        
        Args:
            current_size: Current cache size
            
        Returns:
            bool: True if eviction needed
        """
        pass
    
    @abstractmethod
    def select_victim(self, cache_entries: Dict[str, Any]) -> Optional[str]:
        """
        Select entry to evict.
        
        Args:
            cache_entries: Current cache entries
            
        Returns:
            Optional[str]: Key to evict or None
        """
        pass
    
    @abstractmethod
    def on_access(self, key: str) -> None:
        """
        Update strategy state on cache access.
        
        Args:
            key: Accessed key
        """
        pass
    
    @abstractmethod
    def on_insert(self, key: str, metadata: Dict[str, Any]) -> None:
        """
        Update strategy state on cache insertion.
        
        Args:
            key: Inserted key
            metadata: Entry metadata
        """
        pass
    
    @abstractmethod
    def on_evict(self, key: str) -> None:
        """
        Update strategy state on cache eviction.
        
        Args:
            key: Evicted key
        """
        pass


class LRUStrategy(CacheStrategy):
    """
    Least Recently Used (LRU) eviction strategy.
    
    This strategy evicts the least recently accessed items first,
    maintaining an access order for all cache entries.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize LRU strategy.
        
        Args:
            max_size: Maximum number of cache entries
        """
        super().__init__(max_size)
        self._access_order = OrderedDict()
        self._access_times = {}
    
    def should_evict(self, current_size: int) -> bool:
        """
        Check if eviction is needed based on size.
        
        Args:
            current_size: Current cache size
            
        Returns:
            bool: True if current_size >= max_size
        """
        return current_size >= self.max_size
    
    def select_victim(self, cache_entries: Dict[str, Any]) -> Optional[str]:
        """
        Select least recently used entry.
        
        Args:
            cache_entries: Current cache entries
            
        Returns:
            Optional[str]: LRU key or None
        """
        if not self._access_order:
            # If no access history, select any key
            return next(iter(cache_entries)) if cache_entries else None
        
        # Return least recently used (first in OrderedDict)
        lru_key = next(iter(self._access_order))
        self.logger.debug(f"Selected LRU victim: {lru_key}")
        return lru_key
    
    def on_access(self, key: str) -> None:
        """
        Move accessed key to end (most recently used).
        
        Args:
            key: Accessed key
        """
        if key in self._access_order:
            self._access_order.move_to_end(key)
        else:
            self._access_order[key] = True
        
        self._access_times[key] = datetime.now()
    
    def on_insert(self, key: str, metadata: Dict[str, Any]) -> None:
        """
        Add new key as most recently used.
        
        Args:
            key: Inserted key
            metadata: Entry metadata
        """
        self._access_order[key] = True
        self._access_times[key] = datetime.now()
    
    def on_evict(self, key: str) -> None:
        """
        Remove evicted key from tracking.
        
        Args:
            key: Evicted key
        """
        self._access_order.pop(key, None)
        self._access_times.pop(key, None)


class LFUStrategy(CacheStrategy):
    """
    Least Frequently Used (LFU) eviction strategy.
    
    This strategy evicts the least frequently accessed items,
    maintaining access frequency counters for all entries.
    """
    
    def __init__(self, max_size: int = 1000, decay_factor: float = 0.99):
        """
        Initialize LFU strategy.
        
        Args:
            max_size: Maximum number of cache entries
            decay_factor: Factor for frequency decay over time
        """
        super().__init__(max_size)
        self._frequencies = defaultdict(int)
        self._last_access = {}
        self._min_heap = []  # (frequency, key) pairs
        self.decay_factor = decay_factor
    
    def should_evict(self, current_size: int) -> bool:
        """
        Check if eviction is needed based on size.
        
        Args:
            current_size: Current cache size
            
        Returns:
            bool: True if current_size >= max_size
        """
        return current_size >= self.max_size
    
    def select_victim(self, cache_entries: Dict[str, Any]) -> Optional[str]:
        """
        Select least frequently used entry.
        
        Args:
            cache_entries: Current cache entries
            
        Returns:
            Optional[str]: LFU key or None
        """
        # Rebuild heap if necessary
        self._rebuild_heap()
        
        # Find valid victim (key that still exists in cache)
        while self._min_heap:
            freq, key = heapq.heappop(self._min_heap)
            if key in cache_entries:
                self.logger.debug(f"Selected LFU victim: {key} (frequency: {freq})")
                return key
        
        # Fallback to any key
        return next(iter(cache_entries)) if cache_entries else None
    
    def on_access(self, key: str) -> None:
        """
        Increment access frequency with decay.
        
        Args:
            key: Accessed key
        """
        # Apply time-based decay to existing frequency
        if key in self._last_access:
            time_diff = (datetime.now() - self._last_access[key]).total_seconds()
            decay_multiplier = self.decay_factor ** (time_diff / 3600)  # Decay per hour
            self._frequencies[key] = self._frequencies[key] * decay_multiplier
        
        # Increment frequency
        self._frequencies[key] += 1
        self._last_access[key] = datetime.now()
        
        # Mark heap as needing rebuild
        self._heap_dirty = True
    
    def on_insert(self, key: str, metadata: Dict[str, Any]) -> None:
        """
        Initialize frequency for new entry.
        
        Args:
            key: Inserted key
            metadata: Entry metadata
        """
        self._frequencies[key] = 1
        self._last_access[key] = datetime.now()
        heapq.heappush(self._min_heap, (1, key))
    
    def on_evict(self, key: str) -> None:
        """
        Remove evicted key from tracking.
        
        Args:
            key: Evicted key
        """
        self._frequencies.pop(key, None)
        self._last_access.pop(key, None)
        # Heap will be cleaned on next access
    
    def _rebuild_heap(self) -> None:
        """Rebuild min heap with current frequencies."""
        if hasattr(self, "_heap_dirty") and self._heap_dirty:
            self._min_heap = [(freq, key) for key, freq in self._frequencies.items()]
            heapq.heapify(self._min_heap)
            self._heap_dirty = False


class TTLStrategy(CacheStrategy):
    """
    Time-To-Live (TTL) based eviction strategy.
    
    This strategy evicts entries that have exceeded their TTL,
    regardless of access patterns.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: timedelta = timedelta(hours=1)):
        """
        Initialize TTL strategy.
        
        Args:
            max_size: Maximum number of cache entries
            default_ttl: Default time-to-live for entries
        """
        super().__init__(max_size)
        self._expiry_times = {}
        self._insertion_times = {}
        self.default_ttl = default_ttl
    
    def should_evict(self, current_size: int) -> bool:
        """
        Check if eviction is needed based on size or expiry.
        
        Args:
            current_size: Current cache size
            
        Returns:
            bool: True if size exceeded or expired entries exist
        """
        if current_size >= self.max_size:
            return True
        
        # Check for expired entries
        now = datetime.now()
        return any(expiry <= now for expiry in self._expiry_times.values())
    
    def select_victim(self, cache_entries: Dict[str, Any]) -> Optional[str]:
        """
        Select expired entry or oldest entry.
        
        Args:
            cache_entries: Current cache entries
            
        Returns:
            Optional[str]: Expired or oldest key
        """
        now = datetime.now()
        
        # First, look for expired entries
        for key, expiry in self._expiry_times.items():
            if expiry <= now and key in cache_entries:
                self.logger.debug(f"Selected expired victim: {key}")
                return key
        
        # If no expired entries, select oldest
        if self._insertion_times:
            oldest_key = min(self._insertion_times, key=self._insertion_times.get)
            if oldest_key in cache_entries:
                self.logger.debug(f"Selected oldest victim: {oldest_key}")
                return oldest_key
        
        # Fallback to any key
        return next(iter(cache_entries)) if cache_entries else None
    
    def on_access(self, key: str) -> None:
        """
        Optionally extend TTL on access.
        
        Args:
            key: Accessed key
        """
        # TTL strategy typically doesn't change on access
        # But can implement sliding TTL if needed
        pass
    
    def on_insert(self, key: str, metadata: Dict[str, Any]) -> None:
        """
        Set expiry time for new entry.
        
        Args:
            key: Inserted key
            metadata: Entry metadata with optional 'ttl' field
        """
        now = datetime.now()
        ttl = metadata.get("ttl", self.default_ttl)
        
        if isinstance(ttl, (int, float)):
            ttl = timedelta(seconds=ttl)
        
        self._insertion_times[key] = now
        self._expiry_times[key] = now + ttl
    
    def on_evict(self, key: str) -> None:
        """
        Remove evicted key from tracking.
        
        Args:
            key: Evicted key
        """
        self._expiry_times.pop(key, None)
        self._insertion_times.pop(key, None)
    
    def get_ttl(self, key: str) -> Optional[timedelta]:
        """
        Get remaining TTL for key.
        
        Args:
            key: Cache key
            
        Returns:
            Optional[timedelta]: Remaining TTL or None
        """
        if key in self._expiry_times:
            remaining = self._expiry_times[key] - datetime.now()
            return remaining if remaining.total_seconds() > 0 else timedelta(0)
        return None


class FIFOStrategy(CacheStrategy):
    """
    First In First Out (FIFO) eviction strategy.
    
    This strategy evicts the oldest entries first,
    regardless of access patterns.
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize FIFO strategy.
        
        Args:
            max_size: Maximum number of cache entries
        """
        super().__init__(max_size)
        self._insertion_order = OrderedDict()
    
    def should_evict(self, current_size: int) -> bool:
        """
        Check if eviction is needed based on size.
        
        Args:
            current_size: Current cache size
            
        Returns:
            bool: True if current_size >= max_size
        """
        return current_size >= self.max_size
    
    def select_victim(self, cache_entries: Dict[str, Any]) -> Optional[str]:
        """
        Select oldest (first inserted) entry.
        
        Args:
            cache_entries: Current cache entries
            
        Returns:
            Optional[str]: Oldest key or None
        """
        if not self._insertion_order:
            return next(iter(cache_entries)) if cache_entries else None
        
        # Return first inserted (first in OrderedDict)
        oldest_key = next(iter(self._insertion_order))
        self.logger.debug(f"Selected FIFO victim: {oldest_key}")
        return oldest_key
    
    def on_access(self, key: str) -> None:
        """
        No-op for FIFO (access doesn't affect order).
        
        Args:
            key: Accessed key
        """
        pass
    
    def on_insert(self, key: str, metadata: Dict[str, Any]) -> None:
        """
        Add new key to insertion order.
        
        Args:
            key: Inserted key
            metadata: Entry metadata
        """
        self._insertion_order[key] = datetime.now()
    
    def on_evict(self, key: str) -> None:
        """
        Remove evicted key from tracking.
        
        Args:
            key: Evicted key
        """
        self._insertion_order.pop(key, None)


class AdaptiveStrategy(CacheStrategy):
    """
    Adaptive cache strategy that switches between strategies based on workload.
    
    This strategy monitors cache patterns and dynamically selects
    the most appropriate eviction policy.
    """
    
    def __init__(
        self,
        max_size: int = 1000,
        strategies: Optional[List[CacheStrategy]] = None
    ):
        """
        Initialize adaptive strategy.
        
        Args:
            max_size: Maximum number of cache entries
            strategies: List of strategies to adapt between
        """
        super().__init__(max_size)
        
        self.strategies = strategies or [
            LRUStrategy(max_size),
            LFUStrategy(max_size),
            TTLStrategy(max_size)
        ]
        
        self.current_strategy_index = 0
        self.current_strategy = self.strategies[0]
        
        # Performance tracking
        self._hit_counts = defaultdict(int)
        self._miss_counts = defaultdict(int)
        self._evaluation_window = 1000
        self._access_count = 0
    
    def should_evict(self, current_size: int) -> bool:
        """
        Delegate to current strategy.
        
        Args:
            current_size: Current cache size
            
        Returns:
            bool: Result from current strategy
        """
        return self.current_strategy.should_evict(current_size)
    
    def select_victim(self, cache_entries: Dict[str, Any]) -> Optional[str]:
        """
        Delegate to current strategy.
        
        Args:
            cache_entries: Current cache entries
            
        Returns:
            Optional[str]: Result from current strategy
        """
        return self.current_strategy.select_victim(cache_entries)
    
    def on_access(self, key: str) -> None:
        """
        Update all strategies and evaluate performance.
        
        Args:
            key: Accessed key
        """
        # Update all strategies
        for strategy in self.strategies:
            strategy.on_access(key)
        
        # Evaluate and potentially switch strategy
        self._access_count += 1
        if self._access_count % self._evaluation_window == 0:
            self._evaluate_and_switch()
    
    def on_insert(self, key: str, metadata: Dict[str, Any]) -> None:
        """
        Update all strategies.
        
        Args:
            key: Inserted key
            metadata: Entry metadata
        """
        for strategy in self.strategies:
            strategy.on_insert(key, metadata)
    
    def on_evict(self, key: str) -> None:
        """
        Update all strategies.
        
        Args:
            key: Evicted key
        """
        for strategy in self.strategies:
            strategy.on_evict(key)
    
    def _evaluate_and_switch(self) -> None:
        """Evaluate strategies and switch if beneficial."""
        # Calculate hit rates for each strategy
        best_hit_rate = 0
        best_index = self.current_strategy_index
        
        for i, strategy in enumerate(self.strategies):
            hit_count = self._hit_counts[i]
            miss_count = self._miss_counts[i]
            total = hit_count + miss_count
            
            if total > 0:
                hit_rate = hit_count / total
                if hit_rate > best_hit_rate:
                    best_hit_rate = hit_rate
                    best_index = i
        
        # Switch if different strategy performs better
        if best_index != self.current_strategy_index:
            self.current_strategy_index = best_index
            self.current_strategy = self.strategies[best_index]
            self.logger.info(
                f"Switched to {self.current_strategy.__class__.__name__} "
                f"(hit rate: {best_hit_rate:.2%})"
            )
        
        # Reset counters
        self._hit_counts.clear()
        self._miss_counts.clear()
