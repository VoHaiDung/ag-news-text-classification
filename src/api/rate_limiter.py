"""
Rate Limiting Module for API
================================================================================
Implements advanced rate limiting strategies for API endpoints with support
for distributed rate limiting, dynamic adjustment, and multiple algorithms.

This module provides comprehensive rate limiting capabilities following
industry best practices for API throttling and fair resource allocation.

References:
    - Cormen, T. H., et al. (2009). Introduction to Algorithms (Token Bucket)
    - Redis Labs (2021). Rate Limiting Patterns
    - CloudFlare (2017). How We Built Rate Limiting Capable of Scaling to Millions

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import redis.asyncio as redis

from src.core.exceptions import RateLimitExceeded
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RateLimitStrategy(Enum):
    """
    Rate limiting strategies supported by the system.
    
    Each strategy has different characteristics suitable for various use cases.
    """
    FIXED_WINDOW = "fixed_window"          # Simple, memory efficient
    SLIDING_WINDOW = "sliding_window"      # More accurate, higher memory
    TOKEN_BUCKET = "token_bucket"          # Allows bursts
    LEAKY_BUCKET = "leaky_bucket"          # Smooth rate
    ADAPTIVE = "adaptive"                  # Dynamic adjustment
    DISTRIBUTED = "distributed"             # Redis-based for scaling


@dataclass
class RateLimitConfig:
    """
    Configuration for rate limiting.
    
    Attributes:
        requests_per_second: Maximum requests per second
        requests_per_minute: Maximum requests per minute  
        requests_per_hour: Maximum requests per hour
        requests_per_day: Maximum requests per day
        burst_size: Maximum burst size for token bucket
        strategy: Rate limiting strategy to use
        redis_url: Redis URL for distributed rate limiting
        enable_adaptive: Enable adaptive rate limiting
        custom_limits: Custom limits per endpoint or user
    """
    requests_per_second: Optional[int] = None
    requests_per_minute: Optional[int] = 60
    requests_per_hour: Optional[int] = 1000
    requests_per_day: Optional[int] = 10000
    burst_size: int = 10
    strategy: RateLimitStrategy = RateLimitStrategy.TOKEN_BUCKET
    redis_url: Optional[str] = None
    enable_adaptive: bool = False
    custom_limits: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    def get_limit_for_window(self, window: str) -> Optional[int]:
        """
        Get rate limit for specific time window.
        
        Args:
            window: Time window (second, minute, hour, day)
            
        Returns:
            Rate limit for the window
        """
        limits = {
            "second": self.requests_per_second,
            "minute": self.requests_per_minute,
            "hour": self.requests_per_hour,
            "day": self.requests_per_day
        }
        return limits.get(window)


@dataclass
class RateLimitInfo:
    """
    Rate limit information for responses.
    
    Attributes:
        limit: Rate limit threshold
        remaining: Remaining requests
        reset: Reset timestamp
        retry_after: Seconds until retry allowed
    """
    limit: int
    remaining: int
    reset: datetime
    retry_after: Optional[int] = None
    
    def to_headers(self) -> Dict[str, str]:
        """
        Convert to HTTP headers.
        
        Returns:
            Dictionary of rate limit headers
        """
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
            "X-RateLimit-Reset": str(int(self.reset.timestamp()))
        }
        
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        
        return headers


class BaseRateLimiter:
    """
    Base class for rate limiting implementations.
    
    Provides common interface and utilities for different rate limiting strategies.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    async def check_rate_limit(
        self,
        identifier: str,
        resource: str = "default"
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check if request is within rate limits.
        
        Args:
            identifier: Unique identifier (user_id, IP, API key)
            resource: Resource being accessed
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        raise NotImplementedError
    
    async def consume(
        self,
        identifier: str,
        tokens: int = 1,
        resource: str = "default"
    ) -> None:
        """
        Consume rate limit tokens.
        
        Args:
            identifier: Unique identifier
            tokens: Number of tokens to consume
            resource: Resource being accessed
            
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        allowed, info = await self.check_rate_limit(identifier, resource)
        
        if not allowed:
            self.logger.warning(
                f"Rate limit exceeded for {identifier} on {resource}",
                extra={"retry_after": info.retry_after}
            )
            raise RateLimitExceeded(
                f"Rate limit exceeded. Retry after {info.retry_after} seconds",
                retry_after=info.retry_after,
                headers=info.to_headers()
            )
    
    async def reset(self, identifier: str, resource: str = "default") -> None:
        """
        Reset rate limit for identifier.
        
        Args:
            identifier: Unique identifier
            resource: Resource being accessed
        """
        raise NotImplementedError
    
    def _get_custom_limit(
        self,
        identifier: str,
        resource: str
    ) -> Optional[RateLimitConfig]:
        """
        Get custom rate limit if defined.
        
        Args:
            identifier: Unique identifier
            resource: Resource being accessed
            
        Returns:
            Custom rate limit config or None
        """
        # Check user-specific limits
        if identifier in self.config.custom_limits:
            user_limits = self.config.custom_limits[identifier]
            if resource in user_limits:
                return RateLimitConfig(**user_limits[resource])
        
        # Check resource-specific limits
        if resource in self.config.custom_limits:
            return RateLimitConfig(**self.config.custom_limits[resource])
        
        return None


class TokenBucketRateLimiter(BaseRateLimiter):
    """
    Token bucket rate limiting implementation.
    
    Allows burst traffic while maintaining average rate limits.
    Algorithm based on network traffic shaping techniques.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize token bucket rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        super().__init__(config)
        
        # Determine refill rate and bucket size
        if config.requests_per_second:
            self.refill_rate = config.requests_per_second
            self.refill_period = 1.0
        elif config.requests_per_minute:
            self.refill_rate = config.requests_per_minute / 60.0
            self.refill_period = 1.0
        elif config.requests_per_hour:
            self.refill_rate = config.requests_per_hour / 3600.0
            self.refill_period = 1.0
        else:
            self.refill_rate = 10  # Default
            self.refill_period = 1.0
        
        self.bucket_size = config.burst_size or int(self.refill_rate * 2)
        
        # Storage for buckets
        self._buckets: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(
        self,
        identifier: str,
        resource: str = "default"
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check rate limit using token bucket algorithm.
        
        Args:
            identifier: Unique identifier
            resource: Resource being accessed
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        async with self._lock:
            current_time = time.time()
            bucket_key = f"{identifier}:{resource}"
            
            # Initialize bucket if not exists
            if bucket_key not in self._buckets:
                self._buckets[bucket_key] = {
                    "tokens": self.bucket_size,
                    "last_refill": current_time
                }
            
            bucket = self._buckets[bucket_key]
            
            # Refill tokens based on elapsed time
            elapsed = current_time - bucket["last_refill"]
            tokens_to_add = elapsed * self.refill_rate
            bucket["tokens"] = min(self.bucket_size, bucket["tokens"] + tokens_to_add)
            bucket["last_refill"] = current_time
            
            # Check if tokens available
            allowed = bucket["tokens"] >= 1
            
            if allowed:
                bucket["tokens"] -= 1
                remaining = int(bucket["tokens"])
            else:
                remaining = 0
            
            # Calculate reset time
            if not allowed:
                retry_after = int((1 - bucket["tokens"]) / self.refill_rate) + 1
                reset_time = datetime.now(timezone.utc) + timedelta(seconds=retry_after)
            else:
                retry_after = None
                reset_time = datetime.now(timezone.utc) + timedelta(
                    seconds=int((self.bucket_size - bucket["tokens"]) / self.refill_rate)
                )
            
            return allowed, RateLimitInfo(
                limit=self.bucket_size,
                remaining=remaining,
                reset=reset_time,
                retry_after=retry_after
            )
    
    async def reset(self, identifier: str, resource: str = "default") -> None:
        """
        Reset rate limit for identifier.
        
        Args:
            identifier: Unique identifier
            resource: Resource being accessed
        """
        async with self._lock:
            bucket_key = f"{identifier}:{resource}"
            if bucket_key in self._buckets:
                del self._buckets[bucket_key]


class SlidingWindowRateLimiter(BaseRateLimiter):
    """
    Sliding window rate limiting implementation.
    
    Provides more accurate rate limiting than fixed window approach
    by tracking individual request timestamps.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize sliding window rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        super().__init__(config)
        
        # Determine window size and limit
        if config.requests_per_second:
            self.window_size = 1
            self.limit = config.requests_per_second
        elif config.requests_per_minute:
            self.window_size = 60
            self.limit = config.requests_per_minute
        elif config.requests_per_hour:
            self.window_size = 3600
            self.limit = config.requests_per_hour
        else:
            self.window_size = 60
            self.limit = 60
        
        # Storage for request timestamps
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._lock = asyncio.Lock()
    
    async def check_rate_limit(
        self,
        identifier: str,
        resource: str = "default"
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check rate limit using sliding window algorithm.
        
        Args:
            identifier: Unique identifier
            resource: Resource being accessed
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        async with self._lock:
            current_time = time.time()
            window_key = f"{identifier}:{resource}"
            
            # Get or create timestamp deque
            timestamps = self._windows[window_key]
            
            # Remove timestamps outside the window
            cutoff = current_time - self.window_size
            while timestamps and timestamps[0] < cutoff:
                timestamps.popleft()
            
            # Check if under limit
            allowed = len(timestamps) < self.limit
            
            if allowed:
                timestamps.append(current_time)
                remaining = self.limit - len(timestamps)
            else:
                remaining = 0
            
            # Calculate reset time and retry after
            if not allowed and timestamps:
                oldest_in_window = timestamps[0]
                retry_after = int(self.window_size - (current_time - oldest_in_window)) + 1
                reset_time = datetime.now(timezone.utc) + timedelta(seconds=retry_after)
            else:
                retry_after = None
                reset_time = datetime.now(timezone.utc) + timedelta(seconds=self.window_size)
            
            return allowed, RateLimitInfo(
                limit=self.limit,
                remaining=remaining,
                reset=reset_time,
                retry_after=retry_after
            )
    
    async def reset(self, identifier: str, resource: str = "default") -> None:
        """
        Reset rate limit for identifier.
        
        Args:
            identifier: Unique identifier
            resource: Resource being accessed
        """
        async with self._lock:
            window_key = f"{identifier}:{resource}"
            if window_key in self._windows:
                self._windows[window_key].clear()


class DistributedRateLimiter(BaseRateLimiter):
    """
    Distributed rate limiting using Redis.
    
    Provides consistent rate limiting across multiple application instances
    using Redis as a centralized state store.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize distributed rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        super().__init__(config)
        
        if not config.redis_url:
            raise ValueError("Redis URL required for distributed rate limiting")
        
        self.redis_client = None
        self.redis_url = config.redis_url
        
        # Use token bucket algorithm in Redis
        if config.requests_per_minute:
            self.limit = config.requests_per_minute
            self.window = 60
        elif config.requests_per_hour:
            self.limit = config.requests_per_hour
            self.window = 3600
        else:
            self.limit = 60
            self.window = 60
    
    async def _ensure_connected(self):
        """Ensure Redis client is connected."""
        if self.redis_client is None:
            self.redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
    
    async def check_rate_limit(
        self,
        identifier: str,
        resource: str = "default"
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check rate limit using Redis.
        
        Uses Redis INCR with TTL for atomic rate limiting.
        
        Args:
            identifier: Unique identifier
            resource: Resource being accessed
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        await self._ensure_connected()
        
        key = f"rate_limit:{identifier}:{resource}"
        
        try:
            # Atomic increment with TTL
            pipe = self.redis_client.pipeline()
            pipe.incr(key)
            pipe.ttl(key)
            
            results = await pipe.execute()
            current_count = results[0]
            ttl = results[1]
            
            # Set TTL if new key
            if ttl == -1:
                await self.redis_client.expire(key, self.window)
                ttl = self.window
            
            allowed = current_count <= self.limit
            remaining = max(0, self.limit - current_count)
            
            # Calculate reset time
            reset_time = datetime.now(timezone.utc) + timedelta(seconds=ttl)
            retry_after = ttl if not allowed else None
            
            # Decrement if not allowed (to not consume the token)
            if not allowed:
                await self.redis_client.decr(key)
            
            return allowed, RateLimitInfo(
                limit=self.limit,
                remaining=remaining,
                reset=reset_time,
                retry_after=retry_after
            )
            
        except Exception as e:
            self.logger.error(f"Redis rate limit error: {str(e)}")
            # Fail open - allow request if Redis is down
            return True, RateLimitInfo(
                limit=self.limit,
                remaining=self.limit,
                reset=datetime.now(timezone.utc) + timedelta(seconds=self.window)
            )
    
    async def reset(self, identifier: str, resource: str = "default") -> None:
        """
        Reset rate limit for identifier.
        
        Args:
            identifier: Unique identifier
            resource: Resource being accessed
        """
        await self._ensure_connected()
        
        key = f"rate_limit:{identifier}:{resource}"
        await self.redis_client.delete(key)
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


class AdaptiveRateLimiter(BaseRateLimiter):
    """
    Adaptive rate limiting that adjusts based on system load.
    
    Dynamically adjusts rate limits based on system metrics like
    CPU usage, memory, response times, and error rates.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize adaptive rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        super().__init__(config)
        
        # Base limiter (token bucket)
        self.base_limiter = TokenBucketRateLimiter(config)
        
        # Adaptive parameters
        self.load_factor = 1.0
        self.min_factor = 0.1   # Minimum 10% of normal rate
        self.max_factor = 2.0   # Maximum 200% of normal rate
        
        # Metrics for adaptation
        self.response_times = deque(maxlen=100)
        self.error_rates = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=10)
        
        # Update interval
        self.last_update = time.time()
        self.update_interval = 10  # seconds
    
    async def check_rate_limit(
        self,
        identifier: str,
        resource: str = "default"
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check rate limit with adaptive adjustment.
        
        Args:
            identifier: Unique identifier
            resource: Resource being accessed
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        # Update load factor periodically
        current_time = time.time()
        if current_time - self.last_update > self.update_interval:
            await self._update_load_factor()
            self.last_update = current_time
        
        # Apply adjusted rate limit
        original_config = self.base_limiter.config
        
        # Create adjusted config
        adjusted_config = RateLimitConfig(
            requests_per_second=int(
                (original_config.requests_per_second or 0) * self.load_factor
            ) if original_config.requests_per_second else None,
            requests_per_minute=int(
                (original_config.requests_per_minute or 0) * self.load_factor
            ) if original_config.requests_per_minute else None,
            requests_per_hour=int(
                (original_config.requests_per_hour or 0) * self.load_factor
            ) if original_config.requests_per_hour else None,
            burst_size=int(original_config.burst_size * self.load_factor),
            strategy=original_config.strategy
        )
        
        # Temporarily update config
        self.base_limiter.config = adjusted_config
        allowed, info = await self.base_limiter.check_rate_limit(identifier, resource)
        self.base_limiter.config = original_config
        
        # Add adaptive metadata
        info.limit = int(info.limit * self.load_factor)
        
        return allowed, info
    
    async def _update_load_factor(self):
        """Update load factor based on system metrics."""
        try:
            # Get system metrics (simplified)
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage.append(cpu_percent)
            
            # Calculate adjustments
            adjustments = []
            
            # Adjust based on CPU
            if self.cpu_usage:
                avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage)
                if avg_cpu > 80:
                    adjustments.append(0.5)  # Reduce by 50%
                elif avg_cpu > 60:
                    adjustments.append(0.75)  # Reduce by 25%
                elif avg_cpu < 30:
                    adjustments.append(1.5)  # Increase by 50%
                else:
                    adjustments.append(1.0)
            
            # Adjust based on response times
            if self.response_times:
                avg_response = sum(self.response_times) / len(self.response_times)
                if avg_response > 1.0:  # Over 1 second
                    adjustments.append(0.7)
                elif avg_response < 0.1:  # Under 100ms
                    adjustments.append(1.3)
                else:
                    adjustments.append(1.0)
            
            # Adjust based on error rate
            if self.error_rates:
                error_rate = sum(self.error_rates) / len(self.error_rates)
                if error_rate > 0.05:  # Over 5% errors
                    adjustments.append(0.6)
                elif error_rate < 0.01:  # Under 1% errors
                    adjustments.append(1.2)
                else:
                    adjustments.append(1.0)
            
            # Calculate new load factor
            if adjustments:
                new_factor = sum(adjustments) / len(adjustments)
                # Smooth the transition
                self.load_factor = 0.7 * self.load_factor + 0.3 * new_factor
                # Apply bounds
                self.load_factor = max(
                    self.min_factor,
                    min(self.max_factor, self.load_factor)
                )
            
            self.logger.debug(f"Adaptive rate limit factor: {self.load_factor:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to update load factor: {str(e)}")
    
    def record_response(self, response_time: float, is_error: bool):
        """
        Record response metrics for adaptation.
        
        Args:
            response_time: Response time in seconds
            is_error: Whether response was an error
        """
        self.response_times.append(response_time)
        self.error_rates.append(1.0 if is_error else 0.0)
    
    async def reset(self, identifier: str, resource: str = "default") -> None:
        """
        Reset rate limit for identifier.
        
        Args:
            identifier: Unique identifier
            resource: Resource being accessed
        """
        await self.base_limiter.reset(identifier, resource)


class RateLimitManager:
    """
    Central rate limit manager for coordinating multiple limiters.
    
    Provides unified interface for rate limiting with support for
    different strategies, custom limits, and monitoring.
    """
    
    def __init__(self, default_config: RateLimitConfig = None):
        """
        Initialize rate limit manager.
        
        Args:
            default_config: Default rate limit configuration
        """
        self.default_config = default_config or RateLimitConfig()
        self.limiters: Dict[str, BaseRateLimiter] = {}
        self.endpoint_configs: Dict[str, RateLimitConfig] = {}
        self.user_configs: Dict[str, RateLimitConfig] = {}
        self.logger = get_logger(__name__)
        
        # Create default limiter
        self._create_default_limiter()
    
    def _create_default_limiter(self):
        """Create default rate limiter based on strategy."""
        strategy = self.default_config.strategy
        
        if strategy == RateLimitStrategy.TOKEN_BUCKET:
            self.default_limiter = TokenBucketRateLimiter(self.default_config)
        elif strategy == RateLimitStrategy.SLIDING_WINDOW:
            self.default_limiter = SlidingWindowRateLimiter(self.default_config)
        elif strategy == RateLimitStrategy.DISTRIBUTED:
            self.default_limiter = DistributedRateLimiter(self.default_config)
        elif strategy == RateLimitStrategy.ADAPTIVE:
            self.default_limiter = AdaptiveRateLimiter(self.default_config)
        else:
            self.default_limiter = TokenBucketRateLimiter(self.default_config)
    
    def get_limiter(
        self,
        endpoint: str = None,
        user_id: str = None
    ) -> BaseRateLimiter:
        """
        Get appropriate rate limiter.
        
        Args:
            endpoint: API endpoint
            user_id: User identifier
            
        Returns:
            Rate limiter instance
        """
        # Check for user-specific limiter
        if user_id and user_id in self.limiters:
            return self.limiters[user_id]
        
        # Check for endpoint-specific limiter
        if endpoint and endpoint in self.limiters:
            return self.limiters[endpoint]
        
        # Return default limiter
        return self.default_limiter
    
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str = None,
        user_id: str = None
    ) -> Tuple[bool, RateLimitInfo]:
        """
        Check rate limit for request.
        
        Args:
            identifier: Request identifier
            endpoint: API endpoint
            user_id: User identifier
            
        Returns:
            Tuple of (allowed, rate_limit_info)
        """
        limiter = self.get_limiter(endpoint, user_id)
        resource = endpoint or "default"
        
        return await limiter.check_rate_limit(identifier, resource)
    
    async def consume(
        self,
        identifier: str,
        endpoint: str = None,
        user_id: str = None,
        tokens: int = 1
    ) -> None:
        """
        Consume rate limit tokens.
        
        Args:
            identifier: Request identifier
            endpoint: API endpoint
            user_id: User identifier
            tokens: Number of tokens to consume
            
        Raises:
            RateLimitExceeded: If rate limit exceeded
        """
        limiter = self.get_limiter(endpoint, user_id)
        resource = endpoint or "default"
        
        await limiter.consume(identifier, tokens, resource)
    
    def set_endpoint_config(
        self,
        endpoint: str,
        config: RateLimitConfig
    ):
        """
        Set custom configuration for endpoint.
        
        Args:
            endpoint: API endpoint
            config: Rate limit configuration
        """
        self.endpoint_configs[endpoint] = config
        
        # Create limiter for endpoint
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            limiter = TokenBucketRateLimiter(config)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            limiter = SlidingWindowRateLimiter(config)
        elif config.strategy == RateLimitStrategy.DISTRIBUTED:
            limiter = DistributedRateLimiter(config)
        elif config.strategy == RateLimitStrategy.ADAPTIVE:
            limiter = AdaptiveRateLimiter(config)
        else:
            limiter = TokenBucketRateLimiter(config)
        
        self.limiters[endpoint] = limiter
        
        self.logger.info(f"Custom rate limit set for endpoint: {endpoint}")
    
    def set_user_config(
        self,
        user_id: str,
        config: RateLimitConfig
    ):
        """
        Set custom configuration for user.
        
        Args:
            user_id: User identifier
            config: Rate limit configuration
        """
        self.user_configs[user_id] = config
        
        # Create limiter for user
        if config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            limiter = TokenBucketRateLimiter(config)
        elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            limiter = SlidingWindowRateLimiter(config)
        elif config.strategy == RateLimitStrategy.DISTRIBUTED:
            limiter = DistributedRateLimiter(config)
        elif config.strategy == RateLimitStrategy.ADAPTIVE:
            limiter = AdaptiveRateLimiter(config)
        else:
            limiter = TokenBucketRateLimiter(config)
        
        self.limiters[user_id] = limiter
        
        self.logger.info(f"Custom rate limit set for user: {user_id}")
    
    async def reset(
        self,
        identifier: str,
        endpoint: str = None,
        user_id: str = None
    ):
        """
        Reset rate limit.
        
        Args:
            identifier: Request identifier
            endpoint: API endpoint
            user_id: User identifier
        """
        limiter = self.get_limiter(endpoint, user_id)
        resource = endpoint or "default"
        
        await limiter.reset(identifier, resource)
        
        self.logger.info(f"Rate limit reset for {identifier} on {resource}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get rate limiting statistics.
        
        Returns:
            Dictionary of statistics
        """
        stats = {
            "default_config": {
                "strategy": self.default_config.strategy.value,
                "limits": {
                    "per_second": self.default_config.requests_per_second,
                    "per_minute": self.default_config.requests_per_minute,
                    "per_hour": self.default_config.requests_per_hour,
                    "per_day": self.default_config.requests_per_day
                }
            },
            "custom_endpoints": len(self.endpoint_configs),
            "custom_users": len(self.user_configs),
            "active_limiters": len(self.limiters)
        }
        
        # Add adaptive stats if using adaptive limiter
        if isinstance(self.default_limiter, AdaptiveRateLimiter):
            stats["adaptive"] = {
                "load_factor": self.default_limiter.load_factor,
                "min_factor": self.default_limiter.min_factor,
                "max_factor": self.default_limiter.max_factor
            }
        
        return stats
