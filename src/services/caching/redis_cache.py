"""
Redis Cache Implementation for AG News Text Classification
================================================================================
This module implements a distributed cache using Redis, providing scalable
caching across multiple service instances.

The Redis cache provides:
- Distributed caching with Redis backend
- Atomic operations and transactions
- Pub/Sub support for cache invalidation
- Cluster support for horizontal scaling

References:
    - Carlson, J. L. (2013). Redis in Action
    - Redis Documentation: https://redis.io/documentation

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import json
import pickle
from typing import Any, Optional, Dict, List, Set
from datetime import datetime, timedelta
import redis.asyncio as redis
from redis.asyncio.client import Pipeline
from redis.exceptions import RedisError, ConnectionError

from src.utils.logging_config import get_logger


class RedisCache:
    """
    Redis-based distributed cache implementation.
    
    This cache provides distributed caching capabilities using Redis,
    supporting both standalone and cluster deployments.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: timedelta = timedelta(hours=1),
        key_prefix: str = "agnews:",
        max_connections: int = 50,
        cluster_mode: bool = False
    ):
        """
        Initialize Redis cache.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            default_ttl: Default time-to-live for entries
            key_prefix: Prefix for all cache keys
            max_connections: Maximum number of connections
            cluster_mode: Enable cluster mode
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.default_ttl = default_ttl
        self.key_prefix = key_prefix
        self.max_connections = max_connections
        self.cluster_mode = cluster_mode
        
        # Redis client
        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[redis.client.PubSub] = None
        
        # Connection pool
        self._pool: Optional[redis.ConnectionPool] = None
        
        # Statistics
        self._hit_count = 0
        self._miss_count = 0
        
        # Invalidation channels
        self._invalidation_channel = f"{key_prefix}invalidation"
        self._invalidation_handlers = []
        
        self.logger = get_logger("cache.redis")
    
    async def initialize(self) -> None:
        """Initialize Redis connection."""
        try:
            # Create connection pool
            self._pool = redis.ConnectionPool(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                max_connections=self.max_connections,
                decode_responses=False  # Handle binary data
            )
            
            # Create Redis client
            self._client = redis.Redis(connection_pool=self._pool)
            
            # Test connection
            await self._client.ping()
            
            # Setup pub/sub for cache invalidation
            self._pubsub = self._client.pubsub()
            await self._pubsub.subscribe(self._invalidation_channel)
            
            # Start invalidation listener
            asyncio.create_task(self._invalidation_listener())
            
            self.logger.info(f"Redis cache connected to {self.host}:{self.port}")
            
        except (RedisError, ConnectionError) as e:
            self.logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Cleanup Redis connections."""
        try:
            if self._pubsub:
                await self._pubsub.unsubscribe(self._invalidation_channel)
                await self._pubsub.close()
            
            if self._client:
                await self._client.close()
            
            if self._pool:
                await self._pool.disconnect()
            
            self.logger.info("Redis cache cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def get(self, key: str) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Any: Cached value or None
        """
        if not self._client:
            return None
        
        full_key = self._make_key(key)
        
        try:
            # Get value
            value = await self._client.get(full_key)
            
            if value is None:
                self._miss_count += 1
                return None
            
            # Update access time for LRU-like behavior
            await self._client.expire(full_key, self.default_ttl.total_seconds())
            
            # Deserialize
            try:
                result = pickle.loads(value)
                self._hit_count += 1
                return result
            except (pickle.PickleError, TypeError):
                # Try JSON deserialization as fallback
                try:
                    result = json.loads(value)
                    self._hit_count += 1
                    return result
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to deserialize value for key: {key}")
                    return None
                    
        except RedisError as e:
            self.logger.error(f"Redis get error: {e}")
            return None
    
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
        if not self._client:
            return False
        
        full_key = self._make_key(key)
        ttl_seconds = int((ttl or self.default_ttl).total_seconds())
        
        try:
            # Serialize value
            try:
                serialized = pickle.dumps(value)
            except (pickle.PickleError, TypeError):
                # Fallback to JSON for simple types
                try:
                    serialized = json.dumps(value).encode()
                except (TypeError, ValueError):
                    self.logger.error(f"Failed to serialize value for key: {key}")
                    return False
            
            # Use pipeline for atomic operations
            async with self._client.pipeline(transaction=True) as pipe:
                # Set value with TTL
                pipe.setex(full_key, ttl_seconds, serialized)
                
                # Handle tags
                if tags:
                    for tag in tags:
                        tag_key = self._make_tag_key(tag)
                        pipe.sadd(tag_key, full_key)
                        pipe.expire(tag_key, ttl_seconds * 2)  # Keep tag index longer
                
                # Store metadata
                metadata_key = self._make_metadata_key(key)
                metadata = {
                    "created_at": datetime.now().isoformat(),
                    "ttl": ttl_seconds,
                    "tags": tags or [],
                    "size": len(serialized)
                }
                pipe.hset(metadata_key, mapping={
                    k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                    for k, v in metadata.items()
                })
                pipe.expire(metadata_key, ttl_seconds)
                
                # Execute pipeline
                await pipe.execute()
                
            return True
            
        except RedisError as e:
            self.logger.error(f"Redis set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if successful
        """
        if not self._client:
            return False
        
        full_key = self._make_key(key)
        
        try:
            # Get metadata for tag cleanup
            metadata_key = self._make_metadata_key(key)
            metadata = await self._client.hgetall(metadata_key)
            
            # Use pipeline for atomic deletion
            async with self._client.pipeline(transaction=True) as pipe:
                # Delete main key
                pipe.delete(full_key)
                
                # Delete metadata
                pipe.delete(metadata_key)
                
                # Remove from tag sets
                if metadata and b"tags" in metadata:
                    tags = json.loads(metadata[b"tags"])
                    for tag in tags:
                        tag_key = self._make_tag_key(tag)
                        pipe.srem(tag_key, full_key)
                
                result = await pipe.execute()
            
            # Publish invalidation event
            await self._publish_invalidation(key)
            
            return result[0] > 0  # Check if main key was deleted
            
        except RedisError as e:
            self.logger.error(f"Redis delete error: {e}")
            return False
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        """
        Delete all entries with specified tags.
        
        Args:
            tags: Tags to match
            
        Returns:
            int: Number of deleted entries
        """
        if not self._client:
            return 0
        
        deleted_count = 0
        
        try:
            # Get all keys with specified tags
            keys_to_delete = set()
            
            for tag in tags:
                tag_key = self._make_tag_key(tag)
                members = await self._client.smembers(tag_key)
                keys_to_delete.update(members)
            
            if keys_to_delete:
                # Delete in batches to avoid blocking
                batch_size = 100
                keys_list = list(keys_to_delete)
                
                for i in range(0, len(keys_list), batch_size):
                    batch = keys_list[i:i + batch_size]
                    
                    # Delete batch
                    async with self._client.pipeline(transaction=False) as pipe:
                        for full_key in batch:
                            pipe.delete(full_key)
                            
                            # Extract original key for metadata
                            key = full_key.decode().replace(self.key_prefix, "", 1)
                            metadata_key = self._make_metadata_key(key)
                            pipe.delete(metadata_key)
                        
                        results = await pipe.execute()
                        deleted_count += sum(1 for r in results[::2] if r > 0)
                
                # Clean up tag sets
                for tag in tags:
                    tag_key = self._make_tag_key(tag)
                    await self._client.delete(tag_key)
            
            return deleted_count
            
        except RedisError as e:
            self.logger.error(f"Redis delete_by_tags error: {e}")
            return 0
    
    async def clear(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            bool: True if successful
        """
        if not self._client:
            return False
        
        try:
            # Use SCAN to find all keys with our prefix
            cursor = 0
            pattern = f"{self.key_prefix}*"
            
            while True:
                cursor, keys = await self._client.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                
                if keys:
                    await self._client.delete(*keys)
                
                if cursor == 0:
                    break
            
            self.logger.info("Redis cache cleared")
            return True
            
        except RedisError as e:
            self.logger.error(f"Redis clear error: {e}")
            return False
    
    async def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        if not self._client:
            return {}
        
        try:
            # Get Redis info
            info = await self._client.info("memory")
            
            # Count keys with our prefix
            key_count = 0
            cursor = 0
            pattern = f"{self.key_prefix}*"
            
            while True:
                cursor, keys = await self._client.scan(
                    cursor,
                    match=pattern,
                    count=100
                )
                key_count += len(keys)
                
                if cursor == 0:
                    break
            
            # Calculate hit rate
            total_requests = self._hit_count + self._miss_count
            hit_rate = (self._hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "total_entries": key_count,
                "total_size_bytes": info.get("used_memory", 0),
                "hit_count": self._hit_count,
                "miss_count": self._miss_count,
                "hit_rate": hit_rate,
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B")
            }
            
        except RedisError as e:
            self.logger.error(f"Redis get_stats error: {e}")
            return {}
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if exists
        """
        if not self._client:
            return False
        
        full_key = self._make_key(key)
        
        try:
            return await self._client.exists(full_key) > 0
        except RedisError:
            return False
    
    async def expire(self, key: str, ttl: timedelta) -> bool:
        """
        Update TTL for existing key.
        
        Args:
            key: Cache key
            ttl: New time-to-live
            
        Returns:
            bool: True if successful
        """
        if not self._client:
            return False
        
        full_key = self._make_key(key)
        ttl_seconds = int(ttl.total_seconds())
        
        try:
            return await self._client.expire(full_key, ttl_seconds)
        except RedisError:
            return False
    
    async def _invalidation_listener(self) -> None:
        """Listen for cache invalidation events."""
        if not self._pubsub:
            return
        
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    key = message["data"].decode() if isinstance(message["data"], bytes) else message["data"]
                    
                    # Trigger invalidation handlers
                    for handler in self._invalidation_handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(key)
                            else:
                                handler(key)
                        except Exception as e:
                            self.logger.error(f"Invalidation handler error: {e}")
                            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Invalidation listener error: {e}")
    
    async def _publish_invalidation(self, key: str) -> None:
        """
        Publish cache invalidation event.
        
        Args:
            key: Invalidated key
        """
        if not self._client:
            return
        
        try:
            await self._client.publish(self._invalidation_channel, key)
        except RedisError as e:
            self.logger.error(f"Failed to publish invalidation: {e}")
    
    def add_invalidation_handler(self, handler: callable) -> None:
        """
        Add cache invalidation handler.
        
        Args:
            handler: Handler function
        """
        self._invalidation_handlers.append(handler)
    
    def _make_key(self, key: str) -> str:
        """
        Create full cache key with prefix.
        
        Args:
            key: Original key
            
        Returns:
            str: Full key with prefix
        """
        return f"{self.key_prefix}{key}"
    
    def _make_tag_key(self, tag: str) -> str:
        """
        Create tag set key.
        
        Args:
            tag: Tag name
            
        Returns:
            str: Tag set key
        """
        return f"{self.key_prefix}tag:{tag}"
    
    def _make_metadata_key(self, key: str) -> str:
        """
        Create metadata key.
        
        Args:
            key: Original key
            
        Returns:
            str: Metadata key
        """
        return f"{self.key_prefix}meta:{key}"
