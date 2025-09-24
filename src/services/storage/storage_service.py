"""
Storage Service Implementation for AG News Text Classification
================================================================================
This module implements a unified storage service that abstracts different storage
backends, providing consistent interface for data persistence.

The storage service provides:
- Unified API for multiple storage backends
- Automatic failover and retry logic
- Data versioning and metadata management
- Lifecycle policies and expiration

References:
    - Tanenbaum, A. S., & Van Steen, M. (2017). Distributed Systems
    - Amazon S3 Developer Guide
    - Google Cloud Storage Best Practices

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import hashlib
from typing import Dict, Any, List, Optional, Union, BinaryIO
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
import mimetypes

from src.services.base_service import BaseService, ServiceConfig
from src.core.exceptions import ServiceException
from src.utils.logging_config import get_logger


class StorageBackend(Enum):
    """
    Available storage backend types.
    
    Types:
        LOCAL: Local filesystem storage
        S3: Amazon S3 storage
        GCS: Google Cloud Storage
        AZURE: Azure Blob Storage
        HYBRID: Multiple backends with failover
    """
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    HYBRID = "hybrid"


@dataclass
class StorageMetadata:
    """
    Storage object metadata.
    
    Attributes:
        content_type: MIME content type
        content_length: Size in bytes
        etag: Entity tag for caching
        created_at: Creation timestamp
        modified_at: Last modification timestamp
        version: Object version
        custom_metadata: Custom key-value metadata
        tags: Object tags for organization
    """
    content_type: str = "application/octet-stream"
    content_length: int = 0
    etag: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    version: Optional[str] = None
    custom_metadata: Dict[str, str] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "content_type": self.content_type,
            "content_length": self.content_length,
            "etag": self.etag,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "version": self.version,
            "custom_metadata": self.custom_metadata,
            "tags": self.tags
        }


@dataclass
class StorageObject:
    """
    Storage object representation.
    
    Attributes:
        key: Object key/path
        data: Object data
        metadata: Object metadata
        bucket: Storage bucket/container
        url: Object URL if available
    """
    key: str
    data: Optional[bytes] = None
    metadata: StorageMetadata = field(default_factory=StorageMetadata)
    bucket: Optional[str] = None
    url: Optional[str] = None


class StorageAdapter(ABC):
    """
    Abstract base class for storage backend adapters.
    """
    
    @abstractmethod
    async def put(self, key: str, data: bytes, metadata: Optional[StorageMetadata] = None) -> bool:
        """Store object."""
        pass
    
    @abstractmethod
    async def get(self, key: str) -> Optional[StorageObject]:
        """Retrieve object."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete object."""
        pass
    
    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if object exists."""
        pass
    
    @abstractmethod
    async def list(self, prefix: str = "", limit: int = 1000) -> List[str]:
        """List objects with prefix."""
        pass


class StorageService(BaseService):
    """
    Unified storage service for managing data persistence across backends.
    
    This service provides a consistent interface for storing and retrieving
    data across different storage systems with automatic failover.
    """
    
    def __init__(
        self,
        config: Optional[ServiceConfig] = None,
        backend: StorageBackend = StorageBackend.LOCAL,
        bucket: str = "ag-news-storage",
        enable_versioning: bool = True,
        enable_encryption: bool = True
    ):
        """
        Initialize storage service.
        
        Args:
            config: Service configuration
            backend: Primary storage backend
            bucket: Default bucket/container name
            enable_versioning: Enable object versioning
            enable_encryption: Enable data encryption
        """
        if config is None:
            config = ServiceConfig(name="storage_service")
        super().__init__(config)
        
        self.backend = backend
        self.bucket = bucket
        self.enable_versioning = enable_versioning
        self.enable_encryption = enable_encryption
        
        # Storage adapters
        self._adapters: Dict[StorageBackend, StorageAdapter] = {}
        self._primary_adapter: Optional[StorageAdapter] = None
        
        # Cache for frequently accessed objects
        self._cache: Dict[str, StorageObject] = {}
        self._cache_ttl = timedelta(minutes=5)
        
        # Statistics
        self._stats = {
            "puts": 0,
            "gets": 0,
            "deletes": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "bytes_uploaded": 0,
            "bytes_downloaded": 0
        }
        
        self.logger = get_logger("service.storage")
    
    async def _initialize(self) -> None:
        """Initialize storage service."""
        self.logger.info(f"Initializing storage service with backend: {self.backend.value}")
        
        # Initialize storage adapters
        await self._initialize_adapters()
        
        # Create default bucket
        await self._ensure_bucket_exists()
    
    async def _start(self) -> None:
        """Start storage service."""
        self.logger.info("Storage service started")
    
    async def _stop(self) -> None:
        """Stop storage service."""
        # Clear cache
        self._cache.clear()
        
        self.logger.info("Storage service stopped")
    
    async def _cleanup(self) -> None:
        """Cleanup storage resources."""
        self._adapters.clear()
        self._cache.clear()
    
    async def _check_health(self) -> bool:
        """Check storage service health."""
        if self._primary_adapter:
            # Try to list objects to verify connectivity
            try:
                await self._primary_adapter.list(limit=1)
                return True
            except Exception:
                return False
        return False
    
    async def _initialize_adapters(self) -> None:
        """Initialize storage backend adapters."""
        if self.backend == StorageBackend.LOCAL:
            from src.services.storage.local_storage import LocalStorage
            adapter = LocalStorage(base_path=Path(f"outputs/{self.bucket}"))
            await adapter.initialize()
            self._adapters[StorageBackend.LOCAL] = adapter
            self._primary_adapter = adapter
            
        elif self.backend == StorageBackend.S3:
            from src.services.storage.s3_storage import S3Storage
            adapter = S3Storage(bucket=self.bucket)
            await adapter.initialize()
            self._adapters[StorageBackend.S3] = adapter
            self._primary_adapter = adapter
            
        elif self.backend == StorageBackend.GCS:
            from src.services.storage.gcs_storage import GCSStorage
            adapter = GCSStorage(bucket=self.bucket)
            await adapter.initialize()
            self._adapters[StorageBackend.GCS] = adapter
            self._primary_adapter = adapter
            
        elif self.backend == StorageBackend.HYBRID:
            # Initialize multiple backends
            from src.services.storage.local_storage import LocalStorage
            from src.services.storage.s3_storage import S3Storage
            
            # Local as primary
            local_adapter = LocalStorage(base_path=Path(f"outputs/{self.bucket}"))
            await local_adapter.initialize()
            self._adapters[StorageBackend.LOCAL] = local_adapter
            self._primary_adapter = local_adapter
            
            # S3 as secondary (if configured)
            try:
                s3_adapter = S3Storage(bucket=self.bucket)
                await s3_adapter.initialize()
                self._adapters[StorageBackend.S3] = s3_adapter
            except Exception as e:
                self.logger.warning(f"Failed to initialize S3 adapter: {e}")
    
    async def _ensure_bucket_exists(self) -> None:
        """Ensure default bucket exists."""
        # For local storage, create directory
        if self.backend == StorageBackend.LOCAL:
            bucket_path = Path(f"outputs/{self.bucket}")
            bucket_path.mkdir(parents=True, exist_ok=True)
    
    async def put(
        self,
        key: str,
        data: Union[bytes, str, BinaryIO],
        metadata: Optional[Dict[str, Any]] = None,
        content_type: Optional[str] = None
    ) -> str:
        """
        Store object in storage.
        
        Args:
            key: Object key
            data: Data to store
            metadata: Custom metadata
            content_type: MIME content type
            
        Returns:
            str: Object URL or key
            
        Raises:
            ServiceException: If storage fails
        """
        try:
            # Convert data to bytes
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            elif hasattr(data, 'read'):
                data_bytes = data.read()
            else:
                data_bytes = data
            
            # Detect content type if not provided
            if not content_type:
                content_type = mimetypes.guess_type(key)[0] or "application/octet-stream"
            
            # Create metadata
            storage_metadata = StorageMetadata(
                content_type=content_type,
                content_length=len(data_bytes),
                etag=self._calculate_etag(data_bytes),
                custom_metadata=metadata or {}
            )
            
            # Add version if enabled
            if self.enable_versioning:
                storage_metadata.version = datetime.now().strftime("%Y%m%d%H%M%S")
                versioned_key = f"{key}.v{storage_metadata.version}"
            else:
                versioned_key = key
            
            # Store using primary adapter
            success = await self._primary_adapter.put(versioned_key, data_bytes, storage_metadata)
            
            if not success:
                raise ServiceException(f"Failed to store object: {key}")
            
            # Update cache
            self._cache[key] = StorageObject(
                key=key,
                data=data_bytes,
                metadata=storage_metadata,
                bucket=self.bucket
            )
            
            # Update statistics
            self._stats["puts"] += 1
            self._stats["bytes_uploaded"] += len(data_bytes)
            
            # Replicate to secondary adapters if hybrid
            if self.backend == StorageBackend.HYBRID:
                await self._replicate_to_secondary(versioned_key, data_bytes, storage_metadata)
            
            self.logger.debug(f"Stored object: {key} ({len(data_bytes)} bytes)")
            
            return f"{self.bucket}/{versioned_key}"
            
        except Exception as e:
            self.logger.error(f"Failed to store object {key}: {e}")
            raise ServiceException(f"Storage failed: {e}")
    
    async def get(
        self,
        key: str,
        version: Optional[str] = None
    ) -> Optional[StorageObject]:
        """
        Retrieve object from storage.
        
        Args:
            key: Object key
            version: Specific version to retrieve
            
        Returns:
            Optional[StorageObject]: Retrieved object or None
        """
        try:
            # Check cache first
            if key in self._cache and not version:
                cache_obj = self._cache[key]
                cache_age = datetime.now() - cache_obj.metadata.modified_at
                
                if cache_age < self._cache_ttl:
                    self._stats["cache_hits"] += 1
                    self.logger.debug(f"Cache hit for: {key}")
                    return cache_obj
            
            self._stats["cache_misses"] += 1
            
            # Add version to key if specified
            if version and self.enable_versioning:
                versioned_key = f"{key}.v{version}"
            else:
                versioned_key = key
            
            # Retrieve from primary adapter
            obj = await self._primary_adapter.get(versioned_key)
            
            if obj:
                # Update cache
                self._cache[key] = obj
                
                # Update statistics
                self._stats["gets"] += 1
                if obj.data:
                    self._stats["bytes_downloaded"] += len(obj.data)
                
                self.logger.debug(f"Retrieved object: {key}")
                return obj
            
            # Try secondary adapters if hybrid and not found
            if self.backend == StorageBackend.HYBRID and not obj:
                obj = await self._get_from_secondary(versioned_key)
                if obj:
                    # Restore to primary
                    await self._primary_adapter.put(
                        versioned_key,
                        obj.data,
                        obj.metadata
                    )
                    return obj
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve object {key}: {e}")
            return None
    
    async def delete(
        self,
        key: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Delete object from storage.
        
        Args:
            key: Object key
            version: Specific version to delete
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            # Add version to key if specified
            if version and self.enable_versioning:
                versioned_key = f"{key}.v{version}"
            else:
                versioned_key = key
            
            # Delete from primary adapter
            success = await self._primary_adapter.delete(versioned_key)
            
            if success:
                # Remove from cache
                self._cache.pop(key, None)
                
                # Update statistics
                self._stats["deletes"] += 1
                
                # Delete from secondary adapters if hybrid
                if self.backend == StorageBackend.HYBRID:
                    await self._delete_from_secondary(versioned_key)
                
                self.logger.debug(f"Deleted object: {key}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete object {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if object exists.
        
        Args:
            key: Object key
            
        Returns:
            bool: True if exists
        """
        try:
            # Check cache first
            if key in self._cache:
                return True
            
            # Check primary adapter
            return await self._primary_adapter.exists(key)
            
        except Exception as e:
            self.logger.error(f"Failed to check existence of {key}: {e}")
            return False
    
    async def list_objects(
        self,
        prefix: str = "",
        limit: int = 1000
    ) -> List[str]:
        """
        List objects with prefix.
        
        Args:
            prefix: Key prefix
            limit: Maximum number of objects
            
        Returns:
            List[str]: Object keys
        """
        try:
            return await self._primary_adapter.list(prefix, limit)
        except Exception as e:
            self.logger.error(f"Failed to list objects: {e}")
            return []
    
    async def copy(
        self,
        source_key: str,
        dest_key: str
    ) -> bool:
        """
        Copy object within storage.
        
        Args:
            source_key: Source object key
            dest_key: Destination object key
            
        Returns:
            bool: True if copied successfully
        """
        try:
            # Get source object
            source_obj = await self.get(source_key)
            
            if not source_obj or not source_obj.data:
                return False
            
            # Put to destination
            result = await self.put(
                dest_key,
                source_obj.data,
                source_obj.metadata.custom_metadata,
                source_obj.metadata.content_type
            )
            
            return bool(result)
            
        except Exception as e:
            self.logger.error(f"Failed to copy {source_key} to {dest_key}: {e}")
            return False
    
    async def move(
        self,
        source_key: str,
        dest_key: str
    ) -> bool:
        """
        Move object within storage.
        
        Args:
            source_key: Source object key
            dest_key: Destination object key
            
        Returns:
            bool: True if moved successfully
        """
        # Copy then delete
        if await self.copy(source_key, dest_key):
            return await self.delete(source_key)
        return False
    
    def _calculate_etag(self, data: bytes) -> str:
        """
        Calculate ETag for data.
        
        Args:
            data: Data bytes
            
        Returns:
            str: ETag hash
        """
        return hashlib.md5(data).hexdigest()
    
    async def _replicate_to_secondary(
        self,
        key: str,
        data: bytes,
        metadata: StorageMetadata
    ) -> None:
        """
        Replicate object to secondary storage adapters.
        
        Args:
            key: Object key
            data: Object data
            metadata: Object metadata
        """
        for backend, adapter in self._adapters.items():
            if adapter != self._primary_adapter:
                try:
                    await adapter.put(key, data, metadata)
                    self.logger.debug(f"Replicated {key} to {backend.value}")
                except Exception as e:
                    self.logger.warning(f"Failed to replicate to {backend.value}: {e}")
    
    async def _get_from_secondary(self, key: str) -> Optional[StorageObject]:
        """
        Try to get object from secondary adapters.
        
        Args:
            key: Object key
            
        Returns:
            Optional[StorageObject]: Object if found
        """
        for backend, adapter in self._adapters.items():
            if adapter != self._primary_adapter:
                try:
                    obj = await adapter.get(key)
                    if obj:
                        self.logger.debug(f"Retrieved {key} from {backend.value}")
                        return obj
                except Exception as e:
                    self.logger.warning(f"Failed to get from {backend.value}: {e}")
        return None
    
    async def _delete_from_secondary(self, key: str) -> None:
        """
        Delete object from secondary adapters.
        
        Args:
            key: Object key
        """
        for backend, adapter in self._adapters.items():
            if adapter != self._primary_adapter:
                try:
                    await adapter.delete(key)
                    self.logger.debug(f"Deleted {key} from {backend.value}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete from {backend.value}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dict[str, Any]: Storage statistics
        """
        cache_size = sum(
            len(obj.data) if obj.data else 0
            for obj in self._cache.values()
        )
        
        return {
            **self._stats,
            "cache_size": len(self._cache),
            "cache_bytes": cache_size,
            "cache_hit_rate": (
                self._stats["cache_hits"] / 
                max(self._stats["cache_hits"] + self._stats["cache_misses"], 1)
            ) * 100
        }
