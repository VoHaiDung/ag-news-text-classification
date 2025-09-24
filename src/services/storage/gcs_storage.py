"""
Google Cloud Storage Implementation for AG News Text Classification
================================================================================
This module implements Google Cloud Storage adapter for cloud-based object storage,
providing scalable storage with Google Cloud Platform integration.

The GCS adapter supports:
- Object versioning and lifecycle policies
- Customer-managed encryption keys
- Resumable uploads for large files
- Signed URLs for secure access

References:
    - Google Cloud Storage Documentation
    - Python Client for Google Cloud Storage
    - GCS Best Practices Guide

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError
from google.api_core import retry
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import io

from src.services.storage.storage_service import StorageAdapter, StorageObject, StorageMetadata
from src.utils.logging_config import get_logger


class GCSStorage(StorageAdapter):
    """
    Google Cloud Storage adapter.
    
    This adapter provides storage operations using Google Cloud Storage,
    suitable for production deployments on Google Cloud Platform.
    """
    
    def __init__(
        self,
        bucket: str,
        project: Optional[str] = None,
        credentials_path: Optional[str] = None,
        location: str = "us-central1",
        storage_class: str = "STANDARD",
        chunk_size: int = 100 * 1024 * 1024  # 100MB
    ):
        """
        Initialize GCS storage adapter.
        
        Args:
            bucket: GCS bucket name
            project: GCP project ID
            credentials_path: Path to service account credentials
            location: Bucket location
            storage_class: Storage class (STANDARD, NEARLINE, COLDLINE, ARCHIVE)
            chunk_size: Chunk size for resumable uploads
        """
        self.bucket_name = bucket
        self.project = project
        self.credentials_path = credentials_path
        self.location = location
        self.storage_class = storage_class
        self.chunk_size = chunk_size
        
        self.client = None
        self.bucket = None
        
        self.logger = get_logger("storage.gcs")
    
    async def initialize(self) -> None:
        """Initialize GCS connection and ensure bucket exists."""
        try:
            # Create GCS client
            if self.credentials_path:
                self.client = storage.Client.from_service_account_json(
                    self.credentials_path,
                    project=self.project
                )
            else:
                self.client = storage.Client(project=self.project)
            
            # Get or create bucket
            try:
                self.bucket = self.client.get_bucket(self.bucket_name)
                self.logger.info(f"Connected to GCS bucket: {self.bucket_name}")
            except NotFound:
                await self._create_bucket()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize GCS storage: {e}")
            raise
    
    async def _create_bucket(self) -> None:
        """Create GCS bucket with proper configuration."""
        try:
            # Create bucket
            self.bucket = self.client.create_bucket(
                self.bucket_name,
                location=self.location
            )
            
            # Set storage class
            self.bucket.storage_class = self.storage_class
            
            # Enable versioning
            self.bucket.versioning_enabled = True
            
            # Set lifecycle rules
            self.bucket.add_lifecycle_delete_rule(age=365)  # Delete after 1 year
            self.bucket.add_lifecycle_set_storage_class_rule(
                storage_class="NEARLINE",
                age=30  # Move to nearline after 30 days
            )
            
            # Update bucket configuration
            self.bucket.patch()
            
            self.logger.info(f"Created GCS bucket: {self.bucket_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create bucket: {e}")
            raise
    
    async def put(
        self,
        key: str,
        data: bytes,
        metadata: Optional[StorageMetadata] = None
    ) -> bool:
        """
        Store object in GCS.
        
        Args:
            key: Object key
            data: Object data
            metadata: Object metadata
            
        Returns:
            bool: True if stored successfully
        """
        try:
            # Create blob
            blob = self.bucket.blob(key)
            
            # Set metadata
            if metadata:
                blob.content_type = metadata.content_type
                
                # Set custom metadata
                blob.metadata = metadata.custom_metadata.copy()
                blob.metadata["created_at"] = metadata.created_at.isoformat()
                if metadata.version:
                    blob.metadata["version"] = metadata.version
                
                # Set cache control
                blob.cache_control = "private, max-age=3600"
            
            # Upload data
            if len(data) > self.chunk_size:
                # Use resumable upload for large files
                await self._resumable_upload(blob, data)
            else:
                # Regular upload
                await self._run_async(
                    blob.upload_from_string,
                    data,
                    content_type=metadata.content_type if metadata else None,
                    retry=retry.Retry(deadline=30)
                )
            
            self.logger.debug(f"Stored GCS object: {key} ({len(data)} bytes)")
            return True
            
        except GoogleCloudError as e:
            self.logger.error(f"Failed to store GCS object {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[StorageObject]:
        """
        Retrieve object from GCS.
        
        Args:
            key: Object key
            
        Returns:
            Optional[StorageObject]: Retrieved object or None
        """
        try:
            # Get blob
            blob = self.bucket.blob(key)
            
            # Check if exists
            if not await self._run_async(blob.exists):
                return None
            
            # Download data
            data = await self._run_async(blob.download_as_bytes)
            
            # Get metadata
            await self._run_async(blob.reload)  # Reload to get metadata
            
            metadata = StorageMetadata(
                content_type=blob.content_type or "application/octet-stream",
                content_length=blob.size or len(data),
                etag=blob.etag,
                created_at=blob.time_created or datetime.now(),
                modified_at=blob.updated or datetime.now()
            )
            
            # Extract custom metadata
            if blob.metadata:
                metadata.custom_metadata = blob.metadata.copy()
                if "version" in blob.metadata:
                    metadata.version = blob.metadata["version"]
            
            return StorageObject(
                key=key,
                data=data,
                metadata=metadata,
                bucket=self.bucket_name
            )
            
        except NotFound:
            return None
        except GoogleCloudError as e:
            self.logger.error(f"Failed to retrieve GCS object {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete object from GCS.
        
        Args:
            key: Object key
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            blob = self.bucket.blob(key)
            await self._run_async(blob.delete)
            
            self.logger.debug(f"Deleted GCS object: {key}")
            return True
            
        except NotFound:
            return True  # Already deleted
        except GoogleCloudError as e:
            self.logger.error(f"Failed to delete GCS object {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if object exists in GCS.
        
        Args:
            key: Object key
            
        Returns:
            bool: True if exists
        """
        try:
            blob = self.bucket.blob(key)
            return await self._run_async(blob.exists)
        except GoogleCloudError as e:
            self.logger.error(f"Failed to check GCS object {key}: {e}")
            return False
    
    async def list(self, prefix: str = "", limit: int = 1000) -> List[str]:
        """
        List objects with prefix.
        
        Args:
            prefix: Key prefix
            limit: Maximum number of objects
            
        Returns:
            List[str]: Object keys
        """
        try:
            keys = []
            
            # List blobs
            blobs = await self._run_async(
                self.client.list_blobs,
                self.bucket_name,
                prefix=prefix,
                max_results=limit
            )
            
            for blob in blobs:
                keys.append(blob.name)
                if len(keys) >= limit:
                    break
            
            return keys
            
        except GoogleCloudError as e:
            self.logger.error(f"Failed to list GCS objects: {e}")
            return []
    
    async def _resumable_upload(self, blob, data: bytes) -> None:
        """
        Upload large file using resumable upload.
        
        Args:
            blob: GCS blob object
            data: Data to upload
        """
        try:
            # Create a file-like object from bytes
            file_obj = io.BytesIO(data)
            
            # Use resumable upload
            await self._run_async(
                blob.upload_from_file,
                file_obj,
                rewind=True,
                size=len(data),
                resumable=True,
                retry=retry.Retry(deadline=300)
            )
            
            self.logger.debug(f"Completed resumable upload for {blob.name}")
            
        except Exception as e:
            self.logger.error(f"Resumable upload failed: {e}")
            raise
    
    async def generate_signed_url(
        self,
        key: str,
        method: str = "GET",
        expires_in: int = 3600
    ) -> Optional[str]:
        """
        Generate signed URL for secure access.
        
        Args:
            key: Object key
            method: HTTP method (GET, PUT)
            expires_in: URL expiration in seconds
            
        Returns:
            Optional[str]: Signed URL or None
        """
        try:
            blob = self.bucket.blob(key)
            
            url = await self._run_async(
                blob.generate_signed_url,
                version="v4",
                expiration=timedelta(seconds=expires_in),
                method=method
            )
            
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to generate signed URL: {e}")
            return None
    
    async def _run_async(self, func, *args, **kwargs):
        """
        Run synchronous function asynchronously.
        
        Args:
            func: Function to run
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)
