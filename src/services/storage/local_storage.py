"""
Local Storage Implementation for AG News Text Classification
================================================================================
This module implements local filesystem storage adapter for the storage service,
providing file-based persistence with versioning and metadata support.

References:
    - POSIX Filesystem Standards
    - Python pathlib Documentation

Author: Võ Hải Dũng
License: MIT
"""

import os
import json
import shutil
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from src.services.storage.storage_service import StorageAdapter, StorageObject, StorageMetadata
from src.utils.logging_config import get_logger


class LocalStorage(StorageAdapter):
    """
    Local filesystem storage adapter.
    
    This adapter provides storage operations using the local filesystem,
    suitable for development and single-node deployments.
    """
    
    def __init__(self, base_path: Path = Path("outputs/storage")):
        """
        Initialize local storage adapter.
        
        Args:
            base_path: Base directory for storage
        """
        self.base_path = base_path
        self.metadata_suffix = ".meta"
        self.logger = get_logger("storage.local")
    
    async def initialize(self) -> None:
        """Initialize local storage."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Local storage initialized at: {self.base_path}")
    
    async def put(
        self,
        key: str,
        data: bytes,
        metadata: Optional[StorageMetadata] = None
    ) -> bool:
        """
        Store object in local filesystem.
        
        Args:
            key: Object key
            data: Object data
            metadata: Object metadata
            
        Returns:
            bool: True if stored successfully
        """
        try:
            # Create full path
            file_path = self.base_path / key
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write data
            with open(file_path, 'wb') as f:
                f.write(data)
            
            # Write metadata
            if metadata:
                meta_path = Path(str(file_path) + self.metadata_suffix)
                with open(meta_path, 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
            
            self.logger.debug(f"Stored local file: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store local file {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[StorageObject]:
        """
        Retrieve object from local filesystem.
        
        Args:
            key: Object key
            
        Returns:
            Optional[StorageObject]: Retrieved object or None
        """
        try:
            file_path = self.base_path / key
            
            if not file_path.exists():
                return None
            
            # Read data
            with open(file_path, 'rb') as f:
                data = f.read()
            
            # Read metadata
            metadata = StorageMetadata()
            meta_path = Path(str(file_path) + self.metadata_suffix)
            
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    meta_dict = json.load(f)
                    metadata.content_type = meta_dict.get("content_type", "application/octet-stream")
                    metadata.content_length = meta_dict.get("content_length", len(data))
                    metadata.custom_metadata = meta_dict.get("custom_metadata", {})
                    metadata.tags = meta_dict.get("tags", {})
            else:
                # Get file stats
                stat = file_path.stat()
                metadata.content_length = stat.st_size
                metadata.created_at = datetime.fromtimestamp(stat.st_ctime)
                metadata.modified_at = datetime.fromtimestamp(stat.st_mtime)
            
            return StorageObject(
                key=key,
                data=data,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve local file {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete object from local filesystem.
        
        Args:
            key: Object key
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            file_path = self.base_path / key
            
            # Delete data file
            if file_path.exists():
                file_path.unlink()
            
            # Delete metadata file
            meta_path = Path(str(file_path) + self.metadata_suffix)
            if meta_path.exists():
                meta_path.unlink()
            
            # Remove empty parent directories
            try:
                file_path.parent.rmdir()
            except OSError:
                pass  # Directory not empty
            
            self.logger.debug(f"Deleted local file: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete local file {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if object exists in local filesystem.
        
        Args:
            key: Object key
            
        Returns:
            bool: True if exists
        """
        file_path = self.base_path / key
        return file_path.exists()
    
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
            search_path = self.base_path / prefix if prefix else self.base_path
            
            for file_path in search_path.rglob("*"):
                if file_path.is_file() and not file_path.name.endswith(self.metadata_suffix):
                    # Get relative path as key
                    key = str(file_path.relative_to(self.base_path))
                    keys.append(key.replace(os.sep, "/"))  # Normalize path separator
                    
                    if len(keys) >= limit:
                        break
            
            return keys
            
        except Exception as e:
            self.logger.error(f"Failed to list local files: {e}")
            return []
