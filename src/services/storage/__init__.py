"""
Storage Services Module for AG News Text Classification
================================================================================
This module provides unified storage abstraction for various storage backends,
enabling seamless data persistence across local, cloud, and distributed systems.

The storage services provide:
- Unified storage interface
- Multiple backend support (S3, GCS, local)
- Automatic failover and replication
- Data versioning and lifecycle management

References:
    - Martin Kleppmann (2017). Designing Data-Intensive Applications
    - AWS S3 Best Practices
    - Google Cloud Storage Documentation

Author: Võ Hải Dũng
License: MIT
"""

from src.services.storage.storage_service import (
    StorageService,
    StorageBackend,
    StorageObject,
    StorageMetadata
)
from src.services.storage.s3_storage import S3Storage
from src.services.storage.gcs_storage import GCSStorage
from src.services.storage.local_storage import LocalStorage

__all__ = [
    "StorageService",
    "StorageBackend",
    "StorageObject",
    "StorageMetadata",
    "S3Storage",
    "GCSStorage",
    "LocalStorage"
]

# Module version
__version__ = "1.0.0"
