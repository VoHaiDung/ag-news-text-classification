"""
Amazon S3 Storage Implementation for AG News Text Classification
================================================================================
This module implements Amazon S3 storage adapter for cloud-based object storage,
providing scalable and durable storage with advanced features.

The S3 adapter supports:
- Object versioning and lifecycle management
- Server-side encryption
- Multipart upload for large files
- Pre-signed URLs for secure access

References:
    - AWS S3 Developer Guide
    - Boto3 Documentation
    - AWS Best Practices for S3

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import io

from src.services.storage.storage_service import StorageAdapter, StorageObject, StorageMetadata
from src.utils.logging_config import get_logger


class S3Storage(StorageAdapter):
    """
    Amazon S3 storage adapter.
    
    This adapter provides storage operations using Amazon S3,
    suitable for production deployments requiring scalable cloud storage.
    """
    
    def __init__(
        self,
        bucket: str,
        region: str = "us-east-1",
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        use_ssl: bool = True,
        multipart_threshold: int = 100 * 1024 * 1024  # 100MB
    ):
        """
        Initialize S3 storage adapter.
        
        Args:
            bucket: S3 bucket name
            region: AWS region
            access_key_id: AWS access key ID
            secret_access_key: AWS secret access key
            endpoint_url: Custom S3 endpoint (for S3-compatible services)
            use_ssl: Use SSL for connections
            multipart_threshold: Threshold for multipart upload
        """
        self.bucket = bucket
        self.region = region
        self.multipart_threshold = multipart_threshold
        
        # S3 client configuration
        self.client_config = {
            "region_name": region,
            "use_ssl": use_ssl
        }
        
        if endpoint_url:
            self.client_config["endpoint_url"] = endpoint_url
        
        if access_key_id and secret_access_key:
            self.client_config["aws_access_key_id"] = access_key_id
            self.client_config["aws_secret_access_key"] = secret_access_key
        
        self.s3_client = None
        self.s3_resource = None
        
        self.logger = get_logger("storage.s3")
    
    async def initialize(self) -> None:
        """Initialize S3 connection and ensure bucket exists."""
        try:
            # Create S3 client and resource
            self.s3_client = boto3.client("s3", **self.client_config)
            self.s3_resource = boto3.resource("s3", **self.client_config)
            
            # Check if bucket exists
            try:
                self.s3_client.head_bucket(Bucket=self.bucket)
                self.logger.info(f"Connected to S3 bucket: {self.bucket}")
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code == "404":
                    # Create bucket if it doesn't exist
                    await self._create_bucket()
                else:
                    raise
                    
        except NoCredentialsError:
            self.logger.error("AWS credentials not found")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize S3 storage: {e}")
            raise
    
    async def _create_bucket(self) -> None:
        """Create S3 bucket with proper configuration."""
        try:
            # Create bucket with location constraint for non-us-east-1 regions
            if self.region != "us-east-1":
                self.s3_client.create_bucket(
                    Bucket=self.bucket,
                    CreateBucketConfiguration={"LocationConstraint": self.region}
                )
            else:
                self.s3_client.create_bucket(Bucket=self.bucket)
            
            # Enable versioning
            self.s3_client.put_bucket_versioning(
                Bucket=self.bucket,
                VersioningConfiguration={"Status": "Enabled"}
            )
            
            # Enable server-side encryption by default
            self.s3_client.put_bucket_encryption(
                Bucket=self.bucket,
                ServerSideEncryptionConfiguration={
                    "Rules": [{
                        "ApplyServerSideEncryptionByDefault": {
                            "SSEAlgorithm": "AES256"
                        }
                    }]
                }
            )
            
            self.logger.info(f"Created S3 bucket: {self.bucket}")
            
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
        Store object in S3.
        
        Args:
            key: Object key
            data: Object data
            metadata: Object metadata
            
        Returns:
            bool: True if stored successfully
        """
        try:
            # Prepare metadata for S3
            s3_metadata = {}
            if metadata:
                s3_metadata = {
                    "ContentType": metadata.content_type,
                    "Metadata": {
                        **metadata.custom_metadata,
                        "created_at": metadata.created_at.isoformat(),
                        "version": metadata.version or ""
                    }
                }
                
                # Add tags if present
                if metadata.tags:
                    tag_set = [
                        {"Key": k, "Value": v}
                        for k, v in metadata.tags.items()
                    ]
                    s3_metadata["Tagging"] = {"TagSet": tag_set}
            
            # Use multipart upload for large files
            if len(data) > self.multipart_threshold:
                await self._multipart_upload(key, data, s3_metadata)
            else:
                # Regular upload
                await self._run_async(
                    self.s3_client.put_object,
                    Bucket=self.bucket,
                    Key=key,
                    Body=data,
                    **s3_metadata
                )
            
            self.logger.debug(f"Stored S3 object: {key} ({len(data)} bytes)")
            return True
            
        except ClientError as e:
            self.logger.error(f"Failed to store S3 object {key}: {e}")
            return False
    
    async def get(self, key: str) -> Optional[StorageObject]:
        """
        Retrieve object from S3.
        
        Args:
            key: Object key
            
        Returns:
            Optional[StorageObject]: Retrieved object or None
        """
        try:
            # Get object
            response = await self._run_async(
                self.s3_client.get_object,
                Bucket=self.bucket,
                Key=key
            )
            
            # Read data
            data = response["Body"].read()
            
            # Extract metadata
            metadata = StorageMetadata(
                content_type=response.get("ContentType", "application/octet-stream"),
                content_length=response.get("ContentLength", len(data)),
                etag=response.get("ETag", "").strip('"'),
                modified_at=response.get("LastModified", datetime.now())
            )
            
            # Extract custom metadata
            if "Metadata" in response:
                metadata.custom_metadata = response["Metadata"]
                if "version" in response["Metadata"]:
                    metadata.version = response["Metadata"]["version"]
            
            # Get tags
            try:
                tag_response = await self._run_async(
                    self.s3_client.get_object_tagging,
                    Bucket=self.bucket,
                    Key=key
                )
                metadata.tags = {
                    tag["Key"]: tag["Value"]
                    for tag in tag_response.get("TagSet", [])
                }
            except ClientError:
                pass  # Tags may not be present
            
            return StorageObject(
                key=key,
                data=data,
                metadata=metadata,
                bucket=self.bucket
            )
            
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "NoSuchKey":
                return None
            self.logger.error(f"Failed to retrieve S3 object {key}: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """
        Delete object from S3.
        
        Args:
            key: Object key
            
        Returns:
            bool: True if deleted successfully
        """
        try:
            await self._run_async(
                self.s3_client.delete_object,
                Bucket=self.bucket,
                Key=key
            )
            
            self.logger.debug(f"Deleted S3 object: {key}")
            return True
            
        except ClientError as e:
            self.logger.error(f"Failed to delete S3 object {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if object exists in S3.
        
        Args:
            key: Object key
            
        Returns:
            bool: True if exists
        """
        try:
            await self._run_async(
                self.s3_client.head_object,
                Bucket=self.bucket,
                Key=key
            )
            return True
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                return False
            self.logger.error(f"Failed to check S3 object {key}: {e}")
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
            
            # Use paginator for large listings
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(
                Bucket=self.bucket,
                Prefix=prefix,
                PaginationConfig={"MaxItems": limit}
            )
            
            async for page in self._async_paginate(page_iterator):
                for obj in page.get("Contents", []):
                    keys.append(obj["Key"])
                    if len(keys) >= limit:
                        break
            
            return keys[:limit]
            
        except ClientError as e:
            self.logger.error(f"Failed to list S3 objects: {e}")
            return []
    
    async def _multipart_upload(
        self,
        key: str,
        data: bytes,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Upload large file using multipart upload.
        
        Args:
            key: Object key
            data: Object data
            metadata: S3 metadata
        """
        try:
            # Initiate multipart upload
            response = await self._run_async(
                self.s3_client.create_multipart_upload,
                Bucket=self.bucket,
                Key=key,
                **metadata
            )
            upload_id = response["UploadId"]
            
            # Calculate part size (minimum 5MB except last part)
            part_size = max(5 * 1024 * 1024, len(data) // 100)
            parts = []
            
            # Upload parts
            for i, offset in enumerate(range(0, len(data), part_size), 1):
                part_data = data[offset:offset + part_size]
                
                response = await self._run_async(
                    self.s3_client.upload_part,
                    Bucket=self.bucket,
                    Key=key,
                    PartNumber=i,
                    UploadId=upload_id,
                    Body=part_data
                )
                
                parts.append({
                    "ETag": response["ETag"],
                    "PartNumber": i
                })
            
            # Complete multipart upload
            await self._run_async(
                self.s3_client.complete_multipart_upload,
                Bucket=self.bucket,
                Key=key,
                UploadId=upload_id,
                MultipartUpload={"Parts": parts}
            )
            
            self.logger.debug(f"Completed multipart upload for {key}")
            
        except Exception as e:
            # Abort multipart upload on error
            try:
                await self._run_async(
                    self.s3_client.abort_multipart_upload,
                    Bucket=self.bucket,
                    Key=key,
                    UploadId=upload_id
                )
            except:
                pass
            raise e
    
    async def generate_presigned_url(
        self,
        key: str,
        operation: str = "get_object",
        expires_in: int = 3600
    ) -> Optional[str]:
        """
        Generate pre-signed URL for secure access.
        
        Args:
            key: Object key
            operation: S3 operation (get_object, put_object)
            expires_in: URL expiration in seconds
            
        Returns:
            Optional[str]: Pre-signed URL or None
        """
        try:
            url = await self._run_async(
                self.s3_client.generate_presigned_url,
                ClientMethod=operation,
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=expires_in
            )
            return url
        except Exception as e:
            self.logger.error(f"Failed to generate pre-signed URL: {e}")
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
    
    async def _async_paginate(self, page_iterator):
        """
        Asynchronously iterate through pages.
        
        Args:
            page_iterator: Boto3 page iterator
            
        Yields:
            Page results
        """
        loop = asyncio.get_event_loop()
        for page in await loop.run_in_executor(None, list, page_iterator):
            yield page
