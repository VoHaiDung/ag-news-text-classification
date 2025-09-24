"""
Common Schemas
================================================================================
This module defines common schemas shared across API endpoints, implementing
reusable data structures and types following DRY principles.

Provides common schemas for:
- Pagination parameters
- Filtering and sorting
- File uploads
- Metadata structures
- Common enumerations

References:
    - JSON Schema Specification Draft 2020-12
    - OpenAPI Specification v3.1.0
    - ISO 8601 Date and Time Format

Author: Võ Hải Dũng
License: MIT
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, constr, conint
from uuid import UUID

class SortOrder(str, Enum):
    """Sort order enumeration."""
    ASC = "asc"
    DESC = "desc"

class FileFormat(str, Enum):
    """Supported file format enumeration."""
    JSON = "json"
    CSV = "csv"
    TXT = "txt"
    JSONL = "jsonl"
    PARQUET = "parquet"
    PICKLE = "pickle"

class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class ResourceType(str, Enum):
    """Resource type enumeration."""
    MODEL = "model"
    DATASET = "dataset"
    EXPERIMENT = "experiment"
    CHECKPOINT = "checkpoint"
    CONFIGURATION = "configuration"

class PaginationParams(BaseModel):
    """
    Pagination parameters schema.
    
    Attributes:
        page: Page number (1-indexed)
        per_page: Items per page
        offset: Item offset
        limit: Maximum items to return
    """
    page: conint(ge=1) = Field(
        1,
        description="Page number (1-indexed)"
    )
    per_page: conint(ge=1, le=1000) = Field(
        20,
        description="Items per page"
    )
    offset: Optional[conint(ge=0)] = Field(
        None,
        description="Item offset"
    )
    limit: Optional[conint(ge=1, le=1000)] = Field(
        None,
        description="Maximum items"
    )
    
    @validator('offset', always=True)
    def calculate_offset(cls, v, values):
        """Calculate offset from page if not provided."""
        if v is None and 'page' in values and 'per_page' in values:
            return (values['page'] - 1) * values['per_page']
        return v
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "page": 1,
                "per_page": 20
            }
        }

class FilterParams(BaseModel):
    """
    Filter parameters schema.
    
    Attributes:
        field: Field to filter on
        operator: Filter operator
        value: Filter value
        case_sensitive: Case-sensitive comparison
    """
    field: str = Field(
        ...,
        description="Field to filter on"
    )
    operator: str = Field(
        "eq",
        pattern="^(eq|ne|gt|gte|lt|lte|in|nin|contains|regex)$",
        description="Filter operator"
    )
    value: Union[str, int, float, bool, List[Any]] = Field(
        ...,
        description="Filter value"
    )
    case_sensitive: bool = Field(
        False,
        description="Case-sensitive comparison"
    )

class SortParams(BaseModel):
    """
    Sort parameters schema.
    
    Attributes:
        field: Field to sort by
        order: Sort order
        null_first: Place nulls first
    """
    field: str = Field(
        ...,
        description="Field to sort by"
    )
    order: SortOrder = Field(
        SortOrder.ASC,
        description="Sort order"
    )
    null_first: bool = Field(
        False,
        description="Place nulls first"
    )

class SearchParams(BaseModel):
    """
    Search parameters schema.
    
    Attributes:
        query: Search query string
        fields: Fields to search in
        fuzzy: Enable fuzzy matching
        boost: Field boost values
    """
    query: constr(min_length=1, max_length=500) = Field(
        ...,
        description="Search query"
    )
    fields: Optional[List[str]] = Field(
        None,
        description="Fields to search in"
    )
    fuzzy: bool = Field(
        False,
        description="Enable fuzzy matching"
    )
    boost: Optional[Dict[str, float]] = Field(
        None,
        description="Field boost values"
    )

class DateRangeParams(BaseModel):
    """
    Date range parameters schema.
    
    Attributes:
        start_date: Start date
        end_date: End date
        include_start: Include start date
        include_end: Include end date
    """
    start_date: Optional[Union[date, datetime]] = Field(
        None,
        description="Start date"
    )
    end_date: Optional[Union[date, datetime]] = Field(
        None,
        description="End date"
    )
    include_start: bool = Field(
        True,
        description="Include start date"
    )
    include_end: bool = Field(
        True,
        description="Include end date"
    )
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Validate date range."""
        if v and 'start_date' in values and values['start_date']:
            if v < values['start_date']:
                raise ValueError("End date must be after start date")
        return v

class FileUpload(BaseModel):
    """
    File upload schema.
    
    Attributes:
        filename: Original filename
        content_type: MIME type
        size: File size in bytes
        checksum: File checksum
        metadata: Additional metadata
    """
    filename: constr(min_length=1, max_length=255) = Field(
        ...,
        description="Original filename"
    )
    content_type: str = Field(
        ...,
        description="MIME type"
    )
    size: conint(gt=0) = Field(
        ...,
        description="File size in bytes"
    )
    checksum: Optional[str] = Field(
        None,
        description="File checksum (MD5/SHA256)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )

class ResourceIdentifier(BaseModel):
    """
    Resource identifier schema.
    
    Attributes:
        id: Resource ID
        type: Resource type
        version: Resource version
        namespace: Resource namespace
    """
    id: Union[str, UUID] = Field(
        ...,
        description="Resource ID"
    )
    type: ResourceType = Field(
        ...,
        description="Resource type"
    )
    version: Optional[str] = Field(
        None,
        description="Resource version"
    )
    namespace: Optional[str] = Field(
        None,
        description="Resource namespace"
    )

class Metadata(BaseModel):
    """
    Metadata schema for additional information.
    
    Attributes:
        created_at: Creation timestamp
        updated_at: Last update timestamp
        created_by: Creator identifier
        updated_by: Last updater identifier
        tags: Resource tags
        labels: Resource labels
        annotations: Resource annotations
    """
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )
    created_by: Optional[str] = Field(
        None,
        description="Creator identifier"
    )
    updated_by: Optional[str] = Field(
        None,
        description="Last updater identifier"
    )
    tags: Optional[List[str]] = Field(
        None,
        description="Resource tags"
    )
    labels: Optional[Dict[str, str]] = Field(
        None,
        description="Resource labels"
    )
    annotations: Optional[Dict[str, Any]] = Field(
        None,
        description="Resource annotations"
    )

class TaskInfo(BaseModel):
    """
    Task information schema.
    
    Attributes:
        task_id: Task identifier
        status: Task status
        progress: Task progress percentage
        message: Status message
        result: Task result
        error: Error information
        started_at: Task start time
        completed_at: Task completion time
    """
    task_id: str = Field(
        ...,
        description="Task identifier"
    )
    status: TaskStatus = Field(
        ...,
        description="Task status"
    )
    progress: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="Progress percentage"
    )
    message: Optional[str] = Field(
        None,
        description="Status message"
    )
    result: Optional[Any] = Field(
        None,
        description="Task result"
    )
    error: Optional[str] = Field(
        None,
        description="Error information"
    )
    started_at: Optional[datetime] = Field(
        None,
        description="Task start time"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Task completion time"
    )

class ProgressUpdate(BaseModel):
    """
    Progress update schema.
    
    Attributes:
        current: Current progress value
        total: Total progress value
        percentage: Progress percentage
        eta_seconds: Estimated time to completion
        message: Progress message
    """
    current: int = Field(
        ...,
        ge=0,
        description="Current progress"
    )
    total: int = Field(
        ...,
        gt=0,
        description="Total progress"
    )
    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Progress percentage"
    )
    eta_seconds: Optional[float] = Field(
        None,
        ge=0,
        description="ETA in seconds"
    )
    message: Optional[str] = Field(
        None,
        description="Progress message"
    )
    
    @validator('percentage', always=True)
    def calculate_percentage(cls, v, values):
        """Calculate percentage from current and total."""
        if 'current' in values and 'total' in values:
            return (values['current'] / values['total']) * 100
        return v

class Link(BaseModel):
    """
    HATEOAS link schema.
    
    Attributes:
        rel: Link relation
        href: Link URL
        method: HTTP method
        title: Link title
    """
    rel: str = Field(
        ...,
        description="Link relation"
    )
    href: str = Field(
        ...,
        description="Link URL"
    )
    method: str = Field(
        "GET",
        pattern="^(GET|POST|PUT|PATCH|DELETE)$",
        description="HTTP method"
    )
    title: Optional[str] = Field(
        None,
        description="Link title"
    )

class CollectionMeta(BaseModel):
    """
    Collection metadata schema.
    
    Attributes:
        total_count: Total items in collection
        filtered_count: Items after filtering
        aggregations: Aggregation results
        facets: Facet results
    """
    total_count: int = Field(
        ...,
        ge=0,
        description="Total items"
    )
    filtered_count: int = Field(
        ...,
        ge=0,
        description="Filtered items"
    )
    aggregations: Optional[Dict[str, Any]] = Field(
        None,
        description="Aggregation results"
    )
    facets: Optional[Dict[str, List[Dict[str, Any]]]] = Field(
        None,
        description="Facet results"
    )

# Export schemas
__all__ = [
    "SortOrder",
    "FileFormat",
    "TaskStatus",
    "ResourceType",
    "PaginationParams",
    "FilterParams",
    "SortParams",
    "SearchParams",
    "DateRangeParams",
    "FileUpload",
    "ResourceIdentifier",
    "Metadata",
    "TaskInfo",
    "ProgressUpdate",
    "Link",
    "CollectionMeta"
]
