"""
Response Schemas
================================================================================
This module defines Pydantic schemas for API response validation and serialization
following OpenAPI 3.0 specification and RESTful best practices.

Implements standardized response formats including:
- Consistent response structure
- Error handling patterns
- Pagination support
- HATEOAS principles

References:
    - Richardson, L., & Ruby, S. (2007). RESTful Web Services
    - OpenAPI Specification v3.1.0
    - JSON:API Specification v1.1

Author: Võ Hải Dũng
License: MIT
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Generic, TypeVar
from enum import Enum
from pydantic import BaseModel, Field, validator
from pydantic.generics import GenericModel

# Type variable for generic responses
T = TypeVar('T')

class StatusEnum(str, Enum):
    """API response status enumeration."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"

class HealthStatus(str, Enum):
    """Service health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class BaseResponse(BaseModel):
    """
    Base response schema for all API responses.
    
    Attributes:
        status: Response status
        message: Optional message
        timestamp: Response timestamp
        request_id: Request tracking ID
    """
    status: StatusEnum = Field(
        StatusEnum.SUCCESS,
        description="Response status"
    )
    message: Optional[str] = Field(
        None,
        description="Optional message"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )
    request_id: Optional[str] = Field(
        None,
        description="Request tracking ID"
    )
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class DataResponse(GenericModel, Generic[T]):
    """
    Generic data response wrapper.
    
    Attributes:
        data: Response data
        status: Response status
        message: Optional message
        metadata: Additional metadata
    """
    data: T = Field(
        ...,
        description="Response data"
    )
    status: StatusEnum = Field(
        StatusEnum.SUCCESS,
        description="Response status"
    )
    message: Optional[str] = Field(
        None,
        description="Optional message"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata"
    )

class PaginatedResponse(GenericModel, Generic[T]):
    """
    Paginated response schema.
    
    Attributes:
        items: List of items
        total: Total number of items
        page: Current page number
        per_page: Items per page
        pages: Total number of pages
        has_next: Has next page
        has_prev: Has previous page
    """
    items: List[T] = Field(
        ...,
        description="List of items"
    )
    total: int = Field(
        ...,
        ge=0,
        description="Total number of items"
    )
    page: int = Field(
        ...,
        ge=1,
        description="Current page number"
    )
    per_page: int = Field(
        ...,
        ge=1,
        le=1000,
        description="Items per page"
    )
    pages: int = Field(
        ...,
        ge=0,
        description="Total number of pages"
    )
    has_next: bool = Field(
        ...,
        description="Has next page"
    )
    has_prev: bool = Field(
        ...,
        description="Has previous page"
    )
    
    @validator('pages', always=True)
    def calculate_pages(cls, v, values):
        """Calculate total pages."""
        if 'total' in values and 'per_page' in values:
            total = values['total']
            per_page = values['per_page']
            return (total + per_page - 1) // per_page if per_page > 0 else 0
        return v

class ClassificationResponse(BaseResponse):
    """
    Text classification response schema.
    
    Attributes:
        prediction: Predicted class label
        confidence: Prediction confidence score
        probabilities: Class probabilities
        explanations: Model explanations
        model_type: Model used for prediction
        processing_time_ms: Processing time in milliseconds
    """
    prediction: str = Field(
        ...,
        description="Predicted class label",
        example="Business"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score"
    )
    probabilities: Optional[Dict[str, float]] = Field(
        None,
        description="Probability scores for each class"
    )
    explanations: Optional[Dict[str, Any]] = Field(
        None,
        description="Model explanations"
    )
    model_type: str = Field(
        ...,
        description="Model used for prediction"
    )
    processing_time_ms: float = Field(
        ...,
        ge=0,
        description="Processing time in milliseconds"
    )
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "status": "success",
                "prediction": "Business",
                "confidence": 0.95,
                "probabilities": {
                    "Business": 0.95,
                    "Sports": 0.03,
                    "World": 0.01,
                    "Sci/Tech": 0.01
                },
                "model_type": "ensemble",
                "processing_time_ms": 45.2,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

class BatchClassificationResponse(BaseResponse):
    """
    Batch classification response schema.
    
    Attributes:
        predictions: List of predictions
        total_processed: Total items processed
        successful: Number of successful predictions
        failed: Number of failed predictions
        average_confidence: Average confidence score
        total_time_ms: Total processing time
    """
    predictions: List[ClassificationResponse] = Field(
        ...,
        description="List of predictions"
    )
    total_processed: int = Field(
        ...,
        ge=0,
        description="Total items processed"
    )
    successful: int = Field(
        ...,
        ge=0,
        description="Successful predictions"
    )
    failed: int = Field(
        ...,
        ge=0,
        description="Failed predictions"
    )
    average_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Average confidence score"
    )
    total_time_ms: float = Field(
        ...,
        ge=0,
        description="Total processing time"
    )

class TrainingResponse(BaseResponse):
    """
    Model training response schema.
    
    Attributes:
        training_id: Training job ID
        model_id: Trained model ID
        status: Training status
        metrics: Training metrics
        epochs_completed: Number of epochs completed
        best_epoch: Best performing epoch
        training_time_seconds: Total training time
    """
    training_id: str = Field(
        ...,
        description="Training job ID"
    )
    model_id: Optional[str] = Field(
        None,
        description="Trained model ID"
    )
    status: str = Field(
        ...,
        description="Training status",
        example="completed"
    )
    metrics: Dict[str, float] = Field(
        ...,
        description="Training metrics"
    )
    epochs_completed: int = Field(
        ...,
        ge=0,
        description="Epochs completed"
    )
    best_epoch: Optional[int] = Field(
        None,
        description="Best performing epoch"
    )
    training_time_seconds: float = Field(
        ...,
        ge=0,
        description="Total training time"
    )

class ModelInfo(BaseModel):
    """
    Model information schema.
    
    Attributes:
        model_id: Model identifier
        model_type: Type of model
        version: Model version
        created_at: Creation timestamp
        updated_at: Last update timestamp
        metrics: Model performance metrics
        config: Model configuration
        status: Model status
    """
    model_id: str = Field(
        ...,
        description="Model identifier"
    )
    model_type: str = Field(
        ...,
        description="Type of model"
    )
    version: str = Field(
        ...,
        description="Model version"
    )
    created_at: datetime = Field(
        ...,
        description="Creation timestamp"
    )
    updated_at: Optional[datetime] = Field(
        None,
        description="Last update timestamp"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Performance metrics"
    )
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Model configuration"
    )
    status: str = Field(
        "active",
        description="Model status"
    )

class ModelListResponse(PaginatedResponse[ModelInfo]):
    """Response schema for model listing."""
    pass

class DatasetInfo(BaseModel):
    """
    Dataset information schema.
    
    Attributes:
        dataset_id: Dataset identifier
        name: Dataset name
        description: Dataset description
        size: Number of samples
        split_type: Data split type
        created_at: Creation timestamp
        statistics: Dataset statistics
    """
    dataset_id: str = Field(
        ...,
        description="Dataset identifier"
    )
    name: str = Field(
        ...,
        description="Dataset name"
    )
    description: Optional[str] = Field(
        None,
        description="Dataset description"
    )
    size: int = Field(
        ...,
        ge=0,
        description="Number of samples"
    )
    split_type: str = Field(
        ...,
        description="Data split type"
    )
    created_at: datetime = Field(
        ...,
        description="Creation timestamp"
    )
    statistics: Optional[Dict[str, Any]] = Field(
        None,
        description="Dataset statistics"
    )

class ComponentHealth(BaseModel):
    """
    Component health status schema.
    
    Attributes:
        name: Component name
        status: Health status
        latency_ms: Response latency
        details: Additional details
        error: Error message if unhealthy
    """
    name: str = Field(
        ...,
        description="Component name"
    )
    status: str = Field(
        ...,
        description="Health status"
    )
    latency_ms: Optional[float] = Field(
        None,
        ge=0,
        description="Response latency"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional details"
    )
    error: Optional[str] = Field(
        None,
        description="Error message"
    )

class HealthResponse(BaseResponse):
    """
    Health check response schema.
    
    Attributes:
        status: Overall health status
        version: API version
        uptime_seconds: Service uptime
        components: Component health statuses
    """
    status: str = Field(
        ...,
        description="Overall health status"
    )
    version: str = Field(
        ...,
        description="API version"
    )
    uptime_seconds: float = Field(
        ...,
        ge=0,
        description="Service uptime in seconds"
    )
    components: Optional[List[ComponentHealth]] = Field(
        None,
        description="Component health statuses"
    )

class MetricsResponse(BaseResponse):
    """
    Metrics response schema.
    
    Attributes:
        metrics: Metric values
        period_start: Period start time
        period_end: Period end time
        aggregation: Aggregation method
    """
    metrics: Dict[str, Any] = Field(
        ...,
        description="Metric values"
    )
    period_start: datetime = Field(
        ...,
        description="Period start time"
    )
    period_end: datetime = Field(
        ...,
        description="Period end time"
    )
    aggregation: str = Field(
        "average",
        description="Aggregation method"
    )

class AdminActionResponse(BaseResponse):
    """
    Admin action response schema.
    
    Attributes:
        success: Action success status
        action: Action performed
        details: Action details
    """
    success: bool = Field(
        ...,
        description="Action success status"
    )
    action: Optional[str] = Field(
        None,
        description="Action performed"
    )
    details: Optional[Dict[str, Any]] = Field(
        None,
        description="Action details"
    )

class SystemInfoResponse(BaseResponse):
    """
    System information response schema.
    
    Attributes:
        system: System information
        process: Process information
        environment: Environment information
    """
    system: Dict[str, Any] = Field(
        ...,
        description="System information"
    )
    process: Dict[str, Any] = Field(
        ...,
        description="Process information"
    )
    environment: Dict[str, Any] = Field(
        ...,
        description="Environment information"
    )

class ConfigResponse(BaseResponse):
    """
    Configuration response schema.
    
    Attributes:
        section: Configuration section
        values: Configuration values
        last_updated: Last update timestamp
    """
    section: str = Field(
        ...,
        description="Configuration section"
    )
    values: Dict[str, Any] = Field(
        ...,
        description="Configuration values"
    )
    last_updated: datetime = Field(
        ...,
        description="Last update timestamp"
    )

class ExperimentResponse(BaseResponse):
    """
    Experiment response schema.
    
    Attributes:
        experiment_id: Experiment identifier
        name: Experiment name
        status: Experiment status
        results: Experiment results
        created_at: Creation timestamp
        completed_at: Completion timestamp
    """
    experiment_id: str = Field(
        ...,
        description="Experiment identifier"
    )
    name: str = Field(
        ...,
        description="Experiment name"
    )
    status: str = Field(
        ...,
        description="Experiment status"
    )
    results: Optional[Dict[str, Any]] = Field(
        None,
        description="Experiment results"
    )
    created_at: datetime = Field(
        ...,
        description="Creation timestamp"
    )
    completed_at: Optional[datetime] = Field(
        None,
        description="Completion timestamp"
    )

# Export schemas
__all__ = [
    "StatusEnum",
    "HealthStatus",
    "BaseResponse",
    "DataResponse",
    "PaginatedResponse",
    "ClassificationResponse",
    "BatchClassificationResponse",
    "TrainingResponse",
    "ModelInfo",
    "ModelListResponse",
    "DatasetInfo",
    "ComponentHealth",
    "HealthResponse",
    "MetricsResponse",
    "AdminActionResponse",
    "SystemInfoResponse",
    "ConfigResponse",
    "ExperimentResponse"
]
