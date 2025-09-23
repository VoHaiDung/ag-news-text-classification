"""
API Schemas Module
================================================================================
Pydantic schemas for request/response validation and serialization in the
REST API, ensuring type safety and data consistency.

This module provides comprehensive data models following OpenAPI 3.0
specification for automatic documentation generation.

References:
    - Pydantic Documentation: Data validation using Python type annotations
    - OpenAPI Initiative (2021). OpenAPI Specification v3.1.0
    - JSON Schema Specification Draft 2020-12

Author: Võ Hải Dũng
License: MIT
"""

from src.api.rest.schemas.request_schemas import (
    ClassificationRequest,
    BatchClassificationRequest,
    StreamingClassificationRequest,
    TrainingRequest,
    ModelManagementRequest,
    DataUploadRequest,
    MetricsRequest,
    ExplanationRequest,
    FeedbackRequest,
    ModelName,
    ProcessingMode,
    DataFormat
)

from src.api.rest.schemas.response_schemas import (
    ClassificationResponse,
    BatchClassificationResponse,
    PredictionResult,
    ModelInfo,
    ModelListResponse,
    TrainingResponse,
    TrainingStatus,
    MetricsResponse,
    ExplanationResponse,
    ErrorResponse,
    PaginatedResponse,
    BaseResponse,
    CategoryEnum,
    ResponseStatus
)

__all__ = [
    # Request schemas
    "ClassificationRequest",
    "BatchClassificationRequest",
    "StreamingClassificationRequest",
    "TrainingRequest",
    "ModelManagementRequest",
    "DataUploadRequest",
    "MetricsRequest",
    "ExplanationRequest",
    "FeedbackRequest",
    
    # Enums
    "ModelName",
    "ProcessingMode",
    "DataFormat",
    "CategoryEnum",
    "ResponseStatus",
    
    # Response schemas
    "ClassificationResponse",
    "BatchClassificationResponse",
    "PredictionResult",
    "ModelInfo",
    "ModelListResponse",
    "TrainingResponse",
    "TrainingStatus",
    "MetricsResponse",
    "ExplanationResponse",
    "ErrorResponse",
    "PaginatedResponse",
    "BaseResponse"
]

# Schema version for API compatibility
SCHEMA_VERSION = "1.0.0"
