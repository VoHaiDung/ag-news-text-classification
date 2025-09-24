"""
Error Schemas
================================================================================
This module defines standardized error response schemas following RFC 7807
(Problem Details for HTTP APIs) and RESTful error handling best practices.

Implements comprehensive error handling including:
- Standardized error format
- Error categorization and codes
- Validation error details
- Error tracking and debugging information

References:
    - RFC 7807: Problem Details for HTTP APIs
    - Google JSON Style Guide
    - Microsoft REST API Guidelines

Author: Võ Hải Dũng
License: MIT
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator
import traceback

class ErrorType(str, Enum):
    """
    Error type enumeration following RFC 7807.
    
    Based on standard HTTP error categories and application-specific errors.
    """
    # Client errors (4xx)
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    AUTHORIZATION_ERROR = "authorization_error"
    NOT_FOUND = "not_found"
    RATE_LIMIT_ERROR = "rate_limit_error"
    BAD_REQUEST = "bad_request"
    CONFLICT = "conflict"
    
    # Server errors (5xx)
    INTERNAL_ERROR = "internal_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    TIMEOUT_ERROR = "timeout_error"
    DEPENDENCY_ERROR = "dependency_error"
    
    # Application-specific errors
    MODEL_ERROR = "model_error"
    DATA_ERROR = "data_error"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_ERROR = "resource_error"

class ErrorSeverity(str, Enum):
    """Error severity levels for monitoring and alerting."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ValidationErrorDetail(BaseModel):
    """
    Validation error detail schema.
    
    Attributes:
        field: Field name that failed validation
        message: Error message
        type: Validation error type
        context: Additional context
    """
    field: str = Field(
        ...,
        description="Field that failed validation"
    )
    message: str = Field(
        ...,
        description="Validation error message"
    )
    type: str = Field(
        ...,
        description="Type of validation error"
    )
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context"
    )

class ErrorResponse(BaseModel):
    """
    Standard error response schema following RFC 7807.
    
    Attributes:
        type: Error type identifier
        title: Human-readable error title
        status: HTTP status code
        detail: Detailed error message
        instance: Request instance identifier
        timestamp: Error timestamp
        error_code: Application-specific error code
        severity: Error severity level
        trace_id: Request trace ID for debugging
    """
    type: ErrorType = Field(
        ...,
        description="Error type identifier"
    )
    title: str = Field(
        ...,
        description="Human-readable error title"
    )
    status: int = Field(
        ...,
        ge=400,
        le=599,
        description="HTTP status code"
    )
    detail: str = Field(
        ...,
        description="Detailed error message"
    )
    instance: Optional[str] = Field(
        None,
        description="Request instance identifier"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )
    error_code: Optional[str] = Field(
        None,
        description="Application-specific error code"
    )
    severity: ErrorSeverity = Field(
        ErrorSeverity.MEDIUM,
        description="Error severity level"
    )
    trace_id: Optional[str] = Field(
        None,
        description="Request trace ID"
    )
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        schema_extra = {
            "example": {
                "type": "validation_error",
                "title": "Validation Failed",
                "status": 400,
                "detail": "The provided input failed validation",
                "timestamp": "2024-01-15T10:30:00Z",
                "error_code": "VAL001",
                "severity": "low"
            }
        }

class ValidationErrorResponse(ErrorResponse):
    """
    Validation error response with field-level details.
    
    Attributes:
        errors: List of validation error details
        failed_fields: Number of fields that failed validation
    """
    errors: List[ValidationErrorDetail] = Field(
        ...,
        description="List of validation errors"
    )
    failed_fields: int = Field(
        ...,
        ge=1,
        description="Number of failed fields"
    )
    
    def __init__(self, **data):
        """Initialize validation error response."""
        if 'type' not in data:
            data['type'] = ErrorType.VALIDATION_ERROR
        if 'title' not in data:
            data['title'] = "Validation Error"
        if 'status' not in data:
            data['status'] = 400
        if 'severity' not in data:
            data['severity'] = ErrorSeverity.LOW
        
        super().__init__(**data)
    
    @validator('failed_fields', always=True)
    def count_failed_fields(cls, v, values):
        """Count number of failed fields."""
        if 'errors' in values:
            return len(values['errors'])
        return v

class AuthenticationErrorResponse(ErrorResponse):
    """
    Authentication error response.
    
    Attributes:
        auth_type: Authentication type that failed
        realm: Authentication realm
    """
    auth_type: Optional[str] = Field(
        None,
        description="Authentication type"
    )
    realm: Optional[str] = Field(
        None,
        description="Authentication realm"
    )
    
    def __init__(self, **data):
        """Initialize authentication error response."""
        if 'type' not in data:
            data['type'] = ErrorType.AUTHENTICATION_ERROR
        if 'title' not in data:
            data['title'] = "Authentication Failed"
        if 'status' not in data:
            data['status'] = 401
        if 'severity' not in data:
            data['severity'] = ErrorSeverity.MEDIUM
        
        super().__init__(**data)

class RateLimitErrorResponse(ErrorResponse):
    """
    Rate limit error response.
    
    Attributes:
        limit: Rate limit threshold
        remaining: Remaining requests
        reset_at: Rate limit reset timestamp
    """
    limit: int = Field(
        ...,
        ge=0,
        description="Rate limit threshold"
    )
    remaining: int = Field(
        0,
        ge=0,
        description="Remaining requests"
    )
    reset_at: datetime = Field(
        ...,
        description="Rate limit reset time"
    )
    
    def __init__(self, **data):
        """Initialize rate limit error response."""
        if 'type' not in data:
            data['type'] = ErrorType.RATE_LIMIT_ERROR
        if 'title' not in data:
            data['title'] = "Rate Limit Exceeded"
        if 'status' not in data:
            data['status'] = 429
        if 'severity' not in data:
            data['severity'] = ErrorSeverity.LOW
        
        super().__init__(**data)

class ServiceErrorResponse(ErrorResponse):
    """
    Service error response for dependency failures.
    
    Attributes:
        service_name: Name of failed service
        retry_after: Suggested retry delay in seconds
        fallback_available: Whether fallback is available
    """
    service_name: str = Field(
        ...,
        description="Name of failed service"
    )
    retry_after: Optional[int] = Field(
        None,
        ge=0,
        description="Retry after seconds"
    )
    fallback_available: bool = Field(
        False,
        description="Fallback available"
    )
    
    def __init__(self, **data):
        """Initialize service error response."""
        if 'type' not in data:
            data['type'] = ErrorType.DEPENDENCY_ERROR
        if 'title' not in data:
            data['title'] = "Service Dependency Error"
        if 'status' not in data:
            data['status'] = 503
        if 'severity' not in data:
            data['severity'] = ErrorSeverity.HIGH
        
        super().__init__(**data)

class ModelErrorResponse(ErrorResponse):
    """
    Model-related error response.
    
    Attributes:
        model_id: Model identifier
        model_type: Type of model
        error_stage: Stage where error occurred
    """
    model_id: Optional[str] = Field(
        None,
        description="Model identifier"
    )
    model_type: Optional[str] = Field(
        None,
        description="Model type"
    )
    error_stage: Optional[str] = Field(
        None,
        description="Error stage (loading/inference/training)"
    )
    
    def __init__(self, **data):
        """Initialize model error response."""
        if 'type' not in data:
            data['type'] = ErrorType.MODEL_ERROR
        if 'title' not in data:
            data['title'] = "Model Processing Error"
        if 'status' not in data:
            data['status'] = 500
        if 'severity' not in data:
            data['severity'] = ErrorSeverity.HIGH
        
        super().__init__(**data)

class BatchErrorResponse(BaseModel):
    """
    Batch operation error response.
    
    Attributes:
        total_errors: Total number of errors
        errors: List of individual errors
        partial_success: Whether partial success occurred
        successful_count: Number of successful operations
    """
    total_errors: int = Field(
        ...,
        ge=1,
        description="Total number of errors"
    )
    errors: List[ErrorResponse] = Field(
        ...,
        description="List of errors"
    )
    partial_success: bool = Field(
        False,
        description="Partial success occurred"
    )
    successful_count: int = Field(
        0,
        ge=0,
        description="Number of successful operations"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Error timestamp"
    )

def create_error_response(
    error_type: ErrorType,
    title: str,
    detail: str,
    status: int,
    **kwargs
) -> ErrorResponse:
    """
    Factory function to create error responses.
    
    Args:
        error_type: Type of error
        title: Error title
        detail: Error detail message
        status: HTTP status code
        **kwargs: Additional error fields
        
    Returns:
        ErrorResponse: Configured error response
    """
    return ErrorResponse(
        type=error_type,
        title=title,
        detail=detail,
        status=status,
        **kwargs
    )

def create_validation_error(
    errors: List[Dict[str, Any]],
    detail: str = "Validation failed"
) -> ValidationErrorResponse:
    """
    Create validation error response from error list.
    
    Args:
        errors: List of validation errors
        detail: Overall error detail
        
    Returns:
        ValidationErrorResponse: Validation error response
    """
    validation_errors = [
        ValidationErrorDetail(
            field=err.get("field", "unknown"),
            message=err.get("message", "Validation failed"),
            type=err.get("type", "validation"),
            context=err.get("context")
        )
        for err in errors
    ]
    
    return ValidationErrorResponse(
        detail=detail,
        errors=validation_errors,
        failed_fields=len(validation_errors)
    )

# Export schemas
__all__ = [
    "ErrorType",
    "ErrorSeverity",
    "ValidationErrorDetail",
    "ErrorResponse",
    "ValidationErrorResponse",
    "AuthenticationErrorResponse",
    "RateLimitErrorResponse",
    "ServiceErrorResponse",
    "ModelErrorResponse",
    "BatchErrorResponse",
    "create_error_response",
    "create_validation_error"
]
