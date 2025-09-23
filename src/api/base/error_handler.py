"""
Error Handling Module for API
================================================================================
Implements comprehensive error handling mechanisms for API operations following
REST principles and error response standards.

This module provides structured error responses, error recovery strategies,
and detailed error logging for debugging and monitoring purposes.

References:
    - Zalando RESTful API Guidelines (2020). Problem JSON
    - RFC 7807 (2016). Problem Details for HTTP APIs
    - Fowler, M. (2014). Microservices and the First Law of Distributed Objects

Author: Võ Hải Dũng
License: MIT
"""

import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from src.core.exceptions import AGNewsException
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ErrorSeverity(Enum):
    """
    Error severity levels for categorization and handling.
    
    Based on standard logging levels and operational impact assessment.
    """
    CRITICAL = "critical"  # System failure, immediate action required
    HIGH = "high"         # Major functionality impaired
    MEDIUM = "medium"     # Partial functionality affected
    LOW = "low"          # Minor issue, workaround available
    INFO = "info"        # Informational, no action required


class ErrorCategory(Enum):
    """
    Error categories for classification and routing.
    
    Helps in determining appropriate error handling strategies.
    """
    VALIDATION = "validation"      # Input validation errors
    AUTHENTICATION = "authentication"  # Authentication failures
    AUTHORIZATION = "authorization"    # Authorization failures
    BUSINESS_LOGIC = "business_logic"  # Business rule violations
    INTEGRATION = "integration"        # External service errors
    SYSTEM = "system"                 # System-level errors
    NETWORK = "network"               # Network-related errors
    DATA = "data"                     # Data integrity errors


@dataclass
class ErrorDetail:
    """
    Detailed error information for debugging and monitoring.
    
    Implements RFC 7807 Problem Details specification for API errors.
    
    Attributes:
        type: URI reference identifying error type
        title: Short human-readable summary
        status: HTTP status code
        detail: Human-readable explanation
        instance: URI reference for specific occurrence
        timestamp: When the error occurred
        error_code: Application-specific error code
        severity: Error severity level
        category: Error category
        context: Additional context information
        stack_trace: Stack trace for debugging (dev only)
        correlation_id: Request correlation ID
        suggested_action: Suggested remediation action
    """
    type: str
    title: str
    status: int
    detail: str
    instance: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_code: Optional[str] = None
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.SYSTEM
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    correlation_id: Optional[str] = None
    suggested_action: Optional[str] = None
    
    def to_dict(self, include_debug: bool = False) -> Dict[str, Any]:
        """
        Convert error detail to dictionary for API response.
        
        Args:
            include_debug: Whether to include debug information
            
        Returns:
            Dictionary representation of error
        """
        error_dict = {
            "type": self.type,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
            "timestamp": self.timestamp.isoformat(),
            "error_code": self.error_code,
            "severity": self.severity.value,
            "category": self.category.value
        }
        
        if self.instance:
            error_dict["instance"] = self.instance
        
        if self.correlation_id:
            error_dict["correlation_id"] = self.correlation_id
        
        if self.suggested_action:
            error_dict["suggested_action"] = self.suggested_action
        
        if self.context:
            error_dict["context"] = self.context
        
        if include_debug and self.stack_trace:
            error_dict["stack_trace"] = self.stack_trace
        
        return error_dict


class APIError(AGNewsException):
    """
    Base class for API-specific errors.
    
    Extends AGNewsException with API-specific error handling capabilities.
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.SYSTEM,
        details: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None
    ):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            error_code: Application error code
            severity: Error severity
            category: Error category
            details: Additional error details
            suggested_action: Suggested remediation
        """
        super().__init__(message, error_code, status_code, details)
        self.severity = severity
        self.category = category
        self.suggested_action = suggested_action
    
    def to_error_detail(self, instance: Optional[str] = None,
                       correlation_id: Optional[str] = None) -> ErrorDetail:
        """
        Convert exception to ErrorDetail.
        
        Args:
            instance: URI for specific error occurrence
            correlation_id: Request correlation ID
            
        Returns:
            ErrorDetail object
        """
        return ErrorDetail(
            type=f"/errors/{self.error_code or 'unknown'}",
            title=self.__class__.__name__,
            status=self.status_code,
            detail=str(self),
            instance=instance,
            error_code=self.error_code,
            severity=self.severity,
            category=self.category,
            context=self.details or {},
            correlation_id=correlation_id,
            suggested_action=self.suggested_action,
            stack_trace=traceback.format_exc() if logger.level == logging.DEBUG else None
        )


class ValidationError(APIError):
    """
    Error raised for request validation failures.
    
    Indicates that the request data does not meet validation requirements.
    """
    
    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraints: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            field: Field that failed validation
            value: Invalid value
            constraints: Validation constraints that were violated
        """
        details = {}
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)[:100]  # Truncate for security
        if constraints:
            details["constraints"] = constraints
        
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.VALIDATION,
            details=details,
            suggested_action="Review the validation constraints and correct the input data"
        )


class AuthenticationError(APIError):
    """
    Error raised for authentication failures.
    
    Indicates that the request lacks valid authentication credentials.
    """
    
    def __init__(self, message: str = "Authentication required",
                 realm: Optional[str] = None):
        """
        Initialize authentication error.
        
        Args:
            message: Error message
            realm: Authentication realm
        """
        details = {"realm": realm} if realm else {}
        
        super().__init__(
            message=message,
            status_code=401,
            error_code="AUTHENTICATION_REQUIRED",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AUTHENTICATION,
            details=details,
            suggested_action="Provide valid authentication credentials"
        )


class AuthorizationError(APIError):
    """
    Error raised for authorization failures.
    
    Indicates that the authenticated user lacks required permissions.
    """
    
    def __init__(
        self,
        message: str = "Insufficient permissions",
        required_permissions: Optional[List[str]] = None,
        user_permissions: Optional[List[str]] = None
    ):
        """
        Initialize authorization error.
        
        Args:
            message: Error message
            required_permissions: Required permissions
            user_permissions: User's current permissions
        """
        details = {}
        if required_permissions:
            details["required_permissions"] = required_permissions
        if user_permissions:
            details["user_permissions"] = user_permissions
        
        super().__init__(
            message=message,
            status_code=403,
            error_code="INSUFFICIENT_PERMISSIONS",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.AUTHORIZATION,
            details=details,
            suggested_action="Request necessary permissions from administrator"
        )


class RateLimitError(APIError):
    """
    Error raised when rate limits are exceeded.
    
    Indicates that the client has sent too many requests in a given time period.
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        """
        Initialize rate limit error.
        
        Args:
            message: Error message
            limit: Rate limit threshold
            window: Time window (e.g., "1 minute")
            retry_after: Seconds until retry is allowed
        """
        details = {}
        if limit:
            details["limit"] = limit
        if window:
            details["window"] = window
        if retry_after:
            details["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.SYSTEM,
            details=details,
            suggested_action=f"Wait {retry_after} seconds before retrying" if retry_after else "Reduce request frequency"
        )


class NotFoundError(APIError):
    """
    Error raised when requested resource is not found.
    
    Indicates that the server cannot find the requested resource.
    """
    
    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None
    ):
        """
        Initialize not found error.
        
        Args:
            message: Error message
            resource_type: Type of resource
            resource_id: Resource identifier
        """
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        
        super().__init__(
            message=message,
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            severity=ErrorSeverity.LOW,
            category=ErrorCategory.DATA,
            details=details,
            suggested_action="Verify the resource identifier and try again"
        )


class ConflictError(APIError):
    """
    Error raised for resource conflicts.
    
    Indicates that the request conflicts with current state of the resource.
    """
    
    def __init__(
        self,
        message: str = "Resource conflict",
        conflicting_resource: Optional[str] = None,
        current_state: Optional[str] = None
    ):
        """
        Initialize conflict error.
        
        Args:
            message: Error message
            conflicting_resource: Conflicting resource identifier
            current_state: Current resource state
        """
        details = {}
        if conflicting_resource:
            details["conflicting_resource"] = conflicting_resource
        if current_state:
            details["current_state"] = current_state
        
        super().__init__(
            message=message,
            status_code=409,
            error_code="RESOURCE_CONFLICT",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.DATA,
            details=details,
            suggested_action="Resolve the conflict before retrying"
        )


class ServiceUnavailableError(APIError):
    """
    Error raised when service is temporarily unavailable.
    
    Indicates that the server is currently unable to handle the request.
    """
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        service_name: Optional[str] = None,
        retry_after: Optional[int] = None
    ):
        """
        Initialize service unavailable error.
        
        Args:
            message: Error message
            service_name: Name of unavailable service
            retry_after: Seconds until service is expected to be available
        """
        details = {}
        if service_name:
            details["service"] = service_name
        if retry_after:
            details["retry_after"] = retry_after
        
        super().__init__(
            message=message,
            status_code=503,
            error_code="SERVICE_UNAVAILABLE",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            details=details,
            suggested_action=f"Retry after {retry_after} seconds" if retry_after else "Retry later"
        )


class ErrorHandler:
    """
    Central error handler for API operations.
    
    Provides consistent error handling, logging, and recovery strategies
    across all API endpoints.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize error handler.
        
        Args:
            config: Error handler configuration
        """
        self.config = config or {}
        self.include_debug = self.config.get("include_debug", False)
        self.max_retry_attempts = self.config.get("max_retry_attempts", 3)
        self.error_callbacks = []
        
        # Error statistics
        self._error_counts = {}
        self._last_errors = []
        self._max_stored_errors = 100
    
    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> ErrorDetail:
        """
        Handle an error and convert to ErrorDetail.
        
        Args:
            error: Exception to handle
            context: Error context
            correlation_id: Request correlation ID
            
        Returns:
            ErrorDetail object
        """
        # Update statistics
        self._update_error_stats(error)
        
        # Create error detail
        if isinstance(error, APIError):
            error_detail = error.to_error_detail(
                instance=context.get("instance") if context else None,
                correlation_id=correlation_id
            )
        else:
            # Handle generic exceptions
            error_detail = self._create_generic_error_detail(
                error, context, correlation_id
            )
        
        # Log error
        self._log_error(error_detail)
        
        # Execute callbacks
        for callback in self.error_callbacks:
            try:
                callback(error_detail)
            except Exception as e:
                logger.error(f"Error callback failed: {str(e)}")
        
        return error_detail
    
    def _create_generic_error_detail(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]],
        correlation_id: Optional[str]
    ) -> ErrorDetail:
        """
        Create ErrorDetail for generic exceptions.
        
        Args:
            error: Generic exception
            context: Error context
            correlation_id: Request correlation ID
            
        Returns:
            ErrorDetail object
        """
        # Map common exceptions to appropriate status codes
        status_code = 500
        category = ErrorCategory.SYSTEM
        severity = ErrorSeverity.HIGH
        
        if isinstance(error, ValueError):
            status_code = 400
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.LOW
        elif isinstance(error, KeyError):
            status_code = 400
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.LOW
        elif isinstance(error, TimeoutError):
            status_code = 504
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.HIGH
        elif isinstance(error, ConnectionError):
            status_code = 503
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.HIGH
        
        return ErrorDetail(
            type=f"/errors/{error.__class__.__name__.lower()}",
            title=error.__class__.__name__,
            status=status_code,
            detail=str(error),
            error_code="INTERNAL_ERROR",
            severity=severity,
            category=category,
            context=context or {},
            correlation_id=correlation_id,
            stack_trace=traceback.format_exc() if self.include_debug else None,
            suggested_action="Contact support if the problem persists"
        )
    
    def _log_error(self, error_detail: ErrorDetail) -> None:
        """
        Log error based on severity.
        
        Args:
            error_detail: Error detail to log
        """
        log_message = (
            f"Error occurred: {error_detail.title} - {error_detail.detail} "
            f"[{error_detail.error_code}] "
            f"(Severity: {error_detail.severity.value}, "
            f"Category: {error_detail.category.value})"
        )
        
        if error_detail.correlation_id:
            log_message += f" [Correlation ID: {error_detail.correlation_id}]"
        
        # Log based on severity
        if error_detail.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_detail.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_detail.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
        
        # Log stack trace if available
        if error_detail.stack_trace and logger.level == logging.DEBUG:
            logger.debug(f"Stack trace:\n{error_detail.stack_trace}")
    
    def _update_error_stats(self, error: Exception) -> None:
        """
        Update error statistics for monitoring.
        
        Args:
            error: Exception that occurred
        """
        error_type = error.__class__.__name__
        
        # Update count
        if error_type not in self._error_counts:
            self._error_counts[error_type] = 0
        self._error_counts[error_type] += 1
        
        # Store recent error
        self._last_errors.append({
            "type": error_type,
            "message": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Limit stored errors
        if len(self._last_errors) > self._max_stored_errors:
            self._last_errors = self._last_errors[-self._max_stored_errors:]
    
    def get_error_stats(self) -> Dict[str, Any]:
        """
        Get error statistics for monitoring.
        
        Returns:
            Dictionary of error statistics
        """
        return {
            "error_counts": self._error_counts,
            "recent_errors": self._last_errors[-10:],  # Last 10 errors
            "total_errors": sum(self._error_counts.values())
        }
    
    def add_error_callback(self, callback: callable) -> None:
        """
        Add callback to be executed on error.
        
        Args:
            callback: Callback function
        """
        self.error_callbacks.append(callback)
    
    def clear_error_stats(self) -> None:
        """Clear error statistics."""
        self._error_counts.clear()
        self._last_errors.clear()
