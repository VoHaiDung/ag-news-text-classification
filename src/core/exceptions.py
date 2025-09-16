"""
Custom exceptions for AG News Text Classification Framework.

Provides a hierarchy of exceptions for better error handling and debugging.
"""

from typing import Any, Dict, Optional

class AGNewsException(Exception):
    """Base exception for all AG News framework exceptions."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """String representation."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message

# ============================================================================
# Configuration Exceptions
# ============================================================================

class ConfigurationError(AGNewsException):
    """Raised when configuration is invalid or missing."""
    pass

class InvalidConfigError(ConfigurationError):
    """Raised when configuration values are invalid."""
    pass

class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing."""
    pass

# ============================================================================
# Data Exceptions
# ============================================================================

class DataError(AGNewsException):
    """Base exception for data-related errors."""
    pass

class DataNotFoundError(DataError):
    """Raised when data is not found."""
    pass

class DataFormatError(DataError):
    """Raised when data format is invalid."""
    pass

class DataValidationError(DataError):
    """Raised when data validation fails."""
    pass

class InsufficientDataError(DataError):
    """Raised when there is insufficient data for operation."""
    pass

# ============================================================================
# Model Exceptions
# ============================================================================

class ModelError(AGNewsException):
    """Base exception for model-related errors."""
    pass

class ModelNotFoundError(ModelError):
    """Raised when model is not found."""
    pass

class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    pass

class ModelSaveError(ModelError):
    """Raised when model saving fails."""
    pass

class ModelInitializationError(ModelError):
    """Raised when model initialization fails."""
    pass

class IncompatibleModelError(ModelError):
    """Raised when model is incompatible with current configuration."""
    pass

# ============================================================================
# Training Exceptions
# ============================================================================

class TrainingError(AGNewsException):
    """Base exception for training-related errors."""
    pass

class TrainingInterruptedError(TrainingError):
    """Raised when training is interrupted."""
    pass

class ConvergenceError(TrainingError):
    """Raised when model fails to converge."""
    pass

class CheckpointError(TrainingError):
    """Raised when checkpoint operations fail."""
    pass

class OutOfMemoryError(TrainingError):
    """Raised when running out of memory during training."""
    
    def __init__(self, message: str = "Out of memory", batch_size: Optional[int] = None):
        """Initialize with batch size info."""
        details = {}
        if batch_size:
            details["batch_size"] = batch_size
            details["suggestion"] = f"Try reducing batch size from {batch_size}"
        super().__init__(message, details)

# ============================================================================
# Component Exceptions
# ============================================================================

class ComponentError(AGNewsException):
    """Base exception for component-related errors."""
    pass

class ComponentNotFoundError(ComponentError):
    """Raised when component is not found in registry."""
    pass

class ComponentInitializationError(ComponentError):
    """Raised when component initialization fails."""
    pass

class IncompatibleComponentError(ComponentError):
    """Raised when components are incompatible."""
    pass

# ============================================================================
# API Exceptions
# ============================================================================

class APIError(AGNewsException):
    """Base exception for API-related errors."""
    
    def __init__(self, message: str, status_code: int = 500, details: Optional[Dict] = None):
        """Initialize with status code."""
        super().__init__(message, details)
        self.status_code = status_code

class AuthenticationError(APIError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, status_code=401)

class AuthorizationError(APIError):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Unauthorized access"):
        super().__init__(message, status_code=403)

class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", retry_after: Optional[int] = None):
        details = {}
        if retry_after:
            details["retry_after"] = retry_after
        super().__init__(message, status_code=429, details=details)

class ValidationError(APIError):
    """Raised when request validation fails."""
    
    def __init__(self, message: str = "Validation failed", errors: Optional[Dict] = None):
        super().__init__(message, status_code=422, details=errors)

# ============================================================================
# External Service Exceptions
# ============================================================================

class ExternalServiceError(AGNewsException):
    """Base exception for external service errors."""
    pass

class OpenAIError(ExternalServiceError):
    """Raised when OpenAI API calls fail."""
    pass

class HuggingFaceError(ExternalServiceError):
    """Raised when Hugging Face API calls fail."""
    pass

class CloudStorageError(ExternalServiceError):
    """Raised when cloud storage operations fail."""
    pass

class DatabaseError(ExternalServiceError):
    """Raised when database operations fail."""
    pass

# ============================================================================
# Evaluation Exceptions
# ============================================================================

class EvaluationError(AGNewsException):
    """Base exception for evaluation-related errors."""
    pass

class MetricComputationError(EvaluationError):
    """Raised when metric computation fails."""
    pass

class BenchmarkError(EvaluationError):
    """Raised when benchmarking fails."""
    pass

# ============================================================================
# Optimization Exceptions
# ============================================================================

class OptimizationError(AGNewsException):
    """Base exception for optimization-related errors."""
    pass

class QuantizationError(OptimizationError):
    """Raised when quantization fails."""
    pass

class PruningError(OptimizationError):
    """Raised when pruning fails."""
    pass

class DistillationError(OptimizationError):
    """Raised when knowledge distillation fails."""
    pass

# ============================================================================
# Experiment Exceptions
# ============================================================================

class ExperimentError(AGNewsException):
    """Base exception for experiment-related errors."""
    pass

class ExperimentNotFoundError(ExperimentError):
    """Raised when experiment is not found."""
    pass

class ExperimentTrackingError(ExperimentError):
    """Raised when experiment tracking fails."""
    pass

# ============================================================================
# Utility Functions
# ============================================================================

def handle_gpu_error(func):
    """
    Decorator to handle GPU-related errors.
    
    Automatically falls back to CPU if GPU fails.
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                import torch
                torch.cuda.empty_cache()
                raise OutOfMemoryError(
                    "GPU out of memory",
                    batch_size=kwargs.get("batch_size")
                )
            raise ModelError(f"GPU error: {e}")
    return wrapper

def raise_if_not_installed(package_name: str, purpose: str = ""):
    """
    Raise error if package is not installed.
    
    Args:
        package_name: Name of the required package
        purpose: What the package is needed for
    """
    try:
        __import__(package_name)
    except ImportError:
        purpose_msg = f" for {purpose}" if purpose else ""
        raise ImportError(
            f"Package '{package_name}' is required{purpose_msg}. "
            f"Install it with: pip install {package_name}"
        )

# Export public API
__all__ = [
    # Base
    "AGNewsException",
    # Configuration
    "ConfigurationError",
    "InvalidConfigError",
    "MissingConfigError",
    # Data
    "DataError",
    "DataNotFoundError",
    "DataFormatError",
    "DataValidationError",
    "InsufficientDataError",
    # Model
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelSaveError",
    "ModelInitializationError",
    "IncompatibleModelError",
    # Training
    "TrainingError",
    "TrainingInterruptedError",
    "ConvergenceError",
    "CheckpointError",
    "OutOfMemoryError",
    # Component
    "ComponentError",
    "ComponentNotFoundError",
    "ComponentInitializationError",
    "IncompatibleComponentError",
    # API
    "APIError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "ValidationError",
    # External Services
    "ExternalServiceError",
    "OpenAIError",
    "HuggingFaceError",
    "CloudStorageError",
    "DatabaseError",
    # Evaluation
    "EvaluationError",
    "MetricComputationError",
    "BenchmarkError",
    # Optimization
    "OptimizationError",
    "QuantizationError",
    "PruningError",
    "DistillationError",
    # Experiment
    "ExperimentError",
    "ExperimentNotFoundError",
    "ExperimentTrackingError",
    # Utilities
    "handle_gpu_error",
    "raise_if_not_installed",
]
