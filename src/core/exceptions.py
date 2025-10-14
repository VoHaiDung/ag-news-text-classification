"""
Custom Exceptions for AG News Text Classification
==================================================

This module defines a comprehensive hierarchy of exceptions for the
AG News Text Classification framework, enabling precise error handling,
debugging, and recovery mechanisms.

Project: AG News Text Classification (ag-news-text-classification)
Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT

Exception Hierarchy:
    All exceptions inherit from AGNewsException, which provides:
    - Structured error messages
    - Additional context through details dictionary
    - Stack trace preservation
    - Custom string representation

Academic Rationale:
    Exception design follows principles from "Effective Python" (Slatkin, 2019)
    and "Python Best Practices" (Reitz & Schlusser, 2016). The hierarchical
    structure enables:
    
    1. Precise Error Handling: Catch specific exceptions without broad try-except
    2. Error Recovery: Context-aware recovery based on exception type
    3. Debugging Support: Rich error context for troubleshooting
    4. API Design: Clear error contracts for function interfaces
    
    The design pattern follows the Fail-Fast principle from "Clean Code"
    (Martin, 2008), where errors are caught early and reported clearly.

Design Principles:
    1. Single Responsibility: Each exception represents one error condition
    2. Hierarchical Organization: Group related exceptions under base classes
    3. Rich Context: Include relevant details for debugging
    4. Backward Compatibility: Maintain exception signatures across versions
    5. Documentation: Clear docstrings explaining when exceptions are raised

Usage Examples:
    Basic exception handling:
        >>> try:
        ...     model = create_model("invalid_model")
        ... except ModelNotFoundError as e:
        ...     print(f"Model error: {e}")
        ...     print(f"Available models: {e.details.get('available_models')}")
    
    Custom exception with details:
        >>> raise OutOfMemoryError(
        ...     "GPU out of memory during training",
        ...     batch_size=64,
        ...     gpu_memory_allocated=15.2,
        ...     suggestion="Reduce batch size to 32"
        ... )
    
    Decorator for GPU error handling:
        >>> @handle_gpu_error
        ... def train_model(model, batch_size=32):
        ...     return model.train(batch_size=batch_size)

References:
    - Slatkin (2019): "Effective Python: 90 Specific Ways to Write Better Python"
    - Martin (2008): "Clean Code: A Handbook of Agile Software Craftsmanship"
    - Reitz & Schlusser (2016): "The Hitchhiker's Guide to Python"
    - PEP 3151: Reworking the OS and IO exception hierarchy
"""

from typing import Any, Dict, List, Optional, Callable
import functools
import logging

logger = logging.getLogger(__name__)


class AGNewsException(Exception):
    """
    Base exception for all AG News Text Classification framework exceptions.
    
    This is the root of the exception hierarchy. All custom exceptions
    should inherit from this class to enable framework-specific error handling.
    
    Attributes:
        message (str): Human-readable error message
        details (dict): Additional context and debugging information
    
    Examples:
        >>> raise AGNewsException(
        ...     "Configuration error",
        ...     details={"config_file": "model.yaml", "line": 42}
        ... )
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception with message and optional details.
        
        Args:
            message (str): Error message describing what went wrong
            details (dict, optional): Additional context (file paths, parameters, etc.)
        
        Examples:
            >>> e = AGNewsException("Error occurred", details={"step": 5})
            >>> str(e)
            'Error occurred (step=5)'
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """
        String representation with details.
        
        Returns:
            str: Formatted error message with details
        """
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message
    
    def __repr__(self) -> str:
        """
        Developer-friendly representation.
        
        Returns:
            str: Representation string
        """
        return f"{self.__class__.__name__}(message={self.message!r}, details={self.details!r})"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary for serialization.
        
        Returns:
            dict: Dictionary representation
        
        Examples:
            >>> e = AGNewsException("Error", details={"key": "value"})
            >>> e.to_dict()
            {'type': 'AGNewsException', 'message': 'Error', 'details': {'key': 'value'}}
        """
        return {
            "type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }


# ============================================================================
# Configuration Exceptions
# ============================================================================
# These exceptions are raised when configuration files are invalid, missing,
# or contain incompatible values. Configuration is critical for reproducibility
# in machine learning research.

class ConfigurationError(AGNewsException):
    """
    Base exception for configuration-related errors.
    
    Raised when YAML config files, environment variables, or runtime
    configurations are invalid or incompatible.
    
    Examples:
        >>> raise ConfigurationError(
        ...     "Invalid YAML syntax",
        ...     details={"file": "configs/model.yaml", "line": 15}
        ... )
    """
    pass


class InvalidConfigError(ConfigurationError):
    """
    Raised when configuration values are invalid.
    
    This includes type errors, out-of-range values, or logically
    inconsistent parameter combinations.
    
    Examples:
        >>> raise InvalidConfigError(
        ...     "Learning rate must be positive",
        ...     details={"learning_rate": -0.001, "valid_range": "(0, 1]"}
        ... )
    """
    pass


class MissingConfigError(ConfigurationError):
    """
    Raised when required configuration is missing.
    
    Critical for preventing silent failures when mandatory parameters
    are not specified.
    
    Examples:
        >>> raise MissingConfigError(
        ...     "Model name is required",
        ...     details={"required_field": "model_name", "config_file": "train.yaml"}
        ... )
    """
    pass


class ConfigSchemaError(ConfigurationError):
    """
    Raised when configuration does not match expected schema.
    
    Used by configuration validators to ensure type safety and
    structural correctness.
    
    Examples:
        >>> raise ConfigSchemaError(
        ...     "Expected 'batch_size' to be int, got str",
        ...     details={"field": "batch_size", "expected": "int", "actual": "str"}
        ... )
    """
    pass


# ============================================================================
# Data Exceptions
# ============================================================================
# Data exceptions cover dataset loading, preprocessing, validation, and
# augmentation errors. Data quality is paramount in ML research.

class DataError(AGNewsException):
    """
    Base exception for data-related errors.
    
    Covers all stages of data pipeline: loading, preprocessing,
    validation, and augmentation.
    """
    pass


class DataNotFoundError(DataError):
    """
    Raised when dataset files or directories are not found.
    
    Examples:
        >>> raise DataNotFoundError(
        ...     "AG News dataset not found",
        ...     details={"path": "data/raw/ag_news/train.csv", "suggestion": "Run download script"}
        ... )
    """
    pass


class DataFormatError(DataError):
    """
    Raised when data format is invalid or corrupted.
    
    Examples:
        >>> raise DataFormatError(
        ...     "CSV file missing required columns",
        ...     details={"required": ["text", "label"], "found": ["text"]}
        ... )
    """
    pass


class DataValidationError(DataError):
    """
    Raised when data validation fails.
    
    Used by data quality checks to ensure data meets requirements
    (e.g., no null values, correct label distribution).
    
    Examples:
        >>> raise DataValidationError(
        ...     "Found null values in text column",
        ...     details={"null_count": 42, "total_rows": 10000}
        ... )
    """
    pass


class InsufficientDataError(DataError):
    """
    Raised when there is insufficient data for operation.
    
    Critical for preventing training on inadequate data samples.
    
    Examples:
        >>> raise InsufficientDataError(
        ...     "Not enough training samples",
        ...     details={"required": 1000, "available": 100}
        ... )
    """
    pass


class DataLeakageError(DataError):
    """
    Raised when data leakage is detected between train/val/test sets.
    
    Critical for research integrity and overfitting prevention.
    
    Examples:
        >>> raise DataLeakageError(
        ...     "Detected overlap between train and test sets",
        ...     details={"overlap_count": 5, "test_set_hash": "abc123"}
        ... )
    """
    pass


# ============================================================================
# Model Exceptions
# ============================================================================
# Model exceptions handle errors in model creation, loading, saving, and
# initialization. Proper error handling ensures model reproducibility.

class ModelError(AGNewsException):
    """
    Base exception for model-related errors.
    
    Covers model lifecycle: creation, initialization, loading, saving,
    and inference.
    """
    pass


class ModelNotFoundError(ModelError):
    """
    Raised when model is not found in registry or filesystem.
    
    Examples:
        >>> raise ModelNotFoundError(
        ...     "Model 'deberta_v3_xlarge' not found",
        ...     details={"available_models": ["bert", "roberta"], "registry": "models"}
        ... )
    """
    pass


class ModelLoadError(ModelError):
    """
    Raised when model loading fails.
    
    Can be due to corrupted checkpoints, version mismatches, or
    missing dependencies.
    
    Examples:
        >>> raise ModelLoadError(
        ...     "Failed to load model checkpoint",
        ...     details={"checkpoint_path": "outputs/models/model.pt", "error": "File corrupted"}
        ... )
    """
    pass


class ModelSaveError(ModelError):
    """
    Raised when model saving fails.
    
    Can be due to disk space, permissions, or serialization errors.
    
    Examples:
        >>> raise ModelSaveError(
        ...     "Failed to save model checkpoint",
        ...     details={"output_path": "outputs/models/model.pt", "disk_space_available": "100MB"}
        ... )
    """
    pass


class ModelInitializationError(ModelError):
    """
    Raised when model initialization fails.
    
    Examples:
        >>> raise ModelInitializationError(
        ...     "Failed to initialize DeBERTa model",
        ...     details={"model_name": "microsoft/deberta-v3-large", "error": "CUDA out of memory"}
        ... )
    """
    pass


class IncompatibleModelError(ModelError):
    """
    Raised when model is incompatible with current configuration.
    
    Examples:
        >>> raise IncompatibleModelError(
        ...     "Model requires 4 classes, config specifies 2",
        ...     details={"model_classes": 4, "config_classes": 2}
        ... )
    """
    pass


class ModelArchitectureError(ModelError):
    """
    Raised when model architecture is invalid or unsupported.
    
    Examples:
        >>> raise ModelArchitectureError(
        ...     "Unsupported model architecture",
        ...     details={"architecture": "gpt-neo", "supported": ["bert", "roberta", "deberta"]}
        ... )
    """
    pass


# ============================================================================
# Training Exceptions
# ============================================================================
# Training exceptions handle errors during model training, including
# convergence issues, memory errors, and checkpoint failures.

class TrainingError(AGNewsException):
    """
    Base exception for training-related errors.
    
    Covers the entire training lifecycle from initialization to
    final evaluation.
    """
    pass


class TrainingInterruptedError(TrainingError):
    """
    Raised when training is interrupted unexpectedly.
    
    Examples:
        >>> raise TrainingInterruptedError(
        ...     "Training interrupted at epoch 5",
        ...     details={"completed_epochs": 5, "total_epochs": 10, "checkpoint_saved": True}
        ... )
    """
    pass


class ConvergenceError(TrainingError):
    """
    Raised when model fails to converge.
    
    Examples:
        >>> raise ConvergenceError(
        ...     "Model failed to converge after 100 epochs",
        ...     details={"final_loss": 2.5, "min_loss_threshold": 0.1}
        ... )
    """
    pass


class CheckpointError(TrainingError):
    """
    Raised when checkpoint operations fail.
    
    Examples:
        >>> raise CheckpointError(
        ...     "Failed to save checkpoint",
        ...     details={"epoch": 10, "path": "outputs/checkpoints/epoch_10.pt"}
        ... )
    """
    pass


class OutOfMemoryError(TrainingError):
    """
    Raised when running out of memory during training.
    
    Provides suggestions for memory optimization.
    
    Examples:
        >>> raise OutOfMemoryError(
        ...     "CUDA out of memory",
        ...     batch_size=64,
        ...     gpu_memory_allocated=15.2,
        ...     gpu_memory_total=16.0
        ... )
    """
    
    def __init__(
        self,
        message: str = "Out of memory during training",
        batch_size: Optional[int] = None,
        gpu_memory_allocated: Optional[float] = None,
        gpu_memory_total: Optional[float] = None,
        **kwargs
    ):
        """
        Initialize with memory-specific details.
        
        Args:
            message (str): Error message
            batch_size (int, optional): Current batch size
            gpu_memory_allocated (float, optional): Allocated GPU memory in GB
            gpu_memory_total (float, optional): Total GPU memory in GB
            **kwargs: Additional details
        """
        details = kwargs
        if batch_size:
            details["batch_size"] = batch_size
            details["suggestion"] = f"Try reducing batch size from {batch_size} to {batch_size // 2}"
        if gpu_memory_allocated and gpu_memory_total:
            details["gpu_memory_allocated_gb"] = gpu_memory_allocated
            details["gpu_memory_total_gb"] = gpu_memory_total
            details["memory_usage_percent"] = (gpu_memory_allocated / gpu_memory_total) * 100
        
        super().__init__(message, details)


class GradientError(TrainingError):
    """
    Raised when gradient computation or update fails.
    
    Examples:
        >>> raise GradientError(
        ...     "NaN detected in gradients",
        ...     details={"step": 1000, "gradient_norm": float('nan')}
        ... )
    """
    pass


class OverfittingDetectedError(TrainingError):
    """
    Raised when overfitting is detected by monitoring system.
    
    Critical for academic research integrity and model validation.
    
    Examples:
        >>> raise OverfittingDetectedError(
        ...     "Severe overfitting detected",
        ...     details={
        ...         "train_accuracy": 0.99,
        ...         "val_accuracy": 0.75,
        ...         "gap": 0.24,
        ...         "threshold": 0.10
        ...     }
        ... )
    """
    pass


# ============================================================================
# Registry and Factory Exceptions
# ============================================================================
# These exceptions handle component registration and factory pattern errors.

class RegistryError(AGNewsException):
    """
    Base exception for registry operations.
    
    Raised when component registration, retrieval, or validation fails.
    """
    pass


class ComponentNotFoundError(RegistryError):
    """
    Raised when component is not found in registry.
    
    Examples:
        >>> raise ComponentNotFoundError(
        ...     "Trainer 'custom_trainer' not registered",
        ...     details={"registry": "trainers", "available": ["standard", "distributed"]}
        ... )
    """
    pass


class ComponentAlreadyRegisteredError(RegistryError):
    """
    Raised when attempting to register a component that already exists.
    
    Examples:
        >>> raise ComponentAlreadyRegisteredError(
        ...     "Model 'bert' already registered",
        ...     details={"registry": "models", "use_override": True}
        ... )
    """
    pass


class FactoryError(AGNewsException):
    """
    Base exception for factory operations.
    
    Raised when component creation through factory pattern fails.
    """
    pass


class ComponentCreationError(FactoryError):
    """
    Raised when component creation fails.
    
    Examples:
        >>> raise ComponentCreationError(
        ...     "Failed to create model instance",
        ...     details={"model_name": "deberta", "error": "Missing required parameter: num_classes"}
        ... )
    """
    pass


# ============================================================================
# Validation Exceptions
# ============================================================================
# Validation exceptions for input validation, schema validation, and
# constraint checking.

class ValidationError(AGNewsException):
    """
    Base exception for validation errors.
    
    Used for input validation, schema validation, and constraint checking.
    
    Examples:
        >>> raise ValidationError(
        ...     "Input validation failed",
        ...     details={"errors": {"batch_size": "must be positive"}}
        ... )
    """
    pass


class SchemaValidationError(ValidationError):
    """
    Raised when schema validation fails.
    
    Examples:
        >>> raise SchemaValidationError(
        ...     "Config does not match schema",
        ...     details={"field": "learning_rate", "expected_type": "float", "actual_type": "str"}
        ... )
    """
    pass


class ConstraintViolationError(ValidationError):
    """
    Raised when a constraint is violated.
    
    Examples:
        >>> raise ConstraintViolationError(
        ...     "Model size constraint violated",
        ...     details={"max_parameters": 1e9, "actual_parameters": 1.5e9}
        ... )
    """
    pass


# ============================================================================
# API Exceptions
# ============================================================================
# API exceptions for REST API, authentication, and authorization errors.

class APIError(AGNewsException):
    """
    Base exception for API-related errors.
    
    Includes HTTP status code for REST API responses.
    
    Attributes:
        status_code (int): HTTP status code
    """
    
    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize with HTTP status code.
        
        Args:
            message (str): Error message
            status_code (int): HTTP status code (default: 500)
            details (dict, optional): Additional error details
        """
        super().__init__(message, details)
        self.status_code = status_code


class AuthenticationError(APIError):
    """
    Raised when authentication fails.
    
    Examples:
        >>> raise AuthenticationError("Invalid API key")
    """
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, status_code=401, **kwargs)


class AuthorizationError(APIError):
    """
    Raised when authorization fails.
    
    Examples:
        >>> raise AuthorizationError("Insufficient permissions")
    """
    
    def __init__(self, message: str = "Unauthorized access", **kwargs):
        super().__init__(message, status_code=403, **kwargs)


class RateLimitError(APIError):
    """
    Raised when rate limit is exceeded.
    
    Examples:
        >>> raise RateLimitError("Rate limit exceeded", retry_after=60)
    """
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if retry_after:
            details["retry_after_seconds"] = retry_after
        super().__init__(message, status_code=429, details=details)


class RequestValidationError(APIError):
    """
    Raised when API request validation fails.
    
    Examples:
        >>> raise RequestValidationError(
        ...     "Invalid request body",
        ...     errors={"text": "required field missing"}
        ... )
    """
    
    def __init__(
        self,
        message: str = "Request validation failed",
        errors: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        details = kwargs.get("details", {})
        if errors:
            details["validation_errors"] = errors
        super().__init__(message, status_code=422, details=details)


# ============================================================================
# Inference Exceptions
# ============================================================================
# Exceptions raised during model inference and prediction.

class InferenceError(AGNewsException):
    """
    Base exception for inference-related errors.
    
    Raised during model prediction, batch inference, or real-time serving.
    """
    pass


class PredictionError(InferenceError):
    """
    Raised when prediction fails.
    
    Examples:
        >>> raise PredictionError(
        ...     "Failed to generate prediction",
        ...     details={"input_text": "Breaking news...", "error": "Invalid input format"}
        ... )
    """
    pass


class BatchInferenceError(InferenceError):
    """
    Raised when batch inference fails.
    
    Examples:
        >>> raise BatchInferenceError(
        ...     "Batch inference failed",
        ...     details={"batch_size": 32, "failed_at_index": 15}
        ... )
    """
    pass


# ============================================================================
# External Service Exceptions
# ============================================================================
# Exceptions for external API calls and service integrations.

class ExternalServiceError(AGNewsException):
    """
    Base exception for external service errors.
    
    Raised when external API calls or service integrations fail.
    """
    pass


class HuggingFaceError(ExternalServiceError):
    """
    Raised when Hugging Face Hub operations fail.
    
    Examples:
        >>> raise HuggingFaceError(
        ...     "Failed to download model",
        ...     details={"model_id": "microsoft/deberta-v3-large", "error": "Network timeout"}
        ... )
    """
    pass


class OpenAIError(ExternalServiceError):
    """
    Raised when OpenAI API calls fail.
    
    Examples:
        >>> raise OpenAIError(
        ...     "GPT-4 API call failed",
        ...     details={"endpoint": "/v1/chat/completions", "status_code": 429}
        ... )
    """
    pass


class CloudStorageError(ExternalServiceError):
    """
    Raised when cloud storage operations fail.
    
    Examples:
        >>> raise CloudStorageError(
        ...     "Failed to upload to S3",
        ...     details={"bucket": "my-bucket", "key": "models/model.pt"}
        ... )
    """
    pass


# ============================================================================
# Evaluation Exceptions
# ============================================================================
# Exceptions for model evaluation and metrics computation.

class EvaluationError(AGNewsException):
    """
    Base exception for evaluation-related errors.
    
    Raised during model evaluation, metrics computation, or benchmarking.
    """
    pass


class MetricComputationError(EvaluationError):
    """
    Raised when metric computation fails.
    
    Examples:
        >>> raise MetricComputationError(
        ...     "Failed to compute F1 score",
        ...     details={"predictions": 100, "labels": 95, "error": "Length mismatch"}
        ... )
    """
    pass


class BenchmarkError(EvaluationError):
    """
    Raised when benchmarking fails.
    
    Examples:
        >>> raise BenchmarkError(
        ...     "Benchmark execution failed",
        ...     details={"benchmark": "speed_test", "error": "Model not loaded"}
        ... )
    """
    pass


# ============================================================================
# Optimization Exceptions
# ============================================================================
# Exceptions for model optimization techniques.

class OptimizationError(AGNewsException):
    """
    Base exception for optimization-related errors.
    
    Raised during quantization, pruning, distillation, etc.
    """
    pass


class QuantizationError(OptimizationError):
    """
    Raised when quantization fails.
    
    Examples:
        >>> raise QuantizationError(
        ...     "INT8 quantization failed",
        ...     details={"method": "dynamic", "error": "Unsupported layer type"}
        ... )
    """
    pass


class PruningError(OptimizationError):
    """
    Raised when model pruning fails.
    
    Examples:
        >>> raise PruningError(
        ...     "Magnitude pruning failed",
        ...     details={"target_sparsity": 0.5, "achieved_sparsity": 0.3}
        ... )
    """
    pass


class DistillationError(OptimizationError):
    """
    Raised when knowledge distillation fails.
    
    Examples:
        >>> raise DistillationError(
        ...     "Distillation from teacher failed",
        ...     details={"teacher": "llama2-13b", "student": "deberta-large"}
        ... )
    """
    pass


# ============================================================================
# Experiment Tracking Exceptions
# ============================================================================
# Exceptions for experiment management and tracking.

class ExperimentError(AGNewsException):
    """
    Base exception for experiment-related errors.
    
    Raised during experiment creation, tracking, or retrieval.
    """
    pass


class ExperimentNotFoundError(ExperimentError):
    """
    Raised when experiment is not found.
    
    Examples:
        >>> raise ExperimentNotFoundError(
        ...     "Experiment 'exp_123' not found",
        ...     details={"experiment_id": "exp_123", "tracking_system": "mlflow"}
        ... )
    """
    pass


class ExperimentTrackingError(ExperimentError):
    """
    Raised when experiment tracking fails.
    
    Examples:
        >>> raise ExperimentTrackingError(
        ...     "Failed to log metrics",
        ...     details={"metrics": {"accuracy": 0.95}, "error": "MLflow server unavailable"}
        ... )
    """
    pass


# ============================================================================
# Platform Exceptions
# ============================================================================
# Exceptions for platform detection and compatibility.

class PlatformError(AGNewsException):
    """
    Base exception for platform-related errors.
    
    Raised during platform detection or compatibility checks.
    """
    pass


class PlatformNotSupportedError(PlatformError):
    """
    Raised when platform is not supported.
    
    Examples:
        >>> raise PlatformNotSupportedError(
        ...     "Platform 'xyz' not supported",
        ...     details={"platform": "xyz", "supported": ["colab", "kaggle", "local"]}
        ... )
    """
    pass


class PlatformDetectionError(PlatformError):
    """
    Raised when platform detection fails.
    
    Examples:
        >>> raise PlatformDetectionError(
        ...     "Failed to detect platform",
        ...     details={"error": "Ambiguous environment variables"}
        ... )
    """
    pass


# ============================================================================
# Utility Functions
# ============================================================================
# Helper functions for error handling and validation.

def handle_gpu_error(func: Callable) -> Callable:
    """
    Decorator to handle GPU-related errors gracefully.
    
    Automatically catches CUDA out-of-memory errors and provides
    helpful suggestions for resolution.
    
    Args:
        func (Callable): Function to wrap
    
    Returns:
        Callable: Wrapped function
    
    Examples:
        >>> @handle_gpu_error
        ... def train_model(model, batch_size=32):
        ...     return model.train(batch_size=batch_size)
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            error_msg = str(e).lower()
            if "out of memory" in error_msg or "cuda" in error_msg:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                raise OutOfMemoryError(
                    "GPU out of memory during execution",
                    batch_size=kwargs.get("batch_size"),
                    details={"original_error": str(e), "function": func.__name__}
                ) from e
            raise ModelError(f"GPU error in {func.__name__}: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    
    return wrapper


def raise_if_not_installed(
    package_name: str,
    purpose: str = "",
    install_command: Optional[str] = None
) -> None:
    """
    Raise ImportError if required package is not installed.
    
    Provides clear error messages with installation instructions.
    
    Args:
        package_name (str): Name of the required package
        purpose (str, optional): What the package is needed for
        install_command (str, optional): Custom installation command
    
    Raises:
        ImportError: If package is not installed
    
    Examples:
        >>> raise_if_not_installed("peft", "LoRA training")
        ImportError: Package 'peft' is required for LoRA training.
        Install it with: pip install peft
    """
    try:
        __import__(package_name)
    except ImportError as e:
        purpose_msg = f" for {purpose}" if purpose else ""
        install_cmd = install_command or f"pip install {package_name}"
        
        error_message = (
            f"Package '{package_name}' is required{purpose_msg}. "
            f"Install it with: {install_cmd}"
        )
        
        raise ImportError(error_message) from e


def validate_version_compatibility(
    package_name: str,
    min_version: Optional[str] = None,
    max_version: Optional[str] = None
) -> bool:
    """
    Validate package version compatibility.
    
    Args:
        package_name (str): Name of the package
        min_version (str, optional): Minimum required version
        max_version (str, optional): Maximum supported version
    
    Returns:
        bool: True if compatible
    
    Raises:
        ImportError: If package not installed
        ValidationError: If version incompatible
    
    Examples:
        >>> validate_version_compatibility("transformers", min_version="4.35.0")
        True
    """
    try:
        import importlib.metadata
        from packaging import version
        
        installed_version = importlib.metadata.version(package_name)
        installed = version.parse(installed_version)
        
        if min_version and installed < version.parse(min_version):
            raise ValidationError(
                f"Package '{package_name}' version too old",
                details={
                    "installed": installed_version,
                    "minimum_required": min_version,
                    "suggestion": f"pip install --upgrade {package_name}>={min_version}"
                }
            )
        
        if max_version and installed >= version.parse(max_version):
            raise ValidationError(
                f"Package '{package_name}' version too new",
                details={
                    "installed": installed_version,
                    "maximum_supported": max_version,
                    "suggestion": f"pip install {package_name}<{max_version}"
                }
            )
        
        return True
    
    except importlib.metadata.PackageNotFoundError:
        raise_if_not_installed(package_name)


# Export public API
__all__ = [
    # Base exception
    "AGNewsException",
    
    # Configuration exceptions
    "ConfigurationError",
    "InvalidConfigError",
    "MissingConfigError",
    "ConfigSchemaError",
    
    # Data exceptions
    "DataError",
    "DataNotFoundError",
    "DataFormatError",
    "DataValidationError",
    "InsufficientDataError",
    "DataLeakageError",
    
    # Model exceptions
    "ModelError",
    "ModelNotFoundError",
    "ModelLoadError",
    "ModelSaveError",
    "ModelInitializationError",
    "IncompatibleModelError",
    "ModelArchitectureError",
    
    # Training exceptions
    "TrainingError",
    "TrainingInterruptedError",
    "ConvergenceError",
    "CheckpointError",
    "OutOfMemoryError",
    "GradientError",
    "OverfittingDetectedError",
    
    # Registry and factory exceptions
    "RegistryError",
    "ComponentNotFoundError",
    "ComponentAlreadyRegisteredError",
    "FactoryError",
    "ComponentCreationError",
    
    # Validation exceptions
    "ValidationError",
    "SchemaValidationError",
    "ConstraintViolationError",
    
    # API exceptions
    "APIError",
    "AuthenticationError",
    "AuthorizationError",
    "RateLimitError",
    "RequestValidationError",
    
    # Inference exceptions
    "InferenceError",
    "PredictionError",
    "BatchInferenceError",
    
    # External service exceptions
    "ExternalServiceError",
    "HuggingFaceError",
    "OpenAIError",
    "CloudStorageError",
    
    # Evaluation exceptions
    "EvaluationError",
    "MetricComputationError",
    "BenchmarkError",
    
    # Optimization exceptions
    "OptimizationError",
    "QuantizationError",
    "PruningError",
    "DistillationError",
    
    # Experiment exceptions
    "ExperimentError",
    "ExperimentNotFoundError",
    "ExperimentTrackingError",
    
    # Platform exceptions
    "PlatformError",
    "PlatformNotSupportedError",
    "PlatformDetectionError",
    
    # Utility functions
    "handle_gpu_error",
    "raise_if_not_installed",
    "validate_version_compatibility",
]
