"""
Custom Validators for REST API
================================================================================
Implements advanced validation logic for API requests including business rules,
data integrity checks, and security validations.

This module provides reusable validators following the Specification pattern
for complex validation scenarios beyond basic type checking.

References:
    - Evans, E. (2003). Domain-Driven Design: Specification Pattern
    - OWASP Input Validation Cheat Sheet
    - Bean Validation specification (JSR 380)

Author: Võ Hải Dũng
License: MIT
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone

from pydantic import BaseModel, validator, root_validator
from fastapi import HTTPException

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TextValidator:
    """
    Validator for text input in classification requests.
    
    Implements comprehensive text validation including length checks,
    encoding validation, and content filtering.
    """
    
    # Validation constants
    MIN_TEXT_LENGTH = 1
    MAX_TEXT_LENGTH = 10000
    MIN_WORDS = 1
    MAX_WORDS = 2000
    
    # Regex patterns
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\KATEX_INLINE_OPEN\KATEX_INLINE_CLOSE,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    EMAIL_PATTERN = re.compile(
        r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    )
    SCRIPT_TAG_PATTERN = re.compile(
        r'<script[^>]*>.*?</script>', 
        re.IGNORECASE | re.DOTALL
    )
    SQL_INJECTION_PATTERN = re.compile(
        r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)',
        re.IGNORECASE
    )
    
    @classmethod
    def validate_text_length(cls, text: str) -> str:
        """
        Validate text length constraints.
        
        Args:
            text: Input text
            
        Returns:
            Validated text
            
        Raises:
            ValueError: If text length is invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        
        text = text.strip()
        
        if len(text) < cls.MIN_TEXT_LENGTH:
            raise ValueError(f"Text must be at least {cls.MIN_TEXT_LENGTH} characters")
        
        if len(text) > cls.MAX_TEXT_LENGTH:
            raise ValueError(f"Text exceeds maximum length of {cls.MAX_TEXT_LENGTH} characters")
        
        return text
    
    @classmethod
    def validate_word_count(cls, text: str) -> str:
        """
        Validate word count in text.
        
        Args:
            text: Input text
            
        Returns:
            Validated text
            
        Raises:
            ValueError: If word count is invalid
        """
        words = text.split()
        word_count = len(words)
        
        if word_count < cls.MIN_WORDS:
            raise ValueError(f"Text must contain at least {cls.MIN_WORDS} word(s)")
        
        if word_count > cls.MAX_WORDS:
            raise ValueError(f"Text exceeds maximum of {cls.MAX_WORDS} words")
        
        return text
    
    @classmethod
    def validate_encoding(cls, text: str) -> str:
        """
        Validate text encoding and normalize.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
            
        Raises:
            ValueError: If text has encoding issues
        """
        try:
            # Ensure valid UTF-8
            text.encode('utf-8').decode('utf-8')
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            # Remove zero-width characters
            text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '')
            
            return text
            
        except UnicodeError as e:
            raise ValueError(f"Text contains invalid characters: {str(e)}")
    
    @classmethod
    def validate_content_safety(cls, text: str) -> str:
        """
        Validate text for potentially harmful content.
        
        Args:
            text: Input text
            
        Returns:
            Validated text
            
        Raises:
            ValueError: If harmful content detected
        """
        # Check for script tags (XSS prevention)
        if cls.SCRIPT_TAG_PATTERN.search(text):
            raise ValueError("Text contains potentially harmful script content")
        
        # Check for SQL injection patterns
        if cls.SQL_INJECTION_PATTERN.search(text):
            logger.warning("Potential SQL injection pattern detected in text")
            # Don't reject, but log for monitoring
        
        # Check for excessive URLs
        urls = cls.URL_PATTERN.findall(text)
        if len(urls) > 5:
            raise ValueError("Text contains too many URLs")
        
        return text
    
    @classmethod
    def validate_language(cls, text: str, allowed_languages: List[str] = None) -> str:
        """
        Validate text language (simplified check).
        
        Args:
            text: Input text
            allowed_languages: List of allowed language codes
            
        Returns:
            Validated text
        """
        # Simple ASCII check for English
        if allowed_languages and 'en' in allowed_languages:
            non_ascii_ratio = sum(1 for c in text if ord(c) > 127) / len(text)
            if non_ascii_ratio > 0.3:
                logger.warning("Text contains high ratio of non-ASCII characters")
        
        return text
    
    @classmethod
    def validate_all(cls, text: str) -> str:
        """
        Apply all text validations.
        
        Args:
            text: Input text
            
        Returns:
            Fully validated text
            
        Raises:
            ValueError: If any validation fails
        """
        text = cls.validate_text_length(text)
        text = cls.validate_encoding(text)
        text = cls.validate_word_count(text)
        text = cls.validate_content_safety(text)
        text = cls.validate_language(text, allowed_languages=['en'])
        
        return text


class ModelValidator:
    """
    Validator for model selection and configuration.
    
    Ensures valid model names, versions, and configurations.
    """
    
    ALLOWED_MODELS = [
        "deberta-v3-large",
        "roberta-large",
        "xlnet-large",
        "electra-large",
        "longformer-large",
        "ensemble",
        "default"
    ]
    
    ENSEMBLE_MODELS = ["ensemble", "voting", "stacking"]
    
    @classmethod
    def validate_model_name(cls, model_name: str) -> str:
        """
        Validate model name.
        
        Args:
            model_name: Model name to validate
            
        Returns:
            Validated model name
            
        Raises:
            ValueError: If model name is invalid
        """
        if not model_name:
            return "default"
        
        model_name = model_name.lower().strip()
        
        if model_name not in cls.ALLOWED_MODELS:
            raise ValueError(
                f"Invalid model name '{model_name}'. "
                f"Allowed models: {', '.join(cls.ALLOWED_MODELS)}"
            )
        
        return model_name
    
    @classmethod
    def validate_model_version(cls, version: str) -> str:
        """
        Validate model version format.
        
        Args:
            version: Version string
            
        Returns:
            Validated version
            
        Raises:
            ValueError: If version format is invalid
        """
        if not version:
            return "latest"
        
        # Version pattern: major.minor.patch or date-based
        version_pattern = re.compile(r'^\d+\.\d+\.\d+$|^\d{8}_\d{6}$|^latest$')
        
        if not version_pattern.match(version):
            raise ValueError(
                f"Invalid version format '{version}'. "
                "Use semantic versioning (1.0.0) or date format (YYYYMMDD_HHMMSS)"
            )
        
        return version
    
    @classmethod
    def validate_ensemble_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate ensemble model configuration.
        
        Args:
            config: Ensemble configuration
            
        Returns:
            Validated configuration
            
        Raises:
            ValueError: If configuration is invalid
        """
        if not config:
            return {}
        
        # Validate voting weights
        if "voting_weights" in config:
            weights = config["voting_weights"]
            if not isinstance(weights, dict):
                raise ValueError("Voting weights must be a dictionary")
            
            total_weight = sum(weights.values())
            if abs(total_weight - 1.0) > 0.001:
                raise ValueError("Voting weights must sum to 1.0")
        
        # Validate model list
        if "models" in config:
            models = config["models"]
            if not isinstance(models, list) or len(models) < 2:
                raise ValueError("Ensemble must contain at least 2 models")
            
            for model in models:
                cls.validate_model_name(model)
        
        return config


class TrainingValidator:
    """
    Validator for training configuration and parameters.
    
    Ensures valid hyperparameters and training settings.
    """
    
    # Training constraints
    MIN_EPOCHS = 1
    MAX_EPOCHS = 100
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 256
    MIN_LEARNING_RATE = 1e-7
    MAX_LEARNING_RATE = 1e-2
    
    @classmethod
    def validate_epochs(cls, epochs: int) -> int:
        """
        Validate number of epochs.
        
        Args:
            epochs: Number of epochs
            
        Returns:
            Validated epochs
            
        Raises:
            ValueError: If epochs is invalid
        """
        if not isinstance(epochs, int):
            raise ValueError("Epochs must be an integer")
        
        if epochs < cls.MIN_EPOCHS:
            raise ValueError(f"Epochs must be at least {cls.MIN_EPOCHS}")
        
        if epochs > cls.MAX_EPOCHS:
            raise ValueError(f"Epochs cannot exceed {cls.MAX_EPOCHS}")
        
        return epochs
    
    @classmethod
    def validate_batch_size(cls, batch_size: int) -> int:
        """
        Validate batch size.
        
        Args:
            batch_size: Batch size
            
        Returns:
            Validated batch size
            
        Raises:
            ValueError: If batch size is invalid
        """
        if not isinstance(batch_size, int):
            raise ValueError("Batch size must be an integer")
        
        if batch_size < cls.MIN_BATCH_SIZE:
            raise ValueError(f"Batch size must be at least {cls.MIN_BATCH_SIZE}")
        
        if batch_size > cls.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size cannot exceed {cls.MAX_BATCH_SIZE}")
        
        # Check if power of 2 (recommended)
        if batch_size & (batch_size - 1) != 0:
            logger.warning(f"Batch size {batch_size} is not a power of 2")
        
        return batch_size
    
    @classmethod
    def validate_learning_rate(cls, learning_rate: float) -> float:
        """
        Validate learning rate.
        
        Args:
            learning_rate: Learning rate
            
        Returns:
            Validated learning rate
            
        Raises:
            ValueError: If learning rate is invalid
        """
        if not isinstance(learning_rate, (int, float)):
            raise ValueError("Learning rate must be a number")
        
        learning_rate = float(learning_rate)
        
        if learning_rate < cls.MIN_LEARNING_RATE:
            raise ValueError(f"Learning rate must be at least {cls.MIN_LEARNING_RATE}")
        
        if learning_rate > cls.MAX_LEARNING_RATE:
            raise ValueError(f"Learning rate cannot exceed {cls.MAX_LEARNING_RATE}")
        
        return learning_rate
    
    @classmethod
    def validate_validation_split(cls, split: float) -> float:
        """
        Validate validation split ratio.
        
        Args:
            split: Validation split ratio
            
        Returns:
            Validated split
            
        Raises:
            ValueError: If split is invalid
        """
        if not isinstance(split, (int, float)):
            raise ValueError("Validation split must be a number")
        
        split = float(split)
        
        if split < 0.0 or split > 0.5:
            raise ValueError("Validation split must be between 0.0 and 0.5")
        
        return split


class DataValidator:
    """
    Validator for data upload and management operations.
    
    Ensures data quality and format compliance.
    """
    
    ALLOWED_FORMATS = ["csv", "json", "jsonl", "tsv", "txt"]
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    MAX_DATASET_NAME_LENGTH = 100
    DATASET_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$')
    
    @classmethod
    def validate_dataset_name(cls, name: str) -> str:
        """
        Validate dataset name.
        
        Args:
            name: Dataset name
            
        Returns:
            Validated name
            
        Raises:
            ValueError: If name is invalid
        """
        if not name:
            raise ValueError("Dataset name is required")
        
        name = name.strip()
        
        if len(name) > cls.MAX_DATASET_NAME_LENGTH:
            raise ValueError(
                f"Dataset name cannot exceed {cls.MAX_DATASET_NAME_LENGTH} characters"
            )
        
        if not cls.DATASET_NAME_PATTERN.match(name):
            raise ValueError(
                "Dataset name must start with alphanumeric and "
                "contain only letters, numbers, hyphens, and underscores"
            )
        
        # Check for reserved names
        reserved_names = ["test", "train", "validation", "system", "default"]
        if name.lower() in reserved_names:
            raise ValueError(f"'{name}' is a reserved dataset name")
        
        return name
    
    @classmethod
    def validate_file_format(cls, format: str) -> str:
        """
        Validate file format.
        
        Args:
            format: File format
            
        Returns:
            Validated format
            
        Raises:
            ValueError: If format is not supported
        """
        format = format.lower().strip()
        
        if format not in cls.ALLOWED_FORMATS:
            raise ValueError(
                f"Unsupported format '{format}'. "
                f"Allowed formats: {', '.join(cls.ALLOWED_FORMATS)}"
            )
        
        return format
    
    @classmethod
    def validate_file_size(cls, size: int) -> int:
        """
        Validate file size.
        
        Args:
            size: File size in bytes
            
        Returns:
            Validated size
            
        Raises:
            ValueError: If file is too large
        """
        if size > cls.MAX_FILE_SIZE:
            raise ValueError(
                f"File size exceeds maximum of {cls.MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )
        
        if size == 0:
            raise ValueError("File is empty")
        
        return size


class BatchValidator:
    """
    Validator for batch processing requests.
    
    Ensures batch constraints and consistency.
    """
    
    MIN_BATCH_SIZE = 1
    MAX_BATCH_SIZE = 100
    MAX_TOTAL_TEXT_LENGTH = 100000
    
    @classmethod
    def validate_batch_size(cls, items: List[Any]) -> List[Any]:
        """
        Validate batch size.
        
        Args:
            items: Batch items
            
        Returns:
            Validated items
            
        Raises:
            ValueError: If batch size is invalid
        """
        if not items:
            raise ValueError("Batch cannot be empty")
        
        if len(items) < cls.MIN_BATCH_SIZE:
            raise ValueError(f"Batch must contain at least {cls.MIN_BATCH_SIZE} item(s)")
        
        if len(items) > cls.MAX_BATCH_SIZE:
            raise ValueError(f"Batch size exceeds maximum of {cls.MAX_BATCH_SIZE}")
        
        return items
    
    @classmethod
    def validate_batch_consistency(cls, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate batch item consistency.
        
        Args:
            items: Batch items
            
        Returns:
            Validated items
            
        Raises:
            ValueError: If items are inconsistent
        """
        if not items:
            return items
        
        # Check that all items have same structure
        first_keys = set(items[0].keys())
        
        for idx, item in enumerate(items[1:], 1):
            if set(item.keys()) != first_keys:
                raise ValueError(f"Inconsistent item structure at index {idx}")
        
        return items
    
    @classmethod
    def validate_batch_text_length(cls, texts: List[str]) -> List[str]:
        """
        Validate total text length in batch.
        
        Args:
            texts: List of texts
            
        Returns:
            Validated texts
            
        Raises:
            ValueError: If total length exceeds limit
        """
        total_length = sum(len(text) for text in texts)
        
        if total_length > cls.MAX_TOTAL_TEXT_LENGTH:
            raise ValueError(
                f"Total text length {total_length} exceeds "
                f"maximum of {cls.MAX_TOTAL_TEXT_LENGTH} characters"
            )
        
        return texts


class RequestContextValidator:
    """
    Validator for request context and metadata.
    
    Ensures valid request context for tracking and auditing.
    """
    
    @staticmethod
    def validate_request_id(request_id: str) -> str:
        """
        Validate request ID format.
        
        Args:
            request_id: Request ID
            
        Returns:
            Validated request ID
            
        Raises:
            ValueError: If request ID is invalid
        """
        uuid_pattern = re.compile(
            r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            re.IGNORECASE
        )
        
        if not uuid_pattern.match(request_id):
            raise ValueError("Invalid request ID format (expected UUID)")
        
        return request_id.lower()
    
    @staticmethod
    def validate_timestamp(timestamp: Union[str, datetime]) -> datetime:
        """
        Validate and parse timestamp.
        
        Args:
            timestamp: Timestamp string or datetime
            
        Returns:
            Validated datetime
            
        Raises:
            ValueError: If timestamp is invalid
        """
        if isinstance(timestamp, datetime):
            return timestamp
        
        try:
            # Try ISO format
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Ensure timezone aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            
            return dt
            
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid timestamp format: {str(e)}")
    
    @staticmethod
    def validate_user_agent(user_agent: str) -> str:
        """
        Validate user agent string.
        
        Args:
            user_agent: User agent string
            
        Returns:
            Validated user agent
        """
        if not user_agent:
            return "Unknown"
        
        # Truncate if too long
        max_length = 500
        if len(user_agent) > max_length:
            user_agent = user_agent[:max_length]
        
        # Basic sanitization
        user_agent = user_agent.replace('\n', ' ').replace('\r', ' ')
        
        return user_agent


def validate_pagination(page: int, page_size: int) -> tuple[int, int]:
    """
    Validate pagination parameters.
    
    Args:
        page: Page number
        page_size: Items per page
        
    Returns:
        Validated (page, page_size)
        
    Raises:
        HTTPException: If parameters are invalid
    """
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be >= 1")
    
    if page_size < 1:
        raise HTTPException(status_code=400, detail="Page size must be >= 1")
    
    if page_size > 100:
        raise HTTPException(status_code=400, detail="Page size cannot exceed 100")
    
    return page, page_size


def validate_date_range(start_date: str, end_date: str) -> tuple[datetime, datetime]:
    """
    Validate date range.
    
    Args:
        start_date: Start date string
        end_date: End date string
        
    Returns:
        Validated (start_datetime, end_datetime)
        
    Raises:
        HTTPException: If date range is invalid
    """
    try:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if start > end:
            raise HTTPException(
                status_code=400,
                detail="Start date must be before end date"
            )
        
        # Check maximum range (e.g., 1 year)
        max_days = 365
        if (end - start).days > max_days:
            raise HTTPException(
                status_code=400,
                detail=f"Date range cannot exceed {max_days} days"
            )
        
        return start, end
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid date format: {str(e)}"
        )
