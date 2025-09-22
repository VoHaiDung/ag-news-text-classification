"""
REST API Validators
===================

Implements input validation following principles from:
- Martin (2017): "Clean Architecture"
- Evans (2003): "Domain-Driven Design"
- OWASP (2021): "Input Validation Cheat Sheet"

Author: Team SOTA AGNews
License: MIT
"""

import re
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import json
from datetime import datetime

from pydantic import BaseModel, Field, validator, ValidationError
from fastapi import HTTPException, status
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.core.exceptions import ValidationError as CustomValidationError
from configs.constants import (
    AG_NEWS_CLASSES,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH
)

logger = setup_logging(__name__)

# Validation patterns following OWASP guidelines
PATTERNS = {
    "email": re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"),
    "url": re.compile(r"^https?://[^\s/$.?#].[^\s]*$"),
    "html_tag": re.compile(r"<[^>]+>"),
    "script_tag": re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),
    "sql_injection": re.compile(
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|CREATE|ALTER)\b)",
        re.IGNORECASE
    ),
    "special_chars": re.compile(r"[<>\"'`;(){}]")
}

# Model constraints
MODEL_CONSTRAINTS = {
    "deberta-v3": {
        "max_length": 512,
        "min_length": 10,
        "supported_languages": ["en"]
    },
    "roberta-large": {
        "max_length": 512,
        "min_length": 10,
        "supported_languages": ["en"]
    },
    "xlnet-large": {
        "max_length": 512,
        "min_length": 10,
        "supported_languages": ["en"]
    },
    "ensemble": {
        "max_length": 512,
        "min_length": 10,
        "supported_languages": ["en"]
    }
}

def validate_text_input(text: str) -> str:
    """
    Validate single text input.
    
    Implements validation strategies from:
    - OWASP (2021): "Input Validation Cheat Sheet"
    - Stallings (2017): "Computer Security: Principles and Practice"
    
    Args:
        text: Input text to validate
        
    Returns:
        Validated and sanitized text
        
    Raises:
        ValidationError: If validation fails
    """
    if not text or not isinstance(text, str):
        raise CustomValidationError("Text input must be a non-empty string")
    
    # Length validation
    text_length = len(text.strip())
    if text_length < MIN_SEQUENCE_LENGTH:
        raise CustomValidationError(
            f"Text too short. Minimum length: {MIN_SEQUENCE_LENGTH} characters"
        )
    
    if text_length > MAX_SEQUENCE_LENGTH * 10:  # Allow some buffer
        raise CustomValidationError(
            f"Text too long. Maximum length: {MAX_SEQUENCE_LENGTH * 10} characters"
        )
    
    # Security validation - check for potential injection attacks
    if PATTERNS["sql_injection"].search(text):
        logger.warning("Potential SQL injection attempt detected")
        # Don't reject, but sanitize
        text = PATTERNS["sql_injection"].sub("", text)
    
    if PATTERNS["script_tag"].search(text):
        logger.warning("Script tags detected in input")
        text = PATTERNS["script_tag"].sub("", text)
    
    # Remove HTML tags
    if PATTERNS["html_tag"].search(text):
        text = PATTERNS["html_tag"].sub("", text)
    
    # Validate encoding
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        raise CustomValidationError("Text contains invalid Unicode characters")
    
    # Check for empty text after cleaning
    cleaned_text = text.strip()
    if not cleaned_text:
        raise CustomValidationError("Text is empty after cleaning")
    
    return cleaned_text

def validate_batch_input(texts: List[str]) -> List[str]:
    """
    Validate batch text input.
    
    Args:
        texts: List of input texts
        
    Returns:
        List of validated texts
        
    Raises:
        ValidationError: If validation fails
    """
    if not texts or not isinstance(texts, list):
        raise CustomValidationError("Batch input must be a non-empty list")
    
    # Batch size validation
    if len(texts) > 1000:
        raise CustomValidationError("Batch size too large. Maximum: 1000 texts")
    
    # Validate each text
    validated_texts = []
    for i, text in enumerate(texts):
        try:
            validated_text = validate_text_input(text)
            validated_texts.append(validated_text)
        except CustomValidationError as e:
            raise CustomValidationError(f"Text at index {i}: {str(e)}")
    
    return validated_texts

def validate_model_name(model_name: str) -> str:
    """
    Validate model name.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Validated model name
        
    Raises:
        ValidationError: If model name is invalid
    """
    if not model_name or not isinstance(model_name, str):
        raise CustomValidationError("Model name must be a non-empty string")
    
    # Normalize model name
    model_name = model_name.lower().strip()
    
    # Check if model is supported
    supported_models = list(MODEL_CONSTRAINTS.keys())
    if model_name not in supported_models:
        raise CustomValidationError(
            f"Unsupported model: {model_name}. "
            f"Supported models: {', '.join(supported_models)}"
        )
    
    return model_name

def validate_prediction_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate prediction parameters.
    
    Args:
        params: Prediction parameters
        
    Returns:
        Validated parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    validated_params = {}
    
    # Temperature validation
    if "temperature" in params:
        temp = params["temperature"]
        if not isinstance(temp, (int, float)) or temp <= 0 or temp > 2:
            raise CustomValidationError(
                "Temperature must be a number between 0 and 2"
            )
        validated_params["temperature"] = float(temp)
    
    # Top-k validation
    if "top_k" in params:
        top_k = params["top_k"]
        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            raise CustomValidationError(
                "Top-k must be an integer between 1 and 100"
            )
        validated_params["top_k"] = top_k
    
    # Top-p validation
    if "top_p" in params:
        top_p = params["top_p"]
        if not isinstance(top_p, (int, float)) or top_p <= 0 or top_p > 1:
            raise CustomValidationError(
                "Top-p must be a number between 0 and 1"
            )
        validated_params["top_p"] = float(top_p)
    
    # Max length validation
    if "max_length" in params:
        max_len = params["max_length"]
        if not isinstance(max_len, int) or max_len < 1 or max_len > MAX_SEQUENCE_LENGTH:
            raise CustomValidationError(
                f"Max length must be between 1 and {MAX_SEQUENCE_LENGTH}"
            )
        validated_params["max_length"] = max_len
    
    return validated_params

def validate_file_upload(
    file_content: bytes,
    file_name: str,
    max_size: int = 10 * 1024 * 1024  # 10MB
) -> Dict[str, Any]:
    """
    Validate file upload.
    
    Implements file validation from:
    - OWASP (2021): "File Upload Cheat Sheet"
    
    Args:
        file_content: File content
        file_name: File name
        max_size: Maximum file size in bytes
        
    Returns:
        Validated file information
        
    Raises:
        ValidationError: If file validation fails
    """
    # Size validation
    if len(file_content) > max_size:
        raise CustomValidationError(
            f"File too large. Maximum size: {max_size / (1024*1024):.1f}MB"
        )
    
    # Extension validation
    allowed_extensions = ['.txt', '.csv', '.json', '.jsonl']
    file_ext = Path(file_name).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise CustomValidationError(
            f"Invalid file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Content validation
    try:
        content_str = file_content.decode('utf-8')
        
        # Check for malicious patterns
        if PATTERNS["script_tag"].search(content_str):
            raise CustomValidationError("File contains potentially malicious content")
        
        # Parse based on file type
        if file_ext == '.json':
            data = json.loads(content_str)
        elif file_ext == '.jsonl':
            lines = content_str.strip().split('\n')
            data = [json.loads(line) for line in lines]
        elif file_ext == '.csv':
            # Basic CSV validation
            lines = content_str.strip().split('\n')
            if len(lines) < 2:
                raise CustomValidationError("CSV file must have header and at least one row")
        else:  # .txt
            lines = content_str.strip().split('\n')
            if not lines:
                raise CustomValidationError("Text file is empty")
        
    except UnicodeDecodeError:
        raise CustomValidationError("File must be UTF-8 encoded")
    except json.JSONDecodeError:
        raise CustomValidationError("Invalid JSON format")
    except Exception as e:
        raise CustomValidationError(f"File parsing error: {str(e)}")
    
    return {
        "file_name": file_name,
        "file_size": len(file_content),
        "file_type": file_ext,
        "content": content_str
    }

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.
    
    Args:
        api_key: API key to validate
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If API key is invalid
    """
    if not api_key or not isinstance(api_key, str):
        raise CustomValidationError("API key must be a non-empty string")
    
    # Length validation (typical for UUID-based keys)
    if len(api_key) < 32 or len(api_key) > 128:
        raise CustomValidationError("Invalid API key length")
    
    # Character validation (alphanumeric and hyphens)
    if not re.match(r"^[a-zA-Z0-9\-_]+$", api_key):
        raise CustomValidationError("API key contains invalid characters")
    
    return True

def validate_pagination_params(
    page: int = 1,
    page_size: int = 10,
    max_page_size: int = 100
) -> Dict[str, int]:
    """
    Validate pagination parameters.
    
    Args:
        page: Page number
        page_size: Items per page
        max_page_size: Maximum allowed page size
        
    Returns:
        Validated pagination parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    if page < 1:
        raise CustomValidationError("Page number must be >= 1")
    
    if page_size < 1:
        raise CustomValidationError("Page size must be >= 1")
    
    if page_size > max_page_size:
        raise CustomValidationError(
            f"Page size too large. Maximum: {max_page_size}"
        )
    
    return {
        "page": page,
        "page_size": page_size,
        "offset": (page - 1) * page_size,
        "limit": page_size
    }

def validate_date_range(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, datetime]:
    """
    Validate date range parameters.
    
    Args:
        start_date: Start date string (ISO format)
        end_date: End date string (ISO format)
        
    Returns:
        Validated date range
        
    Raises:
        ValidationError: If dates are invalid
    """
    dates = {}
    
    # Parse dates
    if start_date:
        try:
            dates["start"] = datetime.fromisoformat(start_date)
        except ValueError:
            raise CustomValidationError(
                "Invalid start date format. Use ISO format (YYYY-MM-DD)"
            )
    
    if end_date:
        try:
            dates["end"] = datetime.fromisoformat(end_date)
        except ValueError:
            raise CustomValidationError(
                "Invalid end date format. Use ISO format (YYYY-MM-DD)"
            )
    
    # Validate range
    if "start" in dates and "end" in dates:
        if dates["start"] > dates["end"]:
            raise CustomValidationError("Start date must be before end date")
        
        # Check for reasonable range (e.g., max 1 year)
        delta = dates["end"] - dates["start"]
        if delta.days > 365:
            raise CustomValidationError("Date range too large. Maximum: 365 days")
    
    return dates

def sanitize_output(data: Any) -> Any:
    """
    Sanitize output data to prevent XSS and data leakage.
    
    Implements output encoding from:
    - OWASP (2021): "XSS Prevention Cheat Sheet"
    
    Args:
        data: Data to sanitize
        
    Returns:
        Sanitized data
    """
    if isinstance(data, str):
        # HTML entity encoding for special characters
        data = (
            data.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#x27;")
        )
    elif isinstance(data, dict):
        return {k: sanitize_output(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_output(item) for item in data]
    
    return data

# Validation decorators for common patterns
def validate_request(validation_func):
    """
    Decorator for request validation.
    
    Args:
        validation_func: Validation function to apply
        
    Returns:
        Decorated function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                # Apply validation
                if "input_data" in kwargs:
                    kwargs["input_data"] = validation_func(kwargs["input_data"])
                
                return await func(*args, **kwargs)
                
            except CustomValidationError as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
        
        return wrapper
    return decorator
