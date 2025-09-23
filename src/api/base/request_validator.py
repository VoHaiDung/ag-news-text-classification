"""
Request Validation Module for API
================================================================================
Implements comprehensive request validation mechanisms using schema definitions,
type checking, and custom validation rules.

This module provides declarative validation capabilities for API requests
following JSON Schema and OpenAPI specifications.

References:
    - JSON Schema Draft 2020-12
    - OpenAPI Specification v3.1.0
    - Martin, R. C. (2008). Clean Code: A Handbook of Agile Software Craftsmanship

Author: Võ Hải Dũng
License: MIT
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from src.api.base.error_handler import ValidationError
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationType(Enum):
    """Types of validation to perform."""
    TYPE = "type"
    RANGE = "range"
    LENGTH = "length"
    PATTERN = "pattern"
    ENUM = "enum"
    CUSTOM = "custom"
    NESTED = "nested"
    CONDITIONAL = "conditional"


@dataclass
class ValidationRule:
    """
    Single validation rule definition.
    
    Attributes:
        type: Type of validation
        params: Parameters for validation
        message: Custom error message
        required: Whether field is required
        nullable: Whether field can be null
    """
    type: ValidationType
    params: Dict[str, Any] = field(default_factory=dict)
    message: Optional[str] = None
    required: bool = False
    nullable: bool = False
    
    def validate(self, value: Any, field_name: str = "field") -> None:
        """
        Execute validation rule.
        
        Args:
            value: Value to validate
            field_name: Name of field being validated
            
        Raises:
            ValidationError: If validation fails
        """
        # Check null values
        if value is None:
            if self.required and not self.nullable:
                raise ValidationError(
                    f"Field '{field_name}' is required",
                    field=field_name
                )
            elif self.nullable:
                return  # Null is allowed
        
        # Perform validation based on type
        validator_method = getattr(self, f"_validate_{self.type.value}", None)
        if validator_method:
            validator_method(value, field_name)
    
    def _validate_type(self, value: Any, field_name: str) -> None:
        """Validate value type."""
        expected_type = self.params.get("type")
        if expected_type and not isinstance(value, expected_type):
            raise ValidationError(
                self.message or f"Field '{field_name}' must be of type {expected_type.__name__}",
                field=field_name,
                value=value,
                constraints={"type": expected_type.__name__}
            )
    
    def _validate_range(self, value: Any, field_name: str) -> None:
        """Validate numeric range."""
        min_val = self.params.get("min")
        max_val = self.params.get("max")
        
        if min_val is not None and value < min_val:
            raise ValidationError(
                self.message or f"Field '{field_name}' must be >= {min_val}",
                field=field_name,
                value=value,
                constraints={"min": min_val}
            )
        
        if max_val is not None and value > max_val:
            raise ValidationError(
                self.message or f"Field '{field_name}' must be <= {max_val}",
                field=field_name,
                value=value,
                constraints={"max": max_val}
            )
    
    def _validate_length(self, value: Any, field_name: str) -> None:
        """Validate string/collection length."""
        if not hasattr(value, "__len__"):
            return
        
        min_len = self.params.get("min_length")
        max_len = self.params.get("max_length")
        
        if min_len is not None and len(value) < min_len:
            raise ValidationError(
                self.message or f"Field '{field_name}' must have minimum length {min_len}",
                field=field_name,
                value=value,
                constraints={"min_length": min_len}
            )
        
        if max_len is not None and len(value) > max_len:
            raise ValidationError(
                self.message or f"Field '{field_name}' must have maximum length {max_len}",
                field=field_name,
                value=value,
                constraints={"max_length": max_len}
            )
    
    def _validate_pattern(self, value: Any, field_name: str) -> None:
        """Validate string pattern."""
        pattern = self.params.get("pattern")
        if pattern and isinstance(value, str):
            if not re.match(pattern, value):
                raise ValidationError(
                    self.message or f"Field '{field_name}' does not match required pattern",
                    field=field_name,
                    value=value,
                    constraints={"pattern": pattern}
                )
    
    def _validate_enum(self, value: Any, field_name: str) -> None:
        """Validate enum values."""
        allowed_values = self.params.get("values", [])
        if value not in allowed_values:
            raise ValidationError(
                self.message or f"Field '{field_name}' must be one of {allowed_values}",
                field=field_name,
                value=value,
                constraints={"allowed_values": allowed_values}
            )
    
    def _validate_custom(self, value: Any, field_name: str) -> None:
        """Execute custom validation function."""
        validator_func = self.params.get("validator")
        if validator_func and callable(validator_func):
            try:
                is_valid = validator_func(value)
                if not is_valid:
                    raise ValidationError(
                        self.message or f"Field '{field_name}' failed custom validation",
                        field=field_name,
                        value=value
                    )
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(
                    f"Custom validation error for field '{field_name}': {str(e)}",
                    field=field_name,
                    value=value
                )


@dataclass
class FieldValidator:
    """
    Field-level validator with multiple rules.
    
    Attributes:
        field_name: Name of field to validate
        rules: List of validation rules
        description: Field description
        example: Example value
    """
    field_name: str
    rules: List[ValidationRule] = field(default_factory=list)
    description: Optional[str] = None
    example: Optional[Any] = None
    
    def validate(self, value: Any) -> None:
        """
        Validate field value against all rules.
        
        Args:
            value: Value to validate
            
        Raises:
            ValidationError: If any validation fails
        """
        for rule in self.rules:
            rule.validate(value, self.field_name)
    
    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule."""
        self.rules.append(rule)


@dataclass
class ValidationSchema:
    """
    Complete validation schema for request/response.
    
    Attributes:
        name: Schema name
        fields: Field validators
        allow_extra_fields: Whether to allow fields not in schema
        strict_mode: Whether to enforce strict validation
    """
    name: str
    fields: Dict[str, FieldValidator] = field(default_factory=dict)
    allow_extra_fields: bool = False
    strict_mode: bool = True
    
    def validate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate data against schema.
        
        Args:
            data: Data dictionary to validate
            
        Returns:
            Validated and cleaned data
            
        Raises:
            ValidationError: If validation fails
        """
        validated_data = {}
        errors = []
        
        # Validate defined fields
        for field_name, validator in self.fields.items():
            try:
                value = data.get(field_name)
                validator.validate(value)
                if value is not None:
                    validated_data[field_name] = value
            except ValidationError as e:
                if self.strict_mode:
                    raise
                errors.append(str(e))
        
        # Check for extra fields
        if not self.allow_extra_fields:
            extra_fields = set(data.keys()) - set(self.fields.keys())
            if extra_fields:
                if self.strict_mode:
                    raise ValidationError(
                        f"Unexpected fields: {extra_fields}",
                        field=",".join(extra_fields)
                    )
                else:
                    logger.warning(f"Ignoring unexpected fields: {extra_fields}")
        else:
            # Include extra fields in validated data
            for key, value in data.items():
                if key not in validated_data:
                    validated_data[key] = value
        
        # Raise accumulated errors in non-strict mode
        if errors and not self.strict_mode:
            logger.warning(f"Validation warnings: {errors}")
        
        return validated_data
    
    def add_field(self, field_validator: FieldValidator) -> None:
        """Add field validator to schema."""
        self.fields[field_validator.field_name] = field_validator
    
    def merge(self, other_schema: 'ValidationSchema') -> None:
        """Merge another schema into this one."""
        self.fields.update(other_schema.fields)


class RequestValidator:
    """
    Main request validator for API endpoints.
    
    Provides comprehensive validation for API requests including
    headers, query parameters, and body content.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize request validator.
        
        Args:
            config: Validator configuration
        """
        self.config = config or {}
        self.schemas: Dict[str, ValidationSchema] = {}
        self._init_common_validators()
    
    def _init_common_validators(self) -> None:
        """Initialize common validation patterns."""
        # Email pattern
        self.email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        # URL pattern
        self.url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        
        # UUID pattern
        self.uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        
        # Common validators
        self.common_validators = {
            "email": lambda v: re.match(self.email_pattern, v) is not None,
            "url": lambda v: re.match(self.url_pattern, v) is not None,
            "uuid": lambda v: re.match(self.uuid_pattern, v) is not None,
            "positive": lambda v: v > 0,
            "non_negative": lambda v: v >= 0,
            "non_empty": lambda v: len(v) > 0 if hasattr(v, "__len__") else v is not None
        }
    
    def register_schema(self, schema: ValidationSchema) -> None:
        """
        Register validation schema.
        
        Args:
            schema: Validation schema to register
        """
        self.schemas[schema.name] = schema
        logger.debug(f"Registered validation schema: {schema.name}")
    
    def validate_request(
        self,
        data: Dict[str, Any],
        schema_name: str = None,
        schema: ValidationSchema = None
    ) -> Dict[str, Any]:
        """
        Validate request data.
        
        Args:
            data: Request data to validate
            schema_name: Name of registered schema
            schema: Direct schema object
            
        Returns:
            Validated data
            
        Raises:
            ValidationError: If validation fails
        """
        # Get schema
        if schema:
            validation_schema = schema
        elif schema_name:
            validation_schema = self.schemas.get(schema_name)
            if not validation_schema:
                raise ValidationError(f"Schema '{schema_name}' not found")
        else:
            raise ValidationError("No validation schema provided")
        
        # Validate
        try:
            validated_data = validation_schema.validate(data)
            logger.debug(f"Request validated successfully with schema: {validation_schema.name}")
            return validated_data
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Validation error: {str(e)}")
            raise ValidationError(f"Validation failed: {str(e)}")
    
    def create_text_validator(
        self,
        min_length: int = 1,
        max_length: int = 10000,
        required: bool = True
    ) -> FieldValidator:
        """
        Create validator for text input.
        
        Args:
            min_length: Minimum text length
            max_length: Maximum text length
            required: Whether field is required
            
        Returns:
            FieldValidator for text
        """
        return FieldValidator(
            field_name="text",
            rules=[
                ValidationRule(
                    type=ValidationType.TYPE,
                    params={"type": str},
                    required=required
                ),
                ValidationRule(
                    type=ValidationType.LENGTH,
                    params={"min_length": min_length, "max_length": max_length}
                ),
                ValidationRule(
                    type=ValidationType.CUSTOM,
                    params={"validator": self.common_validators["non_empty"]},
                    message="Text cannot be empty"
                )
            ],
            description="Text content for classification",
            example="Latest technology news about AI advancements"
        )
    
    def create_model_validator(
        self,
        allowed_models: List[str],
        required: bool = False
    ) -> FieldValidator:
        """
        Create validator for model selection.
        
        Args:
            allowed_models: List of allowed model names
            required: Whether field is required
            
        Returns:
            FieldValidator for model
        """
        return FieldValidator(
            field_name="model",
            rules=[
                ValidationRule(
                    type=ValidationType.TYPE,
                    params={"type": str},
                    required=required,
                    nullable=not required
                ),
                ValidationRule(
                    type=ValidationType.ENUM,
                    params={"values": allowed_models},
                    message=f"Model must be one of: {allowed_models}"
                )
            ],
            description="Model to use for classification",
            example="deberta-v3-large"
        )
    
    def create_batch_validator(
        self,
        max_batch_size: int = 100,
        item_validator: FieldValidator = None
    ) -> FieldValidator:
        """
        Create validator for batch requests.
        
        Args:
            max_batch_size: Maximum batch size
            item_validator: Validator for individual items
            
        Returns:
            FieldValidator for batch
        """
        def validate_batch(items):
            if not isinstance(items, list):
                return False
            if len(items) > max_batch_size:
                return False
            if item_validator:
                for item in items:
                    try:
                        item_validator.validate(item)
                    except ValidationError:
                        return False
            return True
        
        return FieldValidator(
            field_name="items",
            rules=[
                ValidationRule(
                    type=ValidationType.TYPE,
                    params={"type": list},
                    required=True
                ),
                ValidationRule(
                    type=ValidationType.LENGTH,
                    params={"min_length": 1, "max_length": max_batch_size}
                ),
                ValidationRule(
                    type=ValidationType.CUSTOM,
                    params={"validator": validate_batch},
                    message=f"Invalid batch format or size exceeds {max_batch_size}"
                )
            ],
            description="Batch of items to process",
            example=["Text 1", "Text 2", "Text 3"]
        )


def validate_request(schema: ValidationSchema):
    """
    Decorator for request validation.
    
    Args:
        schema: Validation schema to apply
        
    Returns:
        Decorated function
    """
    def decorator(func):
        async def wrapper(request_data: Dict[str, Any], *args, **kwargs):
            # Validate request
            validated_data = schema.validate(request_data)
            
            # Call original function with validated data
            return await func(validated_data, *args, **kwargs)
        
        return wrapper
    return decorator


def validate_field(
    field_name: str,
    field_type: Type = None,
    required: bool = False,
    min_value: Any = None,
    max_value: Any = None,
    pattern: str = None,
    custom_validator: Callable = None
):
    """
    Decorator for field validation.
    
    Args:
        field_name: Field to validate
        field_type: Expected type
        required: Whether field is required
        min_value: Minimum value
        max_value: Maximum value
        pattern: Regex pattern
        custom_validator: Custom validation function
        
    Returns:
        Decorated function
    """
    def decorator(func):
        async def wrapper(data: Dict[str, Any], *args, **kwargs):
            value = data.get(field_name)
            
            # Check required
            if required and value is None:
                raise ValidationError(
                    f"Field '{field_name}' is required",
                    field=field_name
                )
            
            if value is not None:
                # Type check
                if field_type and not isinstance(value, field_type):
                    raise ValidationError(
                        f"Field '{field_name}' must be of type {field_type.__name__}",
                        field=field_name,
                        value=value
                    )
                
                # Range check
                if min_value is not None and value < min_value:
                    raise ValidationError(
                        f"Field '{field_name}' must be >= {min_value}",
                        field=field_name,
                        value=value
                    )
                
                if max_value is not None and value > max_value:
                    raise ValidationError(
                        f"Field '{field_name}' must be <= {max_value}",
                        field=field_name,
                        value=value
                    )
                
                # Pattern check
                if pattern and isinstance(value, str):
                    if not re.match(pattern, value):
                        raise ValidationError(
                            f"Field '{field_name}' does not match required pattern",
                            field=field_name,
                            value=value
                        )
                
                # Custom validation
                if custom_validator and not custom_validator(value):
                    raise ValidationError(
                        f"Field '{field_name}' failed custom validation",
                        field=field_name,
                        value=value
                    )
            
            # Call original function
            return await func(data, *args, **kwargs)
        
        return wrapper
    return decorator
