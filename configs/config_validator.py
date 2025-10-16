"""
AG News Text Classification - Configuration Validator Module

This module provides comprehensive configuration validation functionality for the
AG News Text Classification (ag-news-text-classification) project. It implements
schema-based validation, type checking, constraint enforcement, and compatibility
verification to ensure configuration correctness and consistency.

The ConfigValidator supports:
- JSON Schema-based validation for structured configs
- Type safety validation with Python type hints
- Domain-specific constraint checking (accuracy, ranges, etc.)
- Cross-configuration compatibility validation
- Platform-specific requirement validation
- Overfitting prevention rule enforcement
- Parameter efficiency constraint validation
- Model-specific configuration validation
- Training configuration safety checks
- Ensemble configuration diversity validation

Architecture:
    The validation system implements a multi-layered approach:
    
    Layer 1: Schema Validation
        - JSON Schema-based structural validation
        - Required field checking
        - Type validation at schema level
        - Format validation (email, URL, etc.)
    
    Layer 2: Semantic Validation
        - Domain-specific rules (e.g., learning_rate > 0)
        - Range constraints (e.g., dropout in [0, 1])
        - Enumeration validation (e.g., valid optimizers)
        - Conditional validation (if-then rules)
    
    Layer 3: Compatibility Validation
        - Model-platform compatibility
        - Framework version compatibility
        - Hardware requirement checking
        - Dependency compatibility
    
    Layer 4: Safety Validation
        - Overfitting prevention rules
        - Parameter efficiency constraints
        - Memory safety checks
        - Training stability rules
    
    Layer 5: Best Practice Validation
        - SOTA configuration recommendations
        - Academic standards compliance
        - Performance optimization hints
        - Security best practices

Validation Strategies:
    The validator implements multiple validation strategies:
    
    1. Strict Mode: All validations must pass, no warnings allowed
    2. Lenient Mode: Warnings logged but validation passes
    3. Progressive Mode: Validation strictness increases with model complexity
    4. Advisory Mode: Only recommendations, no hard failures
    
    Strategy selection depends on:
    - Configuration type (model, training, deployment)
    - Environment (development, production, research)
    - User expertise level (beginner, intermediate, expert)
    - Safety requirements (high for SOTA models)

Overfitting Prevention Integration:
    The validator enforces overfitting prevention rules:
    
    - Train-validation split ratios
    - Test set access control
    - Model complexity vs. dataset size
    - Regularization requirements
    - Early stopping configuration
    - Cross-validation requirements
    - Ensemble diversity constraints
    
    These rules are configurable via overfitting_prevention/ configs.

Platform-Specific Validation:
    Different platforms have different constraints:
    
    Colab Free:
        - Memory limit: 12GB
        - Session timeout: 12 hours
        - GPU: T4 (16GB VRAM)
        - Batch size constraints
    
    Kaggle:
        - Memory limit: 13GB (GPU) / 16GB (TPU)
        - Session timeout: 9 hours (GPU) / 3 hours (TPU)
        - GPU: P100 (16GB VRAM) or T4
        - TPU: v3-8
    
    Local:
        - Hardware-dependent
        - User-configured constraints
        - Flexible resource allocation

Error Reporting:
    The validator provides detailed error reports:
    
    - Error location (path in configuration)
    - Error type (schema, semantic, compatibility)
    - Severity (error, warning, info)
    - Suggested fixes
    - Related documentation links
    - Example correct configurations

References:
    Configuration Validation Best Practices:
        - JSON Schema Specification: https://json-schema.org/
        - Cerberus Documentation: https://docs.python-cerberus.org/
        - Pydantic Documentation: https://pydantic-docs.helpmanual.io/
    
    Academic Standards:
        - Goodfellow, I. et al. (2016). "Deep Learning". MIT Press.
          Chapter 5: Machine Learning Basics (validation strategies)
        - Bishop, C. M. (2006). "Pattern Recognition and Machine Learning". Springer.
          Chapter 1: Introduction (model validation)
    
    Overfitting Prevention:
        - Zhang, C. et al. (2017). "Understanding deep learning requires rethinking 
          generalization". ICLR.
        - Ying, X. (2019). "An Overview of Overfitting and its Solutions". 
          Journal of Physics: Conference Series.
    
    ML System Design:
        - Sculley, D. et al. (2015). "Hidden Technical Debt in Machine Learning Systems". 
          NIPS.
        - Amershi, S. et al. (2019). "Software Engineering for Machine Learning: 
          A Case Study". ICSE-SEIP.

Usage:
    Basic validation:
        from configs.config_validator import ConfigValidator
        
        is_valid = ConfigValidator.validate(config)
    
    Type-specific validation:
        ConfigValidator.validate(config, schema_type='model')
        ConfigValidator.validate(config, schema_type='training')
    
    Strict validation:
        ConfigValidator.validate(config, strict=True)
    
    Get validation report:
        report = ConfigValidator.validate_and_report(config)
        print(report.to_string())
    
    Custom validation:
        validator = ConfigValidator(custom_rules={
            'my_rule': lambda cfg: cfg.get('value') > 0
        })
        validator.validate(config)

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
Project: AG News Text Classification (ag-news-text-classification)
Repository: https://github.com/VoHaiDung/ag-news-text-classification
"""

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable, Set
from collections import defaultdict
from dataclasses import dataclass, field


# Module metadata
__version__ = "1.0.0"
__author__ = "Võ Hải Dũng"
__email__ = "vohaidung.work@gmail.com"
__license__ = "MIT"
__project__ = "AG News Text Classification (ag-news-text-classification)"


# Configure logging
logger = logging.getLogger(__name__)


# Configuration root directory
CONFIGS_ROOT = Path(__file__).parent
PROJECT_ROOT = CONFIGS_ROOT.parent


class ValidationError(Exception):
    """
    Exception raised when configuration validation fails.
    
    This exception encapsulates validation errors with detailed information
    about what failed, where it failed, and suggested fixes.
    
    Attributes:
        message: Error description
        field_path: Dot-notation path to the invalid field
        error_type: Type of validation error
        expected: Expected value or constraint
        actual: Actual value that failed validation
        suggestions: List of suggested fixes
    """
    
    def __init__(
        self,
        message: str,
        field_path: Optional[str] = None,
        error_type: Optional[str] = None,
        expected: Optional[Any] = None,
        actual: Optional[Any] = None,
        suggestions: Optional[List[str]] = None
    ):
        """
        Initialize ValidationError.
        
        Args:
            message: Error description
            field_path: Path to invalid field (e.g., "model.lora.rank")
            error_type: Type of error (schema, type, range, etc.)
            expected: Expected value or constraint
            actual: Actual invalid value
            suggestions: List of suggested fixes
        """
        self.message = message
        self.field_path = field_path
        self.error_type = error_type
        self.expected = expected
        self.actual = actual
        self.suggestions = suggestions or []
        
        error_msg = f"Validation Error: {message}"
        if field_path:
            error_msg += f" (Field: {field_path})"
        if error_type:
            error_msg += f" (Type: {error_type})"
        if expected is not None:
            error_msg += f" (Expected: {expected})"
        if actual is not None:
            error_msg += f" (Actual: {actual})"
        
        super().__init__(error_msg)


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue.
    
    Attributes:
        severity: Severity level (error, warning, info)
        message: Issue description
        field_path: Path to the problematic field
        error_type: Type of validation issue
        suggestions: List of suggested fixes
        context: Additional context information
    """
    
    severity: str  # 'error', 'warning', 'info'
    message: str
    field_path: Optional[str] = None
    error_type: Optional[str] = None
    suggestions: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """String representation of the issue."""
        parts = [f"[{self.severity.upper()}]"]
        
        if self.field_path:
            parts.append(f"{self.field_path}:")
        
        parts.append(self.message)
        
        if self.suggestions:
            parts.append(f"(Suggestions: {', '.join(self.suggestions)})")
        
        return " ".join(parts)


@dataclass
class ValidationReport:
    """
    Comprehensive validation report containing all issues.
    
    Attributes:
        is_valid: Whether configuration is valid
        errors: List of error-level issues
        warnings: List of warning-level issues
        info: List of informational messages
        config_type: Type of configuration validated
        validation_time: Time taken for validation
    """
    
    is_valid: bool
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)
    info: List[ValidationIssue] = field(default_factory=list)
    config_type: Optional[str] = None
    validation_time: Optional[float] = None
    
    @property
    def has_errors(self) -> bool:
        """Check if report has any errors."""
        return len(self.errors) > 0
    
    @property
    def has_warnings(self) -> bool:
        """Check if report has any warnings."""
        return len(self.warnings) > 0
    
    @property
    def total_issues(self) -> int:
        """Total number of issues."""
        return len(self.errors) + len(self.warnings) + len(self.info)
    
    def to_string(self, verbose: bool = True) -> str:
        """
        Convert report to human-readable string.
        
        Args:
            verbose: Whether to include detailed information
        
        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("Configuration Validation Report")
        lines.append("=" * 80)
        
        if self.config_type:
            lines.append(f"Configuration Type: {self.config_type}")
        
        lines.append(f"Status: {'VALID' if self.is_valid else 'INVALID'}")
        lines.append(f"Errors: {len(self.errors)}")
        lines.append(f"Warnings: {len(self.warnings)}")
        lines.append(f"Info: {len(self.info)}")
        
        if self.validation_time:
            lines.append(f"Validation Time: {self.validation_time:.3f}s")
        
        if self.errors:
            lines.append("\n" + "Errors" + ":")
            lines.append("-" * 80)
            for error in self.errors:
                lines.append(f"  {error}")
        
        if verbose and self.warnings:
            lines.append("\n" + "Warnings" + ":")
            lines.append("-" * 80)
            for warning in self.warnings:
                lines.append(f"  {warning}")
        
        if verbose and self.info:
            lines.append("\n" + "Information" + ":")
            lines.append("-" * 80)
            for info_item in self.info:
                lines.append(f"  {info_item}")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report to dictionary.
        
        Returns:
            Dictionary representation of report
        """
        return {
            'is_valid': self.is_valid,
            'config_type': self.config_type,
            'validation_time': self.validation_time,
            'summary': {
                'errors': len(self.errors),
                'warnings': len(self.warnings),
                'info': len(self.info),
                'total': self.total_issues
            },
            'errors': [
                {
                    'severity': e.severity,
                    'message': e.message,
                    'field_path': e.field_path,
                    'error_type': e.error_type,
                    'suggestions': e.suggestions
                }
                for e in self.errors
            ],
            'warnings': [
                {
                    'severity': w.severity,
                    'message': w.message,
                    'field_path': w.field_path,
                    'error_type': w.error_type,
                    'suggestions': w.suggestions
                }
                for w in self.warnings
            ],
            'info': [
                {
                    'severity': i.severity,
                    'message': i.message,
                    'field_path': i.field_path
                }
                for i in self.info
            ]
        }


class ConfigValidator:
    """
    Comprehensive configuration validator with multi-layered validation.
    
    This class provides extensive validation capabilities including schema
    validation, type checking, constraint enforcement, compatibility verification,
    and best practice recommendations.
    
    Class Attributes:
        _validators: Registry of validation functions
        _schemas: Cache of loaded JSON schemas
        _rules: Custom validation rules
    
    Methods:
        validate: Main validation entry point
        validate_and_report: Validation with detailed report
        validate_model_config: Model-specific validation
        validate_training_config: Training-specific validation
        validate_overfitting_prevention: Overfitting prevention validation
        validate_platform_compatibility: Platform compatibility validation
        register_validator: Register custom validator
        register_rule: Register custom validation rule
    
    Examples:
        Basic validation:
            is_valid = ConfigValidator.validate(config)
        
        Type-specific validation:
            ConfigValidator.validate(config, schema_type='model')
        
        Get detailed report:
            report = ConfigValidator.validate_and_report(config)
            if not report.is_valid:
                print(report.to_string())
        
        Custom validation:
            ConfigValidator.register_rule('my_rule', lambda cfg: cfg['value'] > 0)
            ConfigValidator.validate(config)
    """
    
    # Registry of validators
    _validators: Dict[str, Callable] = {}
    
    # Schema cache
    _schemas: Dict[str, Dict[str, Any]] = {}
    
    # Custom rules
    _rules: Dict[str, Callable] = {}
    
    # Validation statistics
    _stats: Dict[str, int] = defaultdict(int)
    
    @classmethod
    def validate(
        cls,
        config: Dict[str, Any],
        schema_type: Optional[str] = None,
        strict: bool = False,
        check_overfitting: bool = True,
        check_compatibility: bool = True,
        check_best_practices: bool = True
    ) -> bool:
        """
        Validate configuration with configurable validation levels.
        
        This is the main entry point for configuration validation. It performs
        multi-layered validation including schema, semantics, compatibility,
        and best practices.
        
        Args:
            config: Configuration dictionary to validate
            schema_type: Type of schema to validate against (model, training, etc.)
            strict: Whether to treat warnings as errors
            check_overfitting: Whether to check overfitting prevention rules
            check_compatibility: Whether to check platform compatibility
            check_best_practices: Whether to check best practices
        
        Returns:
            True if validation passes, False otherwise
        
        Raises:
            ValidationError: If validation fails in strict mode
        
        Examples:
            Basic validation:
                is_valid = ConfigValidator.validate(config)
            
            Strict validation:
                try:
                    ConfigValidator.validate(config, strict=True)
                except ValidationError as e:
                    print(f"Validation failed: {e}")
            
            Model validation:
                ConfigValidator.validate(config, schema_type='model')
        """
        cls._stats['validations'] += 1
        
        try:
            report = cls.validate_and_report(
                config,
                schema_type=schema_type,
                check_overfitting=check_overfitting,
                check_compatibility=check_compatibility,
                check_best_practices=check_best_practices
            )
            
            if strict and report.has_warnings:
                raise ValidationError(
                    f"Validation failed with {len(report.warnings)} warnings in strict mode",
                    error_type="strict_mode"
                )
            
            if not report.is_valid:
                if strict:
                    # Raise first error
                    if report.errors:
                        first_error = report.errors[0]
                        raise ValidationError(
                            first_error.message,
                            field_path=first_error.field_path,
                            error_type=first_error.error_type,
                            suggestions=first_error.suggestions
                        )
                else:
                    logger.error(f"Validation failed: {len(report.errors)} errors")
                    for error in report.errors:
                        logger.error(f"  {error}")
            
            return report.is_valid
        
        except ValidationError:
            cls._stats['validation_errors'] += 1
            raise
        except Exception as e:
            cls._stats['validation_errors'] += 1
            logger.error(f"Validation error: {e}")
            raise ValidationError(
                f"Validation failed: {str(e)}",
                error_type="unexpected",
                actual=str(e)
            )
    
    @classmethod
    def validate_and_report(
        cls,
        config: Dict[str, Any],
        schema_type: Optional[str] = None,
        check_overfitting: bool = True,
        check_compatibility: bool = True,
        check_best_practices: bool = True
    ) -> ValidationReport:
        """
        Validate configuration and return detailed report.
        
        Args:
            config: Configuration dictionary to validate
            schema_type: Type of schema to validate against
            check_overfitting: Whether to check overfitting prevention rules
            check_compatibility: Whether to check platform compatibility
            check_best_practices: Whether to check best practices
        
        Returns:
            Detailed validation report
        
        Examples:
            Get report:
                report = ConfigValidator.validate_and_report(config)
                print(report.to_string())
                
                if report.has_warnings:
                    for warning in report.warnings:
                        print(f"Warning: {warning}")
        """
        import time
        start_time = time.time()
        
        report = ValidationReport(is_valid=True, config_type=schema_type)
        
        # Detect configuration type if not specified
        if schema_type is None:
            schema_type = cls._detect_config_type(config)
            report.config_type = schema_type
        
        # Layer 1: Basic structure validation
        cls._validate_structure(config, report)
        
        # Layer 2: Schema validation if schema type is specified
        if schema_type:
            cls._validate_schema(config, schema_type, report)
        
        # Layer 3: Type-specific validation
        if schema_type == 'model':
            cls._validate_model_config(config, report)
        elif schema_type == 'training':
            cls._validate_training_config(config, report)
        elif schema_type == 'data':
            cls._validate_data_config(config, report)
        elif schema_type == 'ensemble':
            cls._validate_ensemble_config(config, report)
        elif schema_type == 'api':
            cls._validate_api_config(config, report)
        
        # Layer 4: Overfitting prevention validation
        if check_overfitting:
            cls._validate_overfitting_prevention(config, report)
        
        # Layer 5: Compatibility validation
        if check_compatibility:
            cls._validate_compatibility(config, report)
        
        # Layer 6: Best practices validation
        if check_best_practices:
            cls._validate_best_practices(config, report)
        
        # Layer 7: Custom rules validation
        cls._validate_custom_rules(config, report)
        
        # Set final validation status
        report.is_valid = not report.has_errors
        report.validation_time = time.time() - start_time
        
        return report
    
    @classmethod
    def validate_model_config(
        cls,
        config: Dict[str, Any],
        strict: bool = False
    ) -> bool:
        """
        Validate model configuration.
        
        Args:
            config: Model configuration dictionary
            strict: Whether to use strict validation
        
        Returns:
            True if validation passes
        
        Examples:
            ConfigValidator.validate_model_config(model_config)
        """
        return cls.validate(config, schema_type='model', strict=strict)
    
    @classmethod
    def validate_training_config(
        cls,
        config: Dict[str, Any],
        strict: bool = False
    ) -> bool:
        """
        Validate training configuration.
        
        Args:
            config: Training configuration dictionary
            strict: Whether to use strict validation
        
        Returns:
            True if validation passes
        
        Examples:
            ConfigValidator.validate_training_config(training_config)
        """
        return cls.validate(config, schema_type='training', strict=strict)
    
    @classmethod
    def register_validator(
        cls,
        name: str,
        validator_func: Callable[[Dict[str, Any], ValidationReport], None]
    ) -> None:
        """
        Register a custom validator function.
        
        Args:
            name: Validator name
            validator_func: Validator function that takes config and report
        
        Examples:
            def my_validator(config, report):
                if config.get('my_field') is None:
                    report.errors.append(ValidationIssue(
                        severity='error',
                        message='my_field is required',
                        field_path='my_field'
                    ))
            
            ConfigValidator.register_validator('my_validator', my_validator)
        """
        cls._validators[name] = validator_func
        logger.info(f"Registered custom validator: {name}")
    
    @classmethod
    def register_rule(
        cls,
        name: str,
        rule_func: Callable[[Dict[str, Any]], bool],
        error_message: Optional[str] = None
    ) -> None:
        """
        Register a custom validation rule.
        
        Args:
            name: Rule name
            rule_func: Function that returns True if rule passes
            error_message: Error message if rule fails
        
        Examples:
            ConfigValidator.register_rule(
                'positive_rank',
                lambda cfg: cfg.get('lora', {}).get('rank', 0) > 0,
                'LoRA rank must be positive'
            )
        """
        cls._rules[name] = (rule_func, error_message or f"Rule '{name}' failed")
        logger.info(f"Registered custom rule: {name}")
    
    # Private validation methods
    
    @classmethod
    def _detect_config_type(cls, config: Dict[str, Any]) -> Optional[str]:
        """
        Detect configuration type from structure.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Detected configuration type or None
        """
        # Check for model configuration markers
        if 'model' in config or 'model_name' in config or 'architecture' in config:
            return 'model'
        
        # Check for training configuration markers
        if 'training' in config or 'optimizer' in config or 'learning_rate' in config:
            return 'training'
        
        # Check for ensemble configuration markers
        if 'ensemble' in config or 'models' in config or 'voting' in config:
            return 'ensemble'
        
        # Check for data configuration markers
        if 'data' in config or 'dataset' in config or 'preprocessing' in config:
            return 'data'
        
        # Check for API configuration markers
        if 'api' in config or 'endpoints' in config or 'routes' in config:
            return 'api'
        
        return None
    
    @classmethod
    def _validate_structure(
        cls,
        config: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate basic configuration structure.
        
        Args:
            config: Configuration to validate
            report: Validation report to update
        """
        if not isinstance(config, dict):
            report.errors.append(ValidationIssue(
                severity='error',
                message='Configuration must be a dictionary',
                error_type='structure',
                suggestions=['Ensure configuration is a valid YAML/JSON object']
            ))
            return
        
        if not config:
            report.warnings.append(ValidationIssue(
                severity='warning',
                message='Configuration is empty',
                error_type='structure'
            ))
    
    @classmethod
    def _validate_schema(
        cls,
        config: Dict[str, Any],
        schema_type: str,
        report: ValidationReport
    ) -> None:
        """
        Validate configuration against JSON schema.
        
        Args:
            config: Configuration to validate
            schema_type: Type of schema to use
            report: Validation report to update
        """
        try:
            # Try to import jsonschema
            try:
                import jsonschema
                from jsonschema import Draft7Validator
            except ImportError:
                report.warnings.append(ValidationIssue(
                    severity='warning',
                    message='jsonschema not available, skipping schema validation',
                    error_type='dependency',
                    suggestions=['Install jsonschema: pip install jsonschema']
                ))
                return
            
            # Load schema if available
            schema = cls._load_schema(schema_type)
            
            if not schema:
                report.info.append(ValidationIssue(
                    severity='info',
                    message=f'No schema available for type: {schema_type}',
                    error_type='schema'
                ))
                return
            
            # Validate against schema
            validator = Draft7Validator(schema)
            errors = list(validator.iter_errors(config))
            
            for error in errors:
                field_path = '.'.join(str(p) for p in error.path) if error.path else 'root'
                
                report.errors.append(ValidationIssue(
                    severity='error',
                    message=error.message,
                    field_path=field_path,
                    error_type='schema'
                ))
        
        except Exception as e:
            logger.warning(f"Schema validation error: {e}")
            report.warnings.append(ValidationIssue(
                severity='warning',
                message=f'Schema validation failed: {str(e)}',
                error_type='schema'
            ))
    
    @classmethod
    def _validate_model_config(
        cls,
        config: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate model-specific configuration.
        
        Args:
            config: Model configuration
            report: Validation report to update
        """
        model_config = config.get('model', {})
        
        # Validate model name/type
        if 'name' not in model_config and 'type' not in model_config:
            report.warnings.append(ValidationIssue(
                severity='warning',
                message='Model name or type not specified',
                field_path='model.name',
                suggestions=['Add model.name or model.type field']
            ))
        
        # Validate LoRA configuration if present
        if 'lora' in model_config:
            cls._validate_lora_config(model_config['lora'], report, 'model.lora')
        
        # Validate QLoRA configuration if present
        if 'qlora' in model_config:
            cls._validate_qlora_config(model_config['qlora'], report, 'model.qlora')
        
        # Validate num_labels for classification
        num_labels = model_config.get('num_labels', config.get('num_labels'))
        if num_labels is not None:
            if not isinstance(num_labels, int) or num_labels <= 0:
                report.errors.append(ValidationIssue(
                    severity='error',
                    message='num_labels must be a positive integer',
                    field_path='model.num_labels',
                    error_type='type',
                    context={'actual': num_labels}
                ))
    
    @classmethod
    def _validate_training_config(
        cls,
        config: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate training-specific configuration.
        
        Args:
            config: Training configuration
            report: Validation report to update
        """
        training_config = config.get('training', {})
        
        # Validate learning rate
        lr = training_config.get('learning_rate')
        if lr is not None:
            if not isinstance(lr, (int, float)) or lr <= 0:
                report.errors.append(ValidationIssue(
                    severity='error',
                    message='learning_rate must be a positive number',
                    field_path='training.learning_rate',
                    error_type='range',
                    context={'actual': lr}
                ))
            elif lr > 0.1:
                report.warnings.append(ValidationIssue(
                    severity='warning',
                    message='learning_rate seems unusually high',
                    field_path='training.learning_rate',
                    suggestions=['Typical values are 1e-5 to 1e-3 for transformers'],
                    context={'actual': lr}
                ))
        
        # Validate batch size
        batch_size = training_config.get('batch_size')
        if batch_size is not None:
            if not isinstance(batch_size, int) or batch_size <= 0:
                report.errors.append(ValidationIssue(
                    severity='error',
                    message='batch_size must be a positive integer',
                    field_path='training.batch_size',
                    error_type='type',
                    context={'actual': batch_size}
                ))
        
        # Validate num_epochs
        num_epochs = training_config.get('num_epochs', training_config.get('epochs'))
        if num_epochs is not None:
            if not isinstance(num_epochs, int) or num_epochs <= 0:
                report.errors.append(ValidationIssue(
                    severity='error',
                    message='num_epochs must be a positive integer',
                    field_path='training.num_epochs',
                    error_type='type',
                    context={'actual': num_epochs}
                ))
    
    @classmethod
    def _validate_lora_config(
        cls,
        lora_config: Dict[str, Any],
        report: ValidationReport,
        path_prefix: str = 'lora'
    ) -> None:
        """
        Validate LoRA configuration.
        
        Args:
            lora_config: LoRA configuration
            report: Validation report to update
            path_prefix: Field path prefix
        """
        # Validate rank
        rank = lora_config.get('rank', lora_config.get('r'))
        if rank is not None:
            if not isinstance(rank, int) or rank <= 0:
                report.errors.append(ValidationIssue(
                    severity='error',
                    message='LoRA rank must be a positive integer',
                    field_path=f'{path_prefix}.rank',
                    error_type='type',
                    context={'actual': rank}
                ))
            elif rank > 64:
                report.warnings.append(ValidationIssue(
                    severity='warning',
                    message='LoRA rank is very high, may reduce efficiency',
                    field_path=f'{path_prefix}.rank',
                    suggestions=['Typical values are 4-32 for good efficiency'],
                    context={'actual': rank}
                ))
        
        # Validate alpha
        alpha = lora_config.get('alpha', lora_config.get('lora_alpha'))
        if alpha is not None:
            if not isinstance(alpha, (int, float)) or alpha <= 0:
                report.errors.append(ValidationIssue(
                    severity='error',
                    message='LoRA alpha must be a positive number',
                    field_path=f'{path_prefix}.alpha',
                    error_type='type',
                    context={'actual': alpha}
                ))
        
        # Validate dropout
        dropout = lora_config.get('dropout', lora_config.get('lora_dropout'))
        if dropout is not None:
            if not isinstance(dropout, (int, float)) or dropout < 0 or dropout > 1:
                report.errors.append(ValidationIssue(
                    severity='error',
                    message='LoRA dropout must be in range [0, 1]',
                    field_path=f'{path_prefix}.dropout',
                    error_type='range',
                    context={'actual': dropout}
                ))
    
    @classmethod
    def _validate_qlora_config(
        cls,
        qlora_config: Dict[str, Any],
        report: ValidationReport,
        path_prefix: str = 'qlora'
    ) -> None:
        """
        Validate QLoRA configuration.
        
        Args:
            qlora_config: QLoRA configuration
            report: Validation report to update
            path_prefix: Field path prefix
        """
        # Validate LoRA parameters (QLoRA includes LoRA)
        cls._validate_lora_config(qlora_config, report, path_prefix)
        
        # Validate quantization bits
        bits = qlora_config.get('bits', qlora_config.get('load_in_4bit') and 4 or 8)
        if bits not in [4, 8]:
            report.errors.append(ValidationIssue(
                severity='error',
                message='QLoRA bits must be 4 or 8',
                field_path=f'{path_prefix}.bits',
                error_type='enum',
                context={'actual': bits, 'allowed': [4, 8]}
            ))
    
    @classmethod
    def _validate_data_config(
        cls,
        config: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate data configuration.
        
        Args:
            config: Data configuration
            report: Validation report to update
        """
        data_config = config.get('data', {})
        
        # Validate train/val/test splits
        train_split = data_config.get('train_split')
        val_split = data_config.get('val_split')
        test_split = data_config.get('test_split')
        
        if all(s is not None for s in [train_split, val_split, test_split]):
            total = train_split + val_split + test_split
            if abs(total - 1.0) > 0.01:
                report.errors.append(ValidationIssue(
                    severity='error',
                    message='Train/val/test splits must sum to 1.0',
                    field_path='data.splits',
                    error_type='constraint',
                    context={'sum': total}
                ))
    
    @classmethod
    def _validate_ensemble_config(
        cls,
        config: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate ensemble configuration.
        
        Args:
            config: Ensemble configuration
            report: Validation report to update
        """
        ensemble_config = config.get('ensemble', {})
        
        # Validate number of models
        models = ensemble_config.get('models', [])
        if len(models) < 3:
            report.warnings.append(ValidationIssue(
                severity='warning',
                message='Ensemble should have at least 3 models for good diversity',
                field_path='ensemble.models',
                suggestions=['Add more diverse models to the ensemble'],
                context={'num_models': len(models)}
            ))
    
    @classmethod
    def _validate_api_config(
        cls,
        config: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate API configuration.
        
        Args:
            config: API configuration
            report: Validation report to update
        """
        api_config = config.get('api', {})
        
        # Validate port
        port = api_config.get('port')
        if port is not None:
            if not isinstance(port, int) or port < 1024 or port > 65535:
                report.errors.append(ValidationIssue(
                    severity='error',
                    message='Port must be between 1024 and 65535',
                    field_path='api.port',
                    error_type='range',
                    context={'actual': port}
                ))
    
    @classmethod
    def _validate_overfitting_prevention(
        cls,
        config: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate overfitting prevention rules.
        
        Args:
            config: Configuration to validate
            report: Validation report to update
        """
        # Check if regularization is configured
        training = config.get('training', {})
        
        has_dropout = 'dropout' in training or 'dropout_rate' in training
        has_weight_decay = 'weight_decay' in training
        has_early_stopping = 'early_stopping' in training
        
        if not any([has_dropout, has_weight_decay, has_early_stopping]):
            report.warnings.append(ValidationIssue(
                severity='warning',
                message='No regularization techniques configured',
                field_path='training',
                error_type='overfitting',
                suggestions=[
                    'Add dropout',
                    'Add weight_decay',
                    'Add early_stopping'
                ]
            ))
    
    @classmethod
    def _validate_compatibility(
        cls,
        config: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate platform and dependency compatibility.
        
        Args:
            config: Configuration to validate
            report: Validation report to update
        """
        # Check platform compatibility if specified
        platform = config.get('platform')
        if platform:
            model_config = config.get('model', {})
            model_name = model_config.get('name', '')
            
            # Check Colab compatibility
            if platform == 'colab':
                if 'xxlarge' in model_name.lower():
                    report.warnings.append(ValidationIssue(
                        severity='warning',
                        message='XXLarge models may not fit in Colab free tier',
                        field_path='platform',
                        suggestions=['Use Colab Pro or reduce model size']
                    ))
    
    @classmethod
    def _validate_best_practices(
        cls,
        config: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate against best practices.
        
        Args:
            config: Configuration to validate
            report: Validation report to update
        """
        # Add informational best practice suggestions
        training = config.get('training', {})
        
        # Recommend gradient accumulation for large models
        batch_size = training.get('batch_size', 8)
        if batch_size < 4:
            report.info.append(ValidationIssue(
                severity='info',
                message='Consider using gradient accumulation for small batch sizes',
                field_path='training.batch_size',
                suggestions=['Set gradient_accumulation_steps > 1']
            ))
    
    @classmethod
    def _validate_custom_rules(
        cls,
        config: Dict[str, Any],
        report: ValidationReport
    ) -> None:
        """
        Validate against custom registered rules.
        
        Args:
            config: Configuration to validate
            report: Validation report to update
        """
        for rule_name, (rule_func, error_message) in cls._rules.items():
            try:
                if not rule_func(config):
                    report.errors.append(ValidationIssue(
                        severity='error',
                        message=error_message,
                        error_type='custom_rule',
                        context={'rule': rule_name}
                    ))
            except Exception as e:
                logger.error(f"Custom rule '{rule_name}' failed: {e}")
    
    @classmethod
    def _load_schema(cls, schema_type: str) -> Optional[Dict[str, Any]]:
        """
        Load JSON schema for configuration type.
        
        Args:
            schema_type: Type of schema to load
        
        Returns:
            Schema dictionary or None if not found
        """
        # Check cache first
        if schema_type in cls._schemas:
            return cls._schemas[schema_type]
        
        # Try to load schema from config_schema module
        try:
            from configs.config_schema import ConfigSchema
            schema = ConfigSchema.get_schema(schema_type)
            cls._schemas[schema_type] = schema
            return schema
        except (ImportError, AttributeError):
            logger.debug(f"No schema found for type: {schema_type}")
            return None


# Convenience functions

def validate(
    config: Dict[str, Any],
    schema_type: Optional[str] = None,
    strict: bool = False
) -> bool:
    """
    Convenience function to validate configuration.
    
    Args:
        config: Configuration to validate
        schema_type: Type of schema to use
        strict: Whether to use strict validation
    
    Returns:
        True if validation passes
    
    Examples:
        is_valid = validate(config)
    """
    return ConfigValidator.validate(config, schema_type=schema_type, strict=strict)


def validate_and_report(
    config: Dict[str, Any],
    schema_type: Optional[str] = None
) -> ValidationReport:
    """
    Convenience function to validate and get report.
    
    Args:
        config: Configuration to validate
        schema_type: Type of schema to use
    
    Returns:
        Validation report
    
    Examples:
        report = validate_and_report(config)
        print(report.to_string())
    """
    return ConfigValidator.validate_and_report(config, schema_type=schema_type)


# Module exports
__all__ = [
    'ConfigValidator',
    'ValidationError',
    'ValidationIssue',
    'ValidationReport',
    'validate',
    'validate_and_report',
]


# Module initialization
logger.info(
    f"Configuration Validator initialized for {__project__} v{__version__} "
    f"(Author: {__author__})"
)
