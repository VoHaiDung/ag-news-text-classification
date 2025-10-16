"""
AG News Text Classification - Configuration Schema Module

This module provides comprehensive JSON Schema definitions for all configuration
types in the AG News Text Classification (ag-news-text-classification) project.
It implements a hierarchical schema system with reusable components, validation
rules, and automatic schema generation capabilities.

The ConfigSchema supports:
- JSON Schema Draft 7 compliant schema definitions
- Hierarchical schema composition and inheritance
- Reusable schema components (definitions)
- Type-specific schemas (model, training, data, ensemble, etc.)
- Domain-specific validation constraints
- Custom format validators
- Schema versioning and migration
- Automatic schema generation from examples
- Schema documentation generation
- Cross-schema reference resolution

Architecture:
    The schema system is organized hierarchically:
    
    Base Schemas (Primitives):
        - Numeric types with constraints (positive, unit_interval, etc.)
        - String types with patterns (email, url, path, etc.)
        - Enum types for categorical values
        - Array types with item constraints
        - Object types with property schemas
    
    Component Schemas (Reusable):
        - LoRA configuration schema
        - QLoRA configuration schema
        - Optimizer configuration schema
        - Scheduler configuration schema
        - Regularization configuration schema
    
    Domain Schemas (Type-specific):
        - Model configuration schema
        - Training configuration schema
        - Data configuration schema
        - Ensemble configuration schema
        - API configuration schema
        - Deployment configuration schema
        - Overfitting prevention schema
    
    Composite Schemas (Combined):
        - Full training pipeline schema
        - SOTA model configuration schema
        - Platform-optimized configuration schema

Schema Composition:
    Schemas can be composed using JSON Schema features:
    
    1. $ref: Reference to another schema or definition
    2. allOf: All schemas must validate (intersection)
    3. anyOf: At least one schema must validate (union)
    4. oneOf: Exactly one schema must validate (exclusive union)
    5. not: Schema must not validate (negation)
    
    Example:
        {
            "allOf": [
                {"$ref": "#/definitions/base_model"},
                {"$ref": "#/definitions/lora_config"}
            ]
        }

Validation Constraints:
    The schemas enforce multiple types of constraints:
    
    Type Constraints:
        - type: Primitive type (string, number, integer, boolean, etc.)
        - enum: Allowed values from enumeration
        - const: Exact constant value
    
    Numeric Constraints:
        - minimum, maximum: Range boundaries
        - exclusiveMinimum, exclusiveMaximum: Exclusive boundaries
        - multipleOf: Value must be multiple of this number
    
    String Constraints:
        - minLength, maxLength: String length bounds
        - pattern: Regex pattern matching
        - format: Predefined formats (email, uri, etc.)
    
    Array Constraints:
        - minItems, maxItems: Array size bounds
        - uniqueItems: No duplicate items
        - items: Schema for array items
        - contains: At least one item matching schema
    
    Object Constraints:
        - required: Required property names
        - properties: Schema for each property
        - additionalProperties: Allow/disallow extra properties
        - minProperties, maxProperties: Object size bounds
        - dependencies: Property dependencies

Domain-Specific Constraints:
    The module defines domain-specific validation rules:
    
    Machine Learning Constraints:
        - Learning rate: (0, 1], typically [1e-6, 1e-1]
        - Dropout rate: [0, 1]
        - Batch size: Positive integer, power of 2 recommended
        - Epochs: Positive integer
        - LoRA rank: Positive integer, typically [4, 64]
        - Model size limits based on platform
    
    Overfitting Prevention Constraints:
        - Train/val/test split ratios must sum to 1.0
        - Minimum validation set size
        - Maximum model complexity for dataset size
        - Required regularization for large models
    
    Platform-Specific Constraints:
        - Memory limits (Colab: 12GB, Kaggle: 13GB, etc.)
        - GPU VRAM limits
        - Session timeout constraints
        - API quota limits

Schema Versioning:
    Schemas are versioned to support migration:
    
    Version Format: MAJOR.MINOR.PATCH
    - MAJOR: Breaking changes
    - MINOR: Backward-compatible additions
    - PATCH: Bug fixes and clarifications
    
    Migration Support:
        - Automatic schema migration between versions
        - Deprecation warnings for old fields
        - Default value insertion for new required fields

Custom Validators:
    Beyond JSON Schema standard validation, custom validators are provided:
    
    - validate_model_platform_compatibility()
    - validate_lora_efficiency()
    - validate_ensemble_diversity()
    - validate_overfitting_risk()
    - validate_resource_constraints()
    
    These validators implement complex business logic that cannot be
    expressed in pure JSON Schema.

References:
    JSON Schema Specification:
        - Wright, A. et al. "JSON Schema: A Media Type for Describing JSON Documents". 
          Internet-Draft draft-handrews-json-schema-02, 2019.
        - JSON Schema Specification: https://json-schema.org/specification.html
        - Understanding JSON Schema: https://json-schema.org/understanding-json-schema/
    
    Schema Design Patterns:
        - Pezoa, F. et al. (2016). "Foundations of JSON Schema". 
          Proceedings of the 25th International Conference on World Wide Web.
        - Bourhis, P. et al. (2017). "JSON: Data model, Query languages and Schema 
          specification". ACM SIGMOD/PODS.
    
    Configuration Validation Best Practices:
        - Humble, J. & Farley, D. (2010). "Continuous Delivery". Addison-Wesley.
          Chapter 2: Configuration Management.
        - Newman, S. (2015). "Building Microservices". O'Reilly.
          Chapter 11: Microservices at Scale (configuration management).
    
    Machine Learning Configuration:
        - Baylor, D. et al. (2017). "TFX: A TensorFlow-Based Production-Scale Machine 
          Learning Platform". KDD.
        - Polyzotis, N. et al. (2018). "Data Lifecycle Challenges in Production Machine 
          Learning: A Survey". SIGMOD Record.

Usage:
    Get schema for configuration type:
        from configs.config_schema import ConfigSchema
        
        schema = ConfigSchema.get_schema('model')
    
    Validate configuration against schema:
        is_valid = ConfigSchema.validate(config, 'model')
    
    Get all available schemas:
        schemas = ConfigSchema.list_schemas()
    
    Register custom schema:
        ConfigSchema.register_schema('custom', my_schema)
    
    Generate schema from example:
        schema = ConfigSchema.generate_from_example(example_config)

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
Project: AG News Text Classification (ag-news-text-classification)
Repository: https://github.com/VoHaiDung/ag-news-text-classification
"""

import logging
from typing import Any, Dict, List, Optional, Union, Set, Callable
from pathlib import Path
from copy import deepcopy


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


class SchemaError(Exception):
    """
    Exception raised for schema-related errors.
    
    This exception is raised when schema operations fail, such as
    schema not found, invalid schema definition, or schema composition errors.
    
    Attributes:
        message: Error description
        schema_type: Type of schema that caused the error
        details: Additional error details
    """
    
    def __init__(
        self,
        message: str,
        schema_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize SchemaError.
        
        Args:
            message: Error description
            schema_type: Type of schema
            details: Additional error details
        """
        self.message = message
        self.schema_type = schema_type
        self.details = details or {}
        
        error_msg = f"Schema Error: {message}"
        if schema_type:
            error_msg += f" (Schema Type: {schema_type})"
        if details:
            error_msg += f" (Details: {details})"
        
        super().__init__(error_msg)


class ConfigSchema:
    """
    Comprehensive configuration schema registry and validator.
    
    This class provides centralized access to all configuration schemas
    used in the AG News Text Classification project. It supports schema
    registration, retrieval, validation, composition, and generation.
    
    The schema system is based on JSON Schema Draft 7 specification with
    custom extensions for domain-specific validation rules.
    
    Class Attributes:
        _schemas: Registry of all schemas
        _definitions: Reusable schema components
        _validators: Custom validation functions
        _schema_version: Current schema version
    
    Methods:
        get_schema: Get schema by type
        validate: Validate configuration against schema
        register_schema: Register custom schema
        list_schemas: List all available schemas
        get_definitions: Get reusable schema definitions
        compose_schema: Compose schema from multiple parts
        generate_from_example: Generate schema from example config
    
    Examples:
        Get model schema:
            schema = ConfigSchema.get_schema('model')
        
        Validate configuration:
            is_valid = ConfigSchema.validate(config, 'training')
        
        Register custom schema:
            ConfigSchema.register_schema('my_type', my_schema)
        
        List available schemas:
            types = ConfigSchema.list_schemas()
    """
    
    # Schema registry
    _schemas: Dict[str, Dict[str, Any]] = {}
    
    # Reusable definitions
    _definitions: Dict[str, Dict[str, Any]] = {}
    
    # Custom validators
    _validators: Dict[str, Callable] = {}
    
    # Schema version
    _schema_version = "1.0.0"
    
    # Initialization flag
    _initialized = False
    
    @classmethod
    def initialize(cls) -> None:
        """
        Initialize schema registry with all predefined schemas.
        
        This method is called automatically on first use. It loads all
        schema definitions and registers them in the schema registry.
        """
        if cls._initialized:
            return
        
        # Register reusable definitions
        cls._register_definitions()
        
        # Register type-specific schemas
        cls._register_model_schemas()
        cls._register_training_schemas()
        cls._register_data_schemas()
        cls._register_ensemble_schemas()
        cls._register_api_schemas()
        cls._register_deployment_schemas()
        cls._register_overfitting_prevention_schemas()
        cls._register_platform_schemas()
        cls._register_service_schemas()
        
        cls._initialized = True
        logger.info(f"Schema registry initialized with {len(cls._schemas)} schemas")
    
    @classmethod
    def get_schema(
        cls,
        schema_type: str,
        include_definitions: bool = True
    ) -> Dict[str, Any]:
        """
        Get schema by type.
        
        Args:
            schema_type: Type of schema to retrieve
            include_definitions: Whether to include definitions section
        
        Returns:
            Schema dictionary
        
        Raises:
            SchemaError: If schema type not found
        
        Examples:
            Get model schema:
                schema = ConfigSchema.get_schema('model')
            
            Get schema without definitions:
                schema = ConfigSchema.get_schema('training', include_definitions=False)
        """
        cls.initialize()
        
        if schema_type not in cls._schemas:
            available = ', '.join(cls._schemas.keys())
            raise SchemaError(
                f"Schema type '{schema_type}' not found",
                schema_type=schema_type,
                details={'available_types': available}
            )
        
        schema = deepcopy(cls._schemas[schema_type])
        
        if include_definitions and cls._definitions:
            if 'definitions' not in schema:
                schema['definitions'] = {}
            schema['definitions'].update(cls._definitions)
        
        return schema
    
    @classmethod
    def validate(
        cls,
        config: Dict[str, Any],
        schema_type: str,
        strict: bool = False
    ) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema_type: Type of schema to validate against
            strict: Whether to raise exception on validation failure
        
        Returns:
            True if valid, False otherwise
        
        Raises:
            SchemaError: If strict=True and validation fails
        
        Examples:
            Validate model config:
                is_valid = ConfigSchema.validate(config, 'model')
            
            Strict validation:
                try:
                    ConfigSchema.validate(config, 'training', strict=True)
                except SchemaError as e:
                    print(f"Validation failed: {e}")
        """
        try:
            import jsonschema
            from jsonschema import Draft7Validator, ValidationError
        except ImportError:
            logger.warning("jsonschema not available, skipping validation")
            return True
        
        try:
            schema = cls.get_schema(schema_type)
            validator = Draft7Validator(schema)
            validator.validate(config)
            return True
        
        except ValidationError as e:
            if strict:
                raise SchemaError(
                    f"Schema validation failed: {e.message}",
                    schema_type=schema_type,
                    details={'path': list(e.path), 'validator': e.validator}
                )
            else:
                logger.error(f"Schema validation failed for {schema_type}: {e.message}")
                return False
        except Exception as e:
            logger.error(f"Schema validation error: {e}")
            if strict:
                raise
            return False
    
    @classmethod
    def register_schema(
        cls,
        schema_type: str,
        schema: Dict[str, Any],
        overwrite: bool = False
    ) -> None:
        """
        Register a custom schema.
        
        Args:
            schema_type: Type identifier for the schema
            schema: Schema dictionary
            overwrite: Whether to overwrite existing schema
        
        Raises:
            SchemaError: If schema type already exists and overwrite=False
        
        Examples:
            Register custom schema:
                ConfigSchema.register_schema('my_type', my_schema)
            
            Overwrite existing:
                ConfigSchema.register_schema('model', new_schema, overwrite=True)
        """
        if schema_type in cls._schemas and not overwrite:
            raise SchemaError(
                f"Schema type '{schema_type}' already registered",
                schema_type=schema_type,
                details={'hint': 'Use overwrite=True to replace'}
            )
        
        cls._schemas[schema_type] = schema
        logger.info(f"Registered schema: {schema_type}")
    
    @classmethod
    def list_schemas(cls) -> List[str]:
        """
        List all available schema types.
        
        Returns:
            List of schema type identifiers
        
        Examples:
            Get all schema types:
                types = ConfigSchema.list_schemas()
                print(f"Available schemas: {types}")
        """
        cls.initialize()
        return list(cls._schemas.keys())
    
    @classmethod
    def get_definitions(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all reusable schema definitions.
        
        Returns:
            Dictionary of schema definitions
        
        Examples:
            Get definitions:
                defs = ConfigSchema.get_definitions()
                lora_def = defs['lora_config']
        """
        cls.initialize()
        return deepcopy(cls._definitions)
    
    # Private methods for schema registration
    
    @classmethod
    def _register_definitions(cls) -> None:
        """Register reusable schema definitions."""
        
        # Positive number
        cls._definitions['positive_number'] = {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "A positive number greater than zero"
        }
        
        # Non-negative number
        cls._definitions['non_negative_number'] = {
            "type": "number",
            "minimum": 0,
            "description": "A non-negative number (zero or positive)"
        }
        
        # Positive integer
        cls._definitions['positive_integer'] = {
            "type": "integer",
            "minimum": 1,
            "description": "A positive integer"
        }
        
        # Unit interval [0, 1]
        cls._definitions['unit_interval'] = {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "A number in the range [0, 1]"
        }
        
        # Learning rate
        cls._definitions['learning_rate'] = {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 1,
            "description": "Learning rate, typically in range (0, 1e-1]"
        }
        
        # Dropout rate
        cls._definitions['dropout_rate'] = {
            "$ref": "#/definitions/unit_interval",
            "description": "Dropout probability in range [0, 1]"
        }
        
        # Batch size
        cls._definitions['batch_size'] = {
            "type": "integer",
            "minimum": 1,
            "description": "Batch size for training/inference"
        }
        
        # Number of epochs
        cls._definitions['num_epochs'] = {
            "type": "integer",
            "minimum": 1,
            "description": "Number of training epochs"
        }
        
        # Model name
        cls._definitions['model_name'] = {
            "type": "string",
            "minLength": 1,
            "description": "Model name or identifier"
        }
        
        # File path
        cls._definitions['file_path'] = {
            "type": "string",
            "minLength": 1,
            "description": "File system path"
        }
        
        # LoRA configuration
        cls._definitions['lora_config'] = {
            "type": "object",
            "properties": {
                "rank": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 256,
                    "description": "LoRA rank (r), typical values: 4-64"
                },
                "alpha": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "LoRA alpha scaling parameter"
                },
                "dropout": {
                    "$ref": "#/definitions/dropout_rate",
                    "description": "LoRA dropout probability"
                },
                "target_modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": "Target modules for LoRA adaptation"
                },
                "bias": {
                    "type": "string",
                    "enum": ["none", "all", "lora_only"],
                    "default": "none",
                    "description": "Bias training strategy"
                },
                "task_type": {
                    "type": "string",
                    "enum": ["SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS"],
                    "default": "SEQ_CLS",
                    "description": "Task type for PEFT"
                }
            },
            "required": ["rank"],
            "additionalProperties": False,
            "description": "LoRA (Low-Rank Adaptation) configuration"
        }
        
        # QLoRA configuration
        cls._definitions['qlora_config'] = {
            "type": "object",
            "allOf": [
                {"$ref": "#/definitions/lora_config"}
            ],
            "properties": {
                "bits": {
                    "type": "integer",
                    "enum": [4, 8],
                    "default": 4,
                    "description": "Quantization bits (4 or 8)"
                },
                "quant_type": {
                    "type": "string",
                    "enum": ["nf4", "fp4"],
                    "default": "nf4",
                    "description": "Quantization type"
                },
                "double_quant": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use double quantization"
                },
                "compute_dtype": {
                    "type": "string",
                    "enum": ["float16", "bfloat16", "float32"],
                    "default": "bfloat16",
                    "description": "Compute dtype for QLoRA"
                }
            },
            "description": "QLoRA (Quantized LoRA) configuration"
        }
        
        # Optimizer configuration
        cls._definitions['optimizer_config'] = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["adamw", "adam", "sgd", "adafactor", "lamb"],
                    "description": "Optimizer type"
                },
                "lr": {
                    "$ref": "#/definitions/learning_rate",
                    "description": "Learning rate"
                },
                "weight_decay": {
                    "$ref": "#/definitions/non_negative_number",
                    "description": "Weight decay (L2 regularization)"
                },
                "betas": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/unit_interval"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": "Adam beta parameters"
                },
                "eps": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": "Epsilon for numerical stability"
                }
            },
            "required": ["type", "lr"],
            "description": "Optimizer configuration"
        }
        
        # Scheduler configuration
        cls._definitions['scheduler_config'] = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "linear",
                        "cosine",
                        "cosine_with_restarts",
                        "polynomial",
                        "constant",
                        "constant_with_warmup"
                    ],
                    "description": "Learning rate scheduler type"
                },
                "num_warmup_steps": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Number of warmup steps"
                },
                "num_training_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Total number of training steps"
                }
            },
            "required": ["type"],
            "description": "Learning rate scheduler configuration"
        }
        
        # Regularization configuration
        cls._definitions['regularization_config'] = {
            "type": "object",
            "properties": {
                "dropout": {
                    "$ref": "#/definitions/dropout_rate"
                },
                "attention_dropout": {
                    "$ref": "#/definitions/dropout_rate"
                },
                "hidden_dropout": {
                    "$ref": "#/definitions/dropout_rate"
                },
                "weight_decay": {
                    "$ref": "#/definitions/non_negative_number"
                },
                "label_smoothing": {
                    "$ref": "#/definitions/unit_interval"
                },
                "mixup_alpha": {
                    "$ref": "#/definitions/non_negative_number"
                },
                "cutmix_alpha": {
                    "$ref": "#/definitions/non_negative_number"
                }
            },
            "description": "Regularization techniques configuration"
        }
        
        # Early stopping configuration
        cls._definitions['early_stopping_config'] = {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True
                },
                "patience": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Number of epochs to wait for improvement"
                },
                "min_delta": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Minimum change to qualify as improvement"
                },
                "monitor": {
                    "type": "string",
                    "enum": ["val_loss", "val_accuracy", "val_f1"],
                    "default": "val_loss",
                    "description": "Metric to monitor"
                },
                "mode": {
                    "type": "string",
                    "enum": ["min", "max"],
                    "description": "Whether to minimize or maximize monitored metric"
                }
            },
            "required": ["patience"],
            "description": "Early stopping configuration"
        }
        
        # Platform specification
        cls._definitions['platform_spec'] = {
            "type": "string",
            "enum": ["local", "colab", "kaggle", "gitpod", "codespaces"],
            "description": "Execution platform"
        }
        
        # Resource constraints
        cls._definitions['resource_constraints'] = {
            "type": "object",
            "properties": {
                "max_memory_gb": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Maximum RAM in GB"
                },
                "max_vram_gb": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Maximum GPU VRAM in GB"
                },
                "max_batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum batch size"
                },
                "max_sequence_length": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum sequence length"
                }
            },
            "description": "Resource constraint specifications"
        }
    
    @classmethod
    def _register_model_schemas(cls) -> None:
        """Register model configuration schemas."""
        
        # Base model schema
        base_model_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Model Configuration Schema",
            "description": "Schema for model configurations in AG News Text Classification",
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "$ref": "#/definitions/model_name",
                            "description": "Model name or path"
                        },
                        "type": {
                            "type": "string",
                            "enum": [
                                "deberta",
                                "roberta",
                                "electra",
                                "xlnet",
                                "longformer",
                                "t5",
                                "llama",
                                "mistral",
                                "falcon",
                                "mpt",
                                "phi"
                            ],
                            "description": "Model type/family"
                        },
                        "num_labels": {
                            "type": "integer",
                            "minimum": 2,
                            "default": 4,
                            "description": "Number of classification labels (AG News: 4)"
                        },
                        "max_length": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 8192,
                            "default": 512,
                            "description": "Maximum sequence length"
                        },
                        "hidden_dropout_prob": {
                            "$ref": "#/definitions/dropout_rate"
                        },
                        "attention_
