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
        - Adapter configuration schema
        - Prefix tuning configuration schema
        - Prompt tuning configuration schema
    
    Domain Schemas (Type-specific):
        - Model configuration schema (transformers, LLMs, ensemble)
        - Training configuration schema (standard, efficient, advanced)
        - Data configuration schema (preprocessing, augmentation, validation)
        - Ensemble configuration schema (voting, stacking, blending)
        - API configuration schema (REST, local)
        - Deployment configuration schema (local, free-tier, platform)
        - Overfitting prevention schema
        - Platform configuration schema
        - Service configuration schema
        - Quota management schema
    
    Composite Schemas (Combined):
        - Full training pipeline schema
        - SOTA model configuration schema
        - Platform-optimized configuration schema
        - Tier-specific configuration schemas

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
        - Parameter efficiency requirements
    
    Overfitting Prevention Constraints:
        - Train/val/test split ratios must sum to 1.0
        - Minimum validation set size (recommended: 0.1-0.2)
        - Maximum model complexity for dataset size
        - Required regularization for large models
        - Test set access protection
        - Cross-validation requirements
    
    Platform-Specific Constraints:
        - Memory limits (Colab Free: 12GB, Colab Pro: 25GB, Kaggle: 13GB)
        - GPU VRAM limits (T4: 16GB, P100: 16GB, V100: 16GB)
        - Session timeout constraints
        - API quota limits
        - Storage quota limits
        - Compute quota limits

Schema Versioning:
    Schemas are versioned to support migration:
    
    Version Format: MAJOR.MINOR.PATCH (Semantic Versioning 2.0.0)
    - MAJOR: Breaking changes (incompatible API changes)
    - MINOR: Backward-compatible additions (new features)
    - PATCH: Bug fixes and clarifications
    
    Migration Support:
        - Automatic schema migration between versions
        - Deprecation warnings for old fields
        - Default value insertion for new required fields
        - Version compatibility matrix

Custom Validators:
    Beyond JSON Schema standard validation, custom validators are provided:
    
    - validate_model_platform_compatibility: Check model fits platform resources
    - validate_lora_efficiency: Validate LoRA configuration efficiency
    - validate_qlora_configuration: Validate QLoRA quantization settings
    - validate_ensemble_diversity: Ensure ensemble model diversity
    - validate_overfitting_risk: Assess overfitting risk based on configuration
    - validate_resource_constraints: Check resource usage within limits
    - validate_quota_compliance: Verify quota usage compliance
    - validate_test_set_protection: Ensure test set access protection
    - validate_parameter_efficiency: Check parameter efficiency ratio
    
    These validators implement complex business logic that cannot be
    expressed in pure JSON Schema.

References:
    JSON Schema Specification:
        - Wright, A. et al. (2019). "JSON Schema: A Media Type for Describing 
          JSON Documents". Internet-Draft draft-handrews-json-schema-02.
        - JSON Schema Specification: https://json-schema.org/specification.html
        - Understanding JSON Schema: https://json-schema.org/understanding-json-schema/
    
    Schema Design Patterns:
        - Pezoa, F., Reutter, J. L., Suarez, F., Ugarte, M., & Vrgoč, D. (2016). 
          "Foundations of JSON Schema". Proceedings of the 25th International 
          Conference on World Wide Web (WWW), 263-273.
        - Bourhis, P., Reutter, J. L., Suárez, F., & Vrgoč, D. (2017). 
          "JSON: Data model, Query languages and Schema specification". 
          ACM SIGMOD/PODS, 1-6.
    
    Configuration Validation Best Practices:
        - Humble, J. & Farley, D. (2010). "Continuous Delivery: Reliable Software 
          Releases through Build, Test, and Deployment Automation". Addison-Wesley.
          Chapter 2: Configuration Management.
        - Newman, S. (2015). "Building Microservices: Designing Fine-Grained Systems". 
          O'Reilly Media. Chapter 11: Microservices at Scale.
        - Kim, G., Humble, J., Debois, P., & Willis, J. (2016). "The DevOps Handbook". 
          IT Revolution Press. Part III: The Technical Practices of Flow.
    
    Machine Learning Configuration:
        - Baylor, D., Breck, E., Cheng, H. T., Fiedel, N., Foo, C. Y., Haque, Z., 
          Haykal, S., Ispir, M., Jain, V., Koc, L., Koo, C. Y., Lew, L., Mewald, C., 
          Modi, A. N., Polyzotis, N., Ramesh, S., Roy, S., Whang, S. E., Wicke, M., 
          Wilkiewicz, J., Zhang, X., & Zinkevich, M. (2017). "TFX: A TensorFlow-Based 
          Production-Scale Machine Learning Platform". KDD, 1387-1395.
        - Polyzotis, N., Roy, S., Whang, S. E., & Zinkevich, M. (2018). 
          "Data Lifecycle Challenges in Production Machine Learning: A Survey". 
          SIGMOD Record, 47(2), 17-28.
        - Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., 
          Chaudhary, V., Young, M., Crespo, J. F., & Dennison, D. (2015). 
          "Hidden Technical Debt in Machine Learning Systems". NIPS, 2503-2511.
    
    Parameter-Efficient Fine-Tuning:
        - Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., 
          Wang, L., & Chen, W. (2021). "LoRA: Low-Rank Adaptation of Large Language Models". 
          arXiv:2106.09685.
        - Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). 
          "QLoRA: Efficient Finetuning of Quantized LLMs". arXiv:2305.14314.
        - Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., 
          Gesmundo, A., Attariyan, M., & Gelly, S. (2019). "Parameter-Efficient 
          Transfer Learning for NLP". ICML, 2790-2799.

Usage:
    Get schema for configuration type:
        from configs.config_schema import ConfigSchema
        
        schema = ConfigSchema.get_schema('model')
    
    Validate configuration against schema:
        is_valid = ConfigSchema.validate(config, 'model')
        
        try:
            ConfigSchema.validate(config, 'training', strict=True)
        except SchemaError as e:
            print(f"Validation failed: {e}")
    
    Get all available schemas:
        schemas = ConfigSchema.list_schemas()
    
    Register custom schema:
        ConfigSchema.register_schema('custom', my_schema)
    
    Generate schema from example:
        schema = ConfigSchema.generate_from_example(example_config)
    
    Compose schemas:
        composed = ConfigSchema.compose_schema([
            ConfigSchema.get_schema('model'),
            ConfigSchema.get_schema('lora_config')
        ])
    
    Get schema version:
        version = ConfigSchema.get_version()

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
Project: AG News Text Classification (ag-news-text-classification)
Repository: https://github.com/VoHaiDung/ag-news-text-classification
"""

import logging
from typing import Any, Dict, List, Optional, Union, Set, Callable, Tuple
from pathlib import Path
from copy import deepcopy
import json


__version__ = "1.0.0"
__author__ = "Võ Hải Dũng"
__email__ = "vohaidung.work@gmail.com"
__license__ = "MIT"
__project__ = "AG News Text Classification (ag-news-text-classification)"


logger = logging.getLogger(__name__)


CONFIGS_ROOT = Path(__file__).parent
PROJECT_ROOT = CONFIGS_ROOT.parent


class SchemaError(Exception):
    """
    Exception raised for schema-related errors.
    
    This exception is raised when schema operations fail, such as
    schema not found, invalid schema definition, schema composition errors,
    or validation failures in strict mode.
    
    Attributes:
        message: Human-readable error description
        schema_type: Type of schema that caused the error
        details: Additional error context and debugging information
    
    Examples:
        Raise schema error:
            raise SchemaError(
                "Invalid schema definition",
                schema_type="model",
                details={"missing_field": "type"}
            )
        
        Catch schema error:
            try:
                schema = ConfigSchema.get_schema('unknown')
            except SchemaError as e:
                print(f"Error: {e.message}")
                print(f"Details: {e.details}")
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
            schema_type: Type of schema (e.g., 'model', 'training')
            details: Additional error details dictionary
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


class ValidationError(Exception):
    """
    Exception raised for configuration validation errors.
    
    This exception is raised when a configuration fails schema validation.
    It provides detailed information about the validation failure including
    the validation path, failed constraint, and suggested fixes.
    
    Attributes:
        message: Human-readable error description
        path: JSONPath to the invalid field
        validator: Name of the failed validator (e.g., 'minimum', 'enum')
        constraint: The constraint that was violated
        actual_value: The actual value that failed validation
        expected: Expected value or constraint description
    
    Examples:
        Raise validation error:
            raise ValidationError(
                "Learning rate out of range",
                path="optimizer.lr",
                validator="maximum",
                constraint=1.0,
                actual_value=2.0
            )
    """
    
    def __init__(
        self,
        message: str,
        path: Optional[str] = None,
        validator: Optional[str] = None,
        constraint: Optional[Any] = None,
        actual_value: Optional[Any] = None,
        expected: Optional[str] = None
    ):
        """
        Initialize ValidationError.
        
        Args:
            message: Error description
            path: JSONPath to invalid field
            validator: Failed validator name
            constraint: Violated constraint
            actual_value: Actual value
            expected: Expected value description
        """
        self.message = message
        self.path = path
        self.validator = validator
        self.constraint = constraint
        self.actual_value = actual_value
        self.expected = expected
        
        error_msg = f"Validation Error: {message}"
        if path:
            error_msg += f" (Path: {path})"
        if validator:
            error_msg += f" (Validator: {validator})"
        if constraint is not None:
            error_msg += f" (Constraint: {constraint})"
        if actual_value is not None:
            error_msg += f" (Actual: {actual_value})"
        if expected:
            error_msg += f" (Expected: {expected})"
        
        super().__init__(error_msg)


class ConfigSchema:
    """
    Comprehensive configuration schema registry and validator.
    
    This class provides centralized access to all configuration schemas
    used in the AG News Text Classification (ag-news-text-classification) project.
    It supports schema registration, retrieval, validation, composition,
    and generation.
    
    The schema system is based on JSON Schema Draft 7 specification with
    custom extensions for domain-specific validation rules specific to
    machine learning, natural language processing, and the AG News dataset.
    
    Class Attributes:
        _schemas: Registry of all schemas indexed by type
        _definitions: Reusable schema components (shared definitions)
        _validators: Custom validation functions for complex constraints
        _schema_version: Current schema version (Semantic Versioning)
        _initialized: Initialization flag to prevent duplicate registration
    
    Methods:
        Class Methods:
            initialize: Initialize schema registry
            get_schema: Get schema by type
            validate: Validate configuration against schema
            register_schema: Register custom schema
            list_schemas: List all available schema types
            get_definitions: Get reusable schema definitions
            compose_schema: Compose schema from multiple parts
            generate_from_example: Generate schema from example config
            get_version: Get current schema version
            migrate_schema: Migrate schema to different version
        
        Private Methods:
            _register_definitions: Register reusable definitions
            _register_model_schemas: Register model configuration schemas
            _register_training_schemas: Register training configuration schemas
            _register_data_schemas: Register data configuration schemas
            _register_ensemble_schemas: Register ensemble configuration schemas
            _register_api_schemas: Register API configuration schemas
            _register_deployment_schemas: Register deployment configuration schemas
            _register_overfitting_prevention_schemas: Register overfitting prevention schemas
            _register_platform_schemas: Register platform configuration schemas
            _register_service_schemas: Register service configuration schemas
            _register_quota_schemas: Register quota management schemas
            _register_experiment_schemas: Register experiment configuration schemas
    
    Examples:
        Get model schema:
            schema = ConfigSchema.get_schema('model')
        
        Validate configuration:
            is_valid = ConfigSchema.validate(config, 'training')
        
        Strict validation:
            try:
                ConfigSchema.validate(config, 'training', strict=True)
            except ValidationError as e:
                print(f"Validation failed at {e.path}: {e.message}")
        
        Register custom schema:
            ConfigSchema.register_schema('my_type', my_schema)
        
        List available schemas:
            types = ConfigSchema.list_schemas()
            print(f"Available: {', '.join(types)}")
        
        Compose schemas:
            model_with_lora = ConfigSchema.compose_schema([
                ConfigSchema.get_schema('model_base'),
                ConfigSchema.get_schema('lora_config')
            ])
    """
    
    _schemas: Dict[str, Dict[str, Any]] = {}
    _definitions: Dict[str, Dict[str, Any]] = {}
    _validators: Dict[str, Callable] = {}
    _schema_version = "1.0.0"
    _initialized = False
    
    @classmethod
    def initialize(cls) -> None:
        """
        Initialize schema registry with all predefined schemas.
        
        This method is called automatically on first use. It loads all
        schema definitions and registers them in the schema registry.
        The initialization process includes:
        
        1. Register reusable definitions (primitives, components)
        2. Register type-specific schemas (model, training, data, etc.)
        3. Register composite schemas (full pipelines, SOTA configs)
        4. Register custom validators
        
        The method is idempotent and safe to call multiple times.
        
        Raises:
            SchemaError: If schema initialization fails
        """
        if cls._initialized:
            return
        
        try:
            cls._register_definitions()
            cls._register_model_schemas()
            cls._register_training_schemas()
            cls._register_data_schemas()
            cls._register_ensemble_schemas()
            cls._register_api_schemas()
            cls._register_deployment_schemas()
            cls._register_overfitting_prevention_schemas()
            cls._register_platform_schemas()
            cls._register_service_schemas()
            cls._register_quota_schemas()
            cls._register_experiment_schemas()
            
            cls._initialized = True
            logger.info(
                f"Schema registry initialized successfully "
                f"with {len(cls._schemas)} schemas and "
                f"{len(cls._definitions)} definitions"
            )
        except Exception as e:
            logger.error(f"Schema initialization failed: {e}")
            raise SchemaError(
                "Failed to initialize schema registry",
                details={"error": str(e)}
            )
    
    @classmethod
    def get_schema(
        cls,
        schema_type: str,
        include_definitions: bool = True,
        resolve_refs: bool = False
    ) -> Dict[str, Any]:
        """
        Get schema by type.
        
        Args:
            schema_type: Type of schema to retrieve (e.g., 'model', 'training')
            include_definitions: Whether to include definitions section
            resolve_refs: Whether to resolve $ref references inline
        
        Returns:
            Schema dictionary conforming to JSON Schema Draft 7
        
        Raises:
            SchemaError: If schema type not found
        
        Examples:
            Get model schema:
                schema = ConfigSchema.get_schema('model')
            
            Get schema without definitions:
                schema = ConfigSchema.get_schema('training', include_definitions=False)
            
            Get schema with resolved references:
                schema = ConfigSchema.get_schema('model', resolve_refs=True)
        """
        cls.initialize()
        
        if schema_type not in cls._schemas:
            available = ', '.join(sorted(cls._schemas.keys()))
            raise SchemaError(
                f"Schema type '{schema_type}' not found",
                schema_type=schema_type,
                details={
                    'available_types': available,
                    'total_schemas': len(cls._schemas)
                }
            )
        
        schema = deepcopy(cls._schemas[schema_type])
        
        if include_definitions and cls._definitions:
            if 'definitions' not in schema:
                schema['definitions'] = {}
            schema['definitions'].update(deepcopy(cls._definitions))
        
        if resolve_refs:
            schema = cls._resolve_references(schema)
        
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
        
        This method performs comprehensive validation including:
        - JSON Schema Draft 7 validation
        - Custom domain-specific validation
        - Platform compatibility validation
        - Resource constraint validation
        
        Args:
            config: Configuration dictionary to validate
            schema_type: Type of schema to validate against
            strict: Whether to raise exception on validation failure
        
        Returns:
            True if validation succeeds, False otherwise
        
        Raises:
            ValidationError: If strict=True and validation fails
            SchemaError: If schema type not found
        
        Examples:
            Validate model config:
                is_valid = ConfigSchema.validate(config, 'model')
            
            Strict validation with exception:
                try:
                    ConfigSchema.validate(config, 'training', strict=True)
                except ValidationError as e:
                    print(f"Invalid at {e.path}: {e.message}")
            
            Custom error handling:
                if not ConfigSchema.validate(config, 'model'):
                    print("Configuration is invalid")
        """
        try:
            import jsonschema
            from jsonschema import Draft7Validator, ValidationError as JSONValidationError
        except ImportError:
            logger.warning(
                "jsonschema package not available, validation skipped. "
                "Install with: pip install jsonschema"
            )
            return True
        
        try:
            schema = cls.get_schema(schema_type)
            validator = Draft7Validator(schema)
            validator.validate(config)
            
            if schema_type in cls._validators:
                custom_validator = cls._validators[schema_type]
                custom_validator(config)
            
            return True
        
        except JSONValidationError as e:
            error_path = '.'.join(str(p) for p in e.path) if e.path else 'root'
            
            if strict:
                raise ValidationError(
                    message=e.message,
                    path=error_path,
                    validator=e.validator,
                    constraint=e.validator_value,
                    actual_value=e.instance
                )
            else:
                logger.error(
                    f"Schema validation failed for {schema_type} "
                    f"at {error_path}: {e.message}"
                )
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
        overwrite: bool = False,
        validate_schema: bool = True
    ) -> None:
        """
        Register a custom schema.
        
        Args:
            schema_type: Type identifier for the schema
            schema: Schema dictionary conforming to JSON Schema Draft 7
            overwrite: Whether to overwrite existing schema
            validate_schema: Whether to validate the schema itself
        
        Raises:
            SchemaError: If schema type already exists and overwrite=False
            SchemaError: If schema is invalid
        
        Examples:
            Register custom schema:
                custom_schema = {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {...}
                }
                ConfigSchema.register_schema('my_type', custom_schema)
            
            Overwrite existing schema:
                ConfigSchema.register_schema('model', new_schema, overwrite=True)
            
            Skip schema validation:
                ConfigSchema.register_schema('fast', schema, validate_schema=False)
        """
        if schema_type in cls._schemas and not overwrite:
            raise SchemaError(
                f"Schema type '{schema_type}' already registered",
                schema_type=schema_type,
                details={'hint': 'Use overwrite=True to replace existing schema'}
            )
        
        if validate_schema:
            try:
                import jsonschema
                Draft7Validator.check_schema(schema)
            except Exception as e:
                raise SchemaError(
                    f"Invalid schema definition: {e}",
                    schema_type=schema_type,
                    details={'error': str(e)}
                )
        
        cls._schemas[schema_type] = schema
        logger.info(f"Registered schema: {schema_type}")
    
    @classmethod
    def list_schemas(cls) -> List[str]:
        """
        List all available schema types.
        
        Returns:
            Sorted list of schema type identifiers
        
        Examples:
            Get all schema types:
                types = ConfigSchema.list_schemas()
                print(f"Available schemas ({len(types)}): {', '.join(types)}")
            
            Check if schema exists:
                if 'model' in ConfigSchema.list_schemas():
                    schema = ConfigSchema.get_schema('model')
        """
        cls.initialize()
        return sorted(cls._schemas.keys())
    
    @classmethod
    def get_definitions(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get all reusable schema definitions.
        
        Returns:
            Dictionary of schema definitions indexed by name
        
        Examples:
            Get all definitions:
                defs = ConfigSchema.get_definitions()
                lora_def = defs['lora_config']
            
            Check available definitions:
                defs = ConfigSchema.get_definitions()
                print(f"Available definitions: {list(defs.keys())}")
        """
        cls.initialize()
        return deepcopy(cls._definitions)
    
    @classmethod
    def compose_schema(
        cls,
        schemas: List[Dict[str, Any]],
        composition_type: str = "allOf"
    ) -> Dict[str, Any]:
        """
        Compose multiple schemas into a single schema.
        
        Args:
            schemas: List of schema dictionaries to compose
            composition_type: Type of composition ('allOf', 'anyOf', 'oneOf')
        
        Returns:
            Composed schema dictionary
        
        Raises:
            SchemaError: If composition type is invalid
        
        Examples:
            Compose with allOf (intersection):
                schema = ConfigSchema.compose_schema([
                    base_schema,
                    lora_schema
                ], composition_type="allOf")
            
            Compose with anyOf (union):
                schema = ConfigSchema.compose_schema([
                    fast_config,
                    accurate_config
                ], composition_type="anyOf")
        """
        if composition_type not in ["allOf", "anyOf", "oneOf"]:
            raise SchemaError(
                f"Invalid composition type: {composition_type}",
                details={'valid_types': ['allOf', 'anyOf', 'oneOf']}
            )
        
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            composition_type: schemas
        }
    
    @classmethod
    def get_version(cls) -> str:
        """
        Get current schema version.
        
        Returns:
            Schema version string in Semantic Versioning format
        
        Examples:
            Get version:
                version = ConfigSchema.get_version()
                print(f"Schema version: {version}")
        """
        return cls._schema_version
    
    @classmethod
    def _resolve_references(cls, schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve all $ref references in schema.
        
        Args:
            schema: Schema with references
        
        Returns:
            Schema with resolved references
        """
        if isinstance(schema, dict):
            if "$ref" in schema:
                ref_path = schema["$ref"]
                if ref_path.startswith("#/definitions/"):
                    def_name = ref_path.split("/")[-1]
                    if def_name in cls._definitions:
                        return deepcopy(cls._definitions[def_name])
                return schema
            else:
                return {k: cls._resolve_references(v) for k, v in schema.items()}
        elif isinstance(schema, list):
            return [cls._resolve_references(item) for item in schema]
        else:
            return schema
    
    @classmethod
    def _register_definitions(cls) -> None:
        """
        Register reusable schema definitions.
        
        This method registers all reusable schema components that can be
        referenced from other schemas using $ref. Definitions include:
        
        Primitive Types:
            - positive_number: Number > 0
            - non_negative_number: Number >= 0
            - positive_integer: Integer >= 1
            - unit_interval: Number in [0, 1]
        
        ML-Specific Types:
            - learning_rate: Typically (0, 0.1]
            - dropout_rate: [0, 1]
            - batch_size: Positive integer
            - num_epochs: Positive integer
        
        Model Components:
            - lora_config: LoRA configuration
            - qlora_config: QLoRA configuration
            - adapter_config: Adapter configuration
            - prefix_tuning_config: Prefix tuning configuration
            - prompt_tuning_config: Prompt tuning configuration
        
        Training Components:
            - optimizer_config: Optimizer configuration
            - scheduler_config: Learning rate scheduler configuration
            - regularization_config: Regularization techniques
            - early_stopping_config: Early stopping configuration
        
        Platform Components:
            - platform_spec: Platform identifier
            - resource_constraints: Resource limits
            - quota_limits: Usage quota specifications
        """
        
        cls._definitions['positive_number'] = {
            "type": "number",
            "exclusiveMinimum": 0,
            "description": "A positive number greater than zero"
        }
        
        cls._definitions['non_negative_number'] = {
            "type": "number",
            "minimum": 0,
            "description": "A non-negative number (zero or positive)"
        }
        
        cls._definitions['positive_integer'] = {
            "type": "integer",
            "minimum": 1,
            "description": "A positive integer greater than or equal to 1"
        }
        
        cls._definitions['unit_interval'] = {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "A number in the closed interval [0, 1]"
        }
        
        cls._definitions['learning_rate'] = {
            "type": "number",
            "exclusiveMinimum": 0,
            "maximum": 1,
            "description": (
                "Learning rate for optimization, typically in range (0, 1e-1]. "
                "Common values: 1e-5 for large models, 1e-4 for medium, 1e-3 for small."
            )
        }
        
        cls._definitions['dropout_rate'] = {
            "$ref": "#/definitions/unit_interval",
            "description": (
                "Dropout probability in range [0, 1]. "
                "Typical values: 0.1 for robust models, 0.3 for regularization, 0.5 for strong regularization."
            )
        }
        
        cls._definitions['batch_size'] = {
            "type": "integer",
            "minimum": 1,
            "description": (
                "Batch size for training or inference. "
                "Power of 2 recommended for GPU efficiency (8, 16, 32, 64, etc.)."
            )
        }
        
        cls._definitions['num_epochs'] = {
            "type": "integer",
            "minimum": 1,
            "description": "Number of training epochs (full passes through the dataset)"
        }
        
        cls._definitions['model_name'] = {
            "type": "string",
            "minLength": 1,
            "description": "Model name or identifier (e.g., 'microsoft/deberta-v3-large')"
        }
        
        cls._definitions['file_path'] = {
            "type": "string",
            "minLength": 1,
            "pattern": "^[^\\0]+$",
            "description": "File system path (absolute or relative)"
        }
        
        cls._definitions['lora_config'] = {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to enable LoRA"
                },
                "rank": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 256,
                    "description": (
                        "LoRA rank (r), determines the dimensionality of low-rank matrices. "
                        "Typical values: 4-8 for efficiency, 16-32 for balance, 64+ for accuracy."
                    )
                },
                "alpha": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "description": (
                        "LoRA alpha scaling parameter. "
                        "Scaling factor is alpha/rank. Common: alpha=16 for rank=8, alpha=32 for rank=16."
                    )
                },
                "dropout": {
                    "$ref": "#/definitions/dropout_rate",
                    "description": "LoRA dropout probability for regularization"
                },
                "target_modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": (
                        "Target modules for LoRA adaptation. "
                        "Common: ['query', 'value'] for Q-V, ['query', 'key', 'value'] for Q-K-V, "
                        "['query', 'value', 'dense'] for full attention."
                    )
                },
                "bias": {
                    "type": "string",
                    "enum": ["none", "all", "lora_only"],
                    "default": "none",
                    "description": (
                        "Bias training strategy. "
                        "'none': No bias training, 'all': Train all biases, 'lora_only': Only LoRA biases."
                    )
                },
                "task_type": {
                    "type": "string",
                    "enum": ["SEQ_CLS", "SEQ_2_SEQ_LM", "CAUSAL_LM", "TOKEN_CLS", "QUESTION_ANS"],
                    "default": "SEQ_CLS",
                    "description": "Task type for PEFT library (AG News uses SEQ_CLS)"
                },
                "modules_to_save": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional modules to save and train (e.g., classifier head)"
                }
            },
            "required": ["rank"],
            "additionalProperties": False,
            "description": "LoRA (Low-Rank Adaptation) configuration for parameter-efficient fine-tuning"
        }
        
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
                    "description": (
                        "Quantization bits. "
                        "4-bit: Maximum memory reduction, slight accuracy loss. "
                        "8-bit: Better accuracy, moderate memory reduction."
                    )
                },
                "quant_type": {
                    "type": "string",
                    "enum": ["nf4", "fp4"],
                    "default": "nf4",
                    "description": (
                        "Quantization type. "
                        "nf4: Normal Float 4-bit (recommended for most cases). "
                        "fp4: Float Point 4-bit (alternative)."
                    )
                },
                "double_quant": {
                    "type": "boolean",
                    "default": True,
                    "description": (
                        "Use double quantization (quantize the quantization constants). "
                        "Reduces memory further with minimal accuracy impact."
                    )
                },
                "compute_dtype": {
                    "type": "string",
                    "enum": ["float16", "bfloat16", "float32"],
                    "default": "bfloat16",
                    "description": (
                        "Compute dtype for QLoRA operations. "
                        "bfloat16: Best for modern GPUs (A100, H100). "
                        "float16: Good for older GPUs. "
                        "float32: Highest precision, slowest."
                    )
                },
                "llm_int8_threshold": {
                    "type": "number",
                    "default": 6.0,
                    "description": "Threshold for outlier detection in 8-bit quantization"
                },
                "llm_int8_skip_modules": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Modules to skip in 8-bit quantization"
                }
            },
            "description": "QLoRA (Quantized LoRA) configuration for memory-efficient fine-tuning"
        }
        
        cls._definitions['adapter_config'] = {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True
                },
                "adapter_type": {
                    "type": "string",
                    "enum": ["houlsby", "pfeiffer", "parallel", "scaled_parallel"],
                    "default": "pfeiffer",
                    "description": (
                        "Adapter architecture type. "
                        "houlsby: Original adapter (after attention and FFN). "
                        "pfeiffer: More efficient (after FFN only). "
                        "parallel: Parallel to main layers. "
                        "scaled_parallel: Parallel with learnable scaling."
                    )
                },
                "reduction_factor": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 16,
                    "description": (
                        "Bottleneck reduction factor. "
                        "Higher values = fewer parameters but lower capacity. "
                        "Typical: 16-64."
                    )
                },
                "non_linearity": {
                    "type": "string",
                    "enum": ["relu", "gelu", "swish", "tanh"],
                    "default": "gelu",
                    "description": "Non-linearity function in adapter"
                },
                "adapter_dropout": {
                    "$ref": "#/definitions/dropout_rate",
                    "description": "Dropout probability in adapter"
                }
            },
            "required": ["adapter_type"],
            "description": "Adapter configuration for parameter-efficient fine-tuning"
        }
        
        cls._definitions['prefix_tuning_config'] = {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True
                },
                "num_virtual_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 512,
                    "default": 20,
                    "description": (
                        "Number of virtual tokens (prefix length). "
                        "Typical: 10-50 for classification, 100-200 for generation."
                    )
                },
                "token_dim": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Dimension of virtual tokens (usually model hidden size)"
                },
                "encoder_hidden_size": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Hidden size of prefix encoder"
                },
                "prefix_projection": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to use MLP reparameterization"
                }
            },
            "required": ["num_virtual_tokens"],
            "description": "Prefix Tuning configuration"
        }
        
        cls._definitions['prompt_tuning_config'] = {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True
                },
                "num_virtual_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 512,
                    "default": 8,
                    "description": "Number of soft prompt tokens"
                },
                "prompt_tuning_init": {
                    "type": "string",
                    "enum": ["random", "text"],
                    "default": "random",
                    "description": (
                        "Prompt initialization method. "
                        "'random': Random initialization. "
                        "'text': Initialize from text tokens."
                    )
                },
                "prompt_tuning_init_text": {
                    "type": "string",
                    "description": "Text for initialization (if init='text')"
                },
                "tokenizer_name_or_path": {
                    "type": "string",
                    "description": "Tokenizer for text initialization"
                }
            },
            "required": ["num_virtual_tokens"],
            "description": "Prompt Tuning configuration"
        }
        
        cls._definitions['optimizer_config'] = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["adamw", "adam", "sgd", "adafactor", "lamb", "adagrad", "rmsprop"],
                    "default": "adamw",
                    "description": (
                        "Optimizer type. "
                        "adamw: Adam with weight decay (recommended for transformers). "
                        "adam: Original Adam. "
                        "sgd: Stochastic Gradient Descent. "
                        "adafactor: Memory-efficient optimizer. "
                        "lamb: Layer-wise Adaptive Moments optimizer."
                    )
                },
                "lr": {
                    "$ref": "#/definitions/learning_rate",
                    "description": "Learning rate"
                },
                "weight_decay": {
                    "$ref": "#/definitions/non_negative_number",
                    "default": 0.01,
                    "description": (
                        "Weight decay (L2 regularization) coefficient. "
                        "Typical: 0.01 for most transformers, 0.0 for no regularization."
                    )
                },
                "betas": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/unit_interval"},
                    "minItems": 2,
                    "maxItems": 2,
                    "default": [0.9, 0.999],
                    "description": "Adam beta parameters [beta1, beta2]"
                },
                "eps": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "default": 1e-8,
                    "description": "Epsilon for numerical stability"
                },
                "momentum": {
                    "$ref": "#/definitions/unit_interval",
                    "description": "Momentum for SGD"
                },
                "nesterov": {
                    "type": "boolean",
                    "description": "Use Nesterov momentum"
                }
            },
            "required": ["type", "lr"],
            "additionalProperties": False,
            "description": "Optimizer configuration"
        }
        
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
                        "constant_with_warmup",
                        "inverse_sqrt"
                    ],
                    "default": "linear",
                    "description": (
                        "Learning rate scheduler type. "
                        "linear: Linear decay from initial LR to 0. "
                        "cosine: Cosine annealing. "
                        "cosine_with_restarts: Cosine with periodic restarts. "
                        "polynomial: Polynomial decay. "
                        "constant: No decay. "
                        "constant_with_warmup: Constant after warmup."
                    )
                },
                "num_warmup_steps": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": (
                        "Number of warmup steps (linear increase from 0 to initial LR). "
                        "Recommended: 5-10% of total training steps."
                    )
                },
                "num_training_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Total number of training steps"
                },
                "num_cycles": {
                    "type": "number",
                    "minimum": 0.5,
                    "default": 1.0,
                    "description": "Number of cycles for cosine_with_restarts"
                },
                "lr_end": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Final learning rate (for polynomial)"
                },
                "power": {
                    "type": "number",
                    "minimum": 0,
                    "default": 1.0,
                    "description": "Power for polynomial decay"
                }
            },
            "required": ["type"],
            "description": "Learning rate scheduler configuration"
        }
        
        cls._definitions['regularization_config'] = {
            "type": "object",
            "properties": {
                "dropout": {
                    "$ref": "#/definitions/dropout_rate",
                    "description": "General dropout probability"
                },
                "attention_dropout": {
                    "$ref": "#/definitions/dropout_rate",
                    "description": "Attention-specific dropout"
                },
                "hidden_dropout": {
                    "$ref": "#/definitions/dropout_rate",
                    "description": "Hidden layer dropout"
                },
                "classifier_dropout": {
                    "$ref": "#/definitions/dropout_rate",
                    "description": "Classifier head dropout"
                },
                "weight_decay": {
                    "$ref": "#/definitions/non_negative_number",
                    "description": "Weight decay coefficient"
                },
                "label_smoothing": {
                    "$ref": "#/definitions/unit_interval",
                    "description": (
                        "Label smoothing factor. "
                        "Typical: 0.1 for classification, 0.0 for no smoothing."
                    )
                },
                "mixup_alpha": {
                    "$ref": "#/definitions/non_negative_number",
                    "description": "Mixup alpha parameter (0 = disabled)"
                },
                "cutmix_alpha": {
                    "$ref": "#/definitions/non_negative_number",
                    "description": "CutMix alpha parameter (0 = disabled)"
                },
                "gradient_clip_norm": {
                    "$ref": "#/definitions/positive_number",
                    "description": "Maximum gradient norm for clipping"
                },
                "gradient_clip_value": {
                    "$ref": "#/definitions/positive_number",
                    "description": "Maximum gradient value for clipping"
                }
            },
            "description": "Regularization techniques configuration"
        }
        
        cls._definitions['early_stopping_config'] = {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to enable early stopping"
                },
                "patience": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 3,
                    "description": (
                        "Number of epochs to wait for improvement before stopping. "
                        "Typical: 3-5 for small datasets, 10+ for large datasets."
                    )
                },
                "min_delta": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.0,
                    "description": "Minimum change to qualify as improvement"
                },
                "monitor": {
                    "type": "string",
                    "enum": ["val_loss", "val_accuracy", "val_f1", "val_precision", "val_recall"],
                    "default": "val_loss",
                    "description": "Metric to monitor for early stopping"
                },
                "mode": {
                    "type": "string",
                    "enum": ["min", "max"],
                    "description": (
                        "Whether to minimize or maximize the monitored metric. "
                        "Use 'min' for loss, 'max' for accuracy/F1."
                    )
                },
                "baseline": {
                    "type": "number",
                    "description": "Baseline value for the monitored metric"
                },
                "restore_best_weights": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to restore model weights from best epoch"
                }
            },
            "required": ["patience"],
            "description": "Early stopping configuration"
        }
        
        cls._definitions['platform_spec'] = {
            "type": "string",
            "enum": ["local", "colab", "colab_pro", "kaggle", "gitpod", "codespaces", "auto"],
            "description": (
                "Execution platform specification. "
                "'auto': Auto-detect platform. "
                "'local': Local machine. "
                "'colab': Google Colab Free. "
                "'colab_pro': Google Colab Pro. "
                "'kaggle': Kaggle Notebooks."
            )
        }
        
        cls._definitions['resource_constraints'] = {
            "type": "object",
            "properties": {
                "max_memory_gb": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Maximum RAM in GB (0 = no limit)"
                },
                "max_vram_gb": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Maximum GPU VRAM in GB (0 = no limit)"
                },
                "max_batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum batch size"
                },
                "max_sequence_length": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8192,
                    "description": "Maximum sequence length"
                },
                "max_model_parameters": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of model parameters"
                },
                "gradient_accumulation_steps": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Gradient accumulation steps to simulate larger batch size"
                }
            },
            "description": "Resource constraint specifications for platform adaptation"
        }
        
        cls._definitions['quota_limits'] = {
            "type": "object",
            "properties": {
                "max_session_duration_hours": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Maximum session duration in hours"
                },
                "max_gpu_hours_per_week": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Maximum GPU hours per week"
                },
                "max_storage_gb": {
                    "type": "number",
                    "minimum": 0,
                    "description": "Maximum storage in GB"
                },
                "max_api_calls_per_day": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Maximum API calls per day"
                },
                "idle_timeout_minutes": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Idle timeout in minutes"
                }
            },
            "description": "Platform quota and usage limits"
        }
        
        cls._definitions['data_split_config'] = {
            "type": "object",
            "properties": {
                "train_ratio": {
                    "$ref": "#/definitions/unit_interval",
                    "description": "Training set ratio"
                },
                "val_ratio": {
                    "$ref": "#/definitions/unit_interval",
                    "description": "Validation set ratio"
                },
                "test_ratio": {
                    "$ref": "#/definitions/unit_interval",
                    "description": "Test set ratio"
                },
                "stratify": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to stratify split by labels"
                },
                "shuffle": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to shuffle data before split"
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                }
            },
            "required": ["train_ratio", "val_ratio", "test_ratio"],
            "description": "Data split configuration"
        }
        
        logger.debug(f"Registered {len(cls._definitions)} schema definitions")
    
    @classmethod
    def _register_model_schemas(cls) -> None:
        """
        Register model configuration schemas.
        
        This method registers schemas for all model types supported in the
        AG News Text Classification (ag-news-text-classification) project:
        
        Transformer Models:
            - DeBERTa (v2, v3): Base, Large, XLarge, XXLarge
            - RoBERTa: Base, Large, XLM-RoBERTa
            - ELECTRA: Base, Large
            - XLNet: Base, Large
            - Longformer: Base, Large
            - T5: Base, Large, 3B, Flan-T5
        
        Large Language Models:
            - LLaMA: LLaMA-2 (7B, 13B, 70B), LLaMA-3 (8B, 70B)
            - Mistral: Mistral-7B, Mistral-7B-Instruct, Mixtral-8x7B
            - Falcon: Falcon-7B, Falcon-40B
            - MPT: MPT-7B, MPT-30B
            - Phi: Phi-2, Phi-3
        
        Ensemble Models:
            - Voting ensemble
            - Stacking ensemble
            - Blending ensemble
            - Advanced ensemble methods
        """
        
        base_model_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Model Configuration Schema",
            "description": "Schema for model configurations in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "model": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "$ref": "#/definitions/model_name",
                            "examples": [
                                "microsoft/deberta-v3-large",
                                "roberta-large",
                                "meta-llama/Llama-2-7b-hf"
                            ]
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
                                "phi",
                                "ensemble"
                            ],
                            "description": "Model type/family"
                        },
                        "variant": {
                            "type": "string",
                            "description": "Model variant (e.g., 'base', 'large', 'xlarge')"
                        },
                        "num_labels": {
                            "type": "integer",
                            "minimum": 2,
                            "default": 4,
                            "description": "Number of classification labels (AG News: 4 classes)"
                        },
                        "max_length": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 8192,
                            "default": 512,
                            "description": "Maximum sequence length for tokenization"
                        },
                        "hidden_dropout_prob": {
                            "$ref": "#/definitions/dropout_rate"
                        },
                        "attention_probs_dropout_prob": {
                            "$ref": "#/definitions/dropout_rate"
                        },
                        "classifier_dropout": {
                            "$ref": "#/definitions/dropout_rate"
                        },
                        "use_cache": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to use HuggingFace model cache"
                        },
                        "pretrained": {
                            "type": "boolean",
                            "default": True,
                            "description": "Whether to load pretrained weights"
                        },
                        "trust_remote_code": {
                            "type": "boolean",
                            "default": False,
                            "description": "Whether to trust remote code (required for some models)"
                        }
                    },
                    "required": ["name", "type"],
                    "additionalProperties": True
                },
                "peft": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["lora", "qlora", "adapter", "prefix_tuning", "prompt_tuning", "ia3", "none"],
                            "default": "none",
                            "description": "Parameter-efficient fine-tuning method"
                        },
                        "lora": {
                            "$ref": "#/definitions/lora_config"
                        },
                        "qlora": {
                            "$ref": "#/definitions/qlora_config"
                        },
                        "adapter": {
                            "$ref": "#/definitions/adapter_config"
                        },
                        "prefix_tuning": {
                            "$ref": "#/definitions/prefix_tuning_config"
                        },
                        "prompt_tuning": {
                            "$ref": "#/definitions/prompt_tuning_config"
                        }
                    },
                    "description": "Parameter-efficient fine-tuning configuration"
                },
                "head": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["classification", "multitask", "hierarchical", "attention", "adaptive"],
                            "default": "classification"
                        },
                        "hidden_size": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Hidden size for classification head"
                        },
                        "num_hidden_layers": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "Number of hidden layers in classification head"
                        },
                        "activation": {
                            "type": "string",
                            "enum": ["relu", "gelu", "tanh", "swish"],
                            "default": "gelu"
                        },
                        "pooling_strategy": {
                            "type": "string",
                            "enum": ["cls", "mean", "max", "attention", "weighted"],
                            "default": "cls",
                            "description": "Pooling strategy for sequence representation"
                        }
                    }
                },
                "quantization": {
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "default": False
                        },
                        "bits": {
                            "type": "integer",
                            "enum": [4, 8],
                            "description": "Quantization bits"
                        },
                        "method": {
                            "type": "string",
                            "enum": ["dynamic", "static", "qat"],
                            "default": "dynamic"
                        }
                    }
                }
            },
            "required": ["model"],
            "additionalProperties": False
        }
        
        cls._schemas['model'] = base_model_schema
        cls._schemas['model_base'] = base_model_schema
        
        transformer_model_schema = deepcopy(base_model_schema)
        transformer_model_schema["properties"]["model"]["properties"]["type"]["enum"] = [
            "deberta", "roberta", "electra", "xlnet", "longformer", "t5"
        ]
        cls._schemas['model_transformer'] = transformer_model_schema
        
        llm_model_schema = deepcopy(base_model_schema)
        llm_model_schema["properties"]["model"]["properties"]["type"]["enum"] = [
            "llama", "mistral", "falcon", "mpt", "phi"
        ]
        llm_model_schema["properties"]["model"]["properties"]["load_in_8bit"] = {
            "type": "boolean",
            "default": False,
            "description": "Load model in 8-bit precision"
        }
        llm_model_schema["properties"]["model"]["properties"]["load_in_4bit"] = {
            "type": "boolean",
            "default": False,
            "description": "Load model in 4-bit precision"
        }
        llm_model_schema["properties"]["model"]["properties"]["device_map"] = {
            "type": "string",
            "default": "auto",
            "description": "Device mapping strategy for multi-GPU"
        }
        cls._schemas['model_llm'] = llm_model_schema
        
        logger.debug("Registered model configuration schemas")
    
    @classmethod
    def _register_training_schemas(cls) -> None:
        """
        Register training configuration schemas.
        
        This method registers schemas for training configurations including:
        - Standard training
        - Platform-adaptive training
        - Efficient training (LoRA, QLoRA)
        - Advanced training strategies
        - Multi-stage training
        """
        
        training_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Training Configuration Schema",
            "description": "Schema for training configurations in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "training": {
                    "type": "object",
                    "properties": {
                        "num_epochs": {
                            "$ref": "#/definitions/num_epochs",
                            "default": 10
                        },
                        "batch_size": {
                            "$ref": "#/definitions/batch_size",
                            "default": 16
                        },
                        "eval_batch_size": {
                            "$ref": "#/definitions/batch_size",
                            "description": "Batch size for evaluation (can be larger than training)"
                        },
                        "gradient_accumulation_steps": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 1,
                            "description": "Number of steps to accumulate gradients before update"
                        },
                        "mixed_precision": {
                            "type": "string",
                            "enum": ["no", "fp16", "bf16"],
                            "default": "no",
                            "description": "Mixed precision training mode"
                        },
                        "gradient_checkpointing": {
                            "type": "boolean",
                            "default": False,
                            "description": "Enable gradient checkpointing to save memory"
                        },
                        "seed": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 42,
                            "description": "Random seed for reproducibility"
                        },
                        "logging_steps": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 100,
                            "description": "Log every N steps"
                        },
                        "eval_steps": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Evaluate every N steps"
                        },
                        "save_steps": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Save checkpoint every N steps"
                        },
                        "save_total_limit": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Maximum number of checkpoints to keep"
                        },
                        "evaluation_strategy": {
                            "type": "string",
                            "enum": ["no", "steps", "epoch"],
                            "default": "epoch",
                            "description": "Evaluation strategy"
                        },
                        "save_strategy": {
                            "type": "string",
                            "enum": ["no", "steps", "epoch"],
                            "default": "epoch",
                            "description": "Save strategy"
                        },
                        "load_best_model_at_end": {
                            "type": "boolean",
                            "default": True,
                            "description": "Load best model at end of training"
                        },
                        "metric_for_best_model": {
                            "type": "string",
                            "default": "eval_loss",
                            "description": "Metric to use for best model selection"
                        },
                        "greater_is_better": {
                            "type": "boolean",
                            "description": "Whether higher metric value is better"
                        },
                        "warmup_ratio": {
                            "$ref": "#/definitions/unit_interval",
                            "default": 0.1,
                            "description": "Ratio of total steps for warmup"
                        },
                        "warmup_steps": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Number of warmup steps (overrides warmup_ratio)"
                        },
                        "max_grad_norm": {
                            "type": "number",
                            "minimum": 0,
                            "default": 1.0,
                            "description": "Maximum gradient norm for clipping"
                        },
                        "dataloader_num_workers": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                            "description": "Number of dataloader workers"
                        },
                        "dataloader_pin_memory": {
                            "type": "boolean",
                            "default": True,
                            "description": "Pin memory in dataloader"
                        }
                    },
                    "required": ["num_epochs", "batch_size"]
                },
                "optimizer": {
                    "$ref": "#/definitions/optimizer_config"
                },
                "scheduler": {
                    "$ref": "#/definitions/scheduler_config"
                },
                "regularization": {
                    "$ref": "#/definitions/regularization_config"
                },
                "early_stopping": {
                    "$ref": "#/definitions/early_stopping_config"
                },
                "platform": {
                    "$ref": "#/definitions/platform_spec"
                },
                "resources": {
                    "$ref": "#/definitions/resource_constraints"
                }
            },
            "required": ["training"],
            "additionalProperties": False
        }
        
        cls._schemas['training'] = training_schema
        cls._schemas['training_standard'] = training_schema
        
        logger.debug("Registered training configuration schemas")
    
    @classmethod
    def _register_data_schemas(cls) -> None:
        """Register data configuration schemas."""
        
        data_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Data Configuration Schema",
            "description": "Schema for data configurations in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "properties": {
                        "dataset_name": {
                            "type": "string",
                            "default": "ag_news",
                            "description": "Dataset name"
                        },
                        "data_dir": {
                            "$ref": "#/definitions/file_path",
                            "description": "Data directory path"
                        },
                        "cache_dir": {
                            "$ref": "#/definitions/file_path",
                            "description": "Cache directory for processed data"
                        },
                        "max_samples": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Maximum number of samples to use (for debugging)"
                        },
                        "preprocessing": {
                            "type": "object",
                            "properties": {
                                "lowercase": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Convert text to lowercase"
                                },
                                "remove_html": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Remove HTML tags"
                                },
                                "remove_urls": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Remove URLs"
                                },
                                "remove_special_chars": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Remove special characters"
                                }
                            }
                        },
                        "augmentation": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": False
                                },
                                "methods": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": [
                                            "back_translation",
                                            "paraphrase",
                                            "synonym_replacement",
                                            "random_insertion",
                                            "random_swap",
                                            "random_deletion",
                                            "mixup",
                                            "cutmix"
                                        ]
                                    },
                                    "description": "Augmentation methods to apply"
                                },
                                "augmentation_ratio": {
                                    "$ref": "#/definitions/unit_interval",
                                    "default": 0.1,
                                    "description": "Ratio of data to augment"
                                }
                            }
                        },
                        "split": {
                            "$ref": "#/definitions/data_split_config"
                        }
                    },
                    "required": ["dataset_name"]
                }
            },
            "required": ["data"],
            "additionalProperties": False
        }
        
        cls._schemas['data'] = data_schema
        
        logger.debug("Registered data configuration schemas")
    
    @classmethod
    def _register_ensemble_schemas(cls) -> None:
        """Register ensemble configuration schemas."""
        
        ensemble_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Ensemble Configuration Schema",
            "description": "Schema for ensemble configurations in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "ensemble": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["voting", "stacking", "blending", "bayesian", "snapshot"],
                            "description": "Ensemble method type"
                        },
                        "models": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {
                                        "type": "string",
                                        "description": "Model identifier"
                                    },
                                    "path": {
                                        "type": "string",
                                        "description": "Path to model checkpoint"
                                    },
                                    "weight": {
                                        "type": "number",
                                        "minimum": 0,
                                        "maximum": 1,
                                        "description": "Model weight in ensemble"
                                    }
                                },
                                "required": ["name", "path"]
                            },
                            "minItems": 2,
                            "description": "List of models in ensemble"
                        },
                        "voting": {
                            "type": "object",
                            "properties": {
                                "strategy": {
                                    "type": "string",
                                    "enum": ["hard", "soft", "weighted", "rank"],
                                    "default": "soft",
                                    "description": "Voting strategy"
                                }
                            }
                        },
                        "stacking": {
                            "type": "object",
                            "properties": {
                                "meta_learner": {
                                    "type": "string",
                                    "enum": ["logistic", "xgboost", "lightgbm", "catboost", "neural"],
                                    "description": "Meta-learner type"
                                },
                                "use_probabilities": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Use predicted probabilities as features"
                                },
                                "use_features": {
                                    "type": "boolean",
                                    "default": False,
                                    "description": "Use original features in meta-learner"
                                }
                            }
                        }
                    },
                    "required": ["type", "models"]
                }
            },
            "required": ["ensemble"],
            "additionalProperties": False
        }
        
        cls._schemas['ensemble'] = ensemble_schema
        
        logger.debug("Registered ensemble configuration schemas")
    
    @classmethod
    def _register_api_schemas(cls) -> None:
        """Register API configuration schemas."""
        
        api_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "API Configuration Schema",
            "description": "Schema for API configurations in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "api": {
                    "type": "object",
                    "properties": {
                        "host": {
                            "type": "string",
                            "default": "0.0.0.0",
                            "description": "API host address"
                        },
                        "port": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 65535,
                            "default": 8000,
                            "description": "API port number"
                        },
                        "workers": {
                            "type": "integer",
                            "minimum": 1,
                            "default": 1,
                            "description": "Number of API workers"
                        },
                        "reload": {
                            "type": "boolean",
                            "default": False,
                            "description": "Auto-reload on code changes"
                        },
                        "cors_enabled": {
                            "type": "boolean",
                            "default": True,
                            "description": "Enable CORS"
                        },
                        "cors_origins": {
                            "type": "array",
                            "items": {"type": "string"},
                            "default": ["*"],
                            "description": "Allowed CORS origins"
                        },
                        "rate_limit": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True
                                },
                                "requests_per_minute": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "default": 60
                                }
                            }
                        },
                        "auth": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": False
                                },
                                "type": {
                                    "type": "string",
                                    "enum": ["token", "api_key", "oauth"],
                                    "default": "token"
                                }
                            }
                        }
                    }
                }
            },
            "required": ["api"],
            "additionalProperties": False
        }
        
        cls._schemas['api'] = api_schema
        
        logger.debug("Registered API configuration schemas")
    
    @classmethod
    def _register_deployment_schemas(cls) -> None:
        """Register deployment configuration schemas."""
        
        deployment_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Deployment Configuration Schema",
            "description": "Schema for deployment configurations in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "deployment": {
                    "type": "object",
                    "properties": {
                        "platform": {
                            "$ref": "#/definitions/platform_spec"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["local", "docker", "cloud", "edge"],
                            "default": "local"
                        },
                        "model_path": {
                            "$ref": "#/definitions/file_path",
                            "description": "Path to model for deployment"
                        },
                        "optimization": {
                            "type": "object",
                            "properties": {
                                "quantization": {
                                    "type": "boolean",
                                    "default": False
                                },
                                "pruning": {
                                    "type": "boolean",
                                    "default": False
                                },
                                "onnx_export": {
                                    "type": "boolean",
                                    "default": False
                                }
                            }
                        }
                    },
                    "required": ["platform", "model_path"]
                }
            },
            "required": ["deployment"],
            "additionalProperties": False
        }
        
        cls._schemas['deployment'] = deployment_schema
        
        logger.debug("Registered deployment configuration schemas")
    
    @classmethod
    def _register_overfitting_prevention_schemas(cls) -> None:
        """Register overfitting prevention configuration schemas."""
        
        overfitting_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Overfitting Prevention Configuration Schema",
            "description": "Schema for overfitting prevention in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "overfitting_prevention": {
                    "type": "object",
                    "properties": {
                        "enabled": {
                            "type": "boolean",
                            "default": True
                        },
                        "test_set_protection": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True
                                },
                                "hash_verification": {
                                    "type": "boolean",
                                    "default": True
                                },
                                "access_logging": {
                                    "type": "boolean",
                                    "default": True
                                }
                            }
                        },
                        "monitoring": {
                            "type": "object",
                            "properties": {
                                "track_train_val_gap": {
                                    "type": "boolean",
                                    "default": True
                                },
                                "gap_threshold": {
                                    "type": "number",
                                    "minimum": 0,
                                    "default": 0.05,
                                    "description": "Maximum allowed train-val gap"
                                },
                                "alert_on_overfitting": {
                                    "type": "boolean",
                                    "default": True
                                }
                            }
                        },
                        "constraints": {
                            "type": "object",
                            "properties": {
                                "max_model_complexity": {
                                    "type": "integer",
                                    "description": "Maximum model parameters"
                                },
                                "min_validation_size": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                    "default": 0.1,
                                    "description": "Minimum validation set ratio"
                                },
                                "required_regularization": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Require regularization for large models"
                                }
                            }
                        }
                    }
                }
            },
            "required": ["overfitting_prevention"],
            "additionalProperties": False
        }
        
        cls._schemas['overfitting_prevention'] = overfitting_schema
        
        logger.debug("Registered overfitting prevention schemas")
    
    @classmethod
    def _register_platform_schemas(cls) -> None:
        """Register platform configuration schemas."""
        
        platform_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Platform Configuration Schema",
            "description": "Schema for platform-specific configurations in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "platform": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "$ref": "#/definitions/platform_spec"
                        },
                        "auto_detect": {
                            "type": "boolean",
                            "default": True,
                            "description": "Auto-detect platform"
                        },
                        "resources": {
                            "$ref": "#/definitions/resource_constraints"
                        },
                        "quotas": {
                            "$ref": "#/definitions/quota_limits"
                        },
                        "optimizations": {
                            "type": "object",
                            "properties": {
                                "auto_batch_size": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Auto-select batch size based on available memory"
                                },
                                "auto_precision": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Auto-select precision mode"
                                },
                                "memory_efficient": {
                                    "type": "boolean",
                                    "default": True,
                                    "description": "Enable memory-efficient optimizations"
                                }
                            }
                        }
                    },
                    "required": ["type"]
                }
            },
            "required": ["platform"],
            "additionalProperties": False
        }
        
        cls._schemas['platform'] = platform_schema
        
        logger.debug("Registered platform configuration schemas")
    
    @classmethod
    def _register_service_schemas(cls) -> None:
        """Register service configuration schemas."""
        
        service_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Service Configuration Schema",
            "description": "Schema for service configurations in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "service": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Service name"
                        },
                        "type": {
                            "type": "string",
                            "enum": ["prediction", "training", "data", "model_management"],
                            "description": "Service type"
                        },
                        "enabled": {
                            "type": "boolean",
                            "default": True
                        }
                    },
                    "required": ["name", "type"]
                }
            },
            "required": ["service"],
            "additionalProperties": False
        }
        
        cls._schemas['service'] = service_schema
        
        logger.debug("Registered service configuration schemas")
    
    @classmethod
    def _register_quota_schemas(cls) -> None:
        """Register quota management configuration schemas."""
        
        quota_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Quota Management Configuration Schema",
            "description": "Schema for quota management in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "quota": {
                    "type": "object",
                    "properties": {
                        "tracking_enabled": {
                            "type": "boolean",
                            "default": True,
                            "description": "Enable quota tracking"
                        },
                        "limits": {
                            "$ref": "#/definitions/quota_limits"
                        },
                        "alerts": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True
                                },
                                "threshold_percentage": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 100,
                                    "default": 80,
                                    "description": "Alert when usage exceeds this percentage"
                                }
                            }
                        }
                    }
                }
            },
            "required": ["quota"],
            "additionalProperties": False
        }
        
        cls._schemas['quota'] = quota_schema
        
        logger.debug("Registered quota management schemas")
    
    @classmethod
    def _register_experiment_schemas(cls) -> None:
        """Register experiment configuration schemas."""
        
        experiment_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Experiment Configuration Schema",
            "description": "Schema for experiment configurations in AG News Text Classification (ag-news-text-classification)",
            "type": "object",
            "properties": {
                "experiment": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Experiment name"
                        },
                        "description": {
                            "type": "string",
                            "description": "Experiment description"
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Experiment tags for organization"
                        },
                        "tracking": {
                            "type": "object",
                            "properties": {
                                "enabled": {
                                    "type": "boolean",
                                    "default": True
                                },
                                "backend": {
                                    "type": "string",
                                    "enum": ["tensorboard", "mlflow", "wandb", "local"],
                                    "default": "tensorboard"
                                }
                            }
                        },
                        "reproducibility": {
                            "type": "object",
                            "properties": {
                                "seed": {
                                    "type": "integer",
                                    "default": 42
                                },
                                "deterministic": {
                                    "type": "boolean",
                                    "default": True
                                }
                            }
                        }
                    },
                    "required": ["name"]
                }
            },
            "required": ["experiment"],
            "additionalProperties": False
        }
        
        cls._schemas['experiment'] = experiment_schema
        
        logger.debug("Registered experiment configuration schemas")


if __name__ == "__main__":
    ConfigSchema.initialize()
    
    print(f"AG News Text Classification (ag-news-text-classification)")
    print(f"Configuration Schema Module v{ConfigSchema.get_version()}")
    print(f"\nAvailable schemas ({len(ConfigSchema.list_schemas())}):")
    for schema_type in ConfigSchema.list_schemas():
        print(f"  - {schema_type}")
    
    print(f"\nAvailable definitions ({len(ConfigSchema.get_definitions())}):")
    for def_name in sorted(ConfigSchema.get_definitions().keys()):
        print(f"  - {def_name}")
