"""
AG News Text Classification - Configuration Package

This module provides comprehensive configuration management for the AG News Text
Classification project. It implements a hierarchical configuration system with
validation, schema enforcement, and intelligent defaults.

The configuration system supports:
- Multi-environment configurations (dev, local_prod, colab, kaggle)
- Model configurations (single models, ensembles, LLMs)
- Training configurations (standard, efficient, platform-adaptive)
- Overfitting prevention constraints and monitoring
- Platform-specific optimizations
- Deployment configurations
- Configuration templates with Jinja2
- Configuration generation from specifications
- Feature flags management
- Secrets management
- Quota configurations
- Compatibility matrix checking

Architecture:
    The configuration system follows a layered architecture:
    1. Schema Layer: Defines configuration structure and validation rules
    2. Loader Layer: Handles configuration file loading and merging
    3. Validator Layer: Ensures configuration correctness and compatibility
    4. Registry Layer: Manages configuration instances and lifecycle
    5. Smart Defaults Layer: Provides intelligent default configurations
    6. Template Layer: Jinja2-based configuration templating
    7. Generation Layer: Automated configuration generation

Design Patterns:
    - Singleton Pattern: ConfigRegistry ensures single instance
    - Factory Pattern: Configuration creation through factories
    - Strategy Pattern: Multiple validation and loading strategies
    - Template Method Pattern: Configuration loading pipeline
    - Lazy Loading Pattern: Deferred module imports for performance

Usage:
    Basic usage:
        from configs import load_config, validate_config, get_config
        
        # Load and validate configuration
        config = load_config('models/recommended/tier_1_sota/deberta_v3_xlarge_lora.yaml')
        
        # Get registered configuration
        model_config = get_config('model')
    
    Advanced usage:
        from configs import (
            load_model_config,
            load_training_config,
            load_environment_config,
            render_config_template,
            generate_config_from_spec,
            get_compatible_configs,
            load_feature_flags,
            get_platform_config
        )
        
        # Load tier-based model configuration
        model_config = load_model_config(tier='tier_1_sota', model_name='deberta_v3_xlarge_lora')
        
        # Load platform-adaptive training configuration
        training_config = load_training_config(platform='colab', mode='efficient')
        
        # Render configuration from template
        config = render_config_template('deberta_template.yaml.j2', params={'rank': 16})
        
        # Generate configuration from specification
        config = generate_config_from_spec('model_specs.yaml', model_type='deberta')
        
        # Check configuration compatibility
        compatible = get_compatible_configs('deberta_v3_xlarge', 'colab_free')

References:
    Configuration Management Patterns:
        - Martin Fowler. "Patterns of Enterprise Application Architecture". 
          Addison-Wesley, 2002.
        - Gamma et al. "Design Patterns: Elements of Reusable Object-Oriented Software". 
          Addison-Wesley, 1994.
    
    YAML Best Practices:
        - YAML Specification: https://yaml.org/spec/1.2/spec.html
        - PyYAML Documentation: https://pyyaml.org/wiki/PyYAMLDocumentation
    
    Jinja2 Templating:
        - Jinja2 Documentation: https://jinja.palletsprojects.com/
        - Jinja2 Template Designer Documentation: 
          https://jinja.palletsprojects.com/en/3.0.x/templates/
    
    Configuration Validation:
        - JSON Schema Specification: https://json-schema.org/
        - Pydantic Documentation: https://pydantic-docs.helpmanual.io/
    
    Overfitting Prevention:
        - Goodfellow, I., Bengio, Y., & Courville, A. (2016). 
          "Deep Learning". MIT Press. Chapter 7: Regularization for Deep Learning.
        - Zhang, C., Bengio, S., Hardt, M., Recht, B., & Vinyals, O. (2017). 
          "Understanding deep learning requires rethinking generalization". 
          International Conference on Learning Representations (ICLR).
    
    Parameter-Efficient Fine-Tuning:
        - Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models". 
          arXiv:2106.09685.
        - Dettmers, T., et al. (2023). "QLoRA: Efficient Finetuning of Quantized LLMs". 
          arXiv:2305.14314.

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
Project: AG News Text Classification (ag-news-text-classification)
Repository: https://github.com/VoHaiDung/ag-news-text-classification
"""

import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type, Callable, Tuple
from functools import lru_cache, wraps


# Project metadata
__version__ = "1.0.0"
__author__ = "Võ Hải Dũng"
__email__ = "vohaidung.work@gmail.com"
__license__ = "MIT"
__project__ = "AG News Text Classification (ag-news-text-classification)"
__repository__ = "https://github.com/VoHaiDung/ag-news-text-classification"


# Configuration package root directory
CONFIGS_ROOT = Path(__file__).parent
PROJECT_ROOT = CONFIGS_ROOT.parent


# Logging configuration
logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """
    Exception raised for configuration-related errors.
    
    This exception is raised when configuration loading, validation,
    or processing fails. It provides detailed error messages to help
    diagnose configuration issues.
    
    Attributes:
        message: Error description
        config_path: Path to the problematic configuration file
        details: Additional error details
    
    Examples:
        raise ConfigurationError(
            "Invalid configuration format",
            config_path="/path/to/config.yaml",
            details={"expected": "dict", "got": "list"}
        )
    """
    
    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize ConfigurationError.
        
        Args:
            message: Error description
            config_path: Path to problematic configuration file
            details: Additional error details
        """
        self.message = message
        self.config_path = config_path
        self.details = details or {}
        
        error_msg = f"Configuration Error: {message}"
        if config_path:
            error_msg += f" (Path: {config_path})"
        if details:
            error_msg += f" (Details: {details})"
        
        super().__init__(error_msg)


class ConfigRegistry:
    """
    Registry for managing configuration instances.
    
    This class implements a singleton registry pattern for managing
    configuration objects throughout the application lifecycle. It provides
    thread-safe access to configurations and supports lazy loading.
    
    The registry maintains a hierarchical structure of configurations:
    - Global configurations (environment, features)
    - Model configurations (single, ensemble)
    - Training configurations (standard, efficient, advanced)
    - Service configurations (prediction, training, data)
    - Deployment configurations (local, platform-specific)
    - Overfitting prevention configurations
    - Quota configurations
    - API configurations
    
    Thread Safety:
        The registry uses a simple dictionary without explicit locking.
        For production use in multi-threaded environments, consider adding
        threading.Lock for thread-safe operations.
    
    Attributes:
        _instance: Singleton instance
        _configs: Dictionary storing registered configurations
        _lazy_loaders: Dictionary storing lazy loading functions
        _metadata: Dictionary storing configuration metadata
    
    Examples:
        # Register configuration
        ConfigRegistry.register('my_config', config_dict)
        
        # Get configuration
        config = ConfigRegistry.get('my_config')
        
        # Register lazy loader
        ConfigRegistry.register_lazy('lazy_config', lambda: load_config())
    """
    
    _instance = None
    _configs: Dict[str, Any] = {}
    _lazy_loaders: Dict[str, Callable] = {}
    _metadata: Dict[str, Dict[str, Any]] = {}
    
    def __new__(cls):
        """
        Ensure singleton pattern for the registry.
        
        Returns:
            Singleton instance of ConfigRegistry
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(
        cls,
        name: str,
        config: Any,
        overwrite: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a configuration instance.
        
        Args:
            name: Unique identifier for the configuration
            config: Configuration object to register
            overwrite: Whether to overwrite existing configuration
            metadata: Optional metadata about the configuration
        
        Raises:
            ConfigurationError: If configuration already exists and overwrite is False
        
        Examples:
            ConfigRegistry.register('model_config', model_dict)
            ConfigRegistry.register('custom', config, metadata={'tier': 'tier_1'})
        """
        if name in cls._configs and not overwrite:
            raise ConfigurationError(
                f"Configuration '{name}' already registered. Use overwrite=True to replace.",
                details={"name": name}
            )
        
        cls._configs[name] = config
        if metadata:
            cls._metadata[name] = metadata
        
        logger.debug(f"Registered configuration: {name}")
    
    @classmethod
    def get(
        cls,
        name: str,
        default: Optional[Any] = None,
        load_if_missing: bool = True
    ) -> Any:
        """
        Retrieve a registered configuration.
        
        Args:
            name: Configuration identifier
            default: Default value if configuration not found
            load_if_missing: Whether to attempt lazy loading if not found
        
        Returns:
            Configuration object or default value
        
        Raises:
            ConfigurationError: If configuration not found and no default provided
        
        Examples:
            config = ConfigRegistry.get('model_config')
            config = ConfigRegistry.get('optional', default={})
        """
        if name in cls._configs:
            return cls._configs[name]
        
        if load_if_missing and name in cls._lazy_loaders:
            config = cls._lazy_loaders[name]()
            cls.register(name, config)
            return config
        
        if default is not None:
            return default
        
        raise ConfigurationError(
            f"Configuration '{name}' not found in registry",
            details={"available_configs": list(cls._configs.keys())}
        )
    
    @classmethod
    def register_lazy(
        cls,
        name: str,
        loader: Callable[[], Any]
    ) -> None:
        """
        Register a lazy-loading function for a configuration.
        
        Lazy loading defers configuration loading until it is actually needed,
        improving startup time and memory usage.
        
        Args:
            name: Configuration identifier
            loader: Function that returns the configuration when called
        
        Examples:
            ConfigRegistry.register_lazy('heavy_config', lambda: load_heavy_config())
        """
        cls._lazy_loaders[name] = loader
        logger.debug(f"Registered lazy loader for: {name}")
    
    @classmethod
    def clear(cls) -> None:
        """
        Clear all registered configurations.
        
        This method removes all configurations, lazy loaders, and metadata
        from the registry. Useful for testing or resetting state.
        """
        cls._configs.clear()
        cls._lazy_loaders.clear()
        cls._metadata.clear()
        logger.info("Cleared configuration registry")
    
    @classmethod
    def list_registered(cls) -> List[str]:
        """
        List all registered configuration names.
        
        Returns:
            List of configuration identifiers
        
        Examples:
            configs = ConfigRegistry.list_registered()
            print(f"Available configs: {configs}")
        """
        return list(cls._configs.keys())
    
    @classmethod
    def has(cls, name: str) -> bool:
        """
        Check if a configuration is registered.
        
        Args:
            name: Configuration identifier
        
        Returns:
            True if configuration exists, False otherwise
        
        Examples:
            if ConfigRegistry.has('model_config'):
                config = ConfigRegistry.get('model_config')
        """
        return name in cls._configs or name in cls._lazy_loaders
    
    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered configuration.
        
        Args:
            name: Configuration identifier
        
        Returns:
            Configuration metadata dictionary
        
        Examples:
            metadata = ConfigRegistry.get_metadata('model_config')
            tier = metadata.get('tier')
        """
        return cls._metadata.get(name, {})


def _safe_import(module_path: str, class_name: str):
    """
    Safely import a module and class with error handling.
    
    This function provides safe importing with detailed error messages
    for debugging import issues.
    
    Args:
        module_path: Full path to module (e.g., 'configs.config_loader')
        class_name: Name of class to import
    
    Returns:
        Imported class
    
    Raises:
        ConfigurationError: If import fails
    """
    try:
        from importlib import import_module
        module = import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ConfigurationError(
            f"Failed to import {class_name} from {module_path}",
            details={"error": str(e), "module": module_path, "class": class_name}
        )
    except AttributeError as e:
        raise ConfigurationError(
            f"Class {class_name} not found in {module_path}",
            details={"error": str(e), "module": module_path, "class": class_name}
        )


@lru_cache(maxsize=1)
def get_config_loader():
    """
    Get ConfigLoader class with caching.
    
    Uses lazy loading and caching to avoid circular dependencies
    and improve performance.
    
    Returns:
        ConfigLoader class
    
    Examples:
        ConfigLoader = get_config_loader()
        config = ConfigLoader.load('config.yaml')
    """
    return _safe_import('configs.config_loader', 'ConfigLoader')


@lru_cache(maxsize=1)
def get_config_validator():
    """
    Get ConfigValidator class with caching.
    
    Uses lazy loading and caching to avoid circular dependencies
    and improve performance.
    
    Returns:
        ConfigValidator class
    
    Examples:
        ConfigValidator = get_config_validator()
        ConfigValidator.validate(config)
    """
    return _safe_import('configs.config_validator', 'ConfigValidator')


@lru_cache(maxsize=1)
def get_config_schema():
    """
    Get ConfigSchema class with caching.
    
    Uses lazy loading and caching to avoid circular dependencies
    and improve performance.
    
    Returns:
        ConfigSchema class
    
    Examples:
        ConfigSchema = get_config_schema()
        schema = ConfigSchema.get_schema('model')
    """
    return _safe_import('configs.config_schema', 'ConfigSchema')


@lru_cache(maxsize=1)
def get_constants():
    """
    Get constants module with caching.
    
    Uses lazy loading and caching to avoid circular dependencies
    and improve performance.
    
    Returns:
        constants module
    
    Examples:
        constants = get_constants()
        model_types = constants.MODEL_TYPES
    """
    try:
        from importlib import import_module
        return import_module('configs.constants')
    except ImportError as e:
        raise ConfigurationError(
            "Failed to import constants module",
            details={"error": str(e)}
        )


@lru_cache(maxsize=1)
def get_smart_defaults():
    """
    Get SmartDefaults class with caching.
    
    Uses lazy loading and caching to avoid circular dependencies
    and improve performance.
    
    Returns:
        SmartDefaults class
    
    Examples:
        SmartDefaults = get_smart_defaults()
        defaults = SmartDefaults.get_default('model')
    """
    return _safe_import('configs.smart_defaults', 'SmartDefaults')


def load_config(
    config_path: Union[str, Path],
    validate: bool = True,
    merge_defaults: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Load and optionally validate a configuration file.
    
    This is a high-level convenience function that combines loading,
    validation, and default merging in a single call.
    
    Args:
        config_path: Path to configuration file (relative to configs/ or absolute)
        validate: Whether to validate the configuration
        merge_defaults: Whether to merge with smart defaults
        **kwargs: Additional arguments passed to ConfigLoader
    
    Returns:
        Loaded and validated configuration dictionary
    
    Raises:
        ConfigurationError: If loading or validation fails
    
    Examples:
        # Load with validation
        config = load_config('models/recommended/tier_1_sota/deberta_v3_xlarge_lora.yaml')
        
        # Load without validation
        config = load_config('custom_config.yaml', validate=False)
        
        # Load with custom parameters
        config = load_config('config.yaml', merge_defaults=False, encoding='utf-8')
    """
    ConfigLoader = get_config_loader()
    
    # Resolve path relative to configs directory if not absolute
    config_path_obj = Path(config_path)
    if not config_path_obj.is_absolute():
        config_path_obj = CONFIGS_ROOT / config_path
    
    # Load configuration
    config = ConfigLoader.load(str(config_path_obj), **kwargs)
    
    # Merge with smart defaults if requested
    if merge_defaults:
        SmartDefaults = get_smart_defaults()
        config = SmartDefaults.merge(config)
    
    # Validate configuration if requested
    if validate:
        ConfigValidator = get_config_validator()
        ConfigValidator.validate(config)
    
    return config


def validate_config(
    config: Dict[str, Any],
    schema_type: Optional[str] = None,
    strict: bool = True
) -> bool:
    """
    Validate a configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        schema_type: Optional schema type (e.g., 'model', 'training')
        strict: Whether to use strict validation
    
    Returns:
        True if validation passes
    
    Raises:
        ConfigurationError: If validation fails
    
    Examples:
        # Validate with auto-detected schema
        validate_config(my_config)
        
        # Validate with specific schema
        validate_config(my_config, schema_type='model')
        
        # Validate with lenient mode
        validate_config(my_config, strict=False)
    """
    ConfigValidator = get_config_validator()
    return ConfigValidator.validate(config, schema_type=schema_type, strict=strict)


def get_config(
    name: str,
    default: Optional[Any] = None
) -> Any:
    """
    Get a registered configuration from the registry.
    
    Args:
        name: Configuration identifier
        default: Default value if not found
    
    Returns:
        Configuration object
    
    Examples:
        # Get registered config
        model_config = get_config('model')
        
        # Get with default
        config = get_config('optional_config', default={})
    """
    return ConfigRegistry.get(name, default=default)


def register_config(
    name: str,
    config: Any,
    overwrite: bool = False,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Register a configuration in the global registry.
    
    Args:
        name: Unique configuration identifier
        config: Configuration object
        overwrite: Whether to overwrite existing configuration
        metadata: Optional metadata about the configuration
    
    Examples:
        # Register simple config
        register_config('my_custom_model', custom_config)
        
        # Register with metadata
        register_config('model', config, metadata={'tier': 'tier_1_sota'})
        
        # Overwrite existing
        register_config('model', new_config, overwrite=True)
    """
    ConfigRegistry.register(name, config, overwrite=overwrite, metadata=metadata)


def get_config_path(
    relative_path: str,
    must_exist: bool = False
) -> Path:
    """
    Get absolute path to a configuration file.
    
    Args:
        relative_path: Path relative to configs/ directory
        must_exist: Whether to raise error if path does not exist
    
    Returns:
        Absolute path to configuration file
    
    Raises:
        ConfigurationError: If must_exist is True and path does not exist
    
    Examples:
        # Get path
        path = get_config_path('models/recommended/quick_start.yaml')
        
        # Ensure path exists
        path = get_config_path('config.yaml', must_exist=True)
    """
    abs_path = CONFIGS_ROOT / relative_path
    
    if must_exist and not abs_path.exists():
        raise ConfigurationError(
            f"Configuration path does not exist: {relative_path}",
            config_path=str(abs_path)
        )
    
    return abs_path


def list_configs(
    config_dir: str = "models",
    pattern: str = "*.yaml",
    recursive: bool = True
) -> List[Path]:
    """
    List all configuration files in a directory.
    
    Args:
        config_dir: Directory relative to configs/ (default: 'models')
        pattern: File pattern to match (default: '*.yaml')
        recursive: Whether to search recursively
    
    Returns:
        List of configuration file paths
    
    Examples:
        # List all model configs
        model_configs = list_configs('models/recommended')
        
        # List training configs
        training_configs = list_configs('training', pattern='*.yaml')
        
        # List non-recursively
        configs = list_configs('models', recursive=False)
    """
    search_path = CONFIGS_ROOT / config_dir
    
    if not search_path.exists():
        logger.warning(f"Configuration directory not found: {config_dir}")
        return []
    
    if recursive:
        configs = list(search_path.rglob(pattern))
    else:
        configs = list(search_path.glob(pattern))
    
    return configs


def get_default_config(
    config_type: str,
    tier: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get recommended default configuration for a given type.
    
    This function provides intelligent defaults based on configuration type
    and optional tier specification.
    
    Args:
        config_type: Type of configuration ('model', 'training', 'data', etc.)
        tier: Optional tier specification for models ('tier_1_sota', 'tier_2_llm', etc.)
    
    Returns:
        Default configuration dictionary
    
    Examples:
        # Get default model config
        default_model = get_default_config('model')
        
        # Get tier-specific default
        sota_model = get_default_config('model', tier='tier_1_sota')
        
        # Get training default
        training = get_default_config('training')
    """
    SmartDefaults = get_smart_defaults()
    return SmartDefaults.get_default(config_type, tier=tier)


def load_model_config(
    tier: Optional[str] = None,
    model_name: Optional[str] = None,
    model_type: str = "single",
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load model configuration with tier-based or specific model selection.
    
    Args:
        tier: Tier specification ('tier_1_sota', 'tier_2_llm', 'tier_3_ensemble', etc.)
        model_name: Specific model name (e.g., 'deberta_v3_xlarge_lora')
        model_type: Type of model ('single', 'ensemble')
        validate: Whether to validate configuration
    
    Returns:
        Model configuration dictionary
    
    Raises:
        ConfigurationError: If configuration not found or invalid
    
    Examples:
        # Load tier-based default
        config = load_model_config(tier='tier_1_sota')
        
        # Load specific model
        config = load_model_config(model_name='deberta_v3_xlarge_lora')
        
        # Load ensemble configuration
        config = load_model_config(tier='tier_3_ensemble', model_type='ensemble')
    """
    if model_name:
        if model_type == "ensemble":
            config_path = f"models/ensemble/{model_name}.yaml"
        else:
            if tier:
                config_path = f"models/recommended/{tier}/{model_name}.yaml"
            else:
                config_path = f"models/single/{model_name}.yaml"
    elif tier:
        config_path = f"models/recommended/{tier}/quick_start.yaml"
    else:
        config_path = "models/recommended/quick_start.yaml"
    
    return load_config(config_path, validate=validate)


def load_training_config(
    platform: Optional[str] = None,
    mode: str = "standard",
    efficient_method: Optional[str] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load training configuration with platform-adaptive selection.
    
    Args:
        platform: Platform specification ('colab', 'kaggle', 'local')
        mode: Training mode ('standard', 'efficient', 'advanced', 'safe')
        efficient_method: Efficient training method ('lora', 'qlora', 'adapter', etc.)
        validate: Whether to validate configuration
    
    Returns:
        Training configuration dictionary
    
    Examples:
        # Load platform-adaptive configuration
        config = load_training_config(platform='colab', mode='efficient')
        
        # Load specific efficient method
        config = load_training_config(mode='efficient', efficient_method='lora')
        
        # Load safe training configuration
        config = load_training_config(mode='safe')
    """
    if platform:
        config_path = f"training/platform_adaptive/{platform}_{mode}_training.yaml"
    elif efficient_method:
        config_path = f"training/efficient/{efficient_method}/{efficient_method}_config.yaml"
    elif mode == "safe":
        config_path = "training/safe/xlarge_safe_training.yaml"
    else:
        config_path = f"training/{mode}/base_training.yaml"
    
    return load_config(config_path, validate=validate)


def load_environment_config(
    environment: str = "dev",
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load environment-specific configuration.
    
    Args:
        environment: Environment name ('dev', 'local_prod', 'colab', 'kaggle')
        validate: Whether to validate configuration
    
    Returns:
        Environment configuration dictionary
    
    Examples:
        # Load development environment
        config = load_environment_config('dev')
        
        # Load Colab environment
        config = load_environment_config('colab')
    """
    config_path = f"environments/{environment}.yaml"
    return load_config(config_path, validate=validate)


def load_service_config(
    service_name: str,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load service configuration.
    
    Args:
        service_name: Service name ('prediction_service', 'training_service', etc.)
        validate: Whether to validate configuration
    
    Returns:
        Service configuration dictionary
    
    Examples:
        # Load prediction service config
        config = load_service_config('prediction_service')
        
        # Load training service config
        config = load_service_config('training_service')
    """
    config_path = f"services/{service_name}.yaml"
    return load_config(config_path, validate=validate)


def load_api_config(
    api_type: str = "rest",
    config_name: Optional[str] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load API configuration.
    
    Args:
        api_type: API type ('rest', 'auth', 'rate_limit')
        config_name: Specific configuration name
        validate: Whether to validate configuration
    
    Returns:
        API configuration dictionary
    
    Examples:
        # Load REST API config
        config = load_api_config('rest')
        
        # Load specific config
        config = load_api_config('rest', 'rest_config')
    """
    if config_name:
        config_path = f"api/{config_name}.yaml"
    else:
        config_path = f"api/{api_type}_config.yaml"
    
    return load_config(config_path, validate=validate)


def load_overfitting_prevention_config(
    config_type: str,
    specific_config: Optional[str] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load overfitting prevention configuration.
    
    Args:
        config_type: Configuration type ('constraints', 'monitoring', 'validation', 
                     'recommendations', 'safe_defaults')
        specific_config: Specific configuration within type
        validate: Whether to validate configuration
    
    Returns:
        Overfitting prevention configuration dictionary
    
    Examples:
        # Load constraints for xlarge models
        config = load_overfitting_prevention_config('constraints', 'xlarge_constraints')
        
        # Load monitoring thresholds
        config = load_overfitting_prevention_config('monitoring', 'thresholds')
    """
    if specific_config:
        config_path = f"overfitting_prevention/{config_type}/{specific_config}.yaml"
    else:
        config_path = f"overfitting_prevention/{config_type}/default.yaml"
    
    return load_config(config_path, validate=validate)


def load_data_config(
    config_type: str,
    specific_config: Optional[str] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load data configuration.
    
    Args:
        config_type: Configuration type ('preprocessing', 'augmentation', 'validation', etc.)
        specific_config: Specific configuration within type
        validate: Whether to validate configuration
    
    Returns:
        Data configuration dictionary
    
    Examples:
        # Load preprocessing configuration
        config = load_data_config('preprocessing', 'standard')
        
        # Load augmentation configuration
        config = load_data_config('augmentation', 'safe_augmentation')
    """
    if specific_config:
        config_path = f"data/{config_type}/{specific_config}.yaml"
    else:
        config_path = f"data/{config_type}/default.yaml"
    
    return load_config(config_path, validate=validate)


def load_deployment_config(
    deployment_type: str = "local",
    specific_config: Optional[str] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load deployment configuration.
    
    Args:
        deployment_type: Deployment type ('local', 'free_tier', 'platform_profiles')
        specific_config: Specific configuration within type
        validate: Whether to validate configuration
    
    Returns:
        Deployment configuration dictionary
    
    Examples:
        # Load local deployment configuration
        config = load_deployment_config('local', 'docker_local')
        
        # Load free tier configuration
        config = load_deployment_config('free_tier', 'colab_deployment')
    """
    if specific_config:
        config_path = f"deployment/{deployment_type}/{specific_config}.yaml"
    else:
        config_path = f"deployment/{deployment_type}/default.yaml"
    
    return load_config(config_path, validate=validate)


def load_quota_config(
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load quota configuration.
    
    Args:
        validate: Whether to validate configuration
    
    Returns:
        Quota configuration dictionary
    
    Examples:
        # Load quota limits
        config = load_quota_config()
    """
    config_path = "quotas/quota_limits.yaml"
    return load_config(config_path, validate=validate)


def load_feature_flags(
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load feature flags configuration.
    
    Args:
        validate: Whether to validate configuration
    
    Returns:
        Feature flags configuration dictionary
    
    Examples:
        # Load feature flags
        flags = load_feature_flags()
        if flags.get('enable_llm_models'):
            pass
    """
    config_path = "features/feature_flags.yaml"
    return load_config(config_path, validate=validate)


def load_experiment_config(
    experiment_type: str,
    specific_experiment: Optional[str] = None,
    validate: bool = True
) -> Dict[str, Any]:
    """
    Load experiment configuration.
    
    Args:
        experiment_type: Experiment type ('baselines', 'ablations', 'hyperparameter_search', 
                        'sota_experiments', 'reproducibility')
        specific_experiment: Specific experiment configuration
        validate: Whether to validate configuration
    
    Returns:
        Experiment configuration dictionary
    
    Examples:
        # Load baseline experiment
        config = load_experiment_config('baselines', 'classical_ml')
        
        # Load SOTA experiment
        config = load_experiment_config('sota_experiments', 'phase1_xlarge_models')
    """
    if specific_experiment:
        config_path = f"experiments/{experiment_type}/{specific_experiment}.yaml"
    else:
        config_path = f"experiments/{experiment_type}/default.yaml"
    
    return load_config(config_path, validate=validate)


def render_config_template(
    template_name: str,
    params: Dict[str, Any],
    validate: bool = True
) -> Dict[str, Any]:
    """
    Render a configuration template with parameters.
    
    This function uses Jinja2 templating to generate configurations
    from templates with variable substitution.
    
    Args:
        template_name: Template file name (e.g., 'deberta_template.yaml.j2')
        params: Parameters for template rendering
        validate: Whether to validate rendered configuration
    
    Returns:
        Rendered configuration dictionary
    
    Raises:
        ConfigurationError: If template rendering fails
    
    Examples:
        # Render DeBERTa config
        config = render_config_template(
            'deberta_template.yaml.j2',
            params={'rank': 16, 'alpha': 32, 'dropout': 0.1}
        )
    """
    try:
        from jinja2 import Environment, FileSystemLoader
        import yaml
    except ImportError as e:
        raise ConfigurationError(
            f"Required package not installed: {str(e)}. "
            "Install with: pip install jinja2 pyyaml"
        )
    
    template_dir = CONFIGS_ROOT / "templates"
    
    if not template_dir.exists():
        raise ConfigurationError(
            "Templates directory not found",
            config_path=str(template_dir)
        )
    
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    try:
        template = env.get_template(template_name)
        rendered = template.render(**params)
        config = yaml.safe_load(rendered)
        
        if validate:
            validate_config(config)
        
        return config
    
    except Exception as e:
        raise ConfigurationError(
            f"Failed to render template '{template_name}': {str(e)}",
            config_path=str(template_dir / template_name),
            details={"params": params, "error": str(e)}
        )


def generate_config_from_spec(
    spec_name: str,
    config_type: str,
    validate: bool = True,
    **overrides
) -> Dict[str, Any]:
    """
    Generate configuration from specification.
    
    This function reads a specification file and generates a complete
    configuration based on the specification and overrides.
    
    Args:
        spec_name: Specification file name (e.g., 'model_specs.yaml')
        config_type: Type of configuration to generate
        validate: Whether to validate generated configuration
        **overrides: Override parameters for generation
    
    Returns:
        Generated configuration dictionary
    
    Examples:
        # Generate model config
        config = generate_config_from_spec(
            'model_specs.yaml',
            config_type='deberta_large',
            lora_rank=16
        )
    """
    import yaml
    
    spec_path = CONFIGS_ROOT / "generation" / spec_name
    
    if not spec_path.exists():
        raise ConfigurationError(
            f"Specification file not found: {spec_name}",
            config_path=str(spec_path)
        )
    
    with open(spec_path, 'r', encoding='utf-8') as f:
        specs = yaml.safe_load(f)
    
    if config_type not in specs:
        raise ConfigurationError(
            f"Configuration type '{config_type}' not found in specification",
            config_path=str(spec_path),
            details={"available_types": list(specs.keys())}
        )
    
    config = specs[config_type].copy()
    config.update(overrides)
    
    if validate:
        validate_config(config)
    
    return config


def load_compatibility_matrix(
    validate: bool = False
) -> Dict[str, Any]:
    """
    Load compatibility matrix for models, platforms, and configurations.
    
    The compatibility matrix defines which models are compatible with
    which platforms, resource constraints, and configurations.
    
    Args:
        validate: Whether to validate matrix
    
    Returns:
        Compatibility matrix dictionary
    
    Examples:
        # Load matrix
        matrix = load_compatibility_matrix()
        compatible = matrix['models']['deberta_v3_xlarge']['platforms']['colab_free']
    """
    import yaml
    
    matrix_path = CONFIGS_ROOT / "compatibility_matrix.yaml"
    
    if not matrix_path.exists():
        logger.warning("Compatibility matrix not found, returning empty dict")
        return {}
    
    with open(matrix_path, 'r', encoding='utf-8') as f:
        matrix = yaml.safe_load(f)
    
    return matrix


def get_compatible_configs(
    model_name: str,
    platform: str
) -> Dict[str, Any]:
    """
    Get compatible configurations for a model on a specific platform.
    
    Args:
        model_name: Model name
        platform: Platform name ('colab_free', 'colab_pro', 'kaggle', 'local')
    
    Returns:
        Dictionary of compatible configurations
    
    Examples:
        # Check compatibility
        compatible = get_compatible_configs('deberta_v3_xlarge', 'colab_free')
    """
    matrix = load_compatibility_matrix()
    
    if not matrix:
        logger.warning("Empty compatibility matrix, returning default config")
        return {}
    
    try:
        model_compat = matrix.get('models', {}).get(model_name, {})
        platform_compat = model_compat.get('platforms', {}).get(platform, {})
        return platform_compat
    except Exception as e:
        logger.error(f"Error getting compatible configs: {e}")
        return {}


def get_platform_config(
    platform: str,
    config_category: str = "training",
    validate: bool = True
) -> Dict[str, Any]:
    """
    Get platform-specific configuration.
    
    Args:
        platform: Platform name ('colab', 'kaggle', 'local')
        config_category: Configuration category ('training', 'deployment', 'environment')
        validate: Whether to validate configuration
    
    Returns:
        Platform-specific configuration dictionary
    
    Examples:
        # Get Colab training config
        config = get_platform_config('colab', 'training')
    """
    if config_category == "training":
        return load_training_config(platform=platform, validate=validate)
    elif config_category == "deployment":
        return load_deployment_config('platform_profiles', f"{platform}_profile", validate=validate)
    elif config_category == "environment":
        return load_environment_config(platform, validate=validate)
    else:
        raise ConfigurationError(
            f"Unknown config category: {config_category}",
            details={"supported_categories": ["training", "deployment", "environment"]}
        )


def discover_model_configs(
    tier: Optional[str] = None,
    model_family: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Discover available model configurations.
    
    Args:
        tier: Filter by tier ('tier_1_sota', 'tier_2_llm', etc.)
        model_family: Filter by model family ('deberta', 'roberta', 'llama', etc.)
    
    Returns:
        List of model configuration metadata
    
    Examples:
        # Discover SOTA models
        models = discover_model_configs(tier='tier_1_sota')
        
        # Discover DeBERTa models
        models = discover_model_configs(model_family='deberta')
    """
    discovered = []
    
    if tier:
        search_path = CONFIGS_ROOT / "models" / "recommended" / tier
        if search_path.exists():
            for config_file in search_path.glob("*.yaml"):
                if config_file.name.startswith("README"):
                    continue
                
                discovered.append({
                    'name': config_file.stem,
                    'path': str(config_file),
                    'tier': tier,
                    'type': 'recommended'
                })
    
    if model_family:
        search_path = CONFIGS_ROOT / "models" / "single" / "transformers" / model_family
        if search_path.exists():
            for config_file in search_path.glob("*.yaml"):
                discovered.append({
                    'name': config_file.stem,
                    'path': str(config_file),
                    'family': model_family,
                    'type': 'single'
                })
    
    return discovered


def discover_training_configs(
    mode: Optional[str] = None,
    platform: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Discover available training configurations.
    
    Args:
        mode: Filter by mode ('standard', 'efficient', 'advanced')
        platform: Filter by platform ('colab', 'kaggle', 'local')
    
    Returns:
        List of training configuration metadata
    
    Examples:
        # Discover efficient training configs
        configs = discover_training_configs(mode='efficient')
        
        # Discover Colab configs
        configs = discover_training_configs(platform='colab')
    """
    discovered = []
    
    if platform:
        search_path = CONFIGS_ROOT / "training" / "platform_adaptive"
        pattern = f"{platform}_*.yaml"
    elif mode:
        search_path = CONFIGS_ROOT / "training" / mode
        pattern = "*.yaml"
    else:
        search_path = CONFIGS_ROOT / "training"
        pattern = "**/*.yaml"
    
    if search_path.exists():
        for config_file in search_path.glob(pattern):
            if config_file.name.startswith("README"):
                continue
            
            discovered.append({
                'name': config_file.stem,
                'path': str(config_file),
                'mode': mode,
                'platform': platform
            })
    
    return discovered


def _initialize_config_paths():
    """
    Initialize configuration paths and lazy loaders in the registry.
    
    This function registers commonly used configuration directories and
    sets up lazy loaders for frequently accessed configurations.
    """
    ConfigRegistry.register('configs_root', CONFIGS_ROOT)
    ConfigRegistry.register('project_root', PROJECT_ROOT)
    
    config_dirs = [
        'models', 'training', 'data', 'overfitting_prevention',
        'deployment', 'api', 'services', 'environments',
        'features', 'secrets', 'templates', 'generation',
        'quotas', 'experiments'
    ]
    
    for dir_name in config_dirs:
        ConfigRegistry.register_lazy(
            f'{dir_name}_configs_dir',
            lambda d=dir_name: CONFIGS_ROOT / d
        )
    
    logger.debug("Initialized configuration paths in registry")


_initialize_config_paths()


__all__ = [
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    '__project__',
    '__repository__',
    'ConfigurationError',
    'ConfigRegistry',
    'get_config_loader',
    'get_config_validator',
    'get_config_schema',
    'get_constants',
    'get_smart_defaults',
    'load_config',
    'validate_config',
    'get_config',
    'register_config',
    'get_config_path',
    'list_configs',
    'get_default_config',
    'load_model_config',
    'load_training_config',
    'load_environment_config',
    'load_service_config',
    'load_api_config',
    'load_overfitting_prevention_config',
    'load_data_config',
    'load_deployment_config',
    'load_quota_config',
    'load_feature_flags',
    'load_experiment_config',
    'render_config_template',
    'generate_config_from_spec',
    'load_compatibility_matrix',
    'get_compatible_configs',
    'get_platform_config',
    'discover_model_configs',
    'discover_training_configs',
    'CONFIGS_ROOT',
    'PROJECT_ROOT',
]


logger.info(
    f"Initialized {__project__} configuration package v{__version__} "
    f"(Author: {__author__})"
)
logger.debug(f"Configuration root: {CONFIGS_ROOT}")
logger.debug(f"Project root: {PROJECT_ROOT}")
