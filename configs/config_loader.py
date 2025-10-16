"""
AG News Text Classification - Configuration Loader Module

This module provides comprehensive configuration loading functionality for the
AG News Text Classification (ag-news-text-classification) project. It implements
advanced configuration management patterns including hierarchical loading,
template rendering, environment-based substitution, and intelligent merging.

The ConfigLoader supports:
- YAML configuration file loading with validation
- Jinja2 template rendering for dynamic configurations
- Environment variable substitution and expansion
- Hierarchical configuration inheritance and merging
- Platform-specific configuration loading
- Configuration caching and hot reloading
- Multiple configuration sources (files, environment, defaults)
- Type-safe configuration access with validation

Architecture:
    The configuration loader implements several design patterns:
    
    1. Strategy Pattern: Different loading strategies for various config types
    2. Template Method Pattern: Standardized loading pipeline with customization points
    3. Factory Pattern: Dynamic configuration object creation
    4. Singleton Pattern: Cached configuration instances
    5. Composite Pattern: Hierarchical configuration composition
    
    Loading Pipeline:
    1. Source Resolution: Determine configuration file path
    2. File Loading: Read and parse YAML/JSON content
    3. Template Rendering: Process Jinja2 templates if applicable
    4. Variable Substitution: Replace environment variables
    5. Inheritance Resolution: Merge with parent configurations
    6. Validation: Validate against schema if available
    7. Defaults Merging: Apply smart defaults
    8. Caching: Store in cache for future access

Configuration Hierarchy:
    Configurations can inherit from parent configurations using special keys:
    
    - _inherit_from: Path to parent configuration file
    - _merge_strategy: How to merge with parent (replace, merge_deep, merge_shallow)
    - _template: Path to template file for rendering
    - _variables: Variables for template rendering
    
    Example:
        # configs/models/custom/my_model.yaml
        _inherit_from: ../recommended/tier_1_sota/deberta_v3_xlarge_lora.yaml
        _merge_strategy: merge_deep
        
        model:
          lora:
            rank: 16  # Override parent value

Environment Variable Substitution:
    The loader supports multiple syntax forms for environment variables:
    
    - ${VAR_NAME}: Required variable, raises error if not found
    - ${VAR_NAME:-default}: Optional variable with default value
    - ${VAR_NAME:?error_message}: Required with custom error message
    - $VAR_NAME: Simple substitution (bash-style)
    
    Example:
        model:
          cache_dir: ${CACHE_DIR:-/tmp/model_cache}
          api_key: ${API_KEY:?API key is required}

Template Rendering:
    Jinja2 templates enable dynamic configuration generation:
    
    Example:
        # configs/templates/model_template.yaml.j2
        model:
          name: {{ model_name }}
          lora:
            rank: {{ lora_rank | default(8) }}
        
        # Usage:
        config = ConfigLoader.load_template(
            'model_template.yaml.j2',
            model_name='deberta-v3-large',
            lora_rank=16
        )

Caching Strategy:
    The loader implements intelligent caching to improve performance:
    
    - File-based cache with modification time checking
    - In-memory cache for frequently accessed configs
    - Cache invalidation on file changes
    - Optional cache bypass for development

Error Handling:
    Comprehensive error handling with informative messages:
    
    - FileNotFoundError: Configuration file not found
    - YAMLError: Invalid YAML syntax
    - TemplateError: Jinja2 template rendering error
    - ValidationError: Configuration validation failure
    - InheritanceError: Circular or invalid inheritance

Performance Considerations:
    - Lazy loading: Load configurations only when needed
    - Caching: Reduce file I/O and parsing overhead
    - Efficient merging: Optimized deep merge algorithm
    - Minimal dependencies: Only essential libraries

Thread Safety:
    The ConfigLoader is thread-safe for read operations. For write operations
    (cache invalidation, hot reloading), use proper synchronization mechanisms.

References:
    Configuration Management Patterns:
        - Fowler, M. (2004). "Inversion of Control Containers and the Dependency 
          Injection pattern". martinfowler.com
        - Gamma, E. et al. (1994). "Design Patterns: Elements of Reusable 
          Object-Oriented Software". Addison-Wesley.
    
    YAML Specification:
        - Ben-Kiki, O., Evans, C., & döt Net, I. (2009). "YAML Ain't Markup Language 
          (YAML) Version 1.2". yaml.org
    
    Jinja2 Templating:
        - Ronacher, A. "Jinja2 Documentation". jinja.palletsprojects.com
    
    Configuration Best Practices:
        - Wiggins, A. "The Twelve-Factor App". 12factor.net

Usage:
    Basic loading:
        from configs.config_loader import ConfigLoader
        
        config = ConfigLoader.load('models/recommended/tier_1_sota/deberta_v3_xlarge_lora.yaml')
    
    Template rendering:
        config = ConfigLoader.load_template(
            'deberta_template.yaml.j2',
            model_size='xlarge',
            lora_rank=16
        )
    
    Platform-specific loading:
        config = ConfigLoader.load_for_platform(
            'training/platform_adaptive',
            platform='colab'
        )
    
    With validation:
        config = ConfigLoader.load(
            'models/my_model.yaml',
            validate=True,
            schema_type='model'
        )

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
Project: AG News Text Classification (ag-news-text-classification)
Repository: https://github.com/VoHaiDung/ag-news-text-classification
"""

import os
import re
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from datetime import datetime
from functools import lru_cache
from copy import deepcopy

import yaml


# Module metadata
__version__ = "1.0.0"
__author__ = "Võ Hải Dũng"
__email__ = "vohaidung.work@gmail.com"
__license__ = "MIT"


# Configure logging
logger = logging.getLogger(__name__)


# Configuration root directory
CONFIGS_ROOT = Path(__file__).parent
PROJECT_ROOT = CONFIGS_ROOT.parent


class ConfigurationLoaderError(Exception):
    """
    Base exception for configuration loader errors.
    
    This exception is raised when configuration loading fails at any stage
    of the loading pipeline.
    
    Attributes:
        message: Error description
        config_path: Path to the configuration file that caused the error
        stage: Stage in the loading pipeline where error occurred
        original_error: Original exception if wrapped
    """
    
    def __init__(
        self,
        message: str,
        config_path: Optional[str] = None,
        stage: Optional[str] = None,
        original_error: Optional[Exception] = None
    ):
        """
        Initialize ConfigurationLoaderError.
        
        Args:
            message: Error description
            config_path: Path to problematic configuration file
            stage: Loading pipeline stage where error occurred
            original_error: Original exception being wrapped
        """
        self.message = message
        self.config_path = config_path
        self.stage = stage
        self.original_error = original_error
        
        error_msg = f"Configuration Loader Error: {message}"
        if config_path:
            error_msg += f" (Path: {config_path})"
        if stage:
            error_msg += f" (Stage: {stage})"
        if original_error:
            error_msg += f" (Caused by: {type(original_error).__name__}: {str(original_error)})"
        
        super().__init__(error_msg)


class ConfigLoader:
    """
    Advanced configuration loader with support for multiple formats,
    inheritance, templating, and validation.
    
    This class provides a comprehensive configuration loading system that
    supports YAML files, Jinja2 templates, environment variable substitution,
    hierarchical inheritance, and intelligent caching.
    
    Class Attributes:
        _cache: In-memory configuration cache
        _file_mtimes: File modification times for cache invalidation
        _inheritance_chain: Track inheritance to detect circular references
    
    Methods:
        load: Load configuration from file
        load_template: Load and render Jinja2 template
        load_multiple: Load multiple configurations and merge
        load_for_platform: Load platform-specific configuration
        reload: Force reload configuration from disk
        clear_cache: Clear all cached configurations
        get_cache_stats: Get cache statistics
    
    Examples:
        Load single configuration:
            config = ConfigLoader.load('models/deberta_v3_xlarge_lora.yaml')
        
        Load with validation:
            config = ConfigLoader.load('training/lora_config.yaml', validate=True)
        
        Load template:
            config = ConfigLoader.load_template('model_template.yaml.j2', rank=16)
        
        Load for platform:
            config = ConfigLoader.load_for_platform('training', platform='colab')
    """
    
    # Class-level cache for loaded configurations
    _cache: Dict[str, Dict[str, Any]] = {}
    
    # File modification times for cache invalidation
    _file_mtimes: Dict[str, float] = {}
    
    # Inheritance chain tracking to prevent circular inheritance
    _inheritance_chain: List[str] = []
    
    # Configuration loading statistics
    _stats: Dict[str, int] = {
        'loads': 0,
        'cache_hits': 0,
        'cache_misses': 0,
        'template_renders': 0,
        'validations': 0,
        'errors': 0
    }
    
    @classmethod
    def load(
        cls,
        config_path: Union[str, Path],
        validate: bool = False,
        schema_type: Optional[str] = None,
        use_cache: bool = True,
        merge_defaults: bool = False,
        resolve_inheritance: bool = True,
        substitute_env: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Load configuration from a YAML file with advanced features.
        
        This method implements the complete configuration loading pipeline
        including file resolution, parsing, inheritance resolution, variable
        substitution, validation, and caching.
        
        Args:
            config_path: Path to configuration file (relative to configs/ or absolute)
            validate: Whether to validate configuration against schema
            schema_type: Type of schema for validation (e.g., 'model', 'training')
            use_cache: Whether to use cached configuration if available
            merge_defaults: Whether to merge with smart defaults
            resolve_inheritance: Whether to resolve _inherit_from directives
            substitute_env: Whether to substitute environment variables
            **kwargs: Additional arguments passed to parser
        
        Returns:
            Loaded and processed configuration dictionary
        
        Raises:
            ConfigurationLoaderError: If loading fails at any stage
            FileNotFoundError: If configuration file not found
            yaml.YAMLError: If YAML parsing fails
        
        Examples:
            Basic loading:
                config = ConfigLoader.load('models/deberta_v3_large.yaml')
            
            With validation:
                config = ConfigLoader.load(
                    'training/lora_config.yaml',
                    validate=True,
                    schema_type='training'
                )
            
            Bypass cache:
                config = ConfigLoader.load('config.yaml', use_cache=False)
        """
        cls._stats['loads'] += 1
        
        try:
            # Stage 1: Resolve configuration file path
            resolved_path = cls._resolve_path(config_path)
            
            # Stage 2: Check cache if enabled
            if use_cache:
                cached_config = cls._get_from_cache(resolved_path)
                if cached_config is not None:
                    cls._stats['cache_hits'] += 1
                    logger.debug(f"Cache hit for configuration: {config_path}")
                    return deepcopy(cached_config)
                cls._stats['cache_misses'] += 1
            
            # Stage 3: Load and parse YAML file
            config = cls._load_yaml_file(resolved_path, **kwargs)
            
            # Stage 4: Resolve inheritance if enabled
            if resolve_inheritance and '_inherit_from' in config:
                config = cls._resolve_inheritance(config, resolved_path)
            
            # Stage 5: Substitute environment variables if enabled
            if substitute_env:
                config = cls._substitute_env_vars(config)
            
            # Stage 6: Merge with smart defaults if requested
            if merge_defaults:
                config = cls._merge_with_defaults(config)
            
            # Stage 7: Validate configuration if requested
            if validate:
                cls._validate_config(config, schema_type)
                cls._stats['validations'] += 1
            
            # Stage 8: Cache the configuration
            if use_cache:
                cls._add_to_cache(resolved_path, config)
            
            logger.info(f"Successfully loaded configuration: {config_path}")
            return deepcopy(config)
        
        except Exception as e:
            cls._stats['errors'] += 1
            logger.error(f"Failed to load configuration {config_path}: {e}")
            
            if isinstance(e, ConfigurationLoaderError):
                raise
            else:
                raise ConfigurationLoaderError(
                    f"Failed to load configuration: {str(e)}",
                    config_path=str(config_path),
                    original_error=e
                )
    
    @classmethod
    def load_template(
        cls,
        template_name: str,
        validate: bool = False,
        schema_type: Optional[str] = None,
        **template_vars
    ) -> Dict[str, Any]:
        """
        Load and render a Jinja2 configuration template.
        
        This method loads a Jinja2 template file, renders it with provided
        variables, and returns the resulting configuration dictionary.
        
        Args:
            template_name: Name of template file in configs/templates/
            validate: Whether to validate rendered configuration
            schema_type: Type of schema for validation
            **template_vars: Variables to pass to template renderer
        
        Returns:
            Rendered configuration dictionary
        
        Raises:
            ConfigurationLoaderError: If template loading or rendering fails
            FileNotFoundError: If template file not found
        
        Examples:
            Render model template:
                config = ConfigLoader.load_template(
                    'deberta_template.yaml.j2',
                    model_name='deberta-v3-large',
                    lora_rank=16,
                    lora_alpha=32
                )
            
            Render with validation:
                config = ConfigLoader.load_template(
                    'training_template.yaml.j2',
                    validate=True,
                    schema_type='training',
                    learning_rate=2e-5,
                    batch_size=8
                )
        """
        cls._stats['template_renders'] += 1
        
        try:
            # Import Jinja2 (lazy import to avoid hard dependency)
            try:
                from jinja2 import Environment, FileSystemLoader, TemplateError
            except ImportError:
                raise ConfigurationLoaderError(
                    "Jinja2 is required for template rendering. "
                    "Install with: pip install jinja2",
                    stage="template_import"
                )
            
            # Setup Jinja2 environment
            template_dir = CONFIGS_ROOT / "templates"
            
            if not template_dir.exists():
                raise ConfigurationLoaderError(
                    f"Templates directory not found: {template_dir}",
                    stage="template_directory"
                )
            
            env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True
            )
            
            # Add custom filters if needed
            env.filters['default'] = lambda value, default='': value if value is not None else default
            
            # Load and render template
            try:
                template = env.get_template(template_name)
            except Exception as e:
                raise ConfigurationLoaderError(
                    f"Failed to load template '{template_name}': {str(e)}",
                    config_path=str(template_dir / template_name),
                    stage="template_load",
                    original_error=e
                )
            
            try:
                rendered = template.render(**template_vars)
            except TemplateError as e:
                raise ConfigurationLoaderError(
                    f"Failed to render template '{template_name}': {str(e)}",
                    config_path=str(template_dir / template_name),
                    stage="template_render",
                    original_error=e
                )
            
            # Parse rendered YAML
            try:
                config = yaml.safe_load(rendered)
            except yaml.YAMLError as e:
                raise ConfigurationLoaderError(
                    f"Failed to parse rendered template '{template_name}': {str(e)}",
                    config_path=str(template_dir / template_name),
                    stage="template_parse",
                    original_error=e
                )
            
            # Validate if requested
            if validate:
                cls._validate_config(config, schema_type)
            
            logger.info(f"Successfully rendered template: {template_name}")
            return config
        
        except Exception as e:
            cls._stats['errors'] += 1
            
            if isinstance(e, ConfigurationLoaderError):
                raise
            else:
                raise ConfigurationLoaderError(
                    f"Template loading failed: {str(e)}",
                    config_path=template_name,
                    stage="template",
                    original_error=e
                )
    
    @classmethod
    def load_multiple(
        cls,
        config_paths: List[Union[str, Path]],
        merge_strategy: str = 'deep',
        validate: bool = False,
        schema_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load multiple configuration files and merge them.
        
        This method loads multiple configuration files and merges them
        according to the specified strategy. Useful for composing complex
        configurations from multiple sources.
        
        Args:
            config_paths: List of configuration file paths to load
            merge_strategy: How to merge configurations ('deep', 'shallow', 'replace')
            validate: Whether to validate merged configuration
            schema_type: Type of schema for validation
        
        Returns:
            Merged configuration dictionary
        
        Raises:
            ConfigurationLoaderError: If loading or merging fails
        
        Examples:
            Merge base and override configs:
                config = ConfigLoader.load_multiple([
                    'models/base_config.yaml',
                    'models/my_override.yaml'
                ])
            
            Deep merge with validation:
                config = ConfigLoader.load_multiple(
                    ['base.yaml', 'dev.yaml', 'local.yaml'],
                    merge_strategy='deep',
                    validate=True
                )
        """
        if not config_paths:
            raise ConfigurationLoaderError(
                "No configuration paths provided",
                stage="load_multiple"
            )
        
        try:
            # Load first configuration as base
            merged = cls.load(config_paths[0])
            
            # Merge remaining configurations
            for config_path in config_paths[1:]:
                config = cls.load(config_path)
                
                if merge_strategy == 'deep':
                    merged = cls._deep_merge(merged, config)
                elif merge_strategy == 'shallow':
                    merged = cls._shallow_merge(merged, config)
                elif merge_strategy == 'replace':
                    merged = config
                else:
                    raise ConfigurationLoaderError(
                        f"Unknown merge strategy: {merge_strategy}",
                        stage="merge"
                    )
            
            # Validate merged configuration if requested
            if validate:
                cls._validate_config(merged, schema_type)
            
            logger.info(f"Successfully merged {len(config_paths)} configurations")
            return merged
        
        except Exception as e:
            cls._stats['errors'] += 1
            
            if isinstance(e, ConfigurationLoaderError):
                raise
            else:
                raise ConfigurationLoaderError(
                    f"Failed to load multiple configurations: {str(e)}",
                    stage="load_multiple",
                    original_error=e
                )
    
    @classmethod
    def load_for_platform(
        cls,
        config_category: str,
        platform: str,
        validate: bool = False
    ) -> Dict[str, Any]:
        """
        Load platform-specific configuration.
        
        This method loads configuration optimized for a specific platform
        (Colab, Kaggle, Local, etc.) by searching for platform-specific
        configuration files.
        
        Args:
            config_category: Configuration category (e.g., 'training', 'deployment')
            platform: Platform name ('colab', 'kaggle', 'local', etc.)
            validate: Whether to validate configuration
        
        Returns:
            Platform-specific configuration dictionary
        
        Raises:
            ConfigurationLoaderError: If platform config not found
        
        Examples:
            Load Colab training config:
                config = ConfigLoader.load_for_platform('training', 'colab')
            
            Load Kaggle deployment config:
                config = ConfigLoader.load_for_platform('deployment', 'kaggle')
        """
        try:
            # Try platform-adaptive directory first
            platform_adaptive_path = CONFIGS_ROOT / config_category / "platform_adaptive" / f"{platform}_{config_category}.yaml"
            
            if platform_adaptive_path.exists():
                return cls.load(platform_adaptive_path, validate=validate)
            
            # Try environment-specific configuration
            env_path = CONFIGS_ROOT / "environments" / f"{platform}.yaml"
            
            if env_path.exists():
                return cls.load(env_path, validate=validate)
            
            # Try platform profiles
            profile_path = CONFIGS_ROOT / "deployment" / "platform_profiles" / f"{platform}_profile.yaml"
            
            if profile_path.exists():
                return cls.load(profile_path, validate=validate)
            
            raise ConfigurationLoaderError(
                f"Platform-specific configuration not found for platform '{platform}' "
                f"in category '{config_category}'",
                stage="platform_search"
            )
        
        except Exception as e:
            if isinstance(e, ConfigurationLoaderError):
                raise
            else:
                raise ConfigurationLoaderError(
                    f"Failed to load platform configuration: {str(e)}",
                    stage="load_for_platform",
                    original_error=e
                )
    
    @classmethod
    def reload(
        cls,
        config_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Force reload configuration from disk, bypassing cache.
        
        This method invalidates any cached version of the configuration
        and loads it fresh from disk.
        
        Args:
            config_path: Path to configuration file
            **kwargs: Additional arguments passed to load()
        
        Returns:
            Freshly loaded configuration dictionary
        
        Examples:
            Reload configuration:
                config = ConfigLoader.reload('models/my_model.yaml')
        """
        resolved_path = cls._resolve_path(config_path)
        
        # Invalidate cache
        cache_key = cls._get_cache_key(resolved_path)
        if cache_key in cls._cache:
            del cls._cache[cache_key]
        if str(resolved_path) in cls._file_mtimes:
            del cls._file_mtimes[str(resolved_path)]
        
        # Load fresh configuration
        return cls.load(config_path, use_cache=False, **kwargs)
    
    @classmethod
    def clear_cache(cls) -> None:
        """
        Clear all cached configurations.
        
        This method removes all cached configurations and file modification
        times, forcing all subsequent loads to read from disk.
        
        Examples:
            Clear cache:
                ConfigLoader.clear_cache()
        """
        cls._cache.clear()
        cls._file_mtimes.clear()
        logger.info("Configuration cache cleared")
    
    @classmethod
    def get_cache_stats(cls) -> Dict[str, Any]:
        """
        Get configuration loading statistics.
        
        Returns:
            Dictionary containing loading statistics
        
        Examples:
            Get stats:
                stats = ConfigLoader.get_cache_stats()
                print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
        """
        total_loads = cls._stats['loads']
        cache_hits = cls._stats['cache_hits']
        cache_misses = cls._stats['cache_misses']
        
        hit_rate = cache_hits / total_loads if total_loads > 0 else 0.0
        
        return {
            'total_loads': total_loads,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses,
            'cache_hit_rate': hit_rate,
            'template_renders': cls._stats['template_renders'],
            'validations': cls._stats['validations'],
            'errors': cls._stats['errors'],
            'cached_configs': len(cls._cache),
        }
    
    # Private helper methods
    
    @classmethod
    def _resolve_path(cls, config_path: Union[str, Path]) -> Path:
        """
        Resolve configuration file path to absolute path.
        
        Args:
            config_path: Configuration file path (relative or absolute)
        
        Returns:
            Resolved absolute path
        
        Raises:
            FileNotFoundError: If configuration file not found
        """
        path = Path(config_path)
        
        # If absolute path, use as-is
        if path.is_absolute():
            if not path.exists():
                raise FileNotFoundError(f"Configuration file not found: {path}")
            return path
        
        # Try relative to configs directory
        configs_path = CONFIGS_ROOT / path
        if configs_path.exists():
            return configs_path
        
        # Try relative to project root
        project_path = PROJECT_ROOT / path
        if project_path.exists():
            return project_path
        
        # Try relative to current working directory
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
        
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Searched in:\n"
            f"  - {configs_path}\n"
            f"  - {project_path}\n"
            f"  - {cwd_path}"
        )
    
    @classmethod
    def _load_yaml_file(cls, file_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Load and parse YAML file.
        
        Args:
            file_path: Path to YAML file
            **kwargs: Additional arguments for YAML loader
        
        Returns:
            Parsed YAML content as dictionary
        
        Raises:
            yaml.YAMLError: If YAML parsing fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
            
            if content is None:
                logger.warning(f"Empty configuration file: {file_path}")
                return {}
            
            if not isinstance(content, dict):
                raise ConfigurationLoaderError(
                    f"Configuration file must contain a dictionary, got {type(content).__name__}",
                    config_path=str(file_path),
                    stage="yaml_parse"
                )
            
            return content
        
        except yaml.YAMLError as e:
            raise ConfigurationLoaderError(
                f"Failed to parse YAML: {str(e)}",
                config_path=str(file_path),
                stage="yaml_parse",
                original_error=e
            )
        except Exception as e:
            raise ConfigurationLoaderError(
                f"Failed to read configuration file: {str(e)}",
                config_path=str(file_path),
                stage="file_read",
                original_error=e
            )
    
    @classmethod
    def _resolve_inheritance(
        cls,
        config: Dict[str, Any],
        current_path: Path
    ) -> Dict[str, Any]:
        """
        Resolve configuration inheritance.
        
        Args:
            config: Configuration dictionary with _inherit_from key
            current_path: Path to current configuration file
        
        Returns:
            Configuration with inheritance resolved
        
        Raises:
            ConfigurationLoaderError: If circular inheritance detected
        """
        inherit_from = config.get('_inherit_from')
        
        if not inherit_from:
            return config
        
        # Check for circular inheritance
        current_path_str = str(current_path)
        if current_path_str in cls._inheritance_chain:
            chain = ' -> '.join(cls._inheritance_chain + [current_path_str])
            raise ConfigurationLoaderError(
                f"Circular inheritance detected: {chain}",
                config_path=current_path_str,
                stage="inheritance"
            )
        
        # Add to inheritance chain
        cls._inheritance_chain.append(current_path_str)
        
        try:
            # Resolve parent path
            if inherit_from.startswith('/'):
                parent_path = CONFIGS_ROOT / inherit_from.lstrip('/')
            else:
                parent_path = current_path.parent / inherit_from
            
            # Load parent configuration
            parent_config = cls.load(parent_path, resolve_inheritance=True)
            
            # Get merge strategy
            merge_strategy = config.get('_merge_strategy', 'deep')
            
            # Merge configurations
            if merge_strategy == 'deep':
                merged = cls._deep_merge(parent_config, config)
            elif merge_strategy == 'shallow':
                merged = cls._shallow_merge(parent_config, config)
            elif merge_strategy == 'replace':
                merged = config
            else:
                merged = cls._deep_merge(parent_config, config)
            
            # Remove meta keys
            for key in ['_inherit_from', '_merge_strategy']:
                merged.pop(key, None)
            
            return merged
        
        finally:
            # Remove from inheritance chain
            cls._inheritance_chain.pop()
    
    @classmethod
    def _substitute_env_vars(
        cls,
        config: Union[Dict, List, str, Any],
        visited: Optional[set] = None
    ) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Supports multiple syntax forms:
        - ${VAR_NAME}: Required variable
        - ${VAR_NAME:-default}: Optional with default
        - ${VAR_NAME:?error}: Required with custom error
        - $VAR_NAME: Simple substitution
        
        Args:
            config: Configuration to process
            visited: Set of visited objects (for cycle detection)
        
        Returns:
            Configuration with environment variables substituted
        """
        if visited is None:
            visited = set()
        
        # Handle dictionaries
        if isinstance(config, dict):
            obj_id = id(config)
            if obj_id in visited:
                return config
            visited.add(obj_id)
            
            return {
                key: cls._substitute_env_vars(value, visited)
                for key, value in config.items()
            }
        
        # Handle lists
        elif isinstance(config, list):
            obj_id = id(config)
            if obj_id in visited:
                return config
            visited.add(obj_id)
            
            return [
                cls._substitute_env_vars(item, visited)
                for item in config
            ]
        
        # Handle strings
        elif isinstance(config, str):
            return cls._substitute_string_env_vars(config)
        
        # Return other types as-is
        else:
            return config
    
    @classmethod
    def _substitute_string_env_vars(cls, value: str) -> str:
        """
        Substitute environment variables in a string.
        
        Args:
            value: String potentially containing environment variables
        
        Returns:
            String with environment variables substituted
        """
        # Pattern: ${VAR_NAME:-default} or ${VAR_NAME:?error} or ${VAR_NAME}
        pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*)(:-([^}]*)|:\?([^}]*))?\}'
        
        def replace(match):
            var_name = match.group(1)
            has_operator = match.group(2)
            default_value = match.group(3)
            error_message = match.group(4)
            
            env_value = os.environ.get(var_name)
            
            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            elif error_message is not None:
                raise ConfigurationLoaderError(
                    f"Required environment variable not set: {var_name}. {error_message}",
                    stage="env_substitution"
                )
            else:
                raise ConfigurationLoaderError(
                    f"Required environment variable not set: {var_name}",
                    stage="env_substitution"
                )
        
        # Substitute ${VAR} syntax
        result = re.sub(pattern, replace, value)
        
        # Also handle simple $VAR syntax
        simple_pattern = r'\$([A-Za-z_][A-Za-z0-9_]*)'
        
        def simple_replace(match):
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))
        
        result = re.sub(simple_pattern, simple_replace, result)
        
        return result
    
    @classmethod
    def _merge_with_defaults(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge configuration with smart defaults.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            Configuration merged with defaults
        """
        try:
            # Lazy import to avoid circular dependency
            from configs.smart_defaults import SmartDefaults
            
            return SmartDefaults.merge(config)
        except ImportError:
            logger.warning("SmartDefaults not available, skipping default merging")
            return config
    
    @classmethod
    def _validate_config(
        cls,
        config: Dict[str, Any],
        schema_type: Optional[str] = None
    ) -> None:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema_type: Type of schema to validate against
        
        Raises:
            ConfigurationLoaderError: If validation fails
        """
        try:
            # Lazy import to avoid circular dependency
            from configs.config_validator import ConfigValidator
            
            ConfigValidator.validate(config, schema_type=schema_type)
        except ImportError:
            logger.warning("ConfigValidator not available, skipping validation")
        except Exception as e:
            raise ConfigurationLoaderError(
                f"Configuration validation failed: {str(e)}",
                stage="validation",
                original_error=e
            )
    
    @classmethod
    def _deep_merge(
        cls,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
        
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    @classmethod
    def _shallow_merge(
        cls,
        base: Dict[str, Any],
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Shallow merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Override dictionary
        
        Returns:
            Merged dictionary
        """
        result = deepcopy(base)
        result.update(deepcopy(override))
        return result
    
    @classmethod
    def _get_cache_key(cls, file_path: Path) -> str:
        """
        Generate cache key for configuration file.
        
        Args:
            file_path: Path to configuration file
        
        Returns:
            Cache key string
        """
        return str(file_path.resolve())
    
    @classmethod
    def _get_from_cache(cls, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Get configuration from cache if valid.
        
        Args:
            file_path: Path to configuration file
        
        Returns:
            Cached configuration or None if not cached or invalid
        """
        cache_key = cls._get_cache_key(file_path)
        
        if cache_key not in cls._cache:
            return None
        
        # Check if file has been modified
        try:
            current_mtime = file_path.stat().st_mtime
            cached_mtime = cls._file_mtimes.get(str(file_path))
            
            if cached_mtime is None or current_mtime > cached_mtime:
                # File has been modified, invalidate cache
                del cls._cache[cache_key]
                return None
        except OSError:
            # File no longer exists, invalidate cache
            del cls._cache[cache_key]
            return None
        
        return cls._cache[cache_key]
    
    @classmethod
    def _add_to_cache(cls, file_path: Path, config: Dict[str, Any]) -> None:
        """
        Add configuration to cache.
        
        Args:
            file_path: Path to configuration file
            config: Configuration to cache
        """
        cache_key = cls._get_cache_key(file_path)
        cls._cache[cache_key] = deepcopy(config)
        
        try:
            cls._file_mtimes[str(file_path)] = file_path.stat().st_mtime
        except OSError:
            pass


# Convenience functions for direct module usage

def load(config_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        **kwargs: Additional arguments passed to ConfigLoader.load()
    
    Returns:
        Loaded configuration dictionary
    
    Examples:
        config = load('models/deberta_v3_xlarge_lora.yaml')
    """
    return ConfigLoader.load(config_path, **kwargs)


def load_template(template_name: str, **template_vars) -> Dict[str, Any]:
    """
    Convenience function to load and render template.
    
    Args:
        template_name: Template file name
        **template_vars: Variables for template rendering
    
    Returns:
        Rendered configuration dictionary
    
    Examples:
        config = load_template('model_template.yaml.j2', rank=16)
    """
    return ConfigLoader.load_template(template_name, **template_vars)


def load_for_platform(
    config_category: str,
    platform: str,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to load platform-specific configuration.
    
    Args:
        config_category: Configuration category
        platform: Platform name
        **kwargs: Additional arguments
    
    Returns:
        Platform-specific configuration dictionary
    
    Examples:
        config = load_for_platform('training', 'colab')
    """
    return ConfigLoader.load_for_platform(config_category, platform, **kwargs)


# Module-level exports
__all__ = [
    'ConfigLoader',
    'ConfigurationLoaderError',
    'load',
    'load_template',
    'load_for_platform',
]


# Module initialization
logger.info(
    f"Configuration Loader initialized for {__project__} v{__version__} "
    f"(Author: {__author__})"
)

# Add project metadata
__project__ = "AG News Text Classification (ag-news-text-classification)"
