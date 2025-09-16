"""
Configuration loader for AG News Text Classification Framework.

Provides utilities for loading, merging, and validating configuration files
with support for environment variable substitution and inheritance.
"""

import os
import re
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field, asdict
from collections import ChainMap
from copy import deepcopy
import importlib.util

from omegaconf import OmegaConf, DictConfig
from hydra import compose, initialize_config_dir
from pydantic import BaseModel, validator, Field

logger = logging.getLogger(__name__)

# Configuration directory structure
CONFIG_DIR = Path(__file__).parent
ENVIRONMENTS_DIR = CONFIG_DIR / "environments"
MODELS_DIR = CONFIG_DIR / "models"
TRAINING_DIR = CONFIG_DIR / "training"
DATA_DIR = CONFIG_DIR / "data"
EXPERIMENTS_DIR = CONFIG_DIR / "experiments"
FEATURES_DIR = CONFIG_DIR / "features"
SECRETS_DIR = CONFIG_DIR / "secrets"

class ConfigurationError(Exception):
    """Custom exception for configuration errors."""
    pass

@dataclass
class ConfigSchema:
    """Schema for configuration validation."""
    required_fields: List[str] = field(default_factory=list)
    optional_fields: Dict[str, Any] = field(default_factory=dict)
    field_types: Dict[str, Type] = field(default_factory=dict)
    validators: Dict[str, callable] = field(default_factory=dict)

class ConfigValidator:
    """Validator for configuration dictionaries."""
    
    def __init__(self, schema: Optional[ConfigSchema] = None):
        """
        Initialize validator.
        
        Args:
            schema: Configuration schema
        """
        self.schema = schema or ConfigSchema()
    
    def validate(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if valid
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Check required fields
        for field in self.schema.required_fields:
            if field not in config:
                raise ConfigurationError(f"Required field '{field}' missing from configuration")
        
        # Check field types
        for field, expected_type in self.schema.field_types.items():
            if field in config and not isinstance(config[field], expected_type):
                raise ConfigurationError(
                    f"Field '{field}' has incorrect type. "
                    f"Expected {expected_type}, got {type(config[field])}"
                )
        
        # Run custom validators
        for field, validator_func in self.schema.validators.items():
            if field in config:
                try:
                    if not validator_func(config[field]):
                        raise ConfigurationError(f"Validation failed for field '{field}'")
                except Exception as e:
                    raise ConfigurationError(f"Validator error for field '{field}': {e}")
        
        return True

class EnvironmentVariableResolver:
    """Resolver for environment variables in configuration."""
    
    ENV_PATTERN = re.compile(r'\$\{([^}]+)\}')
    
    @classmethod
    def resolve(cls, value: Any) -> Any:
        """
        Resolve environment variables in value.
        
        Args:
            value: Value to resolve
            
        Returns:
            Resolved value
        """
        if isinstance(value, str):
            return cls._resolve_string(value)
        elif isinstance(value, dict):
            return {k: cls.resolve(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [cls.resolve(item) for item in value]
        return value
    
    @classmethod
    def _resolve_string(cls, value: str) -> Any:
        """Resolve environment variables in string."""
        def replacer(match):
            env_var = match.group(1)
            # Support default values with :
            if ':' in env_var:
                var_name, default = env_var.split(':', 1)
                return os.environ.get(var_name, default)
            return os.environ.get(env_var, match.group(0))
        
        resolved = cls.ENV_PATTERN.sub(replacer, value)
        
        # Try to convert to appropriate type
        if resolved.lower() in ['true', 'false']:
            return resolved.lower() == 'true'
        
        try:
            return int(resolved)
        except ValueError:
            try:
                return float(resolved)
            except ValueError:
                return resolved

class ConfigLoader:
    """
    Main configuration loader with support for multiple formats and features.
    """
    
    def __init__(
        self,
        config_dir: Optional[Path] = None,
        use_env_vars: bool = True,
        use_secrets: bool = True,
        validate: bool = True
    ):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Configuration directory
            use_env_vars: Whether to resolve environment variables
            use_secrets: Whether to load secrets
            validate: Whether to validate configurations
        """
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
        self.use_env_vars = use_env_vars
        self.use_secrets = use_secrets
        self.validate = validate
        self._cache = {}
        self._validators = {}
    
    def load(
        self,
        config_path: Union[str, Path],
        overrides: Optional[Dict[str, Any]] = None,
        resolve_inheritance: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            overrides: Dictionary of overrides
            resolve_inheritance: Whether to resolve inheritance
            
        Returns:
            Loaded configuration
        """
        config_path = Path(config_path)
        
        # Check cache
        cache_key = str(config_path)
        if cache_key in self._cache and not overrides:
            return deepcopy(self._cache[cache_key])
        
        # Make path absolute if relative
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
        
        # Load base configuration
        config = self._load_file(config_path)
        
        # Resolve inheritance
        if resolve_inheritance and "inherit" in config:
            parent_configs = config.pop("inherit")
            if not isinstance(parent_configs, list):
                parent_configs = [parent_configs]
            
            # Load parent configurations
            parent_config = {}
            for parent_path in parent_configs:
                parent_full_path = config_path.parent / parent_path
                parent = self.load(parent_full_path, resolve_inheritance=True)
                parent_config = self._deep_merge(parent_config, parent)
            
            # Merge with current config
            config = self._deep_merge(parent_config, config)
        
        # Resolve environment variables
        if self.use_env_vars:
            config = EnvironmentVariableResolver.resolve(config)
        
        # Load and merge secrets
        if self.use_secrets and "secrets" in config:
            secrets = self._load_secrets(config.pop("secrets"))
            config = self._deep_merge(config, secrets)
        
        # Apply overrides
        if overrides:
            config = self._deep_merge(config, overrides)
        
        # Validate configuration
        if self.validate and cache_key in self._validators:
            validator = self._validators[cache_key]
            validator.validate(config)
        
        # Cache result
        self._cache[cache_key] = config
        
        return deepcopy(config)
    
    def load_multiple(
        self,
        config_paths: List[Union[str, Path]],
        merge: bool = True
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load multiple configuration files.
        
        Args:
            config_paths: List of configuration paths
            merge: Whether to merge configurations
            
        Returns:
            Merged configuration or list of configurations
        """
        configs = [self.load(path) for path in config_paths]
        
        if merge:
            result = {}
            for config in configs:
                result = self._deep_merge(result, config)
            return result
        
        return configs
    
    def load_environment(
        self,
        environment: str,
        base_config: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """
        Load environment-specific configuration.
        
        Args:
            environment: Environment name (dev, staging, prod)
            base_config: Optional base configuration
            
        Returns:
            Environment configuration
        """
        env_config_path = ENVIRONMENTS_DIR / f"{environment}.yaml"
        
        if not env_config_path.exists():
            raise ConfigurationError(f"Environment configuration not found: {environment}")
        
        env_config = self.load(env_config_path)
        
        if base_config:
            base = self.load(base_config) if isinstance(base_config, (str, Path)) else base_config
            env_config = self._deep_merge(base, env_config)
        
        # Set environment in config
        env_config["environment"] = environment
        
        return env_config
    
    def load_experiment(
        self,
        experiment_name: str,
        phase: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Load experiment configuration.
        
        Args:
            experiment_name: Experiment name
            phase: Experiment phase
            
        Returns:
            Experiment configuration
        """
        # Build path
        if phase:
            config_path = EXPERIMENTS_DIR / phase / f"{experiment_name}.yaml"
        else:
            # Search for experiment config
            for phase_dir in EXPERIMENTS_DIR.iterdir():
                if phase_dir.is_dir():
                    potential_path = phase_dir / f"{experiment_name}.yaml"
                    if potential_path.exists():
                        config_path = potential_path
                        break
            else:
                raise ConfigurationError(f"Experiment configuration not found: {experiment_name}")
        
        config = self.load(config_path)
        
        # Add experiment metadata
        config["experiment_name"] = experiment_name
        if phase:
            config["experiment_phase"] = phase
        
        return config
    
    def create_pipeline_config(
        self,
        model_config: Union[str, Dict],
        training_config: Union[str, Dict],
        data_config: Union[str, Dict],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Create complete pipeline configuration.
        
        Args:
            model_config: Model configuration or path
            training_config: Training configuration or path
            data_config: Data configuration or path
            **kwargs: Additional configurations
            
        Returns:
            Complete pipeline configuration
        """
        # Load configurations
        model = self.load(model_config) if isinstance(model_config, str) else model_config
        training = self.load(training_config) if isinstance(training_config, str) else training_config
        data = self.load(data_config) if isinstance(data_config, str) else data_config
        
        # Combine into pipeline config
        pipeline_config = {
            "model": model,
            "training": training,
            "data": data,
        }
        
        # Add additional configurations
        pipeline_config.update(kwargs)
        
        return pipeline_config
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")
        
        with open(file_path, "r") as f:
            if file_path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f) or {}
            elif file_path.suffix == ".json":
                return json.load(f)
            else:
                raise ConfigurationError(f"Unsupported configuration format: {file_path.suffix}")
    
    def _load_secrets(self, secrets_config: Union[str, Dict]) -> Dict[str, Any]:
        """Load secrets configuration."""
        if isinstance(secrets_config, str):
            secrets_path = SECRETS_DIR / secrets_config
            if not secrets_path.exists():
                logger.warning(f"Secrets file not found: {secrets_path}")
                return {}
            return self._load_file(secrets_path)
        return secrets_config
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def register_validator(self, config_path: str, validator: ConfigValidator):
        """
        Register validator for configuration.
        
        Args:
            config_path: Configuration path
            validator: Validator instance
        """
        self._validators[config_path] = validator
    
    def save(
        self,
        config: Dict[str, Any],
        save_path: Union[str, Path],
        format: str = "yaml"
    ):
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            save_path: Save path
            format: Output format (yaml or json)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            if format == "yaml":
                yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True)
            elif format == "json":
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Configuration saved to {save_path}")

class HydraConfigLoader:
    """
    Configuration loader using Hydra framework for advanced features.
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize Hydra config loader.
        
        Args:
            config_dir: Configuration directory
        """
        self.config_dir = Path(config_dir) if config_dir else CONFIG_DIR
    
    def load(
        self,
        config_name: str,
        overrides: Optional[List[str]] = None,
        job_name: str = "ag_news"
    ) -> DictConfig:
        """
        Load configuration using Hydra.
        
        Args:
            config_name: Configuration name
            overrides: List of override strings
            job_name: Job name
            
        Returns:
            Hydra DictConfig
        """
        with initialize_config_dir(config_dir=str(self.config_dir), job_name=job_name):
            cfg = compose(config_name=config_name, overrides=overrides or [])
        
        return cfg
    
    def to_dict(self, cfg: DictConfig) -> Dict[str, Any]:
        """Convert Hydra config to dictionary."""
        return OmegaConf.to_container(cfg, resolve=True)

# Configuration models using Pydantic
class ModelConfig(BaseModel):
    """Model configuration schema."""
    name: str = Field(..., description="Model name")
    type: str = Field("transformer", description="Model type")
    pretrained_model_name: Optional[str] = Field(None, description="Pretrained model name")
    num_labels: int = Field(4, description="Number of output labels")
    hidden_size: Optional[int] = Field(None, description="Hidden size")
    num_hidden_layers: Optional[int] = Field(None, description="Number of hidden layers")
    num_attention_heads: Optional[int] = Field(None, description="Number of attention heads")
    dropout_rate: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")
    
    @validator("num_labels")
    def validate_num_labels(cls, v):
        if v < 2:
            raise ValueError("num_labels must be at least 2")
        return v

class TrainingConfig(BaseModel):
    """Training configuration schema."""
    num_epochs: int = Field(10, ge=1, description="Number of epochs")
    batch_size: int = Field(32, ge=1, description="Batch size")
    learning_rate: float = Field(2e-5, gt=0, description="Learning rate")
    warmup_steps: int = Field(500, ge=0, description="Warmup steps")
    weight_decay: float = Field(0.01, ge=0, description="Weight decay")
    gradient_accumulation_steps: int = Field(1, ge=1, description="Gradient accumulation steps")
    mixed_precision: bool = Field(False, description="Use mixed precision")
    seed: int = Field(42, description="Random seed")

class DataConfig(BaseModel):
    """Data configuration schema."""
    dataset_name: str = Field("ag_news", description="Dataset name")
    max_length: int = Field(512, ge=1, description="Maximum sequence length")
    train_split_ratio: float = Field(0.8, gt=0, lt=1, description="Training split ratio")
    val_split_ratio: float = Field(0.1, gt=0, lt=1, description="Validation split ratio")
    test_split_ratio: float = Field(0.1, gt=0, lt=1, description="Test split ratio")
    augmentation_enabled: bool = Field(False, description="Enable data augmentation")
    
    @validator("test_split_ratio")
    def validate_splits(cls, v, values):
        total = values.get("train_split_ratio", 0) + values.get("val_split_ratio", 0) + v
        if not 0.99 <= total <= 1.01:  # Allow small floating point errors
            raise ValueError("Split ratios must sum to 1.0")
        return v

# Global configuration loader instance
config_loader = ConfigLoader()

# Convenience functions
def load_config(config_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """Load configuration from file."""
    return config_loader.load(config_path, **kwargs)

def load_model_config(model_name: str, model_type: str = "single") -> Dict[str, Any]:
    """Load model configuration."""
    config_path = MODELS_DIR / model_type / f"{model_name}.yaml"
    return config_loader.load(config_path)

def load_training_config(config_name: str, training_type: str = "standard") -> Dict[str, Any]:
    """Load training configuration."""
    config_path = TRAINING_DIR / training_type / f"{config_name}.yaml"
    return config_loader.load(config_path)

def load_experiment_config(experiment_name: str, phase: Optional[str] = None) -> Dict[str, Any]:
    """Load experiment configuration."""
    return config_loader.load_experiment(experiment_name, phase)

# Export public API
__all__ = [
    "ConfigLoader",
    "HydraConfigLoader",
    "ConfigValidator",
    "ConfigSchema",
    "EnvironmentVariableResolver",
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
    "config_loader",
    "load_config",
    "load_model_config",
    "load_training_config",
    "load_experiment_config",
    "CONFIG_DIR",
    "MODELS_DIR",
    "TRAINING_DIR",
    "DATA_DIR",
    "EXPERIMENTS_DIR",
]
