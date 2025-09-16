"""
Configuration Module for AG News Classification
================================================

Central configuration management following:
- The Twelve-Factor App methodology (https://12factor.net/config)
- Martin Fowler's Configuration Management patterns
- Google's Best Practices for Configuration

Author: Võ Hải Dũng
License: MIT
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import json
from dataclasses import dataclass, asdict

# Add project root to path
CONFIGS_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIGS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration version for compatibility checking
CONFIG_VERSION = "1.0.0"

@dataclass
class ConfigManager:
    """
    Centralized configuration management.
    
    Following configuration patterns from:
    - Fowler (2016): "Configuration Management"
    - 12-Factor App: "Store config in the environment"
    """
    
    def __init__(self, config_dir: Path = CONFIGS_DIR):
        """Initialize configuration manager."""
        self.config_dir = config_dir
        self.configs_cache: Dict[str, Any] = {}
        self._load_constants()
    
    def _load_constants(self):
        """Load constants from constants.py."""
        from .constants import (
            PROJECT_NAME,
            PROJECT_VERSION,
            AG_NEWS_CLASSES,
            AG_NEWS_NUM_CLASSES,
            LABEL_TO_ID,
            ID_TO_LABEL
        )
        
        self.constants = {
            "project_name": PROJECT_NAME,
            "project_version": PROJECT_VERSION,
            "classes": AG_NEWS_CLASSES,
            "num_classes": AG_NEWS_NUM_CLASSES,
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL
        }
    
    def load_config(
        self,
        config_path: Union[str, Path],
        override_with_env: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config file
            override_with_env: Whether to override with env variables
            
        Returns:
            Configuration dictionary
        """
        # Convert to Path
        if isinstance(config_path, str):
            # Handle relative paths
            if not config_path.startswith('/'):
                config_path = self.config_dir / config_path
            else:
                config_path = Path(config_path)
        
        # Check cache
        cache_key = str(config_path)
        if cache_key in self.configs_cache:
            return self.configs_cache[cache_key]
        
        # Load based on extension
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            config = self._load_yaml(config_path)
        elif config_path.suffix == '.json':
            config = self._load_json(config_path)
        elif config_path.suffix == '.py':
            config = self._load_python(config_path)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Override with environment variables if requested
        if override_with_env:
            config = self._override_with_env(config)
        
        # Cache and return
        self.configs_cache[cache_key] = config
        return config
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """Load YAML configuration."""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON configuration."""
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_python(self, path: Path) -> Dict[str, Any]:
        """Load Python configuration module."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Extract configuration
        config = {}
        for key in dir(module):
            if not key.startswith('_'):
                config[key] = getattr(module, key)
        
        return config
    
    def _override_with_env(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override configuration with environment variables."""
        for key, value in config.items():
            env_key = f"AG_NEWS_{key.upper()}"
            if env_key in os.environ:
                # Type conversion based on original type
                env_value = os.environ[env_key]
                if isinstance(value, bool):
                    config[key] = env_value.lower() in ('true', '1', 'yes')
                elif isinstance(value, int):
                    config[key] = int(env_value)
                elif isinstance(value, float):
                    config[key] = float(env_value)
                else:
                    config[key] = env_value
        
        return config
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get model-specific configuration."""
        config_path = self.config_dir / "models" / "single" / f"{model_name}.yaml"
        if not config_path.exists():
            # Try baselines
            config_path = self.config_dir / "models" / "baselines" / f"{model_name}.yaml"
        
        if config_path.exists():
            return self.load_config(config_path)
        else:
            raise FileNotFoundError(f"Model config not found: {model_name}")
    
    def get_training_config(self, strategy: str = "standard") -> Dict[str, Any]:
        """Get training configuration."""
        config_path = self.config_dir / "training" / strategy / "base_training.yaml"
        if config_path.exists():
            return self.load_config(config_path)
        else:
            # Return default
            return self.load_config("training/standard/base_training.yaml")
    
    def get_data_config(self, config_type: str = "preprocessing") -> Dict[str, Any]:
        """Get data configuration."""
        if config_type == "augmentation":
            return self.load_config("data/augmentation/basic_augment.yaml")
        elif config_type == "preprocessing":
            return self.load_config("data/preprocessing/advanced.yaml")
        else:
            raise ValueError(f"Unknown data config type: {config_type}")

# Global configuration manager instance
config_manager = ConfigManager()

# Convenience functions
def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration file."""
    return config_manager.load_config(path)

def get_model_config(model_name: str) -> Dict[str, Any]:
    """Get model configuration."""
    return config_manager.get_model_config(model_name)

def get_training_config(strategy: str = "standard") -> Dict[str, Any]:
    """Get training configuration."""
    return config_manager.get_training_config(strategy)

def get_data_config(config_type: str = "preprocessing") -> Dict[str, Any]:
    """Get data configuration."""
    return config_manager.get_data_config(config_type)

# Export key constants for easy access
from .constants import (
    PROJECT_NAME,
    PROJECT_VERSION,
    PYTHON_VERSION,
    
    # AG News specific
    AG_NEWS_CLASSES,
    AG_NEWS_NUM_CLASSES,
    LABEL_TO_ID,
    ID_TO_LABEL,
    
    # Model constants
    SUPPORTED_MODELS,
    MAX_SEQUENCE_LENGTH,
    DEFAULT_BATCH_SIZE,
    
    # Paths
    DATA_DIR,
    MODELS_DIR,
    OUTPUTS_DIR,
    LOGS_DIR,
)

__all__ = [
    # Manager
    'ConfigManager',
    'config_manager',
    
    # Functions
    'load_config',
    'get_model_config',
    'get_training_config',
    'get_data_config',
    
    # Constants
    'PROJECT_NAME',
    'PROJECT_VERSION',
    'PYTHON_VERSION',
    'AG_NEWS_CLASSES',
    'AG_NEWS_NUM_CLASSES',
    'LABEL_TO_ID',
    'ID_TO_LABEL',
    'SUPPORTED_MODELS',
    'MAX_SEQUENCE_LENGTH',
    'DEFAULT_BATCH_SIZE',
    'DATA_DIR',
    'MODELS_DIR',
    'OUTPUTS_DIR',
    'LOGS_DIR',
    'CONFIG_VERSION',
]

__version__ = CONFIG_VERSION
