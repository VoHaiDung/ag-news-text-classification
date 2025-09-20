"""
Test Fixtures for Configuration Module
=======================================

This module provides test fixtures and utilities for testing configuration
loading, validation, and processing components.

The test fixtures follow principles from:
- Meszaros (2007): "xUnit Test Patterns: Refactoring Test Code"
- Freeman & Pryce (2009): "Growing Object-Oriented Software, Guided by Tests"
- Khorikov (2020): "Unit Testing Principles, Practices, and Patterns"

Testing methodology based on:
- Property-based testing (MacIver, 2019)
- Fixture design patterns (Meszaros, 2007)
- Test data builders (Freeman & Pryce, 2009)

Author: Võ Hải Dũng
License: MIT
"""

import os
import json
import yaml
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
import pytest
from copy import deepcopy

# Test configuration constants
TEST_CONFIG_DIR = Path(__file__).parent / "test_configs"
TEMP_CONFIG_DIR = Path(tempfile.gettempdir()) / "ag_news_test_configs"


# ============================================================================
# Configuration Fixtures
# ============================================================================

@dataclass
class ConfigFixture:
    """Base class for configuration fixtures."""
    
    name: str
    type: str
    valid: bool = True
    description: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fixture to dictionary."""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """Convert fixture to YAML string."""
        return yaml.dump(self.data, default_flow_style=False)
    
    def to_json(self) -> str:
        """Convert fixture to JSON string."""
        return json.dumps(self.data, indent=2)


# ============================================================================
# Model Configuration Fixtures
# ============================================================================

class ModelConfigFixtures:
    """Fixtures for model configurations."""
    
    @staticmethod
    def valid_transformer_config() -> Dict[str, Any]:
        """Valid transformer model configuration."""
        return {
            "name": "test_transformer",
            "type": "transformer",
            "architecture": "bert",
            "model": {
                "pretrained_model_name": "bert-base-uncased",
                "model_revision": "main",
                "cache_dir": "./.cache/models",
                "use_auth_token": False
            },
            "architecture_params": {
                "hidden_size": 768,
                "num_hidden_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "max_position_embeddings": 512
            },
            "task": {
                "num_labels": 4,
                "problem_type": "single_label_classification"
            },
            "training": {
                "learning_rate": 2e-5,
                "train_batch_size": 32,
                "num_train_epochs": 5
            }
        }
    
    @staticmethod
    def valid_ensemble_config() -> Dict[str, Any]:
        """Valid ensemble model configuration."""
        return {
            "name": "test_ensemble",
            "type": "ensemble",
            "method": "voting",
            "members": [
                {
                    "name": "model1",
                    "config_path": "models/single/model1.yaml",
                    "weight": 0.5
                },
                {
                    "name": "model2",
                    "config_path": "models/single/model2.yaml",
                    "weight": 0.5
                }
            ],
            "voting": {
                "method": "soft",
                "use_weights": True,
                "normalize_weights": True
            }
        }
    
    @staticmethod
    def invalid_model_config() -> Dict[str, Any]:
        """Invalid model configuration (missing required fields)."""
        return {
            "name": "invalid_model",
            # Missing 'type' field
            "architecture": "bert",
            "task": {
                # Missing 'num_labels' field
                "problem_type": "single_label_classification"
            }
        }
    
    @staticmethod
    def edge_case_model_config() -> Dict[str, Any]:
        """Edge case model configuration."""
        return {
            "name": "edge_case_model",
            "type": "transformer",
            "architecture": "custom",
            "model": {
                "pretrained_model_name": None,  # No pretrained model
                "from_scratch": True
            },
            "task": {
                "num_labels": 1000,  # Unusually high number of labels
                "problem_type": "multi_label_classification"
            },
            "training": {
                "learning_rate": 1e-10,  # Extremely small learning rate
                "train_batch_size": 1,  # Minimum batch size
                "num_train_epochs": 1000  # Very high number of epochs
            }
        }


# ============================================================================
# Training Configuration Fixtures
# ============================================================================

class TrainingConfigFixtures:
    """Fixtures for training configurations."""
    
    @staticmethod
    def valid_standard_training() -> Dict[str, Any]:
        """Valid standard training configuration."""
        return {
            "name": "test_training",
            "type": "standard",
            "description": "Test training configuration",
            "general": {
                "seed": 42,
                "deterministic": True,
                "num_train_epochs": 10,
                "early_stopping": True,
                "early_stopping_patience": 3
            },
            "batch": {
                "per_device_train_batch_size": 32,
                "per_device_eval_batch_size": 64,
                "gradient_accumulation_steps": 1
            },
            "optimizer": {
                "name": "adamw",
                "learning_rate": 2e-5,
                "weight_decay": 0.01
            },
            "scheduler": {
                "name": "cosine",
                "warmup_ratio": 0.1
            }
        }
    
    @staticmethod
    def valid_advanced_training() -> Dict[str, Any]:
        """Valid advanced training configuration."""
        return {
            "name": "test_advanced_training",
            "type": "advanced",
            "inherit": "training/standard/base_training.yaml",
            "adversarial": {
                "enabled": True,
                "method": "fgm",
                "epsilon": 1.0
            },
            "distillation": {
                "enabled": True,
                "temperature": 4.0,
                "alpha": 0.7
            },
            "curriculum": {
                "enabled": True,
                "strategy": "easy_to_hard"
            }
        }
    
    @staticmethod
    def valid_efficient_training() -> Dict[str, Any]:
        """Valid efficient training configuration (LoRA/PEFT)."""
        return {
            "name": "test_lora",
            "type": "efficient_training",
            "lora": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "target_modules": ["q_proj", "v_proj"]
            },
            "training": {
                "learning_rate": 3e-4,
                "num_epochs": 20
            }
        }
    
    @staticmethod
    def invalid_training_config() -> Dict[str, Any]:
        """Invalid training configuration."""
        return {
            "name": "invalid_training",
            "type": "standard",
            "optimizer": {
                "name": "invalid_optimizer",  # Invalid optimizer
                "learning_rate": -1.0  # Negative learning rate
            },
            "batch": {
                "per_device_train_batch_size": 0  # Invalid batch size
            }
        }


# ============================================================================
# Data Configuration Fixtures
# ============================================================================

class DataConfigFixtures:
    """Fixtures for data configurations."""
    
    @staticmethod
    def valid_preprocessing_config() -> Dict[str, Any]:
        """Valid preprocessing configuration."""
        return {
            "name": "test_preprocessing",
            "type": "preprocessing",
            "text_cleaning": {
                "lowercase": False,
                "remove_urls": True,
                "remove_emails": True,
                "normalize_unicode": True,
                "min_length": 10,
                "max_length": 1000
            },
            "tokenization": {
                "tokenizer": "bert-base-uncased",
                "max_length": 512,
                "padding": "max_length",
                "truncation": True
            }
        }
    
    @staticmethod
    def valid_augmentation_config() -> Dict[str, Any]:
        """Valid augmentation configuration."""
        return {
            "name": "test_augmentation",
            "type": "augmentation",
            "pipeline": {
                "enabled": True,
                "augmentation_probability": 0.5,
                "num_augmentations": 2
            },
            "eda": {
                "synonym_replacement": {
                    "enabled": True,
                    "probability": 0.1,
                    "num_replacements": 3
                }
            }
        }
    
    @staticmethod
    def valid_selection_config() -> Dict[str, Any]:
        """Valid data selection configuration."""
        return {
            "name": "test_selection",
            "type": "data_selection",
            "algorithm": {
                "method": "k_center_greedy",
                "k_center_greedy": {
                    "distance_metric": "euclidean"
                }
            },
            "selection": {
                "subset_fraction": 0.1,
                "min_samples_per_class": 10
            }
        }


# ============================================================================
# Environment Configuration Fixtures
# ============================================================================

class EnvironmentConfigFixtures:
    """Fixtures for environment configurations."""
    
    @staticmethod
    def valid_dev_environment() -> Dict[str, Any]:
        """Valid development environment configuration."""
        return {
            "environment": "development",
            "stage": "dev",
            "debug": True,
            "verbose": True,
            "system": {
                "seed": 42,
                "deterministic": True,
                "num_workers": 4
            },
            "hardware": {
                "device": "cuda",
                "gpu_ids": [0],
                "mixed_precision": False
            },
            "data": {
                "base_dir": "./data",
                "cache_dir": "./.cache/dev",
                "max_samples": 10000
            }
        }
    
    @staticmethod
    def valid_prod_environment() -> Dict[str, Any]:
        """Valid production environment configuration."""
        return {
            "environment": "production",
            "stage": "prod",
            "debug": False,
            "verbose": False,
            "system": {
                "seed": 42,
                "deterministic": False,
                "num_workers": 16
            },
            "hardware": {
                "device": "cuda",
                "gpu_ids": [0, 1, 2, 3],
                "mixed_precision": True,
                "distributed": {
                    "backend": "nccl",
                    "world_size": 4
                }
            },
            "security": {
                "enable_auth": True,
                "enable_https": True,
                "api_key_required": True
            }
        }


# ============================================================================
# Experiment Configuration Fixtures
# ============================================================================

class ExperimentConfigFixtures:
    """Fixtures for experiment configurations."""
    
    @staticmethod
    def valid_baseline_experiment() -> Dict[str, Any]:
        """Valid baseline experiment configuration."""
        return {
            "name": "test_baseline",
            "type": "experiment",
            "phase": "baseline",
            "methodology": {
                "reproducibility": {
                    "seeds": [42, 123, 456],
                    "deterministic": True
                }
            },
            "models": {
                "bert_base": {
                    "name": "bert-base-uncased",
                    "hyperparameters": {
                        "learning_rate": [2e-5, 3e-5],
                        "batch_size": [32]
                    }
                }
            }
        }
    
    @staticmethod
    def valid_sota_experiment() -> Dict[str, Any]:
        """Valid SOTA experiment configuration."""
        return {
            "name": "test_sota",
            "type": "experiment",
            "phase": "sota",
            "inherit": ["experiments/baselines/transformer_baseline.yaml"],
            "training_pipeline": {
                "stage1_dapt": {
                    "enabled": True,
                    "duration_hours": 24
                },
                "stage2_fine_tuning": {
                    "enabled": True,
                    "progressive": True
                }
            }
        }


# ============================================================================
# Configuration Validation Utilities
# ============================================================================

class ConfigValidator:
    """Utilities for validating configurations."""
    
    @staticmethod
    def validate_required_fields(config: Dict[str, Any], 
                                required_fields: List[str]) -> bool:
        """
        Validate that all required fields are present.
        
        Args:
            config: Configuration dictionary
            required_fields: List of required field paths (e.g., "model.name")
            
        Returns:
            True if all required fields present
        """
        for field_path in required_fields:
            parts = field_path.split(".")
            current = config
            
            for part in parts:
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
        
        return True
    
    @staticmethod
    def validate_field_types(config: Dict[str, Any],
                            field_types: Dict[str, type]) -> bool:
        """
        Validate field types.
        
        Args:
            config: Configuration dictionary
            field_types: Dictionary mapping field paths to expected types
            
        Returns:
            True if all fields have correct types
        """
        for field_path, expected_type in field_types.items():
            parts = field_path.split(".")
            current = config
            
            for part in parts[:-1]:
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
            
            if parts[-1] not in current:
                return False
                
            if not isinstance(current[parts[-1]], expected_type):
                return False
        
        return True
    
    @staticmethod
    def validate_field_values(config: Dict[str, Any],
                            field_constraints: Dict[str, Any]) -> bool:
        """
        Validate field value constraints.
        
        Args:
            config: Configuration dictionary
            field_constraints: Dictionary of constraints
            
        Returns:
            True if all constraints satisfied
        """
        for field_path, constraint in field_constraints.items():
            parts = field_path.split(".")
            current = config
            
            for part in parts[:-1]:
                if not isinstance(current, dict) or part not in current:
                    return False
                current = current[part]
            
            if parts[-1] not in current:
                return False
            
            value = current[parts[-1]]
            
            # Check constraint type
            if isinstance(constraint, dict):
                if "min" in constraint and value < constraint["min"]:
                    return False
                if "max" in constraint and value > constraint["max"]:
                    return False
                if "choices" in constraint and value not in constraint["choices"]:
                    return False
            elif callable(constraint):
                if not constraint(value):
                    return False
        
        return True


# ============================================================================
# Configuration Factory
# ============================================================================

class ConfigFactory:
    """Factory for creating test configurations."""
    
    @staticmethod
    def create_config(config_type: str, **kwargs) -> Dict[str, Any]:
        """
        Create a test configuration.
        
        Args:
            config_type: Type of configuration to create
            **kwargs: Additional parameters
            
        Returns:
            Configuration dictionary
        """
        factories = {
            "model": ModelConfigFixtures.valid_transformer_config,
            "ensemble": ModelConfigFixtures.valid_ensemble_config,
            "training": TrainingConfigFixtures.valid_standard_training,
            "advanced_training": TrainingConfigFixtures.valid_advanced_training,
            "efficient_training": TrainingConfigFixtures.valid_efficient_training,
            "preprocessing": DataConfigFixtures.valid_preprocessing_config,
            "augmentation": DataConfigFixtures.valid_augmentation_config,
            "selection": DataConfigFixtures.valid_selection_config,
            "dev_env": EnvironmentConfigFixtures.valid_dev_environment,
            "prod_env": EnvironmentConfigFixtures.valid_prod_environment,
            "baseline_exp": ExperimentConfigFixtures.valid_baseline_experiment,
            "sota_exp": ExperimentConfigFixtures.valid_sota_experiment
        }
        
        if config_type not in factories:
            raise ValueError(f"Unknown config type: {config_type}")
        
        config = factories[config_type]()
        
        # Apply any overrides
        for key, value in kwargs.items():
            if "." in key:
                # Nested key
                parts = key.split(".")
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config[key] = value
        
        return config
    
    @staticmethod
    def create_config_file(config: Dict[str, Any],
                          file_path: Path,
                          format: str = "yaml") -> Path:
        """
        Create a configuration file.
        
        Args:
            config: Configuration dictionary
            file_path: Path to save file
            format: File format (yaml or json)
            
        Returns:
            Path to created file
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            if format == "yaml":
                yaml.dump(config, f, default_flow_style=False)
            elif format == "json":
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        return file_path


# ============================================================================
# Mock Configuration Generator
# ============================================================================

class MockConfigGenerator:
    """Generator for mock configurations for testing."""
    
    @staticmethod
    def generate_random_config(config_type: str,
                             valid: bool = True,
                             seed: int = 42) -> Dict[str, Any]:
        """
        Generate a random configuration for testing.
        
        Args:
            config_type: Type of configuration
            valid: Whether to generate valid config
            seed: Random seed for reproducibility
            
        Returns:
            Random configuration
        """
        import random
        random.seed(seed)
        
        base_config = ConfigFactory.create_config(config_type)
        
        if not valid:
            # Introduce errors
            modifications = [
                lambda c: c.pop(list(c.keys())[0]),  # Remove a key
                lambda c: c.update({"invalid_key": "invalid_value"}),  # Add invalid key
                lambda c: c.update({list(c.keys())[0]: None})  # Set value to None
            ]
            
            modification = random.choice(modifications)
            modification(base_config)
        
        return base_config
    
    @staticmethod
    def generate_config_variations(base_config: Dict[str, Any],
                                  num_variations: int = 5) -> List[Dict[str, Any]]:
        """
        Generate variations of a base configuration.
        
        Args:
            base_config: Base configuration
            num_variations: Number of variations to generate
            
        Returns:
            List of configuration variations
        """
        variations = []
        
        for i in range(num_variations):
            config = deepcopy(base_config)
            
            # Modify some values
            if "training" in config and "learning_rate" in config["training"]:
                config["training"]["learning_rate"] *= (1 + i * 0.1)
            
            if "batch" in config and "per_device_train_batch_size" in config["batch"]:
                config["batch"]["per_device_train_batch_size"] = 16 * (2 ** i)
            
            variations.append(config)
        
        return variations


# ============================================================================
# Test Utilities
# ============================================================================

def create_temp_config_dir() -> Path:
    """Create temporary directory for test configs."""
    TEMP_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    return TEMP_CONFIG_DIR


def cleanup_temp_configs():
    """Clean up temporary test configs."""
    import shutil
    if TEMP_CONFIG_DIR.exists():
        shutil.rmtree(TEMP_CONFIG_DIR)


def get_all_config_fixtures() -> Dict[str, ConfigFixture]:
    """Get all available configuration fixtures."""
    fixtures = {}
    
    # Model configs
    fixtures["model_valid"] = ConfigFixture(
        name="model_valid",
        type="model",
        valid=True,
        data=ModelConfigFixtures.valid_transformer_config()
    )
    
    fixtures["model_invalid"] = ConfigFixture(
        name="model_invalid",
        type="model",
        valid=False,
        data=ModelConfigFixtures.invalid_model_config()
    )
    
    # Training configs
    fixtures["training_valid"] = ConfigFixture(
        name="training_valid",
        type="training",
        valid=True,
        data=TrainingConfigFixtures.valid_standard_training()
    )
    
    # Data configs
    fixtures["preprocessing_valid"] = ConfigFixture(
        name="preprocessing_valid",
        type="data",
        valid=True,
        data=DataConfigFixtures.valid_preprocessing_config()
    )
    
    # Environment configs
    fixtures["env_dev"] = ConfigFixture(
        name="env_dev",
        type="environment",
        valid=True,
        data=EnvironmentConfigFixtures.valid_dev_environment()
    )
    
    return fixtures


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def config_factory():
    """Pytest fixture for config factory."""
    return ConfigFactory()


@pytest.fixture
def config_validator():
    """Pytest fixture for config validator."""
    return ConfigValidator()


@pytest.fixture
def mock_generator():
    """Pytest fixture for mock config generator."""
    return MockConfigGenerator()


@pytest.fixture
def temp_config_dir():
    """Pytest fixture for temporary config directory."""
    dir_path = create_temp_config_dir()
    yield dir_path
    cleanup_temp_configs()


@pytest.fixture
def all_fixtures():
    """Pytest fixture providing all config fixtures."""
    return get_all_config_fixtures()


# Export public API
__all__ = [
    # Classes
    "ConfigFixture",
    "ModelConfigFixtures",
    "TrainingConfigFixtures",
    "DataConfigFixtures",
    "EnvironmentConfigFixtures",
    "ExperimentConfigFixtures",
    "ConfigValidator",
    "ConfigFactory",
    "MockConfigGenerator",
    
    # Functions
    "create_temp_config_dir",
    "cleanup_temp_configs",
    "get_all_config_fixtures",
    
    # Pytest fixtures
    "config_factory",
    "config_validator",
    "mock_generator",
    "temp_config_dir",
    "all_fixtures"
]
