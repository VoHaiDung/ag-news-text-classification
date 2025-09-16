"""
Factory Pattern Implementation for Component Creation

Provides factory classes for creating models, trainers, and other components
with automatic configuration management and dependency injection.
"""

import logging
from typing import Any, Dict, Optional, Type, Union
from pathlib import Path
import yaml
import json
from abc import ABC, abstractmethod

from src.core.registry import (
    MODELS, TRAINERS, DATASETS, OPTIMIZERS, 
    SCHEDULERS, LOSSES, AUGMENTERS, ENSEMBLES
)
from src.core.exceptions import ConfigurationError, ComponentNotFoundError

logger = logging.getLogger(__name__)

class BaseFactory(ABC):
    """
    Abstract base factory for creating components.
    
    Implements the Factory Method pattern for component instantiation.
    """
    
    def __init__(self, registry, component_type: str):
        """
        Initialize factory.
        
        Args:
            registry: Component registry to use
            component_type: Type of components (for logging)
        """
        self.registry = registry
        self.component_type = component_type
        self._config_cache = {}
    
    @abstractmethod
    def create(self, name: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
        """
        Create a component instance.
        
        Args:
            name: Component name
            config: Configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Component instance
        """
        pass
    
    def create_from_config(self, config_path: Union[str, Path]) -> Any:
        """
        Create component from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Component instance
        """
        config = self.load_config(config_path)
        
        if "name" not in config:
            raise ConfigurationError(f"'name' field missing in config: {config_path}")
        
        name = config.pop("name")
        return self.create(name, config)
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to config file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        # Check cache
        if str(config_path) in self._config_cache:
            return self._config_cache[str(config_path)].copy()
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        # Load based on extension
        if config_path.suffix in [".yaml", ".yml"]:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == ".json":
            with open(config_path, "r") as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        # Cache config
        self._config_cache[str(config_path)] = config
        
        return config.copy()
    
    def validate_config(self, config: Dict[str, Any], schema: Optional[Dict] = None) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration to validate
            schema: Validation schema (optional)
            
        Returns:
            True if valid
        """
        if schema is None:
            return True
        
        # Basic validation - can be extended with jsonschema
        for key, expected_type in schema.items():
            if key not in config:
                logger.warning(f"Missing config key: {key}")
                return False
            
            if not isinstance(config[key], expected_type):
                logger.warning(f"Invalid type for {key}: expected {expected_type}, got {type(config[key])}")
                return False
        
        return True

class ModelFactory(BaseFactory):
    """Factory for creating model instances."""
    
    def __init__(self):
        super().__init__(MODELS, "model")
        self.default_configs = self._load_default_configs()
    
    def _load_default_configs(self) -> Dict[str, Dict]:
        """Load default model configurations."""
        defaults = {
            "deberta": {
                "pretrained_model_name": "microsoft/deberta-v3-large",
                "num_labels": 4,
                "dropout_rate": 0.1,
            },
            "roberta": {
                "pretrained_model_name": "roberta-large",
                "num_labels": 4,
                "dropout_rate": 0.1,
            },
            "ensemble": {
                "models": ["deberta", "roberta", "xlnet"],
                "voting_type": "soft",
                "weights": None,
            }
        }
        return defaults
    
    def create(
        self, 
        name: str, 
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Create a model instance.
        
        Args:
            name: Model name
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Model instance
            
        Example:
            factory = ModelFactory()
            model = factory.create("deberta", {"num_labels": 4})
        """
        try:
            model_class = self.registry.get(name)
        except KeyError:
            raise ComponentNotFoundError(f"Model '{name}' not found in registry")
        
        # Merge configurations
        final_config = {}
        
        # Start with defaults
        if name in self.default_configs:
            final_config.update(self.default_configs[name])
        
        # Override with provided config
        if config:
            final_config.update(config)
        
        # Override with kwargs
        final_config.update(kwargs)
        
        logger.info(f"Creating model: {name} with config: {final_config}")
        
        return model_class(**final_config)
    
    def create_ensemble(
        self,
        model_names: list,
        ensemble_type: str = "voting",
        **kwargs
    ) -> Any:
        """
        Create an ensemble of models.
        
        Args:
            model_names: List of model names
            ensemble_type: Type of ensemble
            **kwargs: Additional arguments
            
        Returns:
            Ensemble model instance
        """
        ensemble_class = ENSEMBLES.get(ensemble_type)
        
        # Create individual models
        models = []
        for model_name in model_names:
            if isinstance(model_name, dict):
                name = model_name.pop("name")
                model = self.create(name, model_name)
            else:
                model = self.create(model_name)
            models.append(model)
        
        return ensemble_class(models=models, **kwargs)

class TrainerFactory(BaseFactory):
    """Factory for creating trainer instances."""
    
    def __init__(self):
        super().__init__(TRAINERS, "trainer")
    
    def create(
        self,
        name: str,
        model: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Create a trainer instance.
        
        Args:
            name: Trainer name
            model: Model to train
            config: Training configuration
            **kwargs: Additional arguments
            
        Returns:
            Trainer instance
        """
        try:
            trainer_class = self.registry.get(name)
        except KeyError:
            raise ComponentNotFoundError(f"Trainer '{name}' not found in registry")
        
        final_config = config or {}
        final_config.update(kwargs)
        
        logger.info(f"Creating trainer: {name}")
        
        return trainer_class(model=model, **final_config)

class DatasetFactory(BaseFactory):
    """Factory for creating dataset instances."""
    
    def __init__(self):
        super().__init__(DATASETS, "dataset")
    
    def create(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Create a dataset instance.
        
        Args:
            name: Dataset name
            config: Dataset configuration
            **kwargs: Additional arguments
            
        Returns:
            Dataset instance
        """
        try:
            dataset_class = self.registry.get(name)
        except KeyError:
            raise ComponentNotFoundError(f"Dataset '{name}' not found in registry")
        
        final_config = config or {}
        final_config.update(kwargs)
        
        logger.info(f"Creating dataset: {name}")
        
        return dataset_class(**final_config)

class OptimizerFactory(BaseFactory):
    """Factory for creating optimizer instances."""
    
    def __init__(self):
        super().__init__(OPTIMIZERS, "optimizer")
    
    def create(
        self,
        name: str,
        model_parameters: Any,
        config: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Create an optimizer instance.
        
        Args:
            name: Optimizer name
            model_parameters: Model parameters to optimize
            config: Optimizer configuration
            **kwargs: Additional arguments
            
        Returns:
            Optimizer instance
        """
        try:
            optimizer_class = self.registry.get(name)
        except KeyError:
            raise ComponentNotFoundError(f"Optimizer '{name}' not found in registry")
        
        final_config = config or {}
        final_config.update(kwargs)
        
        logger.info(f"Creating optimizer: {name} with lr={final_config.get('lr', 'default')}")
        
        return optimizer_class(model_parameters, **final_config)

class ComponentFactory:
    """
    Main factory class that provides access to all component factories.
    
    This implements the Abstract Factory pattern.
    """
    
    def __init__(self):
        self.model_factory = ModelFactory()
        self.trainer_factory = TrainerFactory()
        self.dataset_factory = DatasetFactory()
        self.optimizer_factory = OptimizerFactory()
    
    def create_model(self, name: str, **kwargs) -> Any:
        """Create a model."""
        return self.model_factory.create(name, **kwargs)
    
    def create_trainer(self, name: str, model: Any, **kwargs) -> Any:
        """Create a trainer."""
        return self.trainer_factory.create(name, model, **kwargs)
    
    def create_dataset(self, name: str, **kwargs) -> Any:
        """Create a dataset."""
        return self.dataset_factory.create(name, **kwargs)
    
    def create_optimizer(self, name: str, params: Any, **kwargs) -> Any:
        """Create an optimizer."""
        return self.optimizer_factory.create(name, params, **kwargs)
    
    def create_pipeline(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Create a complete training pipeline from configuration.
        
        Args:
            config_path: Path to pipeline configuration
            
        Returns:
            Dictionary containing all pipeline components
        """
        config = self.model_factory.load_config(config_path)
        
        # Create components
        pipeline = {}
        
        # Dataset
        if "dataset" in config:
            dataset_config = config["dataset"]
            dataset_name = dataset_config.pop("name")
            pipeline["dataset"] = self.create_dataset(dataset_name, **dataset_config)
        
        # Model
        if "model" in config:
            model_config = config["model"]
            model_name = model_config.pop("name")
            pipeline["model"] = self.create_model(model_name, **model_config)
        
        # Optimizer
        if "optimizer" in config and "model" in pipeline:
            opt_config = config["optimizer"]
            opt_name = opt_config.pop("name")
            pipeline["optimizer"] = self.create_optimizer(
                opt_name, 
                pipeline["model"].parameters(),
                **opt_config
            )
        
        # Trainer
        if "trainer" in config and "model" in pipeline:
            trainer_config = config["trainer"]
            trainer_name = trainer_config.pop("name")
            pipeline["trainer"] = self.create_trainer(
                trainer_name,
                pipeline["model"],
                **trainer_config
            )
        
        logger.info(f"Created pipeline with components: {list(pipeline.keys())}")
        
        return pipeline

# Create global factory instance
factory = ComponentFactory()

# Export public API
__all__ = [
    "BaseFactory",
    "ModelFactory",
    "TrainerFactory",
    "DatasetFactory",
    "OptimizerFactory",
    "ComponentFactory",
    "factory",
]
