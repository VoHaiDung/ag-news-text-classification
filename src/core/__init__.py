import abc
import inspect
import warnings
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union,
    TypeVar, Generic, Protocol, runtime_checkable, cast, overload
)
from functools import wraps, lru_cache
import importlib
import pkgutil
import threading
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
ModelT = TypeVar('ModelT')
DatasetT = TypeVar('DatasetT')
TrainerT = TypeVar('TrainerT')

__all__ = [
    # Exceptions
    'AGNewsException',
    'ConfigurationError',
    'DataError',
    'ModelError',
    'TrainingError',
    'InferenceError',
    'APIError',
    'RegistryError',
    'FactoryError',
    'ValidationError',
    
    # Registry
    'Registry',
    'GlobalRegistry',
    'register_model',
    'register_dataset',
    'register_trainer',
    'register_augmenter',
    'register_metric',
    'register_callback',
    'get_model_class',
    'get_dataset_class',
    'get_trainer_class',
    
    # Factory
    'ModelFactory',
    'DatasetFactory',
    'TrainerFactory',
    'AugmenterFactory',
    'MetricFactory',
    'CallbackFactory',
    'create_model',
    'create_dataset',
    'create_trainer',
    
    # Types
    'ModelType',
    'DatasetType',
    'TrainerType',
    'TaskType',
    'PredictionOutput',
    'EvaluationMetrics',
    'ExperimentConfig',
    'TrainingState',
    'ModelConfig',
    'DataConfig',
    
    # Base classes
    'BaseModel',
    'BaseDataset',
    'BaseTrainer',
    'BaseAugmenter',
    'BaseMetric',
    'BaseCallback',
    'BaseFactory',
    'Configurable',
    'Serializable',
    
    # Utilities
    'SingletonMeta',
    'LazyLoader',
    'ComponentValidator',
    'DependencyInjector',
    'HookManager',
    'EventDispatcher',
    'ConfigManager',
    'CacheManager',
]


class AGNewsException(Exception):
    """Base exception for AG News framework."""
    pass


class ConfigurationError(AGNewsException):
    """Exception raised for configuration errors."""
    pass


class DataError(AGNewsException):
    """Exception raised for data-related errors."""
    pass


class ModelError(AGNewsException):
    """Exception raised for model-related errors."""
    pass


class TrainingError(AGNewsException):
    """Exception raised during training."""
    pass


class InferenceError(AGNewsException):
    """Exception raised during inference."""
    pass


class APIError(AGNewsException):
    """Exception raised for API-related errors."""
    pass


class RegistryError(AGNewsException):
    """Exception raised for registry operations."""
    pass


class FactoryError(AGNewsException):
    """Exception raised for factory operations."""
    pass


class ValidationError(AGNewsException):
    """Exception raised for validation errors."""
    pass


class ModelType(Enum):
    """Enumeration of model types."""
    TRANSFORMER = auto()
    ENSEMBLE = auto()
    PROMPT_BASED = auto()
    EFFICIENT = auto()
    CLASSICAL = auto()
    CUSTOM = auto()


class DatasetType(Enum):
    """Enumeration of dataset types."""
    AG_NEWS = auto()
    EXTERNAL = auto()
    COMBINED = auto()
    PROMPTED = auto()
    CONTRAST = auto()
    AUGMENTED = auto()
    CUSTOM = auto()


class TrainerType(Enum):
    """Enumeration of trainer types."""
    STANDARD = auto()
    DISTRIBUTED = auto()
    PROMPT = auto()
    INSTRUCTION = auto()
    MULTISTAGE = auto()
    ADVERSARIAL = auto()
    CUSTOM = auto()


class TaskType(Enum):
    """Enumeration of task types."""
    CLASSIFICATION = auto()
    MULTI_LABEL = auto()
    HIERARCHICAL = auto()
    MULTITASK = auto()
    GENERATION = auto()


@dataclass
class PredictionOutput:
    """Standard output format for predictions."""
    predictions: List[int]
    probabilities: Optional[List[List[float]]] = None
    labels: Optional[List[str]] = None
    embeddings: Optional[List[List[float]]] = None
    attention_weights: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PredictionOutput':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class EvaluationMetrics:
    """Standard format for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    f1_macro: Optional[float] = None
    f1_micro: Optional[float] = None
    f1_weighted: Optional[float] = None
    matthews_corrcoef: Optional[float] = None
    cohen_kappa: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    class_wise_metrics: Optional[Dict[str, Dict[str, float]]] = None
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def summary(self) -> str:
        """Get summary string."""
        return (
            f"Accuracy: {self.accuracy:.4f}, "
            f"Precision: {self.precision:.4f}, "
            f"Recall: {self.recall:.4f}, "
            f"F1: {self.f1_score:.4f}"
        )


@dataclass
class ExperimentConfig:
    """Configuration for experiments."""
    name: str
    model_config: Dict[str, Any]
    data_config: Dict[str, Any]
    training_config: Dict[str, Any]
    evaluation_config: Dict[str, Any] = field(default_factory=dict)
    environment_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_fields = ['name', 'model_config', 'data_config', 'training_config']
        for field_name in required_fields:
            if not getattr(self, field_name):
                raise ValidationError(f"Missing required field: {field_name}")
        return True


@dataclass
class TrainingState:
    """State tracking for training process."""
    epoch: int = 0
    global_step: int = 0
    best_metric: float = 0.0
    best_model_path: Optional[str] = None
    training_loss: List[float] = field(default_factory=list)
    validation_loss: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    metrics_history: Dict[str, List[float]] = field(default_factory=dict)
    
    def update(self, **kwargs):
        """Update state with new values."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save(self, path: Path):
        """Save state to file."""
        import json
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'TrainingState':
        """Load state from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class ModelConfig:
    """Configuration for models."""
    model_type: ModelType
    model_name: str
    num_classes: int
    pretrained: bool = True
    config_dict: Dict[str, Any] = field(default_factory=dict)
    
    def merge(self, other: Dict[str, Any]):
        """Merge with another configuration."""
        self.config_dict.update(other)


@dataclass
class DataConfig:
    """Configuration for datasets."""
    dataset_type: DatasetType
    dataset_name: str
    data_path: Optional[str] = None
    split: str = "train"
    batch_size: int = 32
    max_length: int = 512
    config_dict: Dict[str, Any] = field(default_factory=dict)


class SingletonMeta(type):
    """Metaclass for singleton pattern."""
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    instance = super().__call__(*args, **kwargs)
                    cls._instances[cls] = instance
        return cls._instances[cls]


class Registry(Generic[T]):
    """Generic registry for managing components."""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Type[T]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, str] = {}
        self._lock = threading.Lock()
        logger.debug(f"Created registry: {name}")
    
    def register(
        self,
        name: str,
        cls: Type[T],
        aliases: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        override: bool = False
    ) -> Type[T]:
        """Register a class with the registry."""
        with self._lock:
            if name in self._registry and not override:
                raise RegistryError(
                    f"'{name}' already registered in {self.name}. "
                    f"Use override=True to replace."
                )
            
            self._registry[name] = cls
            self._metadata[name] = metadata or {}
            
            if aliases:
                for alias in aliases:
                    self._aliases[alias] = name
            
            logger.debug(f"Registered {name} in {self.name} registry")
            return cls
    
    def get(self, name: str) -> Type[T]:
        """Get a registered class."""
        if name in self._aliases:
            name = self._aliases[name]
        
        if name not in self._registry:
            available = ', '.join(self._registry.keys())
            raise RegistryError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        
        return self._registry[name]
    
    def list(self) -> List[str]:
        """List all registered names."""
        return list(self._registry.keys())
    
    def contains(self, name: str) -> bool:
        """Check if name is registered."""
        return name in self._registry or name in self._aliases
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for registered component."""
        if name in self._aliases:
            name = self._aliases[name]
        return self._metadata.get(name, {})
    
    def remove(self, name: str) -> bool:
        """Remove a registered component."""
        with self._lock:
            if name in self._registry:
                del self._registry[name]
                if name in self._metadata:
                    del self._metadata[name]
                
                aliases_to_remove = [
                    alias for alias, target in self._aliases.items()
                    if target == name
                ]
                for alias in aliases_to_remove:
                    del self._aliases[alias]
                
                return True
        return False
    
    def clear(self):
        """Clear all registrations."""
        with self._lock:
            self._registry.clear()
            self._metadata.clear()
            self._aliases.clear()
    
    def decorator(
        self,
        name: Optional[str] = None,
        **kwargs
    ) -> Callable[[Type[T]], Type[T]]:
        """Decorator for registering classes."""
        def decorator_func(cls: Type[T]) -> Type[T]:
            reg_name = name or cls.__name__.lower()
            return self.register(reg_name, cls, **kwargs)
        return decorator_func


class GlobalRegistry(metaclass=SingletonMeta):
    """Global registry manager."""
    
    def __init__(self):
        self.models = Registry[Type]("models")
        self.datasets = Registry[Type]("datasets")
        self.trainers = Registry[Type]("trainers")
        self.augmenters = Registry[Type]("augmenters")
        self.metrics = Registry[Type]("metrics")
        self.callbacks = Registry[Type]("callbacks")
        self.optimizers = Registry[Type]("optimizers")
        self.schedulers = Registry[Type]("schedulers")
        self.losses = Registry[Type]("losses")
        self.transforms = Registry[Type]("transforms")
        self._custom_registries: Dict[str, Registry] = {}
    
    def create_registry(self, name: str) -> Registry:
        """Create a custom registry."""
        if name not in self._custom_registries:
            self._custom_registries[name] = Registry(name)
        return self._custom_registries[name]
    
    def get_registry(self, name: str) -> Registry:
        """Get a registry by name."""
        if hasattr(self, name):
            return getattr(self, name)
        elif name in self._custom_registries:
            return self._custom_registries[name]
        else:
            raise RegistryError(f"Registry '{name}' not found")


_global_registry = GlobalRegistry()


def register_model(name: str = None, **kwargs):
    """Decorator to register a model class."""
    return _global_registry.models.decorator(name, **kwargs)


def register_dataset(name: str = None, **kwargs):
    """Decorator to register a dataset class."""
    return _global_registry.datasets.decorator(name, **kwargs)


def register_trainer(name: str = None, **kwargs):
    """Decorator to register a trainer class."""
    return _global_registry.trainers.decorator(name, **kwargs)


def register_augmenter(name: str = None, **kwargs):
    """Decorator to register an augmenter class."""
    return _global_registry.augmenters.decorator(name, **kwargs)


def register_metric(name: str = None, **kwargs):
    """Decorator to register a metric class."""
    return _global_registry.metrics.decorator(name, **kwargs)


def register_callback(name: str = None, **kwargs):
    """Decorator to register a callback class."""
    return _global_registry.callbacks.decorator(name, **kwargs)


def get_model_class(name: str) -> Type:
    """Get a registered model class."""
    return _global_registry.models.get(name)


def get_dataset_class(name: str) -> Type:
    """Get a registered dataset class."""
    return _global_registry.datasets.get(name)


def get_trainer_class(name: str) -> Type:
    """Get a registered trainer class."""
    return _global_registry.trainers.get(name)


class BaseFactory(abc.ABC, Generic[T]):
    """Base factory class for creating components."""
    
    def __init__(self, registry: Registry[Type[T]]):
        self.registry = registry
        self._cache: Dict[str, T] = {}
        self._config_validators: Dict[str, Callable] = {}
    
    @abc.abstractmethod
    def create(self, name: str, **kwargs) -> T:
        """Create an instance of the component."""
        pass
    
    def register_validator(self, name: str, validator: Callable):
        """Register a configuration validator."""
        self._config_validators[name] = validator
    
    def validate_config(self, name: str, config: Dict[str, Any]) -> bool:
        """Validate configuration for a component."""
        if name in self._config_validators:
            return self._config_validators[name](config)
        return True
    
    def create_with_cache(self, name: str, cache_key: str = None, **kwargs) -> T:
        """Create with caching support."""
        cache_key = cache_key or f"{name}_{hash(frozenset(kwargs.items()))}"
        
        if cache_key not in self._cache:
            self._cache[cache_key] = self.create(name, **kwargs)
        
        return self._cache[cache_key]
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()


class ModelFactory(BaseFactory[ModelT]):
    """Factory for creating model instances."""
    
    def __init__(self):
        super().__init__(_global_registry.models)
    
    def create(self, name: str, **kwargs) -> ModelT:
        """Create a model instance."""
        if not self.validate_config(name, kwargs):
            raise ValidationError(f"Invalid configuration for model: {name}")
        
        model_class = self.registry.get(name)
        
        try:
            if 'config' in kwargs:
                model = model_class(config=kwargs['config'])
            else:
                model = model_class(**kwargs)
            
            logger.info(f"Created model: {name}")
            return model
        except Exception as e:
            raise FactoryError(f"Failed to create model '{name}': {e}")
    
    def create_ensemble(self, models: List[str], ensemble_type: str = "voting", **kwargs):
        """Create an ensemble of models."""
        model_instances = []
        for model_name in models:
            model_instances.append(self.create(model_name, **kwargs))
        
        ensemble_class = self.registry.get(ensemble_type)
        return ensemble_class(models=model_instances, **kwargs)


class DatasetFactory(BaseFactory[DatasetT]):
    """Factory for creating dataset instances."""
    
    def __init__(self):
        super().__init__(_global_registry.datasets)
    
    def create(self, name: str, **kwargs) -> DatasetT:
        """Create a dataset instance."""
        if not self.validate_config(name, kwargs):
            raise ValidationError(f"Invalid configuration for dataset: {name}")
        
        dataset_class = self.registry.get(name)
        
        try:
            dataset = dataset_class(**kwargs)
            logger.info(f"Created dataset: {name}")
            return dataset
        except Exception as e:
            raise FactoryError(f"Failed to create dataset '{name}': {e}")
    
    def create_combined(self, datasets: List[str], **kwargs):
        """Create a combined dataset."""
        dataset_instances = []
        for dataset_name in datasets:
            dataset_instances.append(self.create(dataset_name, **kwargs))
        
        combined_class = self.registry.get("combined")
        return combined_class(datasets=dataset_instances, **kwargs)


class TrainerFactory(BaseFactory[TrainerT]):
    """Factory for creating trainer instances."""
    
    def __init__(self):
        super().__init__(_global_registry.trainers)
    
    def create(self, name: str, model=None, **kwargs) -> TrainerT:
        """Create a trainer instance."""
        if not self.validate_config(name, kwargs):
            raise ValidationError(f"Invalid configuration for trainer: {name}")
        
        trainer_class = self.registry.get(name)
        
        try:
            if model is not None:
                trainer = trainer_class(model=model, **kwargs)
            else:
                trainer = trainer_class(**kwargs)
            
            logger.info(f"Created trainer: {name}")
            return trainer
        except Exception as e:
            raise FactoryError(f"Failed to create trainer '{name}': {e}")


class AugmenterFactory(BaseFactory):
    """Factory for creating augmenter instances."""
    
    def __init__(self):
        super().__init__(_global_registry.augmenters)
    
    def create(self, name: str, **kwargs):
        """Create an augmenter instance."""
        augmenter_class = self.registry.get(name)
        return augmenter_class(**kwargs)
    
    def create_pipeline(self, augmenters: List[str], **kwargs):
        """Create an augmentation pipeline."""
        augmenter_instances = []
        for aug_name in augmenters:
            augmenter_instances.append(self.create(aug_name, **kwargs))
        
        from src.data.augmentation import AugmentationPipeline
        return AugmentationPipeline(augmenters=augmenter_instances)


class MetricFactory(BaseFactory):
    """Factory for creating metric instances."""
    
    def __init__(self):
        super().__init__(_global_registry.metrics)
    
    def create(self, name: str, **kwargs):
        """Create a metric instance."""
        metric_class = self.registry.get(name)
        return metric_class(**kwargs)


class CallbackFactory(BaseFactory):
    """Factory for creating callback instances."""
    
    def __init__(self):
        super().__init__(_global_registry.callbacks)
    
    def create(self, name: str, **kwargs):
        """Create a callback instance."""
        callback_class = self.registry.get(name)
        return callback_class(**kwargs)


_model_factory = ModelFactory()
_dataset_factory = DatasetFactory()
_trainer_factory = TrainerFactory()
_augmenter_factory = AugmenterFactory()
_metric_factory = MetricFactory()
_callback_factory = CallbackFactory()


def create_model(name: str, **kwargs):
    """Create a model instance."""
    return _model_factory.create(name, **kwargs)


def create_dataset(name: str, **kwargs):
    """Create a dataset instance."""
    return _dataset_factory.create(name, **kwargs)


def create_trainer(name: str, **kwargs):
    """Create a trainer instance."""
    return _trainer_factory.create(name, **kwargs)


@runtime_checkable
class Configurable(Protocol):
    """Protocol for configurable components."""
    
    def get_config(self) -> Dict[str, Any]:
        """Get configuration dictionary."""
        ...
    
    def from_config(self, config: Dict[str, Any]) -> None:
        """Load from configuration dictionary."""
        ...


@runtime_checkable
class Serializable(Protocol):
    """Protocol for serializable components."""
    
    def save(self, path: Path) -> None:
        """Save to file."""
        ...
    
    @classmethod
    def load(cls, path: Path) -> 'Serializable':
        """Load from file."""
        ...


class BaseModel(abc.ABC):
    """Base class for all models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._initialized = False
    
    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass."""
        pass
    
    @abc.abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        pass
    
    def save(self, path: Path):
        """Save model."""
        raise NotImplementedError
    
    @classmethod
    def load(cls, path: Path):
        """Load model."""
        raise NotImplementedError


class BaseDataset(abc.ABC):
    """Base class for all datasets."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abc.abstractmethod
    def __len__(self) -> int:
        """Get dataset length."""
        pass
    
    @abc.abstractmethod
    def __getitem__(self, idx: int):
        """Get item by index."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get dataset configuration."""
        return self.config


class BaseTrainer(abc.ABC):
    """Base class for all trainers."""
    
    def __init__(self, model: BaseModel, config: Optional[Dict[str, Any]] = None):
        self.model = model
        self.config = config or {}
        self.state = TrainingState()
    
    @abc.abstractmethod
    def train(self, train_dataset, val_dataset=None, **kwargs):
        """Train the model."""
        pass
    
    @abc.abstractmethod
    def evaluate(self, dataset, **kwargs) -> EvaluationMetrics:
        """Evaluate the model."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get trainer configuration."""
        return self.config


class BaseAugmenter(abc.ABC):
    """Base class for all augmenters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abc.abstractmethod
    def augment(self, text: str) -> List[str]:
        """Augment text."""
        pass
    
    def batch_augment(self, texts: List[str]) -> List[List[str]]:
        """Augment batch of texts."""
        return [self.augment(text) for text in texts]


class BaseMetric(abc.ABC):
    """Base class for all metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abc.abstractmethod
    def compute(self, predictions, labels) -> Dict[str, float]:
        """Compute metric."""
        pass
    
    def reset(self):
        """Reset metric state."""
        pass


class BaseCallback(abc.ABC):
    """Base class for all callbacks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    def on_train_begin(self, trainer, **kwargs):
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer, **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer, epoch, **kwargs):
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch, **kwargs):
        """Called at the end of an epoch."""
        pass
    
    def on_batch_begin(self, trainer, batch, **kwargs):
        """Called at the beginning of a batch."""
        pass
    
    def on_batch_end(self, trainer, batch, **kwargs):
        """Called at the end of a batch."""
        pass


class LazyLoader:
    """Lazy loader for modules."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self._module = None
    
    def __getattr__(self, name):
        if self._module is None:
            self._module = importlib.import_module(self.module_name)
        return getattr(self._module, name)


class ComponentValidator:
    """Validator for components."""
    
    @staticmethod
    def validate_model(model: BaseModel) -> bool:
        """Validate a model."""
        required_methods = ['forward', 'get_config']
        for method in required_methods:
            if not hasattr(model, method):
                raise ValidationError(f"Model missing required method: {method}")
        return True
    
    @staticmethod
    def validate_dataset(dataset: BaseDataset) -> bool:
        """Validate a dataset."""
        if len(dataset) == 0:
            warnings.warn("Dataset is empty")
        return True
    
    @staticmethod
    def validate_trainer(trainer: BaseTrainer) -> bool:
        """Validate a trainer."""
        if trainer.model is None:
            raise ValidationError("Trainer requires a model")
        return True


class DependencyInjector:
    """Simple dependency injection container."""
    
    def __init__(self):
        self._services = {}
        self._singletons = {}
    
    def register(self, name: str, factory: Callable, singleton: bool = False):
        """Register a service."""
        self._services[name] = (factory, singleton)
    
    def get(self, name: str, **kwargs):
        """Get a service instance."""
        if name not in self._services:
            raise KeyError(f"Service '{name}' not registered")
        
        factory, is_singleton = self._services[name]
        
        if is_singleton:
            if name not in self._singletons:
                self._singletons[name] = factory(**kwargs)
            return self._singletons[name]
        
        return factory(**kwargs)


class HookManager:
    """Manager for hooks and callbacks."""
    
    def __init__(self):
        self._hooks = defaultdict(list)
    
    def register_hook(self, event: str, callback: Callable):
        """Register a hook."""
        self._hooks[event].append(callback)
    
    def trigger_hook(self, event: str, *args, **kwargs):
        """Trigger all hooks for an event."""
        results = []
        for callback in self._hooks[event]:
            try:
                result = callback(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Hook failed for event '{event}': {e}")
        return results
    
    def remove_hook(self, event: str, callback: Callable):
        """Remove a hook."""
        if event in self._hooks and callback in self._hooks[event]:
            self._hooks[event].remove(callback)


class EventDispatcher:
    """Event dispatcher for pub-sub pattern."""
    
    def __init__(self):
        self._subscribers = defaultdict(list)
    
    def subscribe(self, event: str, handler: Callable):
        """Subscribe to an event."""
        self._subscribers[event].append(handler)
    
    def unsubscribe(self, event: str, handler: Callable):
        """Unsubscribe from an event."""
        if event in self._subscribers and handler in self._subscribers[event]:
            self._subscribers[event].remove(handler)
    
    def dispatch(self, event: str, data: Any = None):
        """Dispatch an event."""
        for handler in self._subscribers[event]:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler failed for '{event}': {e}")


class ConfigManager:
    """Manager for configuration."""
    
    def __init__(self):
        self._configs = {}
        self._validators = {}
    
    def register_config(self, name: str, config: Dict[str, Any], validator: Optional[Callable] = None):
        """Register a configuration."""
        if validator and not validator(config):
            raise ValidationError(f"Invalid configuration for '{name}'")
        
        self._configs[name] = config
        if validator:
            self._validators[name] = validator
    
    def get_config(self, name: str) -> Dict[str, Any]:
        """Get a configuration."""
        if name not in self._configs:
            raise KeyError(f"Configuration '{name}' not found")
        return self._configs[name].copy()
    
    def update_config(self, name: str, updates: Dict[str, Any]):
        """Update a configuration."""
        if name not in self._configs:
            raise KeyError(f"Configuration '{name}' not found")
        
        config = self._configs[name].copy()
        config.update(updates)
        
        if name in self._validators and not self._validators[name](config):
            raise ValidationError(f"Invalid configuration update for '{name}'")
        
        self._configs[name] = config


class CacheManager:
    """Manager for caching."""
    
    def __init__(self, max_size: int = 100):
        self._cache = OrderedDict()
        self.max_size = max_size
        self._lock = threading.Lock()
    
    def get(self, key: str, default=None):
        """Get a cached value."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return default
    
    def set(self, key: str, value: Any):
        """Set a cached value."""
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self.max_size:
                    self._cache.popitem(last=False)
            self._cache[key] = value
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self._cache.clear()
    
    @contextmanager
    def cached(self, key: str, factory: Callable):
        """Context manager for caching."""
        value = self.get(key)
        if value is None:
            value = factory()
            self.set(key, value)
        yield value


def auto_discover_components(package_name: str, registry: Registry):
    """Auto-discover and register components from a package."""
    try:
        package = importlib.import_module(package_name)
        
        for importer, modname, ispkg in pkgutil.walk_packages(
            package.__path__,
            prefix=package.__name__ + "."
        ):
            try:
                module = importlib.import_module(modname)
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isclass(obj) and not inspect.isabstract(obj):
                        if hasattr(obj, '__register__'):
                            registry.register(obj.__register__, obj)
                            logger.debug(f"Auto-registered {obj.__name__} from {modname}")
            except Exception as e:
                logger.warning(f"Failed to import {modname}: {e}")
    except Exception as e:
        logger.error(f"Failed to auto-discover from {package_name}: {e}")


_dependency_injector = DependencyInjector()
_hook_manager = HookManager()
_event_dispatcher = EventDispatcher()
_config_manager = ConfigManager()
_cache_manager = CacheManager(max_size=1000)
