"""
Core Module for AG News Text Classification
============================================

This module provides the foundational infrastructure for the AG News Text
Classification framework, including registry pattern, factory pattern,
type definitions, and base abstractions.

Project: AG News Text Classification (ag-news-text-classification)
Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT

Module Components:
    1. Registry System: Component registration and discovery
    2. Factory Pattern: Dynamic component instantiation
    3. Type System: Type definitions and protocols
    4. Base Classes: Abstract base classes for extensibility
    5. Utilities: Helper classes and decorators

Academic Rationale:
    The design follows established software engineering patterns from
    "Design Patterns: Elements of Reusable Object-Oriented Software"
    (Gamma et al., 1994) and Python-specific patterns from "Python in a
    Nutshell" (Martelli et al., 2017).
    
    Key design decisions:
    
    1. Registry Pattern:
        Enables dynamic component discovery and loose coupling, critical
        for extensible research frameworks (Fowler, 2002).
    
    2. Factory Pattern:
        Centralizes object creation logic and enables configuration-driven
        instantiation (Gamma et al., 1994).
    
    3. Protocol-Based Design:
        Leverages Python's structural subtyping (PEP 544) for flexible
        interfaces without inheritance constraints.
    
    4. Type Safety:
        Comprehensive type hints enable static analysis and IDE support,
        improving code quality and maintainability (PEP 484, 526).

Design Principles:
    1. Separation of Concerns: Each module has a single responsibility
    2. Open-Closed Principle: Open for extension, closed for modification
    3. Dependency Inversion: Depend on abstractions, not concretions
    4. Interface Segregation: Clients should not depend on unused interfaces
    5. Liskov Substitution: Subtypes must be substitutable for base types

Usage Examples:
    Register a custom model:
        >>> @register_model("my_custom_model")
        ... class MyCustomModel(BaseModel):
        ...     def forward(self, x):
        ...         return self.model(x)
    
    Create model via factory:
        >>> model = create_model("deberta_v3_large", num_classes=4)
    
    Use type-safe configuration:
        >>> config = ExperimentConfig(
        ...     name="experiment_1",
        ...     model_config={"model_type": ModelType.TRANSFORMER},
        ...     data_config={"dataset_type": DatasetType.AG_NEWS},
        ...     training_config={"batch_size": 32}
        ... )

References:
    - Gamma et al. (1994): "Design Patterns"
    - Fowler (2002): "Patterns of Enterprise Application Architecture"
    - Martelli et al. (2017): "Python in a Nutshell"
    - PEP 484: Type Hints
    - PEP 544: Protocols (Structural Subtyping)
    - PEP 526: Syntax for Variable Annotations
"""

from src.core import exceptions
from src.core import registry
from src.core import factory
from src.core import types
from src.core import interfaces

# Re-export commonly used exceptions
from src.core.exceptions import (
    AGNewsException,
    ConfigurationError,
    DataError,
    ModelError,
    TrainingError,
    InferenceError,
    APIError,
    RegistryError,
    FactoryError,
    ValidationError,
)

# Re-export registry components
from src.core.registry import (
    Registry,
    GlobalRegistry,
    get_global_registry,
)

# Re-export factory components
from src.core.factory import (
    BaseFactory,
    ModelFactory,
    DatasetFactory,
    TrainerFactory,
    get_model_factory,
    get_dataset_factory,
    get_trainer_factory,
)

# Re-export type definitions
from src.core.types import (
    ModelType,
    DatasetType,
    TrainerType,
    TaskType,
    PredictionOutput,
    EvaluationMetrics,
    ExperimentConfig,
    TrainingState,
)

# Re-export interfaces
from src.core.interfaces import (
    Configurable,
    Serializable,
    Trainable,
    Evaluable,
)

# Package metadata
__all__ = [
    # Exceptions
    "exceptions",
    "AGNewsException",
    "ConfigurationError",
    "DataError",
    "ModelError",
    "TrainingError",
    "InferenceError",
    "APIError",
    "RegistryError",
    "FactoryError",
    "ValidationError",
    
    # Registry
    "registry",
    "Registry",
    "GlobalRegistry",
    "get_global_registry",
    
    # Factory
    "factory",
    "BaseFactory",
    "ModelFactory",
    "DatasetFactory",
    "TrainerFactory",
    "get_model_factory",
    "get_dataset_factory",
    "get_trainer_factory",
    
    # Types
    "types",
    "ModelType",
    "DatasetType",
    "TrainerType",
    "TaskType",
    "PredictionOutput",
    "EvaluationMetrics",
    "ExperimentConfig",
    "TrainingState",
    
    # Interfaces
    "interfaces",
    "Configurable",
    "Serializable",
    "Trainable",
    "Evaluable",
]

# Version compatibility check
import sys

if sys.version_info < (3, 8):
    raise RuntimeError(
        "AG News Text Classification requires Python 3.8 or higher. "
        f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
    )

# Module-level initialization
import logging

logger = logging.getLogger(__name__)
logger.debug("AG News Text Classification core module initialized")
