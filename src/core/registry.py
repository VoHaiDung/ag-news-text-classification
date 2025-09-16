"""
Model and Component Registry System
====================================

This module implements a centralized registry pattern for managing models, trainers,
data processors, and other components in a pluggable architecture.

The registry pattern is based on design principles from:
- Gamma et al. (1994): "Design Patterns: Elements of Reusable Object-Oriented Software"
- Martin (2003): "Agile Software Development, Principles, Patterns, and Practices"
- Fowler (2002): "Patterns of Enterprise Application Architecture"

The implementation follows dependency injection principles for testability and
modularity as described in:
- Seemann (2011): "Dependency Injection in .NET"
- Prasanna (2009): "Dependency Injection: Design Patterns Using Spring and Guice"

Author: Võ Hải Dũng
License: MIT
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Optional, Type, Union
from functools import wraps
from collections import defaultdict
import importlib
from pathlib import Path

logger = logging.getLogger(__name__)

class Registry:
    """
    A registry for storing and retrieving components by name.
    
    This implements the Registry pattern (Fowler, 2002) for dependency injection
    and plugin architecture support. The pattern provides:
    
    1. Decoupling: Components don't need to know about concrete implementations
    2. Extensibility: New components can be added without modifying existing code
    3. Testability: Components can be easily mocked for testing
    
    The design follows the Open/Closed Principle (Meyer, 1988) and 
    Dependency Inversion Principle (Martin, 2003).
    
    Example:
        >>> registry = Registry("models")
        >>> @registry.register("bert")
        ... class BertModel:
        ...     pass
        >>> model = registry.create("bert")
    
    References:
        Fowler (2002): "Patterns of Enterprise Application Architecture", Chapter 18
        Martin (2003): "Agile Software Development", Chapter 11
    """
    
    def __init__(self, name: str):
        """
        Initialize registry.
        
        Args:
            name: Name of the registry (e.g., "models", "trainers")
        """
        self.name = name
        self._registry: Dict[str, Any] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._aliases: Dict[str, str] = {}
        logger.debug(f"Created registry: {name}")
    
    def register(
        self,
        name: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        override: bool = False,
        **metadata
    ) -> Callable:
        """
        Decorator to register a class or function.
        
        Args:
            name: Registration name (defaults to class/function name)
            aliases: Alternative names for the component
            override: Whether to override existing registration
            **metadata: Additional metadata to store
            
        Returns:
            Decorator function
            
        Example:
            @registry.register("deberta", aliases=["deberta-v3"])
            class DeBERTaModel:
                pass
        """
        def decorator(cls_or_func: Union[Type, Callable]) -> Union[Type, Callable]:
            # Get registration name
            reg_name = name or cls_or_func.__name__.lower()
            
            # Check for existing registration
            if reg_name in self._registry and not override:
                raise ValueError(
                    f"Component '{reg_name}' already registered in {self.name} registry. "
                    f"Use override=True to replace."
                )
            
            # Register component
            self._registry[reg_name] = cls_or_func
            
            # Store metadata
            self._metadata[reg_name] = {
                "module": cls_or_func.__module__,
                "qualname": cls_or_func.__qualname__,
                "doc": cls_or_func.__doc__,
                "type": "class" if inspect.isclass(cls_or_func) else "function",
                **metadata
            }
            
            # Register aliases
            if aliases:
                for alias in aliases:
                    if alias in self._aliases and not override:
                        logger.warning(f"Alias '{alias}' already exists, skipping")
                        continue
                    self._aliases[alias] = reg_name
            
            logger.debug(f"Registered {reg_name} in {self.name} registry")
            return cls_or_func
        
        return decorator
    
    def get(self, name: str) -> Any:
        """
        Get a registered component by name.
        
        Args:
            name: Component name or alias
            
        Returns:
            Registered component
            
        Raises:
            KeyError: If component not found
        """
        # Resolve alias if needed
        resolved_name = self._aliases.get(name, name)
        
        if resolved_name not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            raise KeyError(
                f"Component '{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        
        return self._registry[resolved_name]
    
    def create(self, name: str, *args, **kwargs) -> Any:
        """
        Create an instance of a registered class.
        
        Args:
            name: Component name
            *args: Positional arguments for constructor
            **kwargs: Keyword arguments for constructor
            
        Returns:
            Instance of the component
        """
        cls = self.get(name)
        
        if not inspect.isclass(cls):
            raise TypeError(f"Component '{name}' is not a class")
        
        return cls(*args, **kwargs)
    
    def list(self) -> List[str]:
        """List all registered component names."""
        return sorted(self._registry.keys())
    
    def list_with_aliases(self) -> Dict[str, List[str]]:
        """List all components with their aliases."""
        result = defaultdict(list)
        for alias, name in self._aliases.items():
            result[name].append(alias)
        return dict(result)
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a component."""
        resolved_name = self._aliases.get(name, name)
        return self._metadata.get(resolved_name, {})
    
    def __contains__(self, name: str) -> bool:
        """Check if component is registered."""
        return name in self._registry or name in self._aliases
    
    def __len__(self) -> int:
        """Get number of registered components."""
        return len(self._registry)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Registry(name='{self.name}', components={len(self)})"
    
    def remove(self, name: str) -> Any:
        """
        Remove a component from registry.
        
        Args:
            name: Component name
            
        Returns:
            Removed component
        """
        resolved_name = self._aliases.get(name, name)
        
        if resolved_name not in self._registry:
            raise KeyError(f"Component '{name}' not found")
        
        # Remove component
        component = self._registry.pop(resolved_name)
        self._metadata.pop(resolved_name, None)
        
        # Remove aliases
        aliases_to_remove = [
            alias for alias, target in self._aliases.items()
            if target == resolved_name
        ]
        for alias in aliases_to_remove:
            del self._aliases[alias]
        
        logger.debug(f"Removed {resolved_name} from {self.name} registry")
        return component
    
    def clear(self):
        """Clear all registrations."""
        self._registry.clear()
        self._metadata.clear()
        self._aliases.clear()
        logger.debug(f"Cleared {self.name} registry")
    
    def update(self, other: "Registry"):
        """
        Update registry with components from another registry.
        
        Args:
            other: Another registry to merge from
        """
        self._registry.update(other._registry)
        self._metadata.update(other._metadata)
        self._aliases.update(other._aliases)
    
    def auto_discover(self, module_path: str, pattern: str = "*.py"):
        """
        Auto-discover and register components from modules.
        
        Args:
            module_path: Path to modules directory
            pattern: File pattern to match
        """
        module_dir = Path(module_path)
        
        if not module_dir.exists():
            logger.warning(f"Module path {module_path} does not exist")
            return
        
        for file_path in module_dir.glob(pattern):
            if file_path.name.startswith("_"):
                continue
                
            # Import module
            module_name = file_path.stem
            try:
                module = importlib.import_module(f"{module_path.replace('/', '.')}.{module_name}")
                logger.debug(f"Auto-discovered module: {module_name}")
            except Exception as e:
                logger.error(f"Failed to import {module_name}: {e}")
                continue

class RegistryHolder:
    """
    Singleton holder for all registries in the system.
    
    This provides a centralized access point for all component registries.
    """
    
    _instance = None
    _registries: Dict[str, Registry] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def get_registry(cls, name: str) -> Registry:
        """
        Get or create a registry by name.
        
        Args:
            name: Registry name
            
        Returns:
            Registry instance
        """
        if name not in cls._registries:
            cls._registries[name] = Registry(name)
        return cls._registries[name]
    
    @classmethod
    def list_registries(cls) -> List[str]:
        """List all registry names."""
        return sorted(cls._registries.keys())
    
    @classmethod
    def clear_all(cls):
        """Clear all registries."""
        for registry in cls._registries.values():
            registry.clear()
        logger.info("Cleared all registries")

# Create default registries
MODELS = RegistryHolder.get_registry("models")
TRAINERS = RegistryHolder.get_registry("trainers")
DATASETS = RegistryHolder.get_registry("datasets")
METRICS = RegistryHolder.get_registry("metrics")
OPTIMIZERS = RegistryHolder.get_registry("optimizers")
SCHEDULERS = RegistryHolder.get_registry("schedulers")
LOSSES = RegistryHolder.get_registry("losses")
AUGMENTERS = RegistryHolder.get_registry("augmenters")
CALLBACKS = RegistryHolder.get_registry("callbacks")
EVALUATORS = RegistryHolder.get_registry("evaluators")
INTERPRETERS = RegistryHolder.get_registry("interpreters")
ENSEMBLES = RegistryHolder.get_registry("ensembles")
PROCESSORS = RegistryHolder.get_registry("processors")
SAMPLERS = RegistryHolder.get_registry("samplers")
PREDICTORS = RegistryHolder.get_registry("predictors")

# Export public API
__all__ = [
    "Registry",
    "RegistryHolder",
    "MODELS",
    "TRAINERS",
    "DATASETS",
    "METRICS",
    "OPTIMIZERS",
    "SCHEDULERS",
    "LOSSES",
    "AUGMENTERS",
    "CALLBACKS",
    "EVALUATORS",
    "INTERPRETERS",
    "ENSEMBLES",
    "PROCESSORS",
    "SAMPLERS",
    "PREDICTORS",
]
