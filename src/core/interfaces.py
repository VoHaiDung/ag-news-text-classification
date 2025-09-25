"""
Core Interfaces for AG News Text Classification System
================================================================================
This module defines abstract interfaces and protocols that establish contracts
between different components of the system. These interfaces ensure loose
coupling and enable dependency injection patterns.

The interface design follows SOLID principles, particularly the Interface
Segregation Principle (ISP) and Dependency Inversion Principle (DIP).

References:
    - Martin, R. C. (2017). Clean Architecture: A Craftsman's Guide to Software Structure and Design
    - Gamma, E., et al. (1994). Design Patterns: Elements of Reusable Object-Oriented Software
    - PEP 544 -- Protocols: Structural subtyping (static duck typing)

Author: Võ Hải Dũng
License: MIT
"""

from abc import ABC, abstractmethod
from typing import (
    Any, Dict, List, Optional, Union, Tuple, Iterator, 
    Protocol, TypeVar, Generic, Callable, Awaitable
)
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import torch
from pathlib import Path


# Type variables for generic interfaces
T = TypeVar('T')
ModelType = TypeVar('ModelType')
DataType = TypeVar('DataType')


class IModel(ABC, Generic[ModelType]):
    """
    Abstract interface for machine learning models
    
    This interface defines the contract that all models in the system must
    implement, ensuring consistent behavior across different model types.
    """
    
    @abstractmethod
    def load(self, path: Union[str, Path]) -> None:
        """
        Load model from disk
        
        Args:
            path: Path to model files
        """
        pass
    
    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk
        
        Args:
            path: Path to save model files
        """
        pass
    
    @abstractmethod
    def predict(self, inputs: DataType) -> Any:
        """
        Generate predictions for inputs
        
        Args:
            inputs: Input data
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration
        
        Returns:
            Configuration dictionary
        """
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Get model type identifier"""
        pass
    
    @property
    @abstractmethod
    def is_trained(self) -> bool:
        """Check if model is trained"""
        pass


class ITrainer(ABC):
    """
    Abstract interface for model trainers
    
    Defines the contract for training strategies, enabling different training
    approaches while maintaining a consistent interface.
    """
    
    @abstractmethod
    def train(self, 
              model: IModel,
              train_data: DataType,
              valid_data: Optional[DataType] = None,
              **kwargs) -> Dict[str, Any]:
        """
        Train a model
        
        Args:
            model: Model to train
            train_data: Training data
            valid_data: Validation data
            **kwargs: Additional training parameters
            
        Returns:
            Training results and metrics
        """
        pass
    
    @abstractmethod
    def evaluate(self, 
                 model: IModel,
                 data: DataType,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Model to evaluate
            data: Evaluation data
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation metrics
        """
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save training checkpoint"""
        pass
    
    @abstractmethod
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load training checkpoint"""
        pass


class IDataset(ABC, Generic[DataType]):
    """
    Abstract interface for datasets
    
    Provides a consistent interface for different dataset implementations,
    supporting both standard and custom data sources.
    """
    
    @abstractmethod
    def __len__(self) -> int:
        """Get dataset size"""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> DataType:
        """Get item by index"""
        pass
    
    @abstractmethod
    def get_labels(self) -> List[Any]:
        """Get all labels in dataset"""
        pass
    
    @property
    @abstractmethod
    def num_classes(self) -> int:
        """Get number of classes"""
        pass
    
    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        """Get class names"""
        pass


class IPreprocessor(ABC):
    """
    Abstract interface for data preprocessors
    
    Defines the contract for text preprocessing components, enabling
    modular and composable preprocessing pipelines.
    """
    
    @abstractmethod
    def process(self, text: str) -> str:
        """
        Process single text
        
        Args:
            text: Input text
            
        Returns:
            Processed text
        """
        pass
    
    @abstractmethod
    def process_batch(self, texts: List[str]) -> List[str]:
        """
        Process batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of processed texts
        """
        pass
    
    @abstractmethod
    def fit(self, texts: List[str]) -> None:
        """
        Fit preprocessor on texts
        
        Args:
            texts: Training texts
        """
        pass


class IAugmenter(ABC):
    """
    Abstract interface for data augmenters
    
    Provides a consistent interface for different augmentation techniques,
    supporting both deterministic and stochastic augmentations.
    """
    
    @abstractmethod
    def augment(self, 
                text: str,
                label: Optional[Any] = None,
                **kwargs) -> List[Tuple[str, Any]]:
        """
        Augment single sample
        
        Args:
            text: Input text
            label: Sample label
            **kwargs: Augmentation parameters
            
        Returns:
            List of augmented samples
        """
        pass
    
    @abstractmethod
    def augment_batch(self,
                      texts: List[str],
                      labels: Optional[List[Any]] = None,
                      **kwargs) -> Tuple[List[str], List[Any]]:
        """
        Augment batch of samples
        
        Args:
            texts: Input texts
            labels: Sample labels
            **kwargs: Augmentation parameters
            
        Returns:
            Augmented texts and labels
        """
        pass


class IEvaluator(ABC):
    """
    Abstract interface for model evaluators
    
    Defines the contract for evaluation strategies, supporting various
    metrics and evaluation protocols.
    """
    
    @abstractmethod
    def evaluate(self,
                 predictions: np.ndarray,
                 targets: np.ndarray,
                 **kwargs) -> Dict[str, float]:
        """
        Evaluate predictions
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            **kwargs: Additional parameters
            
        Returns:
            Evaluation metrics
        """
        pass
    
    @abstractmethod
    def get_confusion_matrix(self,
                            predictions: np.ndarray,
                            targets: np.ndarray) -> np.ndarray:
        """
        Generate confusion matrix
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Confusion matrix
        """
        pass


class ICache(ABC, Generic[T]):
    """
    Abstract interface for caching systems
    
    Provides a consistent interface for different caching backends,
    supporting both synchronous and asynchronous operations.
    """
    
    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        pass
    
    @abstractmethod
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete value from cache"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass


class IService(ABC):
    """
    Abstract interface for services
    
    Base interface for all service components in the system,
    ensuring consistent lifecycle management and health checking.
    """
    
    @abstractmethod
    async def start(self) -> None:
        """Start service"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop service"""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        
        Returns:
            Health status information
        """
        pass
    
    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if service is ready"""
        pass
    
    @property
    @abstractmethod
    def service_name(self) -> str:
        """Get service name"""
        pass


class IOrchestrator(ABC):
    """
    Abstract interface for workflow orchestrators
    
    Defines the contract for orchestration components that coordinate
    complex workflows and pipelines.
    """
    
    @abstractmethod
    async def execute_workflow(self,
                              workflow_id: str,
                              params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute workflow
        
        Args:
            workflow_id: Workflow identifier
            params: Workflow parameters
            
        Returns:
            Workflow execution results
        """
        pass
    
    @abstractmethod
    async def get_workflow_status(self, execution_id: str) -> Dict[str, Any]:
        """Get workflow execution status"""
        pass
    
    @abstractmethod
    async def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel workflow execution"""
        pass


class IMonitor(Protocol):
    """
    Protocol for monitoring components
    
    Uses Python's Protocol for structural subtyping, allowing any class
    that implements these methods to be used as a monitor.
    """
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric"""
        ...
    
    def record_event(self, event: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Record an event"""
        ...
    
    def create_alert(self, name: str, condition: str, severity: str) -> None:
        """Create an alert"""
        ...


class IStorage(ABC):
    """
    Abstract interface for storage systems
    
    Provides a consistent interface for different storage backends,
    supporting both local and cloud storage systems.
    """
    
    @abstractmethod
    def save(self, key: str, data: bytes) -> None:
        """Save data to storage"""
        pass
    
    @abstractmethod
    def load(self, key: str) -> bytes:
        """Load data from storage"""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete data from storage"""
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists"""
        pass
    
    @abstractmethod
    def list(self, prefix: str = "") -> List[str]:
        """List keys with prefix"""
        pass


# Export all interfaces
__all__ = [
    'IModel',
    'ITrainer',
    'IDataset',
    'IPreprocessor',
    'IAugmenter',
    'IEvaluator',
    'ICache',
    'IService',
    'IOrchestrator',
    'IMonitor',
    'IStorage'
]
