"""
GraphQL Type Definitions
================================================================================
This module defines GraphQL types for the AG News classification API,
implementing strongly-typed schema objects with validation and documentation.

Type definitions include:
- Object types for domain entities
- Input types for mutations
- Interface types for polymorphism
- Union types for result handling

References:
    - GraphQL Type System
    - Strawberry Type Documentation
    - Domain-Driven Design (Evans, 2003)

Author: Võ Hải Dũng
License: MIT
"""

import strawberry
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Enumerations
@strawberry.enum
class ModelType(Enum):
    """Available model types for classification."""
    DEBERTA_V3 = "deberta_v3"
    ROBERTA = "roberta"
    XLNET = "xlnet"
    ELECTRA = "electra"
    LONGFORMER = "longformer"
    GPT2 = "gpt2"
    T5 = "t5"
    ENSEMBLE = "ensemble"
    PROMPT_BASED = "prompt_based"

@strawberry.enum
class ClassLabel(Enum):
    """AG News classification labels."""
    WORLD = "World"
    SPORTS = "Sports"
    BUSINESS = "Business"
    SCITECH = "Sci/Tech"

@strawberry.enum
class TrainingStatus(Enum):
    """Training job status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@strawberry.enum
class DatasetSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

# Object Types
@strawberry.type
class Classification:
    """
    Text classification result type.
    
    Represents the result of classifying a text document
    into one of the AG News categories.
    """
    
    id: strawberry.ID
    text: str
    label: ClassLabel
    confidence: float
    model_type: ModelType
    processing_time_ms: float
    timestamp: datetime
    
    # Optional fields
    probabilities: Optional[Dict[str, float]] = None
    explanation: Optional[str] = None
    attention_weights: Optional[List[float]] = None
    
    @strawberry.field
    def is_high_confidence(self) -> bool:
        """Check if prediction has high confidence (>0.9)."""
        return self.confidence > 0.9
    
    @strawberry.field
    def top_k_labels(self, k: int = 3) -> List[Dict[str, float]]:
        """Get top-k predicted labels with probabilities."""
        if not self.probabilities:
            return []
        
        sorted_probs = sorted(
            self.probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {"label": label, "probability": prob}
            for label, prob in sorted_probs[:k]
        ]

@strawberry.type
class Model:
    """
    Machine learning model type.
    
    Represents a trained model available for classification.
    """
    
    id: strawberry.ID
    name: str
    type: ModelType
    version: str
    created_at: datetime
    updated_at: Optional[datetime]
    is_deployed: bool
    is_default: bool
    
    # Performance metrics
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    
    # Model metadata
    parameters: Optional[Dict[str, Any]] = None
    training_config: Optional[Dict[str, Any]] = None
    size_mb: Optional[float] = None
    
    @strawberry.field
    def performance_summary(self) -> str:
        """Get model performance summary."""
        if self.accuracy:
            return f"Accuracy: {self.accuracy:.2%}, F1: {self.f1_score:.2%}"
        return "Performance metrics not available"

@strawberry.type
class Dataset:
    """
    Dataset type for training and evaluation.
    
    Represents a dataset used for model training or evaluation.
    """
    
    id: strawberry.ID
    name: str
    description: Optional[str]
    split: DatasetSplit
    size: int
    created_at: datetime
    
    # Dataset statistics
    label_distribution: Optional[Dict[str, int]] = None
    avg_text_length: Optional[float] = None
    vocabulary_size: Optional[int] = None
    
    @strawberry.field
    def is_balanced(self) -> bool:
        """Check if dataset has balanced label distribution."""
        if not self.label_distribution:
            return False
        
        values = list(self.label_distribution.values())
        if not values:
            return False
        
        min_count = min(values)
        max_count = max(values)
        
        # Consider balanced if max/min ratio < 1.5
        return max_count / min_count < 1.5 if min_count > 0 else False

@strawberry.type
class Training:
    """
    Training job type.
    
    Represents a model training job with progress tracking.
    """
    
    id: strawberry.ID
    model_type: ModelType
    status: TrainingStatus
    started_at: datetime
    completed_at: Optional[datetime]
    
    # Training configuration
    epochs: int
    batch_size: int
    learning_rate: float
    dataset_id: strawberry.ID
    
    # Progress tracking
    current_epoch: Optional[int] = None
    current_loss: Optional[float] = None
    best_validation_score: Optional[float] = None
    
    # Results
    final_metrics: Optional[Dict[str, float]] = None
    model_id: Optional[strawberry.ID] = None
    error_message: Optional[str] = None
    
    @strawberry.field
    def progress_percentage(self) -> float:
        """Calculate training progress percentage."""
        if self.status == TrainingStatus.COMPLETED:
            return 100.0
        elif self.current_epoch and self.epochs:
            return (self.current_epoch / self.epochs) * 100
        return 0.0
    
    @strawberry.field
    def estimated_time_remaining(self) -> Optional[int]:
        """Estimate remaining time in seconds."""
        if self.status != TrainingStatus.RUNNING:
            return None
        
        if not self.current_epoch or not self.started_at:
            return None
        
        elapsed = (datetime.utcnow() - self.started_at).total_seconds()
        if self.current_epoch > 0:
            time_per_epoch = elapsed / self.current_epoch
            remaining_epochs = self.epochs - self.current_epoch
            return int(time_per_epoch * remaining_epochs)
        
        return None

@strawberry.type
class Metrics:
    """
    Performance metrics type.
    
    Represents various performance metrics for models and API.
    """
    
    timestamp: datetime
    
    # Model metrics
    model_accuracy: Optional[float] = None
    model_latency_ms: Optional[float] = None
    model_throughput: Optional[float] = None
    
    # API metrics
    api_requests_total: Optional[int] = None
    api_errors_total: Optional[int] = None
    api_latency_p50: Optional[float] = None
    api_latency_p95: Optional[float] = None
    api_latency_p99: Optional[float] = None
    
    # System metrics
    cpu_usage_percent: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    
    @strawberry.field
    def error_rate(self) -> Optional[float]:
        """Calculate API error rate."""
        if self.api_requests_total and self.api_requests_total > 0:
            return self.api_errors_total / self.api_requests_total
        return None

@strawberry.type
class Error:
    """
    Error type for error responses.
    
    Represents an error that occurred during operation execution.
    """
    
    code: str
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

# Input Types
@strawberry.input
class ClassificationInput:
    """Input type for classification requests."""
    
    text: str
    model_type: Optional[ModelType] = ModelType.ENSEMBLE
    return_probabilities: bool = True
    return_explanation: bool = False

@strawberry.input
class BatchClassificationInput:
    """Input type for batch classification requests."""
    
    texts: List[str]
    model_type: Optional[ModelType] = ModelType.ENSEMBLE
    return_probabilities: bool = True

@strawberry.input
class TrainingInput:
    """Input type for training requests."""
    
    model_type: ModelType
    dataset_id: strawberry.ID
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    validation_split: float = 0.2

@strawberry.input
class ModelDeployInput:
    """Input type for model deployment."""
    
    model_id: strawberry.ID
    make_default: bool = False
    replicas: int = 1

@strawberry.input
class DatasetUploadInput:
    """Input type for dataset upload."""
    
    name: str
    description: Optional[str]
    split: DatasetSplit
    format: str = "json"

# Union Types
ClassificationResult = Union[Classification, Error]
TrainingResult = Union[Training, Error]
ModelResult = Union[Model, Error]

# Interface Types
@strawberry.interface
class Node:
    """Node interface for Relay specification."""
    id: strawberry.ID

@strawberry.interface
class Timestamped:
    """Interface for timestamped entities."""
    created_at: datetime
    updated_at: Optional[datetime]

# Export types
__all__ = [
    # Enums
    "ModelType",
    "ClassLabel",
    "TrainingStatus",
    "DatasetSplit",
    
    # Object Types
    "Classification",
    "Model",
    "Dataset",
    "Training",
    "Metrics",
    "Error",
    
    # Input Types
    "ClassificationInput",
    "BatchClassificationInput",
    "TrainingInput",
    "ModelDeployInput",
    "DatasetUploadInput",
    
    # Union Types
    "ClassificationResult",
    "TrainingResult",
    "ModelResult",
    
    # Interfaces
    "Node",
    "Timestamped"
]
