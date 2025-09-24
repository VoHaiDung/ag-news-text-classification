"""
Request Schemas
================================================================================
This module defines Pydantic schemas for API request validation following
OpenAPI 3.0 specification and JSON Schema standards.

Implements comprehensive request validation including:
- Type checking and coercion
- Field validation and constraints
- Custom validators for business logic
- Automatic documentation generation

References:
    - OpenAPI Specification v3.1.0
    - JSON Schema Draft 2020-12
    - Pydantic Documentation v2.0

Author: Võ Hải Dũng
License: MIT
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, validator, constr, conint
import numpy as np

class ModelType(str, Enum):
    """Enumeration of available model types."""
    DEBERTA_V3 = "deberta_v3"
    ROBERTA = "roberta"
    XLNET = "xlnet"
    ELECTRA = "electra"
    LONGFORMER = "longformer"
    GPT2 = "gpt2"
    T5 = "t5"
    ENSEMBLE = "ensemble"
    PROMPT_BASED = "prompt_based"

class DataFormat(str, Enum):
    """Enumeration of supported data formats."""
    JSON = "json"
    CSV = "csv"
    TEXT = "text"
    JSONL = "jsonl"

class TrainingStrategy(str, Enum):
    """Enumeration of training strategies."""
    STANDARD = "standard"
    ADVERSARIAL = "adversarial"
    CURRICULUM = "curriculum"
    DISTILLATION = "distillation"
    LORA = "lora"
    PROMPT_TUNING = "prompt_tuning"

class ClassificationRequest(BaseModel):
    """
    Schema for single text classification request.
    
    Attributes:
        text: Input text to classify
        model_type: Type of model to use
        return_probabilities: Whether to return class probabilities
        return_explanations: Whether to return explanations
        top_k: Number of top predictions to return
    """
    text: constr(min_length=1, max_length=10000) = Field(
        ...,
        description="Text to classify",
        example="The stock market reached new highs today..."
    )
    model_type: Optional[ModelType] = Field(
        ModelType.ENSEMBLE,
        description="Model type to use for classification"
    )
    return_probabilities: bool = Field(
        True,
        description="Return probability scores for each class"
    )
    return_explanations: bool = Field(
        False,
        description="Return model explanations"
    )
    top_k: conint(ge=1, le=10) = Field(
        1,
        description="Number of top predictions to return"
    )
    
    @validator('text')
    def validate_text(cls, v):
        """Validate input text."""
        if not v or v.isspace():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()
    
    class Config:
        """Pydantic configuration."""
        schema_extra = {
            "example": {
                "text": "Google announces new AI breakthrough in natural language processing",
                "model_type": "ensemble",
                "return_probabilities": True,
                "return_explanations": False,
                "top_k": 3
            }
        }

class BatchClassificationRequest(BaseModel):
    """
    Schema for batch text classification request.
    
    Attributes:
        texts: List of texts to classify
        model_type: Type of model to use
        batch_size: Processing batch size
        return_probabilities: Whether to return probabilities
        parallel_processing: Enable parallel processing
    """
    texts: List[constr(min_length=1, max_length=10000)] = Field(
        ...,
        min_items=1,
        max_items=1000,
        description="List of texts to classify"
    )
    model_type: Optional[ModelType] = Field(
        ModelType.ENSEMBLE,
        description="Model type to use"
    )
    batch_size: conint(ge=1, le=128) = Field(
        32,
        description="Batch size for processing"
    )
    return_probabilities: bool = Field(
        True,
        description="Return probability scores"
    )
    parallel_processing: bool = Field(
        True,
        description="Use parallel processing"
    )
    
    @validator('texts')
    def validate_texts(cls, v):
        """Validate text list."""
        cleaned_texts = []
        for text in v:
            if text and not text.isspace():
                cleaned_texts.append(text.strip())
        
        if not cleaned_texts:
            raise ValueError("No valid texts provided")
        
        return cleaned_texts

class TrainingRequest(BaseModel):
    """
    Schema for model training request.
    
    Attributes:
        model_type: Type of model to train
        strategy: Training strategy
        config: Training configuration
        dataset_path: Path to training data
        validation_split: Validation data percentage
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
    """
    model_type: ModelType = Field(
        ...,
        description="Model type to train"
    )
    strategy: TrainingStrategy = Field(
        TrainingStrategy.STANDARD,
        description="Training strategy to use"
    )
    config: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional training configuration"
    )
    dataset_path: Optional[str] = Field(
        None,
        description="Path to training dataset"
    )
    validation_split: float = Field(
        0.2,
        ge=0.0,
        le=0.5,
        description="Validation data split ratio"
    )
    epochs: conint(ge=1, le=100) = Field(
        10,
        description="Number of training epochs"
    )
    batch_size: conint(ge=1, le=256) = Field(
        32,
        description="Training batch size"
    )
    learning_rate: float = Field(
        2e-5,
        gt=0,
        le=1.0,
        description="Learning rate"
    )
    
    @validator('config')
    def validate_config(cls, v):
        """Validate training configuration."""
        if v and not isinstance(v, dict):
            raise ValueError("Config must be a dictionary")
        return v

class DataUploadRequest(BaseModel):
    """
    Schema for data upload request.
    
    Attributes:
        data_format: Format of uploaded data
        dataset_name: Name for the dataset
        description: Dataset description
        split_type: Type of data split
        auto_process: Automatically process data
    """
    data_format: DataFormat = Field(
        ...,
        description="Format of the uploaded data"
    )
    dataset_name: constr(min_length=1, max_length=100) = Field(
        ...,
        description="Name for the dataset"
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="Dataset description"
    )
    split_type: str = Field(
        "train",
        pattern="^(train|validation|test)$",
        description="Type of data split"
    )
    auto_process: bool = Field(
        True,
        description="Automatically preprocess data"
    )

class ModelDeployRequest(BaseModel):
    """
    Schema for model deployment request.
    
    Attributes:
        model_id: Model identifier
        deployment_name: Deployment name
        replicas: Number of replicas
        resource_config: Resource configuration
        auto_scaling: Enable auto-scaling
    """
    model_id: str = Field(
        ...,
        description="Model identifier to deploy"
    )
    deployment_name: constr(min_length=1, max_length=100) = Field(
        ...,
        description="Name for the deployment"
    )
    replicas: conint(ge=1, le=10) = Field(
        1,
        description="Number of replicas"
    )
    resource_config: Optional[Dict[str, Any]] = Field(
        None,
        description="Resource configuration"
    )
    auto_scaling: bool = Field(
        False,
        description="Enable auto-scaling"
    )

class ConfigUpdateRequest(BaseModel):
    """
    Schema for configuration update request.
    
    Attributes:
        config_section: Configuration section to update
        config_values: New configuration values
        apply_immediately: Apply changes immediately
    """
    config_section: str = Field(
        ...,
        description="Configuration section name"
    )
    config_values: Dict[str, Any] = Field(
        ...,
        description="Configuration values to update"
    )
    apply_immediately: bool = Field(
        True,
        description="Apply changes immediately"
    )

class UserManagementRequest(BaseModel):
    """
    Schema for user management request.
    
    Attributes:
        action: Management action
        username: Target username
        role: User role
        permissions: User permissions
    """
    action: str = Field(
        ...,
        pattern="^(create|update|delete|suspend)$",
        description="Management action"
    )
    username: constr(min_length=3, max_length=50) = Field(
        ...,
        description="Username"
    )
    role: Optional[str] = Field(
        "user",
        pattern="^(admin|user|viewer)$",
        description="User role"
    )
    permissions: Optional[List[str]] = Field(
        None,
        description="User permissions"
    )

class ServiceControlRequest(BaseModel):
    """
    Schema for service control request.
    
    Attributes:
        service_name: Name of service
        action: Control action
        graceful: Graceful shutdown/restart
        timeout: Action timeout
    """
    service_name: str = Field(
        ...,
        description="Service name"
    )
    action: str = Field(
        ...,
        pattern="^(start|stop|restart|status)$",
        description="Control action"
    )
    graceful: bool = Field(
        True,
        description="Perform graceful action"
    )
    timeout: conint(ge=1, le=300) = Field(
        30,
        description="Action timeout in seconds"
    )

class ExperimentRequest(BaseModel):
    """
    Schema for experiment request.
    
    Attributes:
        experiment_name: Name of experiment
        model_types: Models to compare
        metrics: Metrics to evaluate
        dataset: Dataset to use
        num_runs: Number of experimental runs
    """
    experiment_name: constr(min_length=1, max_length=100) = Field(
        ...,
        description="Experiment name"
    )
    model_types: List[ModelType] = Field(
        ...,
        min_items=1,
        description="Models to compare"
    )
    metrics: List[str] = Field(
        ["accuracy", "f1_score"],
        description="Metrics to evaluate"
    )
    dataset: str = Field(
        "ag_news",
        description="Dataset to use"
    )
    num_runs: conint(ge=1, le=10) = Field(
        3,
        description="Number of experimental runs"
    )

# Export schemas
__all__ = [
    "ModelType",
    "DataFormat",
    "TrainingStrategy",
    "ClassificationRequest",
    "BatchClassificationRequest",
    "TrainingRequest",
    "DataUploadRequest",
    "ModelDeployRequest",
    "ConfigUpdateRequest",
    "UserManagementRequest",
    "ServiceControlRequest",
    "ExperimentRequest"
]
