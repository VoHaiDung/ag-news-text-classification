"""
Type definitions and data structures for AG News Text Classification.

This module provides type hints, data classes, and enums used throughout
the framework for better type safety and code documentation.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Union, Tuple, Callable,
    TypeVar, Generic, Protocol, Literal, TypedDict
)
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

# Type variables for generic types
T = TypeVar("T")
TensorType = TypeVar("TensorType", torch.Tensor, np.ndarray)
PathLike = Union[str, Path]

# ============================================================================
# Enums
# ============================================================================

class DatasetSplit(Enum):
    """Dataset split types."""
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"
    FULL = "full"

class ModelType(Enum):
    """Model architecture types."""
    TRANSFORMER = "transformer"
    ENSEMBLE = "ensemble"
    CLASSICAL = "classical"
    PROMPT_BASED = "prompt_based"
    EFFICIENT = "efficient"
    HYBRID = "hybrid"

class TaskType(Enum):
    """Task types for multi-task learning."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTI_LABEL = "multi_label"
    TOKEN_CLASSIFICATION = "token_classification"
    GENERATION = "generation"

class OptimizationStrategy(Enum):
    """Optimization strategies."""
    STANDARD = "standard"
    MIXED_PRECISION = "mixed_precision"
    GRADIENT_ACCUMULATION = "gradient_accumulation"
    DISTRIBUTED = "distributed"
    DEEPSPEED = "deepspeed"
    FAIRSCALE = "fairscale"

class AugmentationType(Enum):
    """Data augmentation types."""
    NONE = "none"
    SYNONYM_REPLACEMENT = "synonym_replacement"
    BACK_TRANSLATION = "back_translation"
    PARAPHRASE = "paraphrase"
    TOKEN_MANIPULATION = "token_manipulation"
    MIXUP = "mixup"
    ADVERSARIAL = "adversarial"
    CONTRAST_SET = "contrast_set"

class EnsembleMethod(Enum):
    """Ensemble combination methods."""
    VOTING = "voting"
    STACKING = "stacking"
    BLENDING = "blending"
    BAYESIAN = "bayesian"
    BOOSTING = "boosting"
    CASCADE = "cascade"

class MetricType(Enum):
    """Evaluation metric types."""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    F1_MACRO = "f1_macro"
    F1_MICRO = "f1_micro"
    F1_WEIGHTED = "f1_weighted"
    ROC_AUC = "roc_auc"
    MATTHEWS_CORRCOEF = "matthews_corrcoef"
    COHEN_KAPPA = "cohen_kappa"
    CONFUSION_MATRIX = "confusion_matrix"

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DataSample:
    """Single data sample."""
    text: str
    label: Optional[int] = None
    id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class BatchData:
    """Batch of data for training/inference."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None
    token_type_ids: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to(self, device: torch.device) -> "BatchData":
        """Move batch to device."""
        return BatchData(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            labels=self.labels.to(device) if self.labels is not None else None,
            token_type_ids=self.token_type_ids.to(device) if self.token_type_ids is not None else None,
            position_ids=self.position_ids.to(device) if self.position_ids is not None else None,
            metadata=self.metadata
        )
    
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self.input_ids.size(0)
    
    @property
    def seq_length(self) -> int:
        """Get sequence length."""
        return self.input_ids.size(1)

@dataclass
class ModelOutput:
    """Model output container."""
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.Tensor, ...]] = None
    attentions: Optional[Tuple[torch.Tensor, ...]] = None
    embeddings: Optional[torch.Tensor] = None
    predictions: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TrainingConfig:
    """Training configuration."""
    # Basic settings
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # Advanced settings
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    mixed_precision: bool = False
    gradient_checkpointing: bool = False
    
    # Optimizer settings
    optimizer_name: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Scheduler settings
    scheduler_name: str = "cosine"
    num_warmup_steps: int = 500
    
    # Evaluation settings
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    save_total_limit: int = 3
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True
    
    # Device settings
    device: str = "cuda"
    n_gpu: int = 1
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class ModelConfig:
    """Model configuration."""
    model_type: ModelType = ModelType.TRANSFORMER
    model_name: str = "microsoft/deberta-v3-large"
    num_labels: int = 4
    
    # Architecture settings
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    
    # Pooling settings
    pooling_strategy: str = "cls"
    use_hidden_states: bool = False
    hidden_states_layers: List[int] = field(default_factory=lambda: [-1])
    
    # Fine-tuning settings
    freeze_embeddings: bool = False
    freeze_encoder_layers: int = 0
    use_differential_lr: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

@dataclass
class EvaluationResults:
    """Evaluation results container."""
    metrics: Dict[str, float]
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_primary_metric(self, metric_name: str = "f1_macro") -> float:
        """Get primary metric value."""
        return self.metrics.get(metric_name, 0.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_name: str
    experiment_id: Optional[str] = None
    
    # Components
    model_config: ModelConfig = field(default_factory=ModelConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Data settings
    dataset_name: str = "ag_news"
    data_dir: PathLike = "./data"
    max_length: int = 512
    augmentation_config: Optional[Dict[str, Any]] = None
    
    # Output settings
    output_dir: PathLike = "./outputs"
    checkpoint_dir: Optional[PathLike] = None
    log_dir: Optional[PathLike] = None
    
    # Tracking
    use_wandb: bool = False
    use_mlflow: bool = False
    use_tensorboard: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_name": self.experiment_name,
            "experiment_id": self.experiment_id,
            "model_config": self.model_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "dataset_name": self.dataset_name,
            "data_dir": str(self.data_dir),
            "output_dir": str(self.output_dir),
            "seed": self.seed,
        }

# ============================================================================
# Protocols (Interfaces)
# ============================================================================

class DataProcessor(Protocol):
    """Protocol for data processors."""
    
    def process(self, text: str) -> Any:
        """Process text data."""
        ...
    
    def batch_process(self, texts: List[str]) -> Any:
        """Process batch of texts."""
        ...

class ModelInterface(Protocol):
    """Protocol for models."""
    
    def forward(self, batch: BatchData) -> ModelOutput:
        """Forward pass."""
        ...
    
    def predict(self, batch: BatchData) -> torch.Tensor:
        """Make predictions."""
        ...
    
    def save(self, path: PathLike) -> None:
        """Save model."""
        ...
    
    def load(self, path: PathLike) -> None:
        """Load model."""
        ...

class TrainerInterface(Protocol):
    """Protocol for trainers."""
    
    def train(self) -> Dict[str, Any]:
        """Train model."""
        ...
    
    def evaluate(self) -> EvaluationResults:
        """Evaluate model."""
        ...
    
    def save_checkpoint(self, path: PathLike) -> None:
        """Save checkpoint."""
        ...

# ============================================================================
# Type Aliases
# ============================================================================

# Common type aliases
Labels = Union[List[int], np.ndarray, torch.Tensor]
Texts = Union[List[str], List[List[str]]]
Predictions = Union[np.ndarray, torch.Tensor, List[int]]
Features = Union[np.ndarray, torch.Tensor, Dict[str, Any]]

# Configuration types
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, float]
HyperparamsDict = Dict[str, Union[int, float, str, bool]]

# Callback types
LossCallback = Callable[[float, int], None]
MetricCallback = Callable[[MetricsDict, int], None]
CheckpointCallback = Callable[[PathLike, int], None]

# ============================================================================
# TypedDict definitions for better IDE support
# ============================================================================

class DatasetInfo(TypedDict):
    """Dataset information."""
    name: str
    num_samples: int
    num_classes: int
    splits: List[str]
    label_names: List[str]
    metadata: Dict[str, Any]

class ModelInfo(TypedDict):
    """Model information."""
    name: str
    type: str
    num_parameters: int
    architecture: str
    pretrained: bool
    metadata: Dict[str, Any]

class TrainingState(TypedDict):
    """Training state."""
    epoch: int
    global_step: int
    best_metric: float
    best_model_path: str
    training_loss: float
    validation_loss: float
    learning_rate: float
    metadata: Dict[str, Any]

# ============================================================================
# Constants
# ============================================================================

# AG News class labels
AG_NEWS_LABELS = ["World", "Sports", "Business", "Sci/Tech"]
AG_NEWS_LABEL_IDS = {label: i for i, label in enumerate(AG_NEWS_LABELS)}

# Model size mappings
MODEL_SIZES = {
    "small": {"hidden_size": 256, "num_layers": 6, "num_heads": 4},
    "base": {"hidden_size": 768, "num_layers": 12, "num_heads": 12},
    "large": {"hidden_size": 1024, "num_layers": 24, "num_heads": 16},
    "xlarge": {"hidden_size": 1536, "num_layers": 48, "num_heads": 24},
}

# Default hyperparameters
DEFAULT_HYPERPARAMS = {
    "learning_rate": 2e-5,
    "batch_size": 32,
    "num_epochs": 10,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "max_length": 512,
}

# Export public API
__all__ = [
    # Enums
    "DatasetSplit",
    "ModelType",
    "TaskType",
    "OptimizationStrategy",
    "AugmentationType",
    "EnsembleMethod",
    "MetricType",
    # Data classes
    "DataSample",
    "BatchData",
    "ModelOutput",
    "TrainingConfig",
    "ModelConfig",
    "EvaluationResults",
    "ExperimentConfig",
    # Protocols
    "DataProcessor",
    "ModelInterface",
    "TrainerInterface",
    # Type aliases
    "TensorType",
    "PathLike",
    "Labels",
    "Texts",
    "Predictions",
    "Features",
    "ConfigDict",
    "MetricsDict",
    "HyperparamsDict",
    # TypedDicts
    "DatasetInfo",
    "ModelInfo",
    "TrainingState",
    # Constants
    "AG_NEWS_LABELS",
    "AG_NEWS_LABEL_IDS",
    "MODEL_SIZES",
    "DEFAULT_HYPERPARAMS",
]
