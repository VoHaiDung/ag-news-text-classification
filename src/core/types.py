from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import torch

@dataclass
class DataConfig:
    """Data configuration"""
    dataset_name: str = "ag_news"
    max_length: int = 512
    batch_size: int = 32
    num_workers: int = 4
    
@dataclass
class ModelConfig:
    """Model configuration"""
    model_name: str
    num_labels: int = 4
    dropout: float = 0.1
    hidden_size: Optional[int] = None
    
@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 3
    learning_rate: float = 2e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    name: str
    seed: int = 42
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
