"""
Training Service Module
=======================

Implements training orchestration service following patterns from:
- Martin (2017): "Clean Architecture"
- Kleppmann (2017): "Designing Data-Intensive Applications"
- Google (2017): "Rules of Machine Learning"

This service manages the complete training lifecycle including
experiment tracking, hyperparameter optimization, and distributed training.

Author: Võ Hải Dũng
License: MIT
"""

import logging
import json
import time
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import hashlib
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizer
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
import wandb
import optuna

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_seed
from src.utils.experiment_tracking import ExperimentTracker
from src.core.registry import ModelRegistry
from src.core.exceptions import TrainingError, ConfigurationError
from src.data.datasets.ag_news import AGNewsDataset
from src.services.data_service import DataService, DataServiceConfig
from src.training.trainers.base_trainer import BaseTrainer
from src.evaluation.metrics.classification_metrics import ClassificationMetrics
from configs.config_loader import ConfigLoader
from configs.constants import (
    AG_NEWS_CLASSES,
    MODEL_DIR,
    OUTPUT_DIR
)

logger = setup_logging(__name__)

class TrainingStatus(Enum):
    """Training status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class TrainingConfig:
    """Configuration for training service."""
    
    # Model configuration
    model_name: str = "deberta-v3"
    model_type: str = "transformer"
    num_labels: int = len(AG_NEWS_CLASSES)
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    
    # Optimization
    optimizer_name: str = "adamw"
    scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = torch.cuda.is_available()
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 3
    metric_for_best_model: str = "eval_f1"
    
    # Checkpointing
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Evaluation
    evaluation_strategy: str = "epoch"
    eval_steps: Optional[int] = None
    per_device_eval_batch_size: Optional[int] = None
    
    # Distributed training
    distributed: bool = False
    local_rank: int = -1
    world_size: int = 1
    
    # Experiment tracking
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    use_wandb: bool = False
    use_tensorboard: bool = True
    
    # Output
    output_dir: Path = OUTPUT_DIR / "training"
    logging_dir: Path = OUTPUT_DIR / "logs"
    
    # Reproducibility
    seed: int = 42
    
    def __post_init__(self):
        """Validate and process configuration."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.logging_dir, str):
            self.logging_dir = Path(self.logging_dir)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logging_dir.mkdir(parents=True, exist_ok=True)
        
        # Set per_device_eval_batch_size if not specified
        if self.per_device_eval_batch_size is None:
            self.per_device_eval_batch_size = self.batch_size * 2
        
        # Generate experiment name if not provided
        if self.experiment_name is None:
            self.experiment_name = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

@dataclass
class TrainingResult:
    """Training result container."""
    
    experiment_id: str
    model_name: str
    status: TrainingStatus
    metrics: Dict[str, float]
    best_checkpoint: Optional[Path] = None
    training_time: float = 0.0
    num_parameters: int = 0
    config: Optional[TrainingConfig] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "status": self.status.value,
            "metrics": self.metrics,
            "best_checkpoint": str(self.best_checkpoint) if self.best_checkpoint else None,
            "training_time": self.training_time,
            "num_parameters": self.num_parameters,
            "config": asdict(self.config) if self.config else None,
            "error_message": self.error_message,
            "metadata": self.metadata
        }

class ExperimentManager:
    """
    Experiment management following patterns from:
    - Zaharia et al. (2018): "Accelerating the Machine Learning Lifecycle with MLflow"
    """
    
    def __init__(self, base_dir: Path = OUTPUT_DIR / "experiments"):
        """
        Initialize experiment manager.
        
        Args:
            base_dir: Base directory for experiments
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments = {}
        self.active_experiments = {}
        
        # Load existing experiments
        self._load_experiments()
    
    def create_experiment(
        self,
        name: str,
        config: TrainingConfig,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Create new experiment.
        
        Args:
            name: Experiment name
            config: Training configuration
            tags: Optional tags
            
        Returns:
            Experiment ID
        """
        # Generate experiment ID
        exp_id = hashlib.md5(
            f"{name}_{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Create experiment directory
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment metadata
        experiment = {
            "id": exp_id,
            "name": name,
            "config": asdict(config),
            "tags": tags or [],
            "created_at": datetime.now().isoformat(),
            "status": TrainingStatus.PENDING.value,
            "metrics": {},
            "artifacts": []
        }
        
        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(experiment, f, indent=2)
        
        self.experiments[exp_id] = experiment
        
        logger.info(f"Created experiment {exp_id}: {name}")
        
        return exp_id
    
    def update_experiment(
        self,
        exp_id: str,
        status: Optional[TrainingStatus] = None,
        metrics: Optional[Dict[str, float]] = None,
        artifacts: Optional[List[str]] = None
    ):
        """Update experiment information."""
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        
        experiment = self.experiments[exp_id]
        
        if status:
            experiment["status"] = status.value
        
        if metrics:
            experiment["metrics"].update(metrics)
        
        if artifacts:
            experiment["artifacts"].extend(artifacts)
        
        experiment["updated_at"] = datetime.now().isoformat()
        
        # Save updated experiment
        exp_dir = self.base_dir / exp_id
        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(experiment, f, indent=2)
    
    def get_experiment(self, exp_id: str) -> Dict[str, Any]:
        """Get experiment information."""
        if exp_id not in self.experiments:
            raise ValueError(f"Experiment {exp_id} not found")
        return self.experiments[exp_id]
    
    def list_experiments(
        self,
        status: Optional[TrainingStatus] = None,
        tags: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """List experiments with optional filtering."""
        experiments = list(self.experiments.values())
        
        if status:
            experiments = [
                exp for exp in experiments
                if exp["status"] == status.value
            ]
        
        if tags:
            experiments = [
                exp for exp in experiments
                if any(tag in exp["tags"] for tag in tags)
            ]
        
        return experiments
    
    def _load_experiments(self):
        """Load existing experiments from disk."""
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir():
                exp_file = exp_dir / "experiment.json"
                if exp_file.exists():
                    with open(exp_file, "r") as f:
                        experiment = json.load(f)
                        self.experiments[experiment["id"]] = experiment

class TrainingService:
    """
    Main training service implementing patterns from:
    - Sculley et al. (2015): "Hidden Technical Debt in Machine Learning Systems"
    - Amershi et al. (2019): "Software Engineering for Machine Learning"
    """
    
    def __init__(self):
        """Initialize training service."""
        self.data_service = DataService()
        self.experiment_manager = ExperimentManager()
        self.model_registry = ModelRegistry()
        self.config_loader = ConfigLoader()
        
        # Active training jobs
        self.active_jobs = {}
        
        # Training history
        self.training_history = []
        
        # Statistics
        self.stats = {
            "total_experiments": 0,
            "successful_experiments": 0,
            "failed_experiments": 0,
            "total_training_time": 0.0,
            "models_trained": 0
        }
        
        logger.info("Training service initialized")
    
    def train_model(
        self,
        config: Union[TrainingConfig, Dict[str, Any], str],
        dataset: Optional[Any] = None,
        callbacks: Optional[List[Callable]] = None
    ) -> TrainingResult:
        """
        Train a model with given configuration.
        
        Args:
            config: Training configuration (object, dict, or path)
            dataset: Optional dataset (will load if not provided)
            callbacks: Optional training callbacks
            
        Returns:
            Training result
        """
        # Process configuration
        if isinstance(config, str):
            config = self.config_loader.load_training_config(config)
        elif isinstance(config, dict):
            config = TrainingConfig(**config)
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Create experiment
        exp_id = self.experiment_manager.create_experiment(
            config.experiment_name,
            config,
            config.tags
        )
        
        # Update statistics
        self.stats["total_experiments"] += 1
        
        try:
            # Update experiment status
            self.experiment_manager.update_experiment(
                exp_id,
                status=TrainingStatus.RUNNING
            )
            
            # Start training
            start_time = time.time()
            
            # Load data if not provided
            if dataset is None:
                dataset = self._prepare_datasets(config)
            
            # Initialize model
            model, tokenizer = self._initialize_model(config)
            
            # Create trainer
            trainer = self._create_trainer(
                model,
                tokenizer,
                dataset,
                config,
                callbacks
            )
            
            # Train model
            logger.info(f"Starting training for experiment {exp_id}")
            train_result = trainer.train()
            
            # Evaluate model
            eval_result = trainer.evaluate()
            
            # Save model
            best_checkpoint = config.output_dir / exp_id / "best_model"
            trainer.save_model(str(best_checkpoint))
            tokenizer.save_pretrained(str(best_checkpoint))
            
            # Calculate training time
            training_time = time.time() - start_time
            
            # Prepare metrics
            metrics = {
                **train_result.metrics,
                **eval_result,
                "training_time": training_time
            }
            
            # Update experiment
            self.experiment_manager.update_experiment(
                exp_id,
                status=TrainingStatus.COMPLETED,
                metrics=metrics,
                artifacts=[str(best_checkpoint)]
            )
            
            # Update statistics
            self.stats["successful_experiments"] += 1
            self.stats["total_training_time"] += training_time
            self.stats["models_trained"] += 1
            
            # Create result
            result = TrainingResult(
                experiment_id=exp_id,
                model_name=config.model_name,
                status=TrainingStatus.COMPLETED,
                metrics=metrics,
                best_checkpoint=best_checkpoint,
                training_time=training_time,
                num_parameters=sum(p.numel() for p in model.parameters()),
                config=config,
                metadata={
                    "device": str(trainer.args.device),
                    "n_gpu": trainer.args.n_gpu,
                    "train_samples": len(dataset["train"]),
                    "eval_samples": len(dataset["validation"])
                }
            )
            
            # Add to history
            self.training_history.append(result)
            
            logger.info(f"Training completed for experiment {exp_id}")
            
            return result
            
        except Exception as e:
            # Update experiment status
            self.experiment_manager.update_experiment(
                exp_id,
                status=TrainingStatus.FAILED
            )
            
            # Update statistics
            self.stats["failed_experiments"] += 1
            
            logger.error(f"Training failed for experiment {exp_id}: {str(e)}")
            
            # Create error result
            return TrainingResult(
                experiment_id=exp_id,
                model_name=config.model_name,
                status=TrainingStatus.FAILED,
                metrics={},
                error_message=str(e),
                config=config
            )
    
    def _prepare_datasets(self, config: TrainingConfig) -> Dict[str, Any]:
        """Prepare training and validation datasets."""
        # Load datasets
        train_dataset = self.data_service.load_dataset(
            "ag_news",
            split="train"
        )
        
        val_dataset = self.data_service.load_dataset(
            "ag_news",
            split="validation"
        )
        
        return {
            "train": train_dataset,
            "validation": val_dataset
        }
    
    def _initialize_model(
        self,
        config: TrainingConfig
    ) -> tuple:
        """Initialize model and tokenizer."""
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer
        )
        
        # Model name mapping
        model_mapping = {
            "deberta-v3": "microsoft/deberta-v3-base",
            "roberta-large": "roberta-large",
            "xlnet-large": "xlnet-large-cased"
        }
        
        model_name = model_mapping.get(config.model_name, config.model_name)
        
        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=config.num_labels
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        return model, tokenizer
    
    def _create_trainer(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        dataset: Dict[str, Any],
        config: TrainingConfig,
        callbacks: Optional[List[Callable]] = None
    ) -> Trainer:
        """Create Hugging Face Trainer."""
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(config.output_dir),
            num_train_epochs=config.num_epochs,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_dir=str(config.logging_dir),
            logging_steps=10,
            evaluation_strategy=config.evaluation_strategy,
            eval_steps=config.eval_steps,
            save_strategy=config.save_strategy,
            save_total_limit=config.save_total_limit,
            load_best_model_at_end=config.load_best_model_at_end,
            metric_for_best_model=config.metric_for_best_model,
            greater_is_better=True,
            fp16=config.fp16,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            seed=config.seed,
            report_to=["tensorboard"] if config.use_tensorboard else [],
            push_to_hub=False
        )
        
        # Compute metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="macro"),
                "precision": precision_recall_fscore_support(
                    labels, predictions, average="macro"
                )[0],
                "recall": precision_recall_fscore_support(
                    labels, predictions, average="macro"
                )[1]
            }
        
        # Callbacks
        trainer_callbacks = callbacks or []
        
        if config.early_stopping:
            trainer_callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=config.patience
                )
            )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=trainer_callbacks
        )
        
        return trainer
    
    def hyperparameter_search(
        self,
        base_config: TrainingConfig,
        search_space: Dict[str, Any],
        n_trials: int = 20,
        direction: str = "maximize",
        metric: str = "eval_f1"
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter optimization using Optuna.
        
        Implements optimization strategies from:
        - Bergstra & Bengio (2012): "Random Search for Hyper-Parameter Optimization"
        
        Args:
            base_config: Base training configuration
            search_space: Hyperparameter search space
            n_trials: Number of trials
            direction: Optimization direction
            metric: Metric to optimize
            
        Returns:
            Best hyperparameters and results
        """
        def objective(trial):
            # Sample hyperparameters
            config = TrainingConfig(
                **{
                    **asdict(base_config),
                    **{
                        key: trial.suggest_categorical(key, values)
                        if isinstance(values, list)
                        else trial.suggest_float(key, values[0], values[1], log=values[2])
                        if isinstance(values, tuple) and len(values) == 3
                        else trial.suggest_float(key, values[0], values[1])
                        if isinstance(values, tuple) and len(values) == 2
                        else values
                        for key, values in search_space.items()
                    }
                }
            )
            
            # Train model
            result = self.train_model(config)
            
            # Return metric
            return result.metrics.get(metric, 0.0)
        
        # Create study
        study = optuna.create_study(
            direction=direction,
            study_name=f"hyperparam_search_{base_config.model_name}"
        )
        
        # Optimize
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        logger.info(f"Best hyperparameters: {best_params}")
        logger.info(f"Best {metric}: {best_value}")
        
        return {
            "best_params": best_params,
            "best_value": best_value,
            "study": study
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training service statistics."""
        return {
            **self.stats,
            "active_jobs": len(self.active_jobs),
            "total_experiments": len(self.experiment_manager.experiments),
            "avg_training_time": (
                self.stats["total_training_time"] / 
                max(self.stats["successful_experiments"], 1)
            )
        }
    
    def get_training_history(
        self,
        limit: Optional[int] = None
    ) -> List[TrainingResult]:
        """Get training history."""
        history = self.training_history
        
        if limit:
            history = history[-limit:]
        
        return history

# Global service instance
_training_service = None

def get_training_service() -> TrainingService:
    """Get training service instance (singleton)."""
    global _training_service
    
    if _training_service is None:
        _training_service = TrainingService()
    
    return _training_service
