"""
Training Module for AG News Text Classification
================================================

This module provides comprehensive training functionality following best practices from:
- Goodfellow et al. (2016): "Deep Learning" - Training methodology
- Smith (2017): "A Disciplined Approach to Neural Network Hyper-Parameters"
- You et al. (2020): "Large Batch Optimization for Deep Learning"

The module implements various training paradigms:
1. Standard supervised training
2. Adversarial training for robustness
3. Curriculum learning for sample efficiency
4. Multi-task learning for knowledge transfer
5. Knowledge distillation for model compression

Design Principles:
- Modular architecture for easy extension
- Strategy pattern for interchangeable training methods
- Observer pattern for training callbacks
- Factory pattern for trainer creation

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Type, Any, Union, Callable
from pathlib import Path

# Import trainers
from src.training.trainers.base_trainer import (
    BaseTrainer,
    TrainerConfig,
    EarlyStopping,
    ExponentialMovingAverage
)

# Import training strategies (to be implemented)
try:
    from src.training.strategies.curriculum.curriculum_learning import (
        CurriculumLearning,
        CurriculumConfig,
        DifficultyScorer
    )
    from src.training.strategies.adversarial.fgm import FGMTrainer
    from src.training.strategies.adversarial.pgd import PGDTrainer
    from src.training.strategies.distillation.knowledge_distill import (
        KnowledgeDistillation,
        DistillationConfig
    )
    ADVANCED_STRATEGIES = True
except ImportError:
    ADVANCED_STRATEGIES = False

# Import callbacks
try:
    from src.training.callbacks.early_stopping import EarlyStoppingCallback
    from src.training.callbacks.model_checkpoint import ModelCheckpoint
    from src.training.callbacks.tensorboard_logger import TensorBoardLogger
    from src.training.callbacks.wandb_logger import WandBLogger
    from src.training.callbacks.learning_rate_monitor import LearningRateMonitor
    CALLBACKS_AVAILABLE = True
except ImportError:
    CALLBACKS_AVAILABLE = False

# Import optimization utilities
try:
    from src.training.optimization.optimizers.adamw_custom import AdamWCustom
    from src.training.optimization.schedulers.cosine_warmup import CosineWarmupScheduler
    from src.training.optimization.gradient.gradient_accumulation import GradientAccumulator
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Configure module logger
logger = logging.getLogger(__name__)

# Module version
__version__ = "1.0.0"

# ============================================================================
# Trainer Registry
# ============================================================================

class TrainerRegistry:
    """
    Registry for trainer implementations.
    
    Provides centralized management of available trainers and their configurations.
    """
    
    _trainers: Dict[str, Type[BaseTrainer]] = {}
    _configs: Dict[str, Type[TrainerConfig]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        trainer_class: Type[BaseTrainer],
        config_class: Optional[Type[TrainerConfig]] = None
    ):
        """
        Register a trainer implementation.
        
        Args:
            name: Trainer name
            trainer_class: Trainer class
            config_class: Configuration class
        """
        cls._trainers[name] = trainer_class
        if config_class:
            cls._configs[name] = config_class
        logger.debug(f"Registered trainer: {name}")
    
    @classmethod
    def get_trainer(
        cls,
        name: str,
        **kwargs
    ) -> BaseTrainer:
        """
        Get trainer instance by name.
        
        Args:
            name: Trainer name
            **kwargs: Trainer arguments
            
        Returns:
            Trainer instance
            
        Raises:
            KeyError: If trainer not found
        """
        if name not in cls._trainers:
            available = ", ".join(cls._trainers.keys())
            raise KeyError(
                f"Trainer '{name}' not found. Available trainers: {available}"
            )
        
        trainer_class = cls._trainers[name]
        config_class = cls._configs.get(name, TrainerConfig)
        
        # Create config if not provided
        if "config" not in kwargs:
            kwargs["config"] = config_class()
        
        return trainer_class(**kwargs)
    
    @classmethod
    def list_trainers(cls) -> List[str]:
        """List available trainers."""
        return list(cls._trainers.keys())


# Register default trainers
TrainerRegistry.register("base", BaseTrainer)

if ADVANCED_STRATEGIES:
    TrainerRegistry.register("fgm", FGMTrainer)
    TrainerRegistry.register("pgd", PGDTrainer)

# ============================================================================
# Training Utilities
# ============================================================================

def create_trainer(
    trainer_type: str = "base",
    model=None,
    config: Optional[TrainerConfig] = None,
    **kwargs
) -> BaseTrainer:
    """
    Create a trainer instance.
    
    This factory function provides a unified interface for creating different
    trainer types with appropriate configurations.
    
    Args:
        trainer_type: Type of trainer ("base", "adversarial", "curriculum")
        model: Model to train
        config: Training configuration
        **kwargs: Additional trainer arguments
        
    Returns:
        Configured trainer instance
        
    Example:
        >>> trainer = create_trainer(
        ...     "base",
        ...     model=model,
        ...     config=TrainerConfig(num_epochs=10)
        ... )
        >>> results = trainer.train()
    """
    return TrainerRegistry.get_trainer(
        trainer_type,
        model=model,
        config=config,
        **kwargs
    )


def create_training_config(
    strategy: str = "standard",
    **kwargs
) -> TrainerConfig:
    """
    Create training configuration based on strategy.
    
    Provides optimized configurations for different training strategies
    based on empirical results and best practices.
    
    Args:
        strategy: Training strategy name
        **kwargs: Configuration overrides
        
    Returns:
        Training configuration
        
    Training Strategies:
        - "standard": Basic supervised training
        - "robust": Adversarial training for robustness
        - "efficient": Fast training with mixed precision
        - "accurate": High accuracy with advanced techniques
        - "production": Production-ready configuration
    """
    configs = {
        "standard": {
            "num_epochs": 10,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "use_mixed_precision": False,
        },
        "robust": {
            "num_epochs": 15,
            "batch_size": 16,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "warmup_steps": 1000,
            "use_mixed_precision": True,
            "gradient_penalty": 0.5,
            "noise_std": 0.1,
            "use_ema": True,
        },
        "efficient": {
            "num_epochs": 5,
            "batch_size": 64,
            "learning_rate": 3e-5,
            "weight_decay": 0.01,
            "warmup_steps": 200,
            "use_mixed_precision": True,
            "gradient_accumulation_steps": 2,
        },
        "accurate": {
            "num_epochs": 20,
            "batch_size": 16,
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
            "warmup_steps": 2000,
            "use_mixed_precision": True,
            "use_swa": True,
            "use_ema": True,
            "label_smoothing": 0.1,
        },
        "production": {
            "num_epochs": 10,
            "batch_size": 32,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 500,
            "use_mixed_precision": True,
            "save_best_only": True,
            "early_stopping_patience": 3,
            "use_tensorboard": True,
        }
    }
    
    # Get base configuration
    base_config = configs.get(strategy, configs["standard"])
    
    # Apply overrides
    base_config.update(kwargs)
    
    return TrainerConfig(**base_config)


class TrainingPipeline:
    """
    End-to-end training pipeline with automatic configuration.
    
    Orchestrates the complete training process from data preparation
    to model evaluation and deployment.
    """
    
    def __init__(
        self,
        model,
        data_module,
        trainer_type: str = "base",
        config: Optional[TrainerConfig] = None
    ):
        """
        Initialize training pipeline.
        
        Args:
            model: Model to train
            data_module: Data module with train/val/test loaders
            trainer_type: Type of trainer to use
            config: Training configuration
        """
        self.model = model
        self.data_module = data_module
        self.trainer_type = trainer_type
        self.config = config or create_training_config("standard")
        
        # Create trainer
        self.trainer = create_trainer(
            trainer_type,
            model=model,
            config=config,
            train_loader=data_module.train_dataloader(),
            val_loader=data_module.val_dataloader()
        )
        
        logger.info(f"Initialized TrainingPipeline with {trainer_type} trainer")
    
    def run(
        self,
        num_epochs: Optional[int] = None,
        callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """
        Run complete training pipeline.
        
        Args:
            num_epochs: Number of epochs to train
            callbacks: Training callbacks
            
        Returns:
            Training results dictionary
        """
        logger.info("Starting training pipeline")
        
        # Add callbacks
        if callbacks:
            for callback in callbacks:
                self.trainer.add_callback(callback)
        
        # Train model
        results = self.trainer.train(num_epochs)
        
        # Evaluate on test set
        if hasattr(self.data_module, 'test_dataloader'):
            test_results = self.trainer.evaluate(
                self.data_module.test_dataloader()
            )
            results['test'] = test_results
        
        logger.info("Training pipeline completed")
        
        return results
    
    def save_model(self, path: Union[str, Path]):
        """Save trained model."""
        self.trainer.save_checkpoint(path)
    
    def load_model(self, path: Union[str, Path]):
        """Load model checkpoint."""
        self.trainer.load_checkpoint(path)


# ============================================================================
# Training Utilities
# ============================================================================

def get_parameter_groups(
    model,
    weight_decay: float = 0.01,
    learning_rate: float = 2e-5,
    layer_decay: Optional[float] = None
) -> List[Dict[str, Any]]:
    """
    Get parameter groups for optimizer with weight decay and layer-wise LR.
    
    Implements best practices for transformer fine-tuning:
    - No weight decay for bias and layer norm
    - Layer-wise learning rate decay
    - Different LR for different components
    
    Args:
        model: Model to get parameters from
        weight_decay: Weight decay coefficient
        learning_rate: Base learning rate
        layer_decay: Layer-wise LR decay factor
        
    Returns:
        List of parameter groups
    """
    # Parameters without weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    
    if layer_decay is None:
        # Simple grouping
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": weight_decay,
                "lr": learning_rate,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
                "lr": learning_rate,
            },
        ]
    else:
        # Layer-wise learning rate decay
        optimizer_grouped_parameters = []
        
        # Get layer groups if model supports it
        if hasattr(model, 'get_layer_parameters'):
            optimizer_grouped_parameters = model.get_layer_parameters()
        else:
            # Default layer-wise decay
            layers = {}
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Determine layer depth
                layer_id = 0
                if "layer" in name:
                    layer_id = int(name.split("layer.")[-1].split(".")[0])
                elif "encoder" in name:
                    layer_id = 1
                elif "embeddings" in name:
                    layer_id = 0
                else:
                    layer_id = 12  # Classifier layers
                
                if layer_id not in layers:
                    layers[layer_id] = {"params": [], "lr": learning_rate * (layer_decay ** (12 - layer_id))}
                
                layers[layer_id]["params"].append(param)
                
                # Set weight decay
                if not any(nd in name for nd in no_decay):
                    layers[layer_id]["weight_decay"] = weight_decay
                else:
                    layers[layer_id]["weight_decay"] = 0.0
            
            optimizer_grouped_parameters = list(layers.values())
    
    return optimizer_grouped_parameters


def compute_training_steps(
    num_samples: int,
    batch_size: int,
    num_epochs: int,
    gradient_accumulation_steps: int = 1
) -> Dict[str, int]:
    """
    Compute training step numbers for scheduling.
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        num_epochs: Number of epochs
        gradient_accumulation_steps: Gradient accumulation steps
        
    Returns:
        Dictionary with step calculations
    """
    steps_per_epoch = num_samples // (batch_size * gradient_accumulation_steps)
    total_steps = steps_per_epoch * num_epochs
    
    return {
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": min(total_steps // 10, 1000),
        "eval_steps": steps_per_epoch // 4,
        "save_steps": steps_per_epoch,
        "logging_steps": 50
    }


# ============================================================================
# Training Callbacks
# ============================================================================

class CallbackHandler:
    """
    Manages training callbacks for event handling.
    
    Implements the Observer pattern for training events.
    """
    
    def __init__(self, callbacks: Optional[List] = None):
        """
        Initialize callback handler.
        
        Args:
            callbacks: List of callback objects
        """
        self.callbacks = callbacks or []
    
    def add_callback(self, callback):
        """Add a callback."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback):
        """Remove a callback."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def on_train_begin(self, **kwargs):
        """Called at the beginning of training."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(**kwargs)
    
    def on_train_end(self, **kwargs):
        """Called at the end of training."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(**kwargs)
    
    def on_epoch_begin(self, epoch: int, **kwargs):
        """Called at the beginning of an epoch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch, **kwargs)
    
    def on_epoch_end(self, epoch: int, metrics: Dict, **kwargs):
        """Called at the end of an epoch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, metrics, **kwargs)
    
    def on_batch_begin(self, batch: int, **kwargs):
        """Called at the beginning of a batch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_begin'):
                callback.on_batch_begin(batch, **kwargs)
    
    def on_batch_end(self, batch: int, loss: float, **kwargs):
        """Called at the end of a batch."""
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_end'):
                callback.on_batch_end(batch, loss, **kwargs)


# ============================================================================
# Export Public API
# ============================================================================

__all__ = [
    # Core classes
    "BaseTrainer",
    "TrainerConfig",
    "TrainerRegistry",
    "TrainingPipeline",
    "CallbackHandler",
    
    # Training utilities
    "EarlyStopping",
    "ExponentialMovingAverage",
    
    # Factory functions
    "create_trainer",
    "create_training_config",
    
    # Utilities
    "get_parameter_groups",
    "compute_training_steps",
    
    # Version
    "__version__"
]

# Log module initialization
logger.info(f"Training module initialized (v{__version__})")
if ADVANCED_STRATEGIES:
    logger.info("Advanced training strategies available")
if CALLBACKS_AVAILABLE:
    logger.info("Training callbacks available")
if OPTIMIZATION_AVAILABLE:
    logger.info("Advanced optimization utilities available")
