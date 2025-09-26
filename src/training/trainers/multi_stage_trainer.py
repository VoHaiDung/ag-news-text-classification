"""
Multi-Stage Trainer Implementation for AG News Text Classification
===================================================================

This module implements multi-stage training strategies for progressive model
refinement through curriculum learning, iterative training, and stage-wise optimization.

Key Techniques:
- Progressive training (Karras et al., 2018)
- Curriculum learning (Bengio et al., 2009)
- Stage-wise fine-tuning (Howard & Ruder, 2018)
- Iterative refinement (Xie et al., 2020)

References:
- Bengio et al. (2009): "Curriculum Learning"
- Howard & Ruder (2018): "Universal Language Model Fine-tuning for Text Classification"
- Karras et al. (2018): "Progressive Growing of GANs"
- Xie et al. (2020): "Self-training with Noisy Student"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np

from src.training.trainers.base_trainer import BaseTrainer, TrainerConfig
from src.training.trainers.standard_trainer import StandardTrainer
from src.training.trainers.distributed_trainer import DistributedTrainer
from src.models.base.base_model import AGNewsBaseModel
from src.training.strategies.curriculum.curriculum_learning import CurriculumStrategy
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TrainingStage(Enum):
    """Training stage definitions."""
    PRETRAINING = "pretraining"
    WARMUP = "warmup"
    MAIN = "main"
    FINETUNING = "finetuning"
    REFINEMENT = "refinement"
    DISTILLATION = "distillation"


@dataclass
class StageConfig:
    """Configuration for a single training stage."""
    
    name: str
    stage_type: TrainingStage
    num_epochs: int
    learning_rate: float
    batch_size: int
    
    # Data configuration
    data_fraction: float = 1.0
    use_augmentation: bool = False
    augmentation_prob: float = 0.0
    
    # Model configuration
    freeze_layers: List[str] = field(default_factory=list)
    unfreeze_layers: List[str] = field(default_factory=list)
    dropout_rate: float = 0.1
    
    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    warmup_steps: int = 0
    
    # Loss configuration
    loss_type: str = "cross_entropy"
    loss_weights: Dict[str, float] = field(default_factory=dict)
    
    # Regularization
    gradient_clipping: float = 1.0
    label_smoothing: float = 0.0
    
    # Stage-specific settings
    use_ema: bool = False
    use_swa: bool = False
    use_mixup: bool = False
    
    # Transition criteria
    min_epochs: int = 1
    max_epochs: Optional[int] = None
    early_stopping_patience: int = 3
    target_metric: str = "loss"
    target_threshold: Optional[float] = None


@dataclass
class MultiStageTrainerConfig(TrainerConfig):
    """Configuration for multi-stage trainer."""
    
    # Stage definitions
    stages: List[StageConfig] = field(default_factory=list)
    
    # Stage transition
    transition_strategy: str = "sequential"  # sequential, adaptive, cyclic
    stage_transition_smoothing: int = 0  # Epochs for smooth transition
    
    # Progressive training
    progressive_unfreezing: bool = True
    progressive_data_loading: bool = True
    progressive_batch_size: bool = False
    
    # Curriculum settings
    use_curriculum: bool = True
    curriculum_strategy: str = "difficulty"  # difficulty, confidence, diversity
    curriculum_pacing: str = "linear"  # linear, exponential, step
    
    # Iterative refinement
    iterative_refinement: bool = False
    refinement_iterations: int = 3
    pseudo_labeling: bool = False
    pseudo_label_threshold: float = 0.9
    
    # Knowledge accumulation
    knowledge_distillation: bool = False
    ensemble_teachers: bool = False
    student_teacher_ratio: float = 0.5
    
    # Checkpointing
    save_stage_checkpoints: bool = True
    checkpoint_best_stage_only: bool = False
    
    # Monitoring
    track_stage_metrics: bool = True
    log_stage_transitions: bool = True


class MultiStageTrainer:
    """
    Multi-stage trainer for progressive model training.
    
    Implements sophisticated training strategies including:
    - Stage-wise optimization with different configurations
    - Progressive unfreezing and data loading
    - Curriculum learning integration
    - Iterative refinement with pseudo-labeling
    """
    
    def __init__(
        self,
        model: AGNewsBaseModel,
        config: Optional[MultiStageTrainerConfig] = None,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None
    ):
        """
        Initialize multi-stage trainer.
        
        Args:
            model: Model to train
            config: Multi-stage training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
        """
        self.model = model
        self.config = config or self._create_default_config()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        # Initialize stages
        if not self.config.stages:
            self.config.stages = self._create_default_stages()
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Stage tracking
        self.current_stage_idx = 0
        self.stage_history = []
        self.best_metrics = {}
        
        # Initialize curriculum if enabled
        if self.config.use_curriculum:
            self.curriculum = CurriculumStrategy(
                strategy=self.config.curriculum_strategy,
                pacing=self.config.curriculum_pacing
            )
        
        # Teacher models for distillation
        self.teacher_models = []
        
        logger.info(
            f"Initialized MultiStageTrainer with {len(self.config.stages)} stages"
        )
    
    def _create_default_config(self) -> MultiStageTrainerConfig:
        """Create default multi-stage configuration."""
        return MultiStageTrainerConfig(
            stages=self._create_default_stages(),
            use_curriculum=True,
            progressive_unfreezing=True
        )
    
    def _create_default_stages(self) -> List[StageConfig]:
        """Create default training stages."""
        return [
            # Stage 1: Warmup
            StageConfig(
                name="warmup",
                stage_type=TrainingStage.WARMUP,
                num_epochs=2,
                learning_rate=1e-5,
                batch_size=16,
                data_fraction=0.1,
                freeze_layers=["encoder.layer.11", "encoder.layer.10"],
                warmup_steps=500
            ),
            # Stage 2: Main training
            StageConfig(
                name="main",
                stage_type=TrainingStage.MAIN,
                num_epochs=5,
                learning_rate=2e-5,
                batch_size=32,
                data_fraction=1.0,
                use_augmentation=True,
                augmentation_prob=0.1,
                unfreeze_layers=["encoder.layer.11", "encoder.layer.10"]
            ),
            # Stage 3: Fine-tuning
            StageConfig(
                name="finetuning",
                stage_type=TrainingStage.FINETUNING,
                num_epochs=3,
                learning_rate=5e-6,
                batch_size=32,
                data_fraction=1.0,
                use_mixup=True,
                label_smoothing=0.1,
                use_swa=True
            ),
            # Stage 4: Refinement
            StageConfig(
                name="refinement",
                stage_type=TrainingStage.REFINEMENT,
                num_epochs=2,
                learning_rate=1e-6,
                batch_size=64,
                data_fraction=1.0,
                use_ema=True,
                gradient_clipping=0.5
            )
        ]
    
    def train(self) -> Dict[str, Any]:
        """
        Execute multi-stage training.
        
        Returns:
            Training results across all stages
        """
        logger.info("Starting multi-stage training")
        
        overall_results = {
            "stages": [],
            "best_metrics": {},
            "total_epochs": 0
        }
        
        # Execute each stage
        for stage_idx, stage_config in enumerate(self.config.stages):
            self.current_stage_idx = stage_idx
            
            logger.info(
                f"Starting Stage {stage_idx + 1}/{len(self.config.stages)}: "
                f"{stage_config.name} ({stage_config.stage_type.value})"
            )
            
            # Prepare for stage
            self._prepare_stage(stage_config)
            
            # Execute stage training
            stage_results = self._train_stage(stage_config)
            
            # Post-process stage
            self._finalize_stage(stage_config, stage_results)
            
            # Record results
            overall_results["stages"].append({
                "name": stage_config.name,
                "results": stage_results
            })
            overall_results["total_epochs"] += stage_results.get("epochs", 0)
            
            # Check for early termination
            if self._should_terminate_training(stage_results):
                logger.info("Early termination triggered")
                break
            
            # Stage transition
            if stage_idx < len(self.config.stages) - 1:
                self._transition_stage(
                    stage_config,
                    self.config.stages[stage_idx + 1]
                )
        
        # Final refinement if enabled
        if self.config.iterative_refinement:
            refinement_results = self._iterative_refinement()
            overall_results["refinement"] = refinement_results
        
        # Update best metrics
        overall_results["best_metrics"] = self.best_metrics
        
        logger.info("Multi-stage training completed")
        return overall_results
    
    def _prepare_stage(self, stage_config: StageConfig):
        """
        Prepare model and data for stage.
        
        Args:
            stage_config: Stage configuration
        """
        # Freeze/unfreeze layers
        self._configure_model_layers(stage_config)
        
        # Adjust dropout
        self._set_dropout_rate(stage_config.dropout_rate)
        
        # Prepare data subset if needed
        if stage_config.data_fraction < 1.0:
            self._prepare_data_subset(stage_config.data_fraction)
        
        # Setup stage-specific components
        if stage_config.use_ema:
            self._setup_ema()
        
        if stage_config.use_swa:
            self._setup_swa()
    
    def _configure_model_layers(self, stage_config: StageConfig):
        """Configure model layer freezing/unfreezing."""
        # Freeze specified layers
        for layer_name in stage_config.freeze_layers:
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    param.requires_grad = False
                    logger.debug(f"Froze layer: {name}")
        
        # Unfreeze specified layers
        for layer_name in stage_config.unfreeze_layers:
            for name, param in self.model.named_parameters():
                if layer_name in name:
                    param.requires_grad = True
                    logger.debug(f"Unfroze layer: {name}")
        
        # Progressive unfreezing
        if self.config.progressive_unfreezing:
            self._progressive_unfreeze(self.current_stage_idx)
    
    def _progressive_unfreeze(self, stage_idx: int):
        """Progressively unfreeze layers based on stage."""
        # Unfreeze layers from top to bottom
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            num_layers = len(self.model.encoder.layer)
            layers_to_unfreeze = min(stage_idx + 1, num_layers)
            
            for i in range(num_layers - layers_to_unfreeze, num_layers):
                for param in self.model.encoder.layer[i].parameters():
                    param.requires_grad = True
            
            logger.info(f"Unfroze top {layers_to_unfreeze} layers")
    
    def _set_dropout_rate(self, dropout_rate: float):
        """Set dropout rate for all dropout layers."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.p = dropout_rate
    
    def _prepare_data_subset(self, fraction: float):
        """Prepare data subset for stage."""
        if self.train_dataset:
            total_size = len(self.train_dataset)
            subset_size = int(total_size * fraction)
            
            if self.config.use_curriculum:
                # Use curriculum to select samples
                indices = self.curriculum.select_samples(
                    self.train_dataset,
                    subset_size,
                    self.current_stage_idx
                )
            else:
                # Random selection
                indices = np.random.choice(total_size, subset_size, replace=False)
            
            self.stage_train_dataset = Subset(self.train_dataset, indices)
            logger.info(f"Using {subset_size}/{total_size} training samples")
        else:
            self.stage_train_dataset = self.train_dataset
    
    def _setup_ema(self):
        """Setup Exponential Moving Average."""
        from src.training.trainers.base_trainer import ExponentialMovingAverage
        self.ema = ExponentialMovingAverage(self.model, decay=0.999)
    
    def _setup_swa(self):
        """Setup Stochastic Weight Averaging."""
        from torch.optim.swa_utils import AveragedModel
        self.swa_model = AveragedModel(self.model)
    
    def _train_stage(self, stage_config: StageConfig) -> Dict[str, Any]:
        """
        Train a single stage.
        
        Args:
            stage_config: Stage configuration
            
        Returns:
            Stage training results
        """
        # Create stage-specific trainer
        trainer = self._create_stage_trainer(stage_config)
        
        # Train
        results = trainer.train(num_epochs=stage_config.num_epochs)
        
        # Extract best model if improved
        stage_metric = results.get("best_metric", 0)
        if self._is_best_stage(stage_config.name, stage_metric):
            self.best_metrics[stage_config.name] = stage_metric
            if self.config.save_stage_checkpoints:
                self._save_stage_checkpoint(stage_config, trainer)
        
        return results
    
    def _create_stage_trainer(self, stage_config: StageConfig) -> BaseTrainer:
        """Create trainer for specific stage."""
        # Create stage-specific data loader
        train_loader = DataLoader(
            self.stage_train_dataset or self.train_dataset,
            batch_size=stage_config.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=stage_config.batch_size,
            shuffle=False,
            num_workers=4
        ) if self.val_dataset else None
        
        # Create trainer config
        trainer_config = TrainerConfig(
            num_epochs=stage_config.num_epochs,
            learning_rate=stage_config.learning_rate,
            batch_size=stage_config.batch_size,
            weight_decay=stage_config.weight_decay,
            warmup_steps=stage_config.warmup_steps,
            max_grad_norm=stage_config.gradient_clipping,
            early_stopping_patience=stage_config.early_stopping_patience
        )
        
        # Create appropriate trainer
        if stage_config.stage_type == TrainingStage.DISTILLATION:
            # Use distillation trainer
            from src.training.trainers.standard_trainer import StandardTrainer
            trainer = StandardTrainer(
                model=self.model,
                config=trainer_config,
                train_loader=train_loader,
                val_loader=val_loader,
                teacher_model=self.teacher_models[-1] if self.teacher_models else None
            )
        else:
            # Use standard trainer
            trainer = StandardTrainer(
                model=self.model,
                config=trainer_config,
                train_loader=train_loader,
                val_loader=val_loader
            )
        
        return trainer
    
    def _is_best_stage(self, stage_name: str, metric: float) -> bool:
        """Check if current stage has best metric."""
        if stage_name not in self.best_metrics:
            return True
        return metric > self.best_metrics[stage_name]
    
    def _save_stage_checkpoint(self, stage_config: StageConfig, trainer: BaseTrainer):
        """Save checkpoint for stage."""
        checkpoint_path = f"checkpoints/stage_{stage_config.name}_best.pt"
        trainer.save_checkpoint(checkpoint_path, is_best=True)
        logger.info(f"Saved stage checkpoint: {checkpoint_path}")
    
    def _finalize_stage(self, stage_config: StageConfig, results: Dict[str, Any]):
        """
        Finalize stage after training.
        
        Args:
            stage_config: Stage configuration
            results: Stage results
        """
        # Update SWA model if used
        if hasattr(self, 'swa_model'):
            from torch.optim.swa_utils import update_bn
            update_bn(
                DataLoader(
                    self.stage_train_dataset or self.train_dataset,
                    batch_size=stage_config.batch_size
                ),
                self.swa_model,
                device=self.device
            )
            # Replace model with SWA model
            self.model = self.swa_model
        
        # Apply EMA if used
        if hasattr(self, 'ema'):
            self.ema.apply_shadow()
        
        # Store as teacher for distillation
        if self.config.knowledge_distillation:
            import copy
            teacher = copy.deepcopy(self.model)
            teacher.eval()
            self.teacher_models.append(teacher)
        
        # Record stage history
        self.stage_history.append({
            "stage": stage_config.name,
            "results": results
        })
    
    def _should_terminate_training(self, results: Dict[str, Any]) -> bool:
        """Check if training should be terminated early."""
        # Implement termination criteria
        if "error" in results:
            return True
        
        # Check if target metric reached
        for stage_config in self.config.stages[self.current_stage_idx:]:
            if stage_config.target_threshold:
                metric = results.get(stage_config.target_metric, float('inf'))
                if metric <= stage_config.target_threshold:
                    return False
        
        return False
    
    def _transition_stage(
        self,
        current_stage: StageConfig,
        next_stage: StageConfig
    ):
        """
        Handle transition between stages.
        
        Args:
            current_stage: Current stage configuration
            next_stage: Next stage configuration
        """
        logger.info(f"Transitioning from {current_stage.name} to {next_stage.name}")
        
        # Smooth transition if configured
        if self.config.stage_transition_smoothing > 0:
            self._smooth_transition(current_stage, next_stage)
        
        # Update learning rate
        self._adjust_learning_rate_transition(
            current_stage.learning_rate,
            next_stage.learning_rate
        )
        
        # Clear memory
        torch.cuda.empty_cache()
    
    def _smooth_transition(
        self,
        current_stage: StageConfig,
        next_stage: StageConfig
    ):
        """Implement smooth transition between stages."""
        # Gradually adjust hyperparameters
        transition_steps = self.config.stage_transition_smoothing
        
        for step in range(transition_steps):
            alpha = step / transition_steps
            
            # Interpolate learning rate
            lr = (1 - alpha) * current_stage.learning_rate + alpha * next_stage.learning_rate
            
            # Interpolate dropout
            dropout = (1 - alpha) * current_stage.dropout_rate + alpha * next_stage.dropout_rate
            self._set_dropout_rate(dropout)
            
            logger.debug(f"Transition step {step + 1}/{transition_steps}: lr={lr:.2e}, dropout={dropout:.3f}")
    
    def _adjust_learning_rate_transition(self, current_lr: float, next_lr: float):
        """Adjust learning rate during stage transition."""
        # Implement gradual learning rate change
        if hasattr(self, 'optimizer'):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = next_lr
    
    def _iterative_refinement(self) -> Dict[str, Any]:
        """
        Perform iterative refinement with pseudo-labeling.
        
        Returns:
            Refinement results
        """
        logger.info(f"Starting iterative refinement for {self.config.refinement_iterations} iterations")
        
        refinement_results = []
        
        for iteration in range(self.config.refinement_iterations):
            logger.info(f"Refinement iteration {iteration + 1}/{self.config.refinement_iterations}")
            
            # Generate pseudo labels if enabled
            if self.config.pseudo_labeling:
                pseudo_dataset = self._generate_pseudo_labels()
                
                # Combine with original dataset
                combined_dataset = self._combine_datasets(
                    self.train_dataset,
                    pseudo_dataset
                )
            else:
                combined_dataset = self.train_dataset
            
            # Create refinement stage
            refinement_stage = StageConfig(
                name=f"refinement_{iteration}",
                stage_type=TrainingStage.REFINEMENT,
                num_epochs=2,
                learning_rate=1e-6 * (0.5 ** iteration),  # Decay LR
                batch_size=64,
                use_ema=True
            )
            
            # Train refinement stage
            self.stage_train_dataset = combined_dataset
            results = self._train_stage(refinement_stage)
            
            refinement_results.append(results)
            
            # Early stop if no improvement
            if iteration > 0:
                if results.get("best_metric", 0) <= refinement_results[-2].get("best_metric", 0):
                    logger.info("No improvement in refinement, stopping")
                    break
        
        return {"iterations": refinement_results}
    
    def _generate_pseudo_labels(self) -> Any:
        """Generate pseudo labels for unlabeled data."""
        # This would generate pseudo labels using the current model
        # Implementation depends on specific dataset structure
        pass
    
    def _combine_datasets(self, dataset1: Any, dataset2: Any) -> Any:
        """Combine two datasets."""
        # Combine original and pseudo-labeled datasets
        # Implementation depends on specific dataset structure
        from torch.utils.data import ConcatDataset
        return ConcatDataset([dataset1, dataset2])
