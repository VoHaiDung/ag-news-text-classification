"""
Multi-Stage Progressive Training Script for AG News Text Classification
========================================================================

This script implements multi-stage training strategies following:
- Bengio et al. (2009): "Curriculum Learning"
- Gong et al. (2019): "Efficient Training of BERT by Progressively Stacking"
- Zhang & He (2020): "Accelerating Training of Transformer-Based Language Models"

The multi-stage training implements:
1. Progressive unfreezing of layers
2. Gradual complexity increase
3. Stage-specific optimization
4. Transfer learning between stages

Mathematical Framework:
Stage k: θ_k = argmin L_k(θ_{k-1}) with progressive constraint relaxation.

Author: Võ Hải Dũng
License: MIT
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
from dataclasses import dataclass
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets.ag_news import create_ag_news_datasets
from src.training.trainers.multi_stage_trainer import MultiStageTrainer
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from src.utils.experiment_tracking import create_experiment, log_metrics
from src.utils.io_utils import safe_save, ensure_dir
from configs.constants import AG_NEWS_NUM_CLASSES, MODELS_DIR

logger = setup_logging(__name__)


@dataclass
class StageConfig:
    """
    Configuration for a single training stage.
    
    Based on progressive training strategies from:
    - Karras et al. (2018): "Progressive Growing of GANs"
    """
    
    name: str
    num_epochs: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    dropout_rate: float
    freeze_layers: List[str]
    unfreeze_layers: List[str]
    data_fraction: float
    batch_size: int
    gradient_accumulation_steps: int
    optimizer_type: str
    scheduler_type: str
    loss_type: str
    regularization_weight: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "dropout_rate": self.dropout_rate,
            "freeze_layers": self.freeze_layers,
            "unfreeze_layers": self.unfreeze_layers,
            "data_fraction": self.data_fraction,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer_type": self.optimizer_type,
            "scheduler_type": self.scheduler_type,
            "loss_type": self.loss_type,
            "regularization_weight": self.regularization_weight
        }


class ProgressiveTrainer:
    """
    Multi-stage progressive trainer.
    
    Implements progressive training strategies from:
    - Howard & Ruder (2018): "Universal Language Model Fine-tuning"
    - Peters et al. (2019): "To Tune or Not to Tune? Adapting Pretrained Representations"
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_dataset: Any,
        val_dataset: Any,
        stage_configs: List[StageConfig],
        device: torch.device,
        experiment_name: str
    ):
        """
        Initialize progressive trainer.
        
        Args:
            model: Model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            val_dataset: Validation dataset
            stage_configs: Configuration for each stage
            device: Compute device
            experiment_name: Experiment name
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.stage_configs = stage_configs
        self.device = device
        self.experiment_name = experiment_name
        
        # Training state
        self.current_stage = 0
        self.stage_results = []
        self.best_metric = 0.0
        self.best_stage = 0
        
        logger.info(f"Initialized progressive trainer with {len(stage_configs)} stages")
    
    def freeze_layers(self, layer_names: List[str]):
        """
        Freeze specified layers.
        
        Args:
            layer_names: Names of layers to freeze
        """
        for name, param in self.model.named_parameters():
            if any(layer in name for layer in layer_names):
                param.requires_grad = False
                logger.debug(f"Froze layer: {name}")
    
    def unfreeze_layers(self, layer_names: List[str]):
        """
        Unfreeze specified layers.
        
        Args:
            layer_names: Names of layers to unfreeze
        """
        if "all" in layer_names:
            for param in self.model.parameters():
                param.requires_grad = True
            logger.info("Unfroze all layers")
        else:
            for name, param in self.model.named_parameters():
                if any(layer in name for layer in layer_names):
                    param.requires_grad = True
                    logger.debug(f"Unfroze layer: {name}")
    
    def create_stage_dataloader(
        self,
        dataset: Any,
        stage_config: StageConfig,
        shuffle: bool = True
    ) -> DataLoader:
        """
        Create dataloader for specific stage.
        
        Args:
            dataset: Dataset to use
            stage_config: Stage configuration
            shuffle: Whether to shuffle data
            
        Returns:
            DataLoader for stage
        """
        # Use subset of data if specified
        if stage_config.data_fraction < 1.0:
            num_samples = int(len(dataset) * stage_config.data_fraction)
            indices = np.random.choice(len(dataset), num_samples, replace=False)
            dataset = Subset(dataset, indices)
            logger.info(f"Using {num_samples} samples ({stage_config.data_fraction:.1%})")
        
        dataloader = DataLoader(
            dataset,
            batch_size=stage_config.batch_size,
            shuffle=shuffle,
            num_workers=4,
            pin_memory=True
        )
        
        return dataloader
    
    def create_optimizer(self, stage_config: StageConfig) -> torch.optim.Optimizer:
        """
        Create optimizer for stage.
        
        Args:
            stage_config: Stage configuration
            
        Returns:
            Optimizer
        """
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if stage_config.optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=stage_config.learning_rate,
                weight_decay=stage_config.weight_decay
            )
        elif stage_config.optimizer_type == "sgd":
            optimizer = torch.optim.SGD(
                trainable_params,
                lr=stage_config.learning_rate,
                momentum=0.9,
                weight_decay=stage_config.weight_decay
            )
        elif stage_config.optimizer_type == "rmsprop":
            optimizer = torch.optim.RMSprop(
                trainable_params,
                lr=stage_config.learning_rate,
                weight_decay=stage_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {stage_config.optimizer_type}")
        
        return optimizer
    
    def create_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        stage_config: StageConfig,
        num_training_steps: int
    ):
        """
        Create learning rate scheduler for stage.
        
        Args:
            optimizer: Optimizer
            stage_config: Stage configuration
            num_training_steps: Total training steps
            
        Returns:
            Scheduler
        """
        num_warmup_steps = int(num_training_steps * stage_config.warmup_ratio)
        
        if stage_config.scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
        elif stage_config.scheduler_type == "polynomial":
            scheduler = get_polynomial_decay_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps,
                power=2.0
            )
        elif stage_config.scheduler_type == "constant":
            scheduler = None
        else:
            raise ValueError(f"Unknown scheduler: {stage_config.scheduler_type}")
        
        return scheduler
    
    def train_stage(self, stage_idx: int) -> Dict[str, Any]:
        """
        Train a single stage.
        
        Args:
            stage_idx: Stage index
            
        Returns:
            Stage results
        """
        stage_config = self.stage_configs[stage_idx]
        logger.info(f"\nStarting Stage {stage_idx + 1}: {stage_config.name}")
        
        # Configure model for stage
        self.freeze_layers(stage_config.freeze_layers)
        self.unfreeze_layers(stage_config.unfreeze_layers)
        
        # Update dropout if needed
        if hasattr(self.model, 'dropout'):
            for module in self.model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = stage_config.dropout_rate
        
        # Create data loaders
        train_loader = self.create_stage_dataloader(
            self.train_dataset,
            stage_config,
            shuffle=True
        )
        val_loader = self.create_stage_dataloader(
            self.val_dataset,
            stage_config,
            shuffle=False
        )
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(stage_config)
        num_training_steps = len(train_loader) * stage_config.num_epochs
        scheduler = self.create_scheduler(optimizer, stage_config, num_training_steps)
        
        # Training metrics
        stage_history = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1": []
        }
        
        best_stage_metric = 0.0
        
        # Train for stage epochs
        for epoch in range(1, stage_config.num_epochs + 1):
            # Training
            train_loss = self.train_epoch(
                train_loader,
                optimizer,
                scheduler,
                stage_config
            )
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # Update history
            stage_history["train_loss"].append(train_loss)
            stage_history["val_loss"].append(val_loss)
            stage_history["val_accuracy"].append(val_metrics["accuracy"])
            stage_history["val_f1"].append(val_metrics["f1_macro"])
            
            # Log
            logger.info(
                f"Stage {stage_idx + 1}, Epoch {epoch}/{stage_config.num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1_macro']:.4f}"
            )
            
            # Track best
            if val_metrics["f1_macro"] > best_stage_metric:
                best_stage_metric = val_metrics["f1_macro"]
                self.save_checkpoint(stage_idx, epoch)
        
        return {
            "stage_name": stage_config.name,
            "best_metric": best_stage_metric,
            "history": stage_history,
            "config": stage_config.to_dict()
        }
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        stage_config: StageConfig
    ) -> float:
        """
        Train for one epoch in current stage.
        
        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            stage_config: Stage configuration
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for step, batch in enumerate(progress_bar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss / stage_config.gradient_accumulation_steps
            
            # Add regularization if specified
            if stage_config.regularization_weight > 0:
                reg_loss = self.compute_regularization_loss(stage_config.loss_type)
                loss = loss + stage_config.regularization_weight * reg_loss
            
            # Backward pass
            loss.backward()
            
            if (step + 1) % stage_config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
            
            total_loss += loss.item() * stage_config.gradient_accumulation_steps
            num_batches += 1
            
            progress_bar.set_postfix({"loss": loss.item() * stage_config.gradient_accumulation_steps})
        
        return total_loss / num_batches
    
    def compute_regularization_loss(self, loss_type: str) -> torch.Tensor:
        """
        Compute regularization loss.
        
        Args:
            loss_type: Type of regularization
            
        Returns:
            Regularization loss
        """
        reg_loss = 0.0
        
        if loss_type == "l2":
            for param in self.model.parameters():
                if param.requires_grad:
                    reg_loss += torch.norm(param, p=2)
        elif loss_type == "l1":
            for param in self.model.parameters():
                if param.requires_grad:
                    reg_loss += torch.norm(param, p=1)
        
        return reg_loss
    
    def evaluate(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (loss, metrics)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        return avg_loss, {
            "accuracy": accuracy,
            "f1_macro": f1
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Complete multi-stage training.
        
        Returns:
            Training results
        """
        logger.info(f"Starting multi-stage training with {len(self.stage_configs)} stages")
        
        for stage_idx in range(len(self.stage_configs)):
            stage_results = self.train_stage(stage_idx)
            self.stage_results.append(stage_results)
            
            # Track best stage
            if stage_results["best_metric"] > self.best_metric:
                self.best_metric = stage_results["best_metric"]
                self.best_stage = stage_idx
        
        # Final results
        results = {
            "num_stages": len(self.stage_configs),
            "best_metric": self.best_metric,
            "best_stage": self.best_stage,
            "stage_results": self.stage_results
        }
        
        logger.info(f"\nMulti-stage training completed")
        logger.info(f"Best metric: {self.best_metric:.4f} from stage {self.best_stage + 1}")
        
        return results
    
    def save_checkpoint(self, stage_idx: int, epoch: int):
        """Save model checkpoint."""
        save_dir = Path(MODELS_DIR) / self.experiment_name / f"stage_{stage_idx}"
        ensure_dir(save_dir)
        
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            "stage": stage_idx,
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "stage_config": self.stage_configs[stage_idx].to_dict()
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")


def create_default_stages() -> List[StageConfig]:
    """
    Create default training stages for progressive training.
    
    Returns:
        List of stage configurations
    """
    stages = []
    
    # Stage 1: Frozen backbone, train classifier head
    stage1 = StageConfig(
        name="Classifier Head Training",
        num_epochs=2,
        learning_rate=5e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        dropout_rate=0.1,
        freeze_layers=["embeddings", "encoder"],
        unfreeze_layers=["classifier"],
        data_fraction=0.5,
        batch_size=64,
        gradient_accumulation_steps=1,
        optimizer_type="adamw",
        scheduler_type="linear",
        loss_type="l2",
        regularization_weight=0.01
    )
    stages.append(stage1)
    
    # Stage 2: Unfreeze top layers
    stage2 = StageConfig(
        name="Top Layer Fine-tuning",
        num_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        weight_decay=0.01,
        dropout_rate=0.2,
        freeze_layers=["embeddings", "encoder.layer.0", "encoder.layer.1"],
        unfreeze_layers=["encoder.layer.10", "encoder.layer.11", "classifier"],
        data_fraction=0.75,
        batch_size=32,
        gradient_accumulation_steps=2,
        optimizer_type="adamw",
        scheduler_type="polynomial",
        loss_type="l2",
        regularization_weight=0.005
    )
    stages.append(stage2)
    
    # Stage 3: Full model fine-tuning
    stage3 = StageConfig(
        name="Full Model Fine-tuning",
        num_epochs=5,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        dropout_rate=0.3,
        freeze_layers=[],
        unfreeze_layers=["all"],
        data_fraction=1.0,
        batch_size=16,
        gradient_accumulation_steps=4,
        optimizer_type="adamw",
        scheduler_type="linear",
        loss_type="l2",
        regularization_weight=0.001
    )
    stages.append(stage3)
    
    return stages


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-stage progressive training"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Base model"
    )
    parser.add_argument(
        "--stages-config",
        type=str,
        default=None,
        help="Path to stages configuration file"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    
    return parser.parse_args()


def main():
    """Main multi-stage training pipeline."""
    args = parse_arguments()
    
    # Setup
    logger.info("Setting up multi-stage training")
    device = torch.device(args.device)
    ensure_reproducibility(seed=42)
    
    # Generate experiment name
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"multi_stage_{timestamp}"
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=AG_NEWS_NUM_CLASSES
    )
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_ag_news_datasets(tokenizer=tokenizer)
    
    # Load or create stage configurations
    if args.stages_config:
        with open(args.stages_config, 'r') as f:
            stages_data = json.load(f)
        stage_configs = [StageConfig(**stage) for stage in stages_data]
    else:
        stage_configs = create_default_stages()
    
    # Initialize trainer
    trainer = ProgressiveTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        stage_configs=stage_configs,
        device=device,
        experiment_name=args.experiment_name
    )
    
    # Train
    results = trainer.train()
    
    # Save results
    output_dir = Path(MODELS_DIR) / args.experiment_name
    ensure_dir(output_dir)
    safe_save(results, output_dir / "results.json")
    
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
