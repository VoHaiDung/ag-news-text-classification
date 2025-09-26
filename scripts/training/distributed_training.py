"""
Distributed Training Script for AG News Text Classification
===========================================================

This script implements distributed training across multiple GPUs/nodes following:
- Li et al. (2020): "PyTorch Distributed: Experiences on Accelerating Data Parallel Training"
- Sergeev & Del Balso (2018): "Horovod: fast and easy distributed deep learning"
- Goyal et al. (2017): "Accurate, Large Minibatch SGD"

The distributed training implements:
1. Data Parallel (DP) and Distributed Data Parallel (DDP)
2. Gradient accumulation and synchronization
3. Learning rate scaling for large batch training
4. Mixed precision training with gradient scaling

Mathematical Framework:
Effective batch size = batch_size * gradient_accumulation * world_size
Learning rate scaling: lr_effective = lr_base * sqrt(batch_size_effective / batch_size_base)

Author: Võ Hải Dũng
License: MIT
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets.ag_news import create_ag_news_datasets
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from src.utils.distributed_utils import (
    setup_distributed,
    cleanup_distributed,
    reduce_tensor,
    gather_tensors,
    is_main_process,
    get_rank,
    get_world_size,
    synchronize
)
from src.utils.io_utils import safe_save, ensure_dir
from configs.constants import AG_NEWS_NUM_CLASSES, MODELS_DIR

logger = setup_logging(__name__)


class DistributedTrainer:
    """
    Distributed trainer implementing efficient multi-GPU training.
    
    Follows best practices from:
    - NVIDIA's Apex documentation
    - Facebook's FairSeq distributed training
    - Google's BERT pretraining setup
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_sampler: DistributedSampler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        local_rank: int,
        world_size: int
    ):
        """
        Initialize distributed trainer.
        
        Args:
            model: Model to train
            train_sampler: Distributed sampler for training
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            local_rank: Local GPU rank
            world_size: Total number of processes
        """
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = torch.device(f"cuda:{local_rank}")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Wrap model with DDP
        self.model = DDP(
            self.model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=config.get("find_unused_parameters", False)
        )
        
        self.train_sampler = train_sampler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
        # Scale learning rate based on effective batch size
        self.base_lr = config.get("learning_rate", 2e-5)
        self.scaled_lr = self.scale_learning_rate()
        
        # Initialize optimizer
        self.optimizer = self.create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self.create_scheduler()
        
        # Mixed precision training
        self.use_amp = config.get("use_amp", True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.accumulation_steps = config.get("gradient_accumulation_steps", 1)
        
        if is_main_process():
            logger.info(
                f"Initialized distributed training with {world_size} GPUs\n"
                f"Effective batch size: {self.get_effective_batch_size()}\n"
                f"Scaled learning rate: {self.scaled_lr:.2e}"
            )
    
    def scale_learning_rate(self) -> float:
        """
        Scale learning rate for distributed training.
        
        Implements linear scaling rule from:
        - Goyal et al. (2017): "Accurate, Large Minibatch SGD"
        """
        batch_size = self.config.get("batch_size", 32)
        base_batch_size = self.config.get("base_batch_size", 32)
        
        effective_batch_size = self.get_effective_batch_size()
        
        if self.config.get("lr_scaling", "linear") == "linear":
            # Linear scaling
            scale_factor = effective_batch_size / base_batch_size
        elif self.config.get("lr_scaling") == "sqrt":
            # Square root scaling
            scale_factor = np.sqrt(effective_batch_size / base_batch_size)
        else:
            scale_factor = 1.0
        
        return self.base_lr * scale_factor
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size across all processes."""
        return (
            self.config.get("batch_size", 32) *
            self.accumulation_steps *
            self.world_size
        )
    
    def create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with proper weight decay."""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.get("weight_decay", 0.01),
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.scaled_lr,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        return optimizer
    
    def create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        num_training_steps = (
            len(self.train_loader) // self.accumulation_steps *
            self.config.get("num_epochs", 10)
        )
        num_warmup_steps = int(num_training_steps * self.config.get("warmup_ratio", 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch with distributed data parallel.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss across all processes
        """
        self.model.train()
        self.train_sampler.set_epoch(epoch)  # Ensure different shuffling each epoch
        
        total_loss = torch.tensor(0.0).to(self.device)
        num_steps = 0
        
        # Progress bar only on main process
        if is_main_process():
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            pbar = self.train_loader
        
        for step, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                if (step + 1) % self.accumulation_steps == 0:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("max_grad_norm", 1.0)
                    )
                    
                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
            else:
                # Standard forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.accumulation_steps
                loss.backward()
                
                if (step + 1) % self.accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.get("max_grad_norm", 1.0)
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    if self.scheduler:
                        self.scheduler.step()
            
            total_loss += loss.item() * self.accumulation_steps
            num_steps += 1
            
            # Update progress bar on main process
            if is_main_process() and isinstance(pbar, tqdm):
                pbar.set_postfix({"loss": loss.item() * self.accumulation_steps})
        
        # Reduce loss across all processes
        total_loss = reduce_tensor(total_loss, self.world_size)
        avg_loss = total_loss.item() / num_steps
        
        return avg_loss
    
    def evaluate(self) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation set.
        
        Returns:
            Tuple of (validation loss, metrics)
        """
        self.model.eval()
        
        total_loss = torch.tensor(0.0).to(self.device)
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                
                total_loss += outputs.loss
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.append(predictions)
                all_labels.append(labels)
        
        # Gather predictions from all processes
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        
        gathered_predictions = gather_tensors(all_predictions, self.world_size)
        gathered_labels = gather_tensors(all_labels, self.world_size)
        
        # Calculate metrics on main process
        metrics = {}
        if is_main_process():
            gathered_predictions = gathered_predictions.cpu().numpy()
            gathered_labels = gathered_labels.cpu().numpy()
            
            accuracy = accuracy_score(gathered_labels, gathered_predictions)
            f1 = f1_score(gathered_labels, gathered_predictions, average="macro")
            
            metrics = {
                "accuracy": accuracy,
                "f1_macro": f1
            }
        
        # Reduce loss
        total_loss = reduce_tensor(total_loss, self.world_size)
        avg_loss = total_loss.item() / len(self.val_loader)
        
        return avg_loss, metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Complete distributed training loop.
        
        Returns:
            Training results
        """
        num_epochs = self.config.get("num_epochs", 10)
        best_val_metric = 0.0
        best_epoch = 0
        
        if is_main_process():
            logger.info(f"Starting distributed training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Training
            train_loss = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_metrics = self.evaluate()
            
            # Log on main process
            if is_main_process():
                logger.info(
                    f"Epoch {epoch}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
                
                if val_metrics:
                    logger.info(
                        f"Val Accuracy: {val_metrics['accuracy']:.4f}, "
                        f"Val F1: {val_metrics['f1_macro']:.4f}"
                    )
                    
                    # Check for best model
                    if val_metrics["f1_macro"] > best_val_metric:
                        best_val_metric = val_metrics["f1_macro"]
                        best_epoch = epoch
                        self.save_checkpoint(epoch, is_best=True)
            
            # Synchronize processes
            synchronize()
        
        results = {
            "best_epoch": best_epoch,
            "best_metric": best_val_metric,
            "final_epoch": num_epochs
        }
        
        return results
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        if not is_main_process():
            return
        
        save_dir = Path(self.config.get("output_dir", MODELS_DIR))
        ensure_dir(save_dir)
        
        if is_best:
            save_path = save_dir / "best_model.pt"
        else:
            save_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        
        # Save model state (unwrap DDP)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed training on AG News dataset"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-uncased",
        help="Model name or path"
    )
    
    # Training arguments
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size per GPU"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Base learning rate"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    
    # Distributed arguments
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        help="Distributed backend"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(MODELS_DIR / "distributed"),
        help="Output directory"
    )
    
    return parser.parse_args()


def main():
    """Main distributed training pipeline."""
    args = parse_arguments()
    
    # Setup distributed training
    local_rank, world_size = setup_distributed(
        backend=args.backend,
        local_rank=args.local_rank
    )
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    # Setup reproducibility
    ensure_reproducibility(seed=args.seed + local_rank)
    
    # Only log on main process
    if is_main_process():
        logger.info(f"Starting distributed training with {world_size} GPUs")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = create_ag_news_datasets(tokenizer=tokenizer)
    
    # Create distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=AG_NEWS_NUM_CLASSES
    )
    
    # Prepare config
    config = {
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "output_dir": args.output_dir,
        "use_amp": True,
        "lr_scaling": "linear",
        "base_batch_size": 32
    }
    
    # Initialize trainer
    trainer = DistributedTrainer(
        model=model,
        train_sampler=train_sampler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        local_rank=local_rank,
        world_size=world_size
    )
    
    # Train
    results = trainer.train()
    
    if is_main_process():
        logger.info(f"Training completed. Best F1: {results['best_metric']:.4f}")
        
        # Save results
        output_dir = Path(args.output_dir)
        ensure_dir(output_dir)
        safe_save(results, output_dir / "training_results.json")
    
    # Cleanup distributed
    cleanup_distributed()


if __name__ == "__main__":
    main()
