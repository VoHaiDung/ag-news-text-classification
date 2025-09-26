"""
Prompt-based Training Script for AG News Text Classification
============================================================

This script implements prompt-based learning following methodologies from:
- Liu et al. (2021): "Pre-train, Prompt, and Predict: A Systematic Survey"
- Schick & Schütze (2021): "Exploiting Cloze Questions for Few Shot Text Classification"
- Gao et al. (2021): "Making Pre-trained Language Models Better Few-shot Learners"

The prompt-based training implements:
1. Template-based prompting
2. Verbalizer optimization
3. Continuous prompt tuning
4. Prompt ensemble strategies

Mathematical Framework:
P(y|x) = P(v|prompt(x))
where v is the verbalizer mapping labels to tokens.

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoConfig,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets.prompted_dataset import PromptedAGNewsDataset
from src.models.prompt_based.prompt_model import PromptBasedClassifier
from src.models.prompt_based.soft_prompt import SoftPromptModel
from src.models.prompt_based.template_manager import TemplateManager
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from src.utils.experiment_tracking import create_experiment, log_metrics
from src.utils.io_utils import safe_save, ensure_dir
from src.utils.prompt_utils import (
    create_verbalizer,
    optimize_verbalizer,
    generate_prompt_templates,
    evaluate_prompt_quality
)
from configs.constants import AG_NEWS_NUM_CLASSES, MODELS_DIR, AG_NEWS_CLASSES

logger = setup_logging(__name__)


class PromptTrainer:
    """
    Trainer for prompt-based learning.
    
    Implements strategies from:
    - Lester et al. (2021): "The Power of Scale for Parameter-Efficient Prompt Tuning"
    - Li & Liang (2021): "Prefix-Tuning: Optimizing Continuous Prompts"
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device,
        template_manager: TemplateManager
    ):
        """
        Initialize prompt trainer.
        
        Args:
            model: Prompt-based model
            tokenizer: Tokenizer
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Compute device
            template_manager: Template manager for prompts
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.template_manager = template_manager
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Verbalizer setup
        self.verbalizer = self._setup_verbalizer()
        
        # Training state
        self.best_metric = 0.0
        self.training_history = []
    
    def _setup_verbalizer(self) -> Dict[int, List[str]]:
        """
        Setup verbalizer mapping labels to tokens.
        
        Returns:
            Verbalizer dictionary
        """
        # Default verbalizers for AG News categories
        default_verbalizer = {
            0: ["World", "International", "Global"],
            1: ["Sports", "Athletics", "Games"],
            2: ["Business", "Economy", "Finance"],
            3: ["Technology", "Science", "Tech"]
        }
        
        if self.config.get("optimize_verbalizer", False):
            # Optimize verbalizer using training data
            logger.info("Optimizing verbalizer...")
            optimized_verbalizer = optimize_verbalizer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_loader=self.train_loader,
                initial_verbalizer=default_verbalizer,
                device=self.device
            )
            return optimized_verbalizer
        
        return default_verbalizer
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for prompt tuning."""
        # Different learning rates for different components
        param_groups = []
        
        if self.config.get("prompt_tuning_only", False):
            # Only tune prompt parameters
            if hasattr(self.model, "prompt_embeddings"):
                param_groups.append({
                    "params": self.model.prompt_embeddings.parameters(),
                    "lr": self.config.get("prompt_lr", 1e-3)
                })
        else:
            # Tune all parameters with different rates
            prompt_params = []
            model_params = []
            
            for name, param in self.model.named_parameters():
                if "prompt" in name.lower():
                    prompt_params.append(param)
                else:
                    model_params.append(param)
            
            if prompt_params:
                param_groups.append({
                    "params": prompt_params,
                    "lr": self.config.get("prompt_lr", 1e-3)
                })
            
            if model_params:
                param_groups.append({
                    "params": model_params,
                    "lr": self.config.get("model_lr", 2e-5)
                })
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.get("weight_decay", 0.01)
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_training_steps = len(self.train_loader) * self.config.get("num_epochs", 10)
        num_warmup_steps = int(num_training_steps * self.config.get("warmup_ratio", 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def compute_prompt_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for prompt-based learning.
        
        Args:
            logits: Model output logits
            labels: True labels
            mask_positions: Positions of mask tokens
            
        Returns:
            Loss value
        """
        batch_size = logits.size(0)
        
        # Extract logits at mask positions
        mask_logits = []
        for i in range(batch_size):
            mask_pos = mask_positions[i]
            mask_logit = logits[i, mask_pos, :]
            mask_logits.append(mask_logit)
        
        mask_logits = torch.stack(mask_logits)
        
        # Map labels to verbalizer tokens
        verbalizer_ids = []
        for label in labels:
            label_tokens = self.verbalizer[label.item()]
            # Use first token as target
            token_id = self.tokenizer.convert_tokens_to_ids(label_tokens[0])
            verbalizer_ids.append(token_id)
        
        verbalizer_ids = torch.tensor(verbalizer_ids).to(self.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(mask_logits, verbalizer_ids)
        
        return loss
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch with prompts.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Get mask positions for prompt-based prediction
            mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
            mask_positions = mask_positions.view(-1, 1)  # One mask per sample
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if hasattr(self.model, "forward_prompt"):
                # Custom prompt model
                outputs = self.model.forward_prompt(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            else:
                # Standard model
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
            
            # Compute loss
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = self.compute_prompt_loss(logits, labels, mask_positions)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get("max_grad_norm", 1.0)
            )
            
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({"loss": loss.item()})
        
        return total_loss / num_batches
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model with prompts.
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Get mask positions
                mask_positions = (input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
                mask_positions = mask_positions.view(-1, 1)
                
                # Forward pass
                if hasattr(self.model, "forward_prompt"):
                    outputs = self.model.forward_prompt(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                # Get predictions
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Extract logits at mask positions and predict
                batch_size = logits.size(0)
                for i in range(batch_size):
                    mask_pos = mask_positions[i]
                    mask_logit = logits[i, mask_pos, :].squeeze()
                    
                    # Score each label based on verbalizer
                    label_scores = []
                    for label_idx in range(AG_NEWS_NUM_CLASSES):
                        label_tokens = self.verbalizer[label_idx]
                        token_ids = [
                            self.tokenizer.convert_tokens_to_ids(token)
                            for token in label_tokens
                        ]
                        # Average score across verbalizer tokens
                        scores = [mask_logit[tid].item() for tid in token_ids]
                        label_scores.append(np.mean(scores))
                    
                    pred = np.argmax(label_scores)
                    all_preds.append(pred)
                
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Complete prompt-based training loop.
        
        Returns:
            Training results
        """
        num_epochs = self.config.get("num_epochs", 10)
        
        logger.info(f"Starting prompt-based training for {num_epochs} epochs")
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = self.evaluate()
            
            # Log
            logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Loss: {train_loss:.4f}, "
                f"Acc: {val_metrics['accuracy']:.4f}, "
                f"F1: {val_metrics['f1_macro']:.4f}"
            )
            
            # Track best model
            if val_metrics["f1_macro"] > self.best_metric:
                self.best_metric = val_metrics["f1_macro"]
                self.save_model(epoch)
            
            # Update history
            self.training_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics
            })
        
        return {
            "best_metric": self.best_metric,
            "history": self.training_history
        }
    
    def save_model(self, epoch: int):
        """Save model checkpoint."""
        save_dir = Path(self.config.get("output_dir", MODELS_DIR)) / "prompt_model"
        ensure_dir(save_dir)
        
        # Save model
        model_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(self.model.state_dict(), model_path)
        
        # Save verbalizer
        verbalizer_path = save_dir / "verbalizer.json"
        safe_save(self.verbalizer, verbalizer_path)
        
        # Save config
        config_path = save_dir / "config.json"
        safe_save(self.config, config_path)
        
        logger.info(f"Model saved to {save_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train with prompts on AG News dataset"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="roberta-large",
        help="Base model for prompting"
    )
    parser.add_argument(
        "--prompt-type",
        type=str,
        default="manual",
        choices=["manual", "soft", "mixed"],
        help="Type of prompting"
    )
    parser.add_argument(
        "--template",
        type=str,
        default="This is a [MASK] news: {text}",
        help="Prompt template"
    )
    parser.add_argument(
        "--num-prompt-tokens",
        type=int,
        default=10,
        help="Number of soft prompt tokens"
    )
    parser.add_argument(
        "--prompt-tuning-only",
        action="store_true",
        help="Only tune prompt parameters"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--prompt-lr",
        type=float,
        default=1e-3,
        help="Learning rate for prompts"
    )
    parser.add_argument(
        "--model-lr",
        type=float,
        default=2e-5,
        help="Learning rate for model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    
    return parser.parse_args()


def main():
    """Main prompt training pipeline."""
    args = parse_arguments()
    
    # Setup
    logger.info("Setting up prompt-based training")
    device = torch.device(args.device)
    ensure_reproducibility(seed=42)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if args.prompt_type == "soft":
        # Soft prompt model
        model = SoftPromptModel(
            model_name=args.model_name,
            num_labels=AG_NEWS_NUM_CLASSES,
            num_prompt_tokens=args.num_prompt_tokens
        )
    else:
        # Standard model for manual prompts
        model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    
    # Create template manager
    template_manager = TemplateManager()
    template_manager.add_template("default", args.template)
    
    # Create prompted dataset
    from src.data.datasets.ag_news import AGNewsDataset
    
    train_dataset = PromptedAGNewsDataset(
        dataset=AGNewsDataset(split="train"),
        tokenizer=tokenizer,
        template=args.template,
        max_length=128
    )
    
    val_dataset = PromptedAGNewsDataset(
        dataset=AGNewsDataset(split="validation"),
        tokenizer=tokenizer,
        template=args.template,
        max_length=128
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Prepare config
    config = {
        "num_epochs": args.num_epochs,
        "prompt_lr": args.prompt_lr,
        "model_lr": args.model_lr,
        "prompt_tuning_only": args.prompt_tuning_only,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0
    }
    
    # Initialize trainer
    trainer = PromptTrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        template_manager=template_manager
    )
    
    # Train
    results = trainer.train()
    
    logger.info(f"Training completed. Best F1: {results['best_metric']:.4f}")
    
    # Save results
    output_dir = Path(MODELS_DIR) / "prompt_training"
    ensure_dir(output_dir)
    safe_save(results, output_dir / "results.json")


if __name__ == "__main__":
    main()
