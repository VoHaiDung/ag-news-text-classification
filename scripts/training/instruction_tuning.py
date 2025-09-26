"""
Instruction Tuning Script for AG News Text Classification
==========================================================

This script implements instruction-based fine-tuning following methodologies from:
- Wei et al. (2022): "Finetuned Language Models Are Zero-Shot Learners"
- Sanh et al. (2022): "Multitask Prompted Training Enables Zero-Shot Task Generalization"
- Wang et al. (2022): "Self-Instruct: Aligning Language Models with Self-Generated Instructions"

The instruction tuning pipeline implements:
1. Task instruction formatting
2. Multi-task instruction learning
3. Chain-of-thought reasoning
4. Instruction-following optimization

Mathematical Framework:
L = E[log P(y|instruction, x, θ)]
where instruction provides explicit task description.

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
    GenerationConfig
)
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets.ag_news import AGNewsDataset
from src.models.prompt_based.instruction_model import InstructionTunedModel
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from src.utils.experiment_tracking import create_experiment, log_metrics
from src.utils.io_utils import safe_save, ensure_dir
from configs.constants import AG_NEWS_NUM_CLASSES, MODELS_DIR, AG_NEWS_CLASSES

logger = setup_logging(__name__)


@dataclass
class InstructionTemplate:
    """
    Instruction template for different task formulations.
    
    Based on instruction design principles from:
    - Mishra et al. (2022): "Cross-Task Generalization via Natural Language Crowdsourcing Instructions"
    """
    
    task_description: str
    input_format: str
    output_format: str
    examples: List[Dict[str, str]]
    
    def format_instruction(
        self,
        text: str,
        include_examples: bool = True,
        num_examples: int = 2
    ) -> str:
        """
        Format instruction with input text.
        
        Args:
            text: Input text to classify
            include_examples: Whether to include few-shot examples
            num_examples: Number of examples to include
            
        Returns:
            Formatted instruction string
        """
        instruction = f"{self.task_description}\n\n"
        
        if include_examples and self.examples:
            instruction += "Examples:\n"
            for example in self.examples[:num_examples]:
                instruction += f"Input: {example['input']}\n"
                instruction += f"Output: {example['output']}\n\n"
        
        instruction += f"{self.input_format}\n"
        instruction += f"Input: {text}\n"
        instruction += f"{self.output_format}\n"
        instruction += "Output:"
        
        return instruction


class InstructionDataset(Dataset):
    """
    Dataset for instruction-based learning.
    
    Implements instruction formatting strategies from:
    - Chung et al. (2022): "Scaling Instruction-Finetuned Language Models"
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        tokenizer: Any,
        instruction_templates: List[InstructionTemplate],
        max_length: int = 512,
        include_chain_of_thought: bool = False
    ):
        """
        Initialize instruction dataset.
        
        Args:
            base_dataset: Base AG News dataset
            tokenizer: Tokenizer
            instruction_templates: List of instruction templates
            max_length: Maximum sequence length
            include_chain_of_thought: Whether to include reasoning steps
        """
        self.base_dataset = base_dataset
        self.tokenizer = tokenizer
        self.instruction_templates = instruction_templates
        self.max_length = max_length
        self.include_chain_of_thought = include_chain_of_thought
        
        # Label to text mapping
        self.label_to_text = {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Science/Technology"
        }
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get instruction-formatted sample."""
        sample = self.base_dataset[idx]
        text = sample["text"]
        label = sample["label"]
        
        # Randomly select instruction template
        template = np.random.choice(self.instruction_templates)
        
        # Format instruction
        instruction = template.format_instruction(
            text,
            include_examples=np.random.random() > 0.5
        )
        
        # Format target
        target = self.label_to_text[label]
        
        if self.include_chain_of_thought:
            # Add reasoning steps
            reasoning = self._generate_reasoning(text, label)
            target = f"{reasoning} Therefore, the category is: {target}"
        
        # Tokenize
        input_encoding = self.tokenizer(
            instruction,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target,
            max_length=64,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "original_label": torch.tensor(label)
        }
    
    def _generate_reasoning(self, text: str, label: int) -> str:
        """
        Generate chain-of-thought reasoning.
        
        Args:
            text: Input text
            label: True label
            
        Returns:
            Reasoning string
        """
        # Simple heuristic-based reasoning
        reasoning_templates = {
            0: "This text discusses international events and global affairs.",
            1: "This text mentions sports, athletes, or competitions.",
            2: "This text covers business, finance, or economic topics.",
            3: "This text relates to technology, science, or innovation."
        }
        
        return reasoning_templates.get(label, "This text belongs to a specific category.")


class InstructionTuner:
    """
    Trainer for instruction-based fine-tuning.
    
    Implements instruction tuning strategies from:
    - Ouyang et al. (2022): "Training language models to follow instructions"
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize instruction tuner.
        
        Args:
            model: Model to fine-tune
            tokenizer: Tokenizer
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Compute device
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=64,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        # Training state
        self.best_metric = 0.0
        self.training_history = []
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with appropriate learning rate."""
        param_groups = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if "embed" not in n],
                "lr": self.config.get("learning_rate", 5e-5)
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if "embed" in n],
                "lr": self.config.get("embedding_lr", 1e-5)
            }
        ]
        
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.get("weight_decay", 0.01)
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_training_steps = len(self.train_loader) * self.config.get("num_epochs", 5)
        num_warmup_steps = int(num_training_steps * self.config.get("warmup_ratio", 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def compute_instruction_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss for instruction following.
        
        Args:
            logits: Model output logits
            labels: Target token IDs
            attention_mask: Attention mask
            
        Returns:
            Loss value
        """
        # Shift for autoregressive loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_attention_mask = attention_mask[..., 1:].contiguous()
        
        # Flatten
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        shift_attention_mask = shift_attention_mask.view(-1)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            reduction='none'
        )
        
        # Mask padding tokens
        loss = loss * shift_attention_mask
        loss = loss.sum() / shift_attention_mask.sum()
        
        return loss
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch with instruction tuning.
        
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
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # Compute loss
            if hasattr(outputs, "loss"):
                loss = outputs.loss
            else:
                loss = self.compute_instruction_loss(
                    outputs.logits,
                    labels,
                    attention_mask
                )
            
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
        Evaluate model on validation set.
        
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        label_map = {
            "world": 0, "international": 0, "global": 0,
            "sports": 1, "athletics": 1, "games": 1,
            "business": 2, "economy": 2, "finance": 2,
            "technology": 3, "science": 3, "tech": 3
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                original_labels = batch["original_label"].to(self.device)
                
                # Generate predictions
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    generation_config=self.generation_config
                )
                
                # Decode predictions
                generated_texts = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                
                # Parse predictions
                for text, label in zip(generated_texts, original_labels):
                    # Extract category from generated text
                    text_lower = text.lower()
                    pred_label = 3  # Default to Science/Technology
                    
                    for key, value in label_map.items():
                        if key in text_lower:
                            pred_label = value
                            break
                    
                    all_preds.append(pred_label)
                    all_labels.append(label.item())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1
        }
    
    def train(self) -> Dict[str, Any]:
        """
        Complete instruction tuning training loop.
        
        Returns:
            Training results
        """
        num_epochs = self.config.get("num_epochs", 5)
        
        logger.info(f"Starting instruction tuning for {num_epochs} epochs")
        
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
        save_dir = Path(self.config.get("output_dir", MODELS_DIR)) / "instruction_tuned"
        ensure_dir(save_dir)
        
        self.model.save_pretrained(save_dir / f"checkpoint_epoch_{epoch}")
        self.tokenizer.save_pretrained(save_dir / f"checkpoint_epoch_{epoch}")
        
        logger.info(f"Model saved to {save_dir}")


def create_instruction_templates() -> List[InstructionTemplate]:
    """
    Create diverse instruction templates for AG News classification.
    
    Returns:
        List of instruction templates
    """
    templates = []
    
    # Template 1: Direct classification
    template1 = InstructionTemplate(
        task_description="Classify the following news article into one of four categories: World, Sports, Business, or Science/Technology.",
        input_format="Article to classify:",
        output_format="Category:",
        examples=[
            {
                "input": "The stock market saw significant gains today as tech companies reported strong earnings.",
                "output": "Business"
            },
            {
                "input": "Scientists discover a new exoplanet that could potentially harbor life.",
                "output": "Science/Technology"
            }
        ]
    )
    templates.append(template1)
    
    # Template 2: Question-based
    template2 = InstructionTemplate(
        task_description="Read the news article and answer: What category does this article belong to?",
        input_format="News article:",
        output_format="The category is:",
        examples=[
            {
                "input": "The national team won the championship after a thrilling final match.",
                "output": "Sports"
            },
            {
                "input": "International leaders meet to discuss climate change solutions.",
                "output": "World"
            }
        ]
    )
    templates.append(template2)
    
    # Template 3: Analytical
    template3 = InstructionTemplate(
        task_description="Analyze the content of this news article and determine its primary topic area.",
        input_format="Article content:",
        output_format="Primary topic:",
        examples=[
            {
                "input": "New AI breakthrough enables faster drug discovery process.",
                "output": "Science/Technology"
            },
            {
                "input": "Local businesses struggle with supply chain disruptions.",
                "output": "Business"
            }
        ]
    )
    templates.append(template3)
    
    return templates


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Instruction tuning for AG News classification"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="google/flan-t5-base",
        help="Base model for instruction tuning"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="seq2seq",
        choices=["causal", "seq2seq"],
        help="Model type"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=5,
        help="Number of epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--include-cot",
        action="store_true",
        help="Include chain-of-thought reasoning"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    
    return parser.parse_args()


def main():
    """Main instruction tuning pipeline."""
    args = parse_arguments()
    
    # Setup
    logger.info("Setting up instruction tuning")
    device = torch.device(args.device)
    ensure_reproducibility(seed=42)
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Create instruction templates
    templates = create_instruction_templates()
    
    # Create datasets
    base_train = AGNewsDataset(split="train")
    base_val = AGNewsDataset(split="validation")
    
    train_dataset = InstructionDataset(
        base_dataset=base_train,
        tokenizer=tokenizer,
        instruction_templates=templates,
        include_chain_of_thought=args.include_cot
    )
    
    val_dataset = InstructionDataset(
        base_dataset=base_val,
        tokenizer=tokenizer,
        instruction_templates=templates,
        include_chain_of_thought=args.include_cot
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
        "learning_rate": args.learning_rate,
        "embedding_lr": args.learning_rate / 10,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "max_grad_norm": 1.0
    }
    
    # Initialize tuner
    tuner = InstructionTuner(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    results = tuner.train()
    
    logger.info(f"Instruction tuning completed. Best F1: {results['best_metric']:.4f}")
    
    # Save results
    output_dir = Path(MODELS_DIR) / "instruction_tuning"
    ensure_dir(output_dir)
    safe_save(results, output_dir / "results.json")


if __name__ == "__main__":
    main()
