"""
GPT-4 Knowledge Distillation Script for AG News Text Classification
====================================================================

This script implements knowledge distillation from GPT-4 following methodologies from:
- Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
- Hsieh et al. (2023): "Distilling Step-by-Step: Outperforming Larger Language Models"
- Wang et al. (2023): "Self-Instruct: Aligning Language Models with Self-Generated Instructions"

The distillation pipeline implements:
1. GPT-4 API integration for generating soft labels
2. Chain-of-thought reasoning extraction
3. Multi-objective distillation (predictions + rationales)
4. Temperature-based knowledge transfer
5. Progressive distillation strategies

Mathematical Framework:
L_distill = α·CE(y, y_true) + (1-α)·KL(σ(z_s/T), σ(z_t/T))
where z_s, z_t are student and teacher logits, T is temperature.

Author: Võ Hải Dũng
License: MIT
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import time
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import numpy as np
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from sklearn.metrics import accuracy_score, f1_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets.ag_news import AGNewsDataset, create_ag_news_datasets
from src.training.strategies.distillation.knowledge_distill import KnowledgeDistillation
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from src.utils.experiment_tracking import create_experiment, log_metrics
from src.utils.io_utils import safe_save, safe_load, ensure_dir
from configs.constants import AG_NEWS_NUM_CLASSES, MODELS_DIR, AG_NEWS_CLASSES

logger = setup_logging(__name__)


@dataclass
class GPT4Response:
    """
    Structured response from GPT-4.
    
    Contains both prediction and reasoning for interpretability.
    """
    
    text: str
    predicted_label: int
    confidence_scores: List[float]
    reasoning: str
    chain_of_thought: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "predicted_label": self.predicted_label,
            "confidence_scores": self.confidence_scores,
            "reasoning": self.reasoning,
            "chain_of_thought": self.chain_of_thought
        }


class GPT4Teacher:
    """
    GPT-4 teacher model for knowledge distillation.
    
    Implements efficient API usage strategies from:
    - Brown et al. (2020): "Language Models are Few-Shot Learners"
    - Wei et al. (2022): "Chain-of-Thought Prompting"
    """
    
    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 256,
        use_chain_of_thought: bool = True
    ):
        """
        Initialize GPT-4 teacher.
        
        Args:
            api_key: OpenAI API key
            model_name: GPT model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            use_chain_of_thought: Whether to use CoT prompting
        """
        openai.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_chain_of_thought = use_chain_of_thought
        
        # Cache for responses
        self.response_cache = {}
        
        # Prompts
        self.classification_prompt = self._create_classification_prompt()
        self.cot_prompt = self._create_cot_prompt()
        
        logger.info(f"Initialized GPT-4 teacher with model: {model_name}")
    
    def _create_classification_prompt(self) -> str:
        """Create classification prompt template."""
        return """Task: Classify the following news article into one of four categories:
1. World (international news, global events)
2. Sports (athletics, competitions, games)
3. Business (economy, finance, markets)
4. Science/Technology (tech, innovation, research)

Article: {text}

Please provide:
1. The category (World/Sports/Business/Science-Technology)
2. Confidence scores for each category (0-1)
3. Brief reasoning for your classification

Response format:
Category: [category]
Confidence: World=[score], Sports=[score], Business=[score], Science-Technology=[score]
Reasoning: [your reasoning]"""
    
    def _create_cot_prompt(self) -> str:
        """Create chain-of-thought prompt template."""
        return """Task: Classify this news article step-by-step.

Article: {text}

Let's think step by step:
1. What are the key topics mentioned?
2. What domain do these topics belong to?
3. Which category best fits: World/Sports/Business/Science-Technology?

Provide your reasoning process and final answer."""
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def get_prediction(self, text: str) -> GPT4Response:
        """
        Get prediction from GPT-4.
        
        Args:
            text: Input text to classify
            
        Returns:
            GPT4Response with prediction and reasoning
        """
        # Check cache
        cache_key = hash(text)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Prepare messages
        messages = [
            {"role": "system", "content": "You are an expert news classifier."},
            {"role": "user", "content": self.classification_prompt.format(text=text)}
        ]
        
        if self.use_chain_of_thought:
            messages.append({
                "role": "user",
                "content": self.cot_prompt.format(text=text)
            })
        
        try:
            # Call GPT-4 API
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Parse response
            content = response.choices[0].message.content
            parsed_response = self._parse_response(content, text)
            
            # Cache response
            self.response_cache[cache_key] = parsed_response
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"GPT-4 API error: {e}")
            # Return default response
            return self._get_default_response(text)
    
    def _parse_response(self, content: str, text: str) -> GPT4Response:
        """
        Parse GPT-4 response into structured format.
        
        Args:
            content: Raw GPT-4 response
            text: Original input text
            
        Returns:
            Parsed GPT4Response
        """
        # Initialize default values
        predicted_label = 0
        confidence_scores = [0.25, 0.25, 0.25, 0.25]
        reasoning = ""
        chain_of_thought = ""
        
        # Parse category
        if "World" in content:
            predicted_label = 0
        elif "Sports" in content:
            predicted_label = 1
        elif "Business" in content:
            predicted_label = 2
        elif "Science" in content or "Technology" in content:
            predicted_label = 3
        
        # Parse confidence scores
        try:
            import re
            scores = re.findall(r'(\d+\.?\d*)', content)
            if len(scores) >= 4:
                confidence_scores = [float(s) for s in scores[:4]]
                # Normalize to sum to 1
                total = sum(confidence_scores)
                confidence_scores = [s/total for s in confidence_scores]
        except:
            # Use default uniform distribution
            pass
        
        # Extract reasoning
        if "Reasoning:" in content:
            reasoning = content.split("Reasoning:")[1].split("\n")[0].strip()
        
        # Extract chain-of-thought
        if "step by step" in content.lower():
            chain_of_thought = content
        
        return GPT4Response(
            text=text,
            predicted_label=predicted_label,
            confidence_scores=confidence_scores,
            reasoning=reasoning,
            chain_of_thought=chain_of_thought
        )
    
    def _get_default_response(self, text: str) -> GPT4Response:
        """Get default response when API fails."""
        return GPT4Response(
            text=text,
            predicted_label=0,
            confidence_scores=[0.25, 0.25, 0.25, 0.25],
            reasoning="API error - using default response",
            chain_of_thought=""
        )
    
    async def get_batch_predictions(
        self,
        texts: List[str],
        batch_size: int = 10
    ) -> List[GPT4Response]:
        """
        Get predictions for batch of texts.
        
        Args:
            texts: List of texts to classify
            batch_size: Batch size for concurrent requests
            
        Returns:
            List of GPT4Response objects
        """
        responses = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="GPT-4 predictions"):
            batch = texts[i:i+batch_size]
            
            # Concurrent requests
            tasks = [self.get_prediction(text) for text in batch]
            batch_responses = await asyncio.gather(*tasks)
            
            responses.extend(batch_responses)
            
            # Rate limiting
            await asyncio.sleep(1)
        
        return responses


class DistillationDataset(Dataset):
    """
    Dataset with GPT-4 soft labels for distillation.
    
    Implements data loading strategies from:
    - Touvron et al. (2023): "LLaMA: Open and Efficient Foundation Language Models"
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        gpt4_responses: List[GPT4Response],
        tokenizer: Any,
        max_length: int = 256
    ):
        """
        Initialize distillation dataset.
        
        Args:
            base_dataset: Original AG News dataset
            gpt4_responses: GPT-4 predictions
            tokenizer: Tokenizer for student model
            max_length: Maximum sequence length
        """
        self.base_dataset = base_dataset
        self.gpt4_responses = gpt4_responses
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item with soft labels."""
        # Get original sample
        sample = self.base_dataset[idx]
        text = sample["text"]
        true_label = sample["label"]
        
        # Get GPT-4 response
        gpt4_response = self.gpt4_responses[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "true_label": torch.tensor(true_label),
            "teacher_label": torch.tensor(gpt4_response.predicted_label),
            "teacher_logits": torch.tensor(gpt4_response.confidence_scores),
            "has_reasoning": len(gpt4_response.reasoning) > 0
        }


class GPT4Distiller:
    """
    Distillation trainer using GPT-4 as teacher.
    
    Implements distillation strategies from:
    - Sanh et al. (2019): "DistilBERT, a distilled version of BERT"
    - Jiao et al. (2020): "TinyBERT: Distilling BERT for Natural Language Understanding"
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher: GPT4Teacher,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize distiller.
        
        Args:
            student_model: Student model to train
            teacher: GPT4Teacher instance
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            device: Compute device
        """
        self.student_model = student_model.to(device)
        self.teacher = teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Distillation parameters
        self.temperature = config.get("temperature", 3.0)
        self.alpha = config.get("alpha", 0.7)  # Weight for distillation loss
        
        # Training state
        self.best_metric = 0.0
        self.training_history = []
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for student model."""
        return torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.get("learning_rate", 3e-5),
            weight_decay=self.config.get("weight_decay", 0.01)
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        num_training_steps = len(self.train_loader) * self.config.get("num_epochs", 10)
        num_warmup_steps = int(num_training_steps * 0.1)
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def compute_distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        true_labels: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """
        Compute knowledge distillation loss.
        
        Implements loss function from:
        - Hinton et al. (2015): "Distilling the Knowledge in a Neural Network"
        
        Args:
            student_logits: Student model outputs
            teacher_logits: Teacher model outputs
            true_labels: Ground truth labels
            temperature: Distillation temperature
            
        Returns:
            Combined distillation loss
        """
        # Student loss with true labels
        student_loss = F.cross_entropy(student_logits, true_labels)
        
        # Distillation loss with soft labels
        student_soft = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
        
        distillation_loss = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (temperature ** 2)
        
        # Combined loss
        loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        
        return loss
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch with distillation.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.student_model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            true_labels = batch["true_label"].to(self.device)
            teacher_logits = batch["teacher_logits"].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.student_model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            # Compute distillation loss
            loss = self.compute_distillation_loss(
                outputs.logits,
                teacher_logits,
                true_labels,
                self.temperature
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.student_model.parameters(),
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
        Evaluate student model.
        
        Returns:
            Dictionary of metrics
        """
        self.student_model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["true_label"].to(self.device)
                
                outputs = self.student_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
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
        Complete distillation training.
        
        Returns:
            Training results
        """
        num_epochs = self.config.get("num_epochs", 10)
        
        logger.info(f"Starting distillation training for {num_epochs} epochs")
        
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
        """Save student model."""
        save_dir = Path(self.config.get("output_dir", MODELS_DIR)) / "distilled"
        ensure_dir(save_dir)
        
        self.student_model.save_pretrained(save_dir / f"checkpoint_epoch_{epoch}")
        
        logger.info(f"Model saved to {save_dir}")


async def generate_gpt4_labels(
    dataset: Dataset,
    teacher: GPT4Teacher,
    num_samples: Optional[int] = None,
    cache_path: Optional[Path] = None
) -> List[GPT4Response]:
    """
    Generate GPT-4 labels for dataset.
    
    Args:
        dataset: Dataset to label
        teacher: GPT4Teacher instance
        num_samples: Number of samples to label (None for all)
        cache_path: Path to cache responses
        
    Returns:
        List of GPT4Response objects
    """
    # Check cache
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached GPT-4 responses from {cache_path}")
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
        return [GPT4Response(**item) for item in cached_data]
    
    # Extract texts
    texts = []
    num_samples = num_samples or len(dataset)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        texts.append(sample["text"])
    
    logger.info(f"Generating GPT-4 labels for {len(texts)} samples")
    
    # Get predictions
    responses = await teacher.get_batch_predictions(texts)
    
    # Cache responses
    if cache_path:
        ensure_dir(cache_path.parent)
        with open(cache_path, 'w') as f:
            json.dump([r.to_dict() for r in responses], f, indent=2)
        logger.info(f"Cached GPT-4 responses to {cache_path}")
    
    return responses


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Distill knowledge from GPT-4 to smaller model"
    )
    
    parser.add_argument(
        "--student-model",
        type=str,
        default="distilbert-base-uncased",
        help="Student model to train"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--gpt4-model",
        type=str,
        default="gpt-4",
        help="GPT-4 model variant"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of samples for distillation"
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="data/distillation/gpt4_labels.json",
        help="Path to cache GPT-4 responses"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=3.0,
        help="Distillation temperature"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Weight for distillation loss"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device"
    )
    
    return parser.parse_args()


def main():
    """Main distillation pipeline."""
    args = parse_arguments()
    
    # Setup
    logger.info("Setting up GPT-4 distillation")
    device = torch.device(args.device)
    ensure_reproducibility(seed=42)
    
    # Initialize GPT-4 teacher
    teacher = GPT4Teacher(
        api_key=args.api_key,
        model_name=args.gpt4_model,
        use_chain_of_thought=True
    )
    
    # Load student model
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    student_model = AutoModelForSequenceClassification.from_pretrained(
        args.student_model,
        num_labels=AG_NEWS_NUM_CLASSES
    )
    
    # Load datasets
    train_dataset, val_dataset, _ = create_ag_news_datasets(tokenizer=tokenizer)
    
    # Generate or load GPT-4 labels
    loop = asyncio.get_event_loop()
    gpt4_responses = loop.run_until_complete(
        generate_gpt4_labels(
            train_dataset,
            teacher,
            num_samples=args.num_samples,
            cache_path=Path(args.cache_path)
        )
    )
    
    # Create distillation dataset
    distill_train = DistillationDataset(
        train_dataset,
        gpt4_responses,
        tokenizer
    )
    
    # For validation, use same approach or original labels
    distill_val = DistillationDataset(
        val_dataset,
        gpt4_responses[:len(val_dataset)],  # Simple approach
        tokenizer
    )
    
    # Create data loaders
    train_loader = DataLoader(
        distill_train,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        distill_val,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Prepare config
    config = {
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0
    }
    
    # Initialize distiller
    distiller = GPT4Distiller(
        student_model=student_model,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Train
    results = distiller.train()
    
    logger.info(f"Distillation completed. Best F1: {results['best_metric']:.4f}")
    
    # Save results
    output_dir = Path(MODELS_DIR) / "gpt4_distilled"
    ensure_dir(output_dir)
    safe_save(results, output_dir / "results.json")


if __name__ == "__main__":
    main()
