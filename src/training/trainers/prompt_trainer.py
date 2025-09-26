"""
Prompt-based Trainer Implementation for AG News Text Classification
====================================================================

This module implements prompt-based training strategies including soft prompts,
prompt tuning, and instruction-based learning for efficient model adaptation.

Key Techniques:
- Soft prompt tuning (Lester et al., 2021)
- Prefix tuning (Li & Liang, 2021)
- P-tuning v2 (Liu et al., 2022)
- Instruction tuning (Wei et al., 2022)

References:
- Lester et al. (2021): "The Power of Scale for Parameter-Efficient Prompt Tuning"
- Li & Liang (2021): "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
- Liu et al. (2022): "P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.training.trainers.base_trainer import BaseTrainer, TrainerConfig
from src.models.prompt_based.prompt_model import PromptModel
from src.models.prompt_based.soft_prompt import SoftPrompt
from src.models.prompt_based.template_manager import TemplateManager
from src.utils.prompt_utils import (
    create_prompt_template,
    format_prompt,
    extract_answer
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTrainerConfig(TrainerConfig):
    """Configuration for prompt-based trainer."""
    
    # Prompt configuration
    prompt_type: str = "soft"  # soft, prefix, p-tuning, instruction
    prompt_length: int = 20
    prompt_initialization: str = "random"  # random, vocab, text
    
    # Prompt tuning settings
    freeze_model: bool = True
    tune_embeddings: bool = False
    reparameterization: bool = True
    prompt_dropout: float = 0.1
    
    # Template configuration
    template_type: str = "cloze"  # cloze, prefix, instruction
    verbalizer_type: str = "manual"  # manual, automatic, soft
    answer_mapping: Dict[str, int] = None
    
    # Training strategy
    use_demonstrations: bool = False
    num_demonstrations: int = 4
    demonstration_sampling: str = "random"  # random, similarity, diversity
    
    # Optimization
    prompt_learning_rate: float = 1e-3
    prompt_weight_decay: float = 0.0
    prompt_warmup_steps: int = 500
    
    # Advanced features
    use_ensemble_prompts: bool = False
    num_prompt_ensembles: int = 3
    prompt_attention: bool = True
    continuous_prompt: bool = True


class PromptTrainer(BaseTrainer):
    """
    Trainer for prompt-based learning methods.
    
    Implements various prompt tuning strategies for efficient
    model adaptation with minimal parameter updates.
    """
    
    def __init__(
        self,
        model: Union[PromptModel, nn.Module],
        config: Optional[PromptTrainerConfig] = None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        template_manager: Optional[TemplateManager] = None
    ):
        """
        Initialize prompt trainer.
        
        Args:
            model: Prompt-based model or base model
            config: Prompt training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
            template_manager: Template manager for prompt formatting
        """
        self.config = config or PromptTrainerConfig()
        
        # Setup prompt model
        if not isinstance(model, PromptModel):
            model = self._wrap_with_prompt_model(model)
        
        # Initialize base trainer
        super().__init__(
            model=model,
            config=self.config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Template manager
        self.template_manager = template_manager or TemplateManager()
        
        # Initialize prompt components
        self._initialize_prompts()
        
        # Setup prompt-specific optimizer
        self._setup_prompt_optimizer()
        
        # Freeze model if required
        if self.config.freeze_model:
            self._freeze_base_model()
        
        logger.info(
            f"Initialized PromptTrainer with {self.config.prompt_type} prompts "
            f"of length {self.config.prompt_length}"
        )
    
    def _wrap_with_prompt_model(self, model: nn.Module) -> PromptModel:
        """
        Wrap base model with prompt model.
        
        Args:
            model: Base model
            
        Returns:
            Prompt-wrapped model
        """
        return PromptModel(
            base_model=model,
            prompt_length=self.config.prompt_length,
            prompt_type=self.config.prompt_type,
            initialization=self.config.prompt_initialization
        )
    
    def _initialize_prompts(self):
        """Initialize prompt embeddings and components."""
        if self.config.prompt_type == "soft":
            self.soft_prompt = SoftPrompt(
                length=self.config.prompt_length,
                embedding_dim=self.model.get_embedding_dim(),
                initialization=self.config.prompt_initialization,
                dropout=self.config.prompt_dropout
            ).to(self.device)
        
        elif self.config.prompt_type == "prefix":
            self._initialize_prefix_prompts()
        
        elif self.config.prompt_type == "p-tuning":
            self._initialize_ptuning_prompts()
        
        # Initialize ensemble prompts if needed
        if self.config.use_ensemble_prompts:
            self._initialize_ensemble_prompts()
    
    def _initialize_prefix_prompts(self):
        """Initialize prefix tuning prompts."""
        # Get model configuration
        model_config = self.model.config
        
        # Create prefix embeddings for each layer
        self.prefix_embeddings = nn.ModuleList([
            nn.Embedding(
                self.config.prompt_length,
                model_config.hidden_size
            )
            for _ in range(model_config.num_hidden_layers)
        ])
        
        # Initialize with normal distribution
        for embedding in self.prefix_embeddings:
            nn.init.normal_(embedding.weight, mean=0, std=0.02)
    
    def _initialize_ptuning_prompts(self):
        """Initialize P-tuning v2 prompts."""
        # Create prompt encoder
        hidden_size = self.model.config.hidden_size
        
        self.prompt_encoder = nn.Sequential(
            nn.Embedding(self.config.prompt_length, hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Initialize layers
        for layer in self.prompt_encoder:
            if isinstance(layer, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(layer.weight)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def _initialize_ensemble_prompts(self):
        """Initialize ensemble of prompts."""
        self.prompt_ensemble = nn.ModuleList([
            SoftPrompt(
                length=self.config.prompt_length,
                embedding_dim=self.model.get_embedding_dim(),
                initialization="random",
                dropout=self.config.prompt_dropout
            )
            for _ in range(self.config.num_prompt_ensembles)
        ])
    
    def _freeze_base_model(self):
        """Freeze base model parameters."""
        for name, param in self.model.named_parameters():
            if "prompt" not in name and "soft" not in name:
                param.requires_grad = False
        
        # Optionally tune embeddings
        if self.config.tune_embeddings:
            if hasattr(self.model, 'get_input_embeddings'):
                for param in self.model.get_input_embeddings().parameters():
                    param.requires_grad = True
    
    def _setup_prompt_optimizer(self):
        """Setup optimizer specifically for prompt parameters."""
        # Get prompt parameters
        prompt_params = [
            p for n, p in self.model.named_parameters()
            if p.requires_grad and ("prompt" in n or "soft" in n)
        ]
        
        # Create separate optimizer for prompts
        if prompt_params:
            self.prompt_optimizer = torch.optim.AdamW(
                prompt_params,
                lr=self.config.prompt_learning_rate,
                weight_decay=self.config.prompt_weight_decay
            )
            
            # Create scheduler for prompt optimizer
            from transformers import get_linear_schedule_with_warmup
            
            total_steps = len(self.train_loader) * self.config.num_epochs
            self.prompt_scheduler = get_linear_schedule_with_warmup(
                self.prompt_optimizer,
                num_warmup_steps=self.config.prompt_warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.prompt_optimizer = None
            self.prompt_scheduler = None
    
    def _prepare_prompt_input(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare input with prompts.
        
        Args:
            batch: Original batch
            
        Returns:
            Batch with prompts
        """
        # Get text inputs
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        
        # Apply prompt template
        if self.config.template_type == "cloze":
            prompted_ids, prompted_mask = self._apply_cloze_template(
                input_ids, attention_mask
            )
        elif self.config.template_type == "prefix":
            prompted_ids, prompted_mask = self._apply_prefix_template(
                input_ids, attention_mask
            )
        else:
            prompted_ids, prompted_mask = input_ids, attention_mask
        
        # Add soft prompts if applicable
        if hasattr(self, 'soft_prompt'):
            prompted_ids, prompted_mask = self._add_soft_prompts(
                prompted_ids, prompted_mask
            )
        
        # Add demonstrations if configured
        if self.config.use_demonstrations:
            prompted_ids, prompted_mask = self._add_demonstrations(
                prompted_ids, prompted_mask, batch
            )
        
        return {
            "input_ids": prompted_ids,
            "attention_mask": prompted_mask,
            "labels": batch.get("labels")
        }
    
    def _apply_cloze_template(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cloze-style template.
        
        Args:
            input_ids: Original input IDs
            attention_mask: Attention mask
            
        Returns:
            Templated input IDs and mask
        """
        batch_size = input_ids.size(0)
        
        # Create template tokens
        template = self.template_manager.get_template("cloze")
        template_ids = self.template_manager.encode_template(template)
        
        # Insert input into template
        prompted_ids = []
        prompted_masks = []
        
        for i in range(batch_size):
            # Find mask position in template
            mask_pos = template_ids.index(self.template_manager.mask_token_id)
            
            # Construct prompted input
            prompted = torch.cat([
                template_ids[:mask_pos],
                input_ids[i],
                template_ids[mask_pos+1:]
            ])
            
            prompted_ids.append(prompted)
            
            if attention_mask is not None:
                mask = torch.cat([
                    torch.ones(mask_pos),
                    attention_mask[i],
                    torch.ones(len(template_ids) - mask_pos - 1)
                ])
                prompted_masks.append(mask)
        
        prompted_ids = torch.stack(prompted_ids)
        prompted_masks = torch.stack(prompted_masks) if prompted_masks else None
        
        return prompted_ids, prompted_masks
    
    def _apply_prefix_template(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply prefix-style template."""
        # Get prefix tokens
        prefix = self.template_manager.get_template("prefix")
        prefix_ids = self.template_manager.encode_template(prefix)
        
        # Concatenate prefix with input
        batch_size = input_ids.size(0)
        prefix_tensor = prefix_ids.unsqueeze(0).expand(batch_size, -1)
        
        prompted_ids = torch.cat([prefix_tensor, input_ids], dim=1)
        
        if attention_mask is not None:
            prefix_mask = torch.ones(batch_size, len(prefix_ids)).to(attention_mask.device)
            prompted_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        else:
            prompted_mask = None
        
        return prompted_ids, prompted_mask
    
    def _add_soft_prompts(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add soft prompt embeddings."""
        batch_size = input_ids.size(0)
        
        # Get soft prompt embeddings
        prompt_embeds = self.soft_prompt.get_prompt_embeddings(batch_size)
        
        # Combine with input embeddings
        input_embeds = self.model.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat([prompt_embeds, input_embeds], dim=1)
        
        # Update attention mask
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch_size, self.config.prompt_length
            ).to(attention_mask.device)
            combined_mask = torch.cat([prompt_mask, attention_mask], dim=1)
        else:
            combined_mask = None
        
        return combined_embeds, combined_mask
    
    def _add_demonstrations(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add few-shot demonstrations."""
        # Get demonstration examples
        demonstrations = self._sample_demonstrations(batch)
        
        # Format demonstrations
        demo_ids, demo_mask = self._format_demonstrations(demonstrations)
        
        # Concatenate with input
        prompted_ids = torch.cat([demo_ids, input_ids], dim=1)
        
        if attention_mask is not None:
            prompted_mask = torch.cat([demo_mask, attention_mask], dim=1)
        else:
            prompted_mask = None
        
        return prompted_ids, prompted_mask
    
    def _sample_demonstrations(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        """Sample demonstration examples."""
        # Implementation depends on sampling strategy
        demonstrations = []
        
        if self.config.demonstration_sampling == "random":
            # Random sampling from training set
            pass
        elif self.config.demonstration_sampling == "similarity":
            # Sample based on similarity to input
            pass
        elif self.config.demonstration_sampling == "diversity":
            # Sample diverse examples
            pass
        
        return demonstrations
    
    def _format_demonstrations(
        self,
        demonstrations: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Format demonstrations into input format."""
        # Format each demonstration
        formatted = []
        for demo in demonstrations:
            demo_text = self.template_manager.format_demonstration(demo)
            demo_ids = self.template_manager.encode_template(demo_text)
            formatted.append(demo_ids)
        
        # Concatenate demonstrations
        if formatted:
            demo_ids = torch.cat(formatted)
            demo_mask = torch.ones_like(demo_ids)
        else:
            demo_ids = torch.tensor([])
            demo_mask = torch.tensor([])
        
        return demo_ids, demo_mask
    
    def _train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute single training step with prompts.
        
        Args:
            batch: Input batch
            
        Returns:
            Loss and metrics
        """
        # Prepare prompted input
        prompted_batch = self._prepare_prompt_input(batch)
        
        # Forward pass
        if self.config.use_ensemble_prompts:
            # Ensemble prediction
            ensemble_outputs = []
            for prompt in self.prompt_ensemble:
                outputs = self.model(
                    input_ids=prompted_batch["input_ids"],
                    attention_mask=prompted_batch.get("attention_mask"),
                    labels=prompted_batch.get("labels"),
                    prompt_embeddings=prompt
                )
                ensemble_outputs.append(outputs)
            
            # Aggregate ensemble predictions
            loss = sum(o.loss for o in ensemble_outputs) / len(ensemble_outputs)
            logits = torch.stack([o.logits for o in ensemble_outputs]).mean(dim=0)
        else:
            outputs = self.model(
                input_ids=prompted_batch["input_ids"],
                attention_mask=prompted_batch.get("attention_mask"),
                labels=prompted_batch.get("labels")
            )
            loss = outputs.loss
            logits = outputs.logits
        
        # Apply verbalizer to convert to class predictions
        if self.config.verbalizer_type == "manual":
            class_logits = self._apply_verbalizer(logits)
        else:
            class_logits = logits
        
        # Calculate metrics
        with torch.no_grad():
            predictions = torch.argmax(class_logits, dim=-1)
            if prompted_batch.get("labels") is not None:
                accuracy = (predictions == prompted_batch["labels"]).float().mean().item()
            else:
                accuracy = 0.0
        
        return loss, {"loss": loss.item(), "accuracy": accuracy}
    
    def _apply_verbalizer(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply verbalizer to map token predictions to classes.
        
        Args:
            logits: Model output logits
            
        Returns:
            Class logits
        """
        # Get answer token IDs for each class
        answer_tokens = self.template_manager.get_answer_tokens()
        
        # Extract logits for answer tokens
        batch_size = logits.size(0)
        num_classes = len(answer_tokens)
        
        class_logits = torch.zeros(batch_size, num_classes).to(logits.device)
        
        for class_idx, token_ids in enumerate(answer_tokens):
            # Average logits for all tokens representing this class
            token_logits = logits[:, :, token_ids].mean(dim=-1)
            class_logits[:, class_idx] = token_logits.max(dim=-1)[0]
        
        return class_logits
    
    def optimize_prompts(
        self,
        num_steps: int = 1000,
        eval_steps: int = 100
    ) -> Dict[str, Any]:
        """
        Optimize prompts with gradient-based search.
        
        Args:
            num_steps: Number of optimization steps
            eval_steps: Steps between evaluations
            
        Returns:
            Optimization results
        """
        logger.info(f"Starting prompt optimization for {num_steps} steps")
        
        best_metric = float('-inf')
        optimization_history = []
        
        for step in range(num_steps):
            # Sample batch
            batch = next(iter(self.train_loader))
            batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in batch.items()}
            
            # Forward and backward pass
            loss, metrics = self._train_step(batch)
            loss.backward()
            
            # Update prompts
            if self.prompt_optimizer:
                self.prompt_optimizer.step()
                self.prompt_optimizer.zero_grad()
                
                if self.prompt_scheduler:
                    self.prompt_scheduler.step()
            else:
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Evaluation
            if (step + 1) % eval_steps == 0:
                val_metrics = self._validate()
                
                optimization_history.append({
                    "step": step + 1,
                    "train_loss": loss.item(),
                    "val_metrics": val_metrics
                })
                
                # Check for improvement
                current_metric = val_metrics.get("f1_macro", val_metrics.get("accuracy", 0))
                if current_metric > best_metric:
                    best_metric = current_metric
                    self._save_best_prompts()
                
                logger.info(
                    f"Step {step + 1}: Loss = {loss.item():.4f}, "
                    f"Val F1 = {current_metric:.4f}"
                )
        
        return {
            "best_metric": best_metric,
            "history": optimization_history
        }
    
    def _save_best_prompts(self):
        """Save best prompt parameters."""
        if hasattr(self, 'soft_prompt'):
            torch.save(
                self.soft_prompt.state_dict(),
                self.checkpoint_dir / "best_prompts.pt"
            )
        
        if hasattr(self, 'prompt_ensemble'):
            for i, prompt in enumerate(self.prompt_ensemble):
                torch.save(
                    prompt.state_dict(),
                    self.checkpoint_dir / f"best_prompt_ensemble_{i}.pt"
                )
