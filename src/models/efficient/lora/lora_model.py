"""
LoRA (Low-Rank Adaptation) Model Implementation
================================================

Implementation of LoRA for efficient fine-tuning of large language models,
based on:
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al. (2023): "QLoRA: Efficient Finetuning of Quantized LLMs"

Mathematical Foundation:
LoRA decomposes weight updates into low-rank matrices:
W = W_0 + ΔW = W_0 + BA
where B ∈ R^(d×r), A ∈ R^(r×k), and r << min(d,k)

This reduces trainable parameters from d×k to r×(d+k).

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA adaptation."""
    r: int = 8  # Rank of adaptation
    lora_alpha: int = 16  # LoRA scaling parameter
    lora_dropout: float = 0.1  # Dropout probability
    merge_weights: bool = False  # Merge weights after training
    target_modules: List[str] = None  # Modules to apply LoRA
    modules_to_save: List[str] = None  # Modules to train fully
    init_lora_weights: str = "gaussian"  # Initialization method
    use_rslora: bool = False  # Use rank-stabilized LoRA
    use_dora: bool = False  # Use weight-decomposed LoRA


class LoRALayer(nn.Module):
    """
    Base LoRA layer implementation.
    
    Implements low-rank decomposition for efficient adaptation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: LoRAConfig
    ):
        """
        Initialize LoRA layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            config: LoRA configuration
        """
        super().__init__()
        
        self.r = config.r
        self.lora_alpha = config.lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.merge_weights = config.merge_weights
        self.merged = False
        
        # Create low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, self.r))
        self.lora_B = nn.Parameter(torch.zeros(self.r, out_features))
        
        # Dropout
        self.lora_dropout = nn.Dropout(config.lora_dropout)
        
        # Initialize weights
        self._init_weights(config.init_lora_weights)
        
        # Rank-stabilized scaling
        if config.use_rslora:
            self.scaling = self.lora_alpha / math.sqrt(self.r)
    
    def _init_weights(self, init_method: str):
        """
        Initialize LoRA weights.
        
        Following initialization strategies from the paper:
        - A: Random Gaussian initialization
        - B: Zero initialization
        """
        if init_method == "gaussian":
            nn.init.normal_(self.lora_A, mean=0, std=0.02)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(self.lora_A)
        elif init_method == "kaiming":
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with low-rank adaptation.
        
        Computes: y = Wx + (BAx) * scaling
        """
        if self.merged:
            return x
        
        # Apply LoRA
        lora_output = x @ self.lora_A
        lora_output = self.lora_dropout(lora_output)
        lora_output = lora_output @ self.lora_B
        
        return lora_output * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Wraps a pretrained linear layer with LoRA decomposition.
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        config: LoRAConfig
    ):
        """
        Initialize LoRA linear layer.
        
        Args:
            base_layer: Pretrained linear layer
            config: LoRA configuration
        """
        super().__init__()
        
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Freeze base layer
        for param in base_layer.parameters():
            param.requires_grad = False
        
        # Add LoRA adaptation
        self.lora = LoRALayer(
            self.in_features,
            self.out_features,
            config
        )
        
        self.merged = False
        self.config = config
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining base layer and LoRA.
        
        Computes: y = (W_0 + ΔW)x = W_0x + BAx * scaling
        """
        base_output = self.base_layer(x)
        
        if self.merged:
            return base_output
        
        lora_output = self.lora(x)
        return base_output + lora_output
    
    def merge_weights(self):
        """
        Merge LoRA weights into base weights for inference.
        
        Updates: W = W_0 + BA * scaling
        """
        if self.merged:
            return
        
        with torch.no_grad():
            delta_w = (self.lora.lora_A @ self.lora.lora_B) * self.lora.scaling
            self.base_layer.weight.data += delta_w.T
        
        self.merged = True
        logger.info("Merged LoRA weights into base layer")
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from base weights."""
        if not self.merged:
            return
        
        with torch.no_grad():
            delta_w = (self.lora.lora_A @ self.lora.lora_B) * self.lora.scaling
            self.base_layer.weight.data -= delta_w.T
        
        self.merged = False
        logger.info("Unmerged LoRA weights from base layer")


@MODELS.register("lora", aliases=["lora_model"])
class LoRAModel(AGNewsBaseModel):
    """
    Model with LoRA adaptation for efficient fine-tuning.
    
    Provides:
    1. Selective layer adaptation
    2. Significant parameter reduction
    3. Multiple LoRA configurations
    4. Weight merging for deployment
    """
    
    def __init__(
        self,
        base_model: AGNewsBaseModel,
        config: Optional[LoRAConfig] = None
    ):
        """
        Initialize LoRA model.
        
        Args:
            base_model: Pretrained base model
            config: LoRA configuration
        """
        super().__init__()
        
        self.base_model = base_model
        self.config = config or LoRAConfig()
        
        # Apply LoRA to target modules
        self._apply_lora()
        
        # Calculate parameter statistics
        self._log_parameter_stats()
    
    def _apply_lora(self):
        """Apply LoRA to target modules in the base model."""
        target_modules = self.config.target_modules or [
            "query", "key", "value", "dense"
        ]
        
        lora_layers = []
        
        for name, module in self.base_model.named_modules():
            # Check if module should have LoRA
            if any(target in name for target in target_modules):
                if isinstance(module, nn.Linear):
                    # Replace with LoRA linear
                    parent_name = ".".join(name.split(".")[:-1])
                    module_name = name.split(".")[-1]
                    parent = self.base_model
                    
                    for part in parent_name.split("."):
                        if part:
                            parent = getattr(parent, part)
                    
                    # Create LoRA layer
                    lora_linear = LoRALinear(module, self.config)
                    setattr(parent, module_name, lora_linear)
                    lora_layers.append(lora_linear)
                    
                    logger.debug(f"Applied LoRA to {name}")
        
        self.lora_layers = nn.ModuleList(lora_layers)
        logger.info(f"Applied LoRA to {len(lora_layers)} layers")
    
    def _log_parameter_stats(self):
        """Log parameter statistics after LoRA application."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        reduction_ratio = 1 - (trainable_params / total_params)
        
        logger.info(
            f"LoRA Parameter Statistics:\n"
            f"  Total parameters: {total_params:,}\n"
            f"  Trainable parameters: {trainable_params:,}\n"
            f"  Reduction: {reduction_ratio:.1%}\n"
            f"  Compression ratio: {total_params/trainable_params:.1f}x"
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through LoRA-adapted model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
    
    def merge_and_save(self, save_path: str):
        """
        Merge LoRA weights and save model.
        
        Args:
            save_path: Path to save merged model
        """
        # Merge all LoRA layers
        for layer in self.lora_layers:
            layer.merge_weights()
        
        # Save merged model
        self.base_model.save(save_path)
        
        # Unmerge for continued training
        for layer in self.lora_layers:
            layer.unmerge_weights()
        
        logger.info(f"Saved merged model to {save_path}")
    
    def print_trainable_parameters(self):
        """Print trainable parameters summary."""
        trainable_params = 0
        all_param = 0
        
        for _, param in self.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(
            f"Trainable params: {trainable_params:,} || "
            f"All params: {all_param:,} || "
            f"Trainable%: {100 * trainable_params / all_param:.2f}"
        )
