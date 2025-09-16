"""
Prompted Dataset for Instruction-based Learning
===============================================

Implements prompted/instruction-based dataset following:
- Brown et al. (2020): "Language Models are Few-Shot Learners"
- Wei et al. (2022): "Finetuned Language Models are Zero-Shot Learners"
- Sanh et al. (2022): "Multitask Prompted Training Enables Zero-Shot Task Generalization"

Author: Võ Hải Dũng
License: MIT
"""

import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.constants import AG_NEWS_CLASSES, ID_TO_LABEL
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class PromptConfig:
    """Configuration for prompted dataset."""
    
    # Prompt templates
    templates: List[str] = None
    
    # Instruction templates  
    instructions: List[str] = None
    
    # Few-shot examples
    num_examples: int = 0
    example_separator: str = "\n\n"
    
    # Format options
    add_explanation: bool = False
    add_options: bool = True
    shuffle_options: bool = False
    
    # Special tokens
    input_token: str = "[INPUT]"
    output_token: str = "[OUTPUT]"
    instruction_token: str = "[INSTRUCTION]"
    
    def __post_init__(self):
        """Initialize default templates."""
        if self.templates is None:
            self.templates = [
                "Classify the following news article into one of these categories: {options}.\n\nArticle: {text}\n\nCategory:",
                "What category does this news belong to? Options: {options}\n\nNews: {text}\n\nAnswer:",
                "Read the following news and determine its category from {options}.\n\nText: {text}\n\nThe category is:",
            ]
        
        if self.instructions is None:
            self.instructions = [
                "You are a news classification expert. Classify news articles into the correct category.",
                "Task: Categorize news articles into World, Sports, Business, or Sci/Tech.",
                "Instruction: Identify the topic category of the given news article.",
            ]

class PromptedDataset(Dataset):
    """
    Dataset with prompting/instruction formatting.
    
    Implements prompting strategies from:
    - Liu et al. (2023): "Pre-train, Prompt, and Predict"
    - Schick & Schütze (2021): "Exploiting Cloze Questions for Few Shot Text Classification"
    """
    
    def __init__(
        self,
        base_dataset: Dataset,
        prompt_config: PromptConfig,
        tokenizer: Optional[Any] = None,
        seed: int = 42
    ):
        """
        Initialize prompted dataset.
        
        Args:
            base_dataset: Base dataset to wrap
            prompt_config: Prompt configuration
            tokenizer: Optional tokenizer
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.config = prompt_config
        self.tokenizer = tokenizer
        self.rng = random.Random(seed)
        
        # Cache few-shot examples if needed
        if self.config.num_examples > 0:
            self._prepare_examples()
        
        logger.info(f"Created prompted dataset with {len(self)} samples")
    
    def _prepare_examples(self):
        """Prepare few-shot examples."""
        self.examples_by_class = {i: [] for i in range(len(AG_NEWS_CLASSES))}
        
        # Collect examples for each class
        for item in self.base_dataset:
            label = item.get("label", item.get("labels"))
            if isinstance(label, torch.Tensor):
                label = label.item()
            
            if len(self.examples_by_class[label]) < 5:  # Keep 5 examples per class
                self.examples_by_class[label].append(item)
    
    def _format_prompt(self, text: str, label: Optional[int] = None) -> str:
        """
        Format text with prompt template.
        
        Following prompting best practices from:
        - Reynolds & McDonell (2021): "Prompt Programming for Large Language Models"
        """
        # Select random template
        template = self.rng.choice(self.config.templates)
        
        # Prepare options
        options = list(AG_NEWS_CLASSES)
        if self.config.shuffle_options:
            self.rng.shuffle(options)
        options_str = ", ".join(options)
        
        # Format prompt
        prompt = template.format(
            text=text,
            options=options_str
        )
        
        # Add few-shot examples
        if self.config.num_examples > 0:
            examples = self._get_few_shot_examples(label)
            prompt = examples + self.config.example_separator + prompt
        
        # Add instruction
        if self.config.instructions:
            instruction = self.rng.choice(self.config.instructions)
            prompt = f"{instruction}\n\n{prompt}"
        
        return prompt
    
    def _get_few_shot_examples(self, exclude_label: Optional[int] = None) -> str:
        """Get few-shot examples."""
        examples = []
        
        # Sample examples from each class
        for class_idx in range(len(AG_NEWS_CLASSES)):
            if class_idx == exclude_label:
                continue
                
            if self.examples_by_class[class_idx]:
                example = self.rng.choice(self.examples_by_class[class_idx])
                text = example.get("text", "")
                
                # Format example
                example_prompt = f"Article: {text}\nCategory: {ID_TO_LABEL[class_idx]}"
                examples.append(example_prompt)
        
        # Limit to configured number
        examples = examples[:self.config.num_examples]
        
        return self.config.example_separator.join(examples)
    
    def _format_instruction(self, text: str, label: Optional[int] = None) -> Dict[str, str]:
        """
        Format as instruction-following task.
        
        Following instruction tuning from:
        - Wang et al. (2022): "Self-Instruct: Aligning Language Models with Self-Generated Instructions"
        """
        instruction = self.rng.choice(self.config.instructions)
        
        # Create input-output pair
        formatted = {
            "instruction": instruction,
            "input": text,
            "output": ID_TO_LABEL[label] if label is not None else ""
        }
        
        # Add explanation if configured
        if self.config.add_explanation and label is not None:
            explanations = {
                0: "This article discusses international events and world affairs.",
                1: "This article is about sports, athletes, or sporting events.",
                2: "This article covers business, economics, or financial topics.",
                3: "This article relates to science, technology, or technical innovations."
            }
            formatted["explanation"] = explanations.get(label, "")
        
        return formatted
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.base_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get prompted item.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with prompted inputs
        """
        # Get base item
        item = self.base_dataset[idx]
        
        # Extract text and label
        if isinstance(item, dict):
            text = item.get("text", "")
            label = item.get("label", item.get("labels"))
        else:
            text = str(item[0]) if len(item) > 0 else ""
            label = item[1] if len(item) > 1 else None
        
        if isinstance(label, torch.Tensor):
            label = label.item()
        
        # Format with prompt
        prompted_text = self._format_prompt(text, label)
        
        # Format as instruction if configured
        if self.config.instructions:
            instruction_format = self._format_instruction(text, label)
            
            # Combine into single text
            full_text = f"{instruction_format['instruction']}\n\n"
            full_text += f"Input: {instruction_format['input']}\n\n"
            full_text += f"Output: {instruction_format['output']}"
        else:
            full_text = prompted_text
            if label is not None:
                full_text += f" {ID_TO_LABEL[label]}"
        
        # Tokenize if tokenizer provided
        if self.tokenizer:
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=512,  # Longer for prompts
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(label, dtype=torch.long) if label is not None else torch.tensor(-100)
            }
        else:
            return {
                "prompted_text": full_text,
                "original_text": text,
                "label": label,
                "label_name": ID_TO_LABEL[label] if label is not None else None
            }

def create_prompted_dataset(
    base_dataset: Dataset,
    prompt_type: str = "classification",
    tokenizer: Optional[Any] = None,
    **kwargs
) -> PromptedDataset:
    """
    Factory function to create prompted datasets.
    
    Args:
        base_dataset: Base dataset to wrap
        prompt_type: Type of prompting (classification/instruction/few_shot)
        tokenizer: Optional tokenizer
        **kwargs: Additional configuration
        
    Returns:
        PromptedDataset instance
    """
    # Configure based on prompt type
    if prompt_type == "classification":
        config = PromptConfig(
            num_examples=0,
            add_explanation=False,
            **kwargs
        )
    elif prompt_type == "instruction":
        config = PromptConfig(
            num_examples=0,
            add_explanation=True,
            **kwargs
        )
    elif prompt_type == "few_shot":
        config = PromptConfig(
            num_examples=kwargs.get("num_examples", 3),
            add_explanation=False,
            **kwargs
        )
    else:
        config = PromptConfig(**kwargs)
    
    return PromptedDataset(base_dataset, config, tokenizer)
