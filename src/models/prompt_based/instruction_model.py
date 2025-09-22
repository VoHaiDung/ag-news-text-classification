"""
Instruction-Following Model for AG News Classification
======================================================

Implementation of instruction-tuned models that follow natural language instructions
for text classification, based on:
- Wei et al. (2022): "Finetuned Language Models are Zero-Shot Learners"
- Sanh et al. (2022): "Multitask Prompted Training Enables Zero-Shot Task Generalization"
- Wang et al. (2022): "Self-Instruct: Aligning Language Models with Self-Generated Instructions"

The model reformulates classification as instruction-following, enabling:
1. Zero-shot and few-shot learning
2. Task description understanding
3. Chain-of-thought reasoning
4. Multi-step classification

Mathematical Foundation:
P(y|x, i) = LM(y | concat(instruction, x))
where i is the task instruction and LM is the language model.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    GenerationConfig
)

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class InstructionTemplate:
    """
    Template for formatting instructions and inputs.
    
    Provides structured formatting for instruction-following models,
    ensuring consistent prompt formatting across different tasks.
    """
    
    # AG News specific templates
    AG_NEWS_TEMPLATES = {
        "zero_shot": """Classify the following news article into one of these categories: World, Sports, Business, or Technology.

Article: {text}

Category:""",
        
        "few_shot": """Classify news articles into categories.

Examples:
{examples}

Article: {text}

Category:""",
        
        "chain_of_thought": """Classify the following news article step by step.

First, identify the main topic of the article.
Then, determine which category it best fits: World, Sports, Business, or Technology.
Explain your reasoning before providing the final answer.

Article: {text}

Let me analyze this step by step:
1. Main topic:""",
        
        "detailed": """You are a news categorization expert. Your task is to classify news articles into one of four categories:
- World: International news, politics, global events
- Sports: Sports events, athletes, competitions
- Business: Companies, markets, economy, finance
- Technology: Tech companies, innovations, science

Read the article carefully and determine its category based on the primary focus.

Article: {text}

Analysis:""",
        
        "structured": """Task: News Article Classification
Categories: [World, Sports, Business, Technology]
Instructions: Identify the most appropriate category for the given article.

Input: {text}

Output:"""
    }
    
    def __init__(self, template_type: str = "zero_shot"):
        """
        Initialize instruction template.
        
        Args:
            template_type: Type of template to use
        """
        self.template_type = template_type
        self.template = self.AG_NEWS_TEMPLATES.get(
            template_type,
            self.AG_NEWS_TEMPLATES["zero_shot"]
        )
    
    def format(
        self,
        text: str,
        examples: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Format instruction with input text.
        
        Args:
            text: Input text to classify
            examples: Few-shot examples
            
        Returns:
            Formatted instruction prompt
        """
        if self.template_type == "few_shot" and examples:
            example_str = self._format_examples(examples)
            return self.template.format(text=text, examples=example_str)
        else:
            return self.template.format(text=text)
    
    def _format_examples(self, examples: List[Dict[str, str]]) -> str:
        """Format few-shot examples."""
        formatted = []
        for ex in examples:
            formatted.append(f"Article: {ex['text']}\nCategory: {ex['label']}")
        return "\n\n".join(formatted)


@dataclass
class InstructionModelConfig:
    """Configuration for instruction-following models."""
    
    # Model configuration
    model_name: str = "google/flan-t5-large"
    model_type: str = "seq2seq"  # "seq2seq" or "causal"
    max_length: int = 512
    max_new_tokens: int = 50
    
    # Instruction configuration
    instruction_template: str = "zero_shot"
    use_system_prompt: bool = True
    include_reasoning: bool = False
    
    # Generation configuration
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    do_sample: bool = False
    repetition_penalty: float = 1.0
    
    # Few-shot configuration
    num_examples: int = 0
    example_selection: str = "random"  # "random", "similar", "diverse"
    
    # Optimization
    use_cache: bool = True
    batch_decode: bool = True
    
    # Label mapping
    label_map: Dict[str, int] = field(default_factory=lambda: {
        "world": 0, "sports": 1, "business": 2, "technology": 3
    })
    
    # Advanced features
    use_chain_of_thought: bool = False
    self_consistency: bool = False  # Multiple sampling for consistency
    num_consistency_samples: int = 5


class InstructionFollowingModel(AGNewsBaseModel):
    """
    Instruction-following model for text classification.
    
    Transforms classification into an instruction-following task,
    leveraging large language models' ability to understand and
    follow natural language instructions.
    
    The model can operate in multiple modes:
    1. Zero-shot: Classification based on task description alone
    2. Few-shot: Including examples in the instruction
    3. Chain-of-thought: Step-by-step reasoning
    4. Self-consistency: Multiple reasoning paths
    """
    
    def __init__(self, config: Optional[InstructionModelConfig] = None):
        """
        Initialize instruction-following model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config or InstructionModelConfig()
        
        # Initialize model and tokenizer
        self._init_model()
        
        # Initialize instruction templates
        self.template = InstructionTemplate(self.config.instruction_template)
        
        # Setup generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            num_beams=self.config.num_beams,
            do_sample=self.config.do_sample,
            repetition_penalty=self.config.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Cache for few-shot examples
        self.example_cache = []
        
        logger.info(
            f"Initialized instruction model: {self.config.model_name} "
            f"with template: {self.config.instruction_template}"
        )
    
    def _init_model(self):
        """Initialize the underlying language model."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                use_fast=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if self.config.model_type == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            else:  # causal
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config.model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    device_map="auto" if torch.cuda.is_available() else None
                )
            
            # Set to evaluation mode by default
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def add_examples(self, examples: List[Dict[str, str]]):
        """
        Add few-shot examples for in-context learning.
        
        Args:
            examples: List of example dictionaries with 'text' and 'label'
        """
        self.example_cache.extend(examples)
        logger.info(f"Added {len(examples)} examples to cache (total: {len(self.example_cache)})")
    
    def _select_examples(self, text: str, num_examples: int) -> List[Dict[str, str]]:
        """
        Select relevant examples for few-shot learning.
        
        Args:
            text: Input text
            num_examples: Number of examples to select
            
        Returns:
            Selected examples
        """
        if not self.example_cache or num_examples == 0:
            return []
        
        if self.config.example_selection == "random":
            import random
            return random.sample(
                self.example_cache,
                min(num_examples, len(self.example_cache))
            )
        
        elif self.config.example_selection == "similar":
            # Select most similar examples (requires embeddings)
            # Simplified: return first N examples
            return self.example_cache[:num_examples]
        
        else:  # diverse
            # Select diverse examples
            # Simplified: return evenly spaced examples
            step = max(1, len(self.example_cache) // num_examples)
            return self.example_cache[::step][:num_examples]
    
    def _prepare_instruction(
        self,
        texts: List[str],
        include_examples: bool = True
    ) -> List[str]:
        """
        Prepare instruction prompts for batch processing.
        
        Args:
            texts: Input texts
            include_examples: Whether to include few-shot examples
            
        Returns:
            Formatted instruction prompts
        """
        prompts = []
        
        for text in texts:
            # Select examples if needed
            examples = None
            if include_examples and self.config.num_examples > 0:
                examples = self._select_examples(text, self.config.num_examples)
            
            # Format instruction
            prompt = self.template.format(text, examples)
            
            # Add system prompt if configured
            if self.config.use_system_prompt:
                system = "You are a helpful assistant specialized in news classification."
                prompt = f"{system}\n\n{prompt}"
            
            prompts.append(prompt)
        
        return prompts
    
    def _parse_output(self, generated_text: str) -> int:
        """
        Parse generated text to extract classification label.
        
        Args:
            generated_text: Generated text from model
            
        Returns:
            Predicted label index
        """
        # Clean and normalize text
        text_lower = generated_text.lower().strip()
        
        # Direct label matching
        for label_text, label_id in self.config.label_map.items():
            if label_text in text_lower:
                return label_id
        
        # Check for variations
        label_variations = {
            "world": ["world", "international", "global"],
            "sports": ["sports", "sport", "athletic"],
            "business": ["business", "economy", "finance", "market"],
            "technology": ["technology", "tech", "science", "innovation"]
        }
        
        for label_text, variations in label_variations.items():
            for var in variations:
                if var in text_lower:
                    return self.config.label_map[label_text]
        
        # Default to first label if no match
        logger.warning(f"Could not parse label from: {generated_text[:100]}")
        return 0
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        raw_texts: Optional[List[str]] = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass for instruction-following classification.
        
        Args:
            input_ids: Tokenized input IDs (may be ignored if raw_texts provided)
            attention_mask: Attention mask
            labels: Target labels
            raw_texts: Raw text inputs for instruction formatting
            **kwargs: Additional arguments
            
        Returns:
            Model outputs with predictions
        """
        # Prepare instructions
        if raw_texts is None:
            # Decode input_ids to get raw texts
            raw_texts = self.tokenizer.batch_decode(
                input_ids,
                skip_special_tokens=True
            )
        
        prompts = self._prepare_instruction(
            raw_texts,
            include_examples=(self.config.num_examples > 0)
        )
        
        # Tokenize prompts
        encoded = self.tokenizer(
            prompts,
            max_length=self.config.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        # Generate predictions
        with torch.no_grad():
            if self.config.self_consistency:
                # Generate multiple samples for self-consistency
                all_predictions = []
                
                for _ in range(self.config.num_consistency_samples):
                    outputs = self.model.generate(
                        **encoded,
                        generation_config=self.generation_config,
                        do_sample=True  # Enable sampling for diversity
                    )
                    
                    # Decode outputs
                    generated_texts = self.tokenizer.batch_decode(
                        outputs,
                        skip_special_tokens=True
                    )
                    
                    # Parse predictions
                    predictions = [self._parse_output(text) for text in generated_texts]
                    all_predictions.append(predictions)
                
                # Majority voting for final prediction
                import numpy as np
                all_predictions = np.array(all_predictions)
                final_predictions = []
                
                for i in range(all_predictions.shape[1]):
                    votes = all_predictions[:, i]
                    unique, counts = np.unique(votes, return_counts=True)
                    final_predictions.append(unique[np.argmax(counts)])
                
                predictions = torch.tensor(final_predictions, device=device)
                
            else:
                # Single generation
                outputs = self.model.generate(
                    **encoded,
                    generation_config=self.generation_config
                )
                
                # Decode outputs
                generated_texts = self.tokenizer.batch_decode(
                    outputs,
                    skip_special_tokens=True
                )
                
                # Parse predictions
                predictions = [self._parse_output(text) for text in generated_texts]
                predictions = torch.tensor(predictions, device=device)
        
        # Create logits (pseudo-logits for compatibility)
        batch_size = len(prompts)
        num_classes = len(self.config.label_map)
        logits = torch.zeros(batch_size, num_classes, device=device)
        
        # Set high confidence for predicted classes
        for i, pred in enumerate(predictions):
            logits[i, pred] = 10.0  # High logit for predicted class
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        
        return ModelOutputs(
            loss=loss,
            logits=logits,
            predictions=predictions,
            metadata={
                "generated_texts": generated_texts if not self.config.self_consistency else None,
                "prompts": prompts[:2],  # Store first 2 prompts for inspection
                "template_type": self.config.instruction_template
            }
        )
    
    def generate_explanation(
        self,
        text: str,
        predicted_label: Optional[int] = None
    ) -> str:
        """
        Generate explanation for classification decision.
        
        Args:
            text: Input text
            predicted_label: Predicted label (will predict if not provided)
            
        Returns:
            Explanation text
        """
        # Create explanation prompt
        label_names = {v: k for k, v in self.config.label_map.items()}
        
        if predicted_label is None:
            # First predict the label
            outputs = self.forward(
                input_ids=None,
                raw_texts=[text]
            )
            predicted_label = outputs.predictions[0].item()
        
        label_name = label_names[predicted_label]
        
        explanation_prompt = f"""Explain why the following news article belongs to the category '{label_name}'.
        
Article: {text}

The article is classified as '{label_name}' because:"""
        
        # Generate explanation
        encoded = self.tokenizer(
            explanation_prompt,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
        
        device = next(self.model.parameters()).device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **encoded,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
        
        explanation = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return explanation
    
    def adapt_to_new_categories(
        self,
        new_categories: Dict[str, int],
        examples: Optional[List[Dict[str, str]]] = None
    ):
        """
        Adapt model to new category definitions.
        
        Args:
            new_categories: New category mapping
            examples: Optional examples for the new categories
        """
        # Update label map
        self.config.label_map = new_categories
        
        # Clear and update example cache
        if examples:
            self.example_cache = examples
            
        # Update instruction template if needed
        # This would require modifying the template to mention new categories
        
        logger.info(f"Adapted to new categories: {list(new_categories.keys())}")


# Register model
MODELS.register("instruction_model", InstructionFollowingModel)


# Export public API
__all__ = [
    'InstructionTemplate',
    'InstructionModelConfig',
    'InstructionFollowingModel'
]
