"""
Instruction-based Trainer Implementation for AG News Text Classification
=========================================================================

This module implements instruction tuning and in-context learning strategies
for training models to follow natural language instructions.

Key Techniques:
- Instruction tuning (Wei et al., 2022)
- Chain-of-thought prompting (Wei et al., 2022)
- Self-Instruct training (Wang et al., 2023)
- Constitutional AI principles (Bai et al., 2022)

References:
- Wei et al. (2022): "Finetuned Language Models Are Zero-Shot Learners"
- Wei et al. (2022): "Chain-of-Thought Prompting Elicits Reasoning"
- Wang et al. (2023): "Self-Instruct: Aligning LMs with Self-Generated Instructions"
- Ouyang et al. (2022): "Training language models to follow instructions"

Author: Võ Hải Dũng
License: MIT
"""

import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.training.trainers.base_trainer import BaseTrainer, TrainerConfig
from src.models.base.base_model import AGNewsBaseModel
from src.data.preprocessing.prompt_formatter import PromptFormatter
from src.utils.logging_config import get_logger
from src.utils.prompt_utils import (
    create_instruction_prompt,
    parse_model_response,
    validate_instruction_format
)

logger = get_logger(__name__)


@dataclass
class InstructionTrainerConfig(TrainerConfig):
    """Configuration for instruction-based trainer."""
    
    # Instruction settings
    instruction_template: str = "alpaca"  # alpaca, flan, openai, custom
    use_system_prompt: bool = True
    system_prompt: str = "You are a helpful assistant for text classification."
    
    # Response formatting
    response_format: str = "structured"  # structured, natural, chain_of_thought
    require_explanation: bool = False
    max_response_length: int = 256
    
    # Instruction augmentation
    augment_instructions: bool = True
    num_instruction_variants: int = 3
    instruction_diversity_penalty: float = 0.1
    
    # Chain-of-thought
    use_chain_of_thought: bool = False
    cot_trigger: str = "Let's think step by step:"
    few_shot_cot: bool = True
    num_cot_examples: int = 2
    
    # Self-instruction
    use_self_instruct: bool = False
    self_instruct_iterations: int = 3
    bootstrap_examples: int = 175
    filter_threshold: float = 0.8
    
    # Training strategy
    instruction_dropout: float = 0.05
    response_dropout: float = 0.1
    format_strictness: float = 0.9
    
    # Loss configuration
    instruction_loss_weight: float = 0.3
    response_loss_weight: float = 0.7
    format_penalty: float = 0.1
    
    # Constitutional training
    use_constitutional: bool = False
    constitutional_principles: List[str] = field(default_factory=list)
    critique_model: Optional[str] = None
    
    # Multi-turn dialogue
    enable_multi_turn: bool = False
    max_turns: int = 3
    context_window: int = 2048


class InstructionDataset(Dataset):
    """Dataset for instruction-based training."""
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        config: InstructionTrainerConfig,
        tokenizer: Any
    ):
        """
        Initialize instruction dataset.
        
        Args:
            data: Raw data samples
            config: Instruction configuration
            tokenizer: Tokenizer for encoding
        """
        self.data = data
        self.config = config
        self.tokenizer = tokenizer
        self.formatter = PromptFormatter()
        
        # Prepare instruction templates
        self._prepare_templates()
    
    def _prepare_templates(self):
        """Prepare instruction templates based on configuration."""
        if self.config.instruction_template == "alpaca":
            self.instruction_format = (
                "Below is an instruction that describes a task, paired with an input "
                "that provides further context. Write a response that appropriately "
                "completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n"
                "### Input:\n{input}\n\n"
                "### Response:\n{response}"
            )
        elif self.config.instruction_template == "flan":
            self.instruction_format = (
                "Task: {instruction}\n"
                "Input: {input}\n"
                "Output: {response}"
            )
        else:
            self.instruction_format = "{instruction}\n\nText: {input}\n\nAnswer: {response}"
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get formatted instruction sample."""
        sample = self.data[idx]
        
        # Create instruction
        instruction = self._create_instruction(sample)
        
        # Format with template
        formatted_text = self.instruction_format.format(
            instruction=instruction,
            input=sample["text"],
            response=self._create_response(sample)
        )
        
        # Add system prompt if configured
        if self.config.use_system_prompt:
            formatted_text = f"{self.config.system_prompt}\n\n{formatted_text}"
        
        # Tokenize
        encoded = self.tokenizer(
            formatted_text,
            max_length=self.config.max_response_length + 512,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": sample.get("label", -100)
        }
    
    def _create_instruction(self, sample: Dict[str, Any]) -> str:
        """Create instruction for the sample."""
        base_instruction = "Classify the following news article into one of four categories: World, Sports, Business, or Sci/Tech."
        
        if self.config.augment_instructions:
            variants = [
                "Determine which category this news article belongs to: World, Sports, Business, or Sci/Tech.",
                "Read the article and identify its category from: World, Sports, Business, or Sci/Tech.",
                "Categorize this news text as either World, Sports, Business, or Sci/Tech."
            ]
            instruction = random.choice([base_instruction] + variants)
        else:
            instruction = base_instruction
        
        if self.config.use_chain_of_thought:
            instruction += f"\n{self.config.cot_trigger}"
        
        return instruction
    
    def _create_response(self, sample: Dict[str, Any]) -> str:
        """Create response for the sample."""
        label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
        category = label_map.get(sample.get("label", 0), "Unknown")
        
        if self.config.response_format == "structured":
            response = f"Category: {category}"
        elif self.config.response_format == "natural":
            response = f"This article belongs to the {category} category."
        elif self.config.response_format == "chain_of_thought":
            response = self._create_cot_response(sample, category)
        else:
            response = category
        
        if self.config.require_explanation and "explanation" in sample:
            response += f"\nExplanation: {sample['explanation']}"
        
        return response
    
    def _create_cot_response(self, sample: Dict[str, Any], category: str) -> str:
        """Create chain-of-thought response."""
        # Generate reasoning steps
        reasoning_templates = [
            "First, I need to identify the main topic of the article. {analysis} "
            "Based on these observations, this article is about {category}.",
            "Let me analyze the content: {analysis} "
            "Therefore, this article should be classified as {category}.",
            "Looking at the key elements: {analysis} "
            "This clearly indicates the {category} category."
        ]
        
        # Simple analysis based on keywords
        analysis = self._generate_analysis(sample["text"])
        template = random.choice(reasoning_templates)
        
        return template.format(analysis=analysis, category=category)
    
    def _generate_analysis(self, text: str) -> str:
        """Generate simple analysis for chain-of-thought."""
        # This is a simplified version - in practice, use more sophisticated analysis
        keywords = {
            "World": ["country", "government", "international", "president"],
            "Sports": ["game", "player", "team", "score", "match"],
            "Business": ["company", "market", "stock", "revenue", "CEO"],
            "Sci/Tech": ["technology", "software", "research", "computer", "science"]
        }
        
        found_keywords = []
        for category, words in keywords.items():
            for word in words:
                if word.lower() in text.lower():
                    found_keywords.append(word)
                    break
        
        if found_keywords:
            return f"The text mentions {', '.join(found_keywords[:3])}."
        return "The content suggests a specific domain."


class InstructionTrainer(BaseTrainer):
    """
    Trainer for instruction-based learning.
    
    Implements instruction tuning with support for:
    - Multiple instruction formats
    - Chain-of-thought reasoning
    - Self-instruction generation
    - Constitutional AI principles
    """
    
    def __init__(
        self,
        model: AGNewsBaseModel,
        config: Optional[InstructionTrainerConfig] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        tokenizer: Optional[Any] = None
    ):
        """
        Initialize instruction trainer.
        
        Args:
            model: Model to train
            config: Instruction training configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            tokenizer: Tokenizer for text processing
        """
        self.config = config or InstructionTrainerConfig()
        self.tokenizer = tokenizer
        
        # Create instruction datasets
        if train_dataset and not isinstance(train_dataset, InstructionDataset):
            train_dataset = InstructionDataset(train_dataset, self.config, tokenizer)
        if val_dataset and not isinstance(val_dataset, InstructionDataset):
            val_dataset = InstructionDataset(val_dataset, self.config, tokenizer)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.dataloader_num_workers
        ) if train_dataset else None
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.dataloader_num_workers
        ) if val_dataset else None
        
        # Initialize base trainer
        super().__init__(
            model=model,
            config=self.config,
            train_loader=train_loader,
            val_loader=val_loader
        )
        
        # Initialize instruction-specific components
        self._initialize_instruction_components()
        
        # Setup self-instruction if enabled
        if self.config.use_self_instruct:
            self.self_instruct_pool = []
            self._bootstrap_self_instructions()
        
        logger.info(
            f"Initialized InstructionTrainer with {self.config.instruction_template} template"
        )
    
    def _initialize_instruction_components(self):
        """Initialize components specific to instruction training."""
        # Response validator
        self.response_validator = ResponseValidator(self.config)
        
        # Instruction augmenter
        if self.config.augment_instructions:
            self.instruction_augmenter = InstructionAugmenter(self.config)
        
        # Constitutional critic if enabled
        if self.config.use_constitutional:
            self.constitutional_critic = ConstitutionalCritic(
                self.config.constitutional_principles
            )
    
    def _bootstrap_self_instructions(self):
        """Bootstrap initial self-instructions."""
        # Load seed instructions
        seed_instructions = [
            "Classify this text into a news category.",
            "Determine the topic of this article.",
            "Identify what type of news this is.",
            "Categorize this news content."
        ]
        
        # Generate variations
        for seed in seed_instructions:
            for _ in range(self.config.bootstrap_examples // len(seed_instructions)):
                variant = self._generate_instruction_variant(seed)
                self.self_instruct_pool.append(variant)
        
        logger.info(f"Bootstrapped {len(self.self_instruct_pool)} self-instructions")
    
    def _generate_instruction_variant(self, seed: str) -> str:
        """Generate instruction variant from seed."""
        # Simple variation strategies
        variations = [
            lambda s: s.replace("Classify", "Categorize"),
            lambda s: s.replace("text", "article"),
            lambda s: s.replace("Determine", "Identify"),
            lambda s: s + " Be specific.",
            lambda s: "Please " + s.lower(),
            lambda s: s.replace(".", ". Explain your reasoning.")
        ]
        
        variant = random.choice(variations)(seed)
        return variant
    
    def _compute_instruction_loss(
        self,
        outputs: Any,
        labels: torch.Tensor,
        responses: List[str]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for instruction following.
        
        Args:
            outputs: Model outputs
            labels: Ground truth labels
            responses: Generated responses
            
        Returns:
            Loss and metrics
        """
        # Standard classification loss
        if hasattr(outputs, 'loss'):
            classification_loss = outputs.loss
        else:
            classification_loss = F.cross_entropy(outputs.logits, labels)
        
        # Response format loss
        format_loss = torch.tensor(0.0).to(self.device)
        valid_responses = 0
        
        for response in responses:
            if self.response_validator.validate(response):
                valid_responses += 1
            else:
                format_loss += self.config.format_penalty
        
        format_loss = format_loss / len(responses)
        
        # Combined loss
        total_loss = (
            self.config.response_loss_weight * classification_loss +
            self.config.instruction_loss_weight * format_loss
        )
        
        # Metrics
        metrics = {
            "classification_loss": classification_loss.item(),
            "format_loss": format_loss.item(),
            "valid_response_rate": valid_responses / len(responses)
        }
        
        return total_loss, metrics
    
    def _train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Execute single instruction training step.
        
        Args:
            batch: Input batch
            
        Returns:
            Loss and metrics
        """
        # Apply instruction dropout
        if self.training and random.random() < self.config.instruction_dropout:
            # Skip instruction for robustness
            batch = self._remove_instruction(batch)
        
        # Forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask")
        )
        
        # Generate responses
        with torch.no_grad():
            responses = self._generate_responses(outputs, batch)
        
        # Compute instruction-aware loss
        loss, metrics = self._compute_instruction_loss(
            outputs,
            batch.get("labels"),
            responses
        )
        
        # Constitutional critique if enabled
        if self.config.use_constitutional and self.training:
            critique_loss = self._apply_constitutional_critique(responses)
            loss += critique_loss
            metrics["critique_loss"] = critique_loss.item()
        
        return loss, metrics
    
    def _generate_responses(
        self,
        outputs: Any,
        batch: Dict[str, torch.Tensor]
    ) -> List[str]:
        """Generate text responses from model outputs."""
        responses = []
        
        if hasattr(outputs, 'logits'):
            # Decode predictions to text
            predictions = torch.argmax(outputs.logits, dim=-1)
            
            for pred in predictions:
                # Convert to text response
                label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
                category = label_map.get(pred.item(), "Unknown")
                
                if self.config.response_format == "structured":
                    response = f"Category: {category}"
                else:
                    response = category
                
                responses.append(response)
        
        return responses
    
    def _remove_instruction(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Remove instruction from batch for dropout."""
        # This is a simplified implementation
        # In practice, would need to modify input_ids to remove instruction tokens
        return batch
    
    def _apply_constitutional_critique(self, responses: List[str]) -> torch.Tensor:
        """Apply constitutional AI critique to responses."""
        critique_loss = torch.tensor(0.0).to(self.device)
        
        for response in responses:
            critique = self.constitutional_critic.critique(response)
            if critique["violations"]:
                critique_loss += len(critique["violations"]) * 0.1
        
        return critique_loss / len(responses)
    
    def train_with_self_instruct(
        self,
        num_iterations: int = None
    ) -> Dict[str, Any]:
        """
        Train with self-instruction generation.
        
        Args:
            num_iterations: Number of self-instruct iterations
            
        Returns:
            Training results
        """
        num_iterations = num_iterations or self.config.self_instruct_iterations
        
        logger.info(f"Starting self-instruct training for {num_iterations} iterations")
        
        results = []
        
        for iteration in range(num_iterations):
            logger.info(f"Self-instruct iteration {iteration + 1}/{num_iterations}")
            
            # Generate new instructions
            new_instructions = self._generate_self_instructions()
            
            # Filter instructions
            filtered_instructions = self._filter_instructions(new_instructions)
            
            # Add to pool
            self.self_instruct_pool.extend(filtered_instructions)
            
            # Create augmented dataset
            augmented_dataset = self._create_augmented_dataset(filtered_instructions)
            
            # Train on augmented data
            train_results = self.train(num_epochs=1)
            
            results.append({
                "iteration": iteration + 1,
                "new_instructions": len(filtered_instructions),
                "total_pool_size": len(self.self_instruct_pool),
                "train_results": train_results
            })
            
            logger.info(
                f"Generated {len(filtered_instructions)} new instructions, "
                f"pool size: {len(self.self_instruct_pool)}"
            )
        
        return {"iterations": results}
    
    def _generate_self_instructions(self) -> List[str]:
        """Generate new self-instructions using the model."""
        new_instructions = []
        
        # Sample from existing pool
        num_samples = min(10, len(self.self_instruct_pool))
        seed_instructions = random.sample(self.self_instruct_pool, num_samples)
        
        for seed in seed_instructions:
            # Generate variations
            for _ in range(5):
                variant = self._generate_instruction_variant(seed)
                if variant not in self.self_instruct_pool:
                    new_instructions.append(variant)
        
        return new_instructions
    
    def _filter_instructions(self, instructions: List[str]) -> List[str]:
        """Filter instructions based on quality criteria."""
        filtered = []
        
        for instruction in instructions:
            # Check validity
            if validate_instruction_format(instruction):
                # Check diversity
                if self._check_diversity(instruction):
                    filtered.append(instruction)
        
        return filtered
    
    def _check_diversity(self, instruction: str) -> bool:
        """Check if instruction is diverse enough."""
        # Simple diversity check based on similarity
        for existing in self.self_instruct_pool[-50:]:  # Check last 50
            if self._compute_similarity(instruction, existing) > 0.9:
                return False
        return True
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts."""
        # Simple Jaccard similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _create_augmented_dataset(
        self,
        instructions: List[str]
    ) -> Dataset:
        """Create dataset with augmented instructions."""
        # This would create a new dataset with the generated instructions
        # Implementation depends on specific dataset structure
        pass


class ResponseValidator:
    """Validator for instruction response format."""
    
    def __init__(self, config: InstructionTrainerConfig):
        """Initialize response validator."""
        self.config = config
        
        # Define validation rules
        self.rules = self._define_validation_rules()
    
    def _define_validation_rules(self) -> List[Any]:
        """Define validation rules based on response format."""
        rules = []
        
        if self.config.response_format == "structured":
            rules.append(lambda r: "Category:" in r or "category:" in r)
            rules.append(lambda r: any(cat in r for cat in ["World", "Sports", "Business", "Sci/Tech"]))
        
        if self.config.require_explanation:
            rules.append(lambda r: "Explanation:" in r or "because" in r.lower())
        
        return rules
    
    def validate(self, response: str) -> bool:
        """
        Validate response format.
        
        Args:
            response: Response to validate
            
        Returns:
            Whether response is valid
        """
        if not response:
            return False
        
        # Check all rules
        for rule in self.rules:
            if not rule(response):
                return False
        
        return True


class InstructionAugmenter:
    """Augmenter for instruction variations."""
    
    def __init__(self, config: InstructionTrainerConfig):
        """Initialize instruction augmenter."""
        self.config = config
        
        # Define augmentation strategies
        self.strategies = [
            self._paraphrase,
            self._add_context,
            self._change_tone,
            self._add_constraints
        ]
    
    def augment(self, instruction: str) -> List[str]:
        """
        Generate augmented instruction variants.
        
        Args:
            instruction: Original instruction
            
        Returns:
            List of augmented instructions
        """
        variants = [instruction]
        
        for _ in range(self.config.num_instruction_variants - 1):
            strategy = random.choice(self.strategies)
            variant = strategy(instruction)
            if variant != instruction:
                variants.append(variant)
        
        return variants
    
    def _paraphrase(self, instruction: str) -> str:
        """Paraphrase instruction."""
        paraphrases = {
            "Classify": "Categorize",
            "Determine": "Identify",
            "article": "text",
            "belongs to": "falls under"
        }
        
        result = instruction
        for original, replacement in paraphrases.items():
            if original in result:
                result = result.replace(original, replacement)
                break
        
        return result
    
    def _add_context(self, instruction: str) -> str:
        """Add context to instruction."""
        contexts = [
            "Given the content below, ",
            "Based on the text, ",
            "After reading, "
        ]
        return random.choice(contexts) + instruction.lower()
    
    def _change_tone(self, instruction: str) -> str:
        """Change tone of instruction."""
        if random.random() < 0.5:
            return "Please " + instruction.lower()
        else:
            return instruction + " Be precise."
    
    def _add_constraints(self, instruction: str) -> str:
        """Add constraints to instruction."""
        constraints = [
            " Choose only one category.",
            " Be specific in your answer.",
            " Provide a clear response."
        ]
        return instruction + random.choice(constraints)


class ConstitutionalCritic:
    """Critic for constitutional AI principles."""
    
    def __init__(self, principles: List[str]):
        """
        Initialize constitutional critic.
        
        Args:
            principles: List of constitutional principles
        """
        self.principles = principles or [
            "Be helpful and harmless",
            "Provide accurate information",
            "Avoid biased statements",
            "Be respectful and professional"
        ]
    
    def critique(self, response: str) -> Dict[str, Any]:
        """
        Critique response based on principles.
        
        Args:
            response: Response to critique
            
        Returns:
            Critique results
        """
        violations = []
        suggestions = []
        
        # Check each principle
        for principle in self.principles:
            if not self._check_principle(response, principle):
                violations.append(principle)
                suggestions.append(self._get_suggestion(principle))
        
        return {
            "violations": violations,
            "suggestions": suggestions,
            "score": 1.0 - len(violations) / len(self.principles)
        }
    
    def _check_principle(self, response: str, principle: str) -> bool:
        """Check if response adheres to principle."""
        # Simplified checks - in practice, use more sophisticated methods
        if "accurate" in principle.lower():
            # Check for hedging language that might indicate uncertainty
            uncertain_terms = ["maybe", "possibly", "might be", "could be"]
            return not any(term in response.lower() for term in uncertain_terms)
        
        if "biased" in principle.lower():
            # Check for potentially biased language
            biased_terms = ["obviously", "clearly", "everyone knows"]
            return not any(term in response.lower() for term in biased_terms)
        
        return True
    
    def _get_suggestion(self, principle: str) -> str:
        """Get suggestion for principle violation."""
        suggestions = {
            "accurate": "Ensure factual accuracy and avoid speculation.",
            "biased": "Use neutral language and avoid subjective statements.",
            "helpful": "Provide clear and useful information.",
            "respectful": "Maintain professional tone."
        }
        
        for key, suggestion in suggestions.items():
            if key in principle.lower():
                return suggestion
        
        return "Review and improve response quality."
