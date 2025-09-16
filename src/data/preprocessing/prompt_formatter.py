"""
Prompt Formatting Module
========================

Formats data for prompt-based learning following:
- Brown et al. (2020): "Language Models are Few-Shot Learners"
- Liu et al. (2023): "Pre-train, Prompt, and Predict"
- Wei et al. (2022): "Chain of Thought Prompting"

Author: Võ Hải Dũng
License: MIT
"""

import logging
import random
from typing import List, Dict, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from configs.constants import AG_NEWS_CLASSES, ID_TO_LABEL

logger = setup_logging(__name__)

@dataclass
class PromptFormatterConfig:
    """Configuration for prompt formatting."""
    
    # Template selection
    template_style: str = "classification"  # classification, instruction, chat
    use_demonstrations: bool = False
    num_demonstrations: int = 3
    
    # Chain of thought
    use_cot: bool = False
    cot_trigger: str = "Let's think step by step."
    
    # Format options
    include_options: bool = True
    shuffle_options: bool = False
    use_letters: bool = False  # A), B), C), D) instead of names
    
    # Instructions
    task_description: str = "Classify the following news article into the correct category."
    
    # Special formatting
    highlight_keywords: bool = False
    add_metadata: bool = False
    
    # Templates
    templates: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize default templates if not provided."""
        if not self.templates:
            self.templates = self._get_default_templates()
    
    def _get_default_templates(self) -> List[str]:
        """Get default templates based on style."""
        if self.template_style == "classification":
            return [
                "{task}\n\nArticle: {text}\n\nCategories: {options}\n\nCategory:",
                "Task: {task}\n\nText: {text}\n\nOptions: {options}\n\nAnswer:",
            ]
        elif self.template_style == "instruction":
            return [
                "### Instruction:\n{task}\n\n### Input:\n{text}\n\n### Response:",
                "[INST] {task}\n\n{text} [/INST]",
            ]
        else:  # chat
            return [
                "User: {task}\n\n{text}\n\nAssistant:",
                "Human: {task}\n\nArticle: {text}\n\nAI:",
            ]

class PromptFormatter:
    """
    Format prompts for various prompt-based learning approaches.
    
    Implements formatting strategies from:
    - Schick & Schütze (2021): "It's Not Just Size That Matters"
    - Gao et al. (2021): "Making Pre-trained Language Models Better Few-shot Learners"
    """
    
    def __init__(
        self,
        config: Optional[PromptFormatterConfig] = None,
        demonstrations: Optional[List[Dict[str, Any]]] = None,
        seed: int = 42
    ):
        """
        Initialize prompt formatter.
        
        Args:
            config: Formatting configuration
            demonstrations: Few-shot demonstration examples
            seed: Random seed
        """
        self.config = config or PromptFormatterConfig()
        self.demonstrations = demonstrations or []
        self.rng = random.Random(seed)
        
        logger.info(f"Initialized prompt formatter: style={self.config.template_style}")
    
    def format_single(
        self,
        text: str,
        label: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Format single example as prompt.
        
        Args:
            text: Input text
            label: Optional label
            metadata: Optional metadata
            
        Returns:
            Formatted prompt
        """
        # Select template
        template = self.rng.choice(self.config.templates)
        
        # Prepare options
        options = self._format_options()
        
        # Create prompt
        prompt = template.format(
            task=self.config.task_description,
            text=text,
            options=options
        )
        
        # Add demonstrations if configured
        if self.config.use_demonstrations and self.demonstrations:
            demos = self._format_demonstrations(exclude_label=label)
            prompt = demos + "\n\n" + prompt
        
        # Add chain of thought if configured
        if self.config.use_cot:
            prompt = self._add_chain_of_thought(prompt)
        
        # Add metadata if configured
        if self.config.add_metadata and metadata:
            prompt = self._add_metadata(prompt, metadata)
        
        # Add answer if label provided
        if label is not None:
            answer = self._format_answer(label)
            prompt = prompt + " " + answer
        
        return prompt
    
    def format_for_instruction_tuning(
        self,
        text: str,
        label: int,
        explanation: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Format for instruction tuning.
        
        Following instruction tuning from:
        - Wang et al. (2022): "Super-NaturalInstructions"
        - Ouyang et al. (2022): "Training language models to follow instructions"
        """
        instruction = self.config.task_description
        
        # Add detailed task definition
        instruction += "\n\nThe categories are:\n"
        for i, category in enumerate(AG_NEWS_CLASSES):
            instruction += f"- {category}: "
            
            # Add category descriptions
            descriptions = {
                "World": "International news, politics, and global events",
                "Sports": "Sports news, games, athletes, and competitions",
                "Business": "Business, finance, economy, and markets",
                "Sci/Tech": "Science, technology, and innovation"
            }
            instruction += descriptions.get(category, "")
            instruction += "\n"
        
        # Format input
        formatted_input = f"Article:\n{text}"
        
        # Format output
        output = ID_TO_LABEL[label]
        
        # Add explanation if provided
        if explanation:
            output += f"\n\nExplanation: {explanation}"
        elif self.config.use_cot:
            # Generate simple explanation
            output += f"\n\nExplanation: This article belongs to {ID_TO_LABEL[label]} "
            output += "because it discusses topics related to "
            
            keywords = {
                0: "international affairs, countries, governments, or world events",
                1: "sports, games, teams, or athletic competitions",
                2: "business, companies, markets, or economic matters",
                3: "science, technology, research, or technical innovations"
            }
            output += keywords.get(label, "this category")
        
        return {
            "instruction": instruction,
            "input": formatted_input,
            "output": output
        }
    
    def format_for_chat(
        self,
        text: str,
        label: Optional[int] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Format as chat conversation.
        
        Following chat formats from:
        - OpenAI ChatML format
        - Anthropic Claude format
        """
        messages = []
        
        # System message
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant that classifies news articles."
            })
        
        # User message
        user_content = f"{self.config.task_description}\n\n"
        user_content += f"Article: {text}\n\n"
        user_content += f"Categories: {self._format_options()}"
        
        messages.append({
            "role": "user",
            "content": user_content
        })
        
        # Assistant response
        if label is not None:
            messages.append({
                "role": "assistant",
                "content": ID_TO_LABEL[label]
            })
        
        return messages
    
    def _format_options(self) -> str:
        """Format category options."""
        options = list(AG_NEWS_CLASSES)
        
        if self.config.shuffle_options:
            self.rng.shuffle(options)
        
        if self.config.use_letters:
            # Format as A) World, B) Sports, etc.
            formatted = []
            for i, option in enumerate(options):
                letter = chr(65 + i)  # A, B, C, D
                formatted.append(f"{letter}) {option}")
            return ", ".join(formatted)
        else:
            return ", ".join(options)
    
    def _format_demonstrations(self, exclude_label: Optional[int] = None) -> str:
        """Format few-shot demonstrations."""
        if not self.demonstrations:
            return ""
        
        # Select demonstrations
        demos = [d for d in self.demonstrations if d.get('label') != exclude_label]
        
        if len(demos) > self.config.num_demonstrations:
            demos = self.rng.sample(demos, self.config.num_demonstrations)
        
        # Format each demonstration
        formatted_demos = []
        for demo in demos:
            demo_text = f"Article: {demo['text']}\n"
            demo_text += f"Category: {ID_TO_LABEL[demo['label']]}"
            formatted_demos.append(demo_text)
        
        return "\n\n".join(formatted_demos)
    
    def _add_chain_of_thought(self, prompt: str) -> str:
        """
        Add chain of thought reasoning.
        
        Following CoT prompting from:
        - Wei et al. (2022): "Chain of Thought Prompting"
        - Kojima et al. (2022): "Large Language Models are Zero-Shot Reasoners"
        """
        cot_prompt = prompt + f"\n\n{self.config.cot_trigger}\n"
        return cot_prompt
    
    def _add_metadata(self, prompt: str, metadata: Dict[str, Any]) -> str:
        """Add metadata to prompt."""
        meta_str = "\n[Metadata: "
        meta_items = []
        
        for key, value in metadata.items():
            meta_items.append(f"{key}={value}")
        
        meta_str += ", ".join(meta_items) + "]\n"
        return meta_str + prompt
    
    def _format_answer(self, label: int) -> str:
        """Format answer based on configuration."""
        answer = ID_TO_LABEL[label]
        
        if self.config.use_letters:
            # Map to letter
            options = list(AG_NEWS_CLASSES)
            if answer in options:
                idx = options.index(answer)
                letter = chr(65 + idx)
                return f"{letter}) {answer}"
        
        return answer
    
    def create_prompt_dataset(
        self,
        texts: List[str],
        labels: List[int],
        style: Optional[str] = None
    ) -> List[str]:
        """
        Create prompt dataset from texts and labels.
        
        Args:
            texts: List of texts
            labels: List of labels
            style: Optional style override
            
        Returns:
            List of formatted prompts
        """
        if style:
            original_style = self.config.template_style
            self.config.template_style = style
        
        prompts = []
        for text, label in zip(texts, labels):
            prompt = self.format_single(text, label)
            prompts.append(prompt)
        
        if style:
            self.config.template_style = original_style
        
        return prompts
