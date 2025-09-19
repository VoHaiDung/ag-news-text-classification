"""
Prompt engineering utilities for AG News Text Classification Framework.

Provides tools for creating, managing, and optimizing prompts for
prompt-based learning and instruction tuning approaches.

References:
    - Liu, P., et al. (2023). "Pre-train, Prompt, and Predict: A Systematic 
      Survey of Prompting Methods in Natural Language Processing". ACM Computing Surveys.
    - Wei, J., et al. (2022). "Chain-of-thought prompting elicits reasoning 
      in large language models". NeurIPS.

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
"""

import re
import json
import logging
from typing import Optional, Dict, Any, List, Union, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import warnings

import torch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Enumeration of prompt types."""
    
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    INSTRUCTION = "instruction"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    CLOZE = "cloze"
    PREFIX = "prefix"
    MIXED = "mixed"


@dataclass
class PromptTemplate:
    """
    Container for prompt templates.
    
    Following prompt engineering best practices from Liu et al. (2023).
    """
    
    name: str
    template: str
    type: PromptType
    variables: List[str] = field(default_factory=list)
    examples: List[Dict[str, str]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Extract variables from template."""
        # Find all variables in {variable} format
        pattern = r"\{([^}]+)\}"
        found_vars = re.findall(pattern, self.template)
        
        # Update variables list
        for var in found_vars:
            if var not in self.variables:
                self.variables.append(var)
    
    def format(self, **kwargs) -> str:
        """
        Format template with provided values.
        
        Args:
            **kwargs: Variable values
            
        Returns:
            Formatted prompt
        """
        # Check for missing variables
        missing = set(self.variables) - set(kwargs.keys())
        if missing:
            logger.warning(f"Missing variables for template '{self.name}': {missing}")
        
        # Format template
        try:
            return self.template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Error formatting template '{self.name}': {e}")
            raise
    
    def add_example(self, example: Dict[str, str]):
        """Add example to template."""
        self.examples.append(example)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "template": self.template,
            "type": self.type.value,
            "variables": self.variables,
            "examples": self.examples,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PromptTemplate":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            template=data["template"],
            type=PromptType(data["type"]),
            variables=data.get("variables", []),
            examples=data.get("examples", []),
            metadata=data.get("metadata", {}),
        )


class PromptManager:
    """
    Manager for prompt templates and generation.
    
    Handles prompt creation, optimization, and management following
    best practices from prompt engineering literature.
    """
    
    def __init__(self, templates_dir: Optional[Union[str, Path]] = None):
        """
        Initialize prompt manager.
        
        Args:
            templates_dir: Directory containing prompt templates
        """
        self.templates_dir = Path(templates_dir) if templates_dir else None
        self.templates: Dict[str, PromptTemplate] = {}
        
        # Load default templates
        self._load_default_templates()
        
        # Load custom templates
        if self.templates_dir and self.templates_dir.exists():
            self._load_templates_from_dir()
    
    def _load_default_templates(self):
        """Load default prompt templates."""
        # Zero-shot classification
        self.add_template(
            PromptTemplate(
                name="zero_shot_classification",
                template="Classify the following text into one of these categories: {categories}.\n\nText: {text}\n\nCategory:",
                type=PromptType.ZERO_SHOT,
            )
        )
        
        # Few-shot classification
        self.add_template(
            PromptTemplate(
                name="few_shot_classification",
                template="Classify texts into categories.\n\n{examples}\n\nText: {text}\nCategory:",
                type=PromptType.FEW_SHOT,
            )
        )
        
        # Instruction-based
        self.add_template(
            PromptTemplate(
                name="instruction_classification",
                template="Instruction: {instruction}\n\nInput: {text}\n\nOutput:",
                type=PromptType.INSTRUCTION,
            )
        )
        
        # Chain-of-thought
        self.add_template(
            PromptTemplate(
                name="cot_classification",
                template="Let's think step by step about classifying this text.\n\nText: {text}\n\nReasoning: {reasoning}\n\nTherefore, the category is:",
                type=PromptType.CHAIN_OF_THOUGHT,
            )
        )
        
        # Cloze-style
        self.add_template(
            PromptTemplate(
                name="cloze_classification",
                template="The following text about {text} belongs to the [MASK] category.",
                type=PromptType.CLOZE,
            )
        )
        
        # AG News specific templates
        self.add_template(
            PromptTemplate(
                name="ag_news_zero_shot",
                template="Classify this news article into World, Sports, Business, or Sci/Tech.\n\nArticle: {text}\n\nCategory:",
                type=PromptType.ZERO_SHOT,
            )
        )
        
        self.add_template(
            PromptTemplate(
                name="ag_news_detailed",
                template="You are a news categorization expert. Classify the following news article into exactly one category:\n- World: International news and politics\n- Sports: Sports news and events\n- Business: Business and economic news\n- Sci/Tech: Science and technology news\n\nArticle: {text}\n\nCategory:",
                type=PromptType.INSTRUCTION,
            )
        )
    
    def _load_templates_from_dir(self):
        """Load templates from directory."""
        for filepath in self.templates_dir.glob("*.json"):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                
                template = PromptTemplate.from_dict(data)
                self.add_template(template)
                logger.debug(f"Loaded template: {template.name}")
                
            except Exception as e:
                logger.error(f"Error loading template from {filepath}: {e}")
    
    def add_template(self, template: PromptTemplate):
        """
        Add template to manager.
        
        Args:
            template: Prompt template
        """
        self.templates[template.name] = template
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        Get template by name.
        
        Args:
            name: Template name
            
        Returns:
            Prompt template or None
        """
        return self.templates.get(name)
    
    def create_prompt(
        self,
        template_name: str,
        **kwargs
    ) -> str:
        """
        Create prompt from template.
        
        Args:
            template_name: Name of template
            **kwargs: Template variables
            
        Returns:
            Formatted prompt
        """
        template = self.get_template(template_name)
        
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        return template.format(**kwargs)
    
    def create_few_shot_prompt(
        self,
        template_name: str,
        examples: List[Dict[str, str]],
        query: Dict[str, str],
        num_examples: Optional[int] = None,
    ) -> str:
        """
        Create few-shot prompt with examples.
        
        Args:
            template_name: Template name
            examples: List of example dictionaries
            query: Query dictionary
            num_examples: Number of examples to include
            
        Returns:
            Few-shot prompt
        """
        template = self.get_template(template_name)
        
        if not template:
            raise ValueError(f"Template not found: {template_name}")
        
        # Select examples
        if num_examples:
            examples = examples[:num_examples]
        
        # Format examples
        examples_str = "\n\n".join([
            f"Text: {ex['text']}\nCategory: {ex['label']}"
            for ex in examples
        ])
        
        # Create prompt
        return template.format(examples=examples_str, **query)
    
    def optimize_prompt_length(
        self,
        prompt: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int,
        strategy: str = "truncate",
    ) -> str:
        """
        Optimize prompt length for model constraints.
        
        Args:
            prompt: Original prompt
            tokenizer: Tokenizer
            max_length: Maximum token length
            strategy: Optimization strategy (truncate, summarize)
            
        Returns:
            Optimized prompt
        """
        # Tokenize prompt
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        
        if len(tokens) <= max_length:
            return prompt
        
        if strategy == "truncate":
            # Truncate from the middle to preserve context
            keep_start = max_length // 2
            keep_end = max_length - keep_start
            
            truncated_tokens = tokens[:keep_start] + tokens[-keep_end:]
            optimized = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            
            logger.warning(f"Prompt truncated from {len(tokens)} to {max_length} tokens")
            
        elif strategy == "summarize":
            # Placeholder for summarization strategy
            logger.warning("Summarization strategy not implemented, using truncation")
            return self.optimize_prompt_length(prompt, tokenizer, max_length, "truncate")
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        return optimized
    
    def validate_prompt(
        self,
        prompt: str,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[bool, List[str]]:
        """
        Validate prompt.
        
        Args:
            prompt: Prompt to validate
            tokenizer: Optional tokenizer for length check
            max_length: Maximum length
            
        Returns:
            Tuple of (is_valid, issues)
        """
        issues = []
        
        # Check if prompt is empty
        if not prompt or not prompt.strip():
            issues.append("Prompt is empty")
        
        # Check length
        if tokenizer and max_length:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if len(tokens) > max_length:
                issues.append(f"Prompt exceeds max length ({len(tokens)} > {max_length})")
        
        # Check for common issues
        if prompt.count("{") != prompt.count("}"):
            issues.append("Unbalanced curly braces")
        
        if "[MASK]" in prompt and tokenizer:
            if not hasattr(tokenizer, "mask_token"):
                issues.append("Tokenizer does not support [MASK] token")
        
        is_valid = len(issues) == 0
        
        return is_valid, issues
    
    def save_template(
        self,
        template: PromptTemplate,
        filepath: Union[str, Path],
    ):
        """
        Save template to file.
        
        Args:
            template: Template to save
            filepath: Save path
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(template.to_dict(), f, indent=2)
        
        logger.info(f"Template saved to {filepath}")
    
    def load_template(
        self,
        filepath: Union[str, Path],
    ) -> PromptTemplate:
        """
        Load template from file.
        
        Args:
            filepath: Template file path
            
        Returns:
            Loaded template
        """
        filepath = Path(filepath)
        
        with open(filepath) as f:
            data = json.load(f)
        
        template = PromptTemplate.from_dict(data)
        self.add_template(template)
        
        logger.info(f"Template loaded from {filepath}")
        return template


def create_prompt(
    text: str,
    template: str = "zero_shot_classification",
    categories: Optional[List[str]] = None,
    **kwargs
) -> str:
    """
    Create prompt for text classification.
    
    Args:
        text: Input text
        template: Template name
        categories: List of categories
        **kwargs: Additional template variables
        
    Returns:
        Formatted prompt
    """
    manager = PromptManager()
    
    # Prepare variables
    variables = {"text": text}
    
    if categories:
        variables["categories"] = ", ".join(categories)
    
    variables.update(kwargs)
    
    return manager.create_prompt(template, **variables)


def format_prompt(
    template: str,
    **kwargs
) -> str:
    """
    Format prompt template with variables.
    
    Args:
        template: Template string
        **kwargs: Template variables
        
    Returns:
        Formatted prompt
    """
    prompt_template = PromptTemplate(
        name="custom",
        template=template,
        type=PromptType.MIXED,
    )
    
    return prompt_template.format(**kwargs)


def load_prompt_template(filepath: Union[str, Path]) -> PromptTemplate:
    """
    Load prompt template from file.
    
    Args:
        filepath: Template file path
        
    Returns:
        Loaded template
    """
    manager = PromptManager()
    return manager.load_template(filepath)


def save_prompt_template(
    template: PromptTemplate,
    filepath: Union[str, Path],
):
    """
    Save prompt template to file.
    
    Args:
        template: Template to save
        filepath: Save path
    """
    manager = PromptManager()
    manager.save_template(template, filepath)


def validate_prompt(
    prompt: str,
    tokenizer: Optional[PreTrainedTokenizer] = None,
    max_length: Optional[int] = None,
) -> Tuple[bool, List[str]]:
    """
    Validate prompt.
    
    Args:
        prompt: Prompt to validate
        tokenizer: Optional tokenizer
        max_length: Maximum length
        
    Returns:
        Tuple of (is_valid, issues)
    """
    manager = PromptManager()
    return manager.validate_prompt(prompt, tokenizer, max_length)


def optimize_prompt_length(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    strategy: str = "truncate",
) -> str:
    """
    Optimize prompt length.
    
    Args:
        prompt: Original prompt
        tokenizer: Tokenizer
        max_length: Maximum length
        strategy: Optimization strategy
        
    Returns:
        Optimized prompt
    """
    manager = PromptManager()
    return manager.optimize_prompt_length(prompt, tokenizer, max_length, strategy)


# Predefined prompt templates for AG News
AG_NEWS_PROMPTS = {
    "zero_shot": "Classify this news article into World, Sports, Business, or Sci/Tech.\n\nArticle: {text}\n\nCategory:",
    
    "few_shot": "Examples:\n{examples}\n\nClassify this article:\n{text}\n\nCategory:",
    
    "detailed": """You are a news categorization expert. Carefully read the following news article and classify it into exactly one of these categories:

- World: International news, politics, wars, diplomacy
- Sports: Sports events, athletes, competitions, sports business
- Business: Companies, markets, economy, finance, business deals
- Sci/Tech: Science discoveries, technology, computing, internet, space

Article: {text}

Category:""",
    
    "chain_of_thought": """Let's analyze this news article step by step:

Article: {text}

Step 1: Identify the main topic
Step 2: Look for category-specific keywords
Step 3: Consider the overall context

Based on this analysis, the category is:""",
    
    "instruction": "Task: Categorize the given news article.\nCategories: World, Sports, Business, Sci/Tech\n\nInput: {text}\n\nOutput:",
}


# Export public API
__all__ = [
    "PromptTemplate",
    "PromptManager",
    "PromptType",
    "create_prompt",
    "format_prompt",
    "load_prompt_template",
    "save_prompt_template",
    "validate_prompt",
    "optimize_prompt_length",
    "AG_NEWS_PROMPTS",
]
