"""
Template Manager for Prompt-Based Learning
===========================================

Manages prompt templates for various prompt-based learning strategies,
based on:
- Brown et al. (2020): "Language Models are Few-Shot Learners"
- Schick & Schütze (2021): "Exploiting Cloze Questions for Few Shot Text Classification"
- Liu et al. (2023): "Pre-train, Prompt, and Predict: A Systematic Survey"

Provides a unified interface for managing, selecting, and optimizing
prompt templates for text classification tasks.

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import re
from collections import defaultdict

import torch
import numpy as np

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class TemplateType(Enum):
    """Types of prompt templates"""
    MANUAL = "manual"  # Hand-crafted templates
    GENERATED = "generated"  # Auto-generated templates
    LEARNED = "learned"  # Learned through optimization
    HYBRID = "hybrid"  # Combination of manual and learned


class VerbalizerType(Enum):
    """Types of label verbalizers"""
    ONE_TO_ONE = "one_to_one"  # One word per label
    ONE_TO_MANY = "one_to_many"  # Multiple words per label
    SOFT = "soft"  # Soft verbalizer with continuous weights
    CONTEXTUAL = "contextual"  # Context-dependent verbalizer


@dataclass
class PromptTemplate:
    """
    Represents a single prompt template.
    
    Attributes:
        template: Template string with placeholders
        template_type: Type of template
        task_description: Optional task description
        few_shot_separator: Separator for few-shot examples
        label_position: Position of label in template
        metadata: Additional metadata
    """
    template: str
    template_type: TemplateType = TemplateType.MANUAL
    task_description: Optional[str] = None
    few_shot_separator: str = "\n\n"
    label_position: str = "end"  # "start", "end", "middle"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def format(
        self,
        text: str,
        label: Optional[str] = None,
        examples: Optional[List[Dict[str, str]]] = None,
        **kwargs
    ) -> str:
        """
        Format template with given inputs.
        
        Args:
            text: Input text
            label: Label text (optional)
            examples: Few-shot examples
            **kwargs: Additional placeholders
            
        Returns:
            Formatted prompt string
        """
        # Build the prompt
        prompt_parts = []
        
        # Add task description if available
        if self.task_description:
            prompt_parts.append(self.task_description)
        
        # Add few-shot examples if provided
        if examples:
            example_strings = []
            for ex in examples:
                ex_text = self.template.format(
                    text=ex['text'],
                    label=ex.get('label', '[MASK]'),
                    **kwargs
                )
                example_strings.append(ex_text)
            prompt_parts.extend(example_strings)
            prompt_parts.append(self.few_shot_separator)
        
        # Format main template
        main_prompt = self.template.format(
            text=text,
            label=label if label else '[MASK]',
            **kwargs
        )
        prompt_parts.append(main_prompt)
        
        return '\n'.join(prompt_parts)
    
    def extract_label_position(self, formatted_prompt: str) -> Tuple[int, int]:
        """
        Extract the position of label placeholder in formatted prompt.
        
        Args:
            formatted_prompt: Formatted prompt string
            
        Returns:
            Start and end positions of label
        """
        # Find [MASK] or label position
        match = re.search(r'```math
MASK```|<label>.*?</label>', formatted_prompt)
        if match:
            return match.span()
        return -1, -1


@dataclass
class Verbalizer:
    """
    Maps labels to natural language tokens.
    
    Attributes:
        label_words: Mapping from labels to words/tokens
        verbalizer_type: Type of verbalizer
        calibration_params: Parameters for calibration
    """
    label_words: Dict[Union[int, str], List[str]]
    verbalizer_type: VerbalizerType = VerbalizerType.ONE_TO_ONE
    calibration_params: Optional[Dict[str, float]] = None
    
    def __post_init__(self):
        """Validate and process verbalizer"""
        # Ensure all labels have at least one word
        for label, words in self.label_words.items():
            if not words:
                raise ValueError(f"Label {label} has no associated words")
            if not isinstance(words, list):
                self.label_words[label] = [words]
    
    def get_label_words(self, label: Union[int, str]) -> List[str]:
        """Get words associated with a label"""
        return self.label_words.get(label, [])
    
    def get_all_words(self) -> List[str]:
        """Get all verbalizer words"""
        all_words = []
        for words in self.label_words.values():
            all_words.extend(words)
        return list(set(all_words))


class TemplateManager:
    """
    Manages prompt templates for AG News classification.
    
    Provides functionality for:
    1. Template storage and retrieval
    2. Template selection and optimization
    3. Dynamic template generation
    4. Template evaluation and scoring
    """
    
    def __init__(self, task: str = "ag_news"):
        """
        Initialize template manager.
        
        Args:
            task: Task name
        """
        self.task = task
        self.templates: Dict[str, PromptTemplate] = {}
        self.verbalizers: Dict[str, Verbalizer] = {}
        self.template_scores: Dict[str, float] = {}
        
        # Initialize default templates
        self._init_default_templates()
        self._init_default_verbalizers()
        
        # Template optimization history
        self.optimization_history = []
        
        logger.info(f"Initialized TemplateManager for {task}")
    
    def _init_default_templates(self):
        """Initialize default templates for AG News"""
        # Classification templates
        self.add_template(
            "simple",
            PromptTemplate(
                template="Article: {text}\nCategory: {label}",
                template_type=TemplateType.MANUAL
            )
        )
        
        self.add_template(
            "question",
            PromptTemplate(
                template='What is the category of this news article?\n\n"{text}"\n\nCategory: {label}',
                template_type=TemplateType.MANUAL
            )
        )
        
        self.add_template(
            "context",
            PromptTemplate(
                template="Classify the following news article into World, Sports, Business, or Technology.\n\nArticle: {text}\n\nThis article belongs to {label} news.",
                template_type=TemplateType.MANUAL
            )
        )
        
        self.add_template(
            "cloze",
            PromptTemplate(
                template='News: "{text}" This is about {label}.',
                template_type=TemplateType.MANUAL,
                label_position="end"
            )
        )
        
        self.add_template(
            "instruction",
            PromptTemplate(
                template="Task: Categorize the news article.\nCategories: World, Sports, Business, Technology\n\nArticle: {text}\n\nAnswer: {label}",
                template_type=TemplateType.MANUAL,
                task_description="You are a news categorization expert."
            )
        )
    
    def _init_default_verbalizers(self):
        """Initialize default verbalizers for AG News"""
        # Standard verbalizer
        self.add_verbalizer(
            "standard",
            Verbalizer(
                label_words={
                    0: ["World", "International"],
                    1: ["Sports", "Athletics"],
                    2: ["Business", "Economy"],
                    3: ["Technology", "Tech", "Science"]
                },
                verbalizer_type=VerbalizerType.ONE_TO_MANY
            )
        )
        
        # Simple verbalizer
        self.add_verbalizer(
            "simple",
            Verbalizer(
                label_words={
                    0: ["world"],
                    1: ["sports"],
                    2: ["business"],
                    3: ["technology"]
                },
                verbalizer_type=VerbalizerType.ONE_TO_ONE
            )
        )
        
        # Extended verbalizer
        self.add_verbalizer(
            "extended",
            Verbalizer(
                label_words={
                    0: ["world", "international", "global", "politics"],
                    1: ["sports", "game", "match", "athlete"],
                    2: ["business", "company", "market", "finance"],
                    3: ["technology", "computer", "software", "innovation"]
                },
                verbalizer_type=VerbalizerType.ONE_TO_MANY
            )
        )
    
    def add_template(self, name: str, template: PromptTemplate):
        """Add a new template"""
        self.templates[name] = template
        logger.debug(f"Added template: {name}")
    
    def add_verbalizer(self, name: str, verbalizer: Verbalizer):
        """Add a new verbalizer"""
        self.verbalizers[name] = verbalizer
        logger.debug(f"Added verbalizer: {name}")
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name"""
        return self.templates.get(name)
    
    def get_verbalizer(self, name: str) -> Optional[Verbalizer]:
        """Get verbalizer by name"""
        return self.verbalizers.get(name)
    
    def select_best_template(
        self,
        validation_scores: Optional[Dict[str, float]] = None,
        selection_strategy: str = "max_score"
    ) -> str:
        """
        Select best template based on validation scores.
        
        Args:
            validation_scores: Scores for each template
            selection_strategy: Selection strategy
            
        Returns:
            Name of best template
        """
        if validation_scores:
            self.template_scores.update(validation_scores)
        
        if not self.template_scores:
            # Return first template if no scores
            return list(self.templates.keys())[0]
        
        if selection_strategy == "max_score":
            # Select template with highest score
            best_template = max(self.template_scores, key=self.template_scores.get)
        elif selection_strategy == "weighted_random":
            # Weighted random selection
            templates = list(self.template_scores.keys())
            scores = np.array(list(self.template_scores.values()))
            probabilities = scores / scores.sum()
            best_template = np.random.choice(templates, p=probabilities)
        else:
            best_template = list(self.templates.keys())[0]
        
        logger.info(f"Selected template: {best_template}")
        return best_template
    
    def generate_template(
        self,
        base_template: str,
        mutation_type: str = "paraphrase"
    ) -> PromptTemplate:
        """
        Generate new template from existing one.
        
        Args:
            base_template: Name of base template
            mutation_type: Type of mutation
            
        Returns:
            New template
        """
        base = self.templates.get(base_template)
        if not base:
            raise ValueError(f"Base template {base_template} not found")
        
        if mutation_type == "paraphrase":
            # Simple paraphrasing (would use LLM in practice)
            new_template = base.template.replace("Article:", "News:")
            new_template = new_template.replace("Category:", "Type:")
        elif mutation_type == "reorder":
            # Reorder template components
            parts = base.template.split('\n')
            np.random.shuffle(parts)
            new_template = '\n'.join(parts)
        else:
            new_template = base.template
        
        return PromptTemplate(
            template=new_template,
            template_type=TemplateType.GENERATED,
            metadata={"base": base_template, "mutation": mutation_type}
        )
    
    def optimize_template(
        self,
        template_name: str,
        performance_scores: List[float],
        optimization_steps: int = 10
    ) -> PromptTemplate:
        """
        Optimize template based on performance feedback.
        
        Args:
            template_name: Template to optimize
            performance_scores: Performance history
            optimization_steps: Number of optimization steps
            
        Returns:
            Optimized template
        """
        template = self.templates.get(template_name)
        if not template:
            raise ValueError(f"Template {template_name} not found")
        
        # Simple optimization: adjust template based on scores
        avg_score = np.mean(performance_scores)
        
        # Record optimization history
        self.optimization_history.append({
            'template': template_name,
            'score': avg_score,
            'steps': optimization_steps
        })
        
        # For now, return original template
        # In practice, would use gradient-based optimization or RL
        logger.info(f"Optimized template {template_name}, score: {avg_score:.4f}")
        return template
    
    def save(self, path: Union[str, Path]):
        """Save template manager state"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = {
            'templates': {
                name: {
                    'template': t.template,
                    'type': t.template_type.value,
                    'task_description': t.task_description,
                    'metadata': t.metadata
                }
                for name, t in self.templates.items()
            },
            'verbalizers': {
                name: {
                    'label_words': v.label_words,
                    'type': v.verbalizer_type.value
                }
                for name, v in self.verbalizers.items()
            },
            'scores': self.template_scores,
            'optimization_history': self.optimization_history
        }
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Saved TemplateManager to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TemplateManager':
        """Load template manager from file"""
        path = Path(path)
        
        with open(path, 'r') as f:
            state = json.load(f)
        
        manager = cls()
        
        # Load templates
        for name, t_dict in state['templates'].items():
            template = PromptTemplate(
                template=t_dict['template'],
                template_type=TemplateType(t_dict['type']),
                task_description=t_dict.get('task_description'),
                metadata=t_dict.get('metadata', {})
            )
            manager.add_template(name, template)
        
        # Load verbalizers
        for name, v_dict in state['verbalizers'].items():
            verbalizer = Verbalizer(
                label_words=v_dict['label_words'],
                verbalizer_type=VerbalizerType(v_dict['type'])
            )
            manager.add_verbalizer(name, verbalizer)
        
        manager.template_scores = state.get('scores', {})
        manager.optimization_history = state.get('optimization_history', [])
        
        logger.info(f"Loaded TemplateManager from {path}")
        return manager
