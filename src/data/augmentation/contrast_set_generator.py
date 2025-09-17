"""
Contrast Set Generator Module
=============================

Implements contrast set generation for robustness evaluation following:
- Gardner et al. (2020): "Evaluating Models' Local Decision Boundaries via Contrast Sets"
- Kaushik et al. (2020): "Learning the Difference that Makes a Difference with Counterfactually-Augmented Data"
- Ross et al. (2021): "Tailor: Generating and Perturbing Text with Semantic Controls"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
import random
from pathlib import Path
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation.base_augmenter import BaseAugmenter, AugmentationConfig
from src.utils.logging_config import setup_logging
from configs.constants import AG_NEWS_CLASSES, LABEL_TO_ID

logger = setup_logging(__name__)

@dataclass
class ContrastSetConfig(AugmentationConfig):
    """
    Configuration for contrast set generation.
    
    Following configurations from:
    - Gardner et al. (2020): "Evaluating Models' Local Decision Boundaries"
    - Kaushik et al. (2020): "Learning the Difference that Makes a Difference"
    """
    
    # Generation strategies
    generation_strategy: str = "rule_based"  # rule_based, model_based, human_guided
    contrast_type: str = "minimal"  # minimal, diverse, adversarial
    
    # Perturbation settings
    max_perturbations: int = 3
    perturbation_types: List[str] = None  # entity, number, negation, temporal
    preserve_fluency: bool = True
    
    # Label flipping
    target_label_strategy: str = "nearest"  # nearest, random, all
    ensure_label_change: bool = True
    
    # Domain-specific for AG News
    news_categories: List[str] = None  # World, Sports, Business, Sci/Tech
    category_specific_rules: bool = True
    
    # Quality control
    min_edit_distance: int = 1
    max_edit_distance: int = 10
    semantic_threshold: float = 0.7
    
    def __post_init__(self):
        """Initialize default values."""
        if self.perturbation_types is None:
            self.perturbation_types = ["entity", "number", "negation", "temporal", "location"]
        
        if self.news_categories is None:
            self.news_categories = AG_NEWS_CLASSES

class ContrastSetGenerator(BaseAugmenter):
    """
    Generate contrast sets for robust evaluation.
    
    Implements techniques from:
    - Gardner et al. (2020): "Evaluating Models' Local Decision Boundaries"
    - Wu et al. (2021): "Polyjuice: Generating Counterfactuals for Explaining"
    """
    
    def __init__(
        self,
        config: Optional[ContrastSetConfig] = None
    ):
        """
        Initialize contrast set generator.
        
        Args:
            config: Contrast set configuration
        """
        super().__init__(config or ContrastSetConfig(), name="contrast_set")
        
        # Initialize perturbation rules
        self._initialize_perturbation_rules()
        
        # Initialize NLP tools
        self._initialize_nlp_tools()
        
        logger.info(f"Initialized contrast set generator with strategy: {self.config.generation_strategy}")
    
    def _initialize_perturbation_rules(self):
        """Initialize domain-specific perturbation rules."""
        self.perturbation_rules = {
            "World": {
                "entities": ["country", "leader", "organization"],
                "templates": ["conflict", "diplomacy", "disaster"]
            },
            "Sports": {
                "entities": ["team", "player", "tournament"],
                "templates": ["win", "loss", "injury", "transfer"]
            },
            "Business": {
                "entities": ["company", "CEO", "market"],
                "templates": ["profit", "loss", "merger", "IPO"]
            },
            "Sci/Tech": {
                "entities": ["company", "product", "technology"],
                "templates": ["launch", "update", "security", "innovation"]
            }
        }
        
        # Entity replacements for each category
        self.entity_replacements = {
            "World": {
                "USA": ["China", "Russia", "UK", "Germany"],
                "president": ["prime minister", "chancellor", "leader"],
                "UN": ["NATO", "EU", "WHO", "IMF"]
            },
            "Sports": {
                "football": ["basketball", "baseball", "soccer", "tennis"],
                "championship": ["tournament", "league", "cup", "series"],
                "won": ["lost", "drew", "defeated", "beat"]
            },
            "Business": {
                "profit": ["loss", "revenue", "earnings", "income"],
                "increased": ["decreased", "dropped", "surged", "fell"],
                "merger": ["acquisition", "partnership", "split", "IPO"]
            },
            "Sci/Tech": {
                "launched": ["announced", "released", "unveiled", "introduced"],
                "software": ["hardware", "platform", "service", "application"],
                "security": ["privacy", "performance", "feature", "compatibility"]
            }
        }
    
    def _initialize_nlp_tools(self):
        """Initialize NLP tools for contrast generation."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy for linguistic analysis")
        except:
            logger.warning("spaCy not available, using rule-based approach")
            self.nlp = None
    
    def augment_single(
        self,
        text: str,
        label: Optional[int] = None,
        target_label: Optional[int] = None,
        **kwargs
    ) -> List[Tuple[str, int]]:
        """
        Generate contrast set for single text.
        
        Args:
            text: Input text
            label: Original label
            target_label: Target label for contrast
            **kwargs: Additional arguments
            
        Returns:
            List of (contrast_text, new_label) tuples
        """
        # Check cache
        cached = self.get_from_cache(text, contrast_type=self.config.contrast_type)
        if cached:
            return cached
        
        contrast_set = []
        
        # Determine target labels
        if target_label is not None:
            target_labels = [target_label]
        elif self.config.target_label_strategy == "all":
            target_labels = [i for i in range(len(AG_NEWS_CLASSES)) if i != label]
        elif self.config.target_label_strategy == "nearest":
            target_labels = [self._get_nearest_label(label)]
        else:
            target_labels = [self.rng.choice([i for i in range(len(AG_NEWS_CLASSES)) if i != label])]
        
        # Generate contrasts for each target label
        for target in target_labels:
            if self.config.generation_strategy == "rule_based":
                contrasts = self._rule_based_generation(text, label, target)
            elif self.config.generation_strategy == "model_based":
                contrasts = self._model_based_generation(text, label, target)
            else:
                contrasts = []
            
            for contrast_text in contrasts:
                if self._validate_contrast(text, contrast_text, label, target):
                    contrast_set.append((contrast_text, target))
        
        # Cache results
        self.add_to_cache(text, contrast_set, contrast_type=self.config.contrast_type)
        
        return contrast_set
    
    def _rule_based_generation(
        self,
        text: str,
        source_label: int,
        target_label: int
    ) -> List[str]:
        """
        Generate contrasts using rule-based perturbations.
        
        Following rule-based generation from:
        - Gardner et al. (2020): "Evaluating Models' Local Decision Boundaries"
        """
        contrasts = []
        
        # Get category names
        source_category = AG_NEWS_CLASSES[source_label]
        target_category = AG_NEWS_CLASSES[target_label]
        
        # Apply different perturbation types
        for perturb_type in self.config.perturbation_types:
            if perturb_type == "entity":
                contrast = self._entity_replacement(text, source_category, target_category)
            elif perturb_type == "number":
                contrast = self._number_perturbation(text)
            elif perturb_type == "negation":
                contrast = self._negation_insertion(text)
            elif perturb_type == "temporal":
                contrast = self._temporal_shift(text)
            elif perturb_type == "location":
                contrast = self._location_change(text, source_category, target_category)
            else:
                continue
            
            if contrast and contrast != text:
                contrasts.append(contrast)
        
        return contrasts[:self.config.max_perturbations]
    
    def _model_based_generation(
        self,
        text: str,
        source_label: int,
        target_label: int
    ) -> List[str]:
        """
        Generate contrasts using neural models.
        
        Following model-based generation from:
        - Wu et al. (2021): "Polyjuice: Generating Counterfactuals"
        """
        # This would use a trained model for generation
        # For now, fallback to rule-based
        return self._rule_based_generation(text, source_label, target_label)
    
    def _entity_replacement(
        self,
        text: str,
        source_category: str,
        target_category: str
    ) -> str:
        """Replace entities to change category."""
        if source_category not in self.entity_replacements:
            return text
        
        modified = text
        replacements = self.entity_replacements[source_category]
        
        # Try to replace entities
        for original, candidates in replacements.items():
            if original.lower() in modified.lower():
                replacement = self.rng.choice(candidates)
                # Case-sensitive replacement
                if original[0].isupper():
                    replacement = replacement.capitalize()
                modified = modified.replace(original, replacement)
                break  # One replacement for minimal change
        
        # Add category-specific keywords if needed
        if target_category in self.perturbation_rules:
            keywords = self.perturbation_rules[target_category].get("templates", [])
            if keywords and self.rng.random() < 0.3:
                keyword = self.rng.choice(keywords)
                # Insert keyword naturally
                words = modified.split()
                insert_pos = self.rng.randint(1, max(2, len(words) // 2))
                words.insert(insert_pos, keyword)
                modified = " ".join(words)
        
        return modified
    
    def _number_perturbation(self, text: str) -> str:
        """Perturb numerical values."""
        import re
        
        # Find numbers in text
        numbers = re.findall(r'\d+', text)
        if not numbers:
            return text
        
        modified = text
        # Change one number
        num_to_change = self.rng.choice(numbers)
        
        # Perturb the number
        value = int(num_to_change)
        if value > 100:
            new_value = value + self.rng.randint(-50, 50)
        else:
            new_value = value + self.rng.randint(-5, 5)
        
        new_value = max(0, new_value)  # Ensure non-negative
        modified = modified.replace(num_to_change, str(new_value), 1)
        
        return modified
    
    def _negation_insertion(self, text: str) -> str:
        """Insert or remove negation."""
        negation_words = ["not", "no", "never", "neither", "nor", "n't"]
        
        words = text.split()
        
        # Check if negation exists
        has_negation = any(neg in text.lower() for neg in negation_words)
        
        if has_negation:
            # Remove negation
            for neg in negation_words:
                if neg in words:
                    words.remove(neg)
                    break
        else:
            # Add negation
            # Find a verb to negate
            if self.nlp:
                doc = self.nlp(text)
                for token in doc:
                    if token.pos_ == "VERB":
                        # Insert "not" before the verb
                        verb_idx = words.index(token.text)
                        words.insert(verb_idx, "not")
                        break
            else:
                # Simple heuristic
                insert_pos = len(words) // 2
                words.insert(insert_pos, "not")
        
        return " ".join(words)
    
    def _temporal_shift(self, text: str) -> str:
        """Shift temporal references."""
        temporal_replacements = {
            "today": "yesterday",
            "yesterday": "tomorrow",
            "tomorrow": "today",
            "this week": "last week",
            "last week": "next week",
            "this year": "last year",
            "last year": "next year",
            "will": "did",
            "has": "will have",
            "was": "will be"
        }
        
        modified = text
        for original, replacement in temporal_replacements.items():
            if original in modified.lower():
                # Preserve case
                if original[0].isupper():
                    replacement = replacement.capitalize()
                modified = modified.replace(original, replacement)
                break  # One replacement
        
        return modified
    
    def _location_change(
        self,
        text: str,
        source_category: str,
        target_category: str
    ) -> str:
        """Change location references."""
        location_replacements = {
            "New York": ["London", "Tokyo", "Paris", "Beijing"],
            "United States": ["United Kingdom", "Japan", "France", "China"],
            "Washington": ["London", "Tokyo", "Paris", "Beijing"],
            "Silicon Valley": ["Shenzhen", "Bangalore", "Tel Aviv", "Berlin"]
        }
        
        modified = text
        for location, alternatives in location_replacements.items():
            if location in modified:
                replacement = self.rng.choice(alternatives)
                modified = modified.replace(location, replacement)
                break
        
        return modified
    
    def _get_nearest_label(self, label: int) -> int:
        """Get nearest label for minimal contrast."""
        # Simple heuristic: adjacent labels are nearest
        num_classes = len(AG_NEWS_CLASSES)
        
        if label == 0:
            return 1
        elif label == num_classes - 1:
            return label - 1
        else:
            return label + self.rng.choice([-1, 1])
    
    def _validate_contrast(
        self,
        original: str,
        contrast: str,
        source_label: int,
        target_label: int
    ) -> bool:
        """
        Validate generated contrast.
        
        Following validation from:
        - Gardner et al. (2020): "Evaluating Models"
        """
        # Check edit distance
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, original, contrast).ratio()
        
        if similarity < 0.5 or similarity > 0.95:
            return False
        
        # Check minimum changes
        if original == contrast:
            return False
        
        # Check fluency (simple heuristic)
        if len(contrast.split()) < self.config.min_length:
            return False
        
        return True
    
    def generate_contrast_dataset(
        self,
        dataset: List[Tuple[str, int]],
        num_contrasts_per_sample: int = 2
    ) -> List[Tuple[str, int, str, int]]:
        """
        Generate contrast set for entire dataset.
        
        Returns:
            List of (original_text, original_label, contrast_text, contrast_label)
        """
        contrast_dataset = []
        
        for text, label in dataset:
            contrasts = self.augment_single(text, label)
            
            for contrast_text, contrast_label in contrasts[:num_contrasts_per_sample]:
                contrast_dataset.append((text, label, contrast_text, contrast_label))
        
        logger.info(f"Generated {len(contrast_dataset)} contrast pairs from {len(dataset)} samples")
        
        return contrast_dataset
