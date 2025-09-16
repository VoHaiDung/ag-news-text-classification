"""
Token Replacement Augmentation Module
======================================

Implements token-level augmentation following:
- Wei & Zou (2019): "EDA: Easy Data Augmentation Techniques"
- Kobayashi (2018): "Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations"
- Gao et al. (2021): "SimCSE: Simple Contrastive Learning of Sentence Embeddings"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Dict, Set
from dataclasses import dataclass
import random
from pathlib import Path

import nltk
from nltk.corpus import wordnet
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation.base_augmenter import BaseAugmenter, AugmentationConfig
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

@dataclass
class TokenReplacementConfig(AugmentationConfig):
    """Configuration for token replacement augmentation."""
    
    # Replacement strategies
    synonym_replacement_prob: float = 0.1
    random_insertion_prob: float = 0.1
    random_swap_prob: float = 0.1
    random_deletion_prob: float = 0.1
    
    # Constraints
    max_replacements: int = 5
    preserve_named_entities: bool = True
    preserve_numbers: bool = True
    preserve_stopwords: bool = False
    
    # Word importance
    use_word_importance: bool = True
    importance_threshold: float = 0.3

class TokenReplacementAugmenter(BaseAugmenter):
    """
    Token-level augmentation using various replacement strategies.
    
    Implements EDA techniques from:
    - Wei & Zou (2019): "EDA: Easy Data Augmentation Techniques for Text Classification"
    """
    
    def __init__(
        self,
        config: Optional[TokenReplacementConfig] = None,
        stopwords: Optional[Set[str]] = None
    ):
        """
        Initialize token replacement augmenter.
        
        Args:
            config: Token replacement configuration
            stopwords: Set of stopwords
        """
        super().__init__(config or TokenReplacementConfig(), name="token_replacement")
        
        # Load stopwords
        if stopwords:
            self.stopwords = stopwords
        else:
            from nltk.corpus import stopwords as nltk_stopwords
            try:
                self.stopwords = set(nltk_stopwords.words('english'))
            except:
                nltk.download('stopwords', quiet=True)
                self.stopwords = set(nltk_stopwords.words('english'))
        
        logger.info("Initialized token replacement augmenter")
    
    def augment_single(
        self,
        text: str,
        label: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Apply token replacement augmentation.
        
        Args:
            text: Input text
            label: Optional label
            **kwargs: Additional arguments
            
        Returns:
            List of augmented texts
        """
        # Check cache
        cached = self.get_from_cache(text)
        if cached:
            return cached
        
        words = text.split()
        augmented = []
        
        # Apply different augmentation techniques
        if self.rng.random() < self.config.synonym_replacement_prob:
            aug_text = self._synonym_replacement(words.copy())
            augmented.append(' '.join(aug_text))
        
        if self.rng.random() < self.config.random_insertion_prob:
            aug_text = self._random_insertion(words.copy())
            augmented.append(' '.join(aug_text))
        
        if self.rng.random() < self.config.random_swap_prob:
            aug_text = self._random_swap(words.copy())
            augmented.append(' '.join(aug_text))
        
        if self.rng.random() < self.config.random_deletion_prob:
            aug_text = self._random_deletion(words.copy())
            if aug_text:  # Ensure not empty
                augmented.append(' '.join(aug_text))
        
        # Remove duplicates
        augmented = list(set(augmented))
        
        # Filter augmentations
        augmented = self.filter_augmentations(text, augmented, label)
        
        # Cache results
        self.add_to_cache(text, augmented)
        
        return augmented if augmented else [text]
    
    def _synonym_replacement(self, words: List[str]) -> List[str]:
        """
        Replace random words with synonyms.
        
        Following synonym replacement from:
        - Zhang et al. (2015): "Character-level Convolutional Networks"
        """
        n = min(self.config.max_replacements, max(1, int(len(words) * 0.1)))
        
        # Get candidate words for replacement
        candidates = [
            (i, word) for i, word in enumerate(words)
            if word.lower() not in self.stopwords and word.isalpha()
        ]
        
        if not candidates:
            return words
        
        # Randomly select words to replace
        self.rng.shuffle(candidates)
        
        for i, word in candidates[:n]:
            synonyms = self._get_synonyms(word)
            
            if synonyms:
                synonym = self.rng.choice(synonyms)
                words[i] = synonym
        
        return words
    
    def _random_insertion(self, words: List[str]) -> List[str]:
        """
        Randomly insert synonyms of random words.
        
        Following random insertion from:
        - Wei & Zou (2019): "EDA"
        """
        n = min(self.config.max_replacements, max(1, int(len(words) * 0.1)))
        
        for _ in range(n):
            if not words:
                break
            
            # Find a random word with synonyms
            candidates = [w for w in words if w.isalpha() and w.lower() not in self.stopwords]
            
            if candidates:
                word = self.rng.choice(candidates)
                synonyms = self._get_synonyms(word)
                
                if synonyms:
                    synonym = self.rng.choice(synonyms)
                    # Insert at random position
                    insert_idx = self.rng.randint(0, len(words))
                    words.insert(insert_idx, synonym)
        
        return words
    
    def _random_swap(self, words: List[str]) -> List[str]:
        """
        Randomly swap two words.
        
        Following random swap from:
        - Wei & Zou (2019): "EDA"
        """
        n = min(self.config.max_replacements, max(1, int(len(words) * 0.1)))
        
        for _ in range(n):
            if len(words) < 2:
                break
            
            idx1 = self.rng.randint(0, len(words) - 1)
            idx2 = self.rng.randint(0, len(words) - 1)
            
            if idx1 != idx2:
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return words
    
    def _random_deletion(self, words: List[str]) -> List[str]:
        """
        Randomly delete words.
        
        Following random deletion from:
        - Wei & Zou (2019): "EDA"
        """
        if len(words) == 1:
            return words
        
        # Randomly delete words with probability p
        p = self.config.random_deletion_prob
        
        remaining = []
        for word in words:
            if self.rng.random() > p:
                remaining.append(word)
        
        # If all words deleted, return random word
        if not remaining:
            return [self.rng.choice(words)]
        
        return remaining
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word using WordNet.
        
        Following synonym extraction from:
        - Miller (1995): "WordNet: A Lexical Database for English"
        """
        synonyms = set()
        
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name()
                
                # Clean synonym
                if '_' in synonym:
                    synonym = synonym.replace('_', ' ')
                
                # Avoid same word
                if synonym.lower() != word.lower():
                    synonyms.add(synonym)
        
        return list(synonyms)
    
    def contextual_replacement(
        self,
        text: str,
        model: Any = None
    ) -> str:
        """
        Contextual word replacement using language models.
        
        Following contextual augmentation from:
        - Kobayashi (2018): "Contextual Augmentation"
        """
        # This would use a masked language model for contextual replacements
        # Implementation depends on specific model used
        pass
