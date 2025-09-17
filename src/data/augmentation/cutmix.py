"""
CutMix Augmentation Module
===========================

Implements CutMix and related cutting strategies for text following:
- Yun et al. (2019): "CutMix: Regularization Strategy to Train Strong Classifiers"
- DeVries & Taylor (2017): "Improved Regularization of Convolutional Neural Networks with Cutout"
- Singh et al. (2022): "CutMixNLP: Text Data Augmentation for Natural Language Processing"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import random
from pathlib import Path
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation.base_augmenter import BaseAugmenter, AugmentationConfig
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class CutMixConfig(AugmentationConfig):
    """
    Configuration for CutMix augmentation.
    
    Following configurations from:
    - Yun et al. (2019): "CutMix: Regularization Strategy"
    - Singh et al. (2022): "CutMixNLP"
    """
    
    # CutMix parameters
    alpha: float = 1.0  # Beta distribution parameter
    cutmix_prob: float = 1.0  # Probability of applying CutMix
    
    # Cut strategies
    cut_strategy: str = "continuous"  # continuous, random, sentence, phrase
    min_cut_ratio: float = 0.1
    max_cut_ratio: float = 0.5
    
    # Text-specific settings
    preserve_sentence_boundary: bool = True
    preserve_phrase_structure: bool = True
    use_syntax_tree: bool = False
    
    # Position settings
    random_position: bool = True
    center_cut: bool = False
    
    # Quality control
    min_remaining_length: int = 10
    ensure_coherence: bool = True

class CutMixAugmenter(BaseAugmenter):
    """
    CutMix augmenter for text data.
    
    Implements cutting and mixing strategies from:
    - Yun et al. (2019): "CutMix"
    - Singh et al. (2022): "CutMixNLP: Text Data Augmentation"
    """
    
    def __init__(
        self,
        config: Optional[CutMixConfig] = None
    ):
        """
        Initialize CutMix augmenter.
        
        Args:
            config: CutMix configuration
        """
        super().__init__(config or CutMixConfig(), name="cutmix")
        
        # Initialize syntax parser if needed
        if self.config.use_syntax_tree:
            self._initialize_parser()
        
        logger.info(f"Initialized CutMix augmenter with strategy: {self.config.cut_strategy}")
    
    def _initialize_parser(self):
        """Initialize syntax parser for structure-aware cutting."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model for syntax-aware cutting")
        except:
            logger.warning("spaCy not available, falling back to simple cutting")
            self.config.use_syntax_tree = False
    
    def augment_single(
        self,
        text: str,
        label: Optional[int] = None,
        cut_from: Optional[str] = None,
        cut_label: Optional[int] = None,
        **kwargs
    ) -> Union[str, Tuple[str, np.ndarray]]:
        """
        Apply CutMix augmentation to single text.
        
        Args:
            text: Input text
            label: Optional label
            cut_from: Text to cut from
            cut_label: Label of cut_from text
            **kwargs: Additional arguments
            
        Returns:
            CutMixed text and optionally mixed label
        """
        # Check if should apply CutMix
        if self.rng.random() > self.config.cutmix_prob:
            return text
        
        # Need another text to cut from
        if cut_from is None:
            logger.debug("No text provided for cutting")
            return text
        
        # Sample cut ratio
        lambda_val = self._sample_lambda()
        
        # Apply cutting based on strategy
        if self.config.cut_strategy == "continuous":
            mixed_text = self._continuous_cutmix(text, cut_from, lambda_val)
        elif self.config.cut_strategy == "random":
            mixed_text = self._random_cutmix(text, cut_from, lambda_val)
        elif self.config.cut_strategy == "sentence":
            mixed_text = self._sentence_cutmix(text, cut_from, lambda_val)
        elif self.config.cut_strategy == "phrase":
            mixed_text = self._phrase_cutmix(text, cut_from, lambda_val)
        else:
            mixed_text = text
        
        # Mix labels if provided
        mixed_label = None
        if label is not None and cut_label is not None:
            mixed_label = self._mix_labels(label, cut_label, lambda_val)
        
        # Return based on whether labels are mixed
        if mixed_label is not None:
            return mixed_text, mixed_label
        else:
            return mixed_text
    
    def _sample_lambda(self) -> float:
        """
        Sample cut ratio from Beta distribution.
        
        Following sampling from:
        - Yun et al. (2019): "CutMix"
        """
        if self.config.alpha > 0:
            lambda_val = np.random.beta(self.config.alpha, self.config.alpha)
        else:
            lambda_val = 0.5
        
        # Clip to valid range
        lambda_val = np.clip(
            lambda_val,
            self.config.min_cut_ratio,
            self.config.max_cut_ratio
        )
        
        return lambda_val
    
    def _continuous_cutmix(
        self,
        text1: str,
        text2: str,
        lambda_val: float
    ) -> str:
        """
        Continuous span cutting and mixing.
        
        Following continuous cutting from:
        - Yun et al. (2019): "CutMix"
        """
        words1 = text1.split()
        words2 = text2.split()
        
        # Calculate cut size
        cut_size1 = int(len(words1) * (1 - lambda_val))
        cut_size2 = int(len(words2) * lambda_val)
        
        # Determine cut positions
        if self.config.random_position:
            if len(words1) > cut_size1:
                start1 = self.rng.randint(0, len(words1) - cut_size1)
            else:
                start1 = 0
            
            if len(words2) > cut_size2:
                start2 = self.rng.randint(0, len(words2) - cut_size2)
            else:
                start2 = 0
        else:
            # Center cut
            start1 = max(0, (len(words1) - cut_size1) // 2)
            start2 = max(0, (len(words2) - cut_size2) // 2)
        
        # Perform cutting and mixing
        mixed = (
            words1[:start1] +
            words2[start2:start2 + cut_size2] +
            words1[start1 + cut_size1:]
        )
        
        return " ".join(mixed)
    
    def _random_cutmix(
        self,
        text1: str,
        text2: str,
        lambda_val: float
    ) -> str:
        """
        Random word cutting and mixing.
        
        Following random cutting from:
        - Singh et al. (2022): "CutMixNLP"
        """
        words1 = text1.split()
        words2 = text2.split()
        
        # Randomly select words to keep from each text
        num_words1 = int(len(words1) * lambda_val)
        num_words2 = int(len(words2) * (1 - lambda_val))
        
        # Random sampling
        if num_words1 > 0:
            indices1 = self.rng.sample(range(len(words1)), min(num_words1, len(words1)))
            selected1 = [words1[i] for i in sorted(indices1)]
        else:
            selected1 = []
        
        if num_words2 > 0:
            indices2 = self.rng.sample(range(len(words2)), min(num_words2, len(words2)))
            selected2 = [words2[i] for i in sorted(indices2)]
        else:
            selected2 = []
        
        # Combine selected words
        mixed = selected1 + selected2
        
        return " ".join(mixed)
    
    def _sentence_cutmix(
        self,
        text1: str,
        text2: str,
        lambda_val: float
    ) -> str:
        """
        Sentence-level cutting and mixing.
        
        Following sentence-aware cutting from:
        - Singh et al. (2022): "CutMixNLP"
        """
        import re
        
        # Split into sentences
        sentences1 = re.split(r'[.!?]+', text1)
        sentences2 = re.split(r'[.!?]+', text2)
        
        # Filter empty sentences
        sentences1 = [s.strip() for s in sentences1 if s.strip()]
        sentences2 = [s.strip() for s in sentences2 if s.strip()]
        
        # Calculate number of sentences to keep
        num_sent1 = int(len(sentences1) * lambda_val)
        num_sent2 = int(len(sentences2) * (1 - lambda_val))
        
        # Select sentences
        if self.config.random_position:
            selected1 = self.rng.sample(sentences1, min(num_sent1, len(sentences1)))
            selected2 = self.rng.sample(sentences2, min(num_sent2, len(sentences2)))
        else:
            selected1 = sentences1[:num_sent1]
            selected2 = sentences2[-num_sent2:] if num_sent2 > 0 else []
        
        # Combine sentences
        mixed = selected1 + selected2
        
        return ". ".join(mixed) + "."
    
    def _phrase_cutmix(
        self,
        text1: str,
        text2: str,
        lambda_val: float
    ) -> str:
        """
        Phrase-level cutting and mixing.
        
        Using linguistic structure for cutting following:
        - Singh et al. (2022): "CutMixNLP"
        """
        if hasattr(self, 'nlp'):
            # Use spaCy for phrase extraction
            doc1 = self.nlp(text1)
            doc2 = self.nlp(text2)
            
            # Extract noun phrases
            phrases1 = [chunk.text for chunk in doc1.noun_chunks]
            phrases2 = [chunk.text for chunk in doc2.noun_chunks]
            
            # Mix phrases
            num_phrases1 = int(len(phrases1) * lambda_val)
            num_phrases2 = int(len(phrases2) * (1 - lambda_val))
            
            mixed_phrases = (
                self.rng.sample(phrases1, min(num_phrases1, len(phrases1))) +
                self.rng.sample(phrases2, min(num_phrases2, len(phrases2)))
            )
            
            return " ".join(mixed_phrases)
        else:
            # Fallback to continuous cutting
            return self._continuous_cutmix(text1, text2, lambda_val)
    
    def _mix_labels(
        self,
        label1: int,
        label2: int,
        lambda_val: float
    ) -> np.ndarray:
        """
        Mix labels based on cut ratio.
        
        Following label mixing from:
        - Yun et al. (2019): "CutMix"
        """
        num_classes = max(label1, label2) + 1
        
        # Create one-hot vectors
        y1 = np.zeros(num_classes)
        y2 = np.zeros(num_classes)
        y1[label1] = 1
        y2[label2] = 1
        
        # Mix labels based on area ratio
        mixed_label = lambda_val * y1 + (1 - lambda_val) * y2
        
        return mixed_label
    
    def adaptive_cutmix(
        self,
        text1: str,
        text2: str,
        difficulty_score: float = 0.5
    ) -> str:
        """
        Adaptive CutMix based on sample difficulty.
        
        Following adaptive mixing from:
        - Wang et al. (2021): "Adaptive Consistency Regularization"
        """
        # Adjust cut ratio based on difficulty
        adaptive_lambda = self._sample_lambda() * (1 + difficulty_score)
        adaptive_lambda = np.clip(adaptive_lambda, 0, 1)
        
        return self._continuous_cutmix(text1, text2, adaptive_lambda)
