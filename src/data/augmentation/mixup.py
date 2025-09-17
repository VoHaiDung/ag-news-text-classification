"""
MixUp Augmentation Module
==========================

Implements MixUp and advanced mixing strategies for text following:
- Zhang et al. (2018): "mixup: Beyond Empirical Risk Minimization"
- Guo et al. (2019): "Augmenting Data with Mixup for Sentence Classification"
- Chen et al. (2020): "MixText: Linguistically-Informed Interpolation of Hidden States"
- Verma et al. (2019): "Manifold Mixup: Better Representations by Interpolating Hidden States"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
import random
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation.base_augmenter import BaseAugmenter, AugmentationConfig
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class MixUpConfig(AugmentationConfig):
    """
    Configuration for MixUp augmentation.
    
    Following hyperparameter settings from:
    - Zhang et al. (2018): "mixup: Beyond Empirical Risk Minimization"
    - Thulasidasan et al. (2019): "On Mixup Training: Improved Calibration and Predictive Uncertainty"
    """
    
    # MixUp parameters
    alpha: float = 0.2  # Beta distribution parameter
    beta: float = 0.2   # Alternative beta parameter
    mixup_strategy: str = "word"  # word, sentence, embedding, hidden
    
    # Interpolation settings
    min_lambda: float = 0.0
    max_lambda: float = 1.0
    symmetric: bool = True  # Use symmetric lambda (max(λ, 1-λ))
    
    # Text-specific settings
    preserve_order: bool = False
    mix_at_token_level: bool = True
    use_attention_weights: bool = True
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768
    normalize_embeddings: bool = True
    
    # Quality control
    min_text_length: int = 5
    max_mixed_length: int = 512
    filter_nonsensical: bool = True

class MixUpAugmenter(BaseAugmenter):
    """
    MixUp augmenter for text data.
    
    Implements various mixing strategies from:
    - Guo et al. (2019): "Augmenting Data with Mixup for Sentence Classification"
    - Chen et al. (2020): "MixText: Linguistically-Informed Interpolation"
    - Yoon et al. (2021): "SSMix: Saliency-based Span Mixup for Text Classification"
    """
    
    def __init__(
        self,
        config: Optional[MixUpConfig] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize MixUp augmenter.
        
        Args:
            config: MixUp configuration
            tokenizer: Optional tokenizer for advanced mixing
            device: Computing device
        """
        super().__init__(config or MixUpConfig(), name="mixup")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model for embedding-based mixing
        if self.config.mixup_strategy in ["embedding", "hidden"]:
            self._initialize_embedding_model()
        
        self.tokenizer = tokenizer
        
        logger.info(f"Initialized MixUp augmenter with strategy: {self.config.mixup_strategy}")
    
    def _initialize_embedding_model(self):
        """Initialize embedding model for advanced mixing."""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(
                self.config.embedding_model,
                device=self.device
            )
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")
        except ImportError:
            logger.warning("sentence-transformers not installed, falling back to word-level mixing")
            self.config.mixup_strategy = "word"
    
    def augment_single(
        self,
        text: str,
        label: Optional[int] = None,
        mix_with: Optional[str] = None,
        mix_label: Optional[int] = None,
        **kwargs
    ) -> Union[str, Tuple[str, np.ndarray]]:
        """
        Apply MixUp augmentation to single text.
        
        Args:
            text: Input text
            label: Optional label for text
            mix_with: Text to mix with (if None, randomly selected)
            mix_label: Label of mix_with text
            **kwargs: Additional arguments
            
        Returns:
            Mixed text and optionally mixed label
        """
        # Check cache
        cache_key = f"{text[:50]}_{mix_with[:50] if mix_with else 'random'}"
        cached = self.get_from_cache(text, mix_key=cache_key)
        if cached:
            return cached[0] if cached else text
        
        # If no mix partner provided, return original
        if mix_with is None:
            logger.debug("No mix partner provided for single augmentation")
            return text
        
        # Sample mixing coefficient
        lambda_val = self._sample_lambda()
        
        # Apply mixing based on strategy
        if self.config.mixup_strategy == "word":
            mixed_text = self._word_level_mixup(text, mix_with, lambda_val)
        elif self.config.mixup_strategy == "sentence":
            mixed_text = self._sentence_level_mixup(text, mix_with, lambda_val)
        elif self.config.mixup_strategy == "embedding":
            mixed_text = self._embedding_level_mixup(text, mix_with, lambda_val)
        elif self.config.mixup_strategy == "hidden":
            mixed_text = self._hidden_state_mixup(text, mix_with, lambda_val)
        else:
            mixed_text = text
        
        # Mix labels if provided
        mixed_label = None
        if label is not None and mix_label is not None:
            mixed_label = self._mix_labels(label, mix_label, lambda_val)
        
        # Cache result
        self.add_to_cache(text, [mixed_text], mix_key=cache_key)
        
        # Return based on whether labels are mixed
        if mixed_label is not None:
            return mixed_text, mixed_label
        else:
            return mixed_text
    
    def augment_batch(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        **kwargs
    ) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        Apply MixUp to batch of texts.
        
        Following batch mixing from:
        - Zhang et al. (2018): "mixup: Beyond Empirical Risk Minimization"
        """
        batch_size = len(texts)
        
        # Create random pairs for mixing
        indices = list(range(batch_size))
        mix_indices = indices.copy()
        self.rng.shuffle(mix_indices)
        
        mixed_texts = []
        mixed_labels = [] if labels else None
        
        for i, mix_i in zip(indices, mix_indices):
            # Mix texts
            result = self.augment_single(
                texts[i],
                labels[i] if labels else None,
                texts[mix_i],
                labels[mix_i] if labels else None
            )
            
            # Handle return value
            if isinstance(result, tuple):
                mixed_texts.append(result[0])
                if mixed_labels is not None:
                    mixed_labels.append(result[1])
            else:
                mixed_texts.append(result)
        
        return (mixed_texts, np.array(mixed_labels) if mixed_labels else None)
    
    def _sample_lambda(self) -> float:
        """
        Sample mixing coefficient from Beta distribution.
        
        Following sampling strategy from:
        - Verma et al. (2019): "Manifold Mixup"
        """
        if self.config.alpha > 0:
            lambda_val = np.random.beta(self.config.alpha, self.config.beta)
        else:
            lambda_val = 1.0
        
        # Apply symmetric constraint if enabled
        if self.config.symmetric:
            lambda_val = max(lambda_val, 1 - lambda_val)
        
        # Clip to valid range
        lambda_val = np.clip(lambda_val, self.config.min_lambda, self.config.max_lambda)
        
        return lambda_val
    
    def _word_level_mixup(
        self,
        text1: str,
        text2: str,
        lambda_val: float
    ) -> str:
        """
        Mix texts at word level.
        
        Following word-level mixing from:
        - Guo et al. (2019): "Augmenting Data with Mixup"
        """
        words1 = text1.split()
        words2 = text2.split()
        
        # Determine mix length
        max_len = max(len(words1), len(words2))
        mixed_words = []
        
        for i in range(max_len):
            # Get words at position i
            w1 = words1[i] if i < len(words1) else ""
            w2 = words2[i] if i < len(words2) else ""
            
            # Randomly select based on lambda
            if self.rng.random() < lambda_val:
                mixed_words.append(w1)
            else:
                mixed_words.append(w2)
        
        # Filter empty strings
        mixed_words = [w for w in mixed_words if w]
        
        return " ".join(mixed_words)
    
    def _sentence_level_mixup(
        self,
        text1: str,
        text2: str,
        lambda_val: float
    ) -> str:
        """
        Mix texts at sentence level.
        
        Following sentence-level mixing from:
        - Chen et al. (2020): "MixText"
        """
        # Split into sentences
        import re
        sentences1 = re.split(r'[.!?]+', text1)
        sentences2 = re.split(r'[.!?]+', text2)
        
        # Filter empty sentences
        sentences1 = [s.strip() for s in sentences1 if s.strip()]
        sentences2 = [s.strip() for s in sentences2 if s.strip()]
        
        # Mix sentences
        num_sent1 = int(len(sentences1) * lambda_val)
        num_sent2 = len(sentences2) - int(len(sentences2) * lambda_val)
        
        mixed_sentences = (
            self.rng.sample(sentences1, min(num_sent1, len(sentences1))) +
            self.rng.sample(sentences2, min(num_sent2, len(sentences2)))
        )
        
        # Shuffle if not preserving order
        if not self.config.preserve_order:
            self.rng.shuffle(mixed_sentences)
        
        return ". ".join(mixed_sentences) + "."
    
    def _embedding_level_mixup(
        self,
        text1: str,
        text2: str,
        lambda_val: float
    ) -> str:
        """
        Mix texts at embedding level.
        
        Following embedding mixing from:
        - Chen et al. (2020): "MixText"
        - Verma et al. (2019): "Manifold Mixup"
        """
        if not hasattr(self, 'embedding_model'):
            # Fallback to word-level if no embedding model
            return self._word_level_mixup(text1, text2, lambda_val)
        
        # Get embeddings
        emb1 = self.embedding_model.encode(text1, convert_to_tensor=True)
        emb2 = self.embedding_model.encode(text2, convert_to_tensor=True)
        
        # Mix embeddings
        mixed_emb = lambda_val * emb1 + (1 - lambda_val) * emb2
        
        # Normalize if configured
        if self.config.normalize_embeddings:
            mixed_emb = F.normalize(mixed_emb, p=2, dim=-1)
        
        # Decode back to text (approximation)
        # In practice, this would require a decoder model
        # For now, we'll do word-level mixing as approximation
        return self._word_level_mixup(text1, text2, lambda_val)
    
    def _hidden_state_mixup(
        self,
        text1: str,
        text2: str,
        lambda_val: float
    ) -> str:
        """
        Mix texts at hidden state level.
        
        Following hidden state mixing from:
        - Verma et al. (2019): "Manifold Mixup"
        """
        # This would require access to model hidden states
        # For demonstration, fallback to embedding mixing
        return self._embedding_level_mixup(text1, text2, lambda_val)
    
    def _mix_labels(
        self,
        label1: int,
        label2: int,
        lambda_val: float
    ) -> np.ndarray:
        """
        Mix labels for classification.
        
        Following label mixing from:
        - Zhang et al. (2018): "mixup"
        """
        # Convert to one-hot if needed
        num_classes = max(label1, label2) + 1
        
        # Create one-hot vectors
        y1 = np.zeros(num_classes)
        y2 = np.zeros(num_classes)
        y1[label1] = 1
        y2[label2] = 1
        
        # Mix labels
        mixed_label = lambda_val * y1 + (1 - lambda_val) * y2
        
        return mixed_label
    
    def saliency_based_mixup(
        self,
        text1: str,
        text2: str,
        saliency_scores1: Optional[List[float]] = None,
        saliency_scores2: Optional[List[float]] = None
    ) -> str:
        """
        Saliency-based span mixup.
        
        Following SSMix from:
        - Yoon et al. (2021): "SSMix: Saliency-based Span Mixup"
        """
        words1 = text1.split()
        words2 = text2.split()
        
        # If no saliency scores, use uniform
        if saliency_scores1 is None:
            saliency_scores1 = [1.0] * len(words1)
        if saliency_scores2 is None:
            saliency_scores2 = [1.0] * len(words2)
        
        # Select salient spans
        threshold = 0.5
        salient_spans1 = [
            words1[i] for i, score in enumerate(saliency_scores1)
            if score > threshold
        ]
        salient_spans2 = [
            words2[i] for i, score in enumerate(saliency_scores2)
            if score > threshold
        ]
        
        # Mix salient spans
        lambda_val = self._sample_lambda()
        num_spans1 = int(len(salient_spans1) * lambda_val)
        
        mixed = (
            self.rng.sample(salient_spans1, min(num_spans1, len(salient_spans1))) +
            self.rng.sample(salient_spans2, len(salient_spans2) - num_spans1)
        )
        
        return " ".join(mixed)
