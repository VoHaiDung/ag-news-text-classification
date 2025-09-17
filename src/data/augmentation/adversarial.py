"""
Adversarial Augmentation Module
================================

Implements adversarial text generation following:
- Miyato et al. (2017): "Adversarial Training Methods for Semi-Supervised Text Classification"
- Zhang et al. (2019): "PAWS: Paraphrase Adversaries from Word Scrambling"
- Morris et al. (2020): "TextAttack: A Framework for Adversarial Attacks"
- Jin et al. (2020): "Is BERT Really Robust? A Strong Baseline for Natural Language Attack"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass
import random
from pathlib import Path
import numpy as np

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation.base_augmenter import BaseAugmenter, AugmentationConfig
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class AdversarialConfig(AugmentationConfig):
    """
    Configuration for adversarial augmentation.
    
    Following configurations from:
    - Goodfellow et al. (2015): "Explaining and Harnessing Adversarial Examples"
    - Madry et al. (2018): "Towards Deep Learning Models Resistant to Adversarial Attacks"
    """
    
    # Attack parameters
    epsilon: float = 0.1  # Perturbation budget
    alpha: float = 0.01   # Step size
    num_iterations: int = 5  # PGD iterations
    
    # Attack strategies
    attack_type: str = "word_substitution"  # word_substitution, char_flip, embedding
    use_gradient: bool = True
    targeted: bool = False
    
    # Constraints
    max_perturb_ratio: float = 0.2  # Maximum ratio of words to perturb
    use_synonyms_only: bool = True
    preserve_semantics: bool = True
    
    # Word importance
    use_word_importance: bool = True
    importance_method: str = "gradient"  # gradient, attention, deletion
    
    # Model for attack
    victim_model: str = "bert-base-uncased"
    use_ensemble: bool = False
    
    # Quality control
    min_semantic_similarity: float = 0.8
    max_grammatical_errors: int = 2

class AdversarialAugmenter(BaseAugmenter):
    """
    Adversarial augmenter for generating adversarial examples.
    
    Implements attack methods from:
    - Alzantot et al. (2018): "Generating Natural Language Adversarial Examples"
    - Ren et al. (2019): "Generating Natural Language Adversarial Examples through Probability Weighted Word Saliency"
    - Li et al. (2020): "BERT-ATTACK: Adversarial Attack Against BERT Using BERT"
    """
    
    def __init__(
        self,
        config: Optional[AdversarialConfig] = None,
        victim_model: Optional[Any] = None,
        tokenizer: Optional[AutoTokenizer] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize adversarial augmenter.
        
        Args:
            config: Adversarial configuration
            victim_model: Model to attack
            tokenizer: Tokenizer for the model
            device: Computing device
        """
        super().__init__(config or AdversarialConfig(), name="adversarial")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load victim model if not provided
        if victim_model is None:
            self._load_victim_model()
        else:
            self.victim_model = victim_model
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.config.victim_model)
        
        # Initialize attack components
        self._initialize_attack_components()
        
        logger.info(f"Initialized adversarial augmenter with attack: {self.config.attack_type}")
    
    def _load_victim_model(self):
        """Load the victim model for adversarial attacks."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.victim_model)
            self.victim_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.victim_model,
                num_labels=4  # AG News has 4 classes
            ).to(self.device)
            self.victim_model.eval()
            logger.info(f"Loaded victim model: {self.config.victim_model}")
        except Exception as e:
            logger.error(f"Failed to load victim model: {e}")
            raise
    
    def _initialize_attack_components(self):
        """Initialize components for adversarial attacks."""
        # Load synonym dictionary for word substitution
        if self.config.attack_type == "word_substitution":
            self._load_synonyms()
        
        # Initialize embedding space for embedding attacks
        if self.config.attack_type == "embedding":
            self._initialize_embeddings()
    
    def _load_synonyms(self):
        """Load synonym dictionary for word substitution attacks."""
        from nltk.corpus import wordnet
        self.wordnet = wordnet
        logger.info("Loaded WordNet for synonym-based attacks")
    
    def _initialize_embeddings(self):
        """Initialize embedding space for embedding-based attacks."""
        # This would load pre-trained embeddings
        pass
    
    def augment_single(
        self,
        text: str,
        label: Optional[int] = None,
        target_label: Optional[int] = None,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate adversarial example for single text.
        
        Args:
            text: Input text
            label: True label
            target_label: Target label for targeted attacks
            **kwargs: Additional arguments
            
        Returns:
            Adversarial text(s)
        """
        # Check cache
        cached = self.get_from_cache(text, attack=self.config.attack_type)
        if cached:
            return cached
        
        # Apply attack based on type
        if self.config.attack_type == "word_substitution":
            adv_text = self._word_substitution_attack(text, label, target_label)
        elif self.config.attack_type == "char_flip":
            adv_text = self._character_flip_attack(text)
        elif self.config.attack_type == "embedding":
            adv_text = self._embedding_attack(text, label, target_label)
        else:
            adv_text = text
        
        # Ensure list format
        if isinstance(adv_text, str):
            adv_text = [adv_text]
        
        # Validate adversarial examples
        adv_text = self._validate_adversarial(text, adv_text, label)
        
        # Cache results
        self.add_to_cache(text, adv_text, attack=self.config.attack_type)
        
        return adv_text if adv_text else [text]
    
    def _word_substitution_attack(
        self,
        text: str,
        label: Optional[int] = None,
        target_label: Optional[int] = None
    ) -> str:
        """
        Word substitution attack.
        
        Following attacks from:
        - Alzantot et al. (2018): "Generating Natural Language Adversarial Examples"
        - Jin et al. (2020): "Is BERT Really Robust?"
        """
        words = text.split()
        
        # Get word importance scores
        if self.config.use_word_importance:
            importance_scores = self._get_word_importance(text, words, label)
        else:
            importance_scores = [1.0] * len(words)
        
        # Sort words by importance
        word_indices = sorted(
            range(len(words)),
            key=lambda i: importance_scores[i],
            reverse=True
        )
        
        # Perturb important words
        max_perturb = int(len(words) * self.config.max_perturb_ratio)
        perturbed_words = words.copy()
        num_perturbed = 0
        
        for idx in word_indices:
            if num_perturbed >= max_perturb:
                break
            
            # Get substitution candidates
            candidates = self._get_substitution_candidates(words[idx])
            
            if not candidates:
                continue
            
            # Try each candidate
            best_candidate = None
            best_score = -float('inf')
            
            for candidate in candidates:
                # Create perturbed text
                temp_words = perturbed_words.copy()
                temp_words[idx] = candidate
                temp_text = " ".join(temp_words)
                
                # Evaluate attack effectiveness
                score = self._evaluate_attack(temp_text, label, target_label)
                
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            
            # Apply best substitution
            if best_candidate:
                perturbed_words[idx] = best_candidate
                num_perturbed += 1
        
        return " ".join(perturbed_words)
    
    def _character_flip_attack(self, text: str) -> str:
        """
        Character-level perturbation attack.
        
        Following character attacks from:
        - Ebrahimi et al. (2018): "HotFlip: White-Box Adversarial Examples"
        """
        chars = list(text)
        num_flips = max(1, int(len(chars) * 0.05))  # Flip 5% of characters
        
        # Random positions to flip
        flip_positions = self.rng.sample(
            range(len(chars)),
            min(num_flips, len(chars))
        )
        
        for pos in flip_positions:
            if chars[pos].isalpha():
                # Simple character substitution
                if chars[pos].islower():
                    chars[pos] = chr((ord(chars[pos]) - ord('a') + 1) % 26 + ord('a'))
                else:
                    chars[pos] = chr((ord(chars[pos]) - ord('A') + 1) % 26 + ord('A'))
        
        return "".join(chars)
    
    def _embedding_attack(
        self,
        text: str,
        label: Optional[int] = None,
        target_label: Optional[int] = None
    ) -> str:
        """
        Embedding-based adversarial attack.
        
        Following embedding attacks from:
        - Miyato et al. (2017): "Adversarial Training Methods"
        """
        # This would perform attacks in embedding space
        # For demonstration, fallback to word substitution
        return self._word_substitution_attack(text, label, target_label)
    
    def _get_word_importance(
        self,
        text: str,
        words: List[str],
        label: Optional[int] = None
    ) -> List[float]:
        """
        Calculate word importance scores.
        
        Following importance scoring from:
        - Ren et al. (2019): "Generating Natural Language Adversarial Examples"
        """
        if self.config.importance_method == "gradient":
            return self._gradient_based_importance(text, words, label)
        elif self.config.importance_method == "attention":
            return self._attention_based_importance(text, words)
        elif self.config.importance_method == "deletion":
            return self._deletion_based_importance(text, words, label)
        else:
            return [1.0] * len(words)
    
    def _gradient_based_importance(
        self,
        text: str,
        words: List[str],
        label: Optional[int] = None
    ) -> List[float]:
        """Calculate importance using gradients."""
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Forward pass with gradients
        self.victim_model.zero_grad()
        outputs = self.victim_model(**inputs)
        
        if label is not None:
            loss = F.cross_entropy(outputs.logits, torch.tensor([label]).to(self.device))
            loss.backward()
        
        # Get token embeddings gradient
        # This is simplified; actual implementation would be more complex
        importance = [1.0] * len(words)
        
        return importance
    
    def _attention_based_importance(
        self,
        text: str,
        words: List[str]
    ) -> List[float]:
        """Calculate importance using attention scores."""
        # This would use attention weights from the model
        return [1.0] * len(words)
    
    def _deletion_based_importance(
        self,
        text: str,
        words: List[str],
        label: Optional[int] = None
    ) -> List[float]:
        """Calculate importance by deletion impact."""
        importance = []
        
        # Get original prediction
        original_score = self._get_prediction_score(text, label)
        
        for i in range(len(words)):
            # Delete word and measure impact
            deleted_text = " ".join(words[:i] + words[i+1:])
            deleted_score = self._get_prediction_score(deleted_text, label)
            
            importance.append(abs(original_score - deleted_score))
        
        return importance
    
    def _get_substitution_candidates(self, word: str) -> List[str]:
        """Get substitution candidates for a word."""
        candidates = []
        
        if self.config.use_synonyms_only and hasattr(self, 'wordnet'):
            # Get synonyms from WordNet
            for syn in self.wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        candidates.append(synonym)
        else:
            # Use other methods (e.g., embedding neighbors)
            pass
        
        return candidates[:10]  # Limit candidates
    
    def _evaluate_attack(
        self,
        text: str,
        true_label: Optional[int] = None,
        target_label: Optional[int] = None
    ) -> float:
        """Evaluate attack effectiveness."""
        score = self._get_prediction_score(text, target_label if self.config.targeted else true_label)
        
        if self.config.targeted:
            # For targeted attack, maximize target class probability
            return score
        else:
            # For untargeted attack, minimize true class probability
            return -score if true_label is not None else score
    
    def _get_prediction_score(
        self,
        text: str,
        label: Optional[int] = None
    ) -> float:
        """Get model prediction score."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.victim_model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
        
        if label is not None:
            return probs[0, label].item()
        else:
            return probs.max().item()
    
    def _validate_adversarial(
        self,
        original: str,
        adversarial: List[str],
        label: Optional[int] = None
    ) -> List[str]:
        """Validate adversarial examples."""
        valid = []
        
        for adv in adversarial:
            # Check semantic similarity
            if self.config.preserve_semantics:
                # Would use semantic similarity model here
                pass
            
            # Check grammatical errors
            # Would use grammar checker here
            
            valid.append(adv)
        
        return valid
