"""
Paraphrase Generation Module
============================

Implements paraphrase-based augmentation following:
- Prakash et al. (2016): "Neural Paraphrase Generation with Stacked Residual LSTM Networks"
- Li et al. (2018): "Paraphrase Generation with Deep Reinforcement Learning"
- Kumar et al. (2020): "Data Augmentation using Pre-trained Transformer Models"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    PegasusForConditionalGeneration,
    PegasusTokenizer
)

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation.base_augmenter import BaseAugmenter, AugmentationConfig
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class ParaphraseConfig(AugmentationConfig):
    """Configuration for paraphrase augmentation."""
    
    # Model selection
    model_name: str = "tuner007/pegasus_paraphrase"
    model_type: str = "pegasus"  # pegasus, t5, gpt2
    
    # Generation parameters
    num_return_sequences: int = 3
    num_beams: int = 10
    num_beam_groups: int = 5
    diversity_penalty: float = 1.0
    temperature: float = 1.2
    repetition_penalty: float = 1.2
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 3
    
    # Quality thresholds
    min_word_change_ratio: float = 0.3
    max_word_overlap: float = 0.8

class ParaphraseAugmenter(BaseAugmenter):
    """
    Paraphrase-based augmenter using neural generation models.
    
    Implements techniques from:
    - Wieting & Gimpel (2018): "ParaNMT-50M: Pushing the Limits of Paraphrastic Sentence Embeddings"
    - Thompson & Post (2020): "Automatic Machine Translation Evaluation in Many Languages via Zero-Shot Paraphrasing"
    """
    
    def __init__(
        self,
        config: Optional[ParaphraseConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize paraphrase augmenter.
        
        Args:
            config: Paraphrase configuration
            device: Computing device
        """
        super().__init__(config or ParaphraseConfig(), name="paraphrase")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self._load_model()
        
        logger.info(f"Initialized paraphrase augmenter with {self.config.model_name}")
    
    def _load_model(self):
        """Load paraphrase generation model."""
        try:
            if self.config.model_type == "pegasus":
                self.tokenizer = PegasusTokenizer.from_pretrained(self.config.model_name)
                self.model = PegasusForConditionalGeneration.from_pretrained(
                    self.config.model_name
                ).to(self.device)
            elif self.config.model_type == "t5":
                self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(
                    self.config.model_name
                ).to(self.device)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load paraphrase model: {e}")
            raise
    
    def augment_single(
        self,
        text: str,
        label: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        Generate paraphrases for input text.
        
        Args:
            text: Input text
            label: Optional label
            **kwargs: Generation arguments
            
        Returns:
            List of paraphrases
        """
        # Check cache
        cached = self.get_from_cache(text)
        if cached:
            return cached
        
        # Prepare input
        if self.config.model_type == "t5":
            input_text = f"paraphrase: {text}"
        else:
            input_text = text
        
        # Tokenize
        inputs = self.tokenizer(
            input_text,
            truncation=True,
            padding="longest",
            max_length=256,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate paraphrases
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_return_sequences=self.config.num_return_sequences,
                num_beams=self.config.num_beams,
                num_beam_groups=self.config.num_beam_groups,
                diversity_penalty=self.config.diversity_penalty,
                temperature=self.config.temperature,
                repetition_penalty=self.config.repetition_penalty,
                length_penalty=self.config.length_penalty,
                no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                max_length=256,
                early_stopping=True
            )
        
        # Decode paraphrases
        paraphrases = []
        for output in outputs:
            paraphrase = self.tokenizer.decode(output, skip_special_tokens=True)
            
            # Skip if identical to input
            if paraphrase.lower() != text.lower():
                paraphrases.append(paraphrase)
        
        # Validate paraphrases
        paraphrases = self._validate_paraphrases(text, paraphrases)
        
        # Filter augmentations
        paraphrases = self.filter_augmentations(text, paraphrases, label)
        
        # Cache results
        self.add_to_cache(text, paraphrases)
        
        return paraphrases if paraphrases else [text]
    
    def _validate_paraphrases(
        self,
        original: str,
        paraphrases: List[str]
    ) -> List[str]:
        """
        Validate paraphrase quality.
        
        Following validation from:
        - Liu et al. (2010): "Learning to Paraphrase for Question Answering"
        """
        valid_paraphrases = []
        
        original_words = set(original.lower().split())
        
        for paraphrase in paraphrases:
            para_words = set(paraphrase.lower().split())
            
            # Check word change ratio
            common_words = original_words & para_words
            change_ratio = 1 - (len(common_words) / max(len(original_words), 1))
            
            if change_ratio < self.config.min_word_change_ratio:
                continue
            
            # Check word overlap
            overlap = len(common_words) / len(original_words | para_words)
            
            if overlap > self.config.max_word_overlap:
                continue
            
            valid_paraphrases.append(paraphrase)
        
        return valid_paraphrases
    
    def generate_diverse_paraphrases(
        self,
        text: str,
        num_variants: int = 5
    ) -> List[str]:
        """
        Generate diverse paraphrases using different prompts.
        
        Following diversity techniques from:
        - Cao & Wan (2020): "DivGAN: Towards Diverse Paraphrase Generation"
        """
        prompts = [
            f"paraphrase: {text}",
            f"rephrase: {text}",
            f"rewrite: {text}",
            f"say differently: {text}",
            f"alternative: {text}"
        ]
        
        all_paraphrases = []
        
        for prompt in prompts[:num_variants]:
            # Generate with different prompt
            inputs = self.tokenizer(
                prompt,
                truncation=True,
                padding="longest",
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    num_return_sequences=1,
                    temperature=1.0 + (len(all_paraphrases) * 0.1),
                    max_length=256
                )
            
            paraphrase = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            all_paraphrases.append(paraphrase)
        
        return list(set(all_paraphrases))  # Remove duplicates
