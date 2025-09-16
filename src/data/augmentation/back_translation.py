"""
Back Translation Augmentation Module
=====================================

Implements back-translation for data augmentation following:
- Sennrich et al. (2016): "Improving Neural Machine Translation Models with Monolingual Data"
- Edunov et al. (2018): "Understanding Back-Translation at Scale"
- Xie et al. (2020): "Unsupervised Data Augmentation for Consistency Training"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
import random
from pathlib import Path

import torch
from transformers import MarianMTModel, MarianTokenizer

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.augmentation.base_augmenter import BaseAugmenter, AugmentationConfig
from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class BackTranslationConfig(AugmentationConfig):
    """Configuration for back-translation augmentation."""
    
    # Translation models
    pivot_languages: List[str] = None
    model_names: Dict[str, Tuple[str, str]] = None  # lang -> (forward_model, backward_model)
    
    # Generation parameters
    num_beams: int = 5
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    do_sample: bool = True
    
    # Quality control
    min_bleu_score: float = 0.3
    max_repetition: float = 0.5
    
    def __post_init__(self):
        """Initialize default models."""
        if self.pivot_languages is None:
            self.pivot_languages = ['de', 'fr', 'es']
        
        if self.model_names is None:
            self.model_names = {
                'de': ('Helsinki-NLP/opus-mt-en-de', 'Helsinki-NLP/opus-mt-de-en'),
                'fr': ('Helsinki-NLP/opus-mt-en-fr', 'Helsinki-NLP/opus-mt-fr-en'),
                'es': ('Helsinki-NLP/opus-mt-en-es', 'Helsinki-NLP/opus-mt-es-en'),
                'zh': ('Helsinki-NLP/opus-mt-en-zh', 'Helsinki-NLP/opus-mt-zh-en'),
                'ru': ('Helsinki-NLP/opus-mt-en-ru', 'Helsinki-NLP/opus-mt-ru-en'),
            }

class BackTranslationAugmenter(BaseAugmenter):
    """
    Back-translation augmenter using neural translation models.
    
    Implements techniques from:
    - Fadaee et al. (2017): "Data Augmentation for Low-Resource Neural Machine Translation"
    - Graça et al. (2019): "Generalizing Back-Translation in Neural Machine Translation"
    """
    
    def __init__(
        self,
        config: Optional[BackTranslationConfig] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize back-translation augmenter.
        
        Args:
            config: Back-translation configuration
            device: Computing device
        """
        super().__init__(config or BackTranslationConfig(), name="back_translation")
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load translation models
        self.models = {}
        self.tokenizers = {}
        self._load_models()
        
        logger.info(f"Initialized back-translation with languages: {self.config.pivot_languages}")
    
    def _load_models(self):
        """Load translation models for pivot languages."""
        for lang in self.config.pivot_languages:
            if lang in self.config.model_names:
                forward_name, backward_name = self.config.model_names[lang]
                
                try:
                    # Load forward model (en -> pivot)
                    forward_tokenizer = MarianTokenizer.from_pretrained(forward_name)
                    forward_model = MarianMTModel.from_pretrained(forward_name).to(self.device)
                    forward_model.eval()
                    
                    # Load backward model (pivot -> en)
                    backward_tokenizer = MarianTokenizer.from_pretrained(backward_name)
                    backward_model = MarianMTModel.from_pretrained(backward_name).to(self.device)
                    backward_model.eval()
                    
                    self.models[lang] = {
                        'forward': forward_model,
                        'backward': backward_model
                    }
                    self.tokenizers[lang] = {
                        'forward': forward_tokenizer,
                        'backward': backward_tokenizer
                    }
                    
                    logger.info(f"Loaded translation models for {lang}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load models for {lang}: {e}")
    
    def augment_single(
        self,
        text: str,
        label: Optional[int] = None,
        pivot_language: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Augment text using back-translation.
        
        Args:
            text: Input text
            label: Optional label
            pivot_language: Specific pivot language to use
            **kwargs: Additional generation arguments
            
        Returns:
            List of back-translated texts
        """
        # Check cache
        cached = self.get_from_cache(text, pivot=pivot_language)
        if cached:
            return cached
        
        # Select pivot language
        if pivot_language and pivot_language in self.models:
            languages = [pivot_language]
        else:
            # Use all available languages
            languages = list(self.models.keys())
        
        augmented = []
        
        for lang in languages:
            try:
                # Translate to pivot language
                translated = self._translate(
                    text,
                    self.tokenizers[lang]['forward'],
                    self.models[lang]['forward'],
                    **kwargs
                )
                
                # Translate back to English
                back_translated = self._translate(
                    translated,
                    self.tokenizers[lang]['backward'],
                    self.models[lang]['backward'],
                    **kwargs
                )
                
                # Validate quality
                if self._validate_translation(text, back_translated):
                    augmented.append(back_translated)
                    
            except Exception as e:
                logger.debug(f"Translation failed for {lang}: {e}")
                self.stats['failed'] += 1
        
        # Filter augmentations
        augmented = self.filter_augmentations(text, augmented, label)
        
        # Cache results
        self.add_to_cache(text, augmented, pivot=pivot_language)
        
        return augmented if augmented else [text]
    
    def _translate(
        self,
        text: str,
        tokenizer: MarianTokenizer,
        model: MarianMTModel,
        **kwargs
    ) -> str:
        """
        Translate text using MarianMT model.
        
        Following translation strategies from:
        - Ott et al. (2018): "Scaling Neural Machine Translation"
        """
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Set generation parameters
        gen_kwargs = {
            'num_beams': kwargs.get('num_beams', self.config.num_beams),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'top_k': kwargs.get('top_k', self.config.top_k),
            'top_p': kwargs.get('top_p', self.config.top_p),
            'do_sample': kwargs.get('do_sample', self.config.do_sample),
            'max_length': 512,
            'early_stopping': True,
        }
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
        
        # Decode
        translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated
    
    def _validate_translation(self, original: str, translated: str) -> bool:
        """
        Validate translation quality.
        
        Following quality metrics from:
        - Papineni et al. (2002): "BLEU: a Method for Automatic Evaluation of Machine Translation"
        """
        # Length check
        orig_len = len(original.split())
        trans_len = len(translated.split())
        
        if trans_len < orig_len * 0.5 or trans_len > orig_len * 2.0:
            return False
        
        # Repetition check
        words = translated.split()
        if len(words) > 0:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < (1 - self.config.max_repetition):
                return False
        
        # Basic similarity check (avoid identical translations)
        if translated.lower() == original.lower():
            return False
        
        return True
    
    def parallel_augment(
        self,
        texts: List[str],
        num_workers: int = 4
    ) -> List[List[str]]:
        """
        Augment multiple texts in parallel.
        
        Implements parallel processing from:
        - Ott et al. (2019): "fairseq: A Fast, Extensible Toolkit"
        """
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.augment_single, text) for text in texts]
            results = [future.result() for future in futures]
        
        return results
    
    def diversify_translations(
        self,
        text: str,
        num_variants: int = 3
    ) -> List[str]:
        """
        Generate diverse translations using different parameters.
        
        Following diversity strategies from:
        - Vijayakumar et al. (2018): "Diverse Beam Search"
        """
        variants = []
        
        for i in range(num_variants):
            # Vary parameters for diversity
            kwargs = {
                'temperature': 0.8 + (i * 0.2),
                'top_p': 0.9 - (i * 0.1),
                'num_beams': 3 + i
            }
            
            augmented = self.augment_single(text, **kwargs)
            variants.extend(augmented[:1])  # Take first from each
        
        return list(set(variants))  # Remove duplicates
