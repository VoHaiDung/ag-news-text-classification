"""
Text Cleaning and Preprocessing
================================

Implements text cleaning following best practices from:
- Strubell et al. (2018): "Linguistically-Informed Self-Attention"
- Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"

Author: Võ Hải Dũng
License: MIT
"""

import re
import string
import unicodedata
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

# Download NLTK data if needed
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

@dataclass
class CleaningConfig:
    """Configuration for text cleaning."""
    
    # Basic cleaning
    lowercase: bool = False
    remove_punctuation: bool = False
    remove_digits: bool = False
    remove_stopwords: bool = False
    
    # Advanced cleaning
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_special_chars: bool = False
    
    # Preserve important elements
    preserve_entities: bool = True
    preserve_hashtags: bool = True
    preserve_mentions: bool = True
    
    # Custom patterns
    custom_patterns: List[tuple] = None
    
    # Language
    language: str = "english"

class TextCleaner:
    """
    Text cleaning utility.
    
    Implements minimal cleaning to preserve information following:
    - Koehn & Knowles (2017): "Six Challenges for Neural Machine Translation"
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """
        Initialize text cleaner.
        
        Args:
            config: Cleaning configuration
        """
        self.config = config or CleaningConfig()
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Load stopwords if needed
        if self.config.remove_stopwords:
            self.stopwords = set(stopwords.words(self.config.language))
        else:
            self.stopwords = set()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.patterns = {}
        
        # URL pattern
        self.patterns['url'] = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\KATEX_INLINE_OPEN\KATEX_INLINE_CLOSE,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        
        # Email pattern
        self.patterns['email'] = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Multiple spaces
        self.patterns['spaces'] = re.compile(r'\s+')
        
        # Special characters
        self.patterns['special'] = re.compile(r'[^a-zA-Z0-9\s\.\,\!\?\-]')
        
        # Hashtags
        self.patterns['hashtag'] = re.compile(r'#\w+')
        
        # Mentions
        self.patterns['mention'] = re.compile(r'@\w+')
    
    def clean(self, text: str) -> str:
        """
        Clean text according to configuration.
        
        Args:
            text: Input text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Unicode normalization
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs
        if self.config.remove_urls:
            text = self.patterns['url'].sub(' ', text)
        
        # Remove emails
        if self.config.remove_emails:
            text = self.patterns['email'].sub(' ', text)
        
        # Lowercase
        if self.config.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove digits
        if self.config.remove_digits:
            text = ''.join(char for char in text if not char.isdigit())
        
        # Remove special characters
        if self.config.remove_special_chars:
            text = self.patterns['special'].sub(' ', text)
        
        # Remove stopwords
        if self.config.remove_stopwords:
            words = word_tokenize(text)
            text = ' '.join(word for word in words if word.lower() not in self.stopwords)
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self.patterns['spaces'].sub(' ', text)
            text = text.strip()
        
        # Apply custom patterns
        if self.config.custom_patterns:
            for pattern, replacement in self.config.custom_patterns:
                text = re.sub(pattern, replacement, text)
        
        return text
    
    def batch_clean(self, texts: List[str]) -> List[str]:
        """
        Clean multiple texts.
        
        Args:
            texts: List of texts
            
        Returns:
            List of cleaned texts
        """
        return [self.clean(text) for text in texts]
    
    def get_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get text statistics before/after cleaning.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with statistics
        """
        cleaned = self.clean(text)
        
        return {
            "original_length": len(text),
            "cleaned_length": len(cleaned),
            "reduction_ratio": 1 - len(cleaned) / max(len(text), 1),
            "original_words": len(text.split()),
            "cleaned_words": len(cleaned.split()),
            "urls_removed": len(self.patterns['url'].findall(text)),
            "emails_removed": len(self.patterns['email'].findall(text)),
        }

# Convenience functions
def get_minimal_cleaner() -> TextCleaner:
    """Get cleaner with minimal cleaning (for transformers)."""
    config = CleaningConfig(
        lowercase=False,
        remove_punctuation=False,
        remove_stopwords=False,
        normalize_whitespace=True,
        remove_urls=True,
        remove_emails=True
    )
    return TextCleaner(config)

def get_aggressive_cleaner() -> TextCleaner:
    """Get cleaner with aggressive cleaning (for classical ML)."""
    config = CleaningConfig(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        remove_digits=True,
        remove_special_chars=True
    )
    return TextCleaner(config)
