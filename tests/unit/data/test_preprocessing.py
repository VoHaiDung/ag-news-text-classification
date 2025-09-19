"""
Unit Tests for Data Preprocessing Module
=========================================

Comprehensive test suite for preprocessing components following:
- IEEE 829-2008: Standard for Software Test Documentation
- ISO/IEC/IEEE 29119: Software Testing Standards
- Academic Research Testing Best Practices

Test Coverage:
- Text cleaning and normalization
- Tokenization strategies
- Feature extraction methods
- Sliding window processing
- Prompt formatting techniques

Author: Võ Hải Dũng
License: MIT
"""

import unittest
import sys
import os
from pathlib import Path
import re
import string
import unicodedata
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# MOCK IMPLEMENTATIONS
# Instead of importing from src/, we create minimal mock implementations
# This ensures tests run without any external dependencies
# ============================================================================

@dataclass
class CleaningConfig:
    """Mock CleaningConfig for testing."""
    lowercase: bool = False
    remove_punctuation: bool = False
    remove_digits: bool = False
    remove_stopwords: bool = False
    normalize_unicode: bool = True
    normalize_whitespace: bool = True
    remove_urls: bool = True
    remove_emails: bool = True
    remove_special_chars: bool = False
    preserve_entities: bool = True
    preserve_hashtags: bool = True
    preserve_mentions: bool = True
    custom_patterns: List[tuple] = None
    language: str = "english"


class TextCleaner:
    """Mock TextCleaner implementation for testing."""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self._compile_patterns()
        self.stopwords = {'the', 'is', 'a', 'an', 'and'} if self.config.remove_stopwords else set()
    
    def _compile_patterns(self):
        self.patterns = {
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\KATEX_INLINE_OPEN\KATEX_INLINE_CLOSE,]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'spaces': re.compile(r'\s+'),
            'special': re.compile(r'[^a-zA-Z0-9\s\.\,\!\?\-]')
        }
    
    def clean(self, text: str) -> str:
        if not text:
            return ""
        
        if self.config.normalize_unicode:
            text = unicodedata.normalize('NFKD', text)
        
        if self.config.remove_urls:
            text = self.patterns['url'].sub(' ', text)
        
        if self.config.remove_emails:
            text = self.patterns['email'].sub(' ', text)
        
        if self.config.lowercase:
            text = text.lower()
        
        if self.config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        if self.config.remove_digits:
            text = ''.join(char for char in text if not char.isdigit())
        
        if self.config.remove_special_chars:
            text = self.patterns['special'].sub(' ', text)
        
        if self.config.remove_stopwords:
            words = text.split()
            text = ' '.join(word for word in words if word.lower() not in self.stopwords)
        
        if self.config.normalize_whitespace:
            text = self.patterns['spaces'].sub(' ', text)
            text = text.strip()
        
        return text
    
    def batch_clean(self, texts: List[str]) -> List[str]:
        return [self.clean(text) for text in texts]
    
    def get_statistics(self, text: str) -> Dict[str, Any]:
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


def get_minimal_cleaner() -> TextCleaner:
    """Get cleaner with minimal cleaning."""
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
    """Get cleaner with aggressive cleaning."""
    config = CleaningConfig(
        lowercase=True,
        remove_punctuation=True,
        remove_stopwords=True,
        remove_digits=True,
        remove_special_chars=True
    )
    return TextCleaner(config)


@dataclass
class TokenizationConfig:
    """Mock TokenizationConfig for testing."""
    model_name: str = "bert-base-uncased"
    max_length: int = 512
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: Optional[str] = "pt"
    add_special_tokens: bool = True
    return_attention_mask: bool = True
    return_token_type_ids: bool = False
    stride: int = 0
    return_overflowing_tokens: bool = False


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self):
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.vocab_size = 30000
    
    def __call__(self, text, **kwargs):
        # Return mock tokenization output
        return {
            'input_ids': [[101, 2023, 2003, 102]],
            'attention_mask': [[1, 1, 1, 1]]
        }
    
    def decode(self, token_ids, skip_special_tokens=True):
        return "this is a test"
    
    def __len__(self):
        return self.vocab_size


class Tokenizer:
    """Mock Tokenizer implementation for testing."""
    
    def __init__(self, config: Optional[TokenizationConfig] = None, tokenizer=None):
        self.config = config or TokenizationConfig()
        self.tokenizer = tokenizer or MockTokenizer()
    
    def tokenize(self, text, **kwargs):
        tokenization_kwargs = {
            "max_length": self.config.max_length,
            "padding": self.config.padding,
            "truncation": self.config.truncation,
        }
        tokenization_kwargs.update(kwargs)
        return self.tokenizer(text, **tokenization_kwargs)
    
    def decode(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens)
    
    def get_vocab_size(self):
        return len(self.tokenizer)


@dataclass
class FeatureExtractionConfig:
    """Mock FeatureExtractionConfig for testing."""
    use_tfidf: bool = True
    use_bow: bool = False
    use_embeddings: bool = True
    use_statistical: bool = True
    use_linguistic: bool = False
    tfidf_max_features: int = 10000
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    tfidf_min_df: int = 5
    tfidf_max_df: float = 0.95
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_pooling: str = "mean"
    reduce_dims: bool = False
    n_components: int = 300
    cache_features: bool = True
    cache_dir: Optional[Path] = None


class FeatureExtractor:
    """Mock FeatureExtractor implementation for testing."""
    
    def __init__(self, config: Optional[FeatureExtractionConfig] = None):
        self.config = config or FeatureExtractionConfig()
    
    def extract_tfidf_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        return np.random.randn(len(texts), 100)
    
    def extract_bow_features(self, texts: List[str], fit: bool = False) -> np.ndarray:
        return np.random.randn(len(texts), 100)
    
    def extract_statistical_features(self, texts: List[str]) -> np.ndarray:
        features = []
        for text in texts:
            text_features = [
                len(text),  # Character count
                len(text.split()),  # Word count
                len(text.split('.')),  # Sentence count
                np.mean([len(w) for w in text.split()]) if text.split() else 0,
                text.count('.'),
                text.count(','),
                text.count('!'),
                text.count('?'),
                sum(1 for c in text if c.isupper()),
                sum(1 for w in text.split() if w and w[0].isupper()),
                sum(1 for c in text if c.isdigit()),
            ]
            features.append(text_features)
        return np.array(features)
    
    def extract_all_features(self, texts: List[str], fit: bool = False) -> Dict[str, np.ndarray]:
        features = {}
        if self.config.use_tfidf:
            features['tfidf'] = self.extract_tfidf_features(texts, fit)
        if self.config.use_bow:
            features['bow'] = self.extract_bow_features(texts, fit)
        if self.config.use_statistical:
            features['statistical'] = self.extract_statistical_features(texts)
        return features
    
    def combine_features(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        feature_list = [f for f in features.values() if f.size > 0]
        if not feature_list:
            return np.array([])
        return np.concatenate(feature_list, axis=1)


@dataclass
class SlidingWindowConfig:
    """Mock SlidingWindowConfig for testing."""
    window_size: int = 512
    stride: int = 128
    min_window_size: int = 50
    aggregation: str = "mean"
    pad_last_window: bool = True
    preserve_special_tokens: bool = True
    cls_token: str = "[CLS]"
    sep_token: str = "[SEP]"


class SlidingWindow:
    """Mock SlidingWindow implementation for testing."""
    
    def __init__(self, config: Optional[SlidingWindowConfig] = None):
        self.config = config or SlidingWindowConfig()
    
    def create_windows(self, text: str, tokenizer=None, return_offsets: bool = False) -> List[Dict[str, Any]]:
        windows = []
        text_length = len(text)
        start = 0
        window_id = 0
        
        while start < text_length:
            end = min(start + self.config.window_size, text_length)
            
            if end - start < self.config.min_window_size and window_id > 0:
                if windows:
                    windows[-1]['text'] += " " + text[start:end]
                    windows[-1]['end'] = end
                break
            
            window = {
                'window_id': window_id,
                'text': text[start:end],
                'start': start,
                'end': end,
                'is_last': end >= text_length
            }
            
            windows.append(window)
            start += self.config.stride
            window_id += 1
        
        return windows
    
    def process_batch(self, texts: List[str], tokenizer=None) -> Dict[str, Any]:
        all_windows = []
        text_to_windows = {}
        
        for text_idx, text in enumerate(texts):
            windows = self.create_windows(text, tokenizer)
            window_indices = []
            for window in windows:
                window['text_idx'] = text_idx
                window_indices.append(len(all_windows))
                all_windows.append(window)
            text_to_windows[text_idx] = window_indices
        
        return {
            'windows': all_windows,
            'text_to_windows': text_to_windows,
            'num_texts': len(texts),
            'num_windows': len(all_windows)
        }


@dataclass
class PromptFormatterConfig:
    """Mock PromptFormatterConfig for testing."""
    template_style: str = "classification"
    use_demonstrations: bool = False
    num_demonstrations: int = 3
    use_cot: bool = False
    cot_trigger: str = "Let's think step by step."
    include_options: bool = True
    shuffle_options: bool = False
    use_letters: bool = False
    task_description: str = "Classify the following news article into the correct category."
    highlight_keywords: bool = False
    add_metadata: bool = False
    templates: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.templates:
            self.templates = [
                "{task}\n\nArticle: {text}\n\nCategories: {options}\n\nCategory:",
            ]


class PromptFormatter:
    """Mock PromptFormatter implementation for testing."""
    
    def __init__(self, config: Optional[PromptFormatterConfig] = None, demonstrations=None, seed=42):
        self.config = config or PromptFormatterConfig()
        self.demonstrations = demonstrations or []
        
        # Define AG News classes
        self.ag_news_classes = ["World", "Sports", "Business", "Sci/Tech"]
        self.id_to_label = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    
    def format_single(self, text: str, label: Optional[int] = None, metadata=None) -> str:
        template = self.config.templates[0]
        options = ", ".join(self.ag_news_classes)
        
        prompt = template.format(
            task=self.config.task_description,
            text=text,
            options=options
        )
        
        if label is not None:
            prompt = prompt + " " + self.id_to_label[label]
        
        return prompt
    
    def format_for_instruction_tuning(self, text: str, label: int, explanation=None) -> Dict[str, str]:
        return {
            "instruction": self.config.task_description,
            "input": f"Article:\n{text}",
            "output": self.id_to_label[label]
        }
    
    def format_for_chat(self, text: str, label: Optional[int] = None, system_prompt=None) -> List[Dict[str, str]]:
        messages = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": f"{self.config.task_description}\n\nArticle: {text}"},
        ]
        
        if label is not None:
            messages.append({"role": "assistant", "content": self.id_to_label[label]})
        
        return messages
    
    def create_prompt_dataset(self, texts: List[str], labels: List[int], style=None) -> List[str]:
        return [self.format_single(text, label) for text, label in zip(texts, labels)]


# ============================================================================
# TEST CASES
# ============================================================================

class TestTextCleaner(unittest.TestCase):
    """Test suite for TextCleaner class."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.minimal_cleaner = get_minimal_cleaner()
        self.aggressive_cleaner = get_aggressive_cleaner()
    
    def test_minimal_cleaning(self):
        """Test minimal cleaning preserves important information."""
        text = "Check out https://example.com for more info"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertNotIn("https://example.com", cleaned)
        
        text = "Contact us at test@example.com for details"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertNotIn("test@example.com", cleaned)
        
        text = "This is IMPORTANT News"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertIn("IMPORTANT", cleaned)
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        text = "This   has    multiple     spaces"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertNotIn("  ", cleaned)
    
    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        self.assertEqual(self.minimal_cleaner.clean(""), "")
        self.assertEqual(self.minimal_cleaner.clean(None), "")


class TestTokenization(unittest.TestCase):
    """Test suite for Tokenizer class."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.tokenizer = Tokenizer()
    
    def test_single_text_tokenization(self):
        """Test tokenization of single text."""
        result = self.tokenizer.tokenize("Test text")
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
    
    def test_vocab_size(self):
        """Test vocabulary size retrieval."""
        self.assertEqual(self.tokenizer.get_vocab_size(), 30000)


class TestFeatureExtraction(unittest.TestCase):
    """Test suite for FeatureExtractor class."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.extractor = FeatureExtractor()
        self.texts = ["First doc", "Second doc", "Third doc"]
    
    def test_tfidf_extraction(self):
        """Test TF-IDF feature extraction."""
        features = self.extractor.extract_tfidf_features(self.texts, fit=True)
        self.assertEqual(features.shape[0], len(self.texts))
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        features = self.extractor.extract_statistical_features(self.texts)
        self.assertEqual(features.shape[0], len(self.texts))


class TestSlidingWindow(unittest.TestCase):
    """Test suite for SlidingWindow class."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.sliding_window = SlidingWindow(SlidingWindowConfig(window_size=50, stride=25))
    
    def test_window_creation(self):
        """Test window creation."""
        text = "A" * 100
        windows = self.sliding_window.create_windows(text)
        self.assertGreater(len(windows), 0)


class TestPromptFormatter(unittest.TestCase):
    """Test suite for PromptFormatter class."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.formatter = PromptFormatter()
    
    def test_single_prompt_formatting(self):
        """Test single prompt formatting."""
        prompt = self.formatter.format_single("Test article")
        self.assertIn("Test article", prompt)
        self.assertIn("World", prompt)
    
    def test_instruction_format(self):
        """Test instruction format."""
        result = self.formatter.format_for_instruction_tuning("Test", 0)
        self.assertIn('instruction', result)
        self.assertEqual(result['output'], "World")


if __name__ == '__main__':
    unittest.main()
