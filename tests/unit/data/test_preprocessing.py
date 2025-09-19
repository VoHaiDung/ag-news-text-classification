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

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock ALL external dependencies BEFORE any imports
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.feature_extraction'] = MagicMock()
sys.modules['sklearn.feature_extraction.text'] = MagicMock()
sys.modules['sklearn.decomposition'] = MagicMock()
sys.modules['spacy'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['nltk.corpus'] = MagicMock()
sys.modules['nltk.tokenize'] = MagicMock()
sys.modules['joblib'] = MagicMock()

# Now import standard libraries
import unittest
import numpy as np
import tempfile
import json
from typing import Dict, List, Any


class TestTextCleaner(unittest.TestCase):
    """
    Test suite for TextCleaner class.
    
    Validates text cleaning functionality following:
    - Minimal cleaning for transformer models
    - Aggressive cleaning for classical ML
    - Unicode normalization standards
    """
    
    def setUp(self):
        """Initialize test fixtures."""
        # Mock NLTK functions
        with patch('nltk.download'):
            with patch('nltk.corpus.stopwords.words', return_value=['the', 'is', 'a', 'an', 'and']):
                with patch('nltk.tokenize.word_tokenize', side_effect=lambda x: x.split()):
                    from src.data.preprocessing.text_cleaner import (
                        TextCleaner, 
                        CleaningConfig,
                        get_minimal_cleaner,
                        get_aggressive_cleaner
                    )
                    self.TextCleaner = TextCleaner
                    self.CleaningConfig = CleaningConfig
                    self.get_minimal_cleaner = get_minimal_cleaner
                    self.get_aggressive_cleaner = get_aggressive_cleaner
                    
                    self.minimal_cleaner = self.get_minimal_cleaner()
                    self.aggressive_cleaner = self.get_aggressive_cleaner()
                    self.custom_config = CleaningConfig(
                        lowercase=True,
                        remove_urls=True,
                        remove_emails=True,
                        normalize_whitespace=True
                    )
                    self.custom_cleaner = TextCleaner(self.custom_config)
    
    def test_minimal_cleaning(self):
        """Test minimal cleaning preserves important information."""
        # Test URL removal
        text = "Check out https://example.com for more info"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertNotIn("https://example.com", cleaned)
        
        # Test email removal
        text = "Contact us at test@example.com for details"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertNotIn("test@example.com", cleaned)
        
        # Test preserves case
        text = "This is IMPORTANT News"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertIn("IMPORTANT", cleaned)
        
        # Test preserves punctuation
        text = "Breaking news! What happened?"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertIn("!", cleaned)
        self.assertIn("?", cleaned)
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        text = "This   has    multiple     spaces"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertNotIn("  ", cleaned)
        
        text = "\n\nMultiple\n\nlines\n\n"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertEqual(cleaned, "Multiple lines")
    
    def test_empty_text_handling(self):
        """Test handling of empty and None texts."""
        self.assertEqual(self.minimal_cleaner.clean(""), "")
        self.assertEqual(self.minimal_cleaner.clean(None), "")
        self.assertEqual(self.minimal_cleaner.clean("   "), "")


class TestTokenization(unittest.TestCase):
    """
    Test suite for Tokenizer class.
    
    Validates tokenization strategies following:
    - Subword tokenization (BPE, WordPiece)
    - Special token handling
    - Truncation and padding strategies
    """
    
    def setUp(self):
        """Initialize test fixtures with mocked tokenizer."""
        # Create mock tensor class
        class MockTensor:
            def __init__(self, data):
                self.data = np.array(data) if isinstance(data, list) else data
                self.shape = self.data.shape
            
            def tolist(self):
                return self.data.tolist()
        
        # Create mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token = "[PAD]"
        self.mock_tokenizer.unk_token = "[UNK]"
        self.mock_tokenizer.cls_token_id = 101
        self.mock_tokenizer.sep_token_id = 102
        self.mock_tokenizer.__len__ = MagicMock(return_value=30000)
        
        # Mock tokenization output
        self.mock_tokenizer.return_value = {
            'input_ids': MockTensor([[101, 2023, 2003, 102]]),
            'attention_mask': MockTensor([[1, 1, 1, 1]])
        }
        
        # Import with mocked dependencies
        with patch('transformers.AutoTokenizer'):
            from src.data.preprocessing.tokenization import (
                Tokenizer,
                TokenizationConfig
            )
            self.Tokenizer = Tokenizer
            self.TokenizationConfig = TokenizationConfig
            
            # Create tokenizer with mocked backend
            self.config = TokenizationConfig(
                model_name="bert-base-uncased",
                max_length=512,
                padding="max_length"
            )
            self.tokenizer = Tokenizer(config=self.config, tokenizer=self.mock_tokenizer)
    
    def test_single_text_tokenization(self):
        """Test tokenization of single text."""
        text = "This is a test sentence."
        result = self.tokenizer.tokenize(text)
        
        # Verify tokenizer was called
        self.mock_tokenizer.assert_called_once()
        
        # Check output structure
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
    
    def test_vocab_size(self):
        """Test vocabulary size retrieval."""
        vocab_size = self.tokenizer.get_vocab_size()
        self.assertEqual(vocab_size, 30000)


class TestFeatureExtraction(unittest.TestCase):
    """
    Test suite for FeatureExtractor class.
    
    Validates feature extraction methods:
    - TF-IDF features
    - Statistical features
    - Embedding features (mocked)
    """
    
    def setUp(self):
        """Initialize test fixtures."""
        # Create mock vectorizers
        class MockVectorizer:
            def __init__(self, **kwargs):
                self.vocabulary_ = {}
            
            def fit_transform(self, texts):
                return np.random.randn(len(texts), 100)
            
            def transform(self, texts):
                return np.random.randn(len(texts), 100)
        
        # Patch sklearn modules
        with patch('sklearn.feature_extraction.text.TfidfVectorizer', MockVectorizer):
            with patch('sklearn.feature_extraction.text.CountVectorizer', MockVectorizer):
                with patch('sklearn.decomposition.TruncatedSVD'):
                    with patch('transformers.AutoModel'):
                        with patch('transformers.AutoTokenizer'):
                            from src.data.preprocessing.feature_extraction import (
                                FeatureExtractor,
                                FeatureExtractionConfig
                            )
                            self.FeatureExtractor = FeatureExtractor
                            self.FeatureExtractionConfig = FeatureExtractionConfig
                            
                            self.config = FeatureExtractionConfig(
                                use_tfidf=True,
                                use_bow=True,
                                use_statistical=True,
                                use_embeddings=False,
                                use_linguistic=False
                            )
                            self.extractor = FeatureExtractor(self.config)
        
        # Sample texts for testing
        self.texts = [
            "This is the first document about technology.",
            "Second document discusses business and finance.",
            "Third document covers sports and entertainment."
        ]
    
    def test_tfidf_extraction(self):
        """Test TF-IDF feature extraction."""
        features = self.extractor.extract_tfidf_features(self.texts, fit=True)
        
        self.assertEqual(features.shape[0], len(self.texts))
        self.assertGreater(features.shape[1], 0)
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        features = self.extractor.extract_statistical_features(self.texts)
        
        self.assertEqual(features.shape[0], len(self.texts))
        self.assertGreater(features.shape[1], 10)
    
    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        empty_texts = ["", "   ", "\n\n"]
        
        stats = self.extractor.extract_statistical_features(empty_texts)
        self.assertEqual(stats.shape[0], len(empty_texts))


class TestSlidingWindow(unittest.TestCase):
    """
    Test suite for SlidingWindow class.
    
    Validates sliding window processing for long sequences:
    - Character-level windows
    - Token-level windows
    - Prediction aggregation
    """
    
    def setUp(self):
        """Initialize test fixtures."""
        from src.data.preprocessing.sliding_window import (
            SlidingWindow,
            SlidingWindowConfig
        )
        self.SlidingWindow = SlidingWindow
        self.SlidingWindowConfig = SlidingWindowConfig
        
        self.config = SlidingWindowConfig(
            window_size=100,
            stride=50,
            min_window_size=20
        )
        self.sliding_window = SlidingWindow(self.config)
        
        # Sample long text
        self.long_text = " ".join(["This is sentence number {}.".format(i) for i in range(50)])
    
    def test_character_window_creation(self):
        """Test character-level sliding window creation."""
        windows = self.sliding_window.create_windows(self.long_text)
        
        self.assertGreater(len(windows), 0)
        
        # Check window properties
        for i, window in enumerate(windows):
            self.assertIn('window_id', window)
            self.assertIn('text', window)
            self.assertIn('start', window)
            self.assertIn('end', window)
            self.assertIn('is_last', window)
            
            self.assertEqual(window['window_id'], i)
    
    def test_small_text_handling(self):
        """Test handling of texts smaller than window size."""
        small_text = "This is a small text."
        windows = self.sliding_window.create_windows(small_text)
        
        # Should create exactly one window
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]['text'], small_text)
        self.assertTrue(windows[0]['is_last'])
    
    def test_batch_processing(self):
        """Test batch processing of multiple texts."""
        texts = [
            "Short text",
            "Medium length text that is a bit longer than the first one",
            self.long_text
        ]
        
        result = self.sliding_window.process_batch(texts)
        
        self.assertIn('windows', result)
        self.assertIn('text_to_windows', result)
        self.assertIn('num_texts', result)
        self.assertIn('num_windows', result)
        
        self.assertEqual(result['num_texts'], len(texts))


class TestPromptFormatter(unittest.TestCase):
    """
    Test suite for PromptFormatter class.
    
    Validates prompt formatting strategies:
    - Zero-shot prompting
    - Few-shot prompting
    - Instruction formatting
    - Chain of thought
    """
    
    def setUp(self):
        """Initialize test fixtures."""
        from src.data.preprocessing.prompt_formatter import (
            PromptFormatter,
            PromptFormatterConfig
        )
        self.PromptFormatter = PromptFormatter
        self.PromptFormatterConfig = PromptFormatterConfig
        
        self.config = PromptFormatterConfig(
            template_style="classification",
            use_demonstrations=False,
            use_cot=False
        )
        self.formatter = PromptFormatter(self.config)
        
        # Sample data
        self.text = "Breaking news: Technology company announces new product."
        self.label = 3  # Sci/Tech
        
        # Sample demonstrations
        self.demonstrations = [
            {"text": "Stock market rises today", "label": 2},
            {"text": "Team wins championship", "label": 1}
        ]
    
    def test_single_prompt_formatting(self):
        """Test formatting single example as prompt."""
        prompt = self.formatter.format_single(self.text)
        
        # Check prompt contains required elements
        self.assertIn(self.text, prompt)
        self.assertIn("World", prompt)
        self.assertIn("Sports", prompt)
        self.assertIn("Business", prompt)
        self.assertIn("Sci/Tech", prompt)
    
    def test_instruction_tuning_format(self):
        """Test instruction tuning format."""
        result = self.formatter.format_for_instruction_tuning(
            self.text,
            self.label
        )
        
        self.assertIn('instruction', result)
        self.assertIn('input', result)
        self.assertIn('output', result)
        
        self.assertIn("Classify", result['instruction'])
        self.assertIn(self.text, result['input'])
        self.assertEqual(result['output'], "Sci/Tech")
    
    def test_chat_format(self):
        """Test chat conversation format."""
        messages = self.formatter.format_for_chat(self.text, self.label)
        
        self.assertIsInstance(messages, list)
        self.assertGreater(len(messages), 0)
        
        # Check message structure
        for message in messages:
            self.assertIn('role', message)
            self.assertIn('content', message)
    
    def test_prompt_dataset_creation(self):
        """Test creation of prompt dataset."""
        texts = [
            "Technology news article",
            "Sports news article",
            "Business news article"
        ]
        labels = [3, 1, 2]
        
        prompts = self.formatter.create_prompt_dataset(texts, labels)
        
        self.assertEqual(len(prompts), len(texts))
        
        for prompt, text in zip(prompts, texts):
            self.assertIn(text, prompt)


# Test runner
if __name__ == '__main__':
    # Configure test runner with verbosity
    unittest.main(verbosity=2)
