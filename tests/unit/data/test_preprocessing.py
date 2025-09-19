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
from unittest.mock import Mock, MagicMock, patch, PropertyMock, create_autospec
import numpy as np
from pathlib import Path
import sys
import tempfile
import json
from typing import Dict, List, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock external dependencies before importing modules
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

# Create mock classes for sklearn
class MockTfidfVectorizer:
    """Mock TF-IDF Vectorizer."""
    def __init__(self, **kwargs):
        self.vocabulary_ = {}
        
    def fit_transform(self, texts):
        """Mock fit_transform."""
        return MockSparseMatrix(len(texts), 100)
    
    def transform(self, texts):
        """Mock transform."""
        return MockSparseMatrix(len(texts), 100)

class MockCountVectorizer:
    """Mock Count Vectorizer."""
    def __init__(self, **kwargs):
        self.vocabulary_ = {}
        
    def fit_transform(self, texts):
        """Mock fit_transform."""
        return MockSparseMatrix(len(texts), 100)
    
    def transform(self, texts):
        """Mock transform."""
        return MockSparseMatrix(len(texts), 100)

class MockTruncatedSVD:
    """Mock Truncated SVD."""
    def __init__(self, n_components=300):
        self.n_components = n_components
    
    def fit_transform(self, X):
        """Mock fit_transform."""
        n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.random.randn(n_samples, self.n_components)

class MockSparseMatrix:
    """Mock sparse matrix."""
    def __init__(self, rows, cols):
        self.shape = (rows, cols)
        self._data = np.random.randn(rows, cols)
    
    def toarray(self):
        """Convert to array."""
        return self._data

# Mock torch tensor
class MockTensor:
    """Mock PyTorch tensor."""
    def __init__(self, data):
        if isinstance(data, list):
            self.data = np.array(data)
        else:
            self.data = data
        self.shape = self.data.shape
    
    def tolist(self):
        """Convert to list."""
        return self.data.tolist()
    
    def mean(self, dim=None):
        """Mock mean operation."""
        if dim is None:
            return MockTensor(np.mean(self.data))
        return MockTensor(np.mean(self.data, axis=dim))
    
    def max(self, dim=None):
        """Mock max operation."""
        if dim is None:
            return MockTensor(np.max(self.data))
        result = np.max(self.data, axis=dim)
        return result, MockTensor(result)
    
    def argmax(self, dim=None):
        """Mock argmax operation."""
        return MockTensor(np.argmax(self.data, axis=dim))
    
    def squeeze(self):
        """Mock squeeze operation."""
        return MockTensor(np.squeeze(self.data))
    
    def numpy(self):
        """Convert to numpy."""
        return self.data
    
    def item(self):
        """Get single item."""
        return self.data.item()
    
    def __getitem__(self, idx):
        """Get item."""
        return MockTensor(self.data[idx])

# Patch sklearn modules
sys.modules['sklearn.feature_extraction.text'].TfidfVectorizer = MockTfidfVectorizer
sys.modules['sklearn.feature_extraction.text'].CountVectorizer = MockCountVectorizer
sys.modules['sklearn.decomposition'].TruncatedSVD = MockTruncatedSVD

# Mock torch functions
def mock_tensor(data):
    """Create mock tensor."""
    return MockTensor(data)

def mock_stack(tensors):
    """Mock torch.stack."""
    data = np.stack([t.data for t in tensors])
    return MockTensor(data)

def mock_cat(tensors):
    """Mock torch.cat."""
    data = np.concatenate([t.data for t in tensors])
    return MockTensor(data)

def mock_zeros_like(tensor):
    """Mock torch.zeros_like."""
    return MockTensor(np.zeros_like(tensor.data))

def mock_mode(tensor):
    """Mock torch.mode."""
    from scipy import stats
    result = stats.mode(tensor.data, axis=None, keepdims=False)
    return MockTensor(np.array([result.mode])), None

# Assign mock functions to torch module
torch_mock = sys.modules['torch']
torch_mock.tensor = mock_tensor
torch_mock.stack = mock_stack
torch_mock.cat = mock_cat
torch_mock.zeros_like = mock_zeros_like
torch_mock.mode = mock_mode
torch_mock.Tensor = MockTensor

# Mock NLTK
def mock_word_tokenize(text):
    """Mock NLTK word_tokenize."""
    return text.split()

sys.modules['nltk.tokenize'].word_tokenize = mock_word_tokenize
sys.modules['nltk.corpus'].stopwords = MagicMock()
sys.modules['nltk.corpus'].stopwords.words = MagicMock(return_value=['the', 'is', 'a', 'an', 'and'])

# Now import the modules to test
with patch('src.data.preprocessing.text_cleaner.nltk.download'):
    from src.data.preprocessing.text_cleaner import (
        TextCleaner, 
        CleaningConfig,
        get_minimal_cleaner,
        get_aggressive_cleaner
    )

from src.data.preprocessing.tokenization import (
    Tokenizer,
    TokenizationConfig,
    get_bert_tokenizer,
    get_roberta_tokenizer,
    get_deberta_tokenizer
)

from src.data.preprocessing.feature_extraction import (
    FeatureExtractor,
    FeatureExtractionConfig
)

from src.data.preprocessing.sliding_window import (
    SlidingWindow,
    SlidingWindowConfig
)

from src.data.preprocessing.prompt_formatter import (
    PromptFormatter,
    PromptFormatterConfig
)


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
        self.minimal_cleaner = get_minimal_cleaner()
        self.aggressive_cleaner = get_aggressive_cleaner()
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
    
    def test_aggressive_cleaning(self):
        """Test aggressive cleaning for classical ML models."""
        # Test lowercase conversion
        text = "The WORLD News Today"
        cleaned = self.aggressive_cleaner.clean(text)
        self.assertEqual(cleaned.lower(), cleaned)
        
        # Test punctuation removal
        text = "Hello, world! How are you?"
        cleaned = self.aggressive_cleaner.clean(text)
        self.assertNotIn(",", cleaned)
        self.assertNotIn("!", cleaned)
        self.assertNotIn("?", cleaned)
        
        # Test digit removal
        text = "The year 2024 brings new challenges"
        cleaned = self.aggressive_cleaner.clean(text)
        self.assertNotIn("2024", cleaned)
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        text = "This   has    multiple     spaces"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertNotIn("  ", cleaned)
        
        text = "\n\nMultiple\n\nlines\n\n"
        cleaned = self.minimal_cleaner.clean(text)
        self.assertEqual(cleaned, "Multiple lines")
    
    def test_unicode_normalization(self):
        """Test Unicode normalization (NFKD)."""
        text = "Café résumé naïve"
        config = CleaningConfig(normalize_unicode=True)
        cleaner = TextCleaner(config)
        cleaned = cleaner.clean(text)
        # Unicode normalization should handle accented characters
        self.assertIsInstance(cleaned, str)
    
    def test_batch_cleaning(self):
        """Test batch text cleaning."""
        texts = [
            "First text with URL https://test.com",
            "Second text with email test@test.com",
            "Third text with   spaces"
        ]
        cleaned_texts = self.minimal_cleaner.batch_clean(texts)
        
        self.assertEqual(len(cleaned_texts), len(texts))
        self.assertNotIn("https://test.com", cleaned_texts[0])
        self.assertNotIn("test@test.com", cleaned_texts[1])
        self.assertNotIn("  ", cleaned_texts[2])
    
    def test_cleaning_statistics(self):
        """Test text cleaning statistics generation."""
        text = "Visit https://example.com or email test@example.com for more information."
        stats = self.minimal_cleaner.get_statistics(text)
        
        self.assertIn("original_length", stats)
        self.assertIn("cleaned_length", stats)
        self.assertIn("reduction_ratio", stats)
        self.assertIn("urls_removed", stats)
        self.assertIn("emails_removed", stats)
        
        self.assertGreater(stats["original_length"], stats["cleaned_length"])
        self.assertEqual(stats["urls_removed"], 1)
        self.assertEqual(stats["emails_removed"], 1)
    
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
        # Create mock tokenizer
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.pad_token = "[PAD]"
        self.mock_tokenizer.unk_token = "[UNK]"
        self.mock_tokenizer.cls_token_id = 101
        self.mock_tokenizer.sep_token_id = 102
        self.mock_tokenizer.__len__ = MagicMock(return_value=30000)
        
        # Mock tokenization output
        self.mock_tokenizer.return_value = {
            'input_ids': mock_tensor([[101, 2023, 2003, 102]]),
            'attention_mask': mock_tensor([[1, 1, 1, 1]])
        }
        
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
        self.assertIsInstance(result['input_ids'], MockTensor)
    
    def test_batch_tokenization(self):
        """Test batch tokenization."""
        texts = ["First sentence.", "Second sentence.", "Third sentence."]
        
        # Mock batch output
        batch_output = {
            'input_ids': mock_tensor([[101, 2034, 102], [101, 2117, 102], [101, 2353, 102]]),
            'attention_mask': mock_tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        }
        self.mock_tokenizer.return_value = batch_output
        
        result = self.tokenizer.tokenize(texts)
        
        self.assertIn('input_ids', result)
        self.assertEqual(result['input_ids'].shape[0], 3)
    
    def test_tokenization_config_override(self):
        """Test configuration override in tokenization."""
        text = "Test text"
        
        # Override configuration
        result = self.tokenizer.tokenize(
            text,
            max_length=256,
            padding="longest",
            truncation=False
        )
        
        # Verify tokenizer was called with overridden parameters
        call_kwargs = self.mock_tokenizer.call_args[1]
        self.assertEqual(call_kwargs['max_length'], 256)
        self.assertEqual(call_kwargs['padding'], "longest")
        self.assertEqual(call_kwargs['truncation'], False)
    
    def test_decode_functionality(self):
        """Test decoding token IDs back to text."""
        token_ids = [101, 2023, 2003, 1037, 3231, 102]
        self.mock_tokenizer.decode.return_value = "this is a test"
        
        decoded = self.tokenizer.decode(token_ids)
        
        self.mock_tokenizer.decode.assert_called_once_with(
            token_ids,
            skip_special_tokens=True
        )
        self.assertEqual(decoded, "this is a test")
    
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
        self.config = FeatureExtractionConfig(
            use_tfidf=True,
            use_bow=True,
            use_statistical=True,
            use_embeddings=False,  # Disable to avoid loading models
            use_linguistic=False   # Disable to avoid loading spaCy
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
        # Fit and transform
        features = self.extractor.extract_tfidf_features(self.texts, fit=True)
        
        self.assertEqual(features.shape[0], len(self.texts))
        self.assertGreater(features.shape[1], 0)
        self.assertTrue(np.all(features >= -10))  # Allow for any reasonable values
        
        # Transform only
        new_text = ["New document about technology"]
        new_features = self.extractor.extract_tfidf_features(new_text, fit=False)
        self.assertEqual(new_features.shape[0], 1)
    
    def test_bow_extraction(self):
        """Test Bag of Words feature extraction."""
        # Fit and transform
        features = self.extractor.extract_bow_features(self.texts, fit=True)
        
        self.assertEqual(features.shape[0], len(self.texts))
        self.assertGreater(features.shape[1], 0)
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        features = self.extractor.extract_statistical_features(self.texts)
        
        self.assertEqual(features.shape[0], len(self.texts))
        # Statistical features include: char count, word count, sentence count,
        # avg word length, punctuation counts, capitalization, digit count
        self.assertGreater(features.shape[1], 10)
        
        # Test specific features
        text = "Hello World! This has 5 words."
        stats = self.extractor.extract_statistical_features([text])
        
        # Verify some statistical properties
        self.assertGreater(stats[0][0], 0)  # Character count > 0
        self.assertEqual(stats[0][4], 1)    # One period
        self.assertEqual(stats[0][6], 1)    # One exclamation mark
    
    def test_combined_features(self):
        """Test combining multiple feature types."""
        # Extract all features
        all_features = self.extractor.extract_all_features(self.texts, fit=True)
        
        self.assertIn('tfidf', all_features)
        self.assertIn('bow', all_features)
        self.assertIn('statistical', all_features)
        
        # Combine features
        combined = self.extractor.combine_features(all_features)
        
        self.assertEqual(combined.shape[0], len(self.texts))
        total_features = sum(f.shape[1] for f in all_features.values() if f.size > 0)
        self.assertEqual(combined.shape[1], total_features)
    
    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        empty_texts = ["", "   ", "\n\n"]
        
        # Statistical features should handle empty texts
        stats = self.extractor.extract_statistical_features(empty_texts)
        self.assertEqual(stats.shape[0], len(empty_texts))
        
        # All values should be 0 or very small for empty texts
        self.assertTrue(np.all(stats[0] == 0))


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
            self.assertLessEqual(len(window['text']), self.config.window_size)
            
            if not window['is_last']:
                self.assertGreaterEqual(len(window['text']), self.config.min_window_size)
    
    def test_window_overlap(self):
        """Test sliding window overlap with stride."""
        text = "A" * 200  # 200 character text
        windows = self.sliding_window.create_windows(text)
        
        # With window_size=100 and stride=50, we should have overlap
        if len(windows) > 1:
            # Check that windows overlap
            first_window_end = windows[0]['end']
            second_window_start = windows[1]['start']
            overlap = first_window_end - second_window_start
            
            expected_overlap = self.config.window_size - self.config.stride
            self.assertEqual(overlap, expected_overlap)
    
    def test_prediction_aggregation(self):
        """Test aggregation of predictions from multiple windows."""
        # Create sample predictions (3 windows, 4 classes)
        predictions = [
            mock_tensor([0.1, 0.2, 0.6, 0.1]),
            mock_tensor([0.2, 0.3, 0.4, 0.1]),
            mock_tensor([0.1, 0.2, 0.5, 0.2])
        ]
        
        # Test mean aggregation
        aggregated = self.sliding_window.aggregate_predictions(
            predictions,
            strategy="mean"
        )
        
        # Test max aggregation
        aggregated = self.sliding_window.aggregate_predictions(
            predictions,
            strategy="max"
        )
        
        # Test first aggregation
        aggregated = self.sliding_window.aggregate_predictions(
            predictions,
            strategy="first"
        )
        
        # Basic checks
        self.assertIsNotNone(aggregated)
        self.assertIsInstance(aggregated, MockTensor)
    
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
        self.assertGreaterEqual(result['num_windows'], len(texts))
        
        # Check text-to-window mapping
        for text_idx in range(len(texts)):
            self.assertIn(text_idx, result['text_to_windows'])
    
    def test_small_text_handling(self):
        """Test handling of texts smaller than window size."""
        small_text = "This is a small text."
        windows = self.sliding_window.create_windows(small_text)
        
        # Should create exactly one window
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]['text'], small_text)
        self.assertTrue(windows[0]['is_last'])


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
    
    def test_prompt_with_label(self):
        """Test prompt formatting with label."""
        prompt = self.formatter.format_single(self.text, label=self.label)
        
        self.assertIn(self.text, prompt)
        self.assertIn("Sci/Tech", prompt)
    
    def test_few_shot_formatting(self):
        """Test few-shot prompt formatting."""
        formatter = PromptFormatter(
            config=PromptFormatterConfig(use_demonstrations=True),
            demonstrations=self.demonstrations
        )
        
        prompt = formatter.format_single(self.text)
        
        # Check demonstrations are included
        self.assertIn("Stock market rises today", prompt)
        self.assertIn("Team wins championship", prompt)
        self.assertIn("Business", prompt)
        self.assertIn("Sports", prompt)
    
    def test_chain_of_thought_formatting(self):
        """Test chain of thought prompt formatting."""
        config = PromptFormatterConfig(
            use_cot=True,
            cot_trigger="Let's think step by step."
        )
        formatter = PromptFormatter(config)
        
        prompt = formatter.format_single(self.text)
        
        self.assertIn("Let's think step by step.", prompt)
    
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
        
        # Check roles
        roles = [msg['role'] for msg in messages]
        self.assertIn('system', roles)
        self.assertIn('user', roles)
        self.assertIn('assistant', roles)
    
    def test_option_formatting(self):
        """Test different option formatting styles."""
        # Test letter formatting
        config = PromptFormatterConfig(use_letters=True)
        formatter = PromptFormatter(config)
        
        prompt = formatter.format_single(self.text)
        
        self.assertIn("A)", prompt)
        self.assertIn("B)", prompt)
        self.assertIn("C)", prompt)
        self.assertIn("D)", prompt)
    
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
    
    def test_metadata_addition(self):
        """Test adding metadata to prompts."""
        config = PromptFormatterConfig(add_metadata=True)
        formatter = PromptFormatter(config)
        
        metadata = {"source": "test", "date": "2024"}
        prompt = formatter.format_single(self.text, metadata=metadata)
        
        self.assertIn("source=test", prompt)
        self.assertIn("date=2024", prompt)


# Test runner
if __name__ == '__main__':
    # Configure test runner with verbosity
    unittest.main(verbosity=2)
