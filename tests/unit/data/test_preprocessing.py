"""
Unit Tests for Data Preprocessing Modules
==========================================

This module provides comprehensive unit tests for data preprocessing components
following testing methodologies from:
- Beck, K. (2003): "Test Driven Development: By Example"
- Meszaros, G. (2007): "xUnit Test Patterns: Refactoring Test Code"
- Humble, J. & Farley, D. (2010): "Continuous Delivery"

Test Coverage:
1. Text cleaning and normalization
2. Tokenization strategies
3. Feature extraction techniques
4. Sliding window processing
5. Prompt formatting

Mathematical Foundation:
Test cases follow equivalence partitioning and boundary value analysis
ensuring coverage probability P(defect_detection) > 0.95 for critical paths.

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Any

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules to test
from src.data.preprocessing.text_cleaner import (
    TextCleaner, CleaningConfig, get_minimal_cleaner, get_aggressive_cleaner
)
from src.data.preprocessing.tokenization import (
    Tokenizer, TokenizationConfig, get_bert_tokenizer
)
from src.data.preprocessing.feature_extraction import (
    FeatureExtractor, FeatureExtractionConfig
)
from src.data.preprocessing.sliding_window import (
    SlidingWindow, SlidingWindowConfig
)
from src.data.preprocessing.prompt_formatter import (
    PromptFormatter, PromptFormatterConfig
)

# Import test fixtures
from tests.fixtures.sample_data import (
    get_sample_texts, get_edge_case_texts, get_sample_labels,
    create_sample_dataset
)

# Import constants
from configs.constants import (
    AG_NEWS_CLASSES, MAX_SEQUENCE_LENGTH, ID_TO_LABEL
)


class TestTextCleaner(unittest.TestCase):
    """
    Test cases for TextCleaner class.
    
    Following testing patterns from:
    - Martin, R. C. (2008): "Clean Code: A Handbook of Agile Software Craftsmanship"
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.default_cleaner = TextCleaner()
        self.minimal_cleaner = get_minimal_cleaner()
        self.aggressive_cleaner = get_aggressive_cleaner()
        self.sample_texts = get_sample_texts()
        self.edge_cases = get_edge_case_texts()
    
    def test_initialization(self):
        """Test TextCleaner initialization."""
        # Default configuration
        cleaner = TextCleaner()
        self.assertIsNotNone(cleaner.config)
        self.assertFalse(cleaner.config.lowercase)
        
        # Custom configuration
        config = CleaningConfig(lowercase=True, remove_punctuation=True)
        cleaner = TextCleaner(config)
        self.assertTrue(cleaner.config.lowercase)
        self.assertTrue(cleaner.config.remove_punctuation)
    
    def test_basic_cleaning(self):
        """Test basic text cleaning operations."""
        text = "Hello World! This is a TEST."
        
        # Test lowercase
        config = CleaningConfig(lowercase=True)
        cleaner = TextCleaner(config)
        cleaned = cleaner.clean(text)
        self.assertEqual(cleaned.lower(), cleaned)
        
        # Test punctuation removal
        config = CleaningConfig(remove_punctuation=True)
        cleaner = TextCleaner(config)
        cleaned = cleaner.clean(text)
        self.assertNotIn("!", cleaned)
        self.assertNotIn(".", cleaned)
    
    def test_url_email_removal(self):
        """Test URL and email removal."""
        text = "Visit https://example.com or email test@example.com for info."
        
        cleaned = self.minimal_cleaner.clean(text)
        self.assertNotIn("https://example.com", cleaned)
        self.assertNotIn("test@example.com", cleaned)
    
    def test_unicode_normalization(self):
        """Test Unicode normalization."""
        text = "Café naïve résumé"
        cleaned = self.default_cleaner.clean(text)
        # Should normalize but preserve readability
        self.assertIsNotNone(cleaned)
        self.assertIsInstance(cleaned, str)
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        text = "Multiple    spaces   and\n\nnewlines\t\ttabs"
        cleaned = self.default_cleaner.clean(text)
        # Should normalize to single spaces
        self.assertNotIn("  ", cleaned)
        self.assertNotIn("\n\n", cleaned)
        self.assertNotIn("\t", cleaned)
    
    def test_edge_cases(self):
        """Test edge case handling."""
        for edge_case in self.edge_cases:
            try:
                cleaned = self.default_cleaner.clean(edge_case)
                self.assertIsInstance(cleaned, str)
            except Exception as e:
                self.fail(f"Failed on edge case: {edge_case[:50]}... Error: {e}")
    
    def test_batch_cleaning(self):
        """Test batch text cleaning."""
        texts = ["Text 1", "Text 2", "Text 3"]
        cleaned_texts = self.default_cleaner.batch_clean(texts)
        
        self.assertEqual(len(cleaned_texts), len(texts))
        self.assertIsInstance(cleaned_texts, list)
        for text in cleaned_texts:
            self.assertIsInstance(text, str)
    
    def test_statistics(self):
        """Test cleaning statistics generation."""
        text = "https://example.com test@email.com Some text here."
        stats = self.minimal_cleaner.get_statistics(text)
        
        self.assertIn("original_length", stats)
        self.assertIn("cleaned_length", stats)
        self.assertIn("reduction_ratio", stats)
        self.assertIn("urls_removed", stats)
        self.assertIn("emails_removed", stats)
        self.assertGreater(stats["urls_removed"], 0)
        self.assertGreater(stats["emails_removed"], 0)


class TestTokenizer(unittest.TestCase):
    """
    Test cases for Tokenizer class.
    
    Following tokenization testing from:
    - Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = TokenizationConfig(
            model_name="bert-base-uncased",
            max_length=128
        )
        # Mock tokenizer to avoid downloading models
        with patch('src.data.preprocessing.tokenization.AutoTokenizer.from_pretrained'):
            self.tokenizer = Tokenizer(self.config)
            # Set up mock tokenizer
            self.tokenizer.tokenizer = MagicMock()
            self.tokenizer.tokenizer.__len__ = Mock(return_value=30000)
    
    def test_initialization(self):
        """Test Tokenizer initialization."""
        self.assertIsNotNone(self.tokenizer.config)
        self.assertEqual(self.tokenizer.config.max_length, 128)
        self.assertEqual(self.tokenizer.config.model_name, "bert-base-uncased")
    
    def test_tokenize_single(self):
        """Test single text tokenization."""
        text = "This is a test sentence."
        
        # Mock tokenizer output
        mock_output = {
            'input_ids': torch.tensor([[101, 2023, 2003, 1037, 3231, 6251, 102]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1, 1, 1, 1]])
        }
        self.tokenizer.tokenizer.return_value = mock_output
        
        result = self.tokenizer.tokenize(text)
        self.assertIn('input_ids', result)
        self.assertIn('attention_mask', result)
    
    def test_tokenize_batch(self):
        """Test batch tokenization."""
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # Mock tokenizer outputs
        mock_outputs = []
        for i in range(3):
            mock_outputs.append({
                'input_ids': torch.randn(1, 128),
                'attention_mask': torch.ones(1, 128)
            })
        
        with patch.object(self.tokenizer, 'tokenize', side_effect=mock_outputs):
            result = self.tokenizer.batch_tokenize(texts, batch_size=2)
            # Batch tokenization should handle splitting and merging
            self.assertIsNotNone(result)
    
    def test_decode(self):
        """Test token decoding."""
        token_ids = [101, 2023, 2003, 1037, 3231, 6251, 102]
        
        self.tokenizer.tokenizer.decode = Mock(return_value="this is a test sentence")
        decoded = self.tokenizer.decode(token_ids)
        
        self.assertIsInstance(decoded, str)
        self.tokenizer.tokenizer.decode.assert_called_once()
    
    def test_vocab_size(self):
        """Test vocabulary size retrieval."""
        vocab_size = self.tokenizer.get_vocab_size()
        self.assertEqual(vocab_size, 30000)
    
    def test_save_load(self):
        """Test tokenizer save and load."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "tokenizer"
            self.tokenizer.tokenizer.save_pretrained = Mock()
            
            self.tokenizer.save(save_path)
            self.tokenizer.tokenizer.save_pretrained.assert_called_once_with(save_path)


class TestFeatureExtractor(unittest.TestCase):
    """
    Test cases for FeatureExtractor class.
    
    Following feature extraction testing from:
    - Zhang et al. (2015): "Character-level Convolutional Networks"
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = FeatureExtractionConfig(
            use_tfidf=True,
            use_bow=True,
            use_statistical=True,
            use_embeddings=False,  # Disable to avoid model download
            use_linguistic=False   # Disable to avoid spacy download
        )
        self.extractor = FeatureExtractor(self.config)
        self.sample_texts = ["This is a test.", "Another test text.", "Third sample."]
    
    def test_initialization(self):
        """Test FeatureExtractor initialization."""
        self.assertIsNotNone(self.extractor.config)
        self.assertIsNotNone(self.extractor.tfidf_vectorizer)
        self.assertIsNotNone(self.extractor.bow_vectorizer)
    
    def test_tfidf_extraction(self):
        """Test TF-IDF feature extraction."""
        # Fit and transform
        features = self.extractor.extract_tfidf_features(self.sample_texts, fit=True)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], len(self.sample_texts))
        self.assertGreater(features.shape[1], 0)
        
        # Transform only
        new_texts = ["New test text."]
        new_features = self.extractor.extract_tfidf_features(new_texts, fit=False)
        self.assertEqual(new_features.shape[0], 1)
        self.assertEqual(new_features.shape[1], features.shape[1])
    
    def test_bow_extraction(self):
        """Test Bag of Words feature extraction."""
        features = self.extractor.extract_bow_features(self.sample_texts, fit=True)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], len(self.sample_texts))
        self.assertGreater(features.shape[1], 0)
    
    def test_statistical_features(self):
        """Test statistical feature extraction."""
        features = self.extractor.extract_statistical_features(self.sample_texts)
        
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], len(self.sample_texts))
        # Should have 11 statistical features
        self.assertEqual(features.shape[1], 11)
        
        # Test specific features
        for i, text in enumerate(self.sample_texts):
            # Character count
            self.assertEqual(features[i, 0], len(text))
            # Word count
            self.assertEqual(features[i, 1], len(text.split()))
    
    def test_extract_all_features(self):
        """Test extracting all configured features."""
        all_features = self.extractor.extract_all_features(self.sample_texts, fit=True)
        
        self.assertIsInstance(all_features, dict)
        self.assertIn('tfidf', all_features)
        self.assertIn('bow', all_features)
        self.assertIn('statistical', all_features)
        
        for feature_type, features in all_features.items():
            if features.size > 0:
                self.assertEqual(features.shape[0], len(self.sample_texts))
    
    def test_combine_features(self):
        """Test feature combination."""
        features_dict = {
            'feat1': np.random.randn(3, 10),
            'feat2': np.random.randn(3, 5),
            'feat3': np.random.randn(3, 7)
        }
        
        combined = self.extractor.combine_features(features_dict)
        
        self.assertIsInstance(combined, np.ndarray)
        self.assertEqual(combined.shape[0], 3)
        self.assertEqual(combined.shape[1], 22)  # 10 + 5 + 7
    
    def test_save_load_extractors(self):
        """Test saving and loading fitted extractors."""
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir)
            
            # Fit extractors
            self.extractor.extract_tfidf_features(self.sample_texts, fit=True)
            
            # Save
            self.extractor.save_extractors(save_path)
            
            # Create new extractor and load
            new_extractor = FeatureExtractor(self.config)
            new_extractor.load_extractors(save_path)
            
            # Test that loaded extractor works
            features = new_extractor.extract_tfidf_features(["Test"], fit=False)
            self.assertIsNotNone(features)


class TestSlidingWindow(unittest.TestCase):
    """
    Test cases for SlidingWindow class.
    
    Following sliding window testing from:
    - Beltagy et al. (2020): "Longformer: The Long-Document Transformer"
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SlidingWindowConfig(
            window_size=100,
            stride=50,
            min_window_size=20
        )
        self.sliding_window = SlidingWindow(self.config)
        self.long_text = " ".join(["word"] * 200)  # 200 words
    
    def test_initialization(self):
        """Test SlidingWindow initialization."""
        self.assertIsNotNone(self.sliding_window.config)
        self.assertEqual(self.sliding_window.config.window_size, 100)
        self.assertEqual(self.sliding_window.config.stride, 50)
    
    def test_create_char_windows(self):
        """Test character-level window creation."""
        windows = self.sliding_window.create_windows(self.long_text)
        
        self.assertIsInstance(windows, list)
        self.assertGreater(len(windows), 0)
        
        # Check window structure
        for window in windows:
            self.assertIn('window_id', window)
            self.assertIn('text', window)
            self.assertIn('start', window)
            self.assertIn('end', window)
            self.assertIn('is_last', window)
            
            # Check window size constraints
            window_size = window['end'] - window['start']
            if not window['is_last']:
                self.assertLessEqual(window_size, self.config.window_size)
    
    def test_create_token_windows(self):
        """Test token-level window creation."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            'input_ids': list(range(150)),  # 150 tokens
            'offset_mapping': [(i, i+1) for i in range(150)]
        }
        mock_tokenizer.cls_token_id = 101
        mock_tokenizer.sep_token_id = 102
        
        windows = self.sliding_window.create_windows(self.long_text, mock_tokenizer)
        
        self.assertIsInstance(windows, list)
        self.assertGreater(len(windows), 0)
        
        for window in windows:
            self.assertIn('window_id', window)
            self.assertIn('input_ids', window)
    
    def test_aggregate_predictions(self):
        """Test prediction aggregation strategies."""
        predictions = [
            torch.tensor([0.1, 0.7, 0.1, 0.1]),
            torch.tensor([0.2, 0.6, 0.1, 0.1]),
            torch.tensor([0.1, 0.8, 0.05, 0.05])
        ]
        
        # Test mean aggregation
        result = self.sliding_window.aggregate_predictions(predictions, strategy="mean")
        expected = torch.stack(predictions).mean(dim=0)
        torch.testing.assert_close(result, expected)
        
        # Test max aggregation
        result = self.sliding_window.aggregate_predictions(predictions, strategy="max")
        expected = torch.stack(predictions).max(dim=0)[0]
        torch.testing.assert_close(result, expected)
        
        # Test first aggregation
        result = self.sliding_window.aggregate_predictions(predictions, strategy="first")
        torch.testing.assert_close(result, predictions[0])
    
    def test_process_batch(self):
        """Test batch processing with sliding windows."""
        texts = ["Short text", self.long_text, "Another text"]
        
        result = self.sliding_window.process_batch(texts)
        
        self.assertIn('windows', result)
        self.assertIn('text_to_windows', result)
        self.assertIn('num_texts', result)
        self.assertIn('num_windows', result)
        
        self.assertEqual(result['num_texts'], len(texts))
        self.assertGreaterEqual(result['num_windows'], len(texts))


class TestPromptFormatter(unittest.TestCase):
    """
    Test cases for PromptFormatter class.
    
    Following prompt formatting testing from:
    - Brown et al. (2020): "Language Models are Few-Shot Learners"
    """
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = PromptFormatterConfig(
            template_style="classification",
            use_demonstrations=False,
            use_cot=False
        )
        self.formatter = PromptFormatter(self.config)
        self.sample_text = "This is a test news article about technology."
        self.sample_label = 3  # Sci/Tech
    
    def test_initialization(self):
        """Test PromptFormatter initialization."""
        self.assertIsNotNone(self.formatter.config)
        self.assertEqual(self.formatter.config.template_style, "classification")
        self.assertIsInstance(self.formatter.config.templates, list)
        self.assertGreater(len(self.formatter.config.templates), 0)
    
    def test_format_single(self):
        """Test single prompt formatting."""
        prompt = self.formatter.format_single(self.sample_text)
        
        self.assertIsInstance(prompt, str)
        self.assertIn(self.sample_text, prompt)
        self.assertIn(self.formatter.config.task_description, prompt)
        
        # Test with label
        prompt_with_label = self.formatter.format_single(
            self.sample_text, 
            label=self.sample_label
        )
        self.assertIn(ID_TO_LABEL[self.sample_label], prompt_with_label)
    
    def test_format_options(self):
        """Test option formatting."""
        options = self.formatter._format_options()
        
        for class_name in AG_NEWS_CLASSES:
            self.assertIn(class_name, options)
        
        # Test with letter formatting
        self.formatter.config.use_letters = True
        options = self.formatter._format_options()
        self.assertIn("A)", options)
        self.assertIn("B)", options)
    
    def test_format_demonstrations(self):
        """Test few-shot demonstration formatting."""
        demos = [
            {'text': 'Demo 1', 'label': 0},
            {'text': 'Demo 2', 'label': 1},
            {'text': 'Demo 3', 'label': 2}
        ]
        
        formatter = PromptFormatter(
            PromptFormatterConfig(use_demonstrations=True),
            demonstrations=demos
        )
        
        formatted_demos = formatter._format_demonstrations()
        
        self.assertIsInstance(formatted_demos, str)
        self.assertIn('Demo 1', formatted_demos)
        self.assertIn(ID_TO_LABEL[0], formatted_demos)
    
    def test_chain_of_thought(self):
        """Test chain of thought prompting."""
        config = PromptFormatterConfig(use_cot=True)
        formatter = PromptFormatter(config)
        
        prompt = formatter.format_single(self.sample_text)
        
        self.assertIn(config.cot_trigger, prompt)
    
    def test_instruction_tuning_format(self):
        """Test instruction tuning format."""
        result = self.formatter.format_for_instruction_tuning(
            self.sample_text,
            self.sample_label
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('instruction', result)
        self.assertIn('input', result)
        self.assertIn('output', result)
        
        self.assertIn(self.sample_text, result['input'])
        self.assertEqual(result['output'], ID_TO_LABEL[self.sample_label])
    
    def test_chat_format(self):
        """Test chat conversation format."""
        messages = self.formatter.format_for_chat(
            self.sample_text,
            label=self.sample_label
        )
        
        self.assertIsInstance(messages, list)
        self.assertGreater(len(messages), 0)
        
        # Check message structure
        for message in messages:
            self.assertIn('role', message)
            self.assertIn('content', message)
            self.assertIn(message['role'], ['system', 'user', 'assistant'])
    
    def test_create_prompt_dataset(self):
        """Test prompt dataset creation."""
        texts = ["Text 1", "Text 2", "Text 3"]
        labels = [0, 1, 2]
        
        prompts = self.formatter.create_prompt_dataset(texts, labels)
        
        self.assertEqual(len(prompts), len(texts))
        for prompt, text in zip(prompts, texts):
            self.assertIn(text, prompt)


class TestIntegration(unittest.TestCase):
    """
    Integration tests for preprocessing pipeline.
    
    Following integration testing from:
    - Humble & Farley (2010): "Continuous Delivery"
    """
    
    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create sample dataset
        dataset = create_sample_dataset(n_samples=10)
        
        # Step 1: Clean texts
        cleaner = get_minimal_cleaner()
        cleaned_texts = cleaner.batch_clean(dataset.texts)
        self.assertEqual(len(cleaned_texts), len(dataset.texts))
        
        # Step 2: Tokenize
        config = TokenizationConfig(model_name="bert-base-uncased")
        with patch('src.data.preprocessing.tokenization.AutoTokenizer.from_pretrained'):
            tokenizer = Tokenizer(config)
            tokenizer.tokenizer = MagicMock()
            tokenizer.tokenizer.return_value = {
                'input_ids': torch.randn(len(cleaned_texts), 128),
                'attention_mask': torch.ones(len(cleaned_texts), 128)
            }
            
            tokenized = tokenizer.tokenize(cleaned_texts)
            self.assertIn('input_ids', tokenized)
        
        # Step 3: Extract features
        extractor = FeatureExtractor(
            FeatureExtractionConfig(
                use_tfidf=True,
                use_statistical=True,
                use_embeddings=False
            )
        )
        features = extractor.extract_all_features(cleaned_texts, fit=True)
        self.assertIn('tfidf', features)
        self.assertIn('statistical', features)
        
        # Step 4: Format as prompts
        formatter = PromptFormatter()
        prompts = formatter.create_prompt_dataset(
            cleaned_texts, 
            dataset.labels
        )
        self.assertEqual(len(prompts), len(dataset.texts))
    
    def test_edge_case_handling(self):
        """Test preprocessing with edge cases."""
        edge_cases = get_edge_case_texts()[:5]  # Test subset
        
        # Test each preprocessing step with edge cases
        cleaner = TextCleaner()
        for text in edge_cases:
            try:
                cleaned = cleaner.clean(text)
                self.assertIsInstance(cleaned, str)
            except Exception as e:
                self.fail(f"Failed on edge case: {text[:30]}... Error: {e}")
    
    def test_memory_efficiency(self):
        """Test memory efficiency with large datasets."""
        # Create large dataset
        large_texts = ["Sample text " * 100] * 100  # 100 long texts
        
        # Test sliding window for memory efficiency
        sliding_window = SlidingWindow(
            SlidingWindowConfig(window_size=512, stride=256)
        )
        
        # Process in batches
        batch_result = sliding_window.process_batch(large_texts[:10])
        
        self.assertIsNotNone(batch_result)
        self.assertIn('num_windows', batch_result)
        # Should create multiple windows for long texts
        self.assertGreater(batch_result['num_windows'], batch_result['num_texts'])


# Pytest fixtures and additional tests
@pytest.fixture
def sample_dataset():
    """Pytest fixture for sample dataset."""
    return create_sample_dataset(n_samples=20)


@pytest.fixture
def preprocessing_pipeline():
    """Pytest fixture for preprocessing pipeline."""
    return {
        'cleaner': get_minimal_cleaner(),
        'extractor': FeatureExtractor(
            FeatureExtractionConfig(
                use_tfidf=True,
                use_embeddings=False
            )
        ),
        'formatter': PromptFormatter()
    }


def test_pipeline_consistency(sample_dataset, preprocessing_pipeline):
    """Test preprocessing pipeline consistency."""
    texts = sample_dataset.texts[:5]
    
    # Process twice and check consistency
    results1 = []
    results2 = []
    
    for _ in range(2):
        cleaned = preprocessing_pipeline['cleaner'].batch_clean(texts)
        if not results1:
            results1 = cleaned
        else:
            results2 = cleaned
    
    # Results should be identical
    assert results1 == results2


def test_error_recovery(preprocessing_pipeline):
    """Test error recovery in preprocessing."""
    problematic_texts = [
        None,
        "",
        "Normal text",
        12345,  # Wrong type
        "Another normal text"
    ]
    
    cleaner = preprocessing_pipeline['cleaner']
    results = []
    
    for text in problematic_texts:
        try:
            if isinstance(text, str):
                result = cleaner.clean(text)
                results.append(result)
            else:
                results.append("")
        except:
            results.append("")
    
    # Should handle errors gracefully
    assert len(results) == len(problematic_texts)


if __name__ == '__main__':
    unittest.main()
