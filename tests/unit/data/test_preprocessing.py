"""
Unit Tests for Data Preprocessing Modules
==========================================

Comprehensive test suite for preprocessing components following:
- IEEE 829-2008: Standard for Software Test Documentation
- ISO/IEC/IEEE 29119: Software Testing Standards
- Academic Testing Best Practices

This module tests:
- Text cleaning and normalization
- Tokenization strategies
- Feature extraction methods
- Sliding window processing
- Prompt formatting

Author: Võ Hải Dũng
License: MIT
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import modules to test
from src.data.preprocessing.text_cleaner import TextCleaner, CleaningConfig
from src.data.preprocessing.tokenization import Tokenizer, TokenizationConfig
from src.data.preprocessing.feature_extraction import FeatureExtractor, FeatureExtractionConfig
from src.data.preprocessing.sliding_window import SlidingWindow, SlidingWindowConfig
from src.data.preprocessing.prompt_formatter import PromptFormatter, PromptFormatterConfig


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return [
        "Breaking: Stock market reaches all-time high amid economic recovery.",
        "Scientists discover new method for quantum computing breakthrough!",
        "Local team wins championship in dramatic overtime victory.",
        "Global leaders meet to discuss climate change solutions."
    ]


@pytest.fixture
def long_text():
    """Provide long text for sliding window testing."""
    return " ".join(["This is sentence number {}.".format(i) for i in range(100)])


@pytest.fixture
def mock_tokenizer():
    """Create mock tokenizer for testing."""
    mock = Mock()
    mock.cls_token_id = 101
    mock.sep_token_id = 102
    mock.pad_token_id = 0
    mock.encode = Mock(return_value=[101, 1000, 2000, 3000, 102])
    mock.decode = Mock(return_value="Decoded text")
    
    # Mock the __call__ method for tokenization
    mock_output = {
        'input_ids': [[101, 1000, 2000, 3000, 102]],
        'attention_mask': [[1, 1, 1, 1, 1]],
        'offset_mapping': [(0, 0), (0, 5), (6, 10), (11, 15), (16, 16)]
    }
    mock.__call__ = Mock(return_value=mock_output)
    
    return mock


# ============================================================================
# Text Cleaner Tests
# ============================================================================

class TestTextCleaner:
    """Test suite for TextCleaner class."""
    
    def test_initialization_default_config(self):
        """Test TextCleaner initialization with default configuration."""
        cleaner = TextCleaner()
        
        assert cleaner.config is not None
        assert isinstance(cleaner.config, CleaningConfig)
        assert cleaner.config.normalize_unicode is True
        assert cleaner.config.lowercase is False
        assert cleaner.patterns is not None
    
    def test_initialization_custom_config(self):
        """Test TextCleaner initialization with custom configuration."""
        config = CleaningConfig(
            lowercase=True,
            remove_punctuation=True,
            remove_stopwords=True
        )
        cleaner = TextCleaner(config)
        
        assert cleaner.config.lowercase is True
        assert cleaner.config.remove_punctuation is True
        assert cleaner.config.remove_stopwords is True
        assert cleaner.stopwords is not None
        assert len(cleaner.stopwords) > 0
    
    def test_url_removal(self):
        """Test URL removal from text."""
        config = CleaningConfig(remove_urls=True)
        cleaner = TextCleaner(config)
        
        text = "Check out https://example.com for more info"
        cleaned = cleaner.clean(text)
        
        assert "https://example.com" not in cleaned
        assert "Check out" in cleaned
        assert "for more info" in cleaned
    
    def test_email_removal(self):
        """Test email removal from text."""
        config = CleaningConfig(remove_emails=True)
        cleaner = TextCleaner(config)
        
        text = "Contact us at support@example.com for assistance"
        cleaned = cleaner.clean(text)
        
        assert "support@example.com" not in cleaned
        assert "Contact us at" in cleaned
        assert "for assistance" in cleaned
    
    def test_lowercase_conversion(self):
        """Test lowercase conversion."""
        config = CleaningConfig(lowercase=True)
        cleaner = TextCleaner(config)
        
        text = "This Is A Mixed Case Text"
        cleaned = cleaner.clean(text)
        
        assert cleaned == "this is a mixed case text"
    
    def test_whitespace_normalization(self):
        """Test whitespace normalization."""
        config = CleaningConfig(normalize_whitespace=True)
        cleaner = TextCleaner(config)
        
        text = "This   has    multiple     spaces"
        cleaned = cleaner.clean(text)
        
        assert cleaned == "This has multiple spaces"
    
    def test_batch_clean(self, sample_texts):
        """Test batch cleaning of multiple texts."""
        cleaner = TextCleaner()
        cleaned_texts = cleaner.batch_clean(sample_texts)
        
        assert len(cleaned_texts) == len(sample_texts)
        assert all(isinstance(text, str) for text in cleaned_texts)
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        cleaner = TextCleaner()
        
        assert cleaner.clean("") == ""
        assert cleaner.clean(None) == ""
        assert cleaner.clean("   ") == ""
    
    def test_get_statistics(self):
        """Test text statistics generation."""
        cleaner = TextCleaner(CleaningConfig(remove_urls=True))
        
        text = "Visit https://example.com for info"
        stats = cleaner.get_statistics(text)
        
        assert 'original_length' in stats
        assert 'cleaned_length' in stats
        assert 'reduction_ratio' in stats
        assert stats['urls_removed'] == 1


# ============================================================================
# Tokenization Tests
# ============================================================================

class TestTokenizer:
    """Test suite for Tokenizer class."""
    
    def test_initialization_default_config(self):
        """Test Tokenizer initialization with default configuration."""
        with patch('src.data.preprocessing.tokenization.AutoTokenizer') as mock_auto:
            mock_auto.from_pretrained.return_value = Mock()
            
            tokenizer = Tokenizer()
            
            assert tokenizer.config is not None
            assert isinstance(tokenizer.config, TokenizationConfig)
            assert tokenizer.config.max_length == 512  # MAX_SEQUENCE_LENGTH
            mock_auto.from_pretrained.assert_called_once()
    
    def test_initialization_custom_config(self):
        """Test Tokenizer initialization with custom configuration."""
        with patch('src.data.preprocessing.tokenization.AutoTokenizer') as mock_auto:
            mock_auto.from_pretrained.return_value = Mock()
            
            config = TokenizationConfig(
                model_name="roberta-base",
                max_length=256,
                padding="longest"
            )
            tokenizer = Tokenizer(config)
            
            assert tokenizer.config.model_name == "roberta-base"
            assert tokenizer.config.max_length == 256
            assert tokenizer.config.padding == "longest"
    
    @patch('src.data.preprocessing.tokenization.AutoTokenizer')
    def test_tokenize_single_text(self, mock_auto):
        """Test tokenization of single text."""
        mock_tokenizer = Mock()
        mock_output = {
            'input_ids': Mock(spec=['tolist']),
            'attention_mask': Mock(spec=['tolist'])
        }
        mock_tokenizer.return_value = mock_output
        mock_auto.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = Tokenizer()
        result = tokenizer.tokenize("Test text")
        
        mock_tokenizer.assert_called_once()
        assert result == mock_output
    
    @patch('src.data.preprocessing.tokenization.AutoTokenizer')
    def test_tokenize_batch(self, mock_auto):
        """Test batch tokenization."""
        mock_tokenizer = Mock()
        mock_auto.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = Tokenizer()
        texts = ["Text 1", "Text 2", "Text 3"]
        
        # Mock return value for batch tokenization
        mock_tokenizer.return_value = {'input_ids': Mock(), 'attention_mask': Mock()}
        
        result = tokenizer.tokenize(texts)
        
        mock_tokenizer.assert_called_once()
        assert result is not None
    
    @patch('src.data.preprocessing.tokenization.AutoTokenizer')
    def test_decode_token_ids(self, mock_auto):
        """Test decoding of token IDs."""
        mock_tokenizer = Mock()
        mock_tokenizer.decode.return_value = "Decoded text"
        mock_auto.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = Tokenizer()
        
        # Test with list
        result = tokenizer.decode([101, 1000, 2000, 102])
        assert result == "Decoded text"
        
        # Test with tensor
        import torch
        tensor_ids = torch.tensor([101, 1000, 2000, 102])
        result = tokenizer.decode(tensor_ids)
        assert result == "Decoded text"
    
    @patch('src.data.preprocessing.tokenization.AutoTokenizer')
    def test_get_vocab_size(self, mock_auto):
        """Test vocabulary size retrieval."""
        mock_tokenizer = Mock()
        mock_tokenizer.__len__.return_value = 30000
        mock_auto.from_pretrained.return_value = mock_tokenizer
        
        tokenizer = Tokenizer()
        vocab_size = tokenizer.get_vocab_size()
        
        assert vocab_size == 30000


# ============================================================================
# Feature Extraction Tests
# ============================================================================

class TestFeatureExtractor:
    """Test suite for FeatureExtractor class."""
    
    def test_initialization_default_config(self):
        """Test FeatureExtractor initialization with default configuration."""
        extractor = FeatureExtractor()
        
        assert extractor.config is not None
        assert isinstance(extractor.config, FeatureExtractionConfig)
        assert extractor.config.use_tfidf is True
        assert extractor.config.use_statistical is True
    
    def test_initialization_custom_config(self):
        """Test FeatureExtractor initialization with custom configuration."""
        config = FeatureExtractionConfig(
            use_tfidf=False,
            use_bow=True,
            use_embeddings=False,
            tfidf_max_features=5000
        )
        extractor = FeatureExtractor(config)
        
        assert extractor.config.use_tfidf is False
        assert extractor.config.use_bow is True
        assert extractor.config.tfidf_max_features == 5000
    
    def test_extract_statistical_features(self, sample_texts):
        """Test statistical feature extraction."""
        config = FeatureExtractionConfig(use_statistical=True)
        extractor = FeatureExtractor(config)
        
        features = extractor.extract_statistical_features(sample_texts)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_texts)
        assert features.shape[1] > 0  # Should have multiple statistical features
    
    def test_extract_tfidf_features(self, sample_texts):
        """Test TF-IDF feature extraction."""
        config = FeatureExtractionConfig(
            use_tfidf=True,
            tfidf_max_features=100
        )
        extractor = FeatureExtractor(config)
        
        # Fit and transform
        features = extractor.extract_tfidf_features(sample_texts, fit=True)
        
        assert features is not None
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == len(sample_texts)
        assert features.shape[1] <= 100  # Max features constraint
    
    def test_extract_all_features(self, sample_texts):
        """Test extraction of all configured features."""
        config = FeatureExtractionConfig(
            use_tfidf=True,
            use_statistical=True,
            use_embeddings=False  # Disable to avoid loading models in tests
        )
        extractor = FeatureExtractor(config)
        
        features = extractor.extract_all_features(sample_texts, fit=True)
        
        assert isinstance(features, dict)
        assert 'tfidf' in features
        assert 'statistical' in features
        assert all(isinstance(v, np.ndarray) for v in features.values())
    
    def test_combine_features(self):
        """Test feature combination."""
        extractor = FeatureExtractor()
        
        features = {
            'feature1': np.array([[1, 2], [3, 4]]),
            'feature2': np.array([[5, 6], [7, 8]])
        }
        
        combined = extractor.combine_features(features)
        
        assert combined.shape == (2, 4)
        np.testing.assert_array_equal(combined, [[1, 2, 5, 6], [3, 4, 7, 8]])
    
    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        extractor = FeatureExtractor()
        
        features = extractor.extract_statistical_features([""])
        assert features.shape[0] == 1
        
        features = extractor.extract_statistical_features([])
        assert features.shape[0] == 0


# ============================================================================
# Sliding Window Tests
# ============================================================================

class TestSlidingWindow:
    """Test suite for SlidingWindow class."""
    
    def test_initialization_default_config(self):
        """Test SlidingWindow initialization with default configuration."""
        window = SlidingWindow()
        
        assert window.config is not None
        assert isinstance(window.config, SlidingWindowConfig)
        assert window.config.window_size == 512  # MAX_SEQUENCE_LENGTH
        assert window.config.stride == 128
    
    def test_initialization_custom_config(self):
        """Test SlidingWindow initialization with custom configuration."""
        config = SlidingWindowConfig(
            window_size=256,
            stride=64,
            aggregation="max"
        )
        window = SlidingWindow(config)
        
        assert window.config.window_size == 256
        assert window.config.stride == 64
        assert window.config.aggregation == "max"
    
    def test_create_char_windows(self, long_text):
        """Test character-level window creation."""
        config = SlidingWindowConfig(window_size=100, stride=50)
        window = SlidingWindow(config)
        
        windows = window.create_windows(long_text)
        
        assert len(windows) > 1
        assert all('window_id' in w for w in windows)
        assert all('text' in w for w in windows)
        assert windows[0]['window_id'] == 0
        assert windows[-1]['is_last'] is True
    
    def test_create_token_windows(self, long_text, mock_tokenizer):
        """Test token-level window creation."""
        config = SlidingWindowConfig(window_size=10, stride=5)
        window = SlidingWindow(config)
        
        # Mock tokenizer output
        mock_tokenizer.return_value = {
            'input_ids': list(range(50)),
            'offset_mapping': [(i, i+1) for i in range(50)]
        }
        
        windows = window.create_windows(long_text, tokenizer=mock_tokenizer)
        
        assert len(windows) > 1
        assert all('window_id' in w for w in windows)
        assert all('input_ids' in w for w in windows)
    
    def test_aggregate_predictions_mean(self):
        """Test mean aggregation of predictions."""
        import torch
        
        window = SlidingWindow()
        
        predictions = [
            torch.tensor([0.1, 0.3, 0.4, 0.2]),
            torch.tensor([0.2, 0.4, 0.3, 0.1]),
            torch.tensor([0.15, 0.35, 0.35, 0.15])
        ]
        
        aggregated = window.aggregate_predictions(predictions, strategy="mean")
        
        assert aggregated.shape == (4,)
        assert torch.allclose(aggregated[0], torch.tensor(0.15))
        assert torch.allclose(aggregated[1], torch.tensor(0.35))
    
    def test_aggregate_predictions_max(self):
        """Test max aggregation of predictions."""
        import torch
        
        window = SlidingWindow()
        
        predictions = [
            torch.tensor([0.1, 0.3, 0.4, 0.2]),
            torch.tensor([0.2, 0.4, 0.3, 0.1]),
            torch.tensor([0.15, 0.35, 0.35, 0.15])
        ]
        
        aggregated = window.aggregate_predictions(predictions, strategy="max")
        
        assert aggregated.shape == (4,)
        assert aggregated[0] == 0.2
        assert aggregated[1] == 0.4
    
    def test_process_batch(self, sample_texts):
        """Test batch processing with sliding window."""
        config = SlidingWindowConfig(window_size=50, stride=25)
        window = SlidingWindow(config)
        
        result = window.process_batch(sample_texts)
        
        assert 'windows' in result
        assert 'text_to_windows' in result
        assert 'num_texts' in result
        assert result['num_texts'] == len(sample_texts)
        assert len(result['text_to_windows']) == len(sample_texts)


# ============================================================================
# Prompt Formatter Tests
# ============================================================================

class TestPromptFormatter:
    """Test suite for PromptFormatter class."""
    
    def test_initialization_default_config(self):
        """Test PromptFormatter initialization with default configuration."""
        formatter = PromptFormatter()
        
        assert formatter.config is not None
        assert isinstance(formatter.config, PromptFormatterConfig)
        assert formatter.config.template_style == "classification"
        assert len(formatter.config.templates) > 0
    
    def test_initialization_custom_config(self):
        """Test PromptFormatter initialization with custom configuration."""
        config = PromptFormatterConfig(
            template_style="instruction",
            use_demonstrations=True,
            num_demonstrations=5
        )
        formatter = PromptFormatter(config)
        
        assert formatter.config.template_style == "instruction"
        assert formatter.config.use_demonstrations is True
        assert formatter.config.num_demonstrations == 5
    
    def test_format_single_text(self):
        """Test single text formatting."""
        formatter = PromptFormatter()
        
        text = "This is a test article about technology"
        prompt = formatter.format_single(text)
        
        assert text in prompt
        assert "Categories:" in prompt or "Options:" in prompt
    
    def test_format_single_with_label(self):
        """Test single text formatting with label."""
        formatter = PromptFormatter()
        
        text = "Sports news article"
        label = 1  # Sports category
        prompt = formatter.format_single(text, label=label)
        
        assert text in prompt
        assert "Sports" in prompt
    
    def test_format_for_instruction_tuning(self):
        """Test instruction tuning format."""
        formatter = PromptFormatter()
        
        text = "Business news article"
        label = 2  # Business category
        
        result = formatter.format_for_instruction_tuning(text, label)
        
        assert 'instruction' in result
        assert 'input' in result
        assert 'output' in result
        assert result['output'] == "Business"
    
    def test_format_for_chat(self):
        """Test chat format."""
        formatter = PromptFormatter()
        
        text = "Science article"
        label = 3  # Sci/Tech category
        
        messages = formatter.format_for_chat(text, label)
        
        assert len(messages) >= 2
        assert messages[0]['role'] == 'system'
        assert messages[1]['role'] == 'user'
        assert messages[-1]['content'] == "Sci/Tech"
    
    def test_format_options(self):
        """Test options formatting."""
        config = PromptFormatterConfig(use_letters=True)
        formatter = PromptFormatter(config)
        
        options = formatter._format_options()
        
        assert "A)" in options
        assert "B)" in options
        assert "C)" in options
        assert "D)" in options
    
    def test_create_prompt_dataset(self, sample_texts):
        """Test prompt dataset creation."""
        formatter = PromptFormatter()
        
        labels = [0, 3, 1, 0]  # World, Sci/Tech, Sports, World
        prompts = formatter.create_prompt_dataset(sample_texts, labels)
        
        assert len(prompts) == len(sample_texts)
        assert all(isinstance(p, str) for p in prompts)
        assert all(text[:50] in prompt for text, prompt in zip(sample_texts, prompts))
    
    def test_chain_of_thought(self):
        """Test chain of thought prompting."""
        config = PromptFormatterConfig(use_cot=True)
        formatter = PromptFormatter(config)
        
        text = "Test article"
        prompt = formatter.format_single(text)
        
        assert "Let's think step by step" in prompt


# ============================================================================
# Integration Tests
# ============================================================================

class TestPreprocessingIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_cleaning_and_tokenization_pipeline(self, sample_texts):
        """Test integration of cleaning and tokenization."""
        # Clean texts
        cleaner = TextCleaner(CleaningConfig(normalize_whitespace=True))
        cleaned_texts = cleaner.batch_clean(sample_texts)
        
        # Tokenize cleaned texts
        with patch('src.data.preprocessing.tokenization.AutoTokenizer') as mock_auto:
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {'input_ids': Mock(), 'attention_mask': Mock()}
            mock_auto.from_pretrained.return_value = mock_tokenizer
            
            tokenizer = Tokenizer()
            tokens = tokenizer.tokenize(cleaned_texts)
            
            assert tokens is not None
            mock_tokenizer.assert_called()
    
    def test_feature_extraction_pipeline(self, sample_texts):
        """Test complete feature extraction pipeline."""
        # Clean texts
        cleaner = TextCleaner()
        cleaned_texts = cleaner.batch_clean(sample_texts)
        
        # Extract features
        config = FeatureExtractionConfig(
            use_tfidf=True,
            use_statistical=True,
            use_embeddings=False
        )
        extractor = FeatureExtractor(config)
        
        features = extractor.extract_all_features(cleaned_texts, fit=True)
        combined = extractor.combine_features(features)
        
        assert combined.shape[0] == len(sample_texts)
        assert combined.shape[1] > 0
    
    def test_prompt_formatting_pipeline(self, sample_texts):
        """Test prompt formatting with preprocessing."""
        # Clean texts
        cleaner = TextCleaner(CleaningConfig(
            normalize_whitespace=True,
            remove_urls=True
        ))
        cleaned_texts = cleaner.batch_clean(sample_texts)
        
        # Format as prompts
        formatter = PromptFormatter()
        labels = [0, 3, 1, 0]
        
        prompts = formatter.create_prompt_dataset(cleaned_texts, labels)
        
        assert len(prompts) == len(sample_texts)
        assert all(len(p) > len(t) for p, t in zip(prompts, cleaned_texts))


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs across all modules."""
        # Text cleaner
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.batch_clean([]) == []
        
        # Feature extractor
        extractor = FeatureExtractor()
        features = extractor.extract_statistical_features([])
        assert features.shape[0] == 0
        
        # Sliding window
        window = SlidingWindow()
        windows = window.create_windows("")
        assert len(windows) == 0 or (len(windows) == 1 and windows[0]['text'] == "")
        
        # Prompt formatter
        formatter = PromptFormatter()
        prompt = formatter.format_single("")
        assert prompt != ""  # Should still have template structure
    
    def test_very_long_text_handling(self):
        """Test handling of very long texts."""
        very_long_text = "word " * 10000  # Very long text
        
        # Text cleaner should handle without error
        cleaner = TextCleaner()
        cleaned = cleaner.clean(very_long_text)
        assert len(cleaned) > 0
        
        # Sliding window should create multiple windows
        window = SlidingWindow(SlidingWindowConfig(window_size=100, stride=50))
        windows = window.create_windows(very_long_text)
        assert len(windows) > 10
        
        # Feature extractor should handle
        extractor = FeatureExtractor()
        features = extractor.extract_statistical_features([very_long_text])
        assert features.shape[0] == 1
    
    def test_special_characters_handling(self):
        """Test handling of special characters."""
        special_text = "Test 测试 🚀 €100 #hashtag @mention"
        
        # Text cleaner
        cleaner = TextCleaner(CleaningConfig(normalize_unicode=True))
        cleaned = cleaner.clean(special_text)
        assert "Test" in cleaned
        
        # Feature extractor
        extractor = FeatureExtractor()
        features = extractor.extract_statistical_features([special_text])
        assert features.shape[0] == 1
        
        # Prompt formatter
        formatter = PromptFormatter()
        prompt = formatter.format_single(special_text)
        assert len(prompt) > len(special_text)
    
    def test_invalid_aggregation_strategy(self):
        """Test invalid aggregation strategy in sliding window."""
        import torch
        
        window = SlidingWindow()
        predictions = [torch.tensor([0.1, 0.2, 0.3, 0.4])]
        
        with pytest.raises(ValueError, match="Unknown aggregation strategy"):
            window.aggregate_predictions(predictions, strategy="invalid")
    
    def test_no_predictions_to_aggregate(self):
        """Test aggregation with no predictions."""
        window = SlidingWindow()
        
        with pytest.raises(ValueError, match="No predictions to aggregate"):
            window.aggregate_predictions([])


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Test performance characteristics of preprocessing modules."""
    
    def test_batch_processing_efficiency(self, sample_texts):
        """Test that batch processing is more efficient than individual processing."""
        import time
        
        cleaner = TextCleaner()
        
        # Measure batch processing time
        start = time.time()
        batch_result = cleaner.batch_clean(sample_texts * 10)
        batch_time = time.time() - start
        
        # Measure individual processing time
        start = time.time()
        individual_result = [cleaner.clean(text) for text in sample_texts * 10]
        individual_time = time.time() - start
        
        # Batch should be at least as fast (usually faster due to vectorization)
        assert len(batch_result) == len(individual_result)
        # Note: This assertion is relaxed as performance can vary
        assert batch_time < individual_time * 2
    
    def test_memory_efficiency(self):
        """Test memory efficiency of feature extraction."""
        # Create moderate-sized dataset
        texts = ["Sample text " + str(i) for i in range(100)]
        
        config = FeatureExtractionConfig(
            use_tfidf=True,
            tfidf_max_features=100  # Limit features for memory
        )
        extractor = FeatureExtractor(config)
        
        features = extractor.extract_tfidf_features(texts, fit=True)
        
        # Check feature matrix is sparse-friendly size
        assert features.shape[1] <= 100
        assert features.nbytes < 1000000  # Less than 1MB for 100 texts


if __name__ == "__main__":
    """Allow running tests directly."""
    pytest.main([__file__, "-v", "--tb=short"])
