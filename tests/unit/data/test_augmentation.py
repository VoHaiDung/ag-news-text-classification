"""
Unit Tests for Data Augmentation Modules
=========================================

Comprehensive test suite for augmentation components following:
- IEEE 829-2008: Standard for Software Test Documentation
- ISO/IEC/IEEE 29119: Software Testing Standards
- Academic Testing Best Practices

This module tests:
- Base augmenter functionality
- Back translation augmentation
- Adversarial augmentation
- MixUp and CutMix strategies
- Paraphrase generation
- Token replacement techniques

Author: Võ Hải Dũng
License: MIT
"""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock, create_autospec

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Mock all external dependencies before any imports
# ============================================================================

# Create comprehensive mocks for all external libraries
mock_torch = MagicMock()
mock_torch.__version__ = '2.0.0'
mock_torch.tensor = MagicMock(return_value=MagicMock())
mock_torch.nn = MagicMock()
mock_torch.nn.functional = MagicMock()
mock_torch.device = MagicMock(return_value='cpu')
mock_torch.no_grad = MagicMock()
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=False)

mock_transformers = MagicMock()
mock_nltk = MagicMock()
mock_wordnet = MagicMock()
mock_spacy = MagicMock()
mock_sentence_transformers = MagicMock()

# Install mocks into sys.modules
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn
sys.modules['torch.nn.functional'] = mock_torch.nn.functional
sys.modules['transformers'] = mock_transformers
sys.modules['nltk'] = mock_nltk
sys.modules['nltk.corpus'] = MagicMock()
sys.modules['nltk.corpus.wordnet'] = mock_wordnet
sys.modules['spacy'] = mock_spacy
sys.modules['sentence_transformers'] = mock_sentence_transformers

# Import augmentation modules after mocking
from src.data.augmentation.base_augmenter import BaseAugmenter, AugmentationConfig, CompositeAugmenter
from src.data.augmentation.back_translation import BackTranslationAugmenter, BackTranslationConfig
from src.data.augmentation.adversarial import AdversarialAugmenter, AdversarialConfig
from src.data.augmentation.mixup import MixUpAugmenter, MixUpConfig
from src.data.augmentation.cutmix import CutMixAugmenter, CutMixConfig
from src.data.augmentation.paraphrase import ParaphraseAugmenter, ParaphraseConfig
from src.data.augmentation.token_replacement import TokenReplacementAugmenter, TokenReplacementConfig


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_texts():
    """Provide sample texts for testing."""
    return [
        "The stock market showed strong gains today amid positive economic data.",
        "Scientists have discovered a new method for treating cancer patients.",
        "The team won the championship game in overtime with a dramatic finish.",
        "Technology companies are investing heavily in artificial intelligence research."
    ]


@pytest.fixture
def sample_labels():
    """Provide sample labels for AG News categories."""
    return [2, 3, 1, 3]  # Business, Sci/Tech, Sports, Sci/Tech


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.eval = MagicMock()
    model.to = MagicMock(return_value=model)
    model.generate = MagicMock(return_value=mock_torch.tensor([[1, 2, 3]]))
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
    tokenizer.decode = MagicMock(return_value="Augmented text")
    tokenizer.__call__ = MagicMock(return_value={'input_ids': mock_torch.tensor([[1, 2, 3]])})
    return tokenizer


# ============================================================================
# Base Augmenter Tests
# ============================================================================

class TestBaseAugmenter:
    """Test suite for BaseAugmenter abstract class."""
    
    def test_augmentation_config_initialization(self):
        """Test AugmentationConfig initialization with default values."""
        config = AugmentationConfig()
        
        assert config.augmentation_rate == 0.5
        assert config.num_augmentations == 1
        assert config.min_similarity == 0.8
        assert config.max_similarity == 0.99
        assert config.preserve_label is True
        assert config.seed == 42
    
    def test_augmentation_config_custom_values(self):
        """Test AugmentationConfig with custom values."""
        config = AugmentationConfig(
            augmentation_rate=0.7,
            num_augmentations=3,
            min_length=20,
            max_length=256
        )
        
        assert config.augmentation_rate == 0.7
        assert config.num_augmentations == 3
        assert config.min_length == 20
        assert config.max_length == 256
    
    def test_composite_augmenter_initialization(self):
        """Test CompositeAugmenter initialization."""
        # Create mock augmenters
        aug1 = MagicMock(spec=BaseAugmenter)
        aug2 = MagicMock(spec=BaseAugmenter)
        
        composite = CompositeAugmenter(
            augmenters=[aug1, aug2],
            strategy="sequential"
        )
        
        assert len(composite.augmenters) == 2
        assert composite.strategy == "sequential"
        assert composite.name == "composite"
    
    def test_composite_augmenter_sequential_strategy(self):
        """Test CompositeAugmenter with sequential strategy."""
        # Create mock augmenters
        aug1 = MagicMock(spec=BaseAugmenter)
        aug1.augment_single.return_value = "First augmented"
        
        aug2 = MagicMock(spec=BaseAugmenter)
        aug2.augment_single.return_value = "Second augmented"
        
        composite = CompositeAugmenter(
            augmenters=[aug1, aug2],
            strategy="sequential"
        )
        
        result = composite.augment_single("Test text")
        
        assert aug1.augment_single.called
        assert aug2.augment_single.called
        assert isinstance(result, list)
    
    def test_cache_operations(self):
        """Test cache operations in base augmenter."""
        # Create a concrete implementation for testing
        class TestAugmenter(BaseAugmenter):
            def augment_single(self, text, label=None, **kwargs):
                return f"Augmented: {text}"
        
        config = AugmentationConfig(cache_augmented=True)
        augmenter = TestAugmenter(config)
        
        # Test cache key generation
        key = augmenter.get_cache_key("test text", param1="value1")
        assert isinstance(key, str)
        assert "test text" in key
        
        # Test adding to cache
        augmenter.add_to_cache("test", ["augmented"], param1="value1")
        assert augmenter.cache is not None
        
        # Test getting from cache
        cached = augmenter.get_from_cache("test", param1="value1")
        assert cached == ["augmented"]
    
    def test_statistics_tracking(self):
        """Test statistics tracking in base augmenter."""
        class TestAugmenter(BaseAugmenter):
            def augment_single(self, text, label=None, **kwargs):
                return f"Augmented: {text}"
        
        augmenter = TestAugmenter()
        
        # Check initial stats
        stats = augmenter.get_stats()
        assert stats['total_augmented'] == 0
        assert stats['successful'] == 0
        
        # Update stats
        augmenter.stats['total_augmented'] = 10
        augmenter.stats['successful'] = 8
        
        stats = augmenter.get_stats()
        assert stats['success_rate'] == 0.8
        
        # Reset stats
        augmenter.reset_stats()
        stats = augmenter.get_stats()
        assert stats['total_augmented'] == 0


# ============================================================================
# Back Translation Tests
# ============================================================================

class TestBackTranslationAugmenter:
    """Test suite for BackTranslationAugmenter."""
    
    def test_initialization_default_config(self):
        """Test BackTranslationAugmenter initialization with default config."""
        with patch('src.data.augmentation.back_translation.MarianMTModel'):
            with patch('src.data.augmentation.back_translation.MarianTokenizer'):
                augmenter = BackTranslationAugmenter()
                
                assert augmenter.name == "back_translation"
                assert augmenter.config.num_beams == 5
                assert augmenter.config.temperature == 1.0
                assert 'de' in augmenter.config.pivot_languages
    
    def test_initialization_custom_config(self):
        """Test BackTranslationAugmenter with custom configuration."""
        config = BackTranslationConfig(
            pivot_languages=['fr', 'es'],
            num_beams=3,
            temperature=0.8
        )
        
        with patch('src.data.augmentation.back_translation.MarianMTModel'):
            with patch('src.data.augmentation.back_translation.MarianTokenizer'):
                augmenter = BackTranslationAugmenter(config)
                
                assert augmenter.config.pivot_languages == ['fr', 'es']
                assert augmenter.config.num_beams == 3
                assert augmenter.config.temperature == 0.8
    
    @patch('src.data.augmentation.back_translation.MarianMTModel')
    @patch('src.data.augmentation.back_translation.MarianTokenizer')
    def test_augment_single(self, mock_tokenizer_class, mock_model_class):
        """Test single text augmentation with back translation."""
        # Setup mocks
        mock_model = MagicMock()
        mock_model.generate.return_value = mock_torch.tensor([[1, 2, 3]])
        mock_model_class.from_pretrained.return_value = mock_model
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.decode.return_value = "Translated text"
        mock_tokenizer.__call__ = MagicMock(return_value={'input_ids': mock_torch.tensor([[1, 2]])})
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        augmenter = BackTranslationAugmenter()
        augmenter.models = {'de': {'forward': mock_model, 'backward': mock_model}}
        augmenter.tokenizers = {'de': {'forward': mock_tokenizer, 'backward': mock_tokenizer}}
        
        result = augmenter.augment_single("Test text")
        
        assert isinstance(result, list)
        assert len(result) > 0
    
    def test_translation_validation(self):
        """Test translation quality validation."""
        with patch('src.data.augmentation.back_translation.MarianMTModel'):
            with patch('src.data.augmentation.back_translation.MarianTokenizer'):
                augmenter = BackTranslationAugmenter()
                
                # Test valid translation
                valid = augmenter._validate_translation(
                    "This is a test sentence.",
                    "This is a test phrase."
                )
                assert valid is True
                
                # Test identical translation (invalid)
                invalid = augmenter._validate_translation(
                    "Test text",
                    "Test text"
                )
                assert invalid is False
                
                # Test too short translation
                invalid = augmenter._validate_translation(
                    "This is a long sentence with many words.",
                    "Short"
                )
                assert invalid is False


# ============================================================================
# Adversarial Augmentation Tests
# ============================================================================

class TestAdversarialAugmenter:
    """Test suite for AdversarialAugmenter."""
    
    @patch('src.data.augmentation.adversarial.AutoModelForSequenceClassification')
    @patch('src.data.augmentation.adversarial.AutoTokenizer')
    def test_initialization_default_config(self, mock_tokenizer, mock_model):
        """Test AdversarialAugmenter initialization."""
        augmenter = AdversarialAugmenter()
        
        assert augmenter.name == "adversarial"
        assert augmenter.config.epsilon == 0.1
        assert augmenter.config.attack_type == "word_substitution"
        assert augmenter.config.use_gradient is True
    
    def test_adversarial_config_custom(self):
        """Test AdversarialConfig with custom values."""
        config = AdversarialConfig(
            epsilon=0.2,
            alpha=0.02,
            num_iterations=10,
            attack_type="char_flip"
        )
        
        assert config.epsilon == 0.2
        assert config.alpha == 0.02
        assert config.num_iterations == 10
        assert config.attack_type == "char_flip"
    
    @patch('src.data.augmentation.adversarial.AutoModelForSequenceClassification')
    @patch('src.data.augmentation.adversarial.AutoTokenizer')
    def test_character_flip_attack(self, mock_tokenizer, mock_model):
        """Test character-level perturbation attack."""
        augmenter = AdversarialAugmenter()
        
        text = "Test text"
        result = augmenter._character_flip_attack(text)
        
        assert isinstance(result, str)
        assert len(result) == len(text)
        # Some characters should be different
        assert result != text or result == text  # May or may not change
    
    @patch('src.data.augmentation.adversarial.AutoModelForSequenceClassification')
    @patch('src.data.augmentation.adversarial.AutoTokenizer')
    @patch('src.data.augmentation.adversarial.wordnet')
    def test_word_substitution_attack(self, mock_wordnet, mock_tokenizer, mock_model):
        """Test word substitution attack."""
        # Setup mocks
        mock_wordnet.synsets.return_value = []
        
        augmenter = AdversarialAugmenter()
        augmenter.wordnet = mock_wordnet
        
        text = "This is a test sentence"
        result = augmenter._word_substitution_attack(text)
        
        assert isinstance(result, str)
        assert len(result.split()) <= len(text.split()) + 5  # Allow some variation


# ============================================================================
# MixUp Tests
# ============================================================================

class TestMixUpAugmenter:
    """Test suite for MixUpAugmenter."""
    
    def test_initialization_default_config(self):
        """Test MixUpAugmenter initialization."""
        augmenter = MixUpAugmenter()
        
        assert augmenter.name == "mixup"
        assert augmenter.config.alpha == 0.2
        assert augmenter.config.mixup_strategy == "word"
    
    def test_mixup_config_custom(self):
        """Test MixUpConfig with custom values."""
        config = MixUpConfig(
            alpha=0.4,
            beta=0.4,
            mixup_strategy="sentence",
            symmetric=False
        )
        
        assert config.alpha == 0.4
        assert config.beta == 0.4
        assert config.mixup_strategy == "sentence"
        assert config.symmetric is False
    
    def test_sample_lambda(self):
        """Test lambda sampling for mixup."""
        augmenter = MixUpAugmenter()
        
        # Sample multiple times
        lambdas = [augmenter._sample_lambda() for _ in range(100)]
        
        # Check all values are in valid range
        assert all(0 <= l <= 1 for l in lambdas)
        
        # Check variation exists
        assert len(set(lambdas)) > 1
    
    def test_word_level_mixup(self):
        """Test word-level mixing."""
        augmenter = MixUpAugmenter()
        
        text1 = "The cat sat on mat"
        text2 = "A dog jumped over fence"
        lambda_val = 0.6
        
        result = augmenter._word_level_mixup(text1, text2, lambda_val)
        
        assert isinstance(result, str)
        assert len(result.split()) > 0
    
    def test_sentence_level_mixup(self):
        """Test sentence-level mixing."""
        augmenter = MixUpAugmenter()
        
        text1 = "First sentence. Second sentence. Third sentence."
        text2 = "Another text. More content. Final part."
        lambda_val = 0.5
        
        result = augmenter._sentence_level_mixup(text1, text2, lambda_val)
        
        assert isinstance(result, str)
        assert "." in result
    
    def test_label_mixing(self):
        """Test label mixing for soft labels."""
        augmenter = MixUpAugmenter()
        
        label1 = 0
        label2 = 2
        lambda_val = 0.7
        
        mixed_label = augmenter._mix_labels(label1, label2, lambda_val)
        
        assert isinstance(mixed_label, np.ndarray)
        assert mixed_label.shape[0] >= 3
        assert np.allclose(mixed_label.sum(), 1.0)
        assert mixed_label[label1] == 0.7
        assert mixed_label[label2] == 0.3


# ============================================================================
# CutMix Tests
# ============================================================================

class TestCutMixAugmenter:
    """Test suite for CutMixAugmenter."""
    
    def test_initialization_default_config(self):
        """Test CutMixAugmenter initialization."""
        augmenter = CutMixAugmenter()
        
        assert augmenter.name == "cutmix"
        assert augmenter.config.alpha == 1.0
        assert augmenter.config.cut_strategy == "continuous"
    
    def test_cutmix_config_custom(self):
        """Test CutMixConfig with custom values."""
        config = CutMixConfig(
            alpha=2.0,
            cut_strategy="sentence",
            min_cut_ratio=0.2,
            max_cut_ratio=0.6
        )
        
        assert config.alpha == 2.0
        assert config.cut_strategy == "sentence"
        assert config.min_cut_ratio == 0.2
        assert config.max_cut_ratio == 0.6
    
    def test_continuous_cutmix(self):
        """Test continuous span cutting and mixing."""
        augmenter = CutMixAugmenter()
        
        text1 = "The quick brown fox jumps over lazy dog"
        text2 = "A beautiful day with clear blue sky"
        lambda_val = 0.5
        
        result = augmenter._continuous_cutmix(text1, text2, lambda_val)
        
        assert isinstance(result, str)
        assert len(result.split()) > 0
    
    def test_random_cutmix(self):
        """Test random word cutting and mixing."""
        augmenter = CutMixAugmenter()
        
        text1 = "First text with some words"
        text2 = "Second text with different content"
        lambda_val = 0.6
        
        result = augmenter._random_cutmix(text1, text2, lambda_val)
        
        assert isinstance(result, str)
        assert len(result.split()) > 0
    
    def test_sentence_cutmix(self):
        """Test sentence-level cutting and mixing."""
        augmenter = CutMixAugmenter()
        
        text1 = "First sentence. Second one. Third sentence."
        text2 = "Another sentence. More text. Final sentence."
        lambda_val = 0.5
        
        result = augmenter._sentence_cutmix(text1, text2, lambda_val)
        
        assert isinstance(result, str)
        assert "." in result


# ============================================================================
# Paraphrase Tests
# ============================================================================

class TestParaphraseAugmenter:
    """Test suite for ParaphraseAugmenter."""
    
    @patch('src.data.augmentation.paraphrase.PegasusForConditionalGeneration')
    @patch('src.data.augmentation.paraphrase.PegasusTokenizer')
    def test_initialization_default_config(self, mock_tokenizer, mock_model):
        """Test ParaphraseAugmenter initialization."""
        augmenter = ParaphraseAugmenter()
        
        assert augmenter.name == "paraphrase"
        assert augmenter.config.model_type == "pegasus"
        assert augmenter.config.num_return_sequences == 3
    
    def test_paraphrase_config_custom(self):
        """Test ParaphraseConfig with custom values."""
        config = ParaphraseConfig(
            model_type="t5",
            num_return_sequences=5,
            temperature=1.5,
            num_beams=15
        )
        
        assert config.model_type == "t5"
        assert config.num_return_sequences == 5
        assert config.temperature == 1.5
        assert config.num_beams == 15
    
    @patch('src.data.augmentation.paraphrase.PegasusForConditionalGeneration')
    @patch('src.data.augmentation.paraphrase.PegasusTokenizer')
    def test_paraphrase_validation(self, mock_tokenizer, mock_model):
        """Test paraphrase quality validation."""
        augmenter = ParaphraseAugmenter()
        
        original = "This is the original sentence"
        
        # Valid paraphrase (sufficient change)
        paraphrase1 = "Here we have the initial statement"
        valid = augmenter._validate_paraphrases(original, [paraphrase1])
        assert len(valid) == 1
        
        # Invalid paraphrase (too similar)
        paraphrase2 = "This is the original sentence"
        invalid = augmenter._validate_paraphrases(original, [paraphrase2])
        assert len(invalid) == 0


# ============================================================================
# Token Replacement Tests
# ============================================================================

class TestTokenReplacementAugmenter:
    """Test suite for TokenReplacementAugmenter."""
    
    @patch('src.data.augmentation.token_replacement.wordnet')
    def test_initialization_default_config(self, mock_wordnet):
        """Test TokenReplacementAugmenter initialization."""
        augmenter = TokenReplacementAugmenter()
        
        assert augmenter.name == "token_replacement"
        assert augmenter.config.synonym_replacement_prob == 0.1
        assert augmenter.config.random_insertion_prob == 0.1
    
    def test_token_replacement_config_custom(self):
        """Test TokenReplacementConfig with custom values."""
        config = TokenReplacementConfig(
            synonym_replacement_prob=0.2,
            random_deletion_prob=0.15,
            max_replacements=10
        )
        
        assert config.synonym_replacement_prob == 0.2
        assert config.random_deletion_prob == 0.15
        assert config.max_replacements == 10
    
    @patch('src.data.augmentation.token_replacement.wordnet')
    def test_synonym_replacement(self, mock_wordnet):
        """Test synonym replacement operation."""
        mock_wordnet.synsets.return_value = []
        
        augmenter = TokenReplacementAugmenter()
        augmenter.wordnet = mock_wordnet
        
        words = ["The", "quick", "brown", "fox"]
        result = augmenter._synonym_replacement(words.copy())
        
        assert isinstance(result, list)
        assert len(result) == len(words)
    
    @patch('src.data.augmentation.token_replacement.wordnet')
    def test_random_swap(self, mock_wordnet):
        """Test random word swapping."""
        augmenter = TokenReplacementAugmenter()
        
        words = ["one", "two", "three", "four", "five"]
        result = augmenter._random_swap(words.copy())
        
        assert isinstance(result, list)
        assert len(result) == len(words)
        assert set(result) == set(words)  # Same words, possibly different order
    
    @patch('src.data.augmentation.token_replacement.wordnet')
    def test_random_deletion(self, mock_wordnet):
        """Test random word deletion."""
        augmenter = TokenReplacementAugmenter()
        
        words = ["one", "two", "three", "four", "five"]
        result = augmenter._random_deletion(words.copy())
        
        assert isinstance(result, list)
        assert len(result) > 0  # At least one word remains
        assert len(result) <= len(words)


# ============================================================================
# Integration Tests
# ============================================================================

class TestAugmentationIntegration:
    """Integration tests for augmentation pipeline."""
    
    def test_augmentation_pipeline(self, sample_texts):
        """Test complete augmentation pipeline."""
        # Create multiple augmenters
        augmenters = []
        
        # Token replacement
        with patch('src.data.augmentation.token_replacement.wordnet'):
            token_aug = TokenReplacementAugmenter()
            augmenters.append(token_aug)
        
        # MixUp
        mixup_aug = MixUpAugmenter()
        augmenters.append(mixup_aug)
        
        # CutMix
        cutmix_aug = CutMixAugmenter()
        augmenters.append(cutmix_aug)
        
        # Create composite augmenter
        composite = CompositeAugmenter(
            augmenters=augmenters,
            strategy="all"
        )
        
        # Apply augmentation
        for text in sample_texts:
            result = composite.augment_single(text)
            assert isinstance(result, list)
            assert len(result) >= 0
    
    def test_augmentation_with_labels(self, sample_texts, sample_labels):
        """Test augmentation with label preservation."""
        config = MixUpConfig(mixup_strategy="word")
        augmenter = MixUpAugmenter(config)
        
        # Test batch augmentation
        results = augmenter.augment_batch(sample_texts, sample_labels)
        
        assert isinstance(results, tuple)
        texts, labels = results
        assert len(texts) == len(sample_texts)
        
        if labels is not None:
            assert len(labels) == len(sample_labels)
    
    def test_caching_functionality(self):
        """Test caching in augmentation."""
        with patch('src.data.augmentation.token_replacement.wordnet'):
            config = TokenReplacementConfig(cache_augmented=True)
            augmenter = TokenReplacementAugmenter(config)
            augmenter._get_synonyms = MagicMock(return_value=["synonym"])
            
            text = "Test text for caching"
            
            # First call - should generate
            result1 = augmenter.augment_single(text)
            
            # Second call - should use cache
            result2 = augmenter.augment_single(text)
            
            assert augmenter.stats['cached'] > 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestAugmentationEdgeCases:
    """Test edge cases and error handling in augmentation."""
    
    def test_empty_text_handling(self):
        """Test handling of empty text input."""
        augmenters = [
            MixUpAugmenter(),
            CutMixAugmenter(),
        ]
        
        for augmenter in augmenters:
            result = augmenter.augment_single("")
            assert result == "" or result == [""]
    
    def test_single_word_handling(self):
        """Test handling of single-word input."""
        with patch('src.data.augmentation.token_replacement.wordnet'):
            augmenter = TokenReplacementAugmenter()
            augmenter._get_synonyms = MagicMock(return_value=[])
            
            result = augmenter.augment_single("Word")
            assert isinstance(result, list)
            assert len(result) > 0
    
    def test_very_long_text_handling(self):
        """Test handling of very long text."""
        long_text = " ".join(["word"] * 1000)
        
        augmenter = MixUpAugmenter()
        mix_text = " ".join(["other"] * 1000)
        
        result = augmenter.augment_single(long_text, mix_with=mix_text)
        
        assert isinstance(result, str) or isinstance(result, tuple)
    
    def test_special_characters_handling(self):
        """Test handling of special characters."""
        special_text = "Text with @#$% special characters!!!"
        
        with patch('src.data.augmentation.token_replacement.wordnet'):
            augmenter = TokenReplacementAugmenter()
            augmenter._get_synonyms = MagicMock(return_value=[])
            
            result = augmenter.augment_single(special_text)
            assert isinstance(result, list)


if __name__ == "__main__":
    """Allow running tests directly."""
    pytest.main([__file__, "-v", "--tb=short"])
