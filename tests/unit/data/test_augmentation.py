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
from unittest.mock import Mock, patch, MagicMock, create_autospec

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# CRITICAL: Mock all external dependencies BEFORE any src imports
# ============================================================================

# Mock joblib first (needed by io_utils)
sys.modules['joblib'] = MagicMock()

# Mock torch and related modules
mock_torch = MagicMock()
mock_torch.__version__ = '2.0.0'
mock_torch.tensor = MagicMock(return_value=MagicMock())
mock_torch.nn = MagicMock()
mock_torch.nn.functional = MagicMock()
mock_torch.device = MagicMock(return_value='cpu')
mock_torch.no_grad = MagicMock()
mock_torch.cuda = MagicMock()
mock_torch.cuda.is_available = MagicMock(return_value=False)

sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn
sys.modules['torch.nn.functional'] = mock_torch.nn.functional
sys.modules['torch.optim'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()

# Mock transformers
sys.modules['transformers'] = MagicMock()

# Mock nltk
mock_nltk = MagicMock()
mock_wordnet = MagicMock()
sys.modules['nltk'] = mock_nltk
sys.modules['nltk.corpus'] = MagicMock()
sys.modules['nltk.corpus.wordnet'] = mock_wordnet
sys.modules['nltk.tokenize'] = MagicMock()

# Mock spacy
sys.modules['spacy'] = MagicMock()

# Mock sklearn
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.feature_extraction'] = MagicMock()
sys.modules['sklearn.feature_extraction.text'] = MagicMock()
sys.modules['sklearn.decomposition'] = MagicMock()

# Mock sentence_transformers
sys.modules['sentence_transformers'] = MagicMock()

# Now we can safely import after all mocks are in place
import pytest
import numpy as np

# Import augmentation modules after all mocking is complete
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
# Simplified Tests for Other Augmenters
# ============================================================================

class TestAugmentersInitialization:
    """Test initialization of various augmenters."""
    
    def test_back_translation_config(self):
        """Test BackTranslationConfig initialization."""
        config = BackTranslationConfig(
            pivot_languages=['fr', 'es'],
            num_beams=3,
            temperature=0.8
        )
        
        assert config.pivot_languages == ['fr', 'es']
        assert config.num_beams == 3
        assert config.temperature == 0.8
    
    def test_adversarial_config(self):
        """Test AdversarialConfig initialization."""
        config = AdversarialConfig(
            epsilon=0.2,
            alpha=0.02,
            attack_type="char_flip"
        )
        
        assert config.epsilon == 0.2
        assert config.alpha == 0.02
        assert config.attack_type == "char_flip"
    
    def test_mixup_config(self):
        """Test MixUpConfig initialization."""
        config = MixUpConfig(
            alpha=0.4,
            beta=0.4,
            mixup_strategy="sentence"
        )
        
        assert config.alpha == 0.4
        assert config.beta == 0.4
        assert config.mixup_strategy == "sentence"
    
    def test_cutmix_config(self):
        """Test CutMixConfig initialization."""
        config = CutMixConfig(
            alpha=2.0,
            cut_strategy="sentence",
            min_cut_ratio=0.2
        )
        
        assert config.alpha == 2.0
        assert config.cut_strategy == "sentence"
        assert config.min_cut_ratio == 0.2
    
    def test_paraphrase_config(self):
        """Test ParaphraseConfig initialization."""
        config = ParaphraseConfig(
            model_type="t5",
            num_return_sequences=5,
            temperature=1.5
        )
        
        assert config.model_type == "t5"
        assert config.num_return_sequences == 5
        assert config.temperature == 1.5
    
    def test_token_replacement_config(self):
        """Test TokenReplacementConfig initialization."""
        config = TokenReplacementConfig(
            synonym_replacement_prob=0.2,
            random_deletion_prob=0.15,
            max_replacements=10
        )
        
        assert config.synonym_replacement_prob == 0.2
        assert config.random_deletion_prob == 0.15
        assert config.max_replacements == 10


if __name__ == "__main__":
    """Allow running tests directly."""
    pytest.main([__file__, "-v", "--tb=short"])
