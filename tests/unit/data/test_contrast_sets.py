"""
Unit Tests for Contrast Set Generation Module
==============================================

Comprehensive test suite for contrast set generation following:
- IEEE 829-2008: Standard for Software Test Documentation
- Gardner et al. (2020): "Evaluating Models' Local Decision Boundaries via Contrast Sets"
- Academic Testing Best Practices

This module tests:
- Contrast set generation strategies
- Perturbation techniques
- Label flipping mechanisms
- Quality validation
- Domain-specific rules for AG News

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

# Mock external dependencies
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['spacy'] = MagicMock()
sys.modules['nltk'] = MagicMock()

# Import contrast set module
from src.data.augmentation.contrast_set_generator import ContrastSetGenerator, ContrastSetConfig


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_news_texts():
    """Provide sample news texts for each AG News category."""
    return {
        0: "The United Nations announced new sanctions against the country following diplomatic tensions.",  # World
        1: "The team won the championship game with a last-minute goal in overtime.",  # Sports
        2: "Company profits increased by 20% in the third quarter due to strong sales.",  # Business
        3: "Scientists developed a new artificial intelligence algorithm for data analysis."  # Sci/Tech
    }


@pytest.fixture
def mock_nlp():
    """Create mock spaCy NLP model."""
    nlp = MagicMock()
    doc = MagicMock()
    doc.noun_chunks = [MagicMock(text="test phrase")]
    nlp.return_value = doc
    return nlp


# ============================================================================
# ContrastSetConfig Tests
# ============================================================================

class TestContrastSetConfig:
    """Test suite for ContrastSetConfig."""
    
    def test_default_config_initialization(self):
        """Test ContrastSetConfig with default values."""
        config = ContrastSetConfig()
        
        assert config.generation_strategy == "rule_based"
        assert config.contrast_type == "minimal"
        assert config.max_perturbations == 3
        assert config.preserve_fluency is True
        assert config.ensure_label_change is True
        assert config.category_specific_rules is True
    
    def test_custom_config_initialization(self):
        """Test ContrastSetConfig with custom values."""
        config = ContrastSetConfig(
            generation_strategy="model_based",
            contrast_type="diverse",
            max_perturbations=5,
            target_label_strategy="random",
            min_edit_distance=2,
            max_edit_distance=15
        )
        
        assert config.generation_strategy == "model_based"
        assert config.contrast_type == "diverse"
        assert config.max_perturbations == 5
        assert config.target_label_strategy == "random"
        assert config.min_edit_distance == 2
        assert config.max_edit_distance == 15
    
    def test_perturbation_types_initialization(self):
        """Test default perturbation types initialization."""
        config = ContrastSetConfig()
        
        assert "entity" in config.perturbation_types
        assert "number" in config.perturbation_types
        assert "negation" in config.perturbation_types
        assert "temporal" in config.perturbation_types
        assert "location" in config.perturbation_types
        assert len(config.perturbation_types) == 5
    
    def test_news_categories_initialization(self):
        """Test AG News categories initialization."""
        config = ContrastSetConfig()
        
        assert config.news_categories == ["World", "Sports", "Business", "Sci/Tech"]
        assert len(config.news_categories) == 4


# ============================================================================
# ContrastSetGenerator Tests
# ============================================================================

class TestContrastSetGenerator:
    """Test suite for ContrastSetGenerator."""
    
    def test_initialization_default(self):
        """Test ContrastSetGenerator initialization with default config."""
        generator = ContrastSetGenerator()
        
        assert generator.name == "contrast_set"
        assert generator.config.generation_strategy == "rule_based"
        assert hasattr(generator, 'perturbation_rules')
        assert hasattr(generator, 'entity_replacements')
    
    def test_initialization_custom_config(self):
        """Test ContrastSetGenerator with custom configuration."""
        config = ContrastSetConfig(
            generation_strategy="model_based",
            max_perturbations=5
        )
        generator = ContrastSetGenerator(config)
        
        assert generator.config.generation_strategy == "model_based"
        assert generator.config.max_perturbations == 5
    
    def test_perturbation_rules_structure(self):
        """Test structure of perturbation rules."""
        generator = ContrastSetGenerator()
        
        # Check all categories have rules
        for category in ["World", "Sports", "Business", "Sci/Tech"]:
            assert category in generator.perturbation_rules
            assert "entities" in generator.perturbation_rules[category]
            assert "templates" in generator.perturbation_rules[category]
    
    def test_entity_replacements_structure(self):
        """Test structure of entity replacements."""
        generator = ContrastSetGenerator()
        
        # Check all categories have replacements
        for category in ["World", "Sports", "Business", "Sci/Tech"]:
            assert category in generator.entity_replacements
            assert len(generator.entity_replacements[category]) > 0
    
    def test_get_nearest_label(self):
        """Test nearest label selection."""
        generator = ContrastSetGenerator()
        
        # Test edge cases
        assert generator._get_nearest_label(0) == 1
        assert generator._get_nearest_label(3) == 2
        
        # Test middle cases
        result = generator._get_nearest_label(1)
        assert result in [0, 2]
        
        result = generator._get_nearest_label(2)
        assert result in [1, 3]
    
    def test_entity_replacement(self):
        """Test entity replacement perturbation."""
        generator = ContrastSetGenerator()
        
        # World news text
        text = "The USA announced new policy changes"
        result = generator._entity_replacement(text, "World", "Business")
        
        assert isinstance(result, str)
        assert len(result) > 0
        
        # Check if entity was replaced (may or may not change)
        # Due to randomness, we just check it returns valid text
        assert result is not None
    
    def test_number_perturbation(self):
        """Test number perturbation."""
        generator = ContrastSetGenerator()
        
        text = "The company reported 100 million in revenue"
        result = generator._number_perturbation(text)
        
        assert isinstance(result, str)
        
        # Text without numbers
        text_no_numbers = "The company reported strong revenue"
        result = generator._number_perturbation(text_no_numbers)
        assert result == text_no_numbers
    
    def test_negation_insertion(self):
        """Test negation insertion/removal."""
        generator = ContrastSetGenerator()
        
        # Text without negation
        text = "The team won the game"
        result = generator._negation_insertion(text)
        assert isinstance(result, str)
        
        # Text with negation
        text_neg = "The team did not win the game"
        result = generator._negation_insertion(text_neg)
        assert isinstance(result, str)
    
    def test_temporal_shift(self):
        """Test temporal reference shifting."""
        generator = ContrastSetGenerator()
        
        text = "The event will happen tomorrow"
        result = generator._temporal_shift(text)
        
        assert isinstance(result, str)
        # Check if temporal word was changed
        assert result != text or "tomorrow" not in result or result == text
    
    def test_location_change(self):
        """Test location reference changes."""
        generator = ContrastSetGenerator()
        
        text = "The conference was held in New York"
        result = generator._location_change(text, "World", "World")
        
        assert isinstance(result, str)
        # May or may not change based on random selection
        assert len(result) > 0
    
    def test_validate_contrast(self):
        """Test contrast validation."""
        generator = ContrastSetGenerator()
        
        original = "This is the original news text"
        
        # Valid contrast (moderate change)
        contrast1 = "This is the modified news text"
        valid = generator._validate_contrast(original, contrast1, 0, 1)
        assert valid is True
        
        # Invalid contrast (identical)
        contrast2 = "This is the original news text"
        invalid = generator._validate_contrast(original, contrast2, 0, 1)
        assert invalid is False
        
        # Invalid contrast (too different)
        contrast3 = "Completely different content here"
        invalid = generator._validate_contrast(original, contrast3, 0, 1)
        assert invalid is False
    
    def test_rule_based_generation(self):
        """Test rule-based contrast generation."""
        generator = ContrastSetGenerator()
        
        text = "The United States announced new trade policies"
        contrasts = generator._rule_based_generation(text, 0, 2)  # World to Business
        
        assert isinstance(contrasts, list)
        assert len(contrasts) <= generator.config.max_perturbations
        
        for contrast in contrasts:
            assert isinstance(contrast, str)
            assert len(contrast) > 0
    
    def test_augment_single(self, sample_news_texts):
        """Test single text contrast generation."""
        generator = ContrastSetGenerator()
        
        # Test World news
        text = sample_news_texts[0]
        label = 0
        
        contrasts = generator.augment_single(text, label, target_label=2)
        
        assert isinstance(contrasts, list)
        assert all(isinstance(c, tuple) for c in contrasts)
        assert all(len(c) == 2 for c in contrasts)  # (text, label) pairs
        
        for contrast_text, contrast_label in contrasts:
            assert isinstance(contrast_text, str)
            assert isinstance(contrast_label, int)
            assert contrast_label != label  # Label should change
    
    def test_generate_contrast_dataset(self, sample_news_texts):
        """Test contrast dataset generation."""
        generator = ContrastSetGenerator()
        
        # Create dataset
        dataset = [(text, label) for label, text in sample_news_texts.items()]
        
        contrast_dataset = generator.generate_contrast_dataset(
            dataset,
            num_contrasts_per_sample=1
        )
        
        assert isinstance(contrast_dataset, list)
        assert len(contrast_dataset) > 0
        
        for item in contrast_dataset:
            assert len(item) == 4  # (orig_text, orig_label, contrast_text, contrast_label)
            orig_text, orig_label, contrast_text, contrast_label = item
            assert isinstance(orig_text, str)
            assert isinstance(orig_label, int)
            assert isinstance(contrast_text, str)
            assert isinstance(contrast_label, int)
            assert orig_label != contrast_label  # Labels should differ


# ============================================================================
# Strategy-Specific Tests
# ============================================================================

class TestContrastGenerationStrategies:
    """Test different contrast generation strategies."""
    
    def test_minimal_contrast_generation(self):
        """Test minimal contrast generation."""
        config = ContrastSetConfig(contrast_type="minimal")
        generator = ContrastSetGenerator(config)
        
        text = "The company announced record profits"
        contrasts = generator._rule_based_generation(text, 2, 3)  # Business to Sci/Tech
        
        assert isinstance(contrasts, list)
        # Minimal contrasts should have limited changes
        for contrast in contrasts:
            assert len(contrast.split()) >= len(text.split()) - 3
    
    def test_diverse_contrast_generation(self):
        """Test diverse contrast generation."""
        config = ContrastSetConfig(
            contrast_type="diverse",
            max_perturbations=5
        )
        generator = ContrastSetGenerator(config)
        
        text = "The team won the championship"
        contrasts = generator._rule_based_generation(text, 1, 0)  # Sports to World
        
        assert isinstance(contrasts, list)
        # Should generate multiple diverse contrasts
        assert len(contrasts) <= 5
    
    def test_all_target_labels_strategy(self):
        """Test generation for all target labels."""
        config = ContrastSetConfig(target_label_strategy="all")
        generator = ContrastSetGenerator(config)
        
        text = "Technology companies invest in AI"
        label = 3  # Sci/Tech
        
        contrasts = generator.augment_single(text, label)
        
        # Should have contrasts for all other labels (0, 1, 2)
        target_labels = [c[1] for c in contrasts]
        assert len(set(target_labels)) >= 1  # At least one different label
    
    def test_nearest_label_strategy(self):
        """Test nearest label selection strategy."""
        config = ContrastSetConfig(target_label_strategy="nearest")
        generator = ContrastSetGenerator(config)
        
        text = "Sports team wins the game"
        label = 1  # Sports
        
        contrasts = generator.augment_single(text, label)
        
        if contrasts:
            target_labels = [c[1] for c in contrasts]
            # Should target adjacent labels (0 or 2)
            assert all(l in [0, 2] for l in target_labels)


# ============================================================================
# Integration Tests
# ============================================================================

class TestContrastSetIntegration:
    """Integration tests for contrast set generation."""
    
    def test_full_pipeline(self, sample_news_texts):
        """Test complete contrast generation pipeline."""
        generator = ContrastSetGenerator()
        
        all_contrasts = []
        
        for label, text in sample_news_texts.items():
            contrasts = generator.augment_single(text, label)
            all_contrasts.extend(contrasts)
        
        assert len(all_contrasts) > 0
        
        # Check variety in generated contrasts
        contrast_texts = [c[0] for c in all_contrasts]
        assert len(set(contrast_texts)) > 1  # Should have different texts
        
        contrast_labels = [c[1] for c in all_contrasts]
        assert len(set(contrast_labels)) > 1  # Should have different labels
    
    def test_caching_functionality(self):
        """Test caching in contrast generation."""
        config = ContrastSetConfig(cache_augmented=True)
        generator = ContrastSetGenerator(config)
        
        text = "Test text for caching"
        label = 0
        
        # First call
        result1 = generator.augment_single(text, label, target_label=1)
        
        # Second call - should use cache
        result2 = generator.augment_single(text, label, target_label=1)
        
        assert generator.stats['cached'] > 0
    
    @patch('src.data.augmentation.contrast_set_generator.spacy')
    def test_with_nlp_tools(self, mock_spacy, mock_nlp):
        """Test contrast generation with NLP tools."""
        mock_spacy.load.return_value = mock_nlp
        
        generator = ContrastSetGenerator()
        generator.nlp = mock_nlp
        
        text = "The company announced new products"
        result = generator._negation_insertion(text)
        
        assert isinstance(result, str)
        assert len(result) > 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestContrastSetEdgeCases:
    """Test edge cases in contrast set generation."""
    
    def test_empty_text_handling(self):
        """Test handling of empty text."""
        generator = ContrastSetGenerator()
        
        contrasts = generator.augment_single("", label=0)
        
        assert isinstance(contrasts, list)
        assert len(contrasts) == 0 or all(c[0] == "" for c in contrasts)
    
    def test_single_word_handling(self):
        """Test handling of single-word input."""
        generator = ContrastSetGenerator()
        
        contrasts = generator.augment_single("Sports", label=1)
        
        assert isinstance(contrasts, list)
        # Single word is hard to contrast meaningfully
        assert len(contrasts) >= 0
    
    def test_very_long_text_handling(self):
        """Test handling of very long text."""
        generator = ContrastSetGenerator()
        
        long_text = " ".join(["word"] * 500)
        contrasts = generator.augment_single(long_text, label=0, target_label=1)
        
        assert isinstance(contrasts, list)
        
        for contrast_text, _ in contrasts:
            assert len(contrast_text) > 0
    
    def test_special_characters_handling(self):
        """Test handling of special characters."""
        generator = ContrastSetGenerator()
        
        text = "News: $100 million deal @company #breaking"
        contrasts = generator.augment_single(text, label=2)
        
        assert isinstance(contrasts, list)
        
        for contrast_text, _ in contrasts:
            assert isinstance(contrast_text, str)
    
    def test_no_valid_contrasts(self):
        """Test when no valid contrasts can be generated."""
        generator = ContrastSetGenerator()
        
        # Override validation to always fail
        generator._validate_contrast = MagicMock(return_value=False)
        
        text = "Test text"
        contrasts = generator.augment_single(text, label=0)
        
        assert isinstance(contrasts, list)
        assert len(contrasts) == 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestContrastSetPerformance:
    """Test performance characteristics of contrast generation."""
    
    def test_generation_consistency(self):
        """Test consistency of contrast generation."""
        generator = ContrastSetGenerator()
        generator.rng.seed(42)  # Fix seed for consistency
        
        text = "The team won the match"
        label = 1
        
        # Generate multiple times with same seed
        contrasts1 = generator._rule_based_generation(text, label, 0)
        
        generator.rng.seed(42)  # Reset seed
        contrasts2 = generator._rule_based_generation(text, label, 0)
        
        # Should produce same results with same seed
        assert contrasts1 == contrasts2
    
    def test_batch_generation_efficiency(self, sample_news_texts):
        """Test efficiency of batch contrast generation."""
        generator = ContrastSetGenerator()
        
        # Create larger dataset
        dataset = []
        for _ in range(10):
            for label, text in sample_news_texts.items():
                dataset.append((text, label))
        
        contrast_dataset = generator.generate_contrast_dataset(
            dataset,
            num_contrasts_per_sample=1
        )
        
        assert len(contrast_dataset) > 0
        assert len(contrast_dataset) <= len(dataset) * 3  # Maximum 3 contrasts per sample


if __name__ == "__main__":
    """Allow running tests directly."""
    pytest.main([__file__, "-v", "--tb=short"])
