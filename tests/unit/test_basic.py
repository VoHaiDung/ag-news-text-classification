#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Basic Unit Tests for AG News Classification Framework
======================================================

This module provides basic unit tests to verify core functionality.
Following testing best practices from:
- Osherove (2013): "The Art of Unit Testing"
- Beck (2002): "Test Driven Development"

Author: Võ Hải Dũng
License: MIT
"""

import sys
import pytest
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

def test_project_structure():
    """Test that project structure is correct."""
    # Check required directories exist
    assert (PROJECT_ROOT / "src").exists(), "src directory missing"
    assert (PROJECT_ROOT / "configs").exists(), "configs directory missing"
    assert (PROJECT_ROOT / "scripts").exists(), "scripts directory missing"
    
def test_python_version():
    """Test Python version requirement."""
    assert sys.version_info >= (3, 8), "Python 3.8+ required"
    
def test_imports():
    """Test that core modules can be imported."""
    try:
        # Test standard library imports
        import os
        import json
        import logging
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import standard library: {e}")
        
def test_constants():
    """Test that constants are defined correctly."""
    from configs.constants import (
        AG_NEWS_CLASSES,
        AG_NEWS_NUM_CLASSES,
        PROJECT_NAME
    )
    
    assert AG_NEWS_CLASSES == ["World", "Sports", "Business", "Sci/Tech"]
    assert AG_NEWS_NUM_CLASSES == 4
    assert PROJECT_NAME == "AG News Text Classification"
    
class TestDataValidation:
    """Test data validation utilities."""
    
    def test_label_validation(self):
        """Test label validation."""
        valid_labels = [0, 1, 2, 3]
        invalid_labels = [-1, 4, 5, 100]
        
        for label in valid_labels:
            assert 0 <= label < 4, f"Invalid label: {label}"
            
        for label in invalid_labels:
            assert not (0 <= label < 4), f"Label should be invalid: {label}"
            
    def test_text_validation(self):
        """Test text validation."""
        valid_texts = [
            "This is a valid news article.",
            "Another valid text with numbers 123.",
            "Text with special chars: @#$%"
        ]
        
        invalid_texts = [
            "",  # Empty
            " ",  # Only whitespace
            None,  # None
        ]
        
        for text in valid_texts:
            assert text and text.strip(), f"Text should be valid: {text}"
            
        for text in invalid_texts:
            assert not (text and str(text).strip() if text else False), \
                   f"Text should be invalid: {text}"

class TestConfiguration:
    """Test configuration loading."""
    
    def test_config_files_exist(self):
        """Test that configuration files exist."""
        config_files = [
            "configs/constants.py",
            "configs/environments/dev.yaml",
            "configs/environments/prod.yaml",
        ]
        
        for config_file in config_files:
            config_path = PROJECT_ROOT / config_file
            assert config_path.exists(), f"Config file missing: {config_file}"
            
    def test_yaml_loading(self):
        """Test YAML configuration loading."""
        import yaml
        
        yaml_file = PROJECT_ROOT / "configs" / "environments" / "dev.yaml"
        if yaml_file.exists():
            with open(yaml_file) as f:
                config = yaml.safe_load(f)
                assert isinstance(config, dict), "Config should be a dictionary"
                assert "name" in config, "Config should have 'name' field"

class TestUtilities:
    """Test utility functions."""
    
    def test_path_utilities(self):
        """Test path utility functions."""
        from pathlib import Path
        
        # Test path creation
        test_path = Path("/tmp/test/path")
        assert isinstance(test_path, Path)
        
        # Test path operations
        parent = test_path.parent
        assert parent == Path("/tmp/test")
        
    def test_logging_setup(self):
        """Test logging configuration."""
        import logging
        
        # Create logger
        logger = logging.getLogger("test_logger")
        logger.setLevel(logging.INFO)
        
        # Test log levels
        assert logger.level == logging.INFO
        
    def test_reproducibility_seed(self):
        """Test reproducibility seed setting."""
        import random
        import numpy as np
        
        # Set seed
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate random numbers
        random_nums = [random.random() for _ in range(5)]
        np_nums = np.random.random(5)
        
        # Reset seed and verify same sequence
        random.seed(seed)
        np.random.seed(seed)
        
        random_nums_2 = [random.random() for _ in range(5)]
        np_nums_2 = np.random.random(5)
        
        assert random_nums == random_nums_2, "Random seed not working"
        assert np.allclose(np_nums, np_nums_2), "NumPy seed not working"

@pytest.mark.parametrize("text,expected_words", [
    ("Hello world", 2),
    ("This is a test sentence.", 5),
    ("Single", 1),
    ("", 0),
])
def test_word_counting(text, expected_words):
    """Test word counting with parametrization."""
    word_count = len(text.split()) if text else 0
    assert word_count == expected_words, f"Expected {expected_words} words, got {word_count}"

def test_performance_baseline():
    """Test basic performance requirements."""
    import time
    
    # Simple performance test
    start = time.time()
    
    # Simulate some work
    result = sum(range(1000000))
    
    elapsed = time.time() - start
    
    # Should complete in reasonable time
    assert elapsed < 1.0, f"Operation too slow: {elapsed:.3f}s"
    assert result > 0, "Result should be positive"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
