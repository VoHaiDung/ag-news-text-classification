"""
Pytest Configuration and Shared Fixtures
=========================================

This module provides centralized test configuration following established
testing frameworks and academic standards:

Standards and Methodologies:
----------------------------
- Meszaros (2007): "xUnit Test Patterns: Refactoring Test Code"
  Defines test fixture patterns and test doubles taxonomy
- Fowler (2018): "Refactoring: Improving the Design of Existing Code"
  Guides test structure and organization principles
- Beck (2002): "Test Driven Development: By Example"
  Informs test-first methodology and fixture design

Fixture Architecture:
--------------------
The fixture system implements Dependency Injection pattern (Martin, 2017)
to provide:
1. Test isolation through mock objects
2. Shared test data and utilities
3. Resource management and cleanup
4. Performance optimization through caching

Mock Strategy:
-------------
Following the Test Double patterns (Meszaros, 2007):
- Dummy objects: Simple placeholders
- Stub objects: Provide canned responses
- Mock objects: Verify interactions
- Fake objects: Simplified implementations

Academic Context:
----------------
Testing methodology aligned with empirical software engineering research:
- Runeson & Höst (2009): "Guidelines for conducting and reporting case study research"
- Wohlin et al. (2012): "Experimentation in Software Engineering"

Author: Võ Hải Dũng
Institution: Academic Research Laboratory
License: MIT
Version: 1.0.0
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, Mock, PropertyMock
import pytest
import numpy as np

# ============================================================================
# Section 1: Path Configuration
# ============================================================================
# Establishing test environment isolation

PROJECT_ROOT = Path(__file__).parent.parent

# Note: We intentionally do NOT add PROJECT_ROOT to sys.path here
# to avoid triggering imports from src that may have missing dependencies.
# Each test module should handle its own imports as needed.

# ============================================================================
# Section 2: Pytest Configuration Hooks
# ============================================================================
# Following pytest best practices documentation

def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.
    
    This hook is called after command line options have been parsed
    and all plugins and initial conftest files have been loaded.
    
    References:
    - pytest documentation: "Writing plugins"
    - Okken (2017): "Python Testing with pytest"
    """
    # Register custom markers for test categorization
    config.addinivalue_line(
        "markers", "unit: Unit tests for isolated component testing"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "slow: Tests with execution time > 1 second"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests requiring CUDA-capable hardware"
    )
    config.addinivalue_line(
        "markers", "benchmark: Performance measurement tests"
    )
    config.addinivalue_line(
        "markers", "experimental: Unstable tests under development"
    )


def pytest_sessionstart(session):
    """
    Initialize test session with diagnostic information.
    
    Provides visibility into test environment for reproducibility,
    following guidelines from:
    - Basili et al. (1986): "Experimentation in Software Engineering"
    """
    print("\n" + "="*80)
    print("AG NEWS TEXT CLASSIFICATION - TEST SUITE")
    print("="*80)
    print(f"Test session directory: {session.startdir}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print(f"NumPy version: {np.__version__}")
    print("="*80 + "\n")


def pytest_sessionfinish(session, exitstatus):
    """
    Finalize test session with summary statistics.
    
    Implements post-test analysis recommended in:
    - IEEE 1008-1987: "Standard for Software Unit Testing"
    """
    print("\n" + "="*80)
    print("TEST SESSION SUMMARY")
    print("-"*80)
    print(f"Exit status: {exitstatus}")
    if hasattr(session, 'testscollected'):
        print(f"Tests collected: {session.testscollected}")
    if hasattr(session, 'testsfailed'):
        print(f"Tests failed: {session.testsfailed}")
    print("="*80 + "\n")


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection for optimization and categorization.
    
    Implements test prioritization strategies from:
    - Rothermel et al. (2001): "Prioritizing Test Cases For Regression Testing"
    """
    for item in items:
        # Auto-categorize tests based on path structure
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        
        # Skip GPU tests in CPU-only environments
        if "gpu" in item.keywords:
            skip_gpu = pytest.mark.skip(reason="GPU tests disabled in CI environment")
            item.add_marker(skip_gpu)


# ============================================================================
# Section 3: Shared Fixtures
# ============================================================================
# Implementing fixture patterns from Meszaros (2007)

@pytest.fixture(scope="session")
def project_root():
    """
    Provide project root path for test resource location.
    
    Scope: Session (shared across all tests)
    Pattern: Object Mother (Meszaros, 2007)
    """
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """
    Provide test data directory with automatic creation.
    
    Implements Shared Fixture pattern for test data management.
    """
    test_dir = project_root / "tests" / "fixtures" / "data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def temp_dir(tmp_path):
    """
    Provide temporary directory for test isolation.
    
    Implements Fresh Fixture pattern ensuring test independence.
    """
    return tmp_path


@pytest.fixture
def sample_texts():
    """
    Provide representative AG News text samples.
    
    Dataset characteristics based on:
    - Zhang et al. (2015): "Character-level Convolutional Networks"
    
    Returns:
        list: Text samples covering all AG News categories
    """
    return [
        "Wall Street closed mixed on Tuesday as investors digested earnings reports.",
        "Researchers at MIT developed a new machine learning algorithm for drug discovery.",
        "The Lakers defeated the Celtics 110-95 in last night's championship game.",
        "Apple announced its latest iPhone with advanced AI capabilities.",
        "The UN Security Council convened to discuss the ongoing humanitarian crisis.",
        "Tennis star wins Grand Slam title after dramatic five-set match."
    ]


@pytest.fixture
def sample_labels():
    """
    Provide corresponding AG News category labels.
    
    Label encoding:
    - 0: World
    - 1: Sports  
    - 2: Business
    - 3: Science/Technology
    """
    return [2, 3, 1, 3, 0, 1]


@pytest.fixture
def ag_news_categories():
    """
    Provide AG News category names following original dataset structure.
    """
    return ["World", "Sports", "Business", "Sci/Tech"]


@pytest.fixture
def sample_config():
    """
    Provide representative configuration for testing.
    
    Configuration values based on empirical studies:
    - Learning rate: Liu et al. (2019) "RoBERTa" recommendations
    - Batch size: McCandlish et al. (2018) "An Empirical Model of Large-Batch Training"
    """
    return {
        'model': {
            'name': 'deberta-v3',
            'num_labels': 4,
            'max_length': 512,
            'hidden_size': 768,
            'num_attention_heads': 12,
            'num_hidden_layers': 12,
            'dropout_prob': 0.1
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 2e-5,
            'num_epochs': 3,
            'warmup_ratio': 0.1,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'optimizer': 'adamw',
            'scheduler': 'linear'
        },
        'data': {
            'train_size': 0.8,
            'validation_size': 0.1,
            'test_size': 0.1,
            'max_length': 512,
            'pad_to_max_length': True,
            'num_workers': 4
        }
    }


# ============================================================================
# Section 4: Mock Objects and Test Doubles
# ============================================================================
# Implementing test double patterns from Meszaros (2007)

@pytest.fixture
def mock_model():
    """
    Create mock model following PyTorch nn.Module interface.
    
    Implements Mock Object pattern for behavior verification.
    """
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=[MagicMock()])
    model.state_dict = MagicMock(return_value={'layer.weight': MagicMock()})
    model.load_state_dict = MagicMock()
    model.forward = MagicMock(return_value=MagicMock(logits=MagicMock()))
    return model


@pytest.fixture
def mock_tokenizer():
    """
    Create mock tokenizer following HuggingFace interface.
    
    Implements Stub pattern providing canned responses.
    """
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[101, 1000, 2000, 3000, 102])
    tokenizer.decode = MagicMock(return_value="Decoded text")
    tokenizer.pad_token_id = 0
    tokenizer.cls_token_id = 101
    tokenizer.sep_token_id = 102
    tokenizer.max_length = 512
    tokenizer.__call__ = MagicMock(return_value={
        'input_ids': [[101, 1000, 2000, 3000, 102]],
        'attention_mask': [[1, 1, 1, 1, 1]]
    })
    return tokenizer


@pytest.fixture
def mock_config_file(temp_dir):
    """
    Create temporary configuration file for testing.
    
    Implements Fake Object pattern with simplified implementation.
    """
    config_file = temp_dir / "config.yaml"
    config_content = """
    model:
      name: test_model
      version: 1.0
      num_labels: 4
    training:
      batch_size: 32
      epochs: 10
      learning_rate: 2e-5
    data:
      max_length: 512
      num_workers: 4
    """
    config_file.write_text(config_content)
    return config_file


# ============================================================================
# Section 5: Assertion Helpers
# ============================================================================
# Custom assertions following assertion pattern guidelines

@pytest.fixture
def assert_shape():
    """
    Provide shape assertion helper for array validation.
    
    Following NumPy testing utilities design.
    """
    def _assert_shape(array, expected_shape):
        """
        Assert array shape matches expectation.
        
        Args:
            array: Array-like object with shape attribute
            expected_shape: Expected shape tuple
        """
        if hasattr(array, 'shape'):
            actual_shape = array.shape
        else:
            actual_shape = np.array(array).shape
        assert actual_shape == expected_shape, \
            f"Shape mismatch: expected {expected_shape}, got {actual_shape}"
    return _assert_shape


@pytest.fixture
def assert_between():
    """
    Provide range assertion helper for boundary testing.
    
    Implements boundary value analysis from:
    Myers et al. (2011): "The Art of Software Testing"
    """
    def _assert_between(value, min_val, max_val, inclusive=True):
        """
        Assert value falls within specified range.
        
        Args:
            value: Value to test
            min_val: Minimum boundary
            max_val: Maximum boundary
            inclusive: Whether boundaries are inclusive
        """
        if inclusive:
            assert min_val <= value <= max_val, \
                f"Value {value} outside range [{min_val}, {max_val}]"
        else:
            assert min_val < value < max_val, \
                f"Value {value} outside range ({min_val}, {max_val})"
    return _assert_between


@pytest.fixture
def assert_close():
    """
    Provide floating-point comparison helper.
    
    Implements approximate equality testing following:
    Goldberg (1991): "What Every Computer Scientist Should Know About Floating-Point"
    """
    def _assert_close(actual, expected, rtol=1e-5, atol=1e-8):
        """
        Assert approximate equality for floating-point values.
        
        Args:
            actual: Actual value
            expected: Expected value
            rtol: Relative tolerance
            atol: Absolute tolerance
        """
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    return _assert_close


# ============================================================================
# Section 6: Test Data Generators
# ============================================================================
# Factory fixtures for test data generation

@pytest.fixture
def create_random_data():
    """
    Factory for random test data generation.
    
    Implements Factory pattern for test data creation with
    controlled randomness for reproducibility.
    """
    def _create_random_data(shape, dtype='float32', seed=None):
        """
        Generate random data with specified characteristics.
        
        Args:
            shape: Data shape tuple
            dtype: Data type specification
            seed: Random seed for reproducibility
            
        Returns:
            np.ndarray: Generated random data
        """
        if seed is not None:
            np.random.seed(seed)
        
        if dtype in ['float32', 'float64']:
            return np.random.randn(*shape).astype(dtype)
        elif dtype in ['int32', 'int64', 'int']:
            return np.random.randint(0, 100, size=shape, dtype=dtype)
        elif dtype == 'bool':
            return np.random.choice([True, False], size=shape)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    return _create_random_data


# ============================================================================
# Section 7: Resource Management
# ============================================================================
# Cleanup and resource management following RAII pattern

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Automatic cleanup after each test.
    
    Implements teardown pattern ensuring test isolation.
    """
    # Setup phase
    yield
    # Teardown phase
    # Resource cleanup handled automatically by pytest


@pytest.fixture(scope="session", autouse=True)
def cleanup_session(request):
    """
    Session-level cleanup for resource finalization.
    
    Ensures proper resource disposal at session end.
    """
    def finalizer():
        """Perform final cleanup operations."""
        pass  # Cleanup operations if needed
    request.addfinalizer(finalizer)
