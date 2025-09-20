"""
Pytest Configuration and Shared Fixtures
=========================================

Central configuration for all tests following:
- Pytest Best Practices
- Academic Software Testing Standards
- Dependency Injection Patterns

This module provides:
- Global fixtures for all tests
- Mock configurations
- Path setup
- Common test utilities

Author: Võ Hải Dũng
License: MIT
"""

import sys
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest
import numpy as np

# ============================================================================
# Path Configuration
# ============================================================================

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# Global Mock Configuration
# ============================================================================

def pytest_configure(config):
    """
    Configure pytest with global mocks before any test collection.
    
    This ensures all external dependencies are mocked before any imports.
    """
    
    # Create comprehensive mocks for all external libraries
    mock_libraries = {
        # Data libraries
        'joblib': MagicMock(),
        'requests': MagicMock(),
        'pandas': MagicMock(),
        
        # Deep learning
        'torch': create_torch_mock(),
        'torch.nn': MagicMock(),
        'torch.nn.functional': MagicMock(),
        'torch.optim': MagicMock(),
        'torch.utils': MagicMock(),
        'torch.utils.data': MagicMock(),
        'transformers': MagicMock(),
        'sentence_transformers': MagicMock(),
        
        # NLP libraries
        'nltk': MagicMock(),
        'nltk.corpus': MagicMock(),
        'nltk.corpus.wordnet': MagicMock(),
        'nltk.tokenize': MagicMock(),
        'spacy': MagicMock(),
        
        # ML libraries
        'sklearn': MagicMock(),
        'sklearn.feature_extraction': MagicMock(),
        'sklearn.feature_extraction.text': MagicMock(),
        'sklearn.decomposition': MagicMock(),
        'sklearn.metrics': MagicMock(),
        
        # Visualization
        'matplotlib': MagicMock(),
        'matplotlib.pyplot': MagicMock(),
        'seaborn': MagicMock(),
        'plotly': MagicMock(),
        
        # Other utilities
        'wandb': MagicMock(),
        'tqdm': MagicMock(),
        'yaml': MagicMock(),
    }
    
    # Install all mocks
    for module_name, mock_obj in mock_libraries.items():
        sys.modules[module_name] = mock_obj


def create_torch_mock():
    """Create a comprehensive torch mock with commonly used attributes."""
    mock_torch = MagicMock()
    mock_torch.__version__ = '2.0.0'
    mock_torch.tensor = MagicMock(return_value=MagicMock())
    mock_torch.zeros = MagicMock(return_value=MagicMock())
    mock_torch.ones = MagicMock(return_value=MagicMock())
    mock_torch.rand = MagicMock(return_value=MagicMock())
    mock_torch.randn = MagicMock(return_value=MagicMock())
    mock_torch.device = MagicMock(return_value='cpu')
    mock_torch.cuda = MagicMock()
    mock_torch.cuda.is_available = MagicMock(return_value=False)
    mock_torch.no_grad = MagicMock()
    mock_torch.enable_grad = MagicMock()
    mock_torch.manual_seed = MagicMock()
    return mock_torch


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Provide project root path."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root):
    """Provide test data directory path."""
    return project_root / "tests" / "fixtures" / "data"


@pytest.fixture
def sample_texts():
    """Provide sample AG News texts for testing."""
    return [
        "The stock market showed strong gains today amid positive economic data.",
        "Scientists have discovered a new method for treating cancer patients.",
        "The team won the championship game in overtime with a dramatic finish.",
        "Technology companies are investing heavily in artificial intelligence research."
    ]


@pytest.fixture
def sample_labels():
    """Provide sample AG News labels."""
    return [2, 3, 1, 3]  # Business, Sci/Tech, Sports, Sci/Tech


@pytest.fixture
def ag_news_categories():
    """Provide AG News category names."""
    return ["World", "Sports", "Business", "Sci/Tech"]


@pytest.fixture
def sample_config():
    """Provide sample configuration dictionary."""
    return {
        'model': {
            'name': 'deberta-v3',
            'num_labels': 4,
            'max_length': 512
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 2e-5,
            'num_epochs': 3
        },
        'data': {
            'train_size': 0.8,
            'validation_size': 0.1,
            'test_size': 0.1
        }
    }


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.eval = MagicMock()
    model.train = MagicMock()
    model.to = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=[])
    model.state_dict = MagicMock(return_value={})
    model.load_state_dict = MagicMock()
    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = MagicMock()
    tokenizer.encode = MagicMock(return_value=[101, 1000, 2000, 3000, 102])
    tokenizer.decode = MagicMock(return_value="Decoded text")
    tokenizer.pad_token_id = 0
    tokenizer.cls_token_id = 101
    tokenizer.sep_token_id = 102
    tokenizer.__call__ = MagicMock(return_value={
        'input_ids': [[101, 1000, 2000, 3000, 102]],
        'attention_mask': [[1, 1, 1, 1, 1]]
    })
    return tokenizer


@pytest.fixture
def mock_dataset():
    """Create a mock dataset for testing."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=100)
    dataset.__getitem__ = MagicMock(return_value={
        'input_ids': [101, 1000, 2000, 3000, 102],
        'attention_mask': [1, 1, 1, 1, 1],
        'labels': 0
    })
    return dataset


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for file operations."""
    return tmp_path


@pytest.fixture
def mock_config_file(temp_dir):
    """Create a mock config file."""
    config_file = temp_dir / "config.yaml"
    config_content = """
    model:
      name: test_model
      version: 1.0
    training:
      batch_size: 32
      epochs: 10
    """
    config_file.write_text(config_content)
    return config_file


# ============================================================================
# Test Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "unit: Unit tests for individual components"
    )
    config.addinivalue_line(
        "markers", "integration: Integration tests for component interactions"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "gpu: Tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "benchmark: Performance benchmark tests"
    )


# ============================================================================
# Test Session Configuration
# ============================================================================

def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    print("\n" + "="*80)
    print("AG NEWS TEXT CLASSIFICATION - TEST SUITE")
    print("="*80)
    print(f"Test session started: {session.startdir}")
    print("="*80 + "\n")


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    print("\n" + "="*80)
    print("TEST SESSION COMPLETED")
    print(f"Exit status: {exitstatus}")
    print("="*80 + "\n")


# ============================================================================
# Test Collection Hooks
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers and configure test order.
    """
    for item in items:
        # Add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        
        # Skip GPU tests if CUDA not available
        if "gpu" in item.keywords:
            try:
                import torch
                if not torch.cuda.is_available():
                    skip_gpu = pytest.mark.skip(reason="GPU not available")
                    item.add_marker(skip_gpu)
            except ImportError:
                pass  # torch is mocked, skip GPU tests


# ============================================================================
# Utility Functions for Tests
# ============================================================================

@pytest.fixture
def assert_shape():
    """Provide assertion helper for array shapes."""
    def _assert_shape(array, expected_shape):
        assert array.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {array.shape}"
    return _assert_shape


@pytest.fixture
def assert_between():
    """Provide assertion helper for value ranges."""
    def _assert_between(value, min_val, max_val):
        assert min_val <= value <= max_val, \
            f"Value {value} not in range [{min_val}, {max_val}]"
    return _assert_between


@pytest.fixture
def create_random_data():
    """Factory fixture for creating random test data."""
    def _create_random_data(shape, dtype='float32'):
        if dtype == 'float32':
            return np.random.randn(*shape).astype(np.float32)
        elif dtype == 'int':
            return np.random.randint(0, 10, size=shape)
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
    return _create_random_data


# ============================================================================
# Cleanup Functions
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Automatically cleanup after each test."""
    yield
    # Cleanup code here if needed
    pass
