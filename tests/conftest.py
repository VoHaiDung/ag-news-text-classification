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
from unittest.mock import MagicMock, Mock, PropertyMock
import pytest
import numpy as np

# ============================================================================
# Mock External Dependencies BEFORE ANY Project Imports
# ============================================================================

def create_torch_mock():
    """Create a comprehensive torch mock with commonly used attributes."""
    mock_torch = MagicMock()
    mock_torch.__version__ = '2.0.0'
    
    # Tensor operations
    mock_torch.tensor = MagicMock(side_effect=lambda x, **kwargs: MagicMock(
        shape=np.array(x).shape if hasattr(x, '__len__') else (),
        dtype=kwargs.get('dtype', 'float32')
    ))
    mock_torch.zeros = MagicMock(side_effect=lambda *shape, **kwargs: MagicMock(shape=shape))
    mock_torch.ones = MagicMock(side_effect=lambda *shape, **kwargs: MagicMock(shape=shape))
    mock_torch.rand = MagicMock(side_effect=lambda *shape, **kwargs: MagicMock(shape=shape))
    mock_torch.randn = MagicMock(side_effect=lambda *shape, **kwargs: MagicMock(shape=shape))
    mock_torch.stack = MagicMock(return_value=MagicMock())
    mock_torch.cat = MagicMock(return_value=MagicMock())
    
    # Device operations
    mock_torch.device = MagicMock(side_effect=lambda x: x)
    mock_torch.cuda = MagicMock()
    mock_torch.cuda.is_available = MagicMock(return_value=False)
    mock_torch.cuda.Stream = MagicMock()
    mock_torch.cuda.current_stream = MagicMock()
    
    # Context managers
    mock_torch.no_grad = MagicMock()
    mock_torch.enable_grad = MagicMock()
    
    # Random operations
    mock_torch.manual_seed = MagicMock()
    
    # Data types
    mock_torch.float32 = 'float32'
    mock_torch.int64 = 'int64'
    mock_torch.long = 'long'
    
    return mock_torch


def create_mock_dataloader():
    """Create mock DataLoader class."""
    mock_dataloader = MagicMock()
    mock_dataloader.return_value = MagicMock(
        __iter__=MagicMock(return_value=iter([])),
        __len__=MagicMock(return_value=0)
    )
    return mock_dataloader


def install_all_mocks():
    """Install all mock modules before any project imports."""
    
    # Create torch mock with submodules
    torch_mock = create_torch_mock()
    
    # torch.utils.data module structure
    torch_mock.utils = MagicMock()
    torch_mock.utils.data = MagicMock()
    torch_mock.utils.data.DataLoader = create_mock_dataloader()
    torch_mock.utils.data.Dataset = MagicMock()
    torch_mock.utils.data.Sampler = MagicMock()
    torch_mock.utils.data.distributed = MagicMock()
    torch_mock.utils.data.distributed.DistributedSampler = MagicMock()
    
    # Install all mocks in sys.modules
    mock_modules = {
        # Core dependencies
        'torch': torch_mock,
        'torch.nn': MagicMock(),
        'torch.nn.functional': MagicMock(),
        'torch.optim': MagicMock(),
        'torch.utils': torch_mock.utils,
        'torch.utils.data': torch_mock.utils.data,
        'torch.utils.data.distributed': torch_mock.utils.data.distributed,
        
        # Progress bars and utilities
        'tqdm': MagicMock(),
        'tqdm.auto': MagicMock(),
        
        # Data processing
        'joblib': MagicMock(),
        'requests': MagicMock(),
        'pandas': MagicMock(),
        'numpy': np,  # Use real numpy for testing
        
        # NLP libraries
        'transformers': MagicMock(),
        'sentence_transformers': MagicMock(),
        'nltk': MagicMock(),
        'nltk.corpus': MagicMock(),
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
        
        # Experiment tracking
        'wandb': MagicMock(),
        'mlflow': MagicMock(),
        'tensorboard': MagicMock(),
        
        # Configuration
        'yaml': MagicMock(),
        'omegaconf': MagicMock(),
        
        # API frameworks
        'fastapi': MagicMock(),
        'uvicorn': MagicMock(),
        'gradio': MagicMock(),
        'streamlit': MagicMock(),
    }
    
    # Install mocks
    for module_name, mock_obj in mock_modules.items():
        sys.modules[module_name] = mock_obj


# ============================================================================
# Install Mocks IMMEDIATELY before any other operations
# ============================================================================
install_all_mocks()


# ============================================================================
# NOW Setup Project Path (AFTER mocks are installed)
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Pytest Configuration Hooks
# ============================================================================

def pytest_configure(config):
    """
    Configure pytest with custom markers and settings.
    Called before test collection.
    """
    # Register custom markers
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
    config.addinivalue_line(
        "markers", "experimental: Experimental tests that may fail"
    )


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    print("\n" + "="*80)
    print("AG NEWS TEXT CLASSIFICATION - TEST SUITE")
    print("="*80)
    print(f"Test session started: {session.startdir}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {sys.platform}")
    print("="*80 + "\n")


def pytest_sessionfinish(session, exitstatus):
    """
    Called after whole test run finished, right before
    returning the exit status to the system.
    """
    print("\n" + "="*80)
    print("TEST SESSION COMPLETED")
    print(f"Exit status: {exitstatus}")
    print(f"Total duration: {session.duration:.2f}s" if hasattr(session, 'duration') else "")
    print("="*80 + "\n")


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers and configure test order.
    """
    for item in items:
        # Auto-add markers based on test location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)
        
        # Skip GPU tests if CUDA not available
        if "gpu" in item.keywords:
            skip_gpu = pytest.mark.skip(reason="GPU not available in test environment")
            item.add_marker(skip_gpu)


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
    test_dir = project_root / "tests" / "fixtures" / "data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture
def temp_dir(tmp_path):
    """Provide temporary directory for file operations."""
    return tmp_path


@pytest.fixture
def sample_texts():
    """Provide sample AG News texts for testing."""
    return [
        "The stock market showed strong gains today amid positive economic data.",
        "Scientists have discovered a new method for treating cancer patients.",
        "The team won the championship game in overtime with a dramatic finish.",
        "Technology companies are investing heavily in artificial intelligence research.",
        "Global warming continues to affect weather patterns worldwide.",
        "The new smartphone features improved battery life and camera quality."
    ]


@pytest.fixture
def sample_labels():
    """Provide sample AG News labels (0: World, 1: Sports, 2: Business, 3: Sci/Tech)."""
    return [2, 3, 1, 3, 0, 3]


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
            'warmup_steps': 500,
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


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.train = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)
    model.parameters = MagicMock(return_value=[MagicMock()])
    model.state_dict = MagicMock(return_value={'layer': 'weights'})
    model.load_state_dict = MagicMock()
    model.forward = MagicMock(return_value=MagicMock(logits=MagicMock()))
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
    tokenizer.max_length = 512
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
    dataset.__getitem__ = MagicMock(side_effect=lambda idx: {
        'input_ids': [101, 1000 + idx, 2000 + idx, 3000 + idx, 102],
        'attention_mask': [1, 1, 1, 1, 1],
        'labels': idx % 4  # 4 classes for AG News
    })
    return dataset


@pytest.fixture
def mock_empty_dataset():
    """Create a mock empty dataset for testing edge cases."""
    dataset = MagicMock()
    dataset.__len__ = MagicMock(return_value=0)
    dataset.__getitem__ = MagicMock(side_effect=IndexError("Empty dataset"))
    return dataset


@pytest.fixture
def mock_config_file(temp_dir):
    """Create a mock config file."""
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
# Utility Fixtures
# ============================================================================

@pytest.fixture
def assert_shape():
    """Provide assertion helper for array shapes."""
    def _assert_shape(array, expected_shape):
        if hasattr(array, 'shape'):
            actual_shape = array.shape
        else:
            actual_shape = np.array(array).shape
        assert actual_shape == expected_shape, \
            f"Expected shape {expected_shape}, got {actual_shape}"
    return _assert_shape


@pytest.fixture
def assert_between():
    """Provide assertion helper for value ranges."""
    def _assert_between(value, min_val, max_val, inclusive=True):
        if inclusive:
            assert min_val <= value <= max_val, \
                f"Value {value} not in range [{min_val}, {max_val}]"
        else:
            assert min_val < value < max_val, \
                f"Value {value} not in range ({min_val}, {max_val})"
    return _assert_between


@pytest.fixture
def assert_close():
    """Provide assertion helper for float comparisons."""
    def _assert_close(actual, expected, rtol=1e-5, atol=1e-8):
        assert np.allclose(actual, expected, rtol=rtol, atol=atol), \
            f"Values not close: actual={actual}, expected={expected}"
    return _assert_close


@pytest.fixture
def create_random_data():
    """Factory fixture for creating random test data."""
    def _create_random_data(shape, dtype='float32', seed=None):
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
# Cleanup and Resource Management
# ============================================================================

@pytest.fixture(autouse=True)
def cleanup_after_test():
    """
    Automatically cleanup after each test.
    This runs after every test function.
    """
    # Setup phase (before test)
    yield
    # Teardown phase (after test)
    # Add cleanup code here if needed
    pass


@pytest.fixture(scope="session", autouse=True)
def cleanup_session(request):
    """
    Session-level cleanup.
    Runs once after all tests complete.
    """
    def finalizer():
        # Cleanup code here
        pass
    request.addfinalizer(finalizer)
