"""
AG News Text Classification Framework
=====================================

A state-of-the-art framework for news article classification using advanced
deep learning and natural language processing techniques.

This framework implements cutting-edge approaches including:
- Transformer-based models (DeBERTa-v3, RoBERTa, XLNet, etc.)
- Advanced ensemble methods with Bayesian optimization
- Prompt-based learning and instruction tuning
- Domain-adaptive pretraining on news corpora
- Knowledge distillation from large language models
- Efficient training strategies (LoRA/QLoRA, mixed precision)
- Comprehensive evaluation with robustness testing

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
"""

from src.__version__ import __version__, __version_info__
from src.core import registry, factory, types, exceptions
from src.utils import logging_config

# Initialize logging
logger = logging_config.setup_logging(__name__)

# Package metadata
__author__ = "Võ Hải Dũng"
__email__ = "vohaidung.work@gmail.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024, AG News Classification Research"
__status__ = "Beta"
__url__ = "https://github.com/VoHaiDung/ag-news-text-classification"

# Public API
__all__ = [
    # Version info
    "__version__",
    "__version_info__",
    "__author__",
    "__email__",
    "__license__",
    # Core modules
    "registry",
    "factory", 
    "types",
    "exceptions",
    # Utils
    "logging_config",
]

# Log package initialization
logger.info(f"AG News Text Classification Framework v{__version__} initialized")
logger.debug(f"Python package location: {__file__}")

# Check for GPU availability on import
try:
    import torch
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        logger.info(f"CUDA available with {gpu_count} GPU(s)")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.debug(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
    else:
        logger.warning("CUDA not available - running on CPU")
except ImportError:
    logger.warning("PyTorch not installed - GPU detection skipped")

# Set default configurations
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # Avoid tokenizer warnings
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")  # Reduce warnings

def get_version() -> str:
    """Get the current version of the package."""
    return __version__

def check_environment() -> dict:
    """
    Check the environment and return system information.
    
    Returns:
        dict: System and environment information
    """
    import platform
    import sys
    
    info = {
        "version": __version__,
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
    }
    
    # Check for key dependencies
    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except ImportError:
        info["transformers_version"] = "Not installed"
    
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
    except ImportError:
        info["torch_version"] = "Not installed"
        info["cuda_available"] = False
    
    return info
