"""
AG News Text Classification Framework
======================================

A comprehensive state-of-the-art framework for news article classification
using advanced deep learning and natural language processing techniques.

Project: AG News Text Classification (ag-news-text-classification)
Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
Repository: https://github.com/VoHaiDung/ag-news-text-classification

Overview:
    This framework implements cutting-edge approaches for text classification
    on the AG News dataset, achieving state-of-the-art performance (97-98%
    accuracy) through ensemble methods, large language models, and advanced
    training strategies.

Key Features:
    - 15+ transformer architectures (DeBERTa-v3, RoBERTa, XLNet, etc.)
    - Large language model integration (LLaMA 2/3, Mistral, Falcon, etc.)
    - Parameter-efficient fine-tuning (LoRA, QLoRA, Adapters)
    - Advanced ensemble methods (voting, stacking, Bayesian optimization)
    - Domain-adaptive pretraining on news corpora
    - Knowledge distillation from large language models
    - Comprehensive overfitting prevention system
    - Platform-adaptive optimization (Colab, Kaggle, Local)
    - Multi-IDE support (VSCode, PyCharm, Jupyter, Vim, etc.)
    - Production-ready deployment with monitoring

Supported Models:
    Tier 1 - SOTA XLarge Models:
        - DeBERTa-v3 XLarge/XXLarge with LoRA
        - RoBERTa Large with LoRA
        - ELECTRA Large with LoRA
        - XLNet Large with LoRA
    
    Tier 2 - Large Language Models:
        - LLaMA 2 (7B, 13B, 70B) with QLoRA
        - LLaMA 3 (8B, 70B) with QLoRA
        - Mistral (7B, 7B-Instruct) with QLoRA
        - Mixtral (8x7B) with QLoRA
        - Falcon (7B, 40B) with QLoRA
    
    Tier 3 - Ensemble Methods:
        - Soft/Hard/Weighted voting
        - Stacking with meta-learners
        - Blending and dynamic blending
        - Bayesian ensemble optimization
    
    Tier 4 - Distilled Models:
        - LLM-to-XLarge distillation
        - Ensemble distillation
        - Self-distillation
    
    Tier 5 - Platform-Optimized:
        - Colab Free/Pro optimized
        - Kaggle GPU/TPU optimized
        - Local CPU/GPU optimized

Training Strategies:
    - Mixed precision training (FP16, BF16)
    - Gradient accumulation and checkpointing
    - Curriculum learning
    - Adversarial training (FGM, PGD, FreeLB, SMART)
    - Advanced regularization (R-Drop, Mixout, SAM, EWC)
    - Multi-stage progressive training
    - Instruction tuning for LLMs

Data Augmentation:
    - Back-translation
    - Paraphrasing
    - Token replacement
    - Mixup and CutMix
    - LLM-based augmentation
    - Contrast set generation

Deployment Options:
    - Local deployment (Docker, systemd)
    - Hugging Face Spaces
    - Streamlit Cloud
    - REST API with FastAPI
    - Batch inference
    - Real-time streaming

Monitoring and Evaluation:
    - TensorBoard integration
    - MLflow experiment tracking
    - Weights & Biases support
    - Overfitting detection and prevention
    - Comprehensive metrics and visualizations
    - Error analysis and interpretability

Academic Rationale:
    This framework follows best practices from recent NLP research:
    
    1. Model Architecture:
        - He et al. (2021): "DeBERTaV3: Improving DeBERTa using ELECTRA-Style
          Pre-Training with Gradient-Disentangled Embedding Sharing"
        - Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining
          Approach"
        - Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
    
    2. Training Strategies:
        - Dettmers et al. (2023): "QLoRA: Efficient Finetuning of Quantized LLMs"
        - Miyato et al. (2018): "Virtual Adversarial Training"
        - Foret et al. (2020): "Sharpness-Aware Minimization"
    
    3. Ensemble Methods:
        - Snoek et al. (2012): "Practical Bayesian Optimization"
        - Dietterich (2000): "Ensemble Methods in Machine Learning"
    
    4. Overfitting Prevention:
        - Goodfellow et al. (2016): "Deep Learning"
        - Bishop (2006): "Pattern Recognition and Machine Learning"

Design Principles:
    1. Reproducibility: All experiments fully reproducible with seed control
    2. Modularity: Pluggable components via factory and registry patterns
    3. Extensibility: Easy to add new models, strategies, and metrics
    4. Efficiency: Parameter-efficient methods for resource-constrained environments
    5. Robustness: Comprehensive testing and overfitting prevention
    6. Usability: Clear documentation and progressive disclosure
    7. Free-tier friendly: Optimizations for Colab, Kaggle platforms

Installation:
    Basic installation:
        pip install -r requirements/base.txt
        pip install -r requirements/ml.txt
    
    Full installation with LLM support:
        pip install -r requirements/all_local.txt
    
    Platform-specific:
        pip install -r requirements/colab.txt      # For Google Colab
        pip install -r requirements/kaggle.txt     # For Kaggle
        pip install -r requirements/minimal.txt    # Minimal footprint

Quick Start:
    Automatic training (recommended for beginners):
        python src/cli.py auto-train
    
    Manual training with specific model:
        python scripts/training/train_single_model.py \\
            --config configs/models/recommended/quick_start.yaml
    
    Ensemble training:
        python scripts/training/ensemble/train_xlarge_ensemble.py
    
    Evaluation:
        python scripts/evaluation/evaluate_all_models.py
    
    Deploy local API:
        python scripts/deployment/deploy_to_local.py

Usage Examples:
    Training DeBERTa-v3 XLarge with LoRA:
        >>> from src.training.trainers import LoRATrainer
        >>> from src.models.transformers.deberta import DeBERTaV3XLargeLoRA
        >>> from src.data.datasets import AGNewsDataset
        >>> 
        >>> model = DeBERTaV3XLargeLoRA()
        >>> dataset = AGNewsDataset()
        >>> trainer = LoRATrainer(model=model, dataset=dataset)
        >>> trainer.train()
    
    Ensemble prediction:
        >>> from src.models.ensemble import SoftVotingEnsemble
        >>> from src.inference.predictors import EnsemblePredictor
        >>> 
        >>> ensemble = SoftVotingEnsemble.from_pretrained("outputs/models/ensembles/xlarge_voting")
        >>> predictor = EnsemblePredictor(ensemble)
        >>> prediction = predictor.predict("Breaking news: Tech stocks rally...")
        >>> print(prediction['label'])  # "Business"
    
    Overfitting prevention:
        >>> from src.core.overfitting_prevention import SafeTrainer
        >>> from src.core.overfitting_prevention.monitors import OverfittingDetector
        >>> 
        >>> detector = OverfittingDetector()
        >>> trainer = SafeTrainer(model=model, detector=detector)
        >>> trainer.train()  # Automatically stops if overfitting detected

Platform Detection:
    The framework automatically detects the execution environment and optimizes
    configurations accordingly:
    
        >>> from src.deployment.platform_detector import detect_platform
        >>> platform_info = detect_platform()
        >>> print(platform_info['platform'])  # 'colab_free', 'kaggle_gpu', 'local_gpu', etc.

Configuration:
    The framework uses YAML-based hierarchical configuration with 300+ preset
    configs organized by tiers. Configurations support:
    - Jinja2 templating for dynamic generation
    - Variable interpolation with OmegaConf
    - Platform-adaptive selection
    - Automatic validation and type checking

Testing:
    Run all tests:
        pytest tests/
    
    Run specific test suite:
        pytest tests/unit/models/
        pytest tests/integration/
        pytest tests/e2e/
    
    Run with coverage:
        pytest --cov=src tests/

Documentation:
    - Quick Start: docs/getting_started/quickstart.md
    - User Guide: docs/user_guide/
    - API Reference: docs/api_reference/
    - Tutorials: notebooks/01_tutorials/
    - Troubleshooting: TROUBLESHOOTING.md

Contributing:
    See CONTRIBUTING.md for guidelines on:
    - Code style and standards
    - Testing requirements
    - Documentation standards
    - Pull request process

Citation:
    If you use this framework in your research, please cite:
    
    @software{ag_news_classification_2024,
        author = {Võ Hải Dũng},
        title = {AG News Text Classification: A State-of-the-Art Framework
                 for News Article Classification},
        year = {2024},
        version = {1.0.0},
        url = {https://github.com/VoHaiDung/ag-news-text-classification},
        license = {MIT}
    }

License:
    MIT License - see LICENSE file for details

References:
    - AG News Dataset: https://huggingface.co/datasets/ag_news
    - Transformers Library: https://huggingface.co/docs/transformers
    - PyTorch: https://pytorch.org/
    - PEFT Library: https://github.com/huggingface/peft
"""

import os
import sys
import warnings
from typing import Dict, Any, Optional

# Version information
from src.__version__ import (
    __version__,
    __version_info__,
    __version_full__,
    get_version_string,
    get_version_info,
    API_VERSION,
    MILESTONES,
    COMPATIBLE_VERSIONS,
    MODEL_VERSIONS,
    PLATFORM_SUPPORT,
    PERFORMANCE_BENCHMARKS,
    CITATION,
)

# Package metadata
__author__ = "Võ Hải Dũng"
__email__ = "vohaidung.work@gmail.com"
__license__ = "MIT"
__copyright__ = "Copyright 2024, Võ Hải Dũng"
__status__ = "Production"
__url__ = "https://github.com/VoHaiDung/ag-news-text-classification"
__title__ = "AG News Text Classification"
__description__ = "State-of-the-art framework for news article classification"

# Conditional imports to avoid circular dependencies
# Core modules are imported lazily when accessed
_core_modules_loaded = False


def _lazy_import_core():
    """
    Lazy import of core modules to improve startup time.
    
    Core modules are only imported when first accessed, reducing initial
    import overhead. This is particularly important for CLI tools and
    scripts that may not need all functionality.
    """
    global _core_modules_loaded
    if _core_modules_loaded:
        return
    
    global registry, factory, types, exceptions
    from src.core import registry, factory, types, exceptions
    
    _core_modules_loaded = True


# Initialize logging subsystem
# Must be done before other imports to ensure all modules use configured logger
try:
    from src.utils import logging_config
    logger = logging_config.setup_logging(__name__)
except ImportError as e:
    # Fallback to basic logging if utils not available
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    logger.warning(f"Could not import logging_config: {e}")

# Public API
# Defines what is available when doing "from src import *"
__all__ = [
    # Metadata
    "__version__",
    "__version_info__",
    "__version_full__",
    "__author__",
    "__email__",
    "__license__",
    "__title__",
    "__description__",
    
    # Version utilities
    "get_version_string",
    "get_version_info",
    "get_version",
    "check_environment",
    "check_gpu_availability",
    
    # Core modules (lazy-loaded)
    "registry",
    "factory",
    "types",
    "exceptions",
    
    # Utils
    "logging_config",
]

# Log package initialization
logger.info(f"Initializing AG News Text Classification Framework v{__version__}")
logger.debug(f"Python package location: {__file__}")
logger.debug(f"Python version: {sys.version}")
logger.debug(f"Platform: {sys.platform}")

# Environment configuration
# Set recommended environment variables for optimal performance and stability
_env_defaults = {
    # Disable tokenizers parallelism to avoid deadlocks in multiprocessing
    "TOKENIZERS_PARALLELISM": "false",
    
    # Reduce Transformers library warnings
    "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
    
    # Set default cache directories
    "HF_HOME": os.path.join(os.path.expanduser("~"), ".cache", "huggingface"),
    "TORCH_HOME": os.path.join(os.path.expanduser("~"), ".cache", "torch"),
    
    # Disable TensorFlow if not needed (some dependencies may import it)
    "TF_CPP_MIN_LOG_LEVEL": "3",
}

for key, value in _env_defaults.items():
    os.environ.setdefault(key, value)

# Check for GPU availability
# This is done at import time to provide early feedback about hardware
_gpu_info = None
try:
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        _gpu_info = {
            "available": True,
            "count": gpu_count,
            "devices": []
        }
        
        logger.info(f"CUDA available with {gpu_count} GPU(s)")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_capability = torch.cuda.get_device_capability(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            
            device_info = {
                "id": i,
                "name": gpu_name,
                "capability": f"{gpu_capability[0]}.{gpu_capability[1]}",
                "memory_gb": round(gpu_memory, 2)
            }
            _gpu_info["devices"].append(device_info)
            
            logger.debug(
                f"  GPU {i}: {gpu_name} "
                f"(Compute {gpu_capability[0]}.{gpu_capability[1]}, "
                f"{gpu_memory:.2f} GB)"
            )
    else:
        _gpu_info = {"available": False, "count": 0, "devices": []}
        logger.warning("CUDA not available - running on CPU only")
        logger.info("For GPU support, ensure CUDA-enabled PyTorch is installed")
        logger.info("Visit: https://pytorch.org/get-started/locally/")

except ImportError:
    logger.warning("PyTorch not installed - GPU detection skipped")
    logger.info("Install PyTorch: pip install torch")
    _gpu_info = {"available": False, "count": 0, "devices": []}

except Exception as e:
    logger.error(f"Error during GPU detection: {e}")
    _gpu_info = {"available": False, "count": 0, "devices": [], "error": str(e)}

# Check for TPU availability (for Kaggle)
_tpu_info = None
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    
    _tpu_info = {
        "available": True,
        "device": xm.xla_device(),
        "ordinal": xm.get_ordinal(),
    }
    logger.info(f"TPU available: {_tpu_info['device']}")
    
except ImportError:
    _tpu_info = {"available": False}
    logger.debug("TPU not available (torch_xla not installed)")

except Exception as e:
    logger.debug(f"TPU detection failed: {e}")
    _tpu_info = {"available": False, "error": str(e)}

# Compatibility warnings
# Check Python version
if sys.version_info < (3, 8):
    logger.error(f"Python {sys.version_info.major}.{sys.version_info.minor} is not supported")
    logger.error("Minimum required version: Python 3.8")
    raise RuntimeError("Unsupported Python version")

if sys.version_info >= (3, 12):
    logger.warning(f"Python {sys.version_info.major}.{sys.version_info.minor} is not fully tested")
    logger.warning("Recommended versions: Python 3.8-3.11")
    logger.warning("Some dependencies may not be compatible")

# Check for common issues
try:
    import transformers
    transformers_version = transformers.__version__
    
    if transformers_version < "4.35.0":
        logger.warning(f"Transformers {transformers_version} is outdated")
        logger.warning("Some models may not be available")
        logger.warning("Recommended: pip install --upgrade transformers>=4.35.0")

except ImportError:
    logger.error("Transformers library not installed")
    logger.error("Install: pip install transformers>=4.35.0")

# Framework state
_framework_initialized = False


def get_version() -> str:
    """
    Get the current version of the framework.
    
    Returns:
        str: Version string in semantic versioning format
    
    Examples:
        >>> from src import get_version
        >>> get_version()
        '1.0.0'
    """
    return __version__


def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return detailed information.
    
    Returns:
        dict: GPU information including availability, count, and device details
            Keys:
                - available (bool): Whether CUDA is available
                - count (int): Number of GPUs
                - devices (list): List of GPU device information
    
    Examples:
        >>> from src import check_gpu_availability
        >>> gpu_info = check_gpu_availability()
        >>> if gpu_info['available']:
        ...     print(f"Found {gpu_info['count']} GPU(s)")
        ...     for device in gpu_info['devices']:
        ...         print(f"  {device['name']}: {device['memory_gb']} GB")
    """
    return _gpu_info.copy() if _gpu_info else {"available": False, "count": 0, "devices": []}


def check_environment() -> Dict[str, Any]:
    """
    Check the execution environment and return comprehensive system information.
    
    Gathers information about:
    - Python version and implementation
    - Operating system and platform
    - Installed dependencies and versions
    - GPU/TPU availability
    - Framework configuration
    
    Returns:
        dict: Complete environment information for debugging and support
            Keys include: version, python_version, platform, processor,
            dependencies (torch, transformers, etc.), gpu_info, tpu_info
    
    Examples:
        >>> from src import check_environment
        >>> env = check_environment()
        >>> print(f"Framework: {env['version']}")
        >>> print(f"PyTorch: {env['torch_version']}")
        >>> print(f"GPU: {env['cuda_available']}")
    
    See Also:
        - scripts/setup/verify_installation.py: Full environment validation
        - src/core/health/health_checker.py: Health check system
    """
    import platform
    
    info = {
        # Framework information
        "framework": __title__,
        "version": __version__,
        "api_version": API_VERSION,
        
        # Python environment
        "python_version": sys.version,
        "python_version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "python_implementation": platform.python_implementation(),
        "python_compiler": platform.python_compiler(),
        
        # System information
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "architecture": platform.architecture()[0],
        
        # Hardware information
        "gpu_info": _gpu_info,
        "tpu_info": _tpu_info,
    }
    
    # Check key dependencies
    dependencies = {}
    
    for package in ["torch", "transformers", "datasets", "accelerate", "peft", "bitsandbytes"]:
        try:
            module = __import__(package)
            dependencies[package] = {
                "installed": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except ImportError:
            dependencies[package] = {
                "installed": False,
                "version": None,
            }
    
    info["dependencies"] = dependencies
    
    # CUDA information (if PyTorch available)
    if dependencies["torch"]["installed"]:
        try:
            import torch
            info["cuda_available"] = torch.cuda.is_available()
            info["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else None
            info["cudnn_version"] = torch.backends.cudnn.version() if torch.cuda.is_available() else None
            info["torch_version"] = torch.__version__
        except Exception as e:
            logger.warning(f"Error getting CUDA information: {e}")
            info["cuda_available"] = False
    else:
        info["torch_version"] = "Not installed"
        info["cuda_available"] = False
    
    return info


def print_system_info():
    """
    Print formatted system information to console.
    
    Displays comprehensive environment details in a human-readable format.
    Useful for debugging and support tickets.
    
    Examples:
        >>> from src import print_system_info
        >>> print_system_info()
        AG News Text Classification Framework v1.0.0
        ============================================
        Python: 3.10.12
        Platform: Linux-5.15.0-1034-gcp-x86_64
        PyTorch: 2.1.0
        CUDA: Available (1 GPU)
        ...
    """
    env = check_environment()
    
    print(f"\n{__title__} v{__version__}")
    print("=" * 60)
    print(f"Python: {env['python_version_info']['major']}.{env['python_version_info']['minor']}.{env['python_version_info']['micro']}")
    print(f"Platform: {env['platform']}")
    print(f"System: {env['system']} ({env['architecture']})")
    
    print("\nDependencies:")
    for pkg, info_dict in env["dependencies"].items():
        if info_dict["installed"]:
            print(f"  {pkg}: {info_dict['version']}")
        else:
            print(f"  {pkg}: Not installed")
    
    print("\nHardware:")
    if env["cuda_available"]:
        gpu_count = env["gpu_info"]["count"]
        print(f"  CUDA: Available ({gpu_count} GPU{'s' if gpu_count > 1 else ''})")
        if env["cuda_version"]:
            print(f"  CUDA Version: {env['cuda_version']}")
        for device in env["gpu_info"]["devices"]:
            print(f"    GPU {device['id']}: {device['name']} ({device['memory_gb']} GB)")
    else:
        print("  CUDA: Not available")
    
    if env["tpu_info"]["available"]:
        print(f"  TPU: Available ({env['tpu_info']['device']})")
    
    print()


# Framework initialization
def initialize_framework(config: Optional[Dict[str, Any]] = None):
    """
    Initialize the framework with optional configuration.
    
    Performs one-time setup operations:
    - Load core modules
    - Configure logging
    - Set up caching
    - Initialize registries
    - Validate environment
    
    Args:
        config (dict, optional): Framework configuration options
            Supported keys:
                - log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
                - cache_dir (str): Directory for caching models and data
                - device (str): Default device ('cuda', 'cpu', 'auto')
                - seed (int): Random seed for reproducibility
    
    Examples:
        >>> from src import initialize_framework
        >>> initialize_framework({
        ...     'log_level': 'INFO',
        ...     'device': 'cuda',
        ...     'seed': 42
        ... })
    """
    global _framework_initialized
    
    if _framework_initialized:
        logger.debug("Framework already initialized")
        return
    
    logger.info("Initializing AG News Text Classification Framework")
    
    if config is None:
        config = {}
    
    # Load core modules
    _lazy_import_core()
    
    # Set random seeds for reproducibility
    if "seed" in config:
        from src.utils.reproducibility import set_seed
        set_seed(config["seed"])
        logger.info(f"Random seed set to {config['seed']}")
    
    # Configure device
    if "device" in config:
        device = config["device"]
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Default device set to {device}")
    
    # Set cache directory
    if "cache_dir" in config:
        cache_dir = config["cache_dir"]
        os.environ["HF_HOME"] = cache_dir
        logger.info(f"Cache directory set to {cache_dir}")
    
    # Set log level
    if "log_level" in config:
        log_level = config["log_level"]
        logging_config.set_log_level(log_level)
        logger.info(f"Log level set to {log_level}")
    
    _framework_initialized = True
    logger.info("Framework initialization complete")


# Module-level initialization hook
def _on_import():
    """
    Perform lightweight initialization on module import.
    
    This function is called automatically when the module is first imported.
    It performs minimal setup to avoid slowing down import time.
    """
    logger.debug(f"AG News Text Classification Framework v{__version__} loaded")
    
    # Check for common misconfigurations
    if "TOKENIZERS_PARALLELISM" not in os.environ:
        logger.debug("Setting TOKENIZERS_PARALLELISM=false")
    
    # Detect platform
    try:
        from src.deployment.platform_detector import detect_platform
        platform_info = detect_platform()
        logger.debug(f"Platform detected: {platform_info.get('platform', 'unknown')}")
    except Exception as e:
        logger.debug(f"Platform detection failed: {e}")


# Call import hook
_on_import()

# Cleanup
del warnings  # Don't expose warnings module

# Final initialization message
logger.debug(f"Module {__name__} initialization complete")
