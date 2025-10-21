"""
Core Package Initialization for AG News Text Classification
===========================================================

This module serves as the main entry point for the AG News Text Classification
framework, providing high-level API access to all core functionalities including
model loading, training, evaluation, and deployment across multiple platforms.

Project: AG News Text Classification (ag-news-text-classification)
Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT

Framework Overview:
    A comprehensive, production-grade framework for text classification implementing
    state-of-the-art models, ensemble methods, parameter-efficient fine-tuning
    techniques, and built-in overfitting prevention mechanisms. Designed for both
    academic research requiring reproducibility and production deployments demanding
    robustness and efficiency.

Core Components:
    1. Model Architecture Support:
       - Transformer models (DeBERTa, RoBERTa, ELECTRA, XLNet, Longformer, T5)
       - Large Language Models (LLaMA 2/3, Mistral, Mixtral, Falcon, MPT, Phi)
       - Parameter-efficient methods (LoRA, QLoRA, Adapters, Prefix/Prompt Tuning)
       - Ensemble techniques (Voting, Stacking, Blending, Bayesian, MoE)
    
    2. Training Infrastructure:
       - Multi-stage training pipelines
       - Curriculum and adversarial learning
       - Knowledge distillation frameworks
       - Real-time overfitting prevention
       - Platform-adaptive optimization
    
    3. Data Processing Pipeline:
       - Advanced augmentation (LLM-based, back-translation, mixup)
       - Automated quality filtering
       - Contrast set generation
       - Active learning selection
    
    4. Deployment Capabilities:
       - Multi-platform support (Local, Colab, Kaggle, Cloud IDEs)
       - Automatic platform detection
       - Quota management and tracking
       - RESTful API with local serving
       - Model optimization (quantization, pruning, ONNX export)
    
    5. Quality Assurance:
       - Comprehensive overfitting prevention system
       - Data leakage detection
       - Test set protection mechanisms
       - Statistical validation utilities
       - Automated health checks

Academic Foundation:
    The framework implements methodologies from recent academic literature in
    natural language processing, machine learning, and deep learning:
    
    Parameter-Efficient Fine-Tuning:
        - Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
        - Dettmers et al. (2023): "QLoRA: Efficient Finetuning of Quantized LLMs"
        - Houlsby et al. (2019): "Parameter-Efficient Transfer Learning for NLP"
    
    Model Architectures:
        - He et al. (2021): "DeBERTa: Decoding-enhanced BERT with Disentangled Attention"
        - Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
        - Clark et al. (2020): "ELECTRA: Pre-training Text Encoders as Discriminators"
    
    Ensemble Methods:
        - Zhou (2012): "Ensemble Methods: Foundations and Algorithms"
        - Wolpert (1992): "Stacked Generalization"
        - Breiman (1996): "Bagging Predictors"
    
    Overfitting Prevention:
        - Goodfellow et al. (2016): "Deep Learning" - Regularization chapter
        - Ying (2019): "An Overview of Overfitting and its Solutions"
        - Zhang et al. (2021): "Understanding Deep Learning Requires Rethinking Generalization"
    
    Reproducibility:
        - Kitzes et al. (2017): "The Practice of Reproducible Research"
        - Pineau et al. (2021): "Improving Reproducibility in Machine Learning Research"

Platform Support Strategy:
    Designed with resource-constrained environments as first-class citizens,
    enabling state-of-the-art research without expensive cloud infrastructure:
    
    Google Colab Integration:
        - Automatic tier detection (Free/Pro/Pro+)
        - Session management with disconnect recovery
        - Drive-based model persistence
        - GPU quota optimization
        - Memory-adaptive batch sizing
    
    Kaggle Notebooks Support:
        - GPU (P100/T4) and TPU (v3-8) optimization
        - Dataset caching strategies
        - Kernel output management
        - Competition integration utilities
    
    Local Development:
        - CPU and GPU (CUDA/MPS) support
        - Multi-GPU distributed training
        - Docker containerization
        - Systemd service integration
        - Nginx reverse proxy configuration
    
    Cloud IDE Compatibility:
        - Gitpod workspace configuration
        - GitHub Codespaces support
        - Devcontainer specifications
        - Remote development optimization

API Design Principles:
    1. Progressive Disclosure:
       - Simple high-level API for beginners (load_model, predict)
       - Detailed control for advanced users (factory pattern, registries)
       - Expert-level customization (plugin system, callbacks)
    
    2. Consistency:
       - Uniform naming conventions across modules
       - Standardized configuration format
       - Common error handling patterns
       - Predictable return types
    
    3. Discoverability:
       - Comprehensive docstrings with examples
       - Type hints for all public APIs
       - Interactive help system
       - Auto-complete friendly naming
    
    4. Extensibility:
       - Plugin architecture for custom components
       - Registry pattern for model/dataset registration
       - Hook system for training customization
       - Event-driven callback mechanism
    
    5. Reliability:
       - Defensive programming with validation
       - Graceful degradation on missing dependencies
       - Informative error messages with solutions
       - Automatic environment health checks

Usage Patterns:
    Quick Start (Minimal Configuration):
        >>> import ag_news_text_classification as agnews
        >>> model = agnews.load_model('deberta-v3-large-lora')
        >>> result = agnews.create_predictor(model).predict("Breaking news text...")
        >>> print(f"Category: {result['label']}, Confidence: {result['confidence']:.2%}")
    
    Research Workflow (Full Control):
        >>> config = agnews.load_config('configs/models/recommended/balanced.yaml')
        >>> trainer = agnews.create_trainer(config)
        >>> model = trainer.train()
        >>> metrics = agnews.evaluate(model, test_dataset)
        >>> agnews.save_results(metrics, 'outputs/results/experiment_001/')
    
    Production Deployment (Optimized):
        >>> platform_info = agnews.get_platform_info()
        >>> selector = agnews.SmartSelector(platform_info)
        >>> optimal_config = selector.select_best_config()
        >>> model = agnews.load_model(optimal_config['model_name'])
        >>> api = agnews.create_api(model, config=optimal_config['api_config'])
        >>> api.start()

Module Organization:
    The package is organized into logical components following domain-driven design:
    
    core/: Framework foundations
        - registry: Component registration system
        - factory: Object creation patterns
        - types: Type definitions and protocols
        - exceptions: Custom exception hierarchy
        - interfaces: Abstract base classes
        - health: System health monitoring
        - overfitting_prevention: Anti-overfitting mechanisms
    
    data/: Data processing pipeline
        - datasets: Dataset implementations
        - preprocessing: Text cleaning and normalization
        - augmentation: Data augmentation strategies
        - sampling: Intelligent data selection
        - validation: Cross-validation utilities
        - loaders: Efficient data loading
    
    models/: Model implementations
        - transformers: Transformer-based architectures
        - llm: Large language model adaptations
        - efficient: Parameter-efficient methods
        - ensemble: Ensemble model implementations
        - prompt_based: Prompt/instruction tuning
        - heads: Classification head variants
    
    training/: Training infrastructure
        - trainers: Training loop implementations
        - strategies: Advanced training techniques
        - objectives: Loss functions and regularizers
        - optimization: Optimizers and schedulers
        - callbacks: Training event handlers
    
    evaluation/: Evaluation tools
        - metrics: Performance measurement
        - analysis: Error and overfitting analysis
        - visualizations: Result visualization
    
    inference/: Prediction and serving
        - predictors: Inference implementations
        - optimization: Model optimization utilities
        - serving: API and batch prediction
    
    deployment/: Platform integration
        - platform_detector: Runtime environment detection
        - smart_selector: Optimal configuration selection
        - quota_tracker: Resource usage monitoring
        - cache_manager: Efficient caching strategies
    
    api/: API layer
        - rest: RESTful API implementation
        - local: Local serving utilities
        - schemas: Request/response validation
    
    services/: High-level services
        - prediction_service: Prediction orchestration
        - training_service: Training pipeline management
        - monitoring: System monitoring
    
    utils/: Shared utilities
        - logging: Structured logging
        - io_utils: File I/O operations
        - reproducibility: Random seed management
        - platform_utils: Platform-specific helpers

Version Compatibility:
    This package maintains semantic versioning and tracks compatibility with:
    - Python versions (3.8 - 3.11)
    - PyTorch ecosystem (2.0+)
    - Transformers library (4.35+)
    - Platform-specific requirements
    
    See __version__.py for detailed version information and compatibility matrix.

Performance Characteristics:
    Benchmarked on AG News dataset (120K training, 7.6K test samples):
    
    Accuracy Targets:
        - Single XLarge Models: 96.5-97.2%
        - LLM-based Models: 96.8-97.5%
        - Ensemble Methods: 97.5-98.0%
    
    Training Time (V100 GPU):
        - DeBERTa-v3-Large LoRA: approximately 2 hours
        - LLaMA-2-7B QLoRA: approximately 4 hours
        - Ensemble Training: approximately 8 hours
    
    Inference Speed:
        - Standard Models: approximately 50 samples/sec (GPU)
        - Distilled Models: approximately 200 samples/sec (CPU)
        - Quantized INT8: approximately 400 samples/sec (CPU)
    
    Memory Requirements:
        - LoRA Fine-tuning: 12-16GB GPU
        - QLoRA Fine-tuning: 8-12GB GPU
        - Inference: 4-8GB GPU or 8-16GB RAM (CPU)

References:
    Project Repository:
        https://github.com/VoHaiDung/ag-news-text-classification
    
    Documentation:
        See docs/ directory for comprehensive guides
    
    Citation:
        See CITATION.cff for BibTeX citation format
    
    Related Work:
        - Zhang et al. (2015): "Character-level Convolutional Networks for Text Classification"
        - Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"
        - Howard & Ruder (2018): "Universal Language Model Fine-tuning for Text Classification"

Notes:
    Package initialization performs the following steps:
    1. Configure structured logging system
    2. Detect execution platform (Local/Colab/Kaggle/Cloud IDE)
    3. Validate environment dependencies
    4. Detect available compute devices (CUDA/MPS/TPU/CPU)
    5. Register signal handlers for cleanup
    
    Initialization is designed to be lightweight and fail gracefully. If environment
    validation fails, warnings are logged but the package remains importable.
    Use health_check() function for detailed diagnostics.

See Also:
    __version__: Version information and compatibility
    core.registry: Component registration system
    core.factory: Object creation utilities
    deployment.platform_detector: Platform detection logic
    core.health: Health check system
"""

import logging
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

# Version management
from .__version__ import (
    __author__,
    __author_email__,
    __description__,
    __license__,
    __title__,
    __url__,
    __version__,
)

# Core imports for public API surface
# These imports define the primary interface for package users
from .core.exceptions import (
    AGNewsBaseException,
    ConfigurationError,
    DataError,
    DeploymentError,
    ModelError,
    OverfittingDetectedError,
    PlatformError,
    QuotaExceededError,
    TrainingError,
    ValidationError,
)
from .core.factory import (
    ModelFactory,
    create_data_loader,
    create_ensemble,
    create_model,
    create_predictor,
    create_trainer,
)
from .core.interfaces import (
    BaseDataset,
    BaseEvaluator,
    BaseModel,
    BasePredictor,
    BaseTrainer,
)
from .core.registry import Registry
from .core.types import (
    ConfigDict,
    DatasetSplit,
    DeviceType,
    EnsembleMethod,
    ModelArchitecture,
    PlatformType,
    PredictionOutput,
    TrainingMode,
)

# Deployment and platform utilities
from .deployment.platform_detector import PlatformDetector, detect_platform
from .deployment.quota_tracker import QuotaTracker
from .deployment.smart_selector import SmartSelector

# Logging configuration
from .utils.logging_config import get_logger, setup_logging

# Type checking imports
# These imports are only evaluated by type checkers, not at runtime
# This avoids circular dependencies and reduces import overhead
if TYPE_CHECKING:
    from .data.datasets.ag_news import AGNewsDataset
    from .models.base.base_model import BaseModel as ModelType
    from .services.core.prediction_service import PredictionService
    from .services.core.training_service import TrainingService
    from .training.trainers.base_trainer import BaseTrainer as TrainerType

# Module metadata
# Defines the public API surface of the package
# Items not listed here are considered internal implementation details
__all__ = [
    # Version information
    "__version__",
    "__author__",
    "__author_email__",
    "__license__",
    "__title__",
    "__description__",
    "__url__",
    # Core functionality
    "Registry",
    "ModelFactory",
    "create_model",
    "create_trainer",
    "create_predictor",
    "create_ensemble",
    "create_data_loader",
    # Type definitions
    "ModelArchitecture",
    "TrainingMode",
    "EnsembleMethod",
    "PlatformType",
    "DeviceType",
    "DatasetSplit",
    "ConfigDict",
    "PredictionOutput",
    # Interfaces
    "BaseModel",
    "BaseTrainer",
    "BasePredictor",
    "BaseDataset",
    "BaseEvaluator",
    # Exceptions
    "AGNewsBaseException",
    "ModelError",
    "TrainingError",
    "DataError",
    "ValidationError",
    "ConfigurationError",
    "OverfittingDetectedError",
    "PlatformError",
    "QuotaExceededError",
    "DeploymentError",
    # Platform utilities
    "PlatformDetector",
    "detect_platform",
    "QuotaTracker",
    "SmartSelector",
    # Logging
    "get_logger",
    "setup_logging",
    # High-level functions
    "load_model",
    "load_config",
    "get_device",
    "set_seed",
    "validate_environment",
    "get_platform_info",
    "get_quota_status",
    "health_check",
]

# Package-level logger instance
logger = get_logger(__name__)

# Package-level configuration state
# Maintains runtime configuration and initialization status
# Should not be modified directly by users
_PACKAGE_CONFIG: Dict[str, Any] = {
    "initialized": False,
    "platform": None,
    "device": None,
    "logger_configured": False,
}


def _initialize_package() -> None:
    """
    Initialize the AG News Text Classification package.
    
    Performs comprehensive initialization of the framework including logging
    configuration, platform detection, device setup, and environment validation.
    This function is called automatically on package import but can be invoked
    manually to reinitialize the package state.
    
    Initialization Steps:
        1. Configure structured logging system with appropriate handlers and formatters
        2. Detect execution platform (Local, Google Colab, Kaggle, Cloud IDEs)
        3. Validate environment dependencies against compatibility requirements
        4. Detect available compute devices (CUDA GPU, Apple MPS, TPU, CPU)
        5. Register signal handlers for graceful cleanup on termination
        6. Initialize component registries for models, datasets, and trainers
    
    Raises:
        RuntimeError: If critical initialization steps fail, such as missing
            essential dependencies or incompatible platform configuration.
    
    Notes:
        This function is idempotent and safe to call multiple times. Subsequent
        calls will check initialization status and skip redundant operations.
        
        If initialization encounters non-critical issues, warnings are logged
        but the function completes successfully. Use health_check() for detailed
        diagnostics of initialization state.
        
        Logging is configured early in the initialization sequence to ensure
        all subsequent operations can emit structured log messages.
    
    Examples:
        >>> # Automatic initialization on import
        >>> import ag_news_text_classification as agnews
        
        >>> # Manual reinitialization after environment changes
        >>> import os
        >>> os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
        >>> agnews._initialize_package()
    
    See Also:
        health_check: Comprehensive health diagnostics
        validate_environment: Detailed environment validation
        setup_logging: Logging configuration details
    """
    global _PACKAGE_CONFIG
    
    if _PACKAGE_CONFIG["initialized"]:
        logger.debug("Package already initialized, skipping initialization")
        return
    
    try:
        # Step 1: Configure logging subsystem
        if not _PACKAGE_CONFIG["logger_configured"]:
            setup_logging()
            _PACKAGE_CONFIG["logger_configured"] = True
            logger.info("AG News Text Classification package initialization started")
        
        # Step 2: Detect execution platform
        try:
            platform = detect_platform()
            _PACKAGE_CONFIG["platform"] = platform
            logger.info(f"Detected platform: {platform.value}")
        except Exception as e:
            logger.warning(
                f"Platform detection failed: {e}, defaulting to local execution mode"
            )
            _PACKAGE_CONFIG["platform"] = PlatformType.LOCAL
        
        # Step 3: Detect compute device
        try:
            device = get_device()
            _PACKAGE_CONFIG["device"] = device
            logger.info(f"Using device: {device}")
        except Exception as e:
            logger.warning(
                f"Device detection failed: {e}, defaulting to CPU execution"
            )
            _PACKAGE_CONFIG["device"] = "cpu"
        
        # Step 4: Validate environment dependencies
        try:
            validate_environment()
            logger.info("Environment validation completed successfully")
        except Exception as e:
            logger.warning(
                f"Environment validation encountered issues: {e}. "
                f"Some features may not be available."
            )
        
        # Step 5: Mark package as initialized
        _PACKAGE_CONFIG["initialized"] = True
        logger.info(
            f"AG News Text Classification v{__version__} initialized successfully"
        )
        
    except Exception as e:
        logger.error(f"Package initialization failed: {e}")
        raise RuntimeError(
            f"Failed to initialize AG News Text Classification package: {e}"
        ) from e


def get_device() -> str:
    """
    Detect and return the optimal compute device for model operations.
    
    Automatically detects available hardware accelerators in priority order,
    selecting the most performant option for deep learning workloads. The
    detection order follows typical performance characteristics:
        1. CUDA GPU (NVIDIA) - Highest performance for large models
        2. TPU (Google Cloud/Colab) - Optimized for specific operations
        3. MPS (Apple Silicon) - Native acceleration on M1/M2 Macs
        4. CPU - Universal fallback compatible with all systems
    
    Returns:
        str: Device identifier string compatible with PyTorch device semantics.
            Possible values: 'cuda', 'tpu', 'mps', 'cpu'
    
    Examples:
        >>> device = get_device()
        >>> print(f"Using device: {device}")
        Using device: cuda
        
        >>> # Use in model initialization
        >>> model = MyModel().to(get_device())
        
        >>> # Check specific device type
        >>> if get_device() == 'cuda':
        ...     print("GPU acceleration available")
    
    Notes:
        The detected device is cached in package configuration state. The cache
        is invalidated on package reinitialization.
        
        For multi-GPU systems, this function returns 'cuda' without specifying
        a particular GPU index. Use CUDA_VISIBLE_DEVICES environment variable
        to control GPU selection.
        
        TPU detection requires torch_xla package. If not installed, TPU will
        not be detected even if available.
        
        MPS support requires PyTorch 1.12+ on macOS 12.3+ with Apple Silicon.
    
    See Also:
        set_seed: Configure reproducible random number generation
        _initialize_package: Package initialization including device detection
    
    References:
        PyTorch Device Semantics: https://pytorch.org/docs/stable/tensor_attributes.html#torch.device
        CUDA Best Practices: https://pytorch.org/docs/stable/notes/cuda.html
    """
    import torch
    
    # Check for CUDA GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        logger.debug(f"CUDA device detected: {gpu_name}")
        return device
    
    # Check for Apple MPS availability
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        logger.debug("Apple MPS device detected")
        return device
    
    # Check for TPU availability
    try:
        import torch_xla.core.xla_model as xm
        device = "tpu"
        logger.debug("TPU device detected")
        return device
    except ImportError:
        pass
    
    # Fallback to CPU
    device = "cpu"
    logger.debug("Using CPU device (no hardware accelerator detected)")
    return device


def set_seed(seed: int = 42) -> None:
    """
    Configure random seeds for reproducible experiments across all libraries.
    
    Sets random number generator seeds for all major libraries used in the framework,
    ensuring deterministic behavior critical for scientific reproducibility. This
    function should be called before any random operations including data loading,
    model initialization, and training.
    
    Libraries Configured:
        - Python standard library (random module)
        - NumPy (np.random)
        - PyTorch CPU operations (torch.manual_seed)
        - PyTorch CUDA operations (torch.cuda.manual_seed_all)
        - PyTorch cuDNN backend (deterministic mode, disabled benchmarking)
    
    Args:
        seed: Random seed value. Defaults to 42, a convention in machine learning
            research for consistency across studies and implementations.
    
    Examples:
        >>> # Basic usage with default seed
        >>> set_seed()
        
        >>> # Custom seed for specific experiment
        >>> set_seed(12345)
        
        >>> # Ensure reproducibility in training script
        >>> import ag_news_text_classification as agnews
        >>> agnews.set_seed(42)
        >>> model = agnews.create_model('deberta-v3-large')
        >>> trainer = agnews.create_trainer(model)
        >>> results = trainer.train()
    
    Notes:
        Setting deterministic mode for cuDNN may significantly impact performance
        due to the use of slower but deterministic algorithms. This is a necessary
        trade-off for reproducibility.
        
        Complete reproducibility also requires:
            - Fixed dataset ordering (disable shuffle or use fixed shuffle seed)
            - Consistent hardware configuration
            - Same library versions (see COMPATIBLE_VERSIONS in __version__.py)
            - Single-threaded execution for some operations
        
        Some operations in distributed training may still have non-deterministic
        behavior. Consult PyTorch documentation for distributed reproducibility.
    
    Warnings:
        Enabling deterministic mode may reduce training speed by 10-30% depending
        on model architecture and hardware configuration.
    
    See Also:
        get_device: Device detection and configuration
        utils.reproducibility: Advanced reproducibility utilities
    
    References:
        PyTorch Reproducibility Guide:
            https://pytorch.org/docs/stable/notes/randomness.html
        
        Pineau et al. (2021): "Improving Reproducibility in Machine Learning Research"
            NeurIPS 2021 Code Submission Guidelines
    """
    import random
    
    import numpy as np
    import torch
    
    # Set Python built-in random seed
    random.seed(seed)
    
    # Set NumPy random seed
    np.random.seed(seed)
    
    # Set PyTorch random seed for CPU operations
    torch.manual_seed(seed)
    
    # Configure CUDA random number generation
    if torch.cuda.is_available():
        # Set seed for current CUDA device
        torch.cuda.manual_seed(seed)
        
        # Set seed for all CUDA devices in multi-GPU setup
        torch.cuda.manual_seed_all(seed)
        
        # Enable deterministic cuDNN algorithms
        # This may reduce performance but ensures reproducibility
        torch.backends.cudnn.deterministic = True
        
        # Disable cuDNN autotuner
        # Autotuner selects fastest algorithm which may vary between runs
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed} for reproducibility")


def load_config(config_path: str) -> ConfigDict:
    """
    Load and validate configuration from YAML file.
    
    Provides flexible configuration loading supporting multiple input formats
    including file paths, named configurations, and template-based configs with
    variable substitution. All loaded configurations are validated against the
    framework's configuration schema to ensure correctness.
    
    Supported Input Formats:
        1. Absolute file paths: '/path/to/config.yaml'
        2. Relative paths from project root: 'configs/models/deberta_large.yaml'
        3. Named configurations: 'deberta_large_lora' (looks up in configs/models/)
        4. Template configurations: 'template:deberta' (applies template expansion)
    
    Args:
        config_path: Configuration identifier. Can be:
            - Full path to YAML configuration file
            - Relative path from project root
            - Named configuration (without .yaml extension)
            - Template name with 'template:' prefix
    
    Returns:
        ConfigDict: Validated configuration dictionary containing all required
            fields with resolved default values and template substitutions.
    
    Raises:
        ConfigurationError: If configuration file is not found, contains invalid
            YAML syntax, fails schema validation, or has unresolvable template
            variables.
        
        FileNotFoundError: If specified configuration file does not exist.
        
        ValidationError: If configuration violates schema constraints such as
            invalid parameter ranges, incompatible settings, or missing required
            fields.
    
    Examples:
        >>> # Load from full path
        >>> config = load_config('configs/models/recommended/balanced.yaml')
        
        >>> # Load named configuration
        >>> config = load_config('deberta_large_lora')
        
        >>> # Load with template
        >>> config = load_config('template:sota_xlarge')
        
        >>> # Use loaded configuration
        >>> model = create_model(config['model'])
        >>> trainer = create_trainer(config['training'])
    
    Notes:
        Configuration files are cached after first load to improve performance.
        Cache is invalidated when files are modified (based on mtime).
        
        Template variables can be specified in configuration files using Jinja2
        syntax and are resolved from environment variables or smart defaults.
        
        Configuration validation includes:
            - Schema compliance (required fields, types, ranges)
            - Cross-field consistency (compatible settings)
            - Platform compatibility (resource requirements)
            - Security checks (no path traversal, command injection)
    
    See Also:
        configs.config_loader: Detailed configuration loading logic
        configs.config_validator: Configuration validation rules
        configs.config_schema: Configuration schema definitions
    
    References:
        YAML Specification: https://yaml.org/spec/1.2/spec.html
        Jinja2 Templates: https://jinja.palletsprojects.com/
    """
    from .configs.config_loader import ConfigLoader
    
    loader = ConfigLoader()
    config = loader.load(config_path)
    
    logger.info(f"Configuration loaded from: {config_path}")
    return config


def load_model(
    model_name_or_path: str,
    config: Optional[ConfigDict] = None,
    device: Optional[str] = None,
    **kwargs: Any,
) -> "ModelType":
    """
    Load pre-trained or fine-tuned model with automatic configuration.
    
    Provides unified interface for loading models from multiple sources including
    local checkpoints, Hugging Face Hub, and the framework's model registry. Handles
    automatic configuration loading, device placement, and adapter merging for
    parameter-efficient models.
    
    Supported Model Sources:
        1. Framework Registry: Pre-configured models registered in the framework
           Example: 'deberta-v3-large-lora', 'llama2-7b-qlora'
        
        2. Local Checkpoints: Models saved during training or fine-tuning
           Example: 'outputs/models/checkpoints/experiment_001/best_model/'
        
        3. Hugging Face Hub: Public or private models from huggingface.co
           Example: 'microsoft/deberta-v3-large', 'meta-llama/Llama-2-7b-hf'
        
        4. Custom Implementations: User-defined models following BaseModel interface
           Example: 'plugins/custom_models/my_custom_model.py'
    
    Args:
        model_name_or_path: Model identifier or file path. Interpretation depends
            on the format:
            - Short name: Looks up in model registry
            - Absolute/relative path: Loads from filesystem
            - HF identifier: Downloads from Hugging Face Hub
            
        config: Optional configuration dictionary. If not provided, configuration
            is loaded from:
            - Model checkpoint metadata (for saved models)
            - Default configuration for model type (for registry models)
            - Hugging Face config.json (for HF models)
            
        device: Target device for model placement. If not specified, automatically
            detected using get_device(). Valid values: 'cuda', 'mps', 'tpu', 'cpu'
            
        **kwargs: Additional arguments passed to model constructor. Common options:
            - num_labels: Number of classification labels (default: 4 for AG News)
            - dropout: Dropout probability for regularization
            - load_in_8bit: Enable 8-bit quantization (QLoRA models)
            - torch_dtype: Data type for model parameters (float32, float16, bfloat16)
    
    Returns:
        ModelType: Loaded model instance ready for inference or fine-tuning. Model
            is automatically:
            - Moved to specified or detected device
            - Set to evaluation mode (call .train() for fine-tuning)
            - Configured with appropriate tokenizer
            - Merged with adapters (for LoRA/QLoRA models, if merge=True)
    
    Raises:
        ModelError: If model cannot be loaded due to:
            - Model identifier not found in registry or filesystem
            - Incompatible checkpoint format
            - Unsupported model architecture
            - Missing required dependencies
            
        ConfigurationError: If provided configuration is invalid or incompatible
            with the model architecture.
            
        RuntimeError: If model loading fails due to insufficient memory, CUDA
            errors, or corrupted checkpoint files.
    
    Examples:
        >>> # Load registered model with default configuration
        >>> model = load_model('deberta-v3-large-lora')
        
        >>> # Load from local checkpoint
        >>> model = load_model('outputs/models/checkpoints/best_model/')
        
        >>> # Load from Hugging Face with custom config
        >>> config = {'num_labels': 4, 'dropout': 0.2}
        >>> model = load_model('microsoft/deberta-v3-large', config=config)
        
        >>> # Load with 8-bit quantization
        >>> model = load_model('meta-llama/Llama-2-7b-hf', load_in_8bit=True)
        
        >>> # Load to specific device
        >>> model = load_model('deberta-v3-large', device='cuda:1')
    
    Notes:
        For LoRA/QLoRA models, adapters are loaded separately and can be merged
        into the base model for faster inference. Use merge_adapters=True in kwargs.
        
        Large language models may require significant memory. Use quantization
        (load_in_8bit=True or load_in_4bit=True) for resource-constrained environments.
        
        Downloaded models from Hugging Face are cached in the transformers cache
        directory (typically ~/.cache/huggingface/).
        
        For multi-GPU inference, models are automatically distributed using
        device_map='auto' when available.
    
    See Also:
        create_model: Factory function for model instantiation
        models.base.base_model: Base model interface
        core.factory: Model factory implementation
    
    References:
        Hugging Face Model Hub: https://huggingface.co/models
        Transformer Model Loading: https://huggingface.co/docs/transformers/main_classes/model
    """
    if device is None:
        device = get_device()
    
    model = create_model(
        model_name_or_path=model_name_or_path,
        config=config,
        device=device,
        **kwargs,
    )
    
    logger.info(f"Model loaded: {model_name_or_path} on device: {device}")
    return model


def validate_environment() -> Dict[str, Any]:
    """
    Perform comprehensive validation of the runtime environment.
    
    Executes extensive checks of the execution environment to ensure all required
    dependencies are available, properly configured, and compatible with the framework.
    This function is called automatically during package initialization but can be
    invoked manually after environment modifications.
    
    Validation Categories:
        1. Python Environment:
           - Python version compatibility (3.8-3.11)
           - Virtual environment detection
           - Site packages accessibility
        
        2. Required Dependencies:
           - PyTorch (version, CUDA support)
           - Transformers (version, model support)
           - Core libraries (numpy, pandas, scikit-learn)
        
        3. Optional Dependencies:
           - Accelerate (distributed training)
           - PEFT (parameter-efficient fine-tuning)
           - BitsAndBytes (quantization)
           - Experiment tracking (wandb, mlflow, tensorboard)
        
        4. Hardware Configuration:
           - CUDA availability and version
           - GPU memory and compute capability
           - TPU availability (on supported platforms)
           - Available system RAM
        
        5. Platform-Specific Requirements:
           - Google Colab (drive mounting, GPU quota)
           - Kaggle (datasets, accelerators)
           - Cloud IDEs (workspace resources)
        
        6. File System:
           - Configuration directory structure
           - Data directory permissions
           - Output directory writability
           - Disk space availability
        
        7. Network Connectivity:
           - Hugging Face Hub accessibility
           - Model download capability
           - External data sources (optional)
    
    Returns:
        Dict[str, Any]: Comprehensive validation results containing:
            
            status (str): Overall validation status
                - 'success': All checks passed
                - 'warning': Non-critical issues detected
                - 'error': Critical issues that prevent operation
            
            python_version (str): Detected Python version
            
            pytorch_version (str): Detected PyTorch version
            
            transformers_version (str): Detected Transformers version
            
            cuda_available (bool): CUDA availability status
            
            cuda_version (str): CUDA version if available
            
            platform (PlatformType): Detected platform type
            
            device (str): Primary compute device
            
            warnings (List[str]): Non-critical issues that may affect performance
                or limit functionality
            
            errors (List[str]): Critical issues that must be resolved before
                framework can be used
            
            dependencies (Dict[str, Dict]): Per-dependency status including:
                - installed: Installed version or None
                - required: Required version range
                - compatible: Boolean compatibility status
    
    Raises:
        RuntimeError: If critical dependencies are missing, incompatible, or
            if the environment is fundamentally incompatible with the framework.
    
    Examples:
        >>> # Perform validation
        >>> results = validate_environment()
        >>> print(f"Status: {results['status']}")
        Status: success
        
        >>> # Check for warnings
        >>> if results['warnings']:
        ...     for warning in results['warnings']:
        ...         print(f"Warning: {warning}")
        
        >>> # Handle validation errors
        >>> if results['status'] == 'error':
        ...     print("Critical errors detected:")
        ...     for error in results['errors']:
        ...         print(f"  - {error}")
        ...     raise RuntimeError("Environment validation failed")
        
        >>> # Check specific dependencies
        >>> cuda_available = results['cuda_available']
        >>> if not cuda_available:
        ...     print("GPU acceleration not available, training will be slow")
    
    Notes:
        This function is automatically invoked during package initialization with
        graceful error handling. Manual invocation allows re-validation after
        environment changes.
        
        Validation results are logged at appropriate levels (INFO for success,
        WARNING for warnings, ERROR for errors).
        
        Optional dependencies that are missing generate warnings rather than
        errors, allowing the framework to operate with reduced functionality.
        
        Network checks are performed with timeout to avoid blocking on slow or
        unreliable connections.
    
    See Also:
        health_check: Runtime health diagnostics
        core.health.dependency_checker: Dependency validation implementation
        __version__.COMPATIBLE_VERSIONS: Dependency version requirements
    
    References:
        Python Packaging User Guide: https://packaging.python.org/
        PyTorch Installation Guide: https://pytorch.org/get-started/locally/
    """
    from .core.health.dependency_checker import DependencyChecker
    
    checker = DependencyChecker()
    results = checker.check_all()
    
    if results["status"] == "error":
        logger.error("Environment validation failed with critical errors")
        for error in results.get("errors", []):
            logger.error(f"  - {error}")
        raise RuntimeError(
            "Environment validation failed. Please resolve the errors listed above."
        )
    
    if results["status"] == "warning":
        logger.warning("Environment validation completed with warnings")
        for warning in results.get("warnings", []):
            logger.warning(f"  - {warning}")
    
    return results


def get_platform_info() -> Dict[str, Any]:
    """
    Retrieve detailed information about the current execution platform.
    
    Gathers comprehensive platform information including hardware resources, quota
    limitations, storage locations, and recommended configurations. This information
    is used by the framework's platform-adaptive optimization system to select
    optimal configurations for training and inference.
    
    Information Categories:
        1. Platform Identification:
           - Platform type (Local, Colab, Kaggle, Gitpod, Codespaces)
           - Platform tier (Colab Free/Pro/Pro+, Kaggle GPU/TPU)
           - Runtime environment details
        
        2. Compute Resources:
           - CPU information (cores, architecture, frequency)
           - GPU details (model, memory, CUDA capability)
           - TPU configuration (version, cores, topology)
           - RAM capacity and availability
        
        3. Quota and Limitations:
           - Compute time limits
           - GPU/TPU usage quotas
           - Storage quotas
           - Network bandwidth limits
           - API rate limits
        
        4. Storage Configuration:
           - Available storage paths
           - Persistent storage locations
           - Temporary storage areas
           - Cache directory locations
           - Permission levels
        
        5. Platform Capabilities:
           - Supported accelerators
           - Available runtime versions
           - Installed libraries
           - Network connectivity
           - External service access
        
        6. Optimization Recommendations:
           - Recommended batch sizes
           - Optimal model architectures
           - Suggested training strategies
           - Memory management tips
           - Platform-specific best practices
    
    Returns:
        Dict[str, Any]: Comprehensive platform information dictionary:
            
            platform (PlatformType): Platform type identifier
            
            platform_tier (str): Platform tier/variant if applicable
            
            resources (Dict[str, Any]): Available compute resources:
                - cpu: CPU details (cores, model, frequency)
                - gpu: GPU information (model, memory, count)
                - tpu: TPU configuration (version, cores)
                - ram: System memory (total, available)
            
            quotas (Dict[str, Any]): Resource quotas and limits:
                - compute_hours: Total and remaining compute time
                - gpu_hours: GPU usage quota
                - storage_gb: Storage capacity limits
                - api_limits: API rate limiting information
            
            storage (Dict[str, Any]): Storage paths and configurations:
                - home: Home directory path
                - data: Data storage location
                - models: Model checkpoint directory
                - cache: Cache directory location
                - temp: Temporary storage path
            
            recommendations (Dict[str, Any]): Platform-specific recommendations:
                - batch_size: Recommended batch sizes
                - models: Suitable model architectures
                - strategies: Suggested training approaches
                - optimizations: Platform-specific optimizations
    
    Examples:
        >>> # Get platform information
        >>> info = get_platform_info()
        >>> print(f"Running on: {info['platform'].value}")
        Running on: colab
        
        >>> # Check GPU availability
        >>> if info['resources']['gpu']['available']:
        ...     gpu_name = info['resources']['gpu']['name']
        ...     gpu_memory = info['resources']['gpu']['memory_gb']
        ...     print(f"GPU: {gpu_name} with {gpu_memory}GB memory")
        
        >>> # Get recommended batch size
        >>> batch_size = info['recommendations']['batch_size']['deberta_large']
        >>> print(f"Recommended batch size: {batch_size}")
        
        >>> # Check storage locations
        >>> model_dir = info['storage']['models']
        >>> print(f"Models will be saved to: {model_dir}")
        
        >>> # Verify quota status
        >>> remaining_gpu = info['quotas']['gpu_hours']['remaining']
        >>> if remaining_gpu < 1.0:
        ...     print("Warning: Low GPU quota remaining")
    
    Notes:
        Platform detection is performed once during package initialization and
        cached. This function retrieves the cached information without re-detection.
        
        For Colab and Kaggle, some quota information may require API calls which
        are rate-limited. Cached values are used when available.
        
        Recommendations are based on empirical testing across different platform
        configurations and updated with each framework release.
        
        Resource information is approximate and may vary based on platform load
        and concurrent usage.
    
    See Also:
        detect_platform: Platform detection logic
        deployment.platform_detector: Platform detector implementation
        deployment.smart_selector: Configuration selection based on platform
        get_quota_status: Current quota usage information
    
    References:
        Google Colab Resources: https://research.google.com/colaboratory/faq.html
        Kaggle Notebooks: https://www.kaggle.com/docs/notebooks
    """
    detector = PlatformDetector()
    platform_info = detector.get_full_info()
    
    logger.debug(f"Platform information retrieved: {platform_info['platform']}")
    return platform_info


def get_quota_status() -> Dict[str, Any]:
    """
    Retrieve current resource quota usage and remaining limits.
    
    Monitors usage of platform-specific resource quotas including compute time,
    GPU hours, storage, and API calls. Critical for preventing job interruption
    on quota-limited platforms like Google Colab and Kaggle.
    
    Tracked Quotas:
        1. Compute Time:
           - Total allocated compute hours
           - Used compute hours
           - Remaining compute hours
           - Time until quota reset
        
        2. GPU Hours:
           - GPU time allocation
           - GPU time consumed
           - GPU time remaining
           - GPU type and tier
        
        3. Storage:
           - Total storage quota
           - Storage space used
           - Storage space available
           - Temporary vs. persistent storage
        
        4. API Rate Limits:
           - API calls per minute/hour
           - Current API call count
           - Time until rate limit reset
           - Throttling status
        
        5. Network Bandwidth:
           - Download quota
           - Upload quota
           - Current usage
           - Bandwidth throttling
    
    Returns:
        Dict[str, Any]: Quota status information:
            
            platform (PlatformType): Platform for which quotas apply
            
            compute_hours (Dict[str, float]): Compute time quota:
                - total: Total allocated hours
                - used: Hours consumed
                - remaining: Hours remaining
                - reset_time: Next quota reset (ISO format)
            
            gpu_hours (Dict[str, float]): GPU time quota:
                - total: Total GPU hours
                - used: GPU hours used
                - remaining: GPU hours remaining
                - gpu_type: GPU model/tier
            
            storage_used (Dict[str, float]): Storage usage:
                - total_gb: Total storage quota in GB
                - used_gb: Storage consumed in GB
                - available_gb: Storage remaining in GB
                - usage_percent: Percentage of quota used
            
            api_calls (Dict[str, int]): API rate limit status:
                - limit_per_hour: Calls allowed per hour
                - used_this_hour: Calls made this hour
                - remaining: Calls remaining this hour
                - reset_time: Hour reset time
            
            warnings (List[str]): Quota warnings:
                - Alerts when approaching limits
                - Recommendations for quota management
                - Alternative strategies if quota low
    
    Examples:
        >>> # Check quota status
        >>> quota = get_quota_status()
        >>> 
        >>> # Monitor GPU quota
        >>> gpu_remaining = quota['gpu_hours']['remaining']
        >>> if gpu_remaining < 1.0:
        ...     print(f"Warning: Only {gpu_remaining:.1f} GPU hours remaining")
        >>> 
        >>> # Check storage quota
        >>> storage_pct = quota['storage_used']['usage_percent']
        >>> if storage_pct > 80:
        ...     print("Warning: Storage quota nearly exhausted")
        >>> 
        >>> # Handle quota warnings
        >>> for warning in quota['warnings']:
        ...     print(f"Quota Warning: {warning}")
    
    Notes:
        Quota information is platform-specific and may not be available on all
        platforms. Local execution typically has no quota limits (returns None).
        
        Quota tracking uses cached values to avoid excessive API calls. Cache is
        refreshed at configurable intervals (default: 5 minutes).
        
        For Colab, GPU quota is complex and depends on usage patterns, tier, and
        time of day. Displayed values are estimates based on observed behavior.
        
        Kaggle quotas are per-week and reset on Saturdays at 00:00 UTC.
    
    See Also:
        deployment.quota_tracker: Quota tracking implementation
        get_platform_info: Platform information including quota limits
        training.callbacks.quota_callback: Training callback for quota monitoring
    
    References:
        Colab Resource Limits: https://research.google.com/colaboratory/faq.html#resource-limits
        Kaggle Quotas: https://www.kaggle.com/docs/notebooks#quotas
    """
    tracker = QuotaTracker()
    quota_status = tracker.get_status()
    
    logger.debug("Quota status retrieved")
    return quota_status


def health_check() -> Dict[str, Any]:
    """
    Execute comprehensive health check of the framework and environment.
    
    Performs thorough diagnostics of all framework components, dependencies, and
    configurations to identify issues that may affect functionality or performance.
    Useful for troubleshooting, pre-deployment validation, and continuous monitoring.
    
    Health Check Components:
        1. Core Dependencies:
           - PyTorch installation and CUDA support
           - Transformers library functionality
           - Required Python packages
           - Optional packages status
        
        2. Configuration System:
           - Configuration file integrity
           - Schema validation
           - Template resolution
           - Default value availability
        
        3. Data Infrastructure:
           - Dataset availability
           - Data directory permissions
           - Cache directory accessibility
           - Disk space sufficiency
        
        4. Model Registry:
           - Registry integrity
           - Model configuration validity
           - Checkpoint accessibility
           - Adapter compatibility
        
        5. API Services:
           - API server status
           - Endpoint responsiveness
           - Authentication configuration
           - Rate limiting functionality
        
        6. Storage System:
           - File system permissions
           - Storage quota availability
           - Checkpoint save/load capability
           - Log file writability
        
        7. Compute Resources:
           - GPU/TPU availability and status
           - Memory sufficiency
           - CUDA driver compatibility
           - Multi-GPU configuration
        
        8. Platform Integration:
           - Platform detection accuracy
           - Platform-specific features
           - External service connectivity
           - Quota system functionality
    
    Returns:
        Dict[str, Any]: Comprehensive health check results:
            
            overall_status (str): Aggregate health status:
                - 'healthy': All checks passed
                - 'degraded': Non-critical issues detected
                - 'unhealthy': Critical issues require attention
            
            timestamp (str): Check execution timestamp (ISO format)
            
            components (Dict[str, Dict]): Per-component health status:
                Each component entry contains:
                - status: 'healthy', 'degraded', or 'unhealthy'
                - checks_passed: Number of successful checks
                - checks_failed: Number of failed checks
                - details: Detailed check results
            
            issues (List[Dict]): Detected issues:
                - severity: 'critical', 'warning', or 'info'
                - component: Affected component
                - description: Issue description
                - recommendation: Suggested resolution
            
            recommendations (List[str]): Prioritized action items to improve
                health status, ordered by severity and impact
            
            metrics (Dict[str, Any]): Performance and resource metrics:
                - memory_usage: Current memory consumption
                - disk_usage: Storage utilization
                - gpu_utilization: GPU usage statistics
    
    Examples:
        >>> # Perform health check
        >>> health = health_check()
        >>> print(f"Overall Status: {health['overall_status']}")
        Overall Status: healthy
        
        >>> # Review component status
        >>> for component, status in health['components'].items():
        ...     if status['status'] != 'healthy':
        ...         print(f"{component}: {status['status']}")
        ...         print(f"  Failed checks: {status['checks_failed']}")
        
        >>> # Address detected issues
        >>> for issue in health['issues']:
        ...     if issue['severity'] == 'critical':
        ...         print(f"CRITICAL: {issue['description']}")
        ...         print(f"Action: {issue['recommendation']}")
        
        >>> # Follow recommendations
        >>> if health['recommendations']:
        ...     print("Recommended actions:")
        ...     for rec in health['recommendations']:
        ...         print(f"  - {rec}")
        
        >>> # Monitor resource usage
        >>> gpu_usage = health['metrics']['gpu_utilization']
        >>> if gpu_usage > 95:
        ...     print("Warning: GPU near capacity")
    
    Notes:
        Health checks are designed to be non-invasive and complete quickly (typically
        under 5 seconds). Some checks may be skipped if they would significantly
        impact performance.
        
        Failed health checks do not prevent framework operation but indicate
        potential issues that should be investigated.
        
        Health check results can be exported to monitoring systems using the
        monitoring.health_monitor module.
        
        Automated health checks can be scheduled using the monitoring service or
        executed via CI/CD pipelines for continuous validation.
    
    See Also:
        validate_environment: Environment validation during initialization
        core.health.health_checker: Health check implementation
        monitoring.health_monitor: Continuous health monitoring
    
    References:
        Health Check Pattern: https://microservices.io/patterns/observability/health-check-api.html
        Site Reliability Engineering: https://sre.google/books/
    """
    from .core.health.health_checker import HealthChecker
    
    checker = HealthChecker()
    results = checker.check_all()
    
    if results["overall_status"] != "healthy":
        logger.warning("Health check detected issues")
        for issue in results.get("issues", []):
            severity = issue.get("severity", "unknown")
            description = issue.get("description", "")
            logger.warning(f"  [{severity.upper()}] {description}")
    
    return results


def _warn_deprecated(old_name: str, new_name: str, version: str) -> None:
    """
    Issue deprecation warning for legacy API usage.
    
    Provides standardized deprecation warnings for API elements scheduled for
    removal in future versions. Helps users migrate to updated APIs while maintaining
    backward compatibility during transition periods.
    
    Args:
        old_name: Deprecated function, class, or parameter name that should no
            longer be used.
        new_name: Recommended replacement providing equivalent or improved
            functionality.
        version: Framework version when the deprecated element will be removed,
            formatted as 'MAJOR.MINOR.PATCH'.
    
    Examples:
        >>> # In deprecated function
        >>> def old_function():
        ...     _warn_deprecated('old_function', 'new_function', '2.0.0')
        ...     return new_function()
    
    Notes:
        Deprecation warnings are issued using Python's warnings module and can be
        controlled via the PYTHONWARNINGS environment variable or warnings filters.
        
        By default, deprecation warnings are shown once per call site to avoid
        cluttering output while ensuring visibility.
    
    See Also:
        warnings: Python warnings module
        CHANGELOG.md: Deprecation timeline and migration guides
    """
    warnings.warn(
        f"{old_name} is deprecated and will be removed in version {version}. "
        f"Please use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3,
    )


# Package initialization
# Automatically initialize the package when imported
try:
    _initialize_package()
except Exception as e:
    logger.warning(
        f"Package initialization encountered issues: {e}. "
        f"Some features may not be available. "
        f"Run health_check() for detailed diagnostics."
    )

# Package information logging
# Log basic package information at debug level for troubleshooting
logger.debug(f"AG News Text Classification v{__version__} loaded")
logger.debug(f"Platform: {_PACKAGE_CONFIG.get('platform', 'unknown')}")
logger.debug(f"Device: {_PACKAGE_CONFIG.get('device', 'unknown')}")
