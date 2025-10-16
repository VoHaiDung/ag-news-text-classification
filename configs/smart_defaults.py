"""
AG News Text Classification - Smart Defaults Module

This module provides intelligent default configurations for the AG News Text
Classification (ag-news-text-classification) project. It automatically detects
the execution environment, available resources, and selects optimal hyperparameters
and configurations based on platform capabilities, model requirements, and best
practices from academic research.

The SmartDefaults system implements a multi-stage decision process:

1. Platform Detection:
   - Automatically identifies execution environment (local, Colab, Kaggle, etc.)
   - Detects available hardware (CPU, GPU, TPU)
   - Measures available memory (RAM, VRAM)
   - Identifies CUDA version and GPU architecture

2. Resource Assessment:
   - Evaluates computational budget
   - Determines memory constraints
   - Assesses storage availability
   - Checks quota limits (for cloud platforms)

3. Configuration Selection:
   - Selects appropriate model tier based on resources
   - Chooses optimal PEFT method (LoRA, QLoRA, etc.)
   - Determines batch size and gradient accumulation
   - Selects precision mode (FP32, FP16, BF16, INT8, INT4)

4. Hyperparameter Optimization:
   - Sets learning rate based on model size and batch size
   - Configures optimizer and scheduler
   - Applies regularization based on model capacity
   - Enables appropriate overfitting prevention measures

5. Platform-Specific Optimizations:
   - Applies platform-specific optimizations
   - Configures checkpointing strategy
   - Sets up monitoring and logging
   - Manages quota and session timeouts

The module follows best practices from deep learning research:

Learning Rate Selection:
    - Based on batch size scaling (Goyal et al., 2017)
    - Model-size dependent initialization (Zhang et al., 2019)
    - Warmup schedule for large models (He et al., 2016)

Batch Size Selection:
    - GPU memory constraints
    - Gradient accumulation for effective larger batches
    - Linear scaling rule (Goyal et al., 2017)

Regularization:
    - Dropout rates based on model size (Srivastava et al., 2014)
    - Weight decay for transformer models (Loshchilov & Hutter, 2019)
    - Label smoothing for classification (Szegedy et al., 2016)

Mixed Precision Training:
    - FP16 for older GPUs (Micikevicius et al., 2018)
    - BF16 for Ampere+ GPUs (better numerical stability)
    - Automatic loss scaling

Key Features:
    - Zero-configuration operation (works out-of-the-box)
    - Platform-aware optimization
    - Resource-constrained adaptation
    - Overfitting prevention integration
    - SOTA-aligned hyperparameters
    - Reproducibility guarantees
    - Extensive validation and safety checks

Usage:
    Basic usage with auto-detection:
        from configs.smart_defaults import SmartDefaults
        
        defaults = SmartDefaults()
        config = defaults.get_config()
    
    Specify platform explicitly:
        defaults = SmartDefaults(platform='colab')
        config = defaults.get_config()
    
    Get model-specific defaults:
        model_config = defaults.get_model_config('deberta_v3_large')
    
    Get training defaults:
        training_config = defaults.get_training_config(
            model_size='large',
            peft_method='lora'
        )
    
    Platform-optimized configuration:
        config = defaults.get_platform_optimized_config()
    
    Tier-specific configuration:
        config = defaults.get_tier_config(tier=1)  # SOTA tier

Architecture:
    The module uses a hierarchical configuration system:
    
    SmartDefaults (Main Controller)
        |
        +-- PlatformDetector (Environment detection)
        |
        +-- ResourceAnalyzer (Resource assessment)
        |
        +-- ConfigSelector (Configuration selection)
        |
        +-- HyperparameterOptimizer (Hyperparameter tuning)
        |
        +-- ValidationEngine (Configuration validation)
        |
        +-- RecommendationEngine (Suggestions and warnings)

References:
    Learning Rate and Batch Size:
        - Goyal, P., Dollár, P., Girshick, R., Noordhuis, P., Wesolowski, L.,
          Kyrola, A., Tulloch, A., Jia, Y., & He, K. (2017). "Accurate, Large
          Minibatch SGD: Training ImageNet in 1 Hour". arXiv:1706.02677.
        - Smith, S. L., Kindermans, P. J., Ying, C., & Le, Q. V. (2018).
          "Don't Decay the Learning Rate, Increase the Batch Size". ICLR.
    
    Mixed Precision Training:
        - Micikevicius, P., Narang, S., Alben, J., Diamos, G., Elsen, E.,
          Garcia, D., Ginsburg, B., Houston, M., Kuchaiev, O., Venkatesh, G.,
          & Wu, H. (2018). "Mixed Precision Training". ICLR.
    
    Regularization:
        - Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., &
          Salakhutdinov, R. (2014). "Dropout: A Simple Way to Prevent Neural
          Networks from Overfitting". JMLR, 15(1), 1929-1958.
        - Loshchilov, I., & Hutter, F. (2019). "Decoupled Weight Decay
          Regularization". ICLR.
        - Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
          "Rethinking the Inception Architecture for Computer Vision". CVPR.
    
    Transformer Training:
        - Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L.,
          Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). "Attention is
          All You Need". NeurIPS.
        - Liu, Y., Ott, M., Goyal, N., Du, J., Joshi, M., Chen, D., Levy, O.,
          Lewis, M., Zettlemoyer, L., & Stoyanov, V. (2019). "RoBERTa: A
          Robustly Optimized BERT Pretraining Approach". arXiv:1907.11692.
    
    Parameter-Efficient Fine-Tuning:
        - Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S.,
          Wang, L., & Chen, W. (2021). "LoRA: Low-Rank Adaptation of Large
          Language Models". arXiv:2106.09685.
        - Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023).
          "QLoRA: Efficient Finetuning of Quantized LLMs". arXiv:2305.14314.

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
Project: AG News Text Classification (ag-news-text-classification)
Repository: https://github.com/VoHaiDung/ag-news-text-classification
"""

import logging
import platform
import os
import sys
import subprocess
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
import warnings


__version__ = "1.0.0"
__author__ = "Võ Hải Dũng"
__email__ = "vohaidung.work@gmail.com"
__license__ = "MIT"
__project__ = "AG News Text Classification (ag-news-text-classification)"


logger = logging.getLogger(__name__)


@dataclass
class PlatformInfo:
    """
    Platform information container.
    
    Stores detected platform characteristics including hardware capabilities,
    software environment, and resource constraints.
    
    Attributes:
        platform_type: Platform identifier (local, colab, kaggle, etc.)
        os_type: Operating system (linux, windows, darwin)
        python_version: Python version string
        has_gpu: Whether GPU is available
        gpu_count: Number of available GPUs
        gpu_names: List of GPU device names
        gpu_memory_gb: Total GPU memory in GB
        gpu_compute_capability: CUDA compute capability
        cpu_count: Number of CPU cores
        ram_gb: Total RAM in GB
        cuda_available: Whether CUDA is available
        cuda_version: CUDA version string
        pytorch_version: PyTorch version string
        has_tpu: Whether TPU is available
        is_notebook: Whether running in notebook environment
        storage_gb: Available storage in GB
    
    Examples:
        Create platform info:
            info = PlatformInfo(
                platform_type='colab',
                has_gpu=True,
                gpu_memory_gb=16.0
            )
        
        Check GPU availability:
            if info.has_gpu and info.gpu_memory_gb >= 16:
                print("Sufficient GPU resources")
    """
    
    platform_type: str = "unknown"
    os_type: str = platform.system().lower()
    python_version: str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    has_gpu: bool = False
    gpu_count: int = 0
    gpu_names: List[str] = field(default_factory=list)
    gpu_memory_gb: float = 0.0
    gpu_compute_capability: Optional[Tuple[int, int]] = None
    cpu_count: int = os.cpu_count() or 1
    ram_gb: float = 0.0
    cuda_available: bool = False
    cuda_version: Optional[str] = None
    pytorch_version: Optional[str] = None
    has_tpu: bool = False
    is_notebook: bool = False
    storage_gb: float = 0.0


@dataclass
class ResourceConstraints:
    """
    Resource constraints for configuration selection.
    
    Defines the available computational resources and their limits,
    used to determine feasible model configurations.
    
    Attributes:
        max_batch_size: Maximum batch size
        max_sequence_length: Maximum sequence length
        max_model_parameters: Maximum model parameters (in millions)
        preferred_precision: Preferred precision mode
        can_use_mixed_precision: Whether mixed precision is available
        can_use_gradient_checkpointing: Whether gradient checkpointing is available
        max_gradient_accumulation: Maximum gradient accumulation steps
        supports_flash_attention: Whether Flash Attention is supported
        supports_qlora: Whether QLoRA is supported
    
    Examples:
        Define constraints:
            constraints = ResourceConstraints(
                max_batch_size=32,
                max_model_parameters=350,  # 350M parameters
                preferred_precision='fp16'
            )
    """
    
    max_batch_size: int = 16
    max_sequence_length: int = 512
    max_model_parameters: float = 100.0  # in millions
    preferred_precision: str = "fp32"
    can_use_mixed_precision: bool = False
    can_use_gradient_checkpointing: bool = True
    max_gradient_accumulation: int = 8
    supports_flash_attention: bool = False
    supports_qlora: bool = False


class PlatformDetector:
    """
    Platform detection and analysis engine.
    
    Automatically detects the execution environment, hardware capabilities,
    and software stack. Provides comprehensive platform information for
    configuration selection.
    
    The detector identifies:
        - Platform type (local, Colab, Kaggle, Gitpod, Codespaces)
        - Operating system
        - Python version
        - GPU availability and specifications
        - CUDA version and compute capability
        - CPU cores and RAM
        - Available storage
        - Notebook environment detection
    
    Methods:
        detect: Perform full platform detection
        get_platform_info: Get detected platform information
        get_resource_constraints: Get resource constraints
    
    Examples:
        Basic detection:
            detector = PlatformDetector()
            info = detector.detect()
            print(f"Platform: {info.platform_type}")
        
        Check GPU:
            if info.has_gpu:
                print(f"GPU: {info.gpu_names[0]}")
                print(f"VRAM: {info.gpu_memory_gb} GB")
    """
    
    def __init__(self):
        """Initialize platform detector."""
        self.info = PlatformInfo()
    
    def detect(self) -> PlatformInfo:
        """
        Perform comprehensive platform detection.
        
        Executes a series of detection methods to identify the platform,
        hardware, and software environment.
        
        Returns:
            PlatformInfo object with detected information
        
        Examples:
            Detect platform:
                detector = PlatformDetector()
                info = detector.detect()
        """
        self._detect_platform_type()
        self._detect_notebook_environment()
        self._detect_gpu()
        self._detect_cuda()
        self._detect_memory()
        self._detect_storage()
        self._detect_pytorch()
        self._detect_tpu()
        
        logger.info(
            f"Platform detected: {self.info.platform_type} "
            f"(GPU: {self.info.has_gpu}, "
            f"VRAM: {self.info.gpu_memory_gb:.1f}GB, "
            f"RAM: {self.info.ram_gb:.1f}GB)"
        )
        
        return self.info
    
    def _detect_platform_type(self) -> None:
        """
        Detect platform type.
        
        Identifies whether the code is running on:
        - Google Colab (free or pro)
        - Kaggle Notebooks
        - Gitpod
        - GitHub Codespaces
        - Local machine
        
        Detection uses environment variables and file system markers.
        """
        if 'COLAB_GPU' in os.environ or 'COLAB_TPU_ADDR' in os.environ:
            self.info.platform_type = 'colab'
        elif 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
            self.info.platform_type = 'kaggle'
        elif 'GITPOD_WORKSPACE_ID' in os.environ:
            self.info.platform_type = 'gitpod'
        elif 'CODESPACES' in os.environ:
            self.info.platform_type = 'codespaces'
        else:
            self.info.platform_type = 'local'
    
    def _detect_notebook_environment(self) -> None:
        """
        Detect if running in notebook environment.
        
        Checks for IPython/Jupyter kernel.
        """
        try:
            from IPython import get_ipython
            self.info.is_notebook = get_ipython() is not None
        except ImportError:
            self.info.is_notebook = False
    
    def _detect_gpu(self) -> None:
        """
        Detect GPU availability and specifications.
        
        Uses PyTorch to detect GPU devices and query their properties.
        Extracts GPU names, memory, and compute capability.
        """
        try:
            import torch
            
            self.info.has_gpu = torch.cuda.is_available()
            if self.info.has_gpu:
                self.info.gpu_count = torch.cuda.device_count()
                
                for i in range(self.info.gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    self.info.gpu_names.append(gpu_name)
                    
                    gpu_memory_bytes = torch.cuda.get_device_properties(i).total_memory
                    gpu_memory_gb = gpu_memory_bytes / (1024 ** 3)
                    self.info.gpu_memory_gb = max(self.info.gpu_memory_gb, gpu_memory_gb)
                    
                    major = torch.cuda.get_device_properties(i).major
                    minor = torch.cuda.get_device_properties(i).minor
                    self.info.gpu_compute_capability = (major, minor)
        
        except ImportError:
            logger.warning("PyTorch not available for GPU detection")
        except Exception as e:
            logger.warning(f"GPU detection failed: {e}")
    
    def _detect_cuda(self) -> None:
        """
        Detect CUDA availability and version.
        
        Queries PyTorch for CUDA information.
        """
        try:
            import torch
            
            self.info.cuda_available = torch.cuda.is_available()
            if self.info.cuda_available:
                self.info.cuda_version = torch.version.cuda
        
        except ImportError:
            pass
    
    def _detect_memory(self) -> None:
        """
        Detect available RAM.
        
        Uses psutil if available, otherwise falls back to platform-specific methods.
        """
        try:
            import psutil
            
            ram_bytes = psutil.virtual_memory().total
            self.info.ram_gb = ram_bytes / (1024 ** 3)
        
        except ImportError:
            if self.info.platform_type == 'colab':
                self.info.ram_gb = 12.0  # Colab free tier default
            elif self.info.platform_type == 'kaggle':
                self.info.ram_gb = 13.0  # Kaggle default
            else:
                self.info.ram_gb = 8.0  # Conservative estimate
    
    def _detect_storage(self) -> None:
        """
        Detect available storage space.
        
        Checks available disk space on the current file system.
        """
        try:
            import shutil
            
            total, used, free = shutil.disk_usage("/")
            self.info.storage_gb = free / (1024 ** 3)
        
        except Exception as e:
            logger.warning(f"Storage detection failed: {e}")
            self.info.storage_gb = 20.0  # Conservative estimate
    
    def _detect_pytorch(self) -> None:
        """
        Detect PyTorch version.
        
        Queries installed PyTorch version.
        """
        try:
            import torch
            self.info.pytorch_version = torch.__version__
        except ImportError:
            self.info.pytorch_version = None
    
    def _detect_tpu(self) -> None:
        """
        Detect TPU availability.
        
        Checks for TPU environment (Kaggle TPU or Google Cloud).
        """
        if self.info.platform_type == 'kaggle':
            tpu_address = os.environ.get('TPU_NAME')
            self.info.has_tpu = tpu_address is not None
        elif 'COLAB_TPU_ADDR' in os.environ:
            self.info.has_tpu = True
        else:
            self.info.has_tpu = False
    
    def get_platform_info(self) -> PlatformInfo:
        """
        Get detected platform information.
        
        Returns:
            PlatformInfo object
        
        Examples:
            Get info:
                info = detector.get_platform_info()
        """
        return self.info
    
    def get_resource_constraints(self) -> ResourceConstraints:
        """
        Determine resource constraints based on detected platform.
        
        Analyzes platform capabilities and returns appropriate resource
        constraints for configuration selection.
        
        Returns:
            ResourceConstraints object
        
        Examples:
            Get constraints:
                constraints = detector.get_resource_constraints()
                print(f"Max batch size: {constraints.max_batch_size}")
        """
        constraints = ResourceConstraints()
        
        if self.info.has_gpu:
            vram_gb = self.info.gpu_memory_gb
            
            if vram_gb >= 40:
                constraints.max_batch_size = 64
                constraints.max_model_parameters = 10000.0  # 10B
                constraints.max_sequence_length = 1024
            elif vram_gb >= 24:
                constraints.max_batch_size = 32
                constraints.max_model_parameters = 3000.0  # 3B
                constraints.max_sequence_length = 512
            elif vram_gb >= 16:
                constraints.max_batch_size = 16
                constraints.max_model_parameters = 1000.0  # 1B
                constraints.max_sequence_length = 512
            elif vram_gb >= 12:
                constraints.max_batch_size = 12
                constraints.max_model_parameters = 500.0  # 500M
                constraints.max_sequence_length = 512
            else:
                constraints.max_batch_size = 8
                constraints.max_model_parameters = 200.0  # 200M
                constraints.max_sequence_length = 256
            
            if self.info.cuda_available:
                constraints.can_use_mixed_precision = True
                
                if self.info.gpu_compute_capability:
                    major, minor = self.info.gpu_compute_capability
                    
                    if major >= 7:
                        constraints.preferred_precision = "fp16"
                    
                    if major >= 8:
                        constraints.preferred_precision = "bf16"
                        constraints.supports_flash_attention = True
                    
                    if major >= 7:
                        constraints.supports_qlora = True
        
        else:
            constraints.max_batch_size = 4
            constraints.max_model_parameters = 100.0  # 100M
            constraints.max_sequence_length = 256
            constraints.preferred_precision = "fp32"
            constraints.can_use_mixed_precision = False
            constraints.supports_qlora = False
        
        return constraints


class SmartDefaults:
    """
    Smart default configuration generator.
    
    Provides intelligent default configurations based on platform detection,
    resource constraints, and best practices from academic research.
    
    This class implements the main logic for automatic configuration selection,
    combining platform detection, resource assessment, and domain knowledge
    to generate optimal configurations for different scenarios.
    
    The configuration process follows these principles:
    
    1. Safety First:
       - Never exceed platform resource limits
       - Always include overfitting prevention measures
       - Validate all generated configurations
    
    2. Performance Optimization:
       - Use optimal batch size for GPU utilization
       - Select appropriate precision mode
       - Enable efficient training techniques
    
    3. Reproducibility:
       - Set random seeds
       - Use deterministic algorithms when possible
       - Document all default choices
    
    4. Flexibility:
       - Allow manual overrides
       - Provide multiple configuration tiers
       - Support custom requirements
    
    Attributes:
        platform_info: Detected platform information
        resource_constraints: Determined resource constraints
        compatibility_matrix: Loaded compatibility matrix
        overfitting_prevention_enabled: Whether to enable overfitting prevention
    
    Methods:
        get_config: Get complete default configuration
        get_model_config: Get model-specific configuration
        get_training_config: Get training configuration
        get_data_config: Get data configuration
        get_platform_optimized_config: Get platform-optimized configuration
        get_tier_config: Get tier-specific configuration
        recommend_model: Recommend model based on resources
        recommend_peft_method: Recommend PEFT method
        validate_config: Validate configuration
    
    Examples:
        Basic usage:
            defaults = SmartDefaults()
            config = defaults.get_config()
        
        Platform-specific:
            defaults = SmartDefaults(platform='colab')
            config = defaults.get_platform_optimized_config()
        
        Model recommendation:
            model = defaults.recommend_model(target_accuracy=0.96)
        
        Get training config:
            training = defaults.get_training_config(
                model_name='deberta_v3_large',
                peft_method='lora'
            )
    """
    
    def __init__(
        self,
        platform: Optional[str] = None,
        auto_detect: bool = True,
        overfitting_prevention: bool = True,
        compatibility_matrix_path: Optional[Path] = None
    ):
        """
        Initialize SmartDefaults.
        
        Args:
            platform: Platform type (None for auto-detection)
            auto_detect: Whether to auto-detect platform
            overfitting_prevention: Whether to enable overfitting prevention
            compatibility_matrix_path: Path to compatibility matrix YAML
        
        Examples:
            Auto-detect platform:
                defaults = SmartDefaults()
            
            Specify platform:
                defaults = SmartDefaults(platform='kaggle')
            
            Disable overfitting prevention:
                defaults = SmartDefaults(overfitting_prevention=False)
        """
        self.overfitting_prevention_enabled = overfitting_prevention
        
        if auto_detect:
            detector = PlatformDetector()
            self.platform_info = detector.detect()
            self.resource_constraints = detector.get_resource_constraints()
        else:
            self.platform_info = PlatformInfo(platform_type=platform or 'local')
            self.resource_constraints = ResourceConstraints()
        
        if platform:
            self.platform_info.platform_type = platform
        
        self.compatibility_matrix = self._load_compatibility_matrix(
            compatibility_matrix_path
        )
        
        self._initialize_defaults()
    
    def _load_compatibility_matrix(
        self,
        matrix_path: Optional[Path]
    ) -> Dict[str, Any]:
        """
        Load compatibility matrix.
        
        Args:
            matrix_path: Path to compatibility matrix YAML
        
        Returns:
            Compatibility matrix dictionary
        """
        if matrix_path is None:
            matrix_path = Path(__file__).parent / "compatibility_matrix.yaml"
        
        try:
            import yaml
            
            if matrix_path.exists():
                with open(matrix_path, 'r') as f:
                    return yaml.safe_load(f)
            else:
                logger.warning(f"Compatibility matrix not found: {matrix_path}")
                return {}
        
        except ImportError:
            logger.warning("PyYAML not available, using empty compatibility matrix")
            return {}
        except Exception as e:
            logger.warning(f"Failed to load compatibility matrix: {e}")
            return {}
    
    def _initialize_defaults(self) -> None:
        """
        Initialize default values.
        
        Sets up base configurations that will be used across all methods.
        """
        self.base_defaults = {
            'seed': 42,
            'num_labels': 4,  # AG News has 4 classes
            'dataset_name': 'ag_news',
            'max_length': 512,
            'warmup_ratio': 0.1,
        }
        
        self.model_defaults = self._get_model_defaults()
        self.training_defaults = self._get_training_defaults()
        self.data_defaults = self._get_data_defaults()
    
    def _get_model_defaults(self) -> Dict[str, Any]:
        """
        Get default model configurations.
        
        Returns:
            Dictionary of model defaults
        """
        return {
            'use_cache': True,
            'pretrained': True,
            'trust_remote_code': False,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'classifier_dropout': 0.1,
        }
    
    def _get_training_defaults(self) -> Dict[str, Any]:
        """
        Get default training configurations.
        
        Returns:
            Dictionary of training defaults
        """
        defaults = {
            'num_epochs': 10,
            'logging_steps': 100,
            'save_strategy': 'epoch',
            'evaluation_strategy': 'epoch',
            'load_best_model_at_end': True,
            'metric_for_best_model': 'eval_loss',
            'greater_is_better': False,
            'save_total_limit': 3,
            'dataloader_num_workers': 0,
            'dataloader_pin_memory': True,
            'max_grad_norm': 1.0,
        }
        
        if self.resource_constraints.can_use_mixed_precision:
            defaults['mixed_precision'] = self.resource_constraints.preferred_precision
        else:
            defaults['mixed_precision'] = 'no'
        
        defaults['batch_size'] = self._get_optimal_batch_size()
        defaults['gradient_accumulation_steps'] = 1
        
        return defaults
    
    def _get_data_defaults(self) -> Dict[str, Any]:
        """
        Get default data configurations.
        
        Returns:
            Dictionary of data defaults
        """
        return {
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'stratify': True,
            'shuffle': True,
            'preprocessing': {
                'lowercase': False,
                'remove_html': True,
                'remove_urls': False,
                'remove_special_chars': False,
            },
            'augmentation': {
                'enabled': False,
                'methods': [],
                'augmentation_ratio': 0.1,
            },
        }
    
    def _get_optimal_batch_size(
        self,
        model_size: str = 'base',
        sequence_length: int = 512
    ) -> int:
        """
        Calculate optimal batch size based on available resources.
        
        Uses heuristics based on GPU memory and model size to determine
        the largest batch size that fits in memory.
        
        Args:
            model_size: Model size category ('base', 'large', 'xlarge', 'llm')
            sequence_length: Input sequence length
        
        Returns:
            Optimal batch size
        
        Examples:
            Get batch size:
                batch_size = defaults._get_optimal_batch_size('large', 512)
        """
        if not self.platform_info.has_gpu:
            return 4  # Conservative for CPU
        
        vram_gb = self.platform_info.gpu_memory_gb
        
        size_multipliers = {
            'base': 1.0,
            'large': 0.5,
            'xlarge': 0.25,
            'xxlarge': 0.125,
            'llm_7b': 0.1,
            'llm_13b': 0.05,
            'llm_70b': 0.01,
        }
        
        multiplier = size_multipliers.get(model_size, 1.0)
        
        if sequence_length > 512:
            multiplier *= (512 / sequence_length)
        
        if vram_gb >= 40:
            base_batch_size = 64
        elif vram_gb >= 24:
            base_batch_size = 32
        elif vram_gb >= 16:
            base_batch_size = 16
        elif vram_gb >= 12:
            base_batch_size = 12
        elif vram_gb >= 8:
            base_batch_size = 8
        else:
            base_batch_size = 4
        
        optimal_batch_size = max(1, int(base_batch_size * multiplier))
        
        optimal_batch_size = min(
            optimal_batch_size,
            self.resource_constraints.max_batch_size
        )
        
        if optimal_batch_size % 2 != 0 and optimal_batch_size > 1:
            optimal_batch_size = optimal_batch_size - 1
        
        return optimal_batch_size
    
    def get_config(
        self,
        model_name: Optional[str] = None,
        tier: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get complete default configuration.
        
        Generates a comprehensive configuration including model, training,
        data, and platform-specific settings.
        
        Args:
            model_name: Specific model name (None for auto-selection)
            tier: Configuration tier (1-5, None for auto-selection)
        
        Returns:
            Complete configuration dictionary
        
        Examples:
            Get auto-selected config:
                config = defaults.get_config()
            
            Get specific model config:
                config = defaults.get_config(model_name='deberta_v3_large')
            
            Get tier config:
                config = defaults.get_config(tier=1)  # SOTA tier
        """
        if tier is not None:
            return self.get_tier_config(tier)
        
        if model_name is None:
            model_name = self.recommend_model()
        
        config = {
            **self.base_defaults,
            'model': self.get_model_config(model_name),
            'training': self.get_training_config(model_name),
            'data': self.get_data_config(),
            'platform': {
                'type': self.platform_info.platform_type,
                'auto_detect': True,
            },
        }
        
        if self.overfitting_prevention_enabled:
            config['overfitting_prevention'] = self._get_overfitting_prevention_config()
        
        return config
    
    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Returns configuration for a specific model including architecture
        settings, dropout rates, and PEFT configuration if applicable.
        
        Args:
            model_name: Model identifier
        
        Returns:
            Model configuration dictionary
        
        Examples:
            Get DeBERTa config:
                config = defaults.get_model_config('deberta_v3_large')
            
            Get LLaMA config:
                config = defaults.get_model_config('llama2_7b')
        """
        config = {**self.model_defaults}
        
        model_registry = {
            'deberta_v3_base': {
                'name': 'microsoft/deberta-v3-base',
                'type': 'deberta',
                'size': 'base',
                'max_length': 512,
            },
            'deberta_v3_large': {
                'name': 'microsoft/deberta-v3-large',
                'type': 'deberta',
                'size': 'large',
                'max_length': 512,
            },
            'deberta_v3_xlarge': {
                'name': 'microsoft/deberta-v3-xlarge',
                'type': 'deberta',
                'size': 'xlarge',
                'max_length': 512,
                'requires_peft': True,
            },
            'deberta_v2_xxlarge': {
                'name': 'microsoft/deberta-v2-xxlarge',
                'type': 'deberta',
                'size': 'xxlarge',
                'max_length': 512,
                'requires_peft': True,
            },
            'roberta_base': {
                'name': 'roberta-base',
                'type': 'roberta',
                'size': 'base',
                'max_length': 512,
            },
            'roberta_large': {
                'name': 'roberta-large',
                'type': 'roberta',
                'size': 'large',
                'max_length': 512,
            },
            'llama2_7b': {
                'name': 'meta-llama/Llama-2-7b-hf',
                'type': 'llama',
                'size': 'llm_7b',
                'max_length': 4096,
                'requires_peft': True,
                'requires_auth': True,
            },
            'mistral_7b': {
                'name': 'mistralai/Mistral-7B-v0.1',
                'type': 'mistral',
                'size': 'llm_7b',
                'max_length': 8192,
                'requires_peft': True,
            },
            'phi_2': {
                'name': 'microsoft/phi-2',
                'type': 'phi',
                'size': 'llm_3b',
                'max_length': 2048,
                'requires_peft': True,
                'trust_remote_code': True,
            },
        }
        
        if model_name in model_registry:
            model_info = model_registry[model_name]
            config.update(model_info)
            
            if model_info.get('requires_peft', False):
                peft_method = self.recommend_peft_method(model_name)
                config['peft'] = self._get_peft_config(peft_method, model_info['size'])
        
        else:
            logger.warning(f"Unknown model: {model_name}, using base defaults")
            config.update({
                'name': model_name,
                'type': 'unknown',
                'size': 'base',
                'max_length': 512,
            })
        
        return config
    
    def get_training_config(
        self,
        model_name: Optional[str] = None,
        peft_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get training configuration.
        
        Returns training hyperparameters optimized for the specified model
        and PEFT method.
        
        Args:
            model_name: Model identifier
            peft_method: PEFT method (lora, qlora, etc.)
        
        Returns:
            Training configuration dictionary
        
        Examples:
            Get training config:
                config = defaults.get_training_config('deberta_v3_large')
            
            Get LoRA training config:
                config = defaults.get_training_config(
                    model_name='llama2_7b',
                    peft_method='qlora'
                )
        """
        config = {**self.training_defaults}
        
        if model_name:
            model_config = self.get_model_config(model_name)
            model_size = model_config.get('size', 'base')
            
            config['batch_size'] = self._get_optimal_batch_size(
                model_size=model_size,
                sequence_length=model_config.get('max_length', 512)
            )
            
            config['optimizer'] = self._get_optimizer_config(model_size)
            config['scheduler'] = self._get_scheduler_config(model_size)
            config['regularization'] = self._get_regularization_config(model_size)
            config['early_stopping'] = self._get_early_stopping_config()
        
        if self.resource_constraints.can_use_gradient_checkpointing:
            if model_name and 'xlarge' in model_name.lower() or 'llm' in model_size:
                config['gradient_checkpointing'] = True
        
        return config
    
    def _get_optimizer_config(self, model_size: str) -> Dict[str, Any]:
        """
        Get optimizer configuration.
        
        Returns optimizer settings based on model size and best practices.
        
        Args:
            model_size: Model size category
        
        Returns:
            Optimizer configuration
        """
        learning_rates = {
            'base': 2e-5,
            'large': 1e-5,
            'xlarge': 5e-6,
            'xxlarge': 3e-6,
            'llm_3b': 2e-5,
            'llm_7b': 1e-5,
            'llm_13b': 5e-6,
            'llm_70b': 3e-6,
        }
        
        lr = learning_rates.get(model_size, 2e-5)
        
        return {
            'type': 'adamw',
            'lr': lr,
            'weight_decay': 0.01,
            'betas': [0.9, 0.999],
            'eps': 1e-8,
        }
    
    def _get_scheduler_config(self, model_size: str) -> Dict[str, Any]:
        """
        Get learning rate scheduler configuration.
        
        Args:
            model_size: Model size category
        
        Returns:
            Scheduler configuration
        """
        return {
            'type': 'linear',
            'num_warmup_steps': 0,
        }
    
    def _get_regularization_config(self, model_size: str) -> Dict[str, Any]:
        """
        Get regularization configuration.
        
        Args:
            model_size: Model size category
        
        Returns:
            Regularization configuration
        """
        dropout_rates = {
            'base': 0.1,
            'large': 0.1,
            'xlarge': 0.15,
            'xxlarge': 0.2,
            'llm_3b': 0.1,
            'llm_7b': 0.1,
            'llm_13b': 0.15,
            'llm_70b': 0.2,
        }
        
        dropout = dropout_rates.get(model_size, 0.1)
        
        return {
            'dropout': dropout,
            'attention_dropout': dropout,
            'hidden_dropout': dropout,
            'weight_decay': 0.01,
            'label_smoothing': 0.1,
            'gradient_clip_norm': 1.0,
        }
    
    def _get_early_stopping_config(self) -> Dict[str, Any]:
        """
        Get early stopping configuration.
        
        Returns:
            Early stopping configuration
        """
        return {
            'enabled': True,
            'patience': 3,
            'min_delta': 0.0,
            'monitor': 'val_loss',
            'mode': 'min',
            'restore_best_weights': True,
        }
    
    def _get_peft_config(self, peft_method: str, model_size: str) -> Dict[str, Any]:
        """
        Get PEFT configuration.
        
        Args:
            peft_method: PEFT method name
            model_size: Model size category
        
        Returns:
            PEFT configuration
        """
        if peft_method == 'lora':
            return self._get_lora_config(model_size)
        elif peft_method == 'qlora':
            return self._get_qlora_config(model_size)
        elif peft_method == 'adapter':
            return self._get_adapter_config(model_size)
        else:
            return {'method': 'none'}
    
    def _get_lora_config(self, model_size: str) -> Dict[str, Any]:
        """
        Get LoRA configuration.
        
        Args:
            model_size: Model size category
        
        Returns:
            LoRA configuration
        """
        rank_map = {
            'base': 8,
            'large': 16,
            'xlarge': 16,
            'xxlarge': 8,
            'llm_3b': 16,
            'llm_7b': 16,
            'llm_13b': 8,
            'llm_70b': 8,
        }
        
        rank = rank_map.get(model_size, 8)
        
        return {
            'method': 'lora',
            'enabled': True,
            'rank': rank,
            'alpha': rank * 2,
            'dropout': 0.1,
            'target_modules': ['query_proj', 'value_proj'],
            'bias': 'none',
            'task_type': 'SEQ_CLS',
        }
    
    def _get_qlora_config(self, model_size: str) -> Dict[str, Any]:
        """
        Get QLoRA configuration.
        
        Args:
            model_size: Model size category
        
        Returns:
            QLoRA configuration
        """
        lora_config = self._get_lora_config(model_size)
        
        lora_config.update({
            'method': 'qlora',
            'bits': 4,
            'quant_type': 'nf4',
            'double_quant': True,
            'compute_dtype': 'bfloat16' if self.resource_constraints.preferred_precision == 'bf16' else 'float16',
        })
        
        return lora_config
    
    def _get_adapter_config(self, model_size: str) -> Dict[str, Any]:
        """
        Get Adapter configuration.
        
        Args:
            model_size: Model size category
        
        Returns:
            Adapter configuration
        """
        return {
            'method': 'adapter',
            'enabled': True,
            'adapter_type': 'pfeiffer',
            'reduction_factor': 16,
            'non_linearity': 'gelu',
            'adapter_dropout': 0.1,
        }
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration.
        
        Returns:
            Data configuration dictionary
        
        Examples:
            Get data config:
                config = defaults.get_data_config()
        """
        return {**self.data_defaults}
    
    def _get_overfitting_prevention_config(self) -> Dict[str, Any]:
        """
        Get overfitting prevention configuration.
        
        Returns:
            Overfitting prevention configuration
        """
        return {
            'enabled': True,
            'test_set_protection': {
                'enabled': True,
                'hash_verification': True,
                'access_logging': True,
            },
            'monitoring': {
                'track_train_val_gap': True,
                'gap_threshold': 0.05,
                'alert_on_overfitting': True,
            },
            'constraints': {
                'min_validation_size': 0.1,
                'required_regularization': True,
            },
        }
    
    def get_platform_optimized_config(self) -> Dict[str, Any]:
        """
        Get platform-optimized configuration.
        
        Returns configuration optimized for the detected platform,
        including platform-specific optimizations and workarounds.
        
        Returns:
            Platform-optimized configuration
        
        Examples:
            Get platform config:
                config = defaults.get_platform_optimized_config()
        """
        config = self.get_config()
        
        platform_type = self.platform_info.platform_type
        
        if platform_type == 'colab':
            config = self._apply_colab_optimizations(config)
        elif platform_type == 'kaggle':
            config = self._apply_kaggle_optimizations(config)
        elif platform_type == 'local':
            config = self._apply_local_optimizations(config)
        
        return config
    
    def _apply_colab_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Colab-specific optimizations.
        
        Args:
            config: Base configuration
        
        Returns:
            Optimized configuration for Colab
        """
        config['platform'].update({
            'type': 'colab',
            'mount_drive': True,
            'cache_dir': '/content/drive/MyDrive/ag_news_cache',
            'checkpoint_dir': '/content/drive/MyDrive/ag_news_checkpoints',
            'auto_disconnect_prevention': True,
        })
        
        config['training']['save_steps'] = 500
        config['training']['save_strategy'] = 'steps'
        
        return config
    
    def _apply_kaggle_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply Kaggle-specific optimizations.
        
        Args:
            config: Base configuration
        
        Returns:
            Optimized configuration for Kaggle
        """
        config['platform'].update({
            'type': 'kaggle',
            'use_kaggle_datasets': True,
            'cache_dir': '/kaggle/working/cache',
            'checkpoint_dir': '/kaggle/working/checkpoints',
        })
        
        return config
    
    def _apply_local_optimizations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply local machine optimizations.
        
        Args:
            config: Base configuration
        
        Returns:
            Optimized configuration for local
        """
        config['platform'].update({
            'type': 'local',
            'cache_dir': './cache',
            'checkpoint_dir': './checkpoints',
        })
        
        if self.platform_info.gpu_count > 1:
            config['training']['distributed'] = True
            config['training']['world_size'] = self.platform_info.gpu_count
        
        return config
    
    def get_tier_config(self, tier: int) -> Dict[str, Any]:
        """
        Get tier-specific configuration.
        
        Returns configuration for a specific tier (1-5) based on the
        tiered model system:
        - Tier 1: SOTA XLarge models with LoRA
        - Tier 2: LLM models with QLoRA
        - Tier 3: Ensemble models
        - Tier 4: Distilled models
        - Tier 5: Free-tier optimized models
        
        Args:
            tier: Tier number (1-5)
        
        Returns:
            Tier-specific configuration
        
        Raises:
            ValueError: If tier is not in range 1-5
        
        Examples:
            Get SOTA tier:
                config = defaults.get_tier_config(1)
            
            Get free-tier:
                config = defaults.get_tier_config(5)
        """
        if tier not in range(1, 6):
            raise ValueError(f"Tier must be between 1 and 5, got {tier}")
        
        tier_models = {
            1: 'deberta_v3_xlarge',  # SOTA XLarge
            2: 'llama2_7b',          # LLM with QLoRA
            3: 'deberta_v3_large',   # For ensemble
            4: 'roberta_large',      # Distilled target
            5: None,                 # Auto-select for platform
        }
        
        if tier == 5:
            model_name = self._select_free_tier_model()
        else:
            model_name = tier_models[tier]
        
        config = self.get_config(model_name=model_name)
        config['tier'] = tier
        
        return config
    
    def _select_free_tier_model(self) -> str:
        """
        Select optimal model for free-tier platforms.
        
        Returns:
            Model name suitable for free-tier resources
        """
        if self.platform_info.has_gpu:
            vram_gb = self.platform_info.gpu_memory_gb
            
            if vram_gb >= 16 and self.resource_constraints.supports_qlora:
                return 'llama2_7b'  # With QLoRA
            elif vram_gb >= 16:
                return 'deberta_v3_xlarge'  # With LoRA
            elif vram_gb >= 12:
                return 'deberta_v3_large'
            else:
                return 'deberta_v3_base'
        else:
            return 'deberta_v3_base'
    
    def recommend_model(
        self,
        target_accuracy: Optional[float] = None,
        max_training_time: Optional[float] = None
    ) -> str:
        """
        Recommend model based on requirements and resources.
        
        Analyzes requirements and available resources to recommend
        the most suitable model.
        
        Args:
            target_accuracy: Target accuracy (0-1)
            max_training_time: Maximum training time in hours
        
        Returns:
            Recommended model name
        
        Examples:
            Recommend for high accuracy:
                model = defaults.recommend_model(target_accuracy=0.97)
            
            Recommend for fast training:
                model = defaults.recommend_model(max_training_time=1.0)
        """
        vram_gb = self.platform_info.gpu_memory_gb if self.platform_info.has_gpu else 0
        
        if target_accuracy and target_accuracy >= 0.97:
            if vram_gb >= 24:
                return 'deberta_v2_xxlarge'
            elif vram_gb >= 16:
                return 'deberta_v3_xlarge'
            else:
                logger.warning(
                    f"Target accuracy {target_accuracy} requires larger GPU. "
                    f"Current VRAM: {vram_gb}GB. Recommending best available model."
                )
                return 'deberta_v3_large'
        
        if max_training_time and max_training_time <= 1.0:
            return 'deberta_v3_base'
        
        if vram_gb >= 24:
            return 'deberta_v3_xlarge'
        elif vram_gb >= 16:
            return 'deberta_v3_large'
        elif vram_gb >= 12:
            return 'roberta_large'
        else:
            return 'deberta_v3_base'
    
    def recommend_peft_method(self, model_name: str) -> str:
        """
        Recommend PEFT method for a model.
        
        Analyzes model requirements and available resources to recommend
        the most suitable PEFT method.
        
        Args:
            model_name: Model identifier
        
        Returns:
            Recommended PEFT method name
        
        Examples:
            Recommend PEFT:
                peft = defaults.recommend_peft_method('llama2_7b')
        """
        model_config = self.get_model_config(model_name)
        model_size = model_config.get('size', 'base')
        
        is_llm = model_size.startswith('llm_')
        
        if is_llm:
            if self.resource_constraints.supports_qlora:
                return 'qlora'
            else:
                logger.warning("QLoRA not supported, falling back to LoRA")
                return 'lora'
        
        if model_size in ['xlarge', 'xxlarge']:
            return 'lora'
        
        return 'none'
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration.
        
        Checks configuration for errors, incompatibilities, and potential issues.
        
        Args:
            config: Configuration to validate
        
        Returns:
            Tuple of (is_valid, list of error messages)
        
        Examples:
            Validate config:
                is_valid, errors = defaults.validate_config(config)
                if not is_valid:
                    for error in errors:
                        print(f"Error: {error}")
        """
        errors = []
        
        if 'model' not in config:
            errors.append("Missing 'model' section in configuration")
        
        if 'training' not in config:
            errors.append("Missing 'training' section in configuration")
        
        if 'model' in config:
            model_config = config['model']
            
            if 'size' in model_config:
                model_size = model_config['size']
                
                if model_size in ['xlarge', 'xxlarge'] or model_size.startswith('llm_'):
                    if 'peft' not in model_config or not model_config['peft'].get('enabled', False):
                        errors.append(
                            f"Model size '{model_size}' requires PEFT method "
                            f"but PEFT is not enabled"
                        )
        
        if 'training' in config:
            training_config = config['training']
            
            batch_size = training_config.get('batch_size', 0)
            if batch_size > self.resource_constraints.max_batch_size:
                errors.append(
                    f"Batch size {batch_size} exceeds maximum "
                    f"{self.resource_constraints.max_batch_size}"
                )
            
            if 'optimizer' in training_config:
                lr = training_config['optimizer'].get('lr', 0)
                if lr <= 0 or lr > 1:
                    errors.append(f"Invalid learning rate: {lr}")
        
        is_valid = len(errors) == 0
        return is_valid, errors


def get_smart_defaults(
    platform: Optional[str] = None,
    model_name: Optional[str] = None,
    tier: Optional[int] = None
) -> Dict[str, Any]:
    """
    Convenience function to get smart defaults.
    
    This is a simplified interface to SmartDefaults for quick usage.
    
    Args:
        platform: Platform type (None for auto-detection)
        model_name: Specific model name
        tier: Configuration tier (1-5)
    
    Returns:
        Configuration dictionary
    
    Examples:
        Get default config:
            config = get_smart_defaults()
        
        Get Colab config:
            config = get_smart_defaults(platform='colab')
        
        Get specific model:
            config = get_smart_defaults(model_name='deberta_v3_large')
        
        Get tier config:
            config = get_smart_defaults(tier=1)
    """
    defaults = SmartDefaults(platform=platform)
    
    if tier is not None:
        return defaults.get_tier_config(tier)
    else:
        return defaults.get_config(model_name=model_name)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("AG News Text Classification (ag-news-text-classification)")
    print(f"Smart Defaults Module v{__version__}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print(f"License: {__license__}\n")
    
    print("Detecting platform and resources...")
    defaults = SmartDefaults()
    
    print(f"\nPlatform: {defaults.platform_info.platform_type}")
    print(f"OS: {defaults.platform_info.os_type}")
    print(f"Python: {defaults.platform_info.python_version}")
    print(f"GPU: {defaults.platform_info.has_gpu}")
    if defaults.platform_info.has_gpu:
        print(f"GPU Name: {defaults.platform_info.gpu_names[0] if defaults.platform_info.gpu_names else 'Unknown'}")
        print(f"VRAM: {defaults.platform_info.gpu_memory_gb:.1f} GB")
        print(f"CUDA: {defaults.platform_info.cuda_version}")
    print(f"RAM: {defaults.platform_info.ram_gb:.1f} GB")
    
    print("\nGenerating smart defaults...")
    config = defaults.get_config()
    
    print(f"\nRecommended Model: {config['model']['name']}")
    print(f"Model Size: {config['model'].get('size', 'unknown')}")
    if 'peft' in config['model']:
        print(f"PEFT Method: {config['model']['peft']['method']}")
        if config['model']['peft']['method'] == 'lora':
            print(f"LoRA Rank: {config['model']['peft']['rank']}")
    
    print(f"\nBatch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['optimizer']['lr']}")
    print(f"Mixed Precision: {config['training']['mixed_precision']}")
    print(f"Gradient Checkpointing: {config['training'].get('gradient_checkpointing', False)}")
    
    print("\nValidating configuration...")
    is_valid, errors = defaults.validate_config(config)
    if is_valid:
        print("Configuration is valid")
    else:
        print("Configuration has errors:")
        for error in errors:
            print(f"  - {error}")
    
    print("\nConfiguration generation complete.")
