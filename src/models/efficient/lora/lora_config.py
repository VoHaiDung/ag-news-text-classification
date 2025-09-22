"""
LoRA Configuration Module
=========================

Configuration management for Low-Rank Adaptation (LoRA) of large language models.

Based on:
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
- Dettmers et al. (2023): "QLoRA: Efficient Finetuning of Quantized LLMs"
- Zhang et al. (2023): "AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning"

Mathematical Foundation:
LoRA decomposes weight updates as: W' = W + BA where B ∈ R^(d×r), A ∈ R^(r×k)
with rank r << min(d, k), reducing parameters from d×k to r×(d+k).

Author: Võ Hải Dũng
License: MIT
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Any
from enum import Enum
import json
from pathlib import Path
import logging

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class LoRATargetModule(Enum):
    """Target modules for LoRA adaptation"""
    QUERY = "query"
    KEY = "key"
    VALUE = "value"
    OUTPUT = "output"
    INTERMEDIATE = "intermediate"
    ALL_LINEAR = "all_linear"
    CUSTOM = "custom"


class LoRAInitMethod(Enum):
    """Initialization methods for LoRA matrices"""
    GAUSSIAN = "gaussian"  # N(0, σ²)
    UNIFORM = "uniform"    # U(-a, a)
    XAVIER = "xavier"      # Xavier/Glorot initialization
    KAIMING = "kaiming"    # He initialization
    ZEROS = "zeros"        # Zero initialization for B matrix
    ORTHOGONAL = "orthogonal"  # Orthogonal initialization


@dataclass
class LoRAConfig:
    """
    Configuration for LoRA adaptation.
    
    Comprehensive configuration for Low-Rank Adaptation including:
    - Rank and scaling parameters
    - Target module selection
    - Initialization strategies
    - Training hyperparameters
    - Memory optimization settings
    """
    
    # Core LoRA parameters
    r: int = 8  # LoRA rank (r << d)
    lora_alpha: float = 16.0  # LoRA scaling factor α
    lora_dropout: float = 0.1  # Dropout probability for LoRA layers
    
    # Target modules configuration
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"  # Standard attention projections
    ])
    modules_to_save: List[str] = field(default_factory=lambda: [
        "classifier", "lm_head"  # Task-specific layers
    ])
    
    # Advanced targeting
    target_module_pattern: Optional[str] = None  # Regex pattern for module selection
    exclude_modules: List[str] = field(default_factory=list)  # Modules to exclude
    layer_wise_rank: bool = False  # Different ranks for different layers
    
    # Initialization configuration
    init_method_a: LoRAInitMethod = LoRAInitMethod.GAUSSIAN
    init_method_b: LoRAInitMethod = LoRAInitMethod.ZEROS
    init_std: float = 0.02  # Standard deviation for Gaussian init
    use_kaiming_scale: bool = True  # Scale initialization by sqrt(2/n)
    
    # Training configuration
    bias: str = "none"  # Options: "none", "all", "lora_only"
    fan_in_fan_out: bool = False  # Set for Conv1D layers (e.g., GPT-2)
    merge_weights: bool = False  # Merge LoRA weights during inference
    
    # Rank adaptation (AdaLoRA style)
    enable_rank_adaptation: bool = False
    initial_rank: int = 12  # Starting rank for adaptation
    target_rank: int = 4    # Target rank after pruning
    rank_pattern: Dict[str, int] = field(default_factory=dict)  # Per-layer ranks
    importance_metric: str = "magnitude"  # "magnitude", "gradient", "fisher"
    
    # Quantization support (QLoRA)
    enable_quantization: bool = False
    quantization_bits: int = 8  # 4-bit or 8-bit quantization
    double_quantization: bool = False  # QLoRA-style double quantization
    quantization_type: str = "nf4"  # "int8", "nf4", "fp4"
    
    # Memory optimization
    gradient_checkpointing: bool = False
    use_rslora: bool = False  # Rank-Stabilized LoRA
    use_dora: bool = False    # Weight-Decomposed LoRA
    memory_efficient_backward: bool = True
    
    # Multi-LoRA configuration
    enable_multi_lora: bool = False
    num_lora_modules: int = 1  # Number of parallel LoRA modules
    lora_composition: str = "add"  # "add", "concat", "gate"
    
    # Performance settings
    compute_dtype: Optional[str] = None  # "float16", "bfloat16", "float32"
    enable_gradient_caching: bool = True
    optimize_memory: bool = True
    
    # Experimental features
    use_adaptive_alpha: bool = False  # Dynamic α adjustment
    orthogonal_regularization: bool = False  # Orthogonality constraint
    importance_sampling: bool = False  # Importance-based rank allocation
    svd_init: bool = False  # Initialize from SVD of pretrained weights
    
    # Metadata
    model_id: str = "model"
    task_type: str = "classification"
    dataset_name: str = "ag_news"
    revision: str = "main"
    
    def __post_init__(self):
        """Validate and process configuration"""
        self._validate_config()
        self._compute_derived_params()
        self._setup_layer_ranks()
    
    def _validate_config(self):
        """Validate configuration parameters"""
        # Validate rank
        if self.r <= 0:
            raise ValueError(f"LoRA rank must be positive, got {self.r}")
        
        # Validate alpha
        if self.lora_alpha <= 0:
            raise ValueError(f"LoRA alpha must be positive, got {self.lora_alpha}")
        
        # Validate dropout
        if not 0 <= self.lora_dropout < 1:
            raise ValueError(f"Dropout must be in [0, 1), got {self.lora_dropout}")
        
        # Validate bias mode
        valid_bias = ["none", "all", "lora_only"]
        if self.bias not in valid_bias:
            raise ValueError(f"Bias must be one of {valid_bias}, got {self.bias}")
        
        # Validate rank adaptation
        if self.enable_rank_adaptation:
            if self.initial_rank < self.target_rank:
                raise ValueError("Initial rank must be >= target rank")
            if self.importance_metric not in ["magnitude", "gradient", "fisher"]:
                raise ValueError(f"Invalid importance metric: {self.importance_metric}")
        
        # Validate quantization
        if self.enable_quantization:
            if self.quantization_bits not in [4, 8]:
                raise ValueError("Quantization bits must be 4 or 8")
            if self.quantization_type not in ["int8", "nf4", "fp4"]:
                raise ValueError(f"Invalid quantization type: {self.quantization_type}")
    
    def _compute_derived_params(self):
        """Compute derived parameters"""
        # Scaling factor for LoRA
        self.scaling = self.lora_alpha / self.r
        
        # Effective number of parameters
        # Original: d × k, LoRA: r × (d + k)
        # Compression ratio ≈ dk / r(d+k) ≈ dk/2r√(dk) for d ≈ k
        
        # Memory footprint estimation (in MB)
        # Assuming d=768 (BERT-base dimension)
        d = 768  # Hidden dimension
        num_target_modules = len(self.target_modules)
        
        # LoRA parameters per module: r × (2d)
        params_per_module = self.r * (2 * d)
        self.total_lora_params = params_per_module * num_target_modules
        
        # Memory in MB (float32)
        self.memory_footprint_mb = (self.total_lora_params * 4) / (1024 * 1024)
        
        if self.enable_quantization:
            self.memory_footprint_mb *= (self.quantization_bits / 32)
        
        # Compute theoretical speedup
        self.theoretical_speedup = d * d / (self.r * (2 * d))
        
        logger.info(
            f"LoRA configuration: rank={self.r}, "
            f"parameters={self.total_lora_params:,}, "
            f"memory={self.memory_footprint_mb:.2f}MB, "
            f"speedup={self.theoretical_speedup:.1f}x"
        )
    
    def _setup_layer_ranks(self):
        """Setup layer-wise ranks if enabled"""
        if self.layer_wise_rank and not self.rank_pattern:
            # Default pattern: decrease rank for higher layers
            self.rank_pattern = {
                "layer_0": self.r,
                "layer_1-3": max(self.r - 2, 1),
                "layer_4-7": max(self.r - 4, 1),
                "layer_8-11": max(self.r - 6, 1)
            }
    
    @property
    def scaling_factor(self) -> float:
        """Get LoRA scaling factor"""
        return self.lora_alpha / self.r
    
    def get_target_modules_regex(self) -> str:
        """Get regex pattern for target modules"""
        if self.target_module_pattern:
            return self.target_module_pattern
        
        # Build pattern from target modules list
        modules = [f".*{module}.*" for module in self.target_modules]
        return f"({'|'.join(modules)})"
    
    def should_adapt_module(self, module_name: str) -> bool:
        """
        Check if a module should be adapted with LoRA.
        
        Args:
            module_name: Name of the module
            
        Returns:
            True if module should be adapted
        """
        # Check exclusion list first
        if any(excluded in module_name for excluded in self.exclude_modules):
            return False
        
        # Check target modules
        return any(target in module_name for target in self.target_modules)
    
    def get_module_rank(self, module_name: str) -> int:
        """
        Get rank for specific module.
        
        Args:
            module_name: Name of the module
            
        Returns:
            Rank for the module
        """
        if not self.layer_wise_rank:
            return self.r
        
        # Check for module-specific rank
        for pattern, rank in self.rank_pattern.items():
            if pattern in module_name:
                return rank
            
            # Handle range patterns (e.g., "layer_1-3")
            if '-' in pattern:
                prefix, range_str = pattern.rsplit('_', 1)
                if prefix in module_name:
                    start, end = map(int, range_str.split('-'))
                    # Extract layer number from module name
                    import re
                    match = re.search(r'(\d+)', module_name.split(prefix)[-1])
                    if match:
                        layer_num = int(match.group(1))
                        if start <= layer_num <= end:
                            return rank
        
        # Return default rank
        return self.r
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {
            'r': self.r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': self.target_modules,
            'modules_to_save': self.modules_to_save,
            'init_method_a': self.init_method_a.value,
            'init_method_b': self.init_method_b.value,
            'init_std': self.init_std,
            'bias': self.bias,
            'fan_in_fan_out': self.fan_in_fan_out,
            'merge_weights': self.merge_weights,
            'enable_rank_adaptation': self.enable_rank_adaptation,
            'enable_quantization': self.enable_quantization,
            'quantization_bits': self.quantization_bits,
            'gradient_checkpointing': self.gradient_checkpointing,
            'model_id': self.model_id,
            'task_type': self.task_type,
            'memory_footprint_mb': self.memory_footprint_mb,
            'total_lora_params': self.total_lora_params,
            'theoretical_speedup': self.theoretical_speedup
        }
        return config_dict
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save configuration to file.
        
        Args:
            path: Path to save configuration
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved LoRA configuration to {path}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LoRAConfig':
        """
        Create configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            LoRAConfig instance
        """
        # Convert string enums back to enum types
        if 'init_method_a' in config_dict:
            config_dict['init_method_a'] = LoRAInitMethod(config_dict['init_method_a'])
        if 'init_method_b' in config_dict:
            config_dict['init_method_b'] = LoRAInitMethod(config_dict['init_method_b'])
        
        # Remove computed fields
        config_dict.pop('memory_footprint_mb', None)
        config_dict.pop('total_lora_params', None)
        config_dict.pop('theoretical_speedup', None)
        
        return cls(**config_dict)
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'LoRAConfig':
        """
        Load configuration from file.
        
        Args:
            path: Path to configuration file
            
        Returns:
            LoRAConfig instance
        """
        path = Path(path)
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"LoRAConfig(\n"
            f"  r={self.r},\n"
            f"  alpha={self.lora_alpha},\n"
            f"  scaling={self.scaling:.2f},\n"
            f"  dropout={self.lora_dropout},\n"
            f"  target_modules={self.target_modules},\n"
            f"  params={self.total_lora_params:,},\n"
            f"  memory={self.memory_footprint_mb:.2f}MB\n"
            f")"
        )


# Preset configurations for common use cases
LORA_PRESETS = {
    "minimal": LoRAConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    ),
    
    "standard": LoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    ),
    
    "aggressive": LoRAConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"]
    ),
    
    "qlora": LoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        enable_quantization=True,
        quantization_bits=4,
        double_quantization=True,
        gradient_checkpointing=True
    ),
    
    "adaptive": LoRAConfig(
        r=12,
        lora_alpha=24,
        lora_dropout=0.1,
        enable_rank_adaptation=True,
        initial_rank=12,
        target_rank=4,
        use_adaptive_alpha=True
    ),
    
    "memory_efficient": LoRAConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        gradient_checkpointing=True,
        memory_efficient_backward=True,
        optimize_memory=True
    )
}


def get_lora_preset(preset_name: str) -> LoRAConfig:
    """
    Get preset LoRA configuration.
    
    Args:
        preset_name: Name of the preset
        
    Returns:
        LoRAConfig instance
    """
    if preset_name not in LORA_PRESETS:
        raise ValueError(
            f"Unknown preset: {preset_name}. "
            f"Available: {list(LORA_PRESETS.keys())}"
        )
    
    config = LORA_PRESETS[preset_name]
    logger.info(f"Loaded LoRA preset: {preset_name}")
    return config
