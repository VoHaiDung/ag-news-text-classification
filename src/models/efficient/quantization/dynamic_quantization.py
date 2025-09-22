"""
Dynamic Quantization for Model Compression
===========================================

Implementation of dynamic quantization techniques for reducing model size
and improving inference speed, based on:
- Jacob et al. (2018): "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
- Zafrir et al. (2019): "Q8BERT: Quantized 8Bit BERT"
- Shen et al. (2020): "Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT"

Dynamic quantization quantizes weights ahead of time but activations dynamically
during inference, providing a good balance between performance and accuracy.

Mathematical Foundation:
Quantization: q = round(x / scale) + zero_point
Dequantization: x = scale * (q - zero_point)

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, Tuple, Union, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import (
    quantize_dynamic,
    QuantStub,
    DeQuantStub,
    prepare_qat,
    convert
)

from src.models.base.base_model import AGNewsBaseModel
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DynamicQuantizationConfig:
    """Configuration for dynamic quantization"""
    
    # Quantization settings
    dtype: torch.dtype = torch.qint8  # Quantization data type
    reduce_range: bool = True  # Reduce range for better accuracy
    
    # Target layers
    quantize_linear: bool = True
    quantize_embedding: bool = False
    quantize_attention: bool = True
    
    # Calibration
    calibration_method: str = "minmax"  # "minmax", "entropy", "percentile"
    calibration_samples: int = 1000
    percentile: float = 99.99  # For percentile calibration
    
    # QAT (Quantization Aware Training)
    use_qat: bool = False
    qat_epochs: int = 3
    qat_learning_rate: float = 1e-5
    
    # Mixed precision
    mixed_precision_layers: List[str] = None  # Layers to keep in fp32
    
    # Performance
    backend: str = "fbgemm"  # "fbgemm" for x86, "qnnpack" for ARM
    
    # Accuracy preservation
    sensitivity_analysis: bool = True
    max_accuracy_drop: float = 0.01  # Maximum acceptable accuracy drop
    
    # Post-quantization optimization
    optimize_for_inference: bool = True
    fold_batch_norm: bool = True
    fuse_modules: bool = True


class QuantizationCalibrator:
    """
    Calibrates quantization parameters using representative data.
    
    Determines optimal scale and zero-point values for quantization
    by analyzing activation distributions.
    """
    
    def __init__(self, config: DynamicQuantizationConfig):
        """
        Initialize calibrator.
        
        Args:
            config: Quantization configuration
        """
        self.config = config
        self.activation_stats = {}
        self.calibration_data = []
        
    def collect_stats(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        num_samples: Optional[int] = None
    ):
        """
        Collect activation statistics for calibration.
        
        Args:
            model: Model to calibrate
            dataloader: Data loader for calibration
            num_samples: Number of samples to use
        """
        model.eval()
        num_samples = num_samples or self.config.calibration_samples
        samples_collected = 0
        
        # Hook to collect activations
        handles = []
        
        def hook_fn(module, input, output):
            """Hook function to collect activation stats"""
            module_name = self._get_module_name(module, model)
            
            if module_name not in self.activation_stats:
                self.activation_stats[module_name] = {
                    'min': [],
                    'max': [],
                    'mean': [],
                    'std': []
                }
            
            # Collect statistics
            if isinstance(output, torch.Tensor):
                self.activation_stats[module_name]['min'].append(output.min().item())
                self.activation_stats[module_name]['max'].append(output.max().item())
                self.activation_stats[module_name]['mean'].append(output.mean().item())
                self.activation_stats[module_name]['std'].append(output.std().item())
        
        # Register hooks for target layers
        for name, module in model.named_modules():
            if self._should_quantize_module(name, module):
                handle = module.register_forward_hook(hook_fn)
                handles.append(handle)
        
        # Collect statistics
        with torch.no_grad():
            for batch in dataloader:
                if samples_collected >= num_samples:
                    break
                
                inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                _ = model(inputs)
                
                samples_collected += inputs.shape[0]
        
        # Remove hooks
        for handle in handles:
            handle.remove()
        
        # Process statistics
        self._process_stats()
        
        logger.info(f"Collected calibration stats from {samples_collected} samples")
    
    def _should_quantize_module(self, name: str, module: nn.Module) -> bool:
        """Check if module should be quantized"""
        if self.config.mixed_precision_layers and name in self.config.mixed_precision_layers:
            return False
        
        if isinstance(module, nn.Linear) and self.config.quantize_linear:
            return True
        if isinstance(module, nn.Embedding) and self.config.quantize_embedding:
            return True
        if 'attention' in name.lower() and self.config.quantize_attention:
            return True
        
        return False
    
    def _get_module_name(self, module: nn.Module, model: nn.Module) -> str:
        """Get module name from model"""
        for name, mod in model.named_modules():
            if mod is module:
                return name
        return "unknown"
    
    def _process_stats(self):
        """Process collected statistics"""
        for module_name, stats in self.activation_stats.items():
            if self.config.calibration_method == "minmax":
                # Use min-max values
                scale = (max(stats['max']) - min(stats['min'])) / 255
                zero_point = -min(stats['min']) / scale
                
            elif self.config.calibration_method == "percentile":
                # Use percentile values
                import numpy as np
                min_val = np.percentile(stats['min'], 100 - self.config.percentile)
                max_val = np.percentile(stats['max'], self.config.percentile)
                scale = (max_val - min_val) / 255
                zero_point = -min_val / scale
                
            elif self.config.calibration_method == "entropy":
                # Use entropy-based calibration
                # Simplified version
                mean = np.mean(stats['mean'])
                std = np.mean(stats['std'])
                min_val = mean - 3 * std
                max_val = mean + 3 * std
                scale = (max_val - min_val) / 255
                zero_point = -min_val / scale
            
            else:
                scale = 1.0
                zero_point = 0
            
            self.activation_stats[module_name]['scale'] = scale
            self.activation_stats[module_name]['zero_point'] = int(zero_point)


class DynamicQuantizer:
    """
    Main class for dynamic quantization of models.
    
    Handles the quantization process including calibration,
    conversion, and optimization.
    """
    
    def __init__(self, config: Optional[DynamicQuantizationConfig] = None):
        """
        Initialize dynamic quantizer.
        
        Args:
            config: Quantization configuration
        """
        self.config = config or DynamicQuantizationConfig()
        self.calibrator = QuantizationCalibrator(config)
        
        # Set backend
        torch.backends.quantized.engine = self.config.backend
        
        logger.info(f"Initialized DynamicQuantizer with backend: {self.config.backend}")
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.utils.data.DataLoader] = None
    ) -> nn.Module:
        """
        Quantize model using dynamic quantization.
        
        Args:
            model: Model to quantize
            calibration_data: Data for calibration
            
        Returns:
            Quantized model
        """
        logger.info("Starting dynamic quantization...")
        
        # Prepare model
        model.eval()
        
        # Fuse modules if configured
        if self.config.fuse_modules:
            model = self._fuse_modules(model)
        
        # Calibrate if data provided
        if calibration_data is not None:
            self.calibrator.collect_stats(model, calibration_data)
        
        # Apply dynamic quantization
        quantized_model = self._apply_dynamic_quantization(model)
        
        # Optimize for inference
        if self.config.optimize_for_inference:
            quantized_model = self._optimize_for_inference(quantized_model)
        
        # Validate quantization
        if self.config.sensitivity_analysis:
            self._sensitivity_analysis(model, quantized_model, calibration_data)
        
        logger.info("Dynamic quantization completed")
        return quantized_model
    
    def _apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model"""
        # Determine layers to quantize
        qconfig_dict = {}
        
        for name, module in model.named_modules():
            if self._should_quantize(name, module):
                qconfig_dict[name] = torch.quantization.default_dynamic_qconfig
        
        # Apply quantization
        quantized_model = quantize_dynamic(
            model,
            qconfig_spec=qconfig_dict,
            dtype=self.config.dtype,
            mapping=None,
            inplace=False
        )
        
        return quantized_model
    
    def _should_quantize(self, name: str, module: nn.Module) -> bool:
        """Determine if module should be quantized"""
        # Skip if in mixed precision list
        if self.config.mixed_precision_layers and name in self.config.mixed_precision_layers:
            return False
        
        # Check module type
        if isinstance(module, nn.Linear) and self.config.quantize_linear:
            return True
        if isinstance(module, nn.Embedding) and self.config.quantize_embedding:
            return True
        if 'attention' in name.lower() and self.config.quantize_attention:
            return True
        
        return False
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse modules for better quantization"""
        # Common fusion patterns
        fusion_patterns = [
            ['conv', 'bn', 'relu'],
            ['conv', 'bn'],
            ['linear', 'relu'],
            ['linear', 'bn']
        ]
        
        # Find and fuse matching patterns
        for pattern in fusion_patterns:
            modules_to_fuse = self._find_fusion_patterns(model, pattern)
            if modules_to_fuse:
                torch.quantization.fuse_modules(
                    model,
                    modules_to_fuse,
                    inplace=True
                )
        
        return model
    
    def _find_fusion_patterns(
        self,
        model: nn.Module,
        pattern: List[str]
    ) -> List[List[str]]:
        """Find modules matching fusion pattern"""
        modules_to_fuse = []
        
        # Simplified pattern matching
        # In practice, would be more sophisticated
        module_names = list(model.named_modules())
        
        for i in range(len(module_names) - len(pattern) + 1):
            match = True
            fusion_group = []
            
            for j, pattern_type in enumerate(pattern):
                name, module = module_names[i + j]
                
                if pattern_type == 'conv' and not isinstance(module, nn.Conv2d):
                    match = False
                    break
                elif pattern_type == 'linear' and not isinstance(module, nn.Linear):
                    match = False
                    break
                elif pattern_type == 'bn' and not isinstance(module, nn.BatchNorm2d):
                    match = False
                    break
                elif pattern_type == 'relu' and not isinstance(module, nn.ReLU):
                    match = False
                    break
                
                fusion_group.append(name)
            
            if match and fusion_group:
                modules_to_fuse.append(fusion_group)
        
        return modules_to_fuse
    
    def _optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Optimize quantized model for inference"""
        # Remove unnecessary operations
        model.eval()
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        # Fold batch normalization if present
        if self.config.fold_batch_norm:
            model = self._fold_batch_norm(model)
        
        return model
    
    def _fold_batch_norm(self, model: nn.Module) -> nn.Module:
        """Fold batch normalization into preceding layers"""
        # Simplified BN folding
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                # Find preceding conv layer
                # In practice, would implement proper folding
                pass
        
        return model
    
    def _sensitivity_analysis(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_data: Optional[torch.utils.data.DataLoader]
    ):
        """Analyze sensitivity of quantization"""
        if test_data is None:
            logger.warning("No test data provided for sensitivity analysis")
            return
        
        # Evaluate both models
        original_acc = self._evaluate_model(original_model, test_data)
        quantized_acc = self._evaluate_model(quantized_model, test_data)
        
        accuracy_drop = original_acc - quantized_acc
        
        logger.info(
            f"Sensitivity Analysis:\n"
            f"  Original accuracy: {original_acc:.4f}\n"
            f"  Quantized accuracy: {quantized_acc:.4f}\n"
            f"  Accuracy drop: {accuracy_drop:.4f}"
        )
        
        if accuracy_drop > self.config.max_accuracy_drop:
            logger.warning(
                f"Accuracy drop ({accuracy_drop:.4f}) exceeds threshold "
                f"({self.config.max_accuracy_drop:.4f})"
            )
    
    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> float:
        """Evaluate model accuracy"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, labels = batch[:2]
                outputs = model(inputs)
                
                if hasattr(outputs, 'logits'):
                    predictions = outputs.logits.argmax(dim=-1)
                else:
                    predictions = outputs.argmax(dim=-1)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        return correct / total if total > 0 else 0
    
    def get_model_size(self, model: nn.Module) -> Dict[str, float]:
        """Get model size statistics"""
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'param_size_mb': param_size / 1024 / 1024,
            'buffer_size_mb': buffer_size / 1024 / 1024,
            'total_size_mb': total_size / 1024 / 1024
        }


@MODELS.register("quantized_model")
class QuantizedAGNewsModel(AGNewsBaseModel):
    """
    Quantized model wrapper for AG News classification.
    
    Wraps any AG News model with dynamic quantization for
    efficient inference.
    """
    
    def __init__(
        self,
        base_model: AGNewsBaseModel,
        config: Optional[DynamicQuantizationConfig] = None
    ):
        """
        Initialize quantized model.
        
        Args:
            base_model: Base model to quantize
            config: Quantization configuration
        """
        super().__init__()
        
        self.config = config or DynamicQuantizationConfig()
        self.quantizer = DynamicQuantizer(config)
        
        # Quantize base model
        self.quantized_model = self.quantizer.quantize_model(base_model)
        
        # Store size reduction
        original_size = self.quantizer.get_model_size(base_model)
        quantized_size = self.quantizer.get_model_size(self.quantized_model)
        
        self.compression_ratio = original_size['total_size_mb'] / quantized_size['total_size_mb']
        
        logger.info(
            f"Model quantized: {original_size['total_size_mb']:.2f}MB -> "
            f"{quantized_size['total_size_mb']:.2f}MB "
            f"(Compression: {self.compression_ratio:.2f}x)"
        )
    
    def forward(self, *args, **kwargs):
        """Forward pass through quantized model"""
        return self.quantized_model(*args, **kwargs)
