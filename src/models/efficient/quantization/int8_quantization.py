"""
INT8 Quantization for Model Compression
========================================

Implementation of INT8 quantization for efficient model deployment,
based on:
- Jacob et al. (2018): "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
- Zafrir et al. (2019): "Q8BERT: Quantized 8Bit BERT"
- Shen et al. (2020): "Q-BERT: Hessian Based Ultra Low Precision Quantization of BERT"

Mathematical Foundation:
Quantization maps floating point values to integers:
q = round(x / scale) + zero_point
x_reconstructed = scale * (q - zero_point)

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.quantization import (
    QuantStub,
    DeQuantStub,
    prepare_qat,
    convert,
    get_default_qat_qconfig
)

from src.models.base.base_model import AGNewsBaseModel, ModelOutputs
from src.core.registry import MODELS
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""
    quantization_type: str = "dynamic"  # "dynamic", "static", "qat"
    backend: str = "fbgemm"  # "fbgemm" (x86), "qnnpack" (ARM)
    calibration_samples: int = 1000
    symmetric: bool = True
    per_channel: bool = True
    reduce_range: bool = False
    
    # Quantization-aware training
    qat_epochs: int = 5
    qat_learning_rate: float = 1e-5
    
    # Mixed precision quantization
    mixed_precision: bool = False
    sensitive_layers: List[str] = None  # Keep these in FP32
    
    # Optimization settings
    optimize_for_mobile: bool = False
    fuse_modules: bool = True


class QuantizationCalibrator:
    """
    Calibrator for static quantization.
    
    Collects statistics for determining quantization parameters.
    """
    
    def __init__(self, num_samples: int = 1000):
        """
        Initialize calibrator.
        
        Args:
            num_samples: Number of calibration samples
        """
        self.num_samples = num_samples
        self.calibration_data = []
    
    def collect_stats(
        self,
        model: nn.Module,
        data_loader,
        device: str = "cpu"
    ):
        """
        Collect activation statistics for calibration.
        
        Args:
            model: Model to calibrate
            data_loader: Data loader for calibration
            device: Device to use
        """
        model.eval()
        model.to(device)
        
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if i >= self.num_samples:
                    break
                
                # Forward pass to collect statistics
                _ = model(
                    batch['input_ids'].to(device),
                    batch.get('attention_mask', None)
                )
        
        logger.info(f"Collected statistics from {i} samples")


class QuantizedLinear(nn.Module):
    """
    Quantized linear layer with INT8 operations.
    
    Implements quantized matrix multiplication for inference speedup.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        weight_bits: int = 8,
        activation_bits: int = 8
    ):
        """
        Initialize quantized linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias
            weight_bits: Bits for weight quantization
            activation_bits: Bits for activation quantization
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits
        
        # Quantized weights
        self.register_buffer(
            'weight_int',
            torch.zeros(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer('weight_scale', torch.ones(out_features))
        self.register_buffer('weight_zero_point', torch.zeros(out_features))
        
        # Bias
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.bias = None
        
        # Activation quantization parameters
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('input_zero_point', torch.tensor(0))
    
    def quantize_weight(self, weight: torch.Tensor):
        """
        Quantize floating point weights to INT8.
        
        Args:
            weight: Floating point weight tensor
        """
        # Calculate quantization parameters
        weight_min = weight.min(dim=1)[0]
        weight_max = weight.max(dim=1)[0]
        
        # Symmetric quantization
        weight_abs_max = torch.max(weight_min.abs(), weight_max.abs())
        self.weight_scale = weight_abs_max / 127.0
        self.weight_zero_point.zero_()
        
        # Quantize weights
        weight_int = torch.round(weight / self.weight_scale.unsqueeze(1))
        self.weight_int = weight_int.clamp(-128, 127).to(torch.int8)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantized forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Quantize input
        x_int = torch.round(x / self.input_scale).clamp(-128, 127).to(torch.int8)
        
        # INT8 matrix multiplication
        # Note: In practice, this would use optimized INT8 kernels
        output_int = torch.matmul(
            x_int.to(torch.float32),
            self.weight_int.t().to(torch.float32)
        )
        
        # Dequantize output
        output = output_int * self.weight_scale * self.input_scale
        
        # Add bias
        if self.bias is not None:
            output += self.bias
        
        return output


@MODELS.register("int8_quantized", aliases=["quantized", "int8"])
class INT8QuantizedModel(AGNewsBaseModel):
    """
    INT8 quantized model for efficient inference.
    
    Provides 4x model size reduction and 2-4x inference speedup
    with minimal accuracy loss through:
    1. Dynamic quantization for weights and activations
    2. Static quantization with calibration
    3. Quantization-aware training
    4. Mixed precision quantization
    """
    
    def __init__(
        self,
        base_model: AGNewsBaseModel,
        config: Optional[QuantizationConfig] = None
    ):
        """
        Initialize quantized model.
        
        Args:
            base_model: Model to quantize
            config: Quantization configuration
        """
        super().__init__()
        
        self.base_model = base_model
        self.config = config or QuantizationConfig()
        
        # Set quantization backend
        torch.backends.quantized.engine = self.config.backend
        
        # Apply quantization based on type
        if self.config.quantization_type == "dynamic":
            self._apply_dynamic_quantization()
        elif self.config.quantization_type == "static":
            self._prepare_static_quantization()
        elif self.config.quantization_type == "qat":
            self._prepare_qat()
        
        # Log model size reduction
        self._log_compression_stats()
    
    def _apply_dynamic_quantization(self):
        """Apply dynamic quantization to model."""
        # Modules to quantize
        modules_to_quantize = [nn.Linear, nn.Conv2d]
        
        # Apply dynamic quantization
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.base_model,
            modules_to_quantize,
            dtype=torch.qint8
        )
        
        logger.info("Applied dynamic INT8 quantization")
    
    def _prepare_static_quantization(self):
        """Prepare model for static quantization."""
        # Add quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        
        # Fuse modules if configured
        if self.config.fuse_modules:
            self._fuse_modules()
        
        # Prepare for quantization
        self.base_model.qconfig = torch.quantization.get_default_qconfig(
            self.config.backend
        )
        
        self.prepared_model = torch.quantization.prepare(
            self.base_model,
            inplace=False
        )
        
        logger.info("Prepared model for static quantization")
    
    def _prepare_qat(self):
        """Prepare model for quantization-aware training."""
        # Set QAT config
        self.base_model.qconfig = get_default_qat_qconfig(self.config.backend)
        
        # Prepare for QAT
        self.qat_model = prepare_qat(self.base_model, inplace=False)
        
        logger.info("Prepared model for quantization-aware training")
    
    def _fuse_modules(self):
        """Fuse modules for optimized quantization."""
        # Identify and fuse conv-bn-relu patterns
        modules_to_fuse = []
        
        for name, module in self.base_model.named_modules():
            # Look for common patterns to fuse
            if isinstance(module, nn.Sequential):
                # Check for Conv-BN-ReLU pattern
                if (len(module) >= 2 and
                    isinstance(module[0], nn.Linear) and
                    isinstance(module[1], (nn.BatchNorm1d, nn.LayerNorm))):
                    modules_to_fuse.append([f"{name}.0", f"{name}.1"])
        
        if modules_to_fuse:
            torch.quantization.fuse_modules(
                self.base_model,
                modules_to_fuse,
                inplace=True
            )
            logger.info(f"Fused {len(modules_to_fuse)} module groups")
    
    def calibrate(self, data_loader):
        """
        Calibrate quantization parameters on data.
        
        Args:
            data_loader: Data loader for calibration
        """
        if self.config.quantization_type != "static":
            logger.warning("Calibration only needed for static quantization")
            return
        
        calibrator = QuantizationCalibrator(
            num_samples=self.config.calibration_samples
        )
        
        calibrator.collect_stats(
            self.prepared_model,
            data_loader,
            device="cpu"  # Quantization typically on CPU
        )
        
        # Convert to quantized model
        self.quantized_model = torch.quantization.convert(
            self.prepared_model,
            inplace=False
        )
        
        logger.info("Completed calibration and conversion to INT8")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ) -> ModelOutputs:
        """
        Forward pass through quantized model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Target labels
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        # Use appropriate model based on quantization type
        if self.config.quantization_type == "dynamic":
            model = self.quantized_model
        elif self.config.quantization_type == "static" and hasattr(self, 'quantized_model'):
            model = self.quantized_model
        elif self.config.quantization_type == "qat":
            model = self.qat_model
        else:
            model = self.base_model
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )
        
        return outputs
    
    def _log_compression_stats(self):
        """Log model compression statistics."""
        # Calculate model sizes
        def get_model_size(model):
            param_size = 0
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            buffer_size = 0
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            return (param_size + buffer_size) / 1024 / 1024  # MB
        
        original_size = get_model_size(self.base_model)
        
        if hasattr(self, 'quantized_model'):
            quantized_size = get_model_size(self.quantized_model)
            compression_ratio = original_size / quantized_size
            
            logger.info(
                f"Model Compression Statistics:\n"
                f"  Original size: {original_size:.2f} MB\n"
                f"  Quantized size: {quantized_size:.2f} MB\n"
                f"  Compression ratio: {compression_ratio:.2f}x\n"
                f"  Size reduction: {(1 - quantized_size/original_size)*100:.1f}%"
            )
    
    def benchmark_inference(
        self,
        input_shape: Tuple[int, int] = (1, 512),
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.
        
        Args:
            input_shape: Input tensor shape
            num_iterations: Number of benchmark iterations
            
        Returns:
            Performance metrics
        """
        import time
        
        # Create dummy input
        dummy_input = torch.randint(0, 1000, input_shape)
        
        # Warmup
        for _ in range(10):
            _ = self.forward(dummy_input)
        
        # Benchmark original model
        start = time.perf_counter()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.base_model(dummy_input)
        original_time = time.perf_counter() - start
        
        # Benchmark quantized model
        start = time.perf_counter()
        for _ in range(num_iterations):
            with torch.no_grad():
                _ = self.forward(dummy_input)
        quantized_time = time.perf_counter() - start
        
        speedup = original_time / quantized_time
        
        return {
            "original_time_ms": (original_time / num_iterations) * 1000,
            "quantized_time_ms": (quantized_time / num_iterations) * 1000,
            "speedup": speedup,
            "throughput_improvement": f"{speedup:.2f}x"
        }
    
    def export_onnx(self, export_path: str):
        """
        Export quantized model to ONNX format.
        
        Args:
            export_path: Path to save ONNX model
        """
        dummy_input = torch.randint(0, 1000, (1, 512))
        
        torch.onnx.export(
            self.quantized_model if hasattr(self, 'quantized_model') else self.base_model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'},
                'logits': {0: 'batch_size'}
            }
        )
        
        logger.info(f"Exported quantized model to {export_path}")
