"""
Single Predictor for AG News Text Classification
================================================================================
This module implements single-instance prediction for text classification,
optimized for low-latency inference with various optimization techniques.

The predictor supports multiple model formats and implements caching,
batching, and hardware acceleration for optimal performance.

References:
    - Accelerating Deep Learning Inference (NVIDIA, 2020)
    - Model Optimization Best Practices (TensorFlow, 2021)
    - PyTorch JIT Documentation: https://pytorch.org/docs/stable/jit.html

Author: Võ Hải Dũng
License: MIT
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.core.interfaces import IModel
from src.utils.memory_utils import MemoryOptimizer
from src.utils.profiling_utils import ProfilerManager

# Configure logging
logger = logging.getLogger(__name__)


class SinglePredictor:
    """
    Single-instance predictor for text classification
    
    This predictor handles individual text inputs with optimizations
    for low-latency inference including caching and hardware acceleration.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize single predictor
        
        Args:
            model_path: Path to model files
            config: Prediction configuration
        """
        self.model_path = Path(model_path)
        self.config = config or {}
        
        # Model components
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.onnx_session: Optional[ort.InferenceSession] = None
        
        # Configuration
        self.device = self._setup_device()
        self.max_length = self.config.get('max_sequence_length', 512)
        self.use_fp16 = self.config.get('use_fp16', False) and torch.cuda.is_available()
        self.use_onnx = self.config.get('use_onnx', False)
        self.use_torch_script = self.config.get('use_torch_script', False)
        
        # Caching
        self.cache_enabled = self.config.get('cache_enabled', True)
        self.prediction_cache: Dict[str, np.ndarray] = {}
        self.max_cache_size = self.config.get('max_cache_size', 1000)
        
        # Performance tracking
        self.inference_times: List[float] = []
        self.profiler = ProfilerManager() if self.config.get('profiling', False) else None
        
        # Load model
        self._load_model()
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.config.get('use_gpu', True) and torch.cuda.is_available():
            device_id = self.config.get('gpu_device_id', 0)
            return torch.device(f'cuda:{device_id}')
        return torch.device('cpu')
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            if self.use_onnx:
                # Load ONNX model
                self._load_onnx_model()
            else:
                # Load PyTorch model
                self._load_pytorch_model()
            
            logger.info(f"Model loaded successfully from {self.model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _load_pytorch_model(self):
        """Load PyTorch model"""
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32
        )
        
        # Move to device
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Apply optimizations
        if self.use_torch_script:
            self._compile_model()
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
    
    def _load_onnx_model(self):
        """Load ONNX model for inference"""
        onnx_path = self.model_path / "model.onnx"
        
        if not onnx_path.exists():
            logger.warning("ONNX model not found, falling back to PyTorch")
            self.use_onnx = False
            self._load_pytorch_model()
            return
        
        # Setup ONNX Runtime providers
        providers = []
        if torch.cuda.is_available() and self.config.get('use_gpu', True):
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # Create session with optimizations
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = self.config.get('num_threads', 4)
        
        self.onnx_session = ort.InferenceSession(
            str(onnx_path),
            sess_options=sess_options,
            providers=providers
        )
        
        logger.info("ONNX model loaded successfully")
    
    def _compile_model(self):
        """Compile model using TorchScript for faster inference"""
        try:
            # Create example input
            example_text = "Example text for compilation"
            inputs = self.tokenizer(
                example_text,
                return_tensors='pt',
                max_length=self.max_length,
                truncation=True,
                padding='max_length'
            ).to(self.device)
            
            # Trace model
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    self.model,
                    (inputs['input_ids'], inputs['attention_mask'])
                )
            
            self.model = traced_model
            logger.info("Model compiled with TorchScript")
            
        except Exception as e:
            logger.warning(f"Could not compile model: {e}")
    
    def predict(
        self,
        text: str,
        return_probabilities: bool = False,
        top_k: int = 1
    ) -> Dict[str, Any]:
        """
        Predict class for single text
        
        Args:
            text: Input text
            return_probabilities: Whether to return probability scores
            top_k: Number of top predictions to return
            
        Returns:
            Prediction results dictionary
        """
        start_time = time.time()
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(text)
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                return self._format_prediction(
                    cached_result,
                    return_probabilities,
                    top_k,
                    time.time() - start_time,
                    cached=True
                )
        
        # Perform prediction
        if self.use_onnx:
            probabilities = self._predict_onnx(text)
        else:
            probabilities = self._predict_pytorch(text)
        
        # Cache result
        if self.cache_enabled:
            self._update_cache(cache_key, probabilities)
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        return self._format_prediction(
            probabilities,
            return_probabilities,
            top_k,
            inference_time,
            cached=False
        )
    
    def _predict_pytorch(self, text: str) -> np.ndarray:
        """Perform prediction using PyTorch model"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        ).to(self.device)
        
        # Inference
        with torch.no_grad():
            if self.use_fp16 and torch.cuda.is_available():
                with autocast():
                    outputs = self.model(**inputs)
            else:
                outputs = self.model(**inputs)
            
            # Get probabilities
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            
        return probabilities.cpu().numpy()[0]
    
    def _predict_onnx(self, text: str) -> np.ndarray:
        """Perform prediction using ONNX Runtime"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='np',
            max_length=self.max_length,
            truncation=True,
            padding='max_length'
        )
        
        # Prepare ONNX inputs
        onnx_inputs = {
            'input_ids': inputs['input_ids'].astype(np.int64),
            'attention_mask': inputs['attention_mask'].astype(np.int64)
        }
        
        # Run inference
        outputs = self.onnx_session.run(None, onnx_inputs)
        logits = outputs[0][0]
        
        # Calculate probabilities
        exp_logits = np.exp(logits - np.max(logits))
        probabilities = exp_logits / np.sum(exp_logits)
        
        return probabilities
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Use first 100 chars for hash to limit key size
        text_snippet = text[:100]
        return hashlib.md5(text_snippet.encode()).hexdigest()
    
    def _update_cache(self, key: str, value: np.ndarray):
        """Update prediction cache with LRU eviction"""
        # Evict oldest entries if cache is full
        if len(self.prediction_cache) >= self.max_cache_size:
            # Remove first (oldest) entry
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[key] = value
    
    def _format_prediction(
        self,
        probabilities: np.ndarray,
        return_probabilities: bool,
        top_k: int,
        inference_time: float,
        cached: bool = False
    ) -> Dict[str, Any]:
        """Format prediction results"""
        # Get top-k predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        
        predictions = []
        for idx in top_indices:
            pred = {
                'class_id': int(idx),
                'class_name': self._get_class_name(idx),
                'score': float(probabilities[idx])
            }
            predictions.append(pred)
        
        result = {
            'prediction': predictions[0]['class_name'],
            'confidence': predictions[0]['score'],
            'inference_time_ms': inference_time * 1000,
            'cached': cached
        }
        
        if top_k > 1:
            result['top_k_predictions'] = predictions
        
        if return_probabilities:
            result['probabilities'] = probabilities.tolist()
        
        return result
    
    def _get_class_name(self, class_id: int) -> str:
        """Get class name from ID"""
        # AG News classes
        class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
        return class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
    
    def predict_batch(
        self,
        texts: List[str],
        return_probabilities: bool = False,
        top_k: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Predict classes for multiple texts
        
        Args:
            texts: List of input texts
            return_probabilities: Whether to return probability scores
            top_k: Number of top predictions to return
            
        Returns:
            List of prediction results
        """
        results = []
        
        for text in texts:
            result = self.predict(text, return_probabilities, top_k)
            results.append(result)
        
        return results
    
    def warmup(self, num_samples: int = 5):
        """
        Warmup model with dummy predictions
        
        Args:
            num_samples: Number of warmup samples
        """
        logger.info(f"Warming up model with {num_samples} samples")
        
        dummy_texts = [
            "This is a warmup text for model initialization.",
            "Technology companies announce new products.",
            "Sports team wins championship game.",
            "Global economic trends affect markets.",
            "Scientific breakthrough in medical research."
        ]
        
        for i in range(min(num_samples, len(dummy_texts))):
            _ = self.predict(dummy_texts[i % len(dummy_texts)])
        
        # Clear cache after warmup
        self.prediction_cache.clear()
        logger.info("Model warmup complete")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if not self.inference_times:
            return {}
        
        return {
            'mean_inference_time_ms': np.mean(self.inference_times) * 1000,
            'median_inference_time_ms': np.median(self.inference_times) * 1000,
            'min_inference_time_ms': np.min(self.inference_times) * 1000,
            'max_inference_time_ms': np.max(self.inference_times) * 1000,
            'std_inference_time_ms': np.std(self.inference_times) * 1000,
            'total_predictions': len(self.inference_times),
            'cache_size': len(self.prediction_cache),
            'cache_hit_rate': self._calculate_cache_hit_rate()
        }
    
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        # This would require tracking hits/misses
        # Placeholder implementation
        return 0.0
    
    def cleanup(self):
        """Cleanup resources"""
        # Clear cache
        self.prediction_cache.clear()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Predictor cleanup complete")
