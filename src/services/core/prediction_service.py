"""
Prediction Service for Text Classification
================================================================================
Implements the core prediction functionality for text classification using
various models and ensemble methods with support for batch processing,
caching, and performance optimization.

This service follows the Repository and Strategy patterns for flexible
model selection and prediction strategies.

References:
    - Fowler, M. (2002). Patterns of Enterprise Application Architecture
    - Bishop, C. M. (2006). Pattern Recognition and Machine Learning
    - Goodfellow, I., et al. (2016). Deep Learning

Author: Võ Hải Dũng
License: MIT
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from src.api.base.base_handler import APIContext
from src.core.exceptions import ModelNotFoundError, PredictionError
from src.models.base.base_model import BaseModel
from src.services.base_service import BaseService, ServiceConfig, ServiceStatus
from src.utils.logging_config import get_logger
from src.utils.memory_utils import optimize_memory
from src.utils.profiling_utils import profile_async

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """
    Container for prediction results.
    
    Attributes:
        text: Input text
        category: Predicted category
        confidence: Confidence score
        probabilities: Class probabilities
        model_name: Model used for prediction
        processing_time: Processing time in seconds
        explanations: Optional explanations
        metadata: Additional metadata
    """
    text: str
    category: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    model_name: str = "default"
    processing_time: float = 0.0
    explanations: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class PredictionCache:
    """
    Cache for prediction results to improve performance.
    
    Implements LRU cache with TTL for prediction results.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize prediction cache.
        
        Args:
            max_size: Maximum cache size
            ttl_seconds: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[PredictionResult, float]] = {}
        self._access_order: List[str] = []
    
    def get(self, key: str) -> Optional[PredictionResult]:
        """
        Get prediction from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached prediction or None
        """
        if key in self._cache:
            result, timestamp = self._cache[key]
            
            # Check TTL
            if time.time() - timestamp < self.ttl_seconds:
                # Update access order
                self._access_order.remove(key)
                self._access_order.append(key)
                return result
            else:
                # Expired
                del self._cache[key]
                self._access_order.remove(key)
        
        return None
    
    def put(self, key: str, result: PredictionResult) -> None:
        """
        Put prediction in cache.
        
        Args:
            key: Cache key
            result: Prediction result
        """
        # Evict if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[key] = (result, time.time())
        
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    @staticmethod
    def generate_key(text: str, model_name: str) -> str:
        """
        Generate cache key.
        
        Args:
            text: Input text
            model_name: Model name
            
        Returns:
            Cache key
        """
        content = f"{text}:{model_name}"
        return hashlib.md5(content.encode()).hexdigest()


class PredictionService(BaseService):
    """
    Service for text classification predictions.
    
    Manages model loading, prediction execution, result caching,
    and performance optimization for classification tasks.
    """
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize prediction service.
        
        Args:
            config: Service configuration
        """
        super().__init__(config)
        self.models: Dict[str, BaseModel] = {}
        self.default_model_name = "deberta-v3-large"
        self.cache = PredictionCache()
        
        # Performance metrics
        self._prediction_count = 0
        self._total_latency = 0.0
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Category mappings
        self.categories = ["World", "Sports", "Business", "Technology"]
        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.idx_to_category = {idx: cat for idx, cat in enumerate(self.categories)}
    
    async def _initialize(self) -> None:
        """Initialize service-specific components."""
        logger.info("Initializing prediction service")
        
        # Load default model
        model_service = self.get_dependency("model_management_service")
        if model_service:
            try:
                await self._load_model(self.default_model_name)
                logger.info(f"Loaded default model: {self.default_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load default model: {str(e)}")
        
        self.status = ServiceStatus.READY
    
    async def _shutdown(self) -> None:
        """Cleanup service resources."""
        logger.info("Shutting down prediction service")
        
        # Clear cache
        self.cache.clear()
        
        # Unload models
        for model_name in list(self.models.keys()):
            await self._unload_model(model_name)
    
    @profile_async
    async def predict(
        self,
        text: str,
        model_name: Optional[str] = None,
        return_probabilities: bool = False,
        return_explanations: bool = False,
        context: Optional[APIContext] = None
    ) -> PredictionResult:
        """
        Perform text classification prediction.
        
        Args:
            text: Input text to classify
            model_name: Model to use (None for default)
            return_probabilities: Whether to return class probabilities
            return_explanations: Whether to return explanations
            context: API context for request tracking
            
        Returns:
            PredictionResult with classification
            
        Raises:
            PredictionError: If prediction fails
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not text or not text.strip():
                raise ValueError("Text input cannot be empty")
            
            if len(text) > 10000:
                raise ValueError("Text exceeds maximum length of 10000 characters")
            
            # Use default model if not specified
            if not model_name:
                model_name = self.default_model_name
            
            # Check cache
            cache_key = self.cache.generate_key(text, model_name)
            cached_result = self.cache.get(cache_key)
            
            if cached_result:
                self._cache_hits += 1
                logger.debug(f"Cache hit for prediction")
                return cached_result
            
            self._cache_misses += 1
            
            # Get model
            model = await self._get_or_load_model(model_name)
            
            # Perform prediction
            prediction = await self._execute_prediction(
                model,
                text,
                return_probabilities
            )
            
            # Generate explanations if requested
            explanations = None
            if return_explanations:
                explanations = await self._generate_explanations(
                    model,
                    text,
                    prediction
                )
            
            # Create result
            result = PredictionResult(
                text=text[:100] + "..." if len(text) > 100 else text,
                category=prediction["category"],
                confidence=prediction["confidence"],
                probabilities=prediction.get("probabilities") if return_probabilities else None,
                model_name=model_name,
                processing_time=time.time() - start_time,
                explanations=explanations,
                metadata={
                    "text_length": len(text),
                    "model_version": getattr(model, "version", "1.0.0")
                }
            )
            
            # Cache result
            self.cache.put(cache_key, result)
            
            # Update metrics
            self._prediction_count += 1
            self._total_latency += result.processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    async def predict_batch(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        return_probabilities: bool = False,
        batch_size: int = 32,
        context: Optional[APIContext] = None
    ) -> List[PredictionResult]:
        """
        Perform batch prediction.
        
        Args:
            texts: List of texts to classify
            model_name: Model to use
            return_probabilities: Whether to return probabilities
            batch_size: Processing batch size
            context: API context
            
        Returns:
            List of PredictionResult objects
        """
        if not texts:
            return []
        
        # Use default model if not specified
        if not model_name:
            model_name = self.default_model_name
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Process batch in parallel
            batch_tasks = [
                self.predict(
                    text,
                    model_name,
                    return_probabilities,
                    context=context
                )
                for text in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle results and exceptions
            for result in batch_results:
                if isinstance(result, Exception):
                    # Create error result
                    results.append(PredictionResult(
                        text="Error",
                        category="Unknown",
                        confidence=0.0,
                        model_name=model_name,
                        metadata={"error": str(result)}
                    ))
                else:
                    results.append(result)
        
        return results
    
    async def _get_or_load_model(self, model_name: str) -> BaseModel:
        """
        Get model from cache or load it.
        
        Args:
            model_name: Model name
            
        Returns:
            Model instance
            
        Raises:
            ModelNotFoundError: If model not found
        """
        if model_name not in self.models:
            await self._load_model(model_name)
        
        return self.models[model_name]
    
    async def _load_model(self, model_name: str) -> None:
        """
        Load a model for prediction.
        
        Args:
            model_name: Model name to load
            
        Raises:
            ModelNotFoundError: If model not found
        """
        logger.info(f"Loading model: {model_name}")
        
        try:
            # Get model from model management service
            model_service = self.get_dependency("model_management_service")
            if not model_service:
                raise ModelNotFoundError("Model management service not available")
            
            model = await model_service.get_model(model_name)
            if not model:
                raise ModelNotFoundError(f"Model '{model_name}' not found")
            
            # Load model weights
            await model.load()
            
            # Set to evaluation mode
            model.eval()
            
            # Store model
            self.models[model_name] = model
            
            logger.info(f"Model '{model_name}' loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {str(e)}")
            raise ModelNotFoundError(f"Failed to load model '{model_name}': {str(e)}")
    
    async def _unload_model(self, model_name: str) -> None:
        """
        Unload a model from memory.
        
        Args:
            model_name: Model name to unload
        """
        if model_name in self.models:
            logger.info(f"Unloading model: {model_name}")
            
            try:
                model = self.models[model_name]
                
                # Clear model from GPU if applicable
                if hasattr(model, "to"):
                    model.to("cpu")
                
                # Delete model
                del self.models[model_name]
                
                # Run garbage collection
                optimize_memory()
                
                logger.info(f"Model '{model_name}' unloaded")
                
            except Exception as e:
                logger.error(f"Error unloading model '{model_name}': {str(e)}")
    
    async def _execute_prediction(
        self,
        model: BaseModel,
        text: str,
        return_probabilities: bool
    ) -> Dict[str, Any]:
        """
        Execute prediction using model.
        
        Args:
            model: Model instance
            text: Input text
            return_probabilities: Whether to return probabilities
            
        Returns:
            Prediction dictionary
        """
        # Tokenize input
        inputs = model.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt"
        )
        
        # Move to device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            
            # Get logits
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Apply softmax
            probabilities = torch.softmax(logits, dim=-1)
            
            # Get prediction
            predicted_idx = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_idx].item()
            
            # Create result
            result = {
                "category": self.idx_to_category[predicted_idx],
                "confidence": float(confidence)
            }
            
            if return_probabilities:
                result["probabilities"] = {
                    self.idx_to_category[i]: float(probabilities[0, i])
                    for i in range(len(self.categories))
                }
            
            return result
    
    async def _generate_explanations(
        self,
        model: BaseModel,
        text: str,
        prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate explanations for prediction.
        
        Args:
            model: Model instance
            text: Input text
            prediction: Prediction result
            
        Returns:
            Explanation dictionary
        """
        explanations = {}
        
        try:
            # Get attention weights if available
            if hasattr(model, "get_attention_weights"):
                attention = await model.get_attention_weights(text)
                explanations["attention"] = attention
            
            # Get feature importance
            if hasattr(model, "get_feature_importance"):
                importance = await model.get_feature_importance(text)
                explanations["feature_importance"] = importance
            
            # Add prediction confidence breakdown
            explanations["confidence_analysis"] = {
                "predicted_class": prediction["category"],
                "confidence": prediction["confidence"],
                "margin": prediction.get("confidence", 0) - 0.25  # Margin above random
            }
            
        except Exception as e:
            logger.warning(f"Failed to generate explanations: {str(e)}")
        
        return explanations
    
    async def _execute(self, *args, **kwargs) -> Any:
        """Execute service operation."""
        # Default to predict method
        return await self.predict(*args, **kwargs)
    
    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a loaded model.
        
        Args:
            model_name: Model name
            
        Returns:
            Model information dictionary
        """
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        
        return {
            "name": model_name,
            "loaded": True,
            "type": type(model).__name__,
            "parameters": sum(p.numel() for p in model.parameters()),
            "device": str(next(model.parameters()).device),
            "memory_usage": self._estimate_model_memory(model)
        }
    
    def _estimate_model_memory(self, model: BaseModel) -> float:
        """
        Estimate model memory usage in MB.
        
        Args:
            model: Model instance
            
        Returns:
            Memory usage in MB
        """
        param_memory = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )
        
        # Add buffer memory
        buffer_memory = sum(
            b.numel() * b.element_size()
            for b in model.buffers()
        )
        
        total_bytes = param_memory + buffer_memory
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    async def warmup(self, model_name: Optional[str] = None) -> None:
        """
        Warmup model for better performance.
        
        Args:
            model_name: Model to warmup (None for all)
        """
        logger.info(f"Warming up model(s)")
        
        # Sample text for warmup
        sample_text = "This is a sample text for model warmup."
        
        if model_name:
            models_to_warmup = [model_name]
        else:
            models_to_warmup = list(self.models.keys())
        
        for name in models_to_warmup:
            try:
                # Run a few predictions
                for _ in range(3):
                    await self.predict(sample_text, name)
                
                logger.info(f"Model '{name}' warmed up")
            except Exception as e:
                logger.warning(f"Failed to warmup model '{name}': {str(e)}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get service metrics.
        
        Returns:
            Metrics dictionary
        """
        avg_latency = (
            self._total_latency / self._prediction_count
            if self._prediction_count > 0
            else 0
        )
        
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses)
            if (self._cache_hits + self._cache_misses) > 0
            else 0
        )
        
        return {
            "predictions_total": self._prediction_count,
            "average_latency_seconds": avg_latency,
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "models_loaded": len(self.models),
            "model_names": list(self.models.keys())
        }
