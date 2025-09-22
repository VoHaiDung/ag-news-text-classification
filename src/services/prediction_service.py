"""
Prediction Service Module
=========================

Implements prediction service following patterns from:
- Fowler (2002): "Patterns of Enterprise Application Architecture"
- Kleppmann (2017): "Designing Data-Intensive Applications"
- Google (2017): "Rules of Machine Learning"

This service handles all prediction-related operations including
single and batch predictions, model loading, and result caching.

Author: Team SOTA AGNews
License: MIT
"""

import logging
import hashlib
import time
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizer
)

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_seed
from src.core.registry import ModelRegistry
from src.core.exceptions import ModelError, PredictionError
from src.data.preprocessing.text_cleaner import TextCleaner, CleaningConfig
from src.data.preprocessing.tokenization import Tokenizer, TokenizationConfig
from src.services.data_service import DataService, DataServiceConfig
from configs.constants import (
    AG_NEWS_CLASSES,
    MAX_SEQUENCE_LENGTH,
    MODEL_DIR
)

logger = setup_logging(__name__)

@dataclass
class PredictionConfig:
    """Configuration for prediction service."""
    
    # Model settings
    model_name: str = "deberta-v3"
    model_path: Optional[Path] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Inference settings
    batch_size: int = 32
    max_length: int = MAX_SEQUENCE_LENGTH
    num_workers: int = 4
    use_fp16: bool = False
    
    # Optimization settings
    use_onnx: bool = False
    use_tensorrt: bool = False
    dynamic_batching: bool = True
    
    # Caching settings
    enable_cache: bool = True
    cache_ttl: int = 3600  # seconds
    max_cache_size: int = 10000
    
    # Post-processing
    return_probabilities: bool = True
    confidence_threshold: float = 0.0
    top_k: int = 1
    
    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)
        
        if self.use_fp16 and self.device == "cpu":
            logger.warning("FP16 not supported on CPU, disabling")
            self.use_fp16 = False

@dataclass
class PredictionResult:
    """Prediction result container."""
    
    text: str
    predicted_class: str
    predicted_label: int
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    processing_time: float = 0.0
    model_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "predicted_class": self.predicted_class,
            "predicted_label": self.predicted_label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "processing_time": self.processing_time,
            "model_name": self.model_name,
            "metadata": self.metadata
        }

class ModelCache:
    """
    Model caching implementation following patterns from:
    - Fitzpatrick (2016): "Designing Distributed Systems"
    """
    
    def __init__(self, max_size: int = 5):
        """
        Initialize model cache.
        
        Args:
            max_size: Maximum number of models to cache
        """
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.load_times = {}
    
    def get(self, model_name: str) -> Optional[PreTrainedModel]:
        """Get model from cache."""
        if model_name in self.cache:
            self.access_times[model_name] = time.time()
            return self.cache[model_name]
        return None
    
    def put(self, model_name: str, model: PreTrainedModel):
        """Put model in cache."""
        # Evict least recently used if cache is full
        if len(self.cache) >= self.max_size:
            lru_model = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[lru_model]
            del self.access_times[lru_model]
            logger.info(f"Evicted model {lru_model} from cache")
        
        self.cache[model_name] = model
        self.access_times[model_name] = time.time()
        self.load_times[model_name] = time.time()
        logger.info(f"Cached model {model_name}")
    
    def clear(self):
        """Clear all cached models."""
        self.cache.clear()
        self.access_times.clear()
        self.load_times.clear()

class PredictionCache:
    """
    Prediction result caching following patterns from:
    - Nishtala et al. (2013): "Scaling Memcache at Facebook"
    """
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        """
        Initialize prediction cache.
        
        Args:
            max_size: Maximum cache size
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.timestamps = {}
        self.hit_count = 0
        self.miss_count = 0
    
    def get_key(self, text: str, model_name: str) -> str:
        """Generate cache key."""
        return hashlib.md5(f"{text}:{model_name}".encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[PredictionResult]:
        """Get prediction from cache."""
        key = self.get_key(text, model_name)
        
        if key in self.cache:
            # Check if expired
            if time.time() - self.timestamps[key] > self.ttl:
                del self.cache[key]
                del self.timestamps[key]
                self.miss_count += 1
                return None
            
            self.hit_count += 1
            return self.cache[key]
        
        self.miss_count += 1
        return None
    
    def put(self, text: str, model_name: str, result: PredictionResult):
        """Put prediction in cache."""
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
        
        key = self.get_key(text, model_name)
        self.cache[key] = result
        self.timestamps[key] = time.time()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(total, 1)
        
        return {
            "size": len(self.cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "total_requests": total
        }

class PredictionService:
    """
    Main prediction service implementing patterns from:
    - Martin (2017): "Clean Architecture"
    - Newman (2015): "Building Microservices"
    """
    
    def __init__(self, config: Optional[PredictionConfig] = None):
        """
        Initialize prediction service.
        
        Args:
            config: Service configuration
        """
        self.config = config or PredictionConfig()
        
        # Initialize components
        self.model_cache = ModelCache(max_size=5)
        self.prediction_cache = PredictionCache(
            max_size=self.config.max_cache_size,
            ttl=self.config.cache_ttl
        )
        
        # Services
        self.data_service = DataService()
        self.text_cleaner = TextCleaner(CleaningConfig())
        
        # Current model
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.config.num_workers)
        
        # Statistics
        self.stats = {
            "total_predictions": 0,
            "successful_predictions": 0,
            "failed_predictions": 0,
            "avg_processing_time": 0.0,
            "models_loaded": 0
        }
        
        logger.info("Prediction service initialized")
    
    def load_model(
        self,
        model_name: str,
        model_path: Optional[Path] = None
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load model with caching.
        
        Args:
            model_name: Name of the model
            model_path: Optional path to model
            
        Returns:
            Tuple of (model, tokenizer)
        """
        # Check cache first
        cached_model = self.model_cache.get(model_name)
        if cached_model:
            logger.info(f"Loaded model {model_name} from cache")
            # Still need to load tokenizer
            tokenizer = self._load_tokenizer(model_name)
            return cached_model, tokenizer
        
        logger.info(f"Loading model {model_name}")
        start_time = time.time()
        
        try:
            # Determine model path
            if model_path is None:
                model_path = MODEL_DIR / model_name
            
            # Load model and tokenizer
            if model_path.exists():
                # Load from local path
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(model_path),
                    num_labels=len(AG_NEWS_CLASSES),
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32
                )
                tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            else:
                # Load from Hugging Face hub
                model_mapping = {
                    "deberta-v3": "microsoft/deberta-v3-base",
                    "roberta-large": "roberta-large",
                    "xlnet-large": "xlnet-large-cased"
                }
                
                hf_model_name = model_mapping.get(model_name, model_name)
                
                model = AutoModelForSequenceClassification.from_pretrained(
                    hf_model_name,
                    num_labels=len(AG_NEWS_CLASSES),
                    torch_dtype=torch.float16 if self.config.use_fp16 else torch.float32
                )
                tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
            
            # Move to device
            model = model.to(self.config.device)
            model.eval()
            
            # Cache model
            self.model_cache.put(model_name, model)
            
            # Update statistics
            load_time = time.time() - start_time
            self.stats["models_loaded"] += 1
            
            logger.info(f"Model {model_name} loaded in {load_time:.2f}s")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise ModelError(f"Model loading failed: {str(e)}")
    
    def _load_tokenizer(self, model_name: str) -> PreTrainedTokenizer:
        """Load tokenizer for model."""
        model_mapping = {
            "deberta-v3": "microsoft/deberta-v3-base",
            "roberta-large": "roberta-large",
            "xlnet-large": "xlnet-large-cased"
        }
        
        hf_model_name = model_mapping.get(model_name, model_name)
        return AutoTokenizer.from_pretrained(hf_model_name)
    
    def predict(
        self,
        text: str,
        model_name: Optional[str] = None,
        return_probabilities: bool = True
    ) -> PredictionResult:
        """
        Perform single text prediction.
        
        Args:
            text: Input text
            model_name: Model to use
            return_probabilities: Whether to return probabilities
            
        Returns:
            Prediction result
        """
        start_time = time.time()
        model_name = model_name or self.config.model_name
        
        # Check cache if enabled
        if self.config.enable_cache:
            cached_result = self.prediction_cache.get(text, model_name)
            if cached_result:
                logger.debug(f"Cache hit for text: {text[:50]}...")
                return cached_result
        
        try:
            # Load model if needed
            if self.current_model_name != model_name:
                self.current_model, self.current_tokenizer = self.load_model(model_name)
                self.current_model_name = model_name
            
            # Clean text
            cleaned_text = self.text_cleaner.clean(text)
            
            # Tokenize
            inputs = self.current_tokenizer(
                cleaned_text,
                padding=True,
                truncation=True,
                max_length=self.config.max_length,
                return_tensors="pt"
            ).to(self.config.device)
            
            # Inference
            with torch.no_grad():
                if self.config.use_fp16:
                    with torch.cuda.amp.autocast():
                        outputs = self.current_model(**inputs)
                else:
                    outputs = self.current_model(**inputs)
                
                logits = outputs.logits
                probabilities = F.softmax(logits, dim=-1)
            
            # Get prediction
            predicted_label = torch.argmax(probabilities, dim=-1).item()
            confidence = probabilities[0, predicted_label].item()
            
            # Prepare result
            result = PredictionResult(
                text=text,
                predicted_class=AG_NEWS_CLASSES[predicted_label],
                predicted_label=predicted_label,
                confidence=confidence,
                probabilities={
                    AG_NEWS_CLASSES[i]: probabilities[0, i].item()
                    for i in range(len(AG_NEWS_CLASSES))
                } if return_probabilities else None,
                processing_time=time.time() - start_time,
                model_name=model_name,
                metadata={
                    "cleaned_text": cleaned_text,
                    "text_length": len(text),
                    "device": self.config.device
                }
            )
            
            # Cache result
            if self.config.enable_cache:
                self.prediction_cache.put(text, model_name, result)
            
            # Update statistics
            self.stats["total_predictions"] += 1
            self.stats["successful_predictions"] += 1
            self._update_avg_processing_time(result.processing_time)
            
            return result
            
        except Exception as e:
            self.stats["total_predictions"] += 1
            self.stats["failed_predictions"] += 1
            logger.error(f"Prediction failed: {str(e)}")
            raise PredictionError(f"Prediction failed: {str(e)}")
    
    def predict_batch(
        self,
        texts: List[str],
        model_name: Optional[str] = None,
        return_probabilities: bool = True,
        parallel: bool = True
    ) -> List[PredictionResult]:
        """
        Perform batch prediction.
        
        Implements batching strategies from:
        - Dean & Barroso (2013): "The Tail at Scale"
        
        Args:
            texts: List of input texts
            model_name: Model to use
            return_probabilities: Whether to return probabilities
            parallel: Whether to process in parallel
            
        Returns:
            List of prediction results
        """
        model_name = model_name or self.config.model_name
        
        if parallel and len(texts) > 1:
            return self._predict_batch_parallel(
                texts, model_name, return_probabilities
            )
        else:
            return self._predict_batch_sequential(
                texts, model_name, return_probabilities
            )
    
    def _predict_batch_sequential(
        self,
        texts: List[str],
        model_name: str,
        return_probabilities: bool
    ) -> List[PredictionResult]:
        """Sequential batch prediction."""
        results = []
        
        for text in texts:
            try:
                result = self.predict(text, model_name, return_probabilities)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict text: {str(e)}")
                # Create error result
                results.append(PredictionResult(
                    text=text,
                    predicted_class="ERROR",
                    predicted_label=-1,
                    confidence=0.0,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def _predict_batch_parallel(
        self,
        texts: List[str],
        model_name: str,
        return_probabilities: bool
    ) -> List[PredictionResult]:
        """Parallel batch prediction using thread pool."""
        futures = []
        
        for text in texts:
            future = self.executor.submit(
                self.predict, text, model_name, return_probabilities
            )
            futures.append((text, future))
        
        results = []
        for text, future in futures:
            try:
                result = future.result(timeout=30)
                results.append(result)
            except Exception as e:
                logger.error(f"Parallel prediction failed for text: {str(e)}")
                results.append(PredictionResult(
                    text=text,
                    predicted_class="ERROR",
                    predicted_label=-1,
                    confidence=0.0,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def predict_with_ensemble(
        self,
        text: str,
        model_names: List[str],
        aggregation: str = "voting"
    ) -> PredictionResult:
        """
        Ensemble prediction using multiple models.
        
        Implements ensemble methods from:
        - Dietterich (2000): "Ensemble Methods in Machine Learning"
        
        Args:
            text: Input text
            model_names: List of models to use
            aggregation: Aggregation method (voting/averaging)
            
        Returns:
            Ensemble prediction result
        """
        predictions = []
        
        # Get predictions from all models
        for model_name in model_names:
            try:
                result = self.predict(text, model_name, return_probabilities=True)
                predictions.append(result)
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {str(e)}")
        
        if not predictions:
            raise PredictionError("All models failed")
        
        # Aggregate predictions
        if aggregation == "voting":
            # Majority voting
            votes = [p.predicted_label for p in predictions]
            predicted_label = max(set(votes), key=votes.count)
            confidence = votes.count(predicted_label) / len(votes)
            
        elif aggregation == "averaging":
            # Average probabilities
            all_probs = np.array([
                [p.probabilities[cls] for cls in AG_NEWS_CLASSES]
                for p in predictions
            ])
            avg_probs = np.mean(all_probs, axis=0)
            predicted_label = np.argmax(avg_probs)
            confidence = avg_probs[predicted_label]
        
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        return PredictionResult(
            text=text,
            predicted_class=AG_NEWS_CLASSES[predicted_label],
            predicted_label=predicted_label,
            confidence=confidence,
            probabilities={
                AG_NEWS_CLASSES[i]: avg_probs[i]
                for i in range(len(AG_NEWS_CLASSES))
            } if aggregation == "averaging" else None,
            model_name=f"ensemble_{aggregation}",
            metadata={
                "models": model_names,
                "aggregation": aggregation,
                "num_models": len(predictions)
            }
        )
    
    def _update_avg_processing_time(self, new_time: float):
        """Update average processing time."""
        n = self.stats["successful_predictions"]
        old_avg = self.stats["avg_processing_time"]
        self.stats["avg_processing_time"] = (old_avg * (n - 1) + new_time) / n
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "cache_stats": self.prediction_cache.get_stats(),
            "models_cached": len(self.model_cache.cache),
            "current_model": self.current_model_name
        }
    
    def clear_caches(self):
        """Clear all caches."""
        self.model_cache.clear()
        self.prediction_cache.cache.clear()
        self.prediction_cache.timestamps.clear()
        logger.info("Cleared all prediction service caches")
    
    def shutdown(self):
        """Shutdown service gracefully."""
        self.executor.shutdown(wait=True)
        self.clear_caches()
        logger.info("Prediction service shut down")

# Global service instance
_prediction_service = None

def get_prediction_service(
    config: Optional[PredictionConfig] = None
) -> PredictionService:
    """
    Get prediction service instance (singleton pattern).
    
    Args:
        config: Optional configuration
        
    Returns:
        Prediction service instance
    """
    global _prediction_service
    
    if _prediction_service is None or config is not None:
        _prediction_service = PredictionService(config)
    
    return _prediction_service
