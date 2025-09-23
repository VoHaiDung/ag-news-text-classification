"""
Model Management Service for Model Lifecycle
================================================================================
Implements comprehensive model management including loading, versioning,
optimization, deployment, and monitoring following MLOps best practices.

This service provides centralized model registry and lifecycle management
ensuring model reproducibility, versioning, and performance tracking.

References:
    - Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems
    - Zaharia, M., et al. (2018). Accelerating the Machine Learning Lifecycle with MLflow
    - Google Cloud (2021). Best Practices for Implementing Machine Learning on Google Cloud

Author: Võ Hải Dũng
License: MIT
"""

import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoConfig, AutoTokenizer

from src.models.base.base_model import BaseModel
from src.services.base_service import BaseService, ServiceConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """
    Model metadata and configuration.
    
    Attributes:
        name: Model name
        version: Model version
        model_type: Type of model (transformer, ensemble, etc.)
        architecture: Model architecture
        parameters: Number of parameters
        size_mb: Model size in MB
        accuracy: Model accuracy score
        created_at: Creation timestamp
        updated_at: Last update timestamp
        checksum: Model checksum
        path: Path to model files
        config: Model configuration
        metrics: Performance metrics
        metadata: Additional metadata
    """
    name: str
    version: str = "1.0.0"
    model_type: str = "transformer"
    architecture: Optional[str] = None
    parameters: Optional[int] = None
    size_mb: Optional[float] = None
    accuracy: Optional[float] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    checksum: Optional[str] = None
    path: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "parameters": self.parameters,
            "size_mb": self.size_mb,
            "accuracy": self.accuracy,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "checksum": self.checksum,
            "path": self.path,
            "config": self.config,
            "metrics": self.metrics,
            "metadata": self.metadata
        }


class ModelManagementService(BaseService):
    """
    Service for comprehensive model lifecycle management.
    
    Manages model loading, versioning, optimization, deployment,
    and monitoring throughout the ML lifecycle.
    """
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize model management service.
        
        Args:
            config: Service configuration
        """
        super().__init__(config)
        
        # Model storage paths
        self.models_dir = Path("outputs/models")
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.pretrained_dir = self.models_dir / "pretrained"
        self.fine_tuned_dir = self.models_dir / "fine_tuned"
        self.exported_dir = self.models_dir / "exported"
        
        # Ensure directories exist
        for dir_path in [
            self.checkpoints_dir,
            self.pretrained_dir,
            self.fine_tuned_dir,
            self.exported_dir
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.model_registry: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, BaseModel] = {}
        self.default_model_name = "deberta-v3-large"
        
        # Statistics
        self._models_loaded = 0
        self._models_saved = 0
        self._total_inference_time = 0.0
    
    async def _initialize(self) -> None:
        """Initialize service components."""
        logger.info("Initializing model management service")
        
        # Load model registry
        await self._load_model_registry()
        
        # Initialize default models
        await self._initialize_default_models()
    
    async def _shutdown(self) -> None:
        """Cleanup service resources."""
        logger.info("Shutting down model management service")
        
        # Save model registry
        await self._save_model_registry()
        
        # Unload all models
        for model_name in list(self.loaded_models.keys()):
            await self.unload_model(model_name)
    
    async def get_model(self, name: str, version: Optional[str] = None) -> Optional[BaseModel]:
        """
        Get a model for inference.
        
        Args:
            name: Model name
            version: Model version (optional)
            
        Returns:
            Model instance or None
        """
        # Check if model is loaded
        model_key = f"{name}:{version}" if version else name
        
        if model_key in self.loaded_models:
            return self.loaded_models[model_key]
        
        # Try to load model
        if await self.load_model(name, version):
            return self.loaded_models[model_key]
        
        return None
    
    async def get_model_for_training(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[BaseModel]:
        """
        Get a model for training (not cached).
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Fresh model instance for training
        """
        if name not in self.model_registry:
            logger.warning(f"Model '{name}' not found in registry")
            return None
        
        model_info = self.model_registry[name]
        
        try:
            # Load model configuration
            if name == "deberta-v3-large":
                from src.models.transformers.deberta.deberta_v3 import DeBERTaV3Model
                model = DeBERTaV3Model()
            elif name == "roberta-large":
                from src.models.transformers.roberta.roberta_enhanced import RoBERTaEnhancedModel
                model = RoBERTaEnhancedModel()
            elif name == "xlnet-large":
                from src.models.transformers.xlnet.xlnet_classifier import XLNetClassifier
                model = XLNetClassifier()
            else:
                logger.warning(f"Unknown model type: {name}")
                return None
            
            # Don't cache training models
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model for training: {str(e)}")
            return None
    
    async def load_model(
        self,
        name: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Load a model into memory.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if successfully loaded
        """
        model_key = f"{name}:{version}" if version else name
        
        # Check if already loaded
        if model_key in self.loaded_models:
            logger.info(f"Model '{model_key}' already loaded")
            return True
        
        if name not in self.model_registry:
            logger.warning(f"Model '{name}' not found in registry")
            return False
        
        model_info = self.model_registry[name]
        
        try:
            logger.info(f"Loading model '{model_key}'")
            
            # Load model based on type
            if name == "deberta-v3-large":
                from src.models.transformers.deberta.deberta_v3 import DeBERTaV3Model
                model = DeBERTaV3Model()
            elif name == "roberta-large":
                from src.models.transformers.roberta.roberta_enhanced import RoBERTaEnhancedModel
                model = RoBERTaEnhancedModel()
            elif name == "xlnet-large":
                from src.models.transformers.xlnet.xlnet_classifier import XLNetClassifier
                model = XLNetClassifier()
            elif name == "ensemble":
                from src.models.ensemble.voting.soft_voting import SoftVotingEnsemble
                model = SoftVotingEnsemble()
            else:
                logger.warning(f"Unknown model type: {name}")
                return False
            
            # Load weights if available
            if model_info.path:
                weights_path = Path(model_info.path) / "model.pt"
                if weights_path.exists():
                    state_dict = torch.load(weights_path, map_location="cpu")
                    model.load_state_dict(state_dict)
                    logger.info(f"Loaded weights from {weights_path}")
            
            # Set to evaluation mode
            model.eval()
            
            # Cache model
            self.loaded_models[model_key] = model
            self._models_loaded += 1
            
            logger.info(f"Model '{model_key}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_key}': {str(e)}")
            return False
    
    async def unload_model(
        self,
        name: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Unload a model from memory.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if successfully unloaded
        """
        model_key = f"{name}:{version}" if version else name
        
        if model_key not in self.loaded_models:
            logger.warning(f"Model '{model_key}' not loaded")
            return False
        
        try:
            logger.info(f"Unloading model '{model_key}'")
            
            # Clear model from memory
            del self.loaded_models[model_key]
            
            # Run garbage collection
            import gc
            gc.collect()
            
            # Clear GPU cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"Model '{model_key}' unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model '{model_key}': {str(e)}")
            return False
    
    async def save_trained_model(
        self,
        model: BaseModel,
        name: str,
        metrics: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save a trained model.
        
        Args:
            model: Trained model
            name: Model name
            metrics: Model metrics
            metadata: Additional metadata
            
        Returns:
            Model ID
        """
        try:
            # Generate version
            version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            
            # Create save path
            save_path = self.fine_tuned_dir / f"{name}_{version}"
            save_path.mkdir(parents=True, exist_ok=True)
            
            # Save model weights
            model_path = save_path / "model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Save configuration
            config_path = save_path / "config.json"
            config = {
                "model_type": type(model).__name__,
                "num_labels": 4,
                "architecture": model.__class__.__name__
            }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            
            # Calculate model size
            size_mb = model_path.stat().st_size / (1024 * 1024)
            
            # Calculate checksum
            with open(model_path, "rb") as f:
                checksum = hashlib.sha256(f.read()).hexdigest()
            
            # Create model info
            model_info = ModelInfo(
                name=name,
                version=version,
                model_type="fine_tuned",
                architecture=model.__class__.__name__,
                parameters=sum(p.numel() for p in model.parameters()),
                size_mb=size_mb,
                accuracy=metrics.get("accuracy") if metrics else None,
                checksum=checksum,
                path=str(save_path),
                config=config,
                metrics=metrics or {},
                metadata=metadata or {}
            )
            
            # Register model
            self.model_registry[name] = model_info
            self._models_saved += 1
            
            # Save registry
            await self._save_model_registry()
            
            model_id = f"{name}:{version}"
            logger.info(f"Saved trained model: {model_id}")
            
            return model_id
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    async def list_models(
        self,
        include_unloaded: bool = True,
        model_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Args:
            include_unloaded: Include models not in memory
            model_type: Filter by model type
            
        Returns:
            List of model information
        """
        models = []
        
        for name, info in self.model_registry.items():
            # Apply filters
            if model_type and info.model_type != model_type:
                continue
            
            # Check if loaded
            loaded = name in self.loaded_models
            
            if not include_unloaded and not loaded:
                continue
            
            model_dict = info.to_dict()
            model_dict["loaded"] = loaded
            model_dict["status"] = "loaded" if loaded else "available"
            
            models.append(model_dict)
        
        # Sort by name
        models.sort(key=lambda m: m["name"])
        
        return models
    
    async def get_model_details(
        self,
        name: str,
        version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed model information.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            Model details or None
        """
        if name not in self.model_registry:
            return None
        
        model_info = self.model_registry[name]
        details = model_info.to_dict()
        
        # Add runtime information if loaded
        model_key = f"{name}:{version}" if version else name
        if model_key in self.loaded_models:
            model = self.loaded_models[model_key]
            details["loaded"] = True
            details["device"] = str(next(model.parameters()).device)
            details["memory_usage_mb"] = self._estimate_memory_usage(model)
        else:
            details["loaded"] = False
        
        return details
    
    async def model_exists(
        self,
        name: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Check if model exists.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if model exists
        """
        return name in self.model_registry
    
    async def is_model_loaded(
        self,
        name: str,
        version: Optional[str] = None
    ) -> bool:
        """
        Check if model is loaded.
        
        Args:
            name: Model name
            version: Model version
            
        Returns:
            True if model is loaded
        """
        model_key = f"{name}:{version}" if version else name
        return model_key in self.loaded_models
    
    async def update_model(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> bool:
        """
        Update model configuration.
        
        Args:
            name: Model name
            config: New configuration
            
        Returns:
            True if successfully updated
        """
        if name not in self.model_registry:
            return False
        
        try:
            model_info = self.model_registry[name]
            model_info.config.update(config)
            model_info.updated_at = datetime.now(timezone.utc)
            
            await self._save_model_registry()
            
            logger.info(f"Updated configuration for model '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update model '{name}': {str(e)}")
            return False
    
    async def delete_model(
        self,
        name: str,
        version: Optional[str] = None,
        force: bool = False
    ) -> bool:
        """
        Delete a model.
        
        Args:
            name: Model name
            version: Model version
            force: Force deletion even if loaded
            
        Returns:
            True if successfully deleted
        """
        model_key = f"{name}:{version}" if version else name
        
        # Check if loaded
        if model_key in self.loaded_models and not force:
            logger.warning(f"Model '{model_key}' is loaded. Use force=True to delete")
            return False
        
        if name not in self.model_registry:
            logger.warning(f"Model '{name}' not found")
            return False
        
        try:
            # Unload if necessary
            if model_key in self.loaded_models:
                await self.unload_model(name, version)
            
            model_info = self.model_registry[name]
            
            # Delete files
            if model_info.path:
                model_path = Path(model_info.path)
                if model_path.exists():
                    if model_path.is_dir():
                        shutil.rmtree(model_path)
                    else:
                        model_path.unlink()
            
            # Remove from registry
            del self.model_registry[name]
            
            # Save registry
            await self._save_model_registry()
            
            logger.info(f"Deleted model '{name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model '{name}': {str(e)}")
            return False
    
    async def get_default_model(self) -> str:
        """Get default model name."""
        return self.default_model_name
    
    async def set_default_model(self, name: str) -> bool:
        """
        Set default model.
        
        Args:
            name: Model name
            
        Returns:
            True if successfully set
        """
        if name not in self.model_registry:
            logger.warning(f"Model '{name}' not found")
            return False
        
        self.default_model_name = name
        logger.info(f"Set default model to '{name}'")
        return True
    
    def _estimate_memory_usage(self, model: BaseModel) -> float:
        """Estimate model memory usage in MB."""
        param_size = sum(
            p.numel() * p.element_size()
            for p in model.parameters()
        )
        buffer_size = sum(
            b.numel() * b.element_size()
            for b in model.buffers()
        )
        
        total_bytes = param_size + buffer_size
        return total_bytes / (1024 * 1024)
    
    async def _initialize_default_models(self) -> None:
        """Initialize default model configurations."""
        default_models = {
            "deberta-v3-large": ModelInfo(
                name="deberta-v3-large",
                model_type="transformer",
                architecture="DeBERTaV3",
                parameters=435000000,
                size_mb=1740,
                accuracy=0.95
            ),
            "roberta-large": ModelInfo(
                name="roberta-large",
                model_type="transformer",
                architecture="RoBERTa",
                parameters=355000000,
                size_mb=1420,
                accuracy=0.94
            ),
            "xlnet-large": ModelInfo(
                name="xlnet-large",
                model_type="transformer",
                architecture="XLNet",
                parameters=340000000,
                size_mb=1360,
                accuracy=0.93
            ),
            "ensemble": ModelInfo(
                name="ensemble",
                model_type="ensemble",
                architecture="VotingEnsemble",
                parameters=1130000000,
                size_mb=4520,
                accuracy=0.96
            )
        }
        
        for name, info in default_models.items():
            if name not in self.model_registry:
                self.model_registry[name] = info
                logger.info(f"Registered default model: {name}")
    
    async def _load_model_registry(self) -> None:
        """Load model registry from disk."""
        registry_file = self.models_dir / "model_registry.json"
        
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, "r") as f:
                registry_data = json.load(f)
            
            for name, info_dict in registry_data.items():
                info = ModelInfo(
                    name=info_dict["name"],
                    version=info_dict.get("version", "1.0.0"),
                    model_type=info_dict.get("model_type", "transformer"),
                    architecture=info_dict.get("architecture"),
                    parameters=info_dict.get("parameters"),
                    size_mb=info_dict.get("size_mb"),
                    accuracy=info_dict.get("accuracy"),
                    checksum=info_dict.get("checksum"),
                    path=info_dict.get("path"),
                    config=info_dict.get("config", {}),
                    metrics=info_dict.get("metrics", {}),
                    metadata=info_dict.get("metadata", {})
                )
                
                # Parse timestamps
                if "created_at" in info_dict:
                    info.created_at = datetime.fromisoformat(info_dict["created_at"])
                if "updated_at" in info_dict:
                    info.updated_at = datetime.fromisoformat(info_dict["updated_at"])
                
                self.model_registry[name] = info
            
            logger.info(f"Loaded {len(self.model_registry)} models from registry")
            
        except Exception as e:
            logger.error(f"Failed to load model registry: {str(e)}")
    
    async def _save_model_registry(self) -> None:
        """Save model registry to disk."""
        registry_file = self.models_dir / "model_registry.json"
        
        try:
            registry_data = {
                name: info.to_dict()
                for name, info in self.model_registry.items()
            }
            
            with open(registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)
            
            logger.debug(f"Saved {len(registry_data)} models to registry")
            
        except Exception as e:
            logger.error(f"Failed to save model registry: {str(e)}")
    
    async def _execute(self, *args, **kwargs) -> Any:
        """Execute service operation."""
        # Not directly callable
        raise NotImplementedError("Model management service operations must be called directly")
