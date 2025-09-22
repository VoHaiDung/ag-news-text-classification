"""
Model Management Service
========================

Implements model lifecycle management following patterns from:
- Sculley et al. (2015): "Hidden Technical Debt in Machine Learning Systems"
- Zaharia et al. (2018): "Accelerating the Machine Learning Lifecycle with MLflow"
- Google (2017): "Rules of Machine Learning"

This service handles model versioning, deployment, monitoring, and lifecycle management.

Author: Võ Hải Dũng
License: MIT
"""

import logging
import json
import shutil
import hashlib
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import pickle
import yaml

import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import numpy as np
from packaging import version

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.core.registry import ModelRegistry
from src.core.exceptions import ModelError
from src.services.prediction_service import PredictionService
from configs.constants import MODEL_DIR, OUTPUT_DIR

logger = setup_logging(__name__)

class ModelStatus(Enum):
    """Model status enumeration."""
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    FAILED = "failed"
    EXPERIMENTAL = "experimental"

class DeploymentEnvironment(Enum):
    """Deployment environment enumeration."""
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

@dataclass
class ModelMetadata:
    """Model metadata container."""
    
    model_id: str
    name: str
    version: str
    status: ModelStatus
    created_at: datetime
    updated_at: datetime
    
    # Model information
    model_type: str = "transformer"
    architecture: str = ""
    num_parameters: int = 0
    model_size_mb: float = 0.0
    
    # Performance metrics
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # Training information
    training_config: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    
    # Deployment information
    deployment_env: Optional[DeploymentEnvironment] = None
    endpoint: Optional[str] = None
    
    # Versioning
    parent_version: Optional[str] = None
    commit_hash: Optional[str] = None
    
    # Tags and metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "model_type": self.model_type,
            "architecture": self.architecture,
            "num_parameters": self.num_parameters,
            "model_size_mb": self.model_size_mb,
            "metrics": self.metrics,
            "training_config": self.training_config,
            "dataset_info": self.dataset_info,
            "deployment_env": self.deployment_env.value if self.deployment_env else None,
            "endpoint": self.endpoint,
            "parent_version": self.parent_version,
            "commit_hash": self.commit_hash,
            "tags": self.tags,
            "metadata": self.metadata
        }

class ModelVersionControl:
    """
    Model version control system.
    
    Implements versioning patterns from:
    - Semantic Versioning Specification (Preston-Werner, 2013)
    """
    
    def __init__(self, base_dir: Path = MODEL_DIR / "versions"):
        """
        Initialize version control.
        
        Args:
            base_dir: Base directory for versioned models
        """
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.versions = {}
        self._load_versions()
    
    def create_version(
        self,
        model_name: str,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        version_str: Optional[str] = None,
        parent_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create new model version.
        
        Args:
            model_name: Model name
            model: Model to version
            tokenizer: Tokenizer
            version_str: Version string (auto-generate if None)
            parent_version: Parent version ID
            metadata: Additional metadata
            
        Returns:
            Version ID
        """
        # Generate version string if not provided
        if version_str is None:
            version_str = self._generate_version_string(model_name)
        
        # Generate version ID
        version_id = hashlib.md5(
            f"{model_name}:{version_str}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Create version directory
        version_dir = self.base_dir / model_name / version_str
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        model.save_pretrained(str(version_dir / "model"))
        tokenizer.save_pretrained(str(version_dir / "tokenizer"))
        
        # Calculate model size
        model_size = sum(
            p.stat().st_size
            for p in (version_dir / "model").rglob("*")
            if p.is_file()
        ) / (1024 * 1024)  # MB
        
        # Create version metadata
        version_metadata = {
            "version_id": version_id,
            "model_name": model_name,
            "version": version_str,
            "parent_version": parent_version,
            "created_at": datetime.now().isoformat(),
            "model_size_mb": model_size,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "metadata": metadata or {}
        }
        
        # Save version metadata
        with open(version_dir / "version.json", "w") as f:
            json.dump(version_metadata, f, indent=2)
        
        self.versions[version_id] = version_metadata
        
        logger.info(f"Created model version {version_id}: {model_name}:{version_str}")
        
        return version_id
    
    def get_version(self, version_id: str) -> Dict[str, Any]:
        """Get version metadata."""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        return self.versions[version_id]
    
    def load_version(
        self,
        version_id: str
    ) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Load versioned model."""
        version_info = self.get_version(version_id)
        
        version_dir = (
            self.base_dir / 
            version_info["model_name"] / 
            version_info["version"]
        )
        
        model = AutoModelForSequenceClassification.from_pretrained(
            str(version_dir / "model")
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(version_dir / "tokenizer")
        )
        
        return model, tokenizer
    
    def _generate_version_string(self, model_name: str) -> str:
        """Generate version string."""
        # Find existing versions
        existing_versions = [
            v["version"] for v in self.versions.values()
            if v["model_name"] == model_name
        ]
        
        if not existing_versions:
            return "1.0.0"
        
        # Parse versions and increment
        latest = max(existing_versions, key=lambda v: version.parse(v))
        major, minor, patch = latest.split(".")
        
        # Increment patch version
        return f"{major}.{minor}.{int(patch) + 1}"
    
    def _load_versions(self):
        """Load existing versions from disk."""
        for model_dir in self.base_dir.iterdir():
            if model_dir.is_dir():
                for version_dir in model_dir.iterdir():
                    if version_dir.is_dir():
                        version_file = version_dir / "version.json"
                        if version_file.exists():
                            with open(version_file, "r") as f:
                                version_info = json.load(f)
                                self.versions[version_info["version_id"]] = version_info

class ModelManagementService:
    """
    Main model management service.
    
    Implements model lifecycle management patterns from:
    - Breck et al. (2019): "The ML Test Score: A Rubric for ML Production Readiness"
    """
    
    def __init__(self):
        """Initialize model management service."""
        self.model_registry = ModelRegistry()
        self.version_control = ModelVersionControl()
        self.prediction_service = PredictionService()
        
        # Model storage
        self.models_dir = MODEL_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Model catalog
        self.catalog = {}
        self._load_catalog()
        
        # Deployment tracking
        self.deployments = {}
        
        # Statistics
        self.stats = {
            "total_models": 0,
            "production_models": 0,
            "archived_models": 0,
            "total_versions": 0,
            "total_deployments": 0
        }
        
        logger.info("Model management service initialized")
    
    def register_model(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        name: str,
        version_str: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ModelMetadata:
        """
        Register a new model.
        
        Args:
            model: Model to register
            tokenizer: Tokenizer
            name: Model name
            version_str: Version string
            metrics: Model metrics
            tags: Model tags
            metadata: Additional metadata
            
        Returns:
            Model metadata
        """
        # Generate model ID
        model_id = hashlib.md5(
            f"{name}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Create version
        version_id = self.version_control.create_version(
            name,
            model,
            tokenizer,
            version_str,
            metadata=metadata
        )
        
        # Get version info
        version_info = self.version_control.get_version(version_id)
        
        # Create model metadata
        model_metadata = ModelMetadata(
            model_id=model_id,
            name=name,
            version=version_info["version"],
            status=ModelStatus.STAGING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            architecture=model.config.architectures[0] if hasattr(model.config, "architectures") else "",
            num_parameters=version_info["num_parameters"],
            model_size_mb=version_info["model_size_mb"],
            metrics=metrics or {},
            tags=tags or [],
            metadata={
                "version_id": version_id,
                **(metadata or {})
            }
        )
        
        # Save to catalog
        self.catalog[model_id] = model_metadata
        self._save_catalog()
        
        # Update statistics
        self.stats["total_models"] += 1
        self.stats["total_versions"] += 1
        
        logger.info(f"Registered model {model_id}: {name}:{version_info['version']}")
        
        return model_metadata
    
    def promote_model(
        self,
        model_id: str,
        target_status: ModelStatus,
        deployment_env: Optional[DeploymentEnvironment] = None
    ) -> ModelMetadata:
        """
        Promote model to new status.
        
        Args:
            model_id: Model ID
            target_status: Target status
            deployment_env: Deployment environment
            
        Returns:
            Updated model metadata
        """
        if model_id not in self.catalog:
            raise ValueError(f"Model {model_id} not found")
        
        model_metadata = self.catalog[model_id]
        
        # Validate promotion
        if target_status == ModelStatus.PRODUCTION:
            # Check metrics
            if not model_metadata.metrics:
                raise ValueError("Cannot promote model without metrics")
            
            # Check minimum performance thresholds
            min_accuracy = 0.9
            if model_metadata.metrics.get("accuracy", 0) < min_accuracy:
                raise ValueError(
                    f"Model accuracy {model_metadata.metrics.get('accuracy', 0)} "
                    f"below minimum threshold {min_accuracy}"
                )
        
        # Update status
        old_status = model_metadata.status
        model_metadata.status = target_status
        model_metadata.updated_at = datetime.now()
        
        if deployment_env:
            model_metadata.deployment_env = deployment_env
        
        # Save updated catalog
        self._save_catalog()
        
        # Update statistics
        if target_status == ModelStatus.PRODUCTION:
            self.stats["production_models"] += 1
        elif target_status == ModelStatus.ARCHIVED:
            self.stats["archived_models"] += 1
        
        logger.info(
            f"Promoted model {model_id} from {old_status.value} "
            f"to {target_status.value}"
        )
        
        return model_metadata
    
    def deploy_model(
        self,
        model_id: str,
        environment: DeploymentEnvironment,
        endpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Deploy model to environment.
        
        Args:
            model_id: Model ID
            environment: Target environment
            endpoint: Optional endpoint URL
            
        Returns:
            Deployment information
        """
        if model_id not in self.catalog:
            raise ValueError(f"Model {model_id} not found")
        
        model_metadata = self.catalog[model_id]
        
        # Check if model is ready for deployment
        if model_metadata.status not in [ModelStatus.STAGING, ModelStatus.PRODUCTION]:
            raise ValueError(
                f"Model status {model_metadata.status.value} "
                f"not ready for deployment"
            )
        
        # Create deployment record
        deployment_id = hashlib.md5(
            f"{model_id}:{environment.value}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        deployment = {
            "deployment_id": deployment_id,
            "model_id": model_id,
            "model_name": model_metadata.name,
            "model_version": model_metadata.version,
            "environment": environment.value,
            "endpoint": endpoint,
            "deployed_at": datetime.now().isoformat(),
            "status": "active"
        }
        
        self.deployments[deployment_id] = deployment
        
        # Update model metadata
        model_metadata.deployment_env = environment
        model_metadata.endpoint = endpoint
        self._save_catalog()
        
        # Load model for deployment (placeholder)
        # In production, this would trigger actual deployment
        
        # Update statistics
        self.stats["total_deployments"] += 1
        
        logger.info(
            f"Deployed model {model_id} to {environment.value} "
            f"(deployment_id: {deployment_id})"
        )
        
        return deployment
    
    def compare_models(
        self,
        model_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple models.
        
        Args:
            model_ids: List of model IDs to compare
            metrics: Specific metrics to compare
            
        Returns:
            Comparison results
        """
        comparison = {
            "models": [],
            "metrics": {},
            "best_model": None
        }
        
        for model_id in model_ids:
            if model_id not in self.catalog:
                logger.warning(f"Model {model_id} not found")
                continue
            
            model_metadata = self.catalog[model_id]
            
            comparison["models"].append({
                "model_id": model_id,
                "name": model_metadata.name,
                "version": model_metadata.version,
                "status": model_metadata.status.value,
                "metrics": model_metadata.metrics,
                "model_size_mb": model_metadata.model_size_mb,
                "num_parameters": model_metadata.num_parameters
            })
            
            # Aggregate metrics
            for metric_name, metric_value in model_metadata.metrics.items():
                if metrics is None or metric_name in metrics:
                    if metric_name not in comparison["metrics"]:
                        comparison["metrics"][metric_name] = []
                    
                    comparison["metrics"][metric_name].append({
                        "model_id": model_id,
                        "value": metric_value
                    })
        
        # Find best model (by accuracy)
        if comparison["models"]:
            best_model = max(
                comparison["models"],
                key=lambda m: m["metrics"].get("accuracy", 0)
            )
            comparison["best_model"] = best_model["model_id"]
        
        return comparison
    
    def get_model_lineage(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get model lineage (version history).
        
        Args:
            model_id: Model ID
            
        Returns:
            List of model versions in lineage
        """
        if model_id not in self.catalog:
            raise ValueError(f"Model {model_id} not found")
        
        model_metadata = self.catalog[model_id]
        lineage = [model_metadata.to_dict()]
        
        # Trace back through parent versions
        current_version = model_metadata.metadata.get("version_id")
        
        while current_version:
            version_info = self.version_control.get_version(current_version)
            
            if version_info.get("parent_version"):
                parent_version = version_info["parent_version"]
                
                # Find model with this version
                for other_id, other_metadata in self.catalog.items():
                    if other_metadata.metadata.get("version_id") == parent_version:
                        lineage.append(other_metadata.to_dict())
                        current_version = parent_version
                        break
                else:
                    break
            else:
                break
        
        return lineage
    
    def cleanup_old_models(
        self,
        days: int = 30,
        keep_production: bool = True
    ) -> int:
        """
        Clean up old models.
        
        Args:
            days: Age threshold in days
            keep_production: Whether to keep production models
            
        Returns:
            Number of models cleaned up
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        cleaned_count = 0
        
        for model_id, model_metadata in list(self.catalog.items()):
            # Skip production models if requested
            if keep_production and model_metadata.status == ModelStatus.PRODUCTION:
                continue
            
            # Check age
            if model_metadata.updated_at < cutoff_date:
                # Archive model
                model_metadata.status = ModelStatus.ARCHIVED
                self._save_catalog()
                
                cleaned_count += 1
                logger.info(f"Archived old model {model_id}")
        
        return cleaned_count
    
    def _load_catalog(self):
        """Load model catalog from disk."""
        catalog_file = self.models_dir / "catalog.json"
        
        if catalog_file.exists():
            with open(catalog_file, "r") as f:
                catalog_data = json.load(f)
                
                for model_id, model_data in catalog_data.items():
                    # Convert to ModelMetadata
                    model_data["status"] = ModelStatus(model_data["status"])
                    model_data["created_at"] = datetime.fromisoformat(model_data["created_at"])
                    model_data["updated_at"] = datetime.fromisoformat(model_data["updated_at"])
                    
                    if model_data.get("deployment_env"):
                        model_data["deployment_env"] = DeploymentEnvironment(
                            model_data["deployment_env"]
                        )
                    
                    self.catalog[model_id] = ModelMetadata(**model_data)
    
    def _save_catalog(self):
        """Save model catalog to disk."""
        catalog_file = self.models_dir / "catalog.json"
        
        catalog_data = {
            model_id: model_metadata.to_dict()
            for model_id, model_metadata in self.catalog.items()
        }
        
        with open(catalog_file, "w") as f:
            json.dump(catalog_data, f, indent=2)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        return {
            **self.stats,
            "catalog_size": len(self.catalog),
            "active_deployments": len([
                d for d in self.deployments.values()
                if d["status"] == "active"
            ])
        }

# Global service instance
_model_management_service = None

def get_model_management_service() -> ModelManagementService:
    """Get model management service instance (singleton)."""
    global _model_management_service
    
    if _model_management_service is None:
        _model_management_service = ModelManagementService()
    
    return _model_management_service
