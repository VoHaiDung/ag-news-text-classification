"""
Data Service for Dataset Management
================================================================================
Implements comprehensive data management capabilities including dataset loading,
preprocessing, augmentation, versioning, and quality validation following
data engineering best practices.

This service provides a centralized interface for all data operations ensuring
data consistency, quality, and traceability throughout the ML pipeline.

References:
    - Polyzotis, N., et al. (2017). Data Management Challenges in Production Machine Learning
    - Schelter, S., et al. (2018). Automating Large-Scale Data Quality Verification
    - Amershi, S., et al. (2019). Software Engineering for Machine Learning

Author: Võ Hải Dũng
License: MIT
"""

import hashlib
import io
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.datasets.ag_news import AGNewsDataset
from src.data.preprocessing.text_cleaner import TextCleaner
from src.data.augmentation.base_augmenter import BaseAugmenter
from src.services.base_service import BaseService, ServiceConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetInfo:
    """
    Dataset metadata and statistics.
    
    Attributes:
        name: Dataset name
        version: Dataset version
        description: Dataset description
        format: Data format
        total_samples: Total number of samples
        num_classes: Number of classes
        class_distribution: Class distribution
        created_at: Creation timestamp
        updated_at: Last update timestamp
        user_id: Owner user ID
        file_path: Path to dataset file
        size_bytes: Dataset size in bytes
        checksum: Dataset checksum for integrity
        preprocessing_applied: List of preprocessing steps
        is_augmented: Whether dataset is augmented
        parent_dataset: Parent dataset if derived
        metadata: Additional metadata
    """
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    format: str = "csv"
    total_samples: int = 0
    num_classes: int = 4
    class_distribution: Dict[str, int] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: Optional[str] = None
    file_path: Optional[str] = None
    size_bytes: int = 0
    checksum: Optional[str] = None
    preprocessing_applied: List[str] = field(default_factory=list)
    is_augmented: bool = False
    parent_dataset: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "format": self.format,
            "total_samples": self.total_samples,
            "num_classes": self.num_classes,
            "class_distribution": self.class_distribution,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "user_id": self.user_id,
            "file_path": self.file_path,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "preprocessing_applied": self.preprocessing_applied,
            "is_augmented": self.is_augmented,
            "parent_dataset": self.parent_dataset,
            "metadata": self.metadata
        }


class DatasetCache:
    """
    Cache for loaded datasets to improve performance.
    
    Implements LRU cache with memory management for datasets.
    """
    
    def __init__(self, max_size: int = 5, max_memory_gb: float = 4.0):
        """
        Initialize dataset cache.
        
        Args:
            max_size: Maximum number of datasets to cache
            max_memory_gb: Maximum memory usage in GB
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self._cache: Dict[str, Tuple[Dataset, int]] = {}
        self._access_order: List[str] = []
        self._total_memory = 0
    
    def get(self, key: str) -> Optional[Dataset]:
        """
        Get dataset from cache.
        
        Args:
            key: Dataset identifier
            
        Returns:
            Cached dataset or None
        """
        if key in self._cache:
            dataset, _ = self._cache[key]
            
            # Update access order
            self._access_order.remove(key)
            self._access_order.append(key)
            
            return dataset
        
        return None
    
    def put(self, key: str, dataset: Dataset, size_bytes: int) -> None:
        """
        Put dataset in cache.
        
        Args:
            key: Dataset identifier
            dataset: Dataset object
            size_bytes: Dataset size in bytes
        """
        # Check memory limit
        while self._total_memory + size_bytes > self.max_memory_bytes and self._cache:
            self._evict_oldest()
        
        # Check size limit
        while len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_oldest()
        
        # Add to cache
        if key in self._cache:
            # Update existing
            old_dataset, old_size = self._cache[key]
            self._total_memory -= old_size
            self._access_order.remove(key)
        
        self._cache[key] = (dataset, size_bytes)
        self._access_order.append(key)
        self._total_memory += size_bytes
    
    def _evict_oldest(self) -> None:
        """Evict oldest dataset from cache."""
        if self._access_order:
            oldest = self._access_order.pop(0)
            dataset, size = self._cache.pop(oldest)
            self._total_memory -= size
            logger.debug(f"Evicted dataset '{oldest}' from cache")
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        self._total_memory = 0


class DataService(BaseService):
    """
    Service for comprehensive data management.
    
    Provides dataset loading, preprocessing, augmentation, versioning,
    and quality validation capabilities for the classification system.
    """
    
    def __init__(self, config: ServiceConfig):
        """
        Initialize data service.
        
        Args:
            config: Service configuration
        """
        super().__init__(config)
        
        # Storage paths
        self.data_dir = Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.augmented_dir = self.data_dir / "augmented"
        
        # Ensure directories exist
        for dir_path in [self.raw_dir, self.processed_dir, self.augmented_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Dataset registry
        self.datasets: Dict[str, DatasetInfo] = {}
        self.dataset_cache = DatasetCache()
        
        # Preprocessing components
        self.text_cleaner = TextCleaner()
        
        # Statistics
        self._datasets_loaded = 0
        self._datasets_created = 0
        self._total_samples_processed = 0
    
    async def _initialize(self) -> None:
        """Initialize service components."""
        logger.info("Initializing data service")
        
        # Load dataset registry
        await self._load_dataset_registry()
        
        # Initialize default AG News dataset if not exists
        if not await self.dataset_exists("ag_news"):
            await self._initialize_ag_news_dataset()
    
    async def _shutdown(self) -> None:
        """Cleanup service resources."""
        logger.info("Shutting down data service")
        
        # Save dataset registry
        await self._save_dataset_registry()
        
        # Clear cache
        self.dataset_cache.clear()
    
    async def upload_dataset(
        self,
        name: str,
        content: bytes,
        format: str,
        description: Optional[str] = None,
        user_id: Optional[str] = None,
        validate: bool = True
    ) -> str:
        """
        Upload and register a new dataset.
        
        Args:
            name: Dataset name
            content: Dataset content
            format: Data format (csv, json, jsonl, etc.)
            description: Dataset description
            user_id: User ID of uploader
            validate: Whether to validate data
            
        Returns:
            Dataset ID
            
        Raises:
            ValueError: If validation fails
        """
        logger.info(f"Uploading dataset '{name}' ({len(content)} bytes)")
        
        try:
            # Parse dataset based on format
            if format == "csv":
                df = pd.read_csv(io.BytesIO(content))
            elif format == "json":
                df = pd.read_json(io.BytesIO(content))
            elif format == "jsonl":
                df = pd.read_json(io.BytesIO(content), lines=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Validate dataset
            if validate:
                validation_result = await self._validate_dataset_df(df)
                if not validation_result["is_valid"]:
                    raise ValueError(f"Dataset validation failed: {validation_result['errors']}")
            
            # Calculate statistics
            class_distribution = {}
            if "category" in df.columns:
                class_distribution = df["category"].value_counts().to_dict()
            elif "label" in df.columns:
                class_distribution = df["label"].value_counts().to_dict()
            
            # Save dataset
            file_path = self.raw_dir / f"{name}.{format}"
            
            if format == "csv":
                df.to_csv(file_path, index=False)
            elif format == "json":
                df.to_json(file_path, orient="records")
            elif format == "jsonl":
                df.to_json(file_path, orient="records", lines=True)
            
            # Calculate checksum
            checksum = hashlib.sha256(content).hexdigest()
            
            # Create dataset info
            dataset_info = DatasetInfo(
                name=name,
                description=description,
                format=format,
                total_samples=len(df),
                num_classes=len(class_distribution),
                class_distribution=class_distribution,
                user_id=user_id,
                file_path=str(file_path),
                size_bytes=len(content),
                checksum=checksum
            )
            
            # Register dataset
            dataset_id = f"{name}_{checksum[:8]}"
            self.datasets[name] = dataset_info
            self._datasets_created += 1
            
            # Save registry
            await self._save_dataset_registry()
            
            logger.info(f"Dataset '{name}' uploaded successfully (ID: {dataset_id})")
            
            return dataset_id
            
        except Exception as e:
            logger.error(f"Failed to upload dataset: {str(e)}")
            raise ValueError(f"Failed to upload dataset: {str(e)}")
    
    async def dataset_exists(self, name: str) -> bool:
        """
        Check if dataset exists.
        
        Args:
            name: Dataset name
            
        Returns:
            True if dataset exists
        """
        return name in self.datasets
    
    async def get_dataset_info(self, name: str) -> Dict[str, Any]:
        """
        Get dataset information.
        
        Args:
            name: Dataset name
            
        Returns:
            Dataset information dictionary
        """
        if name not in self.datasets:
            return None
        
        return self.datasets[name].to_dict()
    
    async def get_dataset(self, name: str) -> Optional[Dataset]:
        """
        Get dataset for training.
        
        Args:
            name: Dataset name
            
        Returns:
            PyTorch Dataset object
        """
        # Check cache
        cached = self.dataset_cache.get(name)
        if cached is not None:
            logger.debug(f"Dataset '{name}' loaded from cache")
            return cached
        
        if name not in self.datasets:
            logger.warning(f"Dataset '{name}' not found")
            return None
        
        dataset_info = self.datasets[name]
        
        try:
            # Load dataset
            if name == "ag_news":
                # Use built-in AG News dataset
                dataset = AGNewsDataset(split="train")
            else:
                # Load custom dataset
                file_path = Path(dataset_info.file_path)
                
                if dataset_info.format == "csv":
                    df = pd.read_csv(file_path)
                elif dataset_info.format == "json":
                    df = pd.read_json(file_path)
                elif dataset_info.format == "jsonl":
                    df = pd.read_json(file_path, lines=True)
                else:
                    raise ValueError(f"Unsupported format: {dataset_info.format}")
                
                # Create PyTorch dataset
                dataset = self._create_pytorch_dataset(df)
            
            # Cache dataset
            size_estimate = dataset_info.size_bytes
            self.dataset_cache.put(name, dataset, size_estimate)
            
            self._datasets_loaded += 1
            logger.info(f"Dataset '{name}' loaded successfully")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset '{name}': {str(e)}")
            return None
    
    async def download_dataset(
        self,
        name: str,
        format: str = "csv",
        sample_size: Optional[int] = None
    ) -> bytes:
        """
        Download dataset in specified format.
        
        Args:
            name: Dataset name
            format: Download format
            sample_size: Optional sample size
            
        Returns:
            Dataset content as bytes
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        dataset_info = self.datasets[name]
        file_path = Path(dataset_info.file_path)
        
        # Load dataset
        if dataset_info.format == "csv":
            df = pd.read_csv(file_path)
        elif dataset_info.format == "json":
            df = pd.read_json(file_path)
        elif dataset_info.format == "jsonl":
            df = pd.read_json(file_path, lines=True)
        else:
            # Try to read as CSV
            df = pd.read_csv(file_path)
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
        
        # Convert to requested format
        buffer = io.BytesIO()
        
        if format == "csv":
            df.to_csv(buffer, index=False)
        elif format == "json":
            df.to_json(buffer, orient="records")
        elif format == "jsonl":
            df.to_json(buffer, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported download format: {format}")
        
        return buffer.getvalue()
    
    async def list_datasets(
        self,
        user_id: Optional[str] = None,
        include_system: bool = False,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        List available datasets.
        
        Args:
            user_id: Filter by user ID
            include_system: Include system datasets
            filters: Additional filters
            
        Returns:
            List of dataset information
        """
        datasets = []
        
        for name, info in self.datasets.items():
            # Apply filters
            if user_id and info.user_id != user_id:
                continue
            
            if not include_system and info.user_id is None:
                continue
            
            if filters:
                # Apply custom filters
                if "min_samples" in filters and info.total_samples < filters["min_samples"]:
                    continue
                
                if "format" in filters and info.format != filters["format"]:
                    continue
            
            datasets.append(info.to_dict())
        
        # Sort by creation time (newest first)
        datasets.sort(key=lambda d: d["created_at"], reverse=True)
        
        return datasets
    
    async def preprocess_dataset(
        self,
        name: str,
        preprocessing_config: Dict[str, Any],
        save_as: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply preprocessing to dataset.
        
        Args:
            name: Dataset name
            preprocessing_config: Preprocessing configuration
            save_as: Optional name for preprocessed dataset
            user_id: User ID
            
        Returns:
            Preprocessing result
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        dataset_info = self.datasets[name]
        file_path = Path(dataset_info.file_path)
        
        # Load dataset
        if dataset_info.format == "csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_json(file_path, lines=True)
        
        # Apply preprocessing
        preprocessing_applied = []
        
        # Text cleaning
        if preprocessing_config.get("clean_text", True):
            df["text"] = df["text"].apply(self.text_cleaner.clean)
            preprocessing_applied.append("text_cleaning")
        
        # Lowercase
        if preprocessing_config.get("lowercase", False):
            df["text"] = df["text"].str.lower()
            preprocessing_applied.append("lowercase")
        
        # Remove short texts
        min_length = preprocessing_config.get("min_length", 10)
        if min_length > 0:
            df = df[df["text"].str.len() >= min_length]
            preprocessing_applied.append(f"min_length_{min_length}")
        
        # Remove duplicates
        if preprocessing_config.get("remove_duplicates", True):
            df = df.drop_duplicates(subset=["text"])
            preprocessing_applied.append("remove_duplicates")
        
        # Save preprocessed dataset
        if save_as:
            new_path = self.processed_dir / f"{save_as}.csv"
            df.to_csv(new_path, index=False)
            
            # Create new dataset info
            new_info = DatasetInfo(
                name=save_as,
                description=f"Preprocessed version of {name}",
                format="csv",
                total_samples=len(df),
                num_classes=dataset_info.num_classes,
                class_distribution=df["category"].value_counts().to_dict() if "category" in df else {},
                user_id=user_id,
                file_path=str(new_path),
                size_bytes=new_path.stat().st_size,
                preprocessing_applied=preprocessing_applied,
                parent_dataset=name
            )
            
            self.datasets[save_as] = new_info
            await self._save_dataset_registry()
        
        self._total_samples_processed += len(df)
        
        return {
            "original_samples": dataset_info.total_samples,
            "processed_samples": len(df),
            "preprocessing_applied": preprocessing_applied,
            "saved_as": save_as
        }
    
    async def augment_dataset(
        self,
        name: str,
        augmentation_config: Dict[str, Any],
        factor: float = 2.0,
        save_as: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Apply data augmentation to dataset.
        
        Args:
            name: Dataset name
            augmentation_config: Augmentation configuration
            factor: Augmentation factor
            save_as: Optional name for augmented dataset
            user_id: User ID
            
        Returns:
            Augmentation result
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        dataset_info = self.datasets[name]
        file_path = Path(dataset_info.file_path)
        
        # Load dataset
        if dataset_info.format == "csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_json(file_path, lines=True)
        
        # Apply augmentation
        augmented_samples = []
        augmentation_methods = []
        
        # Back translation
        if augmentation_config.get("back_translation", False):
            # Simplified mock implementation
            for _, row in df.iterrows():
                augmented_text = row["text"] + " (translated)"
                augmented_samples.append({
                    "text": augmented_text,
                    "category": row.get("category", row.get("label"))
                })
            augmentation_methods.append("back_translation")
        
        # Paraphrase
        if augmentation_config.get("paraphrase", False):
            for _, row in df.iterrows():
                augmented_text = "In other words, " + row["text"]
                augmented_samples.append({
                    "text": augmented_text,
                    "category": row.get("category", row.get("label"))
                })
            augmentation_methods.append("paraphrase")
        
        # Add augmented samples
        if augmented_samples:
            augmented_df = pd.DataFrame(augmented_samples)
            df = pd.concat([df, augmented_df], ignore_index=True)
        
        # Save augmented dataset
        if save_as:
            new_path = self.augmented_dir / f"{save_as}.csv"
            df.to_csv(new_path, index=False)
            
            # Create new dataset info
            new_info = DatasetInfo(
                name=save_as,
                description=f"Augmented version of {name}",
                format="csv",
                total_samples=len(df),
                num_classes=dataset_info.num_classes,
                class_distribution=df["category"].value_counts().to_dict() if "category" in df else {},
                user_id=user_id,
                file_path=str(new_path),
                size_bytes=new_path.stat().st_size,
                is_augmented=True,
                parent_dataset=name,
                metadata={"augmentation_methods": augmentation_methods}
            )
            
            self.datasets[save_as] = new_info
            await self._save_dataset_registry()
        
        return {
            "original_samples": dataset_info.total_samples,
            "augmented_samples": len(df),
            "augmentation_factor": len(df) / dataset_info.total_samples,
            "augmentation_methods": augmentation_methods,
            "saved_as": save_as
        }
    
    async def validate_dataset(
        self,
        name: str,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate dataset quality.
        
        Args:
            name: Dataset name
            validation_rules: Custom validation rules
            
        Returns:
            Validation result
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        dataset_info = self.datasets[name]
        file_path = Path(dataset_info.file_path)
        
        # Load dataset
        if dataset_info.format == "csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_json(file_path, lines=True)
        
        return await self._validate_dataset_df(df, validation_rules)
    
    async def _validate_dataset_df(
        self,
        df: pd.DataFrame,
        validation_rules: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Validate dataframe.
        
        Args:
            df: Dataframe to validate
            validation_rules: Validation rules
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check required columns
        required_columns = ["text"]
        if validation_rules:
            required_columns.extend(validation_rules.get("required_columns", []))
        
        for col in required_columns:
            if col not in df.columns:
                errors.append(f"Missing required column: {col}")
        
        # Check for label column
        if "category" not in df.columns and "label" not in df.columns:
            errors.append("Missing label column (category or label)")
        
        # Check for empty values
        if "text" in df.columns:
            empty_texts = df["text"].isna().sum()
            if empty_texts > 0:
                errors.append(f"Found {empty_texts} empty text values")
        
        # Check text length
        if "text" in df.columns:
            min_length = validation_rules.get("min_text_length", 1) if validation_rules else 1
            short_texts = (df["text"].str.len() < min_length).sum()
            if short_texts > 0:
                warnings.append(f"Found {short_texts} texts shorter than {min_length} characters")
        
        # Check class balance
        if "category" in df.columns:
            class_counts = df["category"].value_counts()
            min_class_size = class_counts.min()
            max_class_size = class_counts.max()
            imbalance_ratio = max_class_size / min_class_size if min_class_size > 0 else float("inf")
            
            if imbalance_ratio > 10:
                warnings.append(f"High class imbalance detected (ratio: {imbalance_ratio:.2f})")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=["text"]).sum()
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate texts")
        
        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "total_samples": len(df),
            "num_columns": len(df.columns),
            "columns": list(df.columns)
        }
    
    async def delete_dataset(self, name: str, force: bool = False) -> None:
        """
        Delete a dataset.
        
        Args:
            name: Dataset name
            force: Force deletion even if referenced
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        # Check if dataset is referenced by others
        if not force:
            for other_name, other_info in self.datasets.items():
                if other_info.parent_dataset == name:
                    raise ValueError(
                        f"Dataset '{name}' is referenced by '{other_name}'. "
                        "Use force=True to delete anyway."
                    )
        
        dataset_info = self.datasets[name]
        
        # Delete file
        if dataset_info.file_path:
            file_path = Path(dataset_info.file_path)
            if file_path.exists():
                file_path.unlink()
        
        # Remove from registry
        del self.datasets[name]
        
        # Clear from cache
        self.dataset_cache.clear()
        
        # Save registry
        await self._save_dataset_registry()
        
        logger.info(f"Dataset '{name}' deleted")
    
    async def get_dataset_statistics(self, name: str) -> Dict[str, Any]:
        """
        Get detailed dataset statistics.
        
        Args:
            name: Dataset name
            
        Returns:
            Statistics dictionary
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        dataset_info = self.datasets[name]
        file_path = Path(dataset_info.file_path)
        
        # Load dataset
        if dataset_info.format == "csv":
            df = pd.read_csv(file_path)
        else:
            df = pd.read_json(file_path, lines=True)
        
        statistics = {
            "basic": {
                "total_samples": len(df),
                "num_columns": len(df.columns),
                "columns": list(df.columns),
                "memory_usage_bytes": df.memory_usage(deep=True).sum()
            }
        }
        
        # Text statistics
        if "text" in df.columns:
            text_lengths = df["text"].str.len()
            statistics["text"] = {
                "avg_length": float(text_lengths.mean()),
                "min_length": int(text_lengths.min()),
                "max_length": int(text_lengths.max()),
                "std_length": float(text_lengths.std()),
                "median_length": float(text_lengths.median())
            }
        
        # Class distribution
        if "category" in df.columns:
            class_dist = df["category"].value_counts()
            statistics["classes"] = {
                "distribution": class_dist.to_dict(),
                "num_classes": len(class_dist),
                "min_class_size": int(class_dist.min()),
                "max_class_size": int(class_dist.max()),
                "class_balance": float(class_dist.min() / class_dist.max())
            }
        
        return statistics
    
    def _create_pytorch_dataset(self, df: pd.DataFrame) -> Dataset:
        """Create PyTorch dataset from dataframe."""
        
        class CustomDataset(Dataset):
            def __init__(self, dataframe):
                self.df = dataframe
                self.texts = dataframe["text"].tolist()
                
                # Handle different label column names
                if "category" in dataframe.columns:
                    self.labels = dataframe["category"].tolist()
                elif "label" in dataframe.columns:
                    self.labels = dataframe["label"].tolist()
                else:
                    self.labels = [0] * len(self.texts)
            
            def __len__(self):
                return len(self.df)
            
            def __getitem__(self, idx):
                return {
                    "text": self.texts[idx],
                    "label": self.labels[idx]
                }
        
        return CustomDataset(df)
    
    async def _initialize_ag_news_dataset(self) -> None:
        """Initialize default AG News dataset."""
        try:
            # Create dataset info for AG News
            dataset_info = DatasetInfo(
                name="ag_news",
                description="AG News Classification Dataset",
                format="internal",
                total_samples=120000,
                num_classes=4,
                class_distribution={
                    "World": 30000,
                    "Sports": 30000,
                    "Business": 30000,
                    "Technology": 30000
                },
                file_path="internal:ag_news"
            )
            
            self.datasets["ag_news"] = dataset_info
            logger.info("Initialized AG News dataset")
            
        except Exception as e:
            logger.error(f"Failed to initialize AG News dataset: {str(e)}")
    
    async def _load_dataset_registry(self) -> None:
        """Load dataset registry from disk."""
        registry_file = self.data_dir / "dataset_registry.json"
        
        if not registry_file.exists():
            return
        
        try:
            with open(registry_file, "r") as f:
                registry_data = json.load(f)
            
            for name, info_dict in registry_data.items():
                # Reconstruct DatasetInfo
                info = DatasetInfo(
                    name=info_dict["name"],
                    version=info_dict.get("version", "1.0.0"),
                    description=info_dict.get("description"),
                    format=info_dict.get("format", "csv"),
                    total_samples=info_dict.get("total_samples", 0),
                    num_classes=info_dict.get("num_classes", 4),
                    class_distribution=info_dict.get("class_distribution", {}),
                    user_id=info_dict.get("user_id"),
                    file_path=info_dict.get("file_path"),
                    size_bytes=info_dict.get("size_bytes", 0),
                    checksum=info_dict.get("checksum"),
                    preprocessing_applied=info_dict.get("preprocessing_applied", []),
                    is_augmented=info_dict.get("is_augmented", False),
                    parent_dataset=info_dict.get("parent_dataset"),
                    metadata=info_dict.get("metadata", {})
                )
                
                # Parse timestamps
                if "created_at" in info_dict:
                    info.created_at = datetime.fromisoformat(info_dict["created_at"])
                if "updated_at" in info_dict:
                    info.updated_at = datetime.fromisoformat(info_dict["updated_at"])
                
                self.datasets[name] = info
            
            logger.info(f"Loaded {len(self.datasets)} datasets from registry")
            
        except Exception as e:
            logger.error(f"Failed to load dataset registry: {str(e)}")
    
    async def _save_dataset_registry(self) -> None:
        """Save dataset registry to disk."""
        registry_file = self.data_dir / "dataset_registry.json"
        
        try:
            registry_data = {
                name: info.to_dict()
                for name, info in self.datasets.items()
            }
            
            with open(registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)
            
            logger.debug(f"Saved {len(registry_data)} datasets to registry")
            
        except Exception as e:
            logger.error(f"Failed to save dataset registry: {str(e)}")
    
    async def _execute(self, *args, **kwargs) -> Any:
        """Execute service operation."""
        # Not directly callable
        raise NotImplementedError("Data service operations must be called directly")
