"""
Data Service Module
===================

Provides high-level data management services for the AG News framework,
implementing service-oriented architecture patterns from:
- Evans (2003): "Domain-Driven Design"
- Richardson (2018): "Microservices Patterns"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import json
import hashlib
from datetime import datetime

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.core.exceptions import DataError, DataValidationError
from src.data.datasets.ag_news import AGNewsDataset
from src.data.datasets.external_news import ExternalNewsDataset, ExternalNewsConfig
from src.data.preprocessing.text_cleaner import TextCleaner, CleaningConfig
from src.data.preprocessing.tokenization import Tokenizer, TokenizationConfig
from src.data.augmentation.base_augmenter import BaseAugmenter
from configs.constants import (
    DATA_DIR,
    AG_NEWS_CLASSES,
    DEFAULT_SPLIT_RATIOS,
    MAX_SEQUENCE_LENGTH
)

logger = setup_logging(__name__)

@dataclass
class DataServiceConfig:
    """Configuration for data service."""
    
    data_dir: Path = DATA_DIR
    cache_dir: Optional[Path] = DATA_DIR / ".cache"
    
    # Processing
    max_length: int = MAX_SEQUENCE_LENGTH
    batch_size: int = 32
    num_workers: int = 4
    
    # Splits
    train_ratio: float = DEFAULT_SPLIT_RATIOS["train"]
    val_ratio: float = DEFAULT_SPLIT_RATIOS["validation"]
    test_ratio: float = DEFAULT_SPLIT_RATIOS["test"]
    
    # Augmentation
    augmentation_enabled: bool = False
    augmentation_factor: float = 1.0
    
    # Caching
    use_cache: bool = True
    cache_ttl: int = 86400  # 24 hours
    
    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

class DataService:
    """
    High-level data management service.
    
    Implements service layer pattern from:
    - Fowler (2002): "Patterns of Enterprise Application Architecture"
    """
    
    def __init__(self, config: Optional[DataServiceConfig] = None):
        """
        Initialize data service.
        
        Args:
            config: Service configuration
        """
        self.config = config or DataServiceConfig()
        
        # Component initialization
        self.text_cleaner = None
        self.tokenizer = None
        self.augmenter = None
        
        # Cache
        self._dataset_cache = {}
        self._loader_cache = {}
        
        # Statistics
        self.stats = {
            "datasets_loaded": 0,
            "samples_processed": 0,
            "augmentations_created": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
        
        logger.info("Data service initialized")
    
    def load_dataset(
        self,
        dataset_name: str = "ag_news",
        split: Optional[str] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        max_samples: Optional[int] = None
    ) -> Dataset:
        """
        Load dataset with caching.
        
        Args:
            dataset_name: Name of dataset
            split: Data split (train/validation/test)
            tokenizer: Optional tokenizer
            max_samples: Maximum samples to load
            
        Returns:
            Dataset instance
        """
        # Generate cache key
        cache_key = self._get_cache_key(dataset_name, split, max_samples)
        
        # Check cache
        if self.config.use_cache and cache_key in self._dataset_cache:
            self.stats["cache_hits"] += 1
            logger.info(f"Loading dataset from cache: {cache_key}")
            return self._dataset_cache[cache_key]
        
        self.stats["cache_misses"] += 1
        
        # Load dataset
        if dataset_name == "ag_news":
            dataset = self._load_ag_news(split, tokenizer, max_samples)
        elif dataset_name == "external_news":
            dataset = self._load_external_news(max_samples)
        else:
            raise DataError(f"Unknown dataset: {dataset_name}")
        
        # Cache dataset
        if self.config.use_cache:
            self._dataset_cache[cache_key] = dataset
        
        self.stats["datasets_loaded"] += 1
        
        return dataset
    
    def _load_ag_news(
        self,
        split: Optional[str],
        tokenizer: Optional[PreTrainedTokenizer],
        max_samples: Optional[int]
    ) -> AGNewsDataset:
        """Load AG News dataset."""
        from src.data.datasets.ag_news import AGNewsConfig
        
        config = AGNewsConfig(
            data_dir=self.config.data_dir,
            split=split or "train",
            max_samples=max_samples
        )
        
        dataset = AGNewsDataset(config, tokenizer=tokenizer)
        
        logger.info(f"Loaded AG News {split} with {len(dataset)} samples")
        
        return dataset
    
    def _load_external_news(
        self,
        max_samples: Optional[int]
    ) -> ExternalNewsDataset:
        """Load external news dataset."""
        config = ExternalNewsConfig(
            data_dir=self.config.data_dir / "external",
            max_samples=max_samples
        )
        
        dataset = ExternalNewsDataset(config)
        
        logger.info(f"Loaded external news with {len(dataset)} samples")
        
        return dataset
    
    def create_data_loaders(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> Dict[str, DataLoader]:
        """
        Create data loaders for datasets.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            batch_size: Batch size
            num_workers: Number of workers
            
        Returns:
            Dictionary of data loaders
        """
        batch_size = batch_size or self.config.batch_size
        num_workers = num_workers or self.config.num_workers
        
        loaders = {}
        
        # Training loader
        loaders["train"] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
        
        # Validation loader
        if val_dataset:
            loaders["validation"] = DataLoader(
                val_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        
        # Test loader
        if test_dataset:
            loaders["test"] = DataLoader(
                test_dataset,
                batch_size=batch_size * 2,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        
        logger.info(f"Created {len(loaders)} data loaders")
        
        return loaders
    
    def prepare_data(
        self,
        texts: List[str],
        labels: Optional[List[int]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        clean: bool = True,
        augment: bool = False
    ) -> Dict[str, Any]:
        """
        Prepare data for training/inference.
        
        Args:
            texts: Input texts
            labels: Optional labels
            tokenizer: Optional tokenizer
            clean: Whether to clean texts
            augment: Whether to augment data
            
        Returns:
            Prepared data dictionary
        """
        logger.info(f"Preparing {len(texts)} samples")
        
        # Text cleaning
        if clean:
            texts = self._clean_texts(texts)
        
        # Data augmentation
        if augment and self.config.augmentation_enabled:
            texts, labels = self._augment_data(texts, labels)
        
        # Tokenization
        if tokenizer:
            encodings = self._tokenize_texts(texts, tokenizer)
        else:
            encodings = None
        
        # Update statistics
        self.stats["samples_processed"] += len(texts)
        
        return {
            "texts": texts,
            "labels": labels,
            "encodings": encodings,
            "metadata": {
                "num_samples": len(texts),
                "cleaned": clean,
                "augmented": augment
            }
        }
    
    def _clean_texts(self, texts: List[str]) -> List[str]:
        """Clean texts using text cleaner."""
        if not self.text_cleaner:
            config = CleaningConfig(
                lowercase=False,
                normalize_whitespace=True,
                remove_urls=True,
                remove_emails=True
            )
            self.text_cleaner = TextCleaner(config)
        
        return self.text_cleaner.batch_clean(texts)
    
    def _tokenize_texts(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts."""
        if not self.tokenizer:
            config = TokenizationConfig(
                max_length=self.config.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            self.tokenizer = Tokenizer(config, tokenizer)
        
        return self.tokenizer.batch_tokenize(texts)
    
    def _augment_data(
        self,
        texts: List[str],
        labels: Optional[List[int]]
    ) -> Tuple[List[str], Optional[List[int]]]:
        """Augment data."""
        if not self.augmenter:
            # Initialize augmenter (placeholder)
            logger.warning("Augmenter not initialized")
            return texts, labels
        
        augmented_texts = []
        augmented_labels = []
        
        for i, text in enumerate(texts):
            # Original
            augmented_texts.append(text)
            if labels:
                augmented_labels.append(labels[i])
            
            # Augmented versions
            aug_texts = self.augmenter.augment_single(text)
            if isinstance(aug_texts, str):
                aug_texts = [aug_texts]
            
            for aug_text in aug_texts[:int(self.config.augmentation_factor)]:
                augmented_texts.append(aug_text)
                if labels:
                    augmented_labels.append(labels[i])
        
        self.stats["augmentations_created"] += len(augmented_texts) - len(texts)
        
        return augmented_texts, augmented_labels if labels else None
    
    def split_data(
        self,
        data: pd.DataFrame,
        stratify: bool = True,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/val/test sets.
        
        Args:
            data: Input DataFrame
            stratify: Whether to stratify by label
            seed: Random seed
            
        Returns:
            Tuple of (train, val, test) DataFrames
        """
        # Calculate split sizes
        train_size = self.config.train_ratio
        val_size = self.config.val_ratio / (1 - train_size)
        
        # First split: train vs (val + test)
        train_data, temp_data = train_test_split(
            data,
            train_size=train_size,
            stratify=data["label"] if stratify else None,
            random_state=seed
        )
        
        # Second split: val vs test
        val_data, test_data = train_test_split(
            temp_data,
            train_size=val_size,
            stratify=temp_data["label"] if stratify else None,
            random_state=seed
        )
        
        logger.info(f"Split data: train={len(train_data)}, "
                   f"val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data
    
    def validate_data(
        self,
        data: Union[pd.DataFrame, Dict[str, Any]]
    ) -> bool:
        """
        Validate data quality.
        
        Args:
            data: Data to validate
            
        Returns:
            True if valid
            
        Raises:
            DataValidationError: If validation fails
        """
        if isinstance(data, pd.DataFrame):
            # Check required columns
            required_columns = ["text", "label"]
            missing = set(required_columns) - set(data.columns)
            if missing:
                raise DataValidationError(f"Missing columns: {missing}")
            
            # Check for empty texts
            if data["text"].isna().any():
                raise DataValidationError("Found empty texts")
            
            # Check label range
            unique_labels = data["label"].unique()
            if not all(0 <= l < len(AG_NEWS_CLASSES) for l in unique_labels):
                raise DataValidationError(f"Invalid labels: {unique_labels}")
        
        elif isinstance(data, dict):
            # Check required keys
            if "texts" not in data:
                raise DataValidationError("Missing 'texts' key")
        
        else:
            raise DataValidationError(f"Unsupported data type: {type(data)}")
        
        return True
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments."""
        key_str = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get service statistics."""
        cache_hit_rate = (
            self.stats["cache_hits"] / 
            max(self.stats["cache_hits"] + self.stats["cache_misses"], 1)
        )
        
        return {
            **self.stats,
            "cache_hit_rate": cache_hit_rate,
            "cached_datasets": len(self._dataset_cache),
            "cached_loaders": len(self._loader_cache)
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self._dataset_cache.clear()
        self._loader_cache.clear()
        logger.info("Cleared data service caches")
    
    def save_processed_data(
        self,
        data: pd.DataFrame,
        output_path: Path,
        format: str = "csv"
    ):
        """
        Save processed data to file.
        
        Args:
            data: Data to save
            output_path: Output path
            format: Output format (csv/json/parquet)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "csv":
            data.to_csv(output_path, index=False)
        elif format == "json":
            data.to_json(output_path, orient="records", lines=True)
        elif format == "parquet":
            data.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved {len(data)} samples to {output_path}")

# Global service instance
_data_service = DataService()

# Convenience functions
def get_data_service(config: Optional[DataServiceConfig] = None) -> DataService:
    """Get data service instance."""
    global _data_service
    if config:
        _data_service = DataService(config)
    return _data_service

def load_dataset(dataset_name: str, **kwargs) -> Dataset:
    """Load dataset using service."""
    return _data_service.load_dataset(dataset_name, **kwargs)

def prepare_data(texts: List[str], **kwargs) -> Dict[str, Any]:
    """Prepare data using service."""
    return _data_service.prepare_data(texts, **kwargs)
