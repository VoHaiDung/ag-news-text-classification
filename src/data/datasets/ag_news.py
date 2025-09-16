"""
AG News Dataset Implementation
==============================

Implements AG News dataset loading and processing following:
- Zhang et al. (2015): "Character-level Convolutional Networks for Text Classification"
- Wolf et al. (2020): "Transformers: State-of-the-Art Natural Language Processing"

Author: Võ Hải Dũng
License: MIT
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizer
from datasets import load_dataset, DatasetDict

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.constants import (
    AG_NEWS_CLASSES,
    AG_NEWS_NUM_CLASSES,
    LABEL_TO_ID,
    ID_TO_LABEL,
    MAX_SEQUENCE_LENGTH,
    DATA_DIR
)
from src.utils.logging_config import setup_logging
from src.core.exceptions import DataError, DataValidationError

logger = setup_logging(__name__)

@dataclass
class AGNewsConfig:
    """
    Configuration for AG News dataset.
    
    Following configuration best practices from:
    - Paszke et al. (2019): "PyTorch: An Imperative Style, High-Performance Deep Learning Library"
    """
    data_dir: Path = field(default_factory=lambda: DATA_DIR / "processed")
    max_length: int = MAX_SEQUENCE_LENGTH
    truncation: bool = True
    padding: str = "max_length"
    return_tensors: str = "pt"
    
    # Data splits
    train_file: str = "train.csv"
    val_file: str = "validation.csv"
    test_file: str = "test.csv"
    
    # Processing options
    lowercase: bool = False
    remove_special_chars: bool = False
    normalize_whitespace: bool = True
    
    # Caching
    cache_dir: Optional[Path] = field(default_factory=lambda: DATA_DIR / ".cache")
    use_cache: bool = True
    
    # Validation
    validate_labels: bool = True
    min_text_length: int = 10
    
    def __post_init__(self):
        """Validate configuration."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
            
        # Create directories if needed
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

class AGNewsDataset(Dataset):
    """
    PyTorch Dataset for AG News.
    
    Implements efficient data loading following:
    - Murray & Chiang (2015): "Auto-Sizing Neural Networks"
    - Tan & Le (2019): "EfficientNet: Rethinking Model Scaling"
    """
    
    def __init__(
        self,
        config: AGNewsConfig,
        split: str = "train",
        tokenizer: Optional[PreTrainedTokenizer] = None,
        transform: Optional[Any] = None
    ):
        """
        Initialize AG News dataset.
        
        Args:
            config: Dataset configuration
            split: Data split (train/validation/test)
            tokenizer: Tokenizer for text processing
            transform: Optional data transformations
        """
        self.config = config
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Load data
        self._load_data()
        
        # Validate data
        if config.validate_labels:
            self._validate_data()
        
        logger.info(f"Loaded {len(self)} samples for {split} split")
        
    def _load_data(self):
        """Load data from files."""
        # Determine file path
        if self.split == "train":
            file_path = self.config.data_dir / self.config.train_file
        elif self.split in ["val", "validation", "dev"]:
            file_path = self.config.data_dir / self.config.val_file
        elif self.split == "test":
            file_path = self.config.data_dir / self.config.test_file
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        # Check if file exists
        if not file_path.exists():
            raise DataError(f"Data file not found: {file_path}")
        
        # Load CSV
        try:
            self.df = pd.read_csv(file_path)
            
            # Ensure required columns
            required_cols = ["text", "label"]
            if not all(col in self.df.columns for col in required_cols):
                raise DataValidationError(f"Missing required columns. Found: {self.df.columns.tolist()}")
            
            # Convert to lists for faster access
            self.texts = self.df["text"].tolist()
            self.labels = self.df["label"].tolist()
            
            # Add label names if available
            if "label_name" in self.df.columns:
                self.label_names = self.df["label_name"].tolist()
            else:
                self.label_names = [ID_TO_LABEL[label] for label in self.labels]
                
        except Exception as e:
            raise DataError(f"Failed to load data from {file_path}: {e}")
    
    def _validate_data(self):
        """
        Validate data quality.
        
        Following validation practices from:
        - Northcutt et al. (2021): "Pervasive Label Errors in Test Sets"
        """
        issues = []
        
        # Check labels
        unique_labels = set(self.labels)
        valid_labels = set(range(AG_NEWS_NUM_CLASSES))
        
        if not unique_labels.issubset(valid_labels):
            invalid = unique_labels - valid_labels
            issues.append(f"Invalid labels found: {invalid}")
        
        # Check text lengths
        text_lengths = [len(text.split()) for text in self.texts]
        
        too_short = sum(1 for length in text_lengths if length < self.config.min_text_length)
        if too_short > 0:
            issues.append(f"{too_short} texts shorter than {self.config.min_text_length} words")
        
        # Check for empty texts
        empty = sum(1 for text in self.texts if not text or not text.strip())
        if empty > 0:
            issues.append(f"{empty} empty texts found")
        
        # Log issues
        if issues:
            logger.warning(f"Data validation issues for {self.split}:")
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with input tensors
        """
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Apply text preprocessing if configured
        if self.config.normalize_whitespace:
            text = " ".join(text.split())
        
        if self.config.lowercase:
            text = text.lower()
        
        # Apply transform if provided
        if self.transform:
            text = self.transform(text)
        
        # Tokenize if tokenizer provided
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=self.config.truncation,
                padding=self.config.padding,
                max_length=self.config.max_length,
                return_tensors=self.config.return_tensors
            )
            
            # Flatten tensors
            item = {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "labels": torch.tensor(label, dtype=torch.long)
            }
            
            # Add token type ids if available
            if "token_type_ids" in encoding:
                item["token_type_ids"] = encoding["token_type_ids"].squeeze()
        else:
            # Return raw data
            item = {
                "text": text,
                "label": label,
                "label_name": self.label_names[idx]
            }
        
        # Add metadata
        item["idx"] = idx
        
        return item
    
    def get_labels(self) -> List[int]:
        """Get all labels."""
        return self.labels
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get label distribution."""
        from collections import Counter
        counts = Counter(self.labels)
        return {ID_TO_LABEL[label]: count for label, count in counts.items()}
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.
        
        Following statistical analysis from:
        - Swayamdipta et al. (2020): "Dataset Cartography"
        """
        text_lengths = [len(text.split()) for text in self.texts]
        
        return {
            "num_samples": len(self),
            "num_classes": AG_NEWS_NUM_CLASSES,
            "label_distribution": self.get_label_distribution(),
            "text_length": {
                "mean": np.mean(text_lengths),
                "std": np.std(text_lengths),
                "min": np.min(text_lengths),
                "max": np.max(text_lengths),
                "median": np.median(text_lengths)
            }
        }
    
    @classmethod
    def from_huggingface(
        cls,
        config: Optional[AGNewsConfig] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None
    ) -> DatasetDict:
        """
        Load AG News from Hugging Face datasets.
        
        Args:
            config: Dataset configuration
            tokenizer: Tokenizer for processing
            
        Returns:
            DatasetDict with train/test splits
        """
        logger.info("Loading AG News from Hugging Face datasets...")
        
        # Load dataset
        dataset = load_dataset("ag_news")
        
        # Create config if not provided
        if config is None:
            config = AGNewsConfig()
        
        # Process if tokenizer provided
        if tokenizer:
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=config.truncation,
                    padding=config.padding,
                    max_length=config.max_length
                )
            
            dataset = dataset.map(tokenize_function, batched=True)
            
            # Set format
            dataset.set_format(
                type="torch",
                columns=["input_ids", "attention_mask", "label"]
            )
        
        return dataset

def create_ag_news_datasets(
    config: AGNewsConfig,
    tokenizer: Optional[PreTrainedTokenizer] = None
) -> Tuple[AGNewsDataset, AGNewsDataset, AGNewsDataset]:
    """
    Create train, validation, and test datasets.
    
    Args:
        config: Dataset configuration
        tokenizer: Tokenizer for processing
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dataset = AGNewsDataset(config, split="train", tokenizer=tokenizer)
    val_dataset = AGNewsDataset(config, split="validation", tokenizer=tokenizer)
    test_dataset = AGNewsDataset(config, split="test", tokenizer=tokenizer)
    
    return train_dataset, val_dataset, test_dataset
