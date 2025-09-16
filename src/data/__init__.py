"""
Data Module for AG News Classification
======================================

This module provides comprehensive data handling following best practices from:
- Gebru et al. (2021): "Datasheets for Datasets"
- Sambasivan et al. (2021): "Everyone wants to do the model work, not the data work"

Author: Võ Hải Dũng
License: MIT
"""

from .datasets.ag_news import AGNewsDataset
from .datasets.external_news import ExternalNewsDataset
from .datasets.combined_dataset import CombinedDataset
from .datasets.prompted_dataset import PromptedDataset

from .preprocessing.text_cleaner import TextCleaner
from .preprocessing.tokenization import Tokenizer
from .preprocessing.feature_extraction import FeatureExtractor

from .loaders.dataloader import get_dataloader, DataLoaderConfig
from .sampling.balanced_sampler import BalancedSampler

__all__ = [
    # Datasets
    "AGNewsDataset",
    "ExternalNewsDataset", 
    "CombinedDataset",
    "PromptedDataset",
    
    # Preprocessing
    "TextCleaner",
    "Tokenizer",
    "FeatureExtractor",
    
    # Loaders
    "get_dataloader",
    "DataLoaderConfig",
    
    # Sampling
    "BalancedSampler",
]

__version__ = "1.0.0"
