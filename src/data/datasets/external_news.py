"""
External News Dataset Module
=============================

Handles external news corpora for domain adaptation following:
- Gururangan et al. (2020): "Don't Stop Pretraining: Adapt Language Models to Domains and Tasks"
- Karpukhin et al. (2020): "Dense Passage Retrieval for Open-Domain Question Answering"
- Lee et al. (2020): "BioBERT: a pre-trained biomedical language representation model"

Author: Võ Hải Dũng
License: MIT
"""

import os
import json
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from dataclasses import dataclass, field
import random

import torch
from torch.utils.data import Dataset, IterableDataset
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizer

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.constants import (
    DATA_DIR,
    MAX_SEQUENCE_LENGTH,
    AG_NEWS_CLASSES
)
from src.utils.logging_config import setup_logging
from src.core.exceptions import DataError, DataValidationError
from src.data.preprocessing.text_cleaner import TextCleaner, CleaningConfig

logger = setup_logging(__name__)

@dataclass
class ExternalNewsConfig:
    """
    Configuration for external news dataset.
    
    Following configuration patterns from:
    - Raffel et al. (2020): "Exploring the Limits of Transfer Learning with T5"
    """
    # Data sources
    data_sources: List[str] = field(default_factory=lambda: ["news_crawl", "cc_news"])
    data_dir: Path = field(default_factory=lambda: DATA_DIR / "external")
    
    # Processing
    max_length: int = MAX_SEQUENCE_LENGTH
    min_length: int = 20
    max_samples: Optional[int] = None
    
    # Domain filtering
    filter_by_domain: bool = True
    domain_keywords: List[str] = field(default_factory=lambda: [
        "world", "politics", "sports", "business", "technology", "science"
    ])
    
    # Sampling
    sample_strategy: str = "uniform"  # uniform, weighted, adaptive
    sample_temperature: float = 1.0
    
    # Caching
    cache_dir: Optional[Path] = field(default_factory=lambda: DATA_DIR / ".cache" / "external")
    use_cache: bool = True
    
    # Quality filtering
    quality_threshold: float = 0.7
    remove_duplicates: bool = True
    
    def __post_init__(self):
        """Validate and setup configuration."""
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.cache_dir, str):
            self.cache_dir = Path(self.cache_dir)
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

class ExternalNewsDataset(Dataset):
    """
    Dataset for external news corpora.
    
    Implements efficient loading strategies from:
    - Shoeybi et al. (2019): "Megatron-LM: Training Multi-Billion Parameter Language Models"
    - Rajbhandari et al. (2020): "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
    """
    
    def __init__(
        self,
        config: ExternalNewsConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        transform: Optional[Any] = None,
        purpose: str = "pretraining"  # pretraining, augmentation, pseudo_labeling
    ):
        """
        Initialize external news dataset.
        
        Args:
            config: Dataset configuration
            tokenizer: Optional tokenizer
            transform: Optional transformations
            purpose: Purpose of dataset (affects processing)
        """
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        self.purpose = purpose
        
        # Initialize text cleaner
        self.cleaner = self._init_cleaner()
        
        # Load data from sources
        self.texts = []
        self.metadata = []
        self._load_external_data()
        
        # Apply quality filtering
        if config.quality_threshold > 0:
            self._filter_quality()
        
        # Remove duplicates
        if config.remove_duplicates:
            self._remove_duplicates()
        
        logger.info(f"Loaded {len(self.texts)} external news samples for {purpose}")
    
    def _init_cleaner(self) -> TextCleaner:
        """Initialize text cleaner for external data."""
        config = CleaningConfig(
            lowercase=False,
            remove_urls=True,
            remove_emails=True,
            normalize_whitespace=True,
            preserve_entities=True
        )
        return TextCleaner(config)
    
    def _load_external_data(self):
        """
        Load external news data from configured sources.
        
        Following data loading strategies from:
        - Liu et al. (2019): "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
        """
        for source in self.config.data_sources:
            source_path = self.config.data_dir / source
            
            if not source_path.exists():
                logger.warning(f"Data source not found: {source_path}")
                continue
            
            logger.info(f"Loading from {source}...")
            
            # Load based on file format
            if source_path.is_dir():
                self._load_from_directory(source_path)
            elif source_path.suffix == ".gz":
                self._load_from_gzip(source_path)
            elif source_path.suffix == ".jsonl":
                self._load_from_jsonl(source_path)
            elif source_path.suffix == ".txt":
                self._load_from_text(source_path)
            else:
                logger.warning(f"Unsupported format: {source_path}")
            
            # Check max samples
            if self.config.max_samples and len(self.texts) >= self.config.max_samples:
                self.texts = self.texts[:self.config.max_samples]
                self.metadata = self.metadata[:self.config.max_samples]
                break
    
    def _load_from_directory(self, dir_path: Path):
        """Load data from directory of files."""
        for file_path in sorted(dir_path.glob("*.txt"))[:100]:  # Limit for efficiency
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read().strip()
                if self._validate_text(text):
                    self.texts.append(self.cleaner.clean(text))
                    self.metadata.append({
                        'source': dir_path.name,
                        'file': file_path.name
                    })
    
    def _load_from_gzip(self, file_path: Path):
        """Load data from gzipped file."""
        with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f):
                if self.config.max_samples and len(self.texts) >= self.config.max_samples:
                    break
                
                text = line.strip()
                if self._validate_text(text):
                    self.texts.append(self.cleaner.clean(text))
                    self.metadata.append({
                        'source': file_path.stem,
                        'line': line_num
                    })
    
    def _load_from_jsonl(self, file_path: Path):
        """Load data from JSONL file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if self.config.max_samples and len(self.texts) >= self.config.max_samples:
                    break
                
                try:
                    data = json.loads(line)
                    text = data.get('text', '') or data.get('content', '')
                    
                    if self._validate_text(text):
                        self.texts.append(self.cleaner.clean(text))
                        self.metadata.append({
                            'source': file_path.stem,
                            'id': data.get('id', line_num)
                        })
                except json.JSONDecodeError:
                    continue
    
    def _load_from_text(self, file_path: Path):
        """Load data from plain text file."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
            # Split into paragraphs or sentences
            paragraphs = content.split('\n\n')
            
            for idx, para in enumerate(paragraphs):
                if self.config.max_samples and len(self.texts) >= self.config.max_samples:
                    break
                
                if self._validate_text(para):
                    self.texts.append(self.cleaner.clean(para))
                    self.metadata.append({
                        'source': file_path.stem,
                        'paragraph': idx
                    })
    
    def _validate_text(self, text: str) -> bool:
        """
        Validate text based on criteria.
        
        Following validation from:
        - Brown et al. (2020): "Language Models are Few-Shot Learners"
        """
        if not text or not text.strip():
            return False
        
        word_count = len(text.split())
        
        # Length checks
        if word_count < self.config.min_length:
            return False
        if word_count > self.config.max_length * 2:  # Allow longer for external
            return False
        
        # Domain filtering
        if self.config.filter_by_domain:
            text_lower = text.lower()
            has_keyword = any(keyword in text_lower for keyword in self.config.domain_keywords)
            if not has_keyword:
                return False
        
        return True
    
    def _filter_quality(self):
        """
        Filter texts by quality score.
        
        Following quality metrics from:
        - Raffel et al. (2020): "Exploring the Limits of Transfer Learning"
        """
        filtered_texts = []
        filtered_metadata = []
        
        for text, meta in zip(self.texts, self.metadata):
            score = self._compute_quality_score(text)
            
            if score >= self.config.quality_threshold:
                filtered_texts.append(text)
                filtered_metadata.append(meta)
        
        logger.info(f"Quality filtering: {len(self.texts)} -> {len(filtered_texts)}")
        
        self.texts = filtered_texts
        self.metadata = filtered_metadata
    
    def _compute_quality_score(self, text: str) -> float:
        """Compute text quality score."""
        scores = []
        
        # Length score
        word_count = len(text.split())
        optimal_length = 100
        length_score = 1.0 - abs(word_count - optimal_length) / optimal_length
        scores.append(max(0, length_score))
        
        # Vocabulary diversity
        words = text.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        scores.append(unique_ratio)
        
        # Alphanumeric ratio
        alpha_chars = sum(c.isalpha() or c.isspace() for c in text)
        alpha_ratio = alpha_chars / max(len(text), 1)
        scores.append(alpha_ratio)
        
        return np.mean(scores)
    
    def _remove_duplicates(self):
        """Remove duplicate texts using hashing."""
        seen_hashes = set()
        unique_texts = []
        unique_metadata = []
        
        for text, meta in zip(self.texts, self.metadata):
            text_hash = hash(text)
            
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_texts.append(text)
                unique_metadata.append(meta)
        
        logger.info(f"Deduplication: {len(self.texts)} -> {len(unique_texts)}")
        
        self.texts = unique_texts
        self.metadata = unique_metadata
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with text and metadata
        """
        text = self.texts[idx]
        meta = self.metadata[idx]
        
        # Apply transform if provided
        if self.transform:
            text = self.transform(text)
        
        # Prepare output based on purpose
        if self.purpose == "pretraining":
            # For MLM pretraining
            if self.tokenizer:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=self.config.max_length,
                    return_tensors="pt"
                )
                
                return {
                    "input_ids": encoding["input_ids"].squeeze(),
                    "attention_mask": encoding["attention_mask"].squeeze(),
                    "source": meta.get("source", "unknown")
                }
            else:
                return {"text": text, "metadata": meta}
        
        elif self.purpose == "augmentation":
            # For data augmentation
            return {
                "text": text,
                "label": None,  # No labels for external data
                "source": "external_news",
                "metadata": meta
            }
        
        elif self.purpose == "pseudo_labeling":
            # For pseudo-labeling
            return {
                "text": text,
                "pseudo_label": None,  # To be filled by model
                "confidence": None,
                "metadata": meta
            }
        
        else:
            return {"text": text, "metadata": meta}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        text_lengths = [len(text.split()) for text in self.texts]
        sources = [meta.get("source", "unknown") for meta in self.metadata]
        
        return {
            "num_samples": len(self.texts),
            "sources": list(set(sources)),
            "source_distribution": dict(pd.Series(sources).value_counts()),
            "text_length": {
                "mean": np.mean(text_lengths),
                "std": np.std(text_lengths),
                "min": np.min(text_lengths) if text_lengths else 0,
                "max": np.max(text_lengths) if text_lengths else 0,
                "median": np.median(text_lengths) if text_lengths else 0
            },
            "purpose": self.purpose
        }

class StreamingExternalNewsDataset(IterableDataset):
    """
    Streaming dataset for large external corpora.
    
    Following streaming strategies from:
    - Rae et al. (2021): "Scaling Language Models: Methods, Analysis & Insights from Training Gopher"
    """
    
    def __init__(
        self,
        config: ExternalNewsConfig,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        buffer_size: int = 10000
    ):
        """
        Initialize streaming dataset.
        
        Args:
            config: Dataset configuration
            tokenizer: Optional tokenizer
            buffer_size: Buffer size for shuffling
        """
        self.config = config
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Stream data from sources."""
        buffer = []
        
        for source in self.config.data_sources:
            source_path = self.config.data_dir / source
            
            if not source_path.exists():
                continue
            
            # Stream from file
            if source_path.suffix == ".gz":
                with gzip.open(source_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        text = line.strip()
                        
                        if len(text.split()) >= self.config.min_length:
                            buffer.append(text)
                            
                            if len(buffer) >= self.buffer_size:
                                # Shuffle and yield
                                random.shuffle(buffer)
                                for item in buffer:
                                    yield self._process_item(item)
                                buffer = []
        
        # Yield remaining items
        random.shuffle(buffer)
        for item in buffer:
            yield self._process_item(item)
    
    def _process_item(self, text: str) -> Dict[str, Any]:
        """Process single text item."""
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_length,
                return_tensors="pt"
            )
            
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze()
            }
        else:
            return {"text": text}
