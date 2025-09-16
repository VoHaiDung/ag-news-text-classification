"""
Dynamic Batching Module
=======================

Implements dynamic batching for efficient GPU utilization following:
- Ott et al. (2019): "fairseq: A Fast, Extensible Toolkit"
- Popel & Bojar (2018): "Training Tips for the Transformer Model"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import List, Dict, Any, Optional, Iterator
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, Sampler
import numpy as np

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class DynamicBatchingConfig:
    """Configuration for dynamic batching."""
    
    max_tokens: int = 10000
    max_sentences: int = 64
    
    # Length-based batching
    sort_by_length: bool = True
    bucket_size: int = 1000
    
    # Padding efficiency
    pad_to_multiple_of: int = 8
    
    # Memory constraints
    max_memory_gb: float = 8.0

class DynamicBatchSampler(Sampler):
    """
    Dynamic batch sampler that groups similar-length sequences.
    
    Following batching strategies from:
    - Morishita et al. (2017): "An Empirical Study of Mini-Batch Creation Strategies"
    """
    
    def __init__(
        self,
        dataset: Dataset,
        config: DynamicBatchingConfig,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize dynamic batch sampler.
        
        Args:
            dataset: Dataset to sample from
            config: Batching configuration
            shuffle: Whether to shuffle
            seed: Random seed
        """
        self.dataset = dataset
        self.config = config
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        
        # Get sequence lengths
        self.lengths = self._get_lengths()
        
        # Create buckets
        self.buckets = self._create_buckets()
        
        logger.info(f"Created dynamic batch sampler with {len(self.buckets)} buckets")
    
    def _get_lengths(self) -> List[int]:
        """Get sequence lengths from dataset."""
        lengths = []
        
        for i in range(len(self.dataset)):
            item = self.dataset[i]
            
            if isinstance(item, dict):
                if 'input_ids' in item:
                    length = len(item['input_ids'])
                elif 'text' in item:
                    length = len(item['text'].split())
                else:
                    length = 1
            else:
                length = 1
            
            lengths.append(length)
        
        return lengths
    
    def _create_buckets(self) -> List[List[int]]:
        """Create buckets of similar-length sequences."""
        # Sort indices by length
        sorted_indices = np.argsort(self.lengths)
        
        # Create buckets
        buckets = []
        current_bucket = []
        current_tokens = 0
        
        for idx in sorted_indices:
            length = self.lengths[idx]
            
            # Check if adding this sample exceeds limits
            new_tokens = current_tokens + length * (len(current_bucket) + 1)
            
            if (new_tokens > self.config.max_tokens or
                len(current_bucket) >= self.config.max_sentences):
                # Start new bucket
                if current_bucket:
                    buckets.append(current_bucket)
                current_bucket = [idx]
                current_tokens = length
            else:
                current_bucket.append(idx)
                current_tokens = new_tokens
        
        # Add last bucket
        if current_bucket:
            buckets.append(current_bucket)
        
        return buckets
    
    def __iter__(self) -> Iterator[List[int]]:
        """Iterate through batches."""
        # Shuffle buckets if needed
        bucket_order = list(range(len(self.buckets)))
        
        if self.shuffle:
            self.rng.shuffle(bucket_order)
        
        # Yield batches
        for bucket_idx in bucket_order:
            bucket = self.buckets[bucket_idx].copy()
            
            if self.shuffle:
                self.rng.shuffle(bucket)
            
            yield bucket
    
    def __len__(self) -> int:
        """Get number of batches."""
        return len(self.buckets)
    
    def set_epoch(self, epoch: int):
        """Set epoch for reproducibility."""
        self.rng = np.random.RandomState(42 + epoch)

class DynamicBatchingDataLoader:
    """
    DataLoader with dynamic batching support.
    
    Optimizes GPU memory usage following:
    - Kitaev et al. (2020): "Reformer: The Efficient Transformer"
    """
    
    def __init__(
        self,
        dataset: Dataset,
        config: DynamicBatchingConfig,
        collate_fn: Optional[Any] = None,
        num_workers: int = 4
    ):
        """
        Initialize dynamic batching dataloader.
        
        Args:
            dataset: Dataset to load
            config: Batching configuration
            collate_fn: Collate function
            num_workers: Number of workers
        """
        self.dataset = dataset
        self.config = config
        self.collate_fn = collate_fn
        
        # Create sampler
        self.sampler = DynamicBatchSampler(dataset, config)
        
        # Create dataloader
        from torch.utils.data import DataLoader
        
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=self.sampler,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True
        )
    
    def __iter__(self):
        """Iterate through batches."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Get number of batches."""
        return len(self.sampler)
