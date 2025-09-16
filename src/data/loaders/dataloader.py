"""
DataLoader Module
=================

Custom dataloaders with advanced features following:
- Smith et al. (2017): "Don't Decay the Learning Rate, Increase the Batch Size"
- McCandlish et al. (2018): "An Empirical Model of Large-Batch Training"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Optional, Dict, Any, List, Union, Callable
from dataclasses import dataclass
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

@dataclass
class DataLoaderConfig:
    """Configuration for DataLoader."""
    
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = False
    
    # Advanced options
    prefetch_factor: int = 2
    persistent_workers: bool = True
    
    # Dynamic batching
    use_dynamic_batching: bool = False
    max_tokens: int = 10000
    
    # Distributed
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    
    # Mixed precision
    mixed_precision: bool = False

class SmartDataLoader:
    """
    Smart DataLoader with advanced features.
    
    Implements efficient data loading following:
    - Johnson & Zhang (2013): "Accelerating Stochastic Gradient Descent"
    """
    
    def __init__(
        self,
        dataset: Dataset,
        config: DataLoaderConfig,
        collate_fn: Optional[Callable] = None,
        sampler: Optional[Sampler] = None
    ):
        """
        Initialize smart dataloader.
        
        Args:
            dataset: PyTorch dataset
            config: DataLoader configuration
            collate_fn: Custom collate function
            sampler: Custom sampler
        """
        self.dataset = dataset
        self.config = config
        self.collate_fn = collate_fn or self._default_collate
        
        # Setup sampler
        if config.distributed and sampler is None:
            sampler = DistributedSampler(
                dataset,
                num_replicas=config.world_size,
                rank=config.rank,
                shuffle=config.shuffle
            )
            shuffle = False
        else:
            shuffle = config.shuffle if sampler is None else False
        
        self.sampler = sampler
        
        # Create dataloader
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=self._get_effective_batch_size(),
            shuffle=shuffle,
            sampler=sampler,
            num_workers=config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
            persistent_workers=config.persistent_workers if config.num_workers > 0 else False
        )
        
        logger.info(f"Created DataLoader: batch_size={config.batch_size}, num_workers={config.num_workers}")
    
    def _get_effective_batch_size(self) -> int:
        """Calculate effective batch size considering gradient accumulation."""
        return self.config.batch_size // self.config.gradient_accumulation_steps
    
    def _default_collate(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Default collate function with padding.
        
        Following batching strategies from:
        - Ott et al. (2019): "fairseq: A Fast, Extensible Toolkit"
        """
        # Separate different types of data
        input_ids = []
        attention_masks = []
        labels = []
        
        for item in batch:
            if 'input_ids' in item:
                input_ids.append(item['input_ids'])
            if 'attention_mask' in item:
                attention_masks.append(item['attention_mask'])
            if 'labels' in item:
                labels.append(item['labels'])
        
        # Stack or pad as needed
        collated = {}
        
        if input_ids:
            if isinstance(input_ids[0], torch.Tensor):
                # Pad sequences to same length
                max_len = max(len(seq) for seq in input_ids)
                padded_input_ids = []
                padded_attention_masks = []
                
                for i, seq in enumerate(input_ids):
                    padding_length = max_len - len(seq)
                    if padding_length > 0:
                        padded_seq = torch.cat([
                            seq,
                            torch.zeros(padding_length, dtype=seq.dtype)
                        ])
                        padded_input_ids.append(padded_seq)
                        
                        if attention_masks:
                            mask = torch.cat([
                                attention_masks[i],
                                torch.zeros(padding_length, dtype=attention_masks[i].dtype)
                            ])
                            padded_attention_masks.append(mask)
                    else:
                        padded_input_ids.append(seq)
                        if attention_masks:
                            padded_attention_masks.append(attention_masks[i])
                
                collated['input_ids'] = torch.stack(padded_input_ids)
                if padded_attention_masks:
                    collated['attention_mask'] = torch.stack(padded_attention_masks)
            else:
                collated['input_ids'] = input_ids
        
        if labels:
            if isinstance(labels[0], torch.Tensor):
                collated['labels'] = torch.stack(labels)
            else:
                collated['labels'] = torch.tensor(labels)
        
        return collated
    
    def __iter__(self):
        """Iterate through dataloader."""
        return iter(self.dataloader)
    
    def __len__(self):
        """Get number of batches."""
        return len(self.dataloader)
    
    def set_epoch(self, epoch: int):
        """Set epoch for distributed sampler."""
        if isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **kwargs
) -> DataLoader:
    """
    Factory function to create dataloader.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of workers
        **kwargs: Additional arguments
        
    Returns:
        Configured DataLoader
    """
    config = DataLoaderConfig(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )
    
    smart_loader = SmartDataLoader(dataset, config)
    return smart_loader.dataloader

def create_train_val_test_loaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    **kwargs
) -> tuple:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of workers
        **kwargs: Additional arguments
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        **kwargs
    )
    
    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        **kwargs
    )
    
    test_loader = get_dataloader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        **kwargs
    )
    
    return train_loader, val_loader, test_loader
