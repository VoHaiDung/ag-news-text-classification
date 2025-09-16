"""
Prefetch DataLoader
===================

Implements data prefetching for improved GPU utilization following:
- NVIDIA DALI documentation
- PyTorch DataLoader best practices

Author: Võ Hải Dũng
License: MIT
"""

import logging
import torch
from torch.utils.data import DataLoader
import threading
import queue

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging

logger = setup_logging(__name__)

class PrefetchLoader:
    """
    DataLoader with prefetching to GPU.
    
    Reduces GPU idle time following:
    - NVIDIA Apex data prefetching
    """
    
    def __init__(self, dataloader: DataLoader, device: torch.device = None):
        """
        Initialize prefetch loader.
        
        Args:
            dataloader: Base dataloader
            device: Target device
        """
        self.dataloader = dataloader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def __iter__(self):
        """Iterate with prefetching."""
        stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        first = True
        
        for next_batch in self.dataloader:
            if stream is not None:
                with torch.cuda.stream(stream):
                    next_batch = self._to_device(next_batch)
            else:
                next_batch = self._to_device(next_batch)
            
            if not first:
                yield batch
            else:
                first = False
            
            if stream is not None:
                torch.cuda.current_stream().wait_stream(stream)
            
            batch = next_batch
        
        yield batch
    
    def _to_device(self, batch):
        """Move batch to device."""
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device, non_blocking=True)
        elif isinstance(batch, dict):
            return {k: self._to_device(v) for k, v in batch.items()}
        elif isinstance(batch, list):
            return [self._to_device(item) for item in batch]
        else:
            return batch
    
    def __len__(self):
        """Get number of batches."""
        return len(self.dataloader)
