#!/usr/bin/env python
"""Download AG News dataset"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets import load_dataset
from src.utils.logging_config import setup_logger

logger = setup_logger(__name__)

def download_ag_news():
    """Download and save AG News dataset"""
    # Create data directory
    data_dir = Path("data/raw/ag_news")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading AG News dataset...")
    dataset = load_dataset("ag_news")
    
    # Save to disk
    dataset.save_to_disk(str(data_dir))
    
    logger.info(f"Dataset saved to {data_dir}")
    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Test samples: {len(dataset['test'])}")
    
    # Print sample
    logger.info("\nSample data:")
    sample = dataset['train'][0]
    logger.info(f"Text: {sample['text'][:100]}...")
    logger.info(f"Label: {sample['label']}")
    
    return dataset

if __name__ == "__main__":
    download_ag_news()
