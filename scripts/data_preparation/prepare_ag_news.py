#!/usr/bin/env python
"""Prepare AG News data pipeline"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from datasets import load_from_disk
from sklearn.model_selection import train_test_split
from src.data.datasets.ag_news import AGNewsDataset
from src.data.preprocessing.text_cleaner import TextCleaner
from src.data.loaders.dataloader import create_dataloaders
from src.utils.logging_config import setup_logger
import torch

logger = setup_logger(__name__)

def prepare_data_pipeline():
    """Complete data preparation pipeline"""
    
    # Load raw data
    logger.info("Loading raw data...")
    dataset = load_from_disk("data/raw/ag_news")
    
    # Clean text
    logger.info("Cleaning text...")
    cleaner = TextCleaner()
    
    # Process train data
    train_data = []
    for item in dataset['train']:
        train_data.append({
            'text': cleaner.clean_text(item['text']),
            'label': item['label']
        })
    
    # Create train/val split
    logger.info("Creating train/val split...")
    train_data, val_data = train_test_split(
        train_data, 
        test_size=0.1, 
        stratify=[item['label'] for item in train_data],
        random_state=42
    )
    
    # Process test data
    test_data = []
    for item in dataset['test']:
        test_data.append({
            'text': cleaner.clean_text(item['text']),
            'label': item['label']
        })
    
    # Create PyTorch datasets
    logger.info("Creating PyTorch datasets...")
    train_dataset = AGNewsDataset(train_data)
    val_dataset = AGNewsDataset(val_data)
    test_dataset = AGNewsDataset(test_data)
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    dataloaders = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=32
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    
    return dataloaders

if __name__ == "__main__":
    dataloaders = prepare_data_pipeline()
