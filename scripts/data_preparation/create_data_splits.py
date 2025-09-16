#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create Data Splits for AG News
===============================

Creates train/validation/test splits with various strategies.

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from configs.constants import DATA_DIR, AG_NEWS_NUM_CLASSES

logger = setup_logging(__name__)

def create_splits(
    data_path: Path,
    output_dir: Path,
    val_size: float = 0.1,
    test_size: float = 0.1,
    stratified: bool = True,
    seed: int = 42
):
    """
    Create data splits.
    
    Args:
        data_path: Path to full dataset
        output_dir: Output directory
        val_size: Validation split size
        test_size: Test split size
        stratified: Use stratified splitting
        seed: Random seed
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Create splits
    if stratified:
        # First split: train+val vs test
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            stratify=df['label'],
            random_state=seed
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            stratify=train_val['label'],
            random_state=seed
        )
    else:
        # Random splits
        train_val, test = train_test_split(
            df,
            test_size=test_size,
            random_state=seed
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        train, val = train_test_split(
            train_val,
            test_size=val_size_adjusted,
            random_state=seed
        )
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train.to_csv(output_dir / 'train.csv', index=False)
    val.to_csv(output_dir / 'validation.csv', index=False)
    test.to_csv(output_dir / 'test.csv', index=False)
    
    # Log statistics
    logger.info(f"Created splits:")
    logger.info(f"  Train: {len(train)} samples")
    logger.info(f"  Val: {len(val)} samples")
    logger.info(f"  Test: {len(test)} samples")
    
    # Check class distribution
    for split_name, split_df in [('Train', train), ('Val', val), ('Test', test)]:
        dist = split_df['label'].value_counts().sort_index()
        logger.info(f"  {split_name} distribution: {dist.tolist()}")

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Create data splits")
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "raw" / "ag_news" / "full.csv",
        help="Path to full dataset"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "processed",
        help="Output directory"
    )
    
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split size"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test split size"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    create_splits(
        args.data_path,
        args.output_dir,
        args.val_size,
        args.test_size,
        seed=args.seed
    )

if __name__ == "__main__":
    main()
