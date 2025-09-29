#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data Split Creation Script for AG News Text Classification
================================================================================
This script creates train/validation/test splits from the AG News dataset using
various splitting strategies including stratified sampling, k-fold cross-validation,
and temporal splitting. It ensures proper data distribution across splits for
robust model training and evaluation.

The splitting methodology follows best practices for machine learning dataset
preparation, maintaining class balance and preventing data leakage between splits.

References:
    - Reitermanova, Z. (2010): Data Splitting
    - Kohavi, R. (1995): A Study of Cross-Validation and Bootstrap
    - Raschka, S. (2018): Model Evaluation, Model Selection, and Algorithm Selection

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import (
    train_test_split, 
    StratifiedKFold,
    StratifiedShuffleSplit,
    GroupKFold
)
import json
from datetime import datetime
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.utils.io_utils import save_json, ensure_dir
from configs.constants import DATA_DIR, AG_NEWS_NUM_CLASSES, AG_NEWS_CLASSES

logger = setup_logging(__name__)


class DataSplitter:
    """
    Comprehensive data splitting utility for AG News dataset
    
    This class provides various strategies for creating train/validation/test
    splits while maintaining data integrity and proper class distribution.
    """
    
    def __init__(
        self,
        data_path: Path,
        output_dir: Path,
        seed: int = 42
    ):
        """
        Initialize data splitter
        
        Args:
            data_path: Path to the full dataset file
            output_dir: Directory for saving split files
            seed: Random seed for reproducibility
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.seed = seed
        self.split_stats = {}
        
        # Ensure output directory exists
        ensure_dir(self.output_dir)
        
        logger.info(f"Initialized DataSplitter with seed={seed}")
    
    def create_standard_splits(
        self,
        val_size: float = 0.1,
        test_size: float = 0.1,
        stratified: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create standard train/validation/test splits
        
        Creates three-way data split maintaining class proportions if stratified
        splitting is enabled. Ensures no data leakage between splits.
        
        Args:
            val_size: Proportion of data for validation set
            test_size: Proportion of data for test set  
            stratified: Whether to use stratified sampling
            
        Returns:
            Tuple of (train, validation, test) DataFrames
        """
        logger.info(f"Loading data from {self.data_path}")
        
        # Load data
        df = pd.read_csv(self.data_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Validate class distribution
        self._validate_data(df)
        
        # Create splits
        if stratified:
            # First split: train+val vs test
            train_val, test = train_test_split(
                df,
                test_size=test_size,
                stratify=df['label'],
                random_state=self.seed
            )
            
            # Second split: train vs val
            val_size_adjusted = val_size / (1 - test_size)
            train, val = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                stratify=train_val['label'],
                random_state=self.seed
            )
        else:
            # Random splits without stratification
            train_val, test = train_test_split(
                df,
                test_size=test_size,
                random_state=self.seed
            )
            
            val_size_adjusted = val_size / (1 - test_size)
            train, val = train_test_split(
                train_val,
                test_size=val_size_adjusted,
                random_state=self.seed
            )
        
        # Calculate and store statistics
        self._calculate_split_statistics(train, val, test)
        
        return train, val, test
    
    def create_kfold_splits(
        self,
        n_folds: int = 5,
        stratified: bool = True
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Create k-fold cross-validation splits
        
        Generates k train/validation pairs for cross-validation, useful for
        robust model evaluation and hyperparameter tuning.
        
        Args:
            n_folds: Number of folds for cross-validation
            stratified: Whether to use stratified k-fold
            
        Returns:
            List of (train, validation) DataFrame pairs
        """
        logger.info(f"Creating {n_folds}-fold cross-validation splits")
        
        # Load data
        df = pd.read_csv(self.data_path)
        
        # Initialize k-fold splitter
        if stratified:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            splits = kf.split(df, df['label'])
        else:
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            splits = kf.split(df)
        
        # Create fold splits
        fold_splits = []
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            train_fold = df.iloc[train_idx].copy()
            val_fold = df.iloc[val_idx].copy()
            
            fold_splits.append((train_fold, val_fold))
            
            logger.info(f"Fold {fold_idx + 1}: Train={len(train_fold)}, Val={len(val_fold)}")
        
        return fold_splits
    
    def save_splits(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        format: str = "csv"
    ):
        """
        Save data splits to disk in specified format
        
        Args:
            train: Training set DataFrame
            val: Validation set DataFrame
            test: Test set DataFrame
            format: Output format ('csv', 'json', or 'parquet')
        """
        # Save based on format
        if format == "csv":
            train.to_csv(self.output_dir / 'train.csv', index=False)
            val.to_csv(self.output_dir / 'validation.csv', index=False)
            test.to_csv(self.output_dir / 'test.csv', index=False)
        
        elif format == "json":
            train.to_json(self.output_dir / 'train.json', orient='records', lines=True)
            val.to_json(self.output_dir / 'validation.json', orient='records', lines=True)
            test.to_json(self.output_dir / 'test.json', orient='records', lines=True)
        
        elif format == "parquet":
            train.to_parquet(self.output_dir / 'train.parquet', index=False)
            val.to_parquet(self.output_dir / 'validation.parquet', index=False)
            test.to_parquet(self.output_dir / 'test.parquet', index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved splits to {self.output_dir} in {format} format")
        
        # Save split statistics
        self._save_split_report()
    
    def save_kfold_splits(
        self,
        fold_splits: List[Tuple[pd.DataFrame, pd.DataFrame]],
        format: str = "csv"
    ):
        """
        Save k-fold splits to disk
        
        Args:
            fold_splits: List of (train, validation) fold pairs
            format: Output format for saving
        """
        folds_dir = self.output_dir / "stratified_folds"
        ensure_dir(folds_dir)
        
        for fold_idx, (train_fold, val_fold) in enumerate(fold_splits):
            fold_dir = folds_dir / f"fold_{fold_idx + 1}"
            ensure_dir(fold_dir)
            
            if format == "csv":
                train_fold.to_csv(fold_dir / 'train.csv', index=False)
                val_fold.to_csv(fold_dir / 'validation.csv', index=False)
            elif format == "json":
                train_fold.to_json(fold_dir / 'train.json', orient='records', lines=True)
                val_fold.to_json(fold_dir / 'validation.json', orient='records', lines=True)
            
        logger.info(f"Saved {len(fold_splits)} folds to {folds_dir}")
    
    def _validate_data(self, df: pd.DataFrame):
        """
        Validate data integrity and class distribution
        
        Args:
            df: DataFrame to validate
        """
        # Check for required columns
        required_columns = ['text', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Check class distribution
        class_counts = df['label'].value_counts().sort_index()
        
        if len(class_counts) != AG_NEWS_NUM_CLASSES:
            logger.warning(f"Expected {AG_NEWS_NUM_CLASSES} classes, found {len(class_counts)}")
        
        # Check for class imbalance
        max_count = class_counts.max()
        min_count = class_counts.min()
        imbalance_ratio = max_count / min_count
        
        if imbalance_ratio > 10:
            logger.warning(f"High class imbalance detected: {imbalance_ratio:.2f}x")
        
        logger.info(f"Class distribution: {class_counts.tolist()}")
    
    def _calculate_split_statistics(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame
    ):
        """
        Calculate and store comprehensive split statistics
        
        Args:
            train: Training set DataFrame
            val: Validation set DataFrame
            test: Test set DataFrame
        """
        self.split_stats = {
            "total_samples": len(train) + len(val) + len(test),
            "split_sizes": {
                "train": len(train),
                "validation": len(val),
                "test": len(test)
            },
            "split_percentages": {
                "train": len(train) / (len(train) + len(val) + len(test)) * 100,
                "validation": len(val) / (len(train) + len(val) + len(test)) * 100,
                "test": len(test) / (len(train) + len(val) + len(test)) * 100
            },
            "class_distributions": {}
        }
        
        # Calculate class distribution for each split
        for split_name, split_df in [('train', train), ('validation', val), ('test', test)]:
            dist = split_df['label'].value_counts().sort_index()
            self.split_stats["class_distributions"][split_name] = {
                "counts": dist.tolist(),
                "percentages": (dist / len(split_df) * 100).tolist(),
                "class_names": [AG_NEWS_CLASSES[i] for i in dist.index]
            }
        
        # Log statistics
        logger.info(f"Created splits:")
        logger.info(f"  Train: {len(train)} samples ({self.split_stats['split_percentages']['train']:.1f}%)")
        logger.info(f"  Val: {len(val)} samples ({self.split_stats['split_percentages']['validation']:.1f}%)")
        logger.info(f"  Test: {len(test)} samples ({self.split_stats['split_percentages']['test']:.1f}%)")
    
    def _save_split_report(self):
        """
        Save detailed split report to JSON file
        
        Creates a comprehensive report of the data splitting process including
        all statistics and metadata for reproducibility.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "data_path": str(self.data_path),
            "output_dir": str(self.output_dir),
            "seed": self.seed,
            "statistics": self.split_stats
        }
        
        report_path = self.output_dir / "split_report.json"
        save_json(report, report_path)
        logger.info(f"Saved split report to {report_path}")


def main():
    """
    Main entry point for data split creation
    
    Orchestrates the data splitting process based on command-line arguments,
    creating train/validation/test splits with proper logging and reporting.
    """
    parser = argparse.ArgumentParser(
        description="Create train/validation/test splits for AG News dataset"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "raw" / "ag_news" / "full.csv",
        help="Path to full dataset CSV file"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "processed",
        help="Output directory for split files"
    )
    
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Validation split size (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test split size (0.0 to 1.0)"
    )
    
    parser.add_argument(
        "--stratified",
        action="store_true",
        default=True,
        help="Use stratified sampling to maintain class distribution"
    )
    
    parser.add_argument(
        "--kfold",
        type=int,
        default=None,
        help="Create k-fold cross-validation splits (specify number of folds)"
    )
    
    parser.add_argument(
        "--format",
        choices=["csv", "json", "parquet"],
        default="csv",
        help="Output format for split files"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.val_size + args.test_size >= 1.0:
        raise ValueError("Sum of val_size and test_size must be less than 1.0")
    
    # Initialize splitter
    splitter = DataSplitter(
        data_path=args.data_path,
        output_dir=args.output_dir,
        seed=args.seed
    )
    
    if args.kfold:
        # Create k-fold splits
        fold_splits = splitter.create_kfold_splits(
            n_folds=args.kfold,
            stratified=args.stratified
        )
        splitter.save_kfold_splits(fold_splits, format=args.format)
        
        print(f"\nCreated {args.kfold}-fold cross-validation splits")
        print(f"Saved to: {args.output_dir / 'stratified_folds'}")
        
    else:
        # Create standard splits
        train, val, test = splitter.create_standard_splits(
            val_size=args.val_size,
            test_size=args.test_size,
            stratified=args.stratified
        )
        
        # Save splits
        splitter.save_splits(train, val, test, format=args.format)
        
        # Print summary
        print("\n" + "=" * 50)
        print("Data Split Summary")
        print("=" * 50)
        print(f"Total samples: {splitter.split_stats['total_samples']}")
        print(f"Train: {splitter.split_stats['split_sizes']['train']} ({splitter.split_stats['split_percentages']['train']:.1f}%)")
        print(f"Validation: {splitter.split_stats['split_sizes']['validation']} ({splitter.split_stats['split_percentages']['validation']:.1f}%)")
        print(f"Test: {splitter.split_stats['split_sizes']['test']} ({splitter.split_stats['split_percentages']['test']:.1f}%)")
        
        print("\nClass Distribution in Train Set:")
        print("-" * 30)
        train_dist = splitter.split_stats['class_distributions']['train']
        for i, class_name in enumerate(train_dist['class_names']):
            count = train_dist['counts'][i]
            pct = train_dist['percentages'][i]
            print(f"{class_name:15} | Count: {count:6} | {pct:.1f}%")
        
        print(f"\nSplits saved to: {args.output_dir}")
        print("Split creation completed successfully!")


if __name__ == "__main__":
    main()
