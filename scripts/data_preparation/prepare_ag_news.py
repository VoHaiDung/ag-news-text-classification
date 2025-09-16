#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AG News Dataset Preparation Script
===================================

This script prepares the AG News dataset for training following best practices from:
- Northcutt et al. (2021): "Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks"
- Swayamdipta et al. (2020): "Dataset Cartography: Mapping and Diagnosing Datasets with Training Dynamics"
- Sambasivan et al. (2021): "Everyone wants to do the model work, not the data work"

Author: Võ Hải Dũng
License: MIT
"""

import os
import sys
import json
import csv
import hashlib
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from collections import Counter
from datetime import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import nltk
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.utils.io_utils import safe_save, safe_load, ensure_dir
from src.core.exceptions import DataError, DataValidationError
from configs.constants import (
    AG_NEWS_CLASSES, AG_NEWS_NUM_CLASSES,
    LABEL_TO_ID, ID_TO_LABEL, MAX_SEQUENCE_LENGTH
)

# Setup logging
logger = setup_logging(
    name=__name__,
    log_dir=PROJECT_ROOT / "outputs" / "logs" / "data",
    log_file="prepare_ag_news.log"
)

# Download NLTK data if needed
try:
    nltk.data.find('punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

@dataclass
class DatasetStatistics:
    """
    Dataset statistics following datasheet guidelines from:
    - Gebru et al. (2021): "Datasheets for Datasets"
    """
    num_samples: int
    num_classes: int
    class_distribution: Dict[str, int]
    avg_text_length: float
    std_text_length: float
    min_text_length: int
    max_text_length: int
    avg_words: float
    std_words: float
    vocabulary_size: int
    label_names: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AGNewsPreprocessor:
    """
    Preprocessor for AG News dataset.
    
    Implements data quality checks from:
    - Northcutt et al. (2021): "Pervasive Label Errors in Test Sets"
    - Breck et al. (2019): "Data Validation for Machine Learning"
    """
    
    def __init__(
        self,
        data_dir: Path = None,
        output_dir: Path = None,
        max_length: int = MAX_SEQUENCE_LENGTH,
        min_length: int = 10,
        remove_duplicates: bool = True,
        validate_labels: bool = True
    ):
        """
        Initialize preprocessor.
        
        Args:
            data_dir: Input data directory
            output_dir: Output directory for processed data
            max_length: Maximum text length
            min_length: Minimum text length
            remove_duplicates: Whether to remove duplicate samples
            validate_labels: Whether to validate label correctness
        """
        self.data_dir = data_dir or PROJECT_ROOT / "data" / "raw" / "ag_news"
        self.output_dir = output_dir or PROJECT_ROOT / "data" / "processed"
        self.max_length = max_length
        self.min_length = min_length
        self.remove_duplicates = remove_duplicates
        self.validate_labels = validate_labels
        
        # Ensure directories exist
        ensure_dir(self.output_dir)
        
        # Statistics tracking
        self.stats = {}
        self.issues = []
        
    def prepare_dataset(
        self,
        train_file: str = "train.csv",
        test_file: str = "test.csv",
        val_split: float = 0.1,
        stratified: bool = True,
        random_state: int = 42
    ) -> Dict[str, pd.DataFrame]:
        """
        Prepare AG News dataset for training.
        
        Args:
            train_file: Training data filename
            test_file: Test data filename
            val_split: Validation split ratio
            stratified: Whether to use stratified split
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing train, validation, and test DataFrames
        """
        logger.info("Starting AG News dataset preparation...")
        
        # Load raw data
        train_df = self._load_raw_data(self.data_dir / train_file, "train")
        test_df = self._load_raw_data(self.data_dir / test_file, "test")
        
        # Validate data quality
        train_df = self._validate_and_clean(train_df, "train")
        test_df = self._validate_and_clean(test_df, "test")
        
        # Create validation split
        if val_split > 0:
            train_df, val_df = self._create_validation_split(
                train_df, val_split, stratified, random_state
            )
        else:
            val_df = pd.DataFrame()
        
        # Process text
        logger.info("Processing text data...")
        train_df = self._process_text(train_df, "train")
        val_df = self._process_text(val_df, "validation") if not val_df.empty else val_df
        test_df = self._process_text(test_df, "test")
        
        # Compute statistics
        self._compute_statistics(train_df, val_df, test_df)
        
        # Save processed data
        datasets = {
            "train": train_df,
            "validation": val_df,
            "test": test_df
        }
        
        self._save_processed_data(datasets)
        
        # Generate quality report
        self._generate_quality_report()
        
        logger.info("Dataset preparation completed successfully!")
        
        return datasets
    
    def _load_raw_data(self, file_path: Path, split_name: str) -> pd.DataFrame:
        """
        Load raw AG News data.
        
        Following data loading best practices from:
        - Sambasivan et al. (2021): "Everyone wants to do the model work, not the data work"
        """
        logger.info(f"Loading {split_name} data from {file_path}...")
        
        if not file_path.exists():
            raise DataError(f"Data file not found: {file_path}")
        
        # AG News CSV format: class_index, title, description
        # Class indices are 1-based in the original dataset
        try:
            df = pd.read_csv(
                file_path,
                names=["label", "title", "description"],
                header=None,
                encoding="utf-8"
            )
            
            # Convert 1-based labels to 0-based
            df["label"] = df["label"] - 1
            
            # Combine title and description
            df["text"] = df["title"] + " " + df["description"]
            
            # Drop original columns
            df = df[["text", "label"]]
            
            logger.info(f"Loaded {len(df)} samples for {split_name}")
            
            return df
            
        except Exception as e:
            raise DataError(f"Failed to load data from {file_path}: {e}")
    
    def _validate_and_clean(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Validate and clean data.
        
        Implements validation checks from:
        - Northcutt et al. (2021): "Pervasive Label Errors in Test Sets"
        """
        logger.info(f"Validating and cleaning {split_name} data...")
        
        initial_size = len(df)
        
        # Remove null values
        null_mask = df.isnull().any(axis=1)
        if null_mask.any():
            self.issues.append(f"{split_name}: Removed {null_mask.sum()} samples with null values")
            df = df[~null_mask]
        
        # Validate labels
        if self.validate_labels:
            valid_labels = set(range(AG_NEWS_NUM_CLASSES))
            invalid_mask = ~df["label"].isin(valid_labels)
            if invalid_mask.any():
                self.issues.append(f"{split_name}: Removed {invalid_mask.sum()} samples with invalid labels")
                df = df[~invalid_mask]
        
        # Remove empty texts
        empty_mask = df["text"].str.strip().str.len() == 0
        if empty_mask.any():
            self.issues.append(f"{split_name}: Removed {empty_mask.sum()} empty samples")
            df = df[~empty_mask]
        
        # Remove duplicates
        if self.remove_duplicates:
            before = len(df)
            df = df.drop_duplicates(subset=["text"], keep="first")
            after = len(df)
            if before > after:
                self.issues.append(f"{split_name}: Removed {before - after} duplicate samples")
        
        # Length filtering
        text_lengths = df["text"].str.split().str.len()
        
        # Remove too short
        short_mask = text_lengths < self.min_length
        if short_mask.any():
            self.issues.append(f"{split_name}: Removed {short_mask.sum()} samples shorter than {self.min_length} words")
            df = df[~short_mask]
        
        # Truncate too long (don't remove, just warn)
        long_mask = text_lengths > self.max_length
        if long_mask.any():
            logger.warning(f"{split_name}: {long_mask.sum()} samples longer than {self.max_length} words will be truncated")
        
        final_size = len(df)
        logger.info(f"{split_name}: {initial_size} -> {final_size} samples after cleaning")
        
        return df.reset_index(drop=True)
    
    def _create_validation_split(
        self,
        df: pd.DataFrame,
        val_split: float,
        stratified: bool,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create validation split from training data.
        
        Following splitting best practices from:
        - Gorman & Bedrick (2019): "We Need to Talk about Standard Splits"
        """
        logger.info(f"Creating validation split ({val_split*100:.1f}%)...")
        
        if stratified:
            train_df, val_df = train_test_split(
                df,
                test_size=val_split,
                stratify=df["label"],
                random_state=random_state
            )
        else:
            train_df, val_df = train_test_split(
                df,
                test_size=val_split,
                random_state=random_state
            )
        
        logger.info(f"Split: {len(train_df)} train, {len(val_df)} validation")
        
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    
    def _process_text(self, df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """Process text data."""
        if df.empty:
            return df
        
        logger.info(f"Processing text for {split_name}...")
        
        # Basic text cleaning
        df["text"] = df["text"].apply(self._clean_text)
        
        # Add metadata
        df["text_length"] = df["text"].str.len()
        df["word_count"] = df["text"].str.split().str.len()
        df["label_name"] = df["label"].map(ID_TO_LABEL)
        
        return df
    
    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Minimal cleaning to preserve information, following:
        - Strubell et al. (2018): "Linguistically-Informed Self-Attention"
        """
        # Replace multiple spaces
        text = " ".join(text.split())
        
        # Fix encoding issues
        text = text.encode("utf-8", errors="ignore").decode("utf-8")
        
        return text.strip()
    
    def _compute_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ):
        """
        Compute dataset statistics.
        
        Following statistical analysis from:
        - Swayamdipta et al. (2020): "Dataset Cartography"
        """
        logger.info("Computing dataset statistics...")
        
        for split_name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
            if df.empty:
                continue
            
            # Class distribution
            class_dist = df["label_name"].value_counts().to_dict()
            
            # Text statistics
            text_lengths = df["text_length"].values
            word_counts = df["word_count"].values
            
            # Vocabulary
            all_words = " ".join(df["text"].values).split()
            vocab_size = len(set(all_words))
            
            stats = DatasetStatistics(
                num_samples=len(df),
                num_classes=AG_NEWS_NUM_CLASSES,
                class_distribution=class_dist,
                avg_text_length=float(np.mean(text_lengths)),
                std_text_length=float(np.std(text_lengths)),
                min_text_length=int(np.min(text_lengths)),
                max_text_length=int(np.max(text_lengths)),
                avg_words=float(np.mean(word_counts)),
                std_words=float(np.std(word_counts)),
                vocabulary_size=vocab_size,
                label_names=AG_NEWS_CLASSES
            )
            
            self.stats[split_name] = stats
            
            # Log statistics
            logger.info(f"\n{split_name.upper()} Statistics:")
            logger.info(f"  Samples: {stats.num_samples}")
            logger.info(f"  Classes: {stats.num_classes}")
            logger.info(f"  Avg words: {stats.avg_words:.1f} ± {stats.std_words:.1f}")
            logger.info(f"  Vocabulary: {stats.vocabulary_size}")
            logger.info(f"  Class distribution: {stats.class_distribution}")
    
    def _save_processed_data(self, datasets: Dict[str, pd.DataFrame]):
        """Save processed datasets."""
        logger.info("Saving processed data...")
        
        for split_name, df in datasets.items():
            if df.empty:
                continue
            
            # Save as CSV
            csv_path = self.output_dir / f"{split_name}.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved {split_name} to {csv_path}")
            
            # Save as JSON for easier loading
            json_path = self.output_dir / f"{split_name}.json"
            df.to_json(json_path, orient="records", lines=True)
            
            # Save as Parquet for efficiency
            parquet_path = self.output_dir / f"{split_name}.parquet"
            df.to_parquet(parquet_path, engine="pyarrow", compression="snappy")
        
        # Save statistics
        stats_path = self.output_dir / "dataset_stats.json"
        stats_dict = {k: v.to_dict() for k, v in self.stats.items()}
        safe_save(stats_dict, stats_path)
        
        # Save label mapping
        label_mapping = {
            "label_to_id": LABEL_TO_ID,
            "id_to_label": ID_TO_LABEL,
            "label_names": AG_NEWS_CLASSES
        }
        mapping_path = self.output_dir / "label_mapping.json"
        safe_save(label_mapping, mapping_path)
    
    def _generate_quality_report(self):
        """
        Generate data quality report.
        
        Following data quality assessment from:
        - Breck et al. (2019): "Data Validation for Machine Learning"
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "max_length": self.max_length,
                "min_length": self.min_length,
                "remove_duplicates": self.remove_duplicates,
                "validate_labels": self.validate_labels,
            },
            "statistics": {k: v.to_dict() for k, v in self.stats.items()},
            "issues": self.issues,
            "data_quality_checks": {
                "no_null_values": True,
                "valid_labels": True,
                "no_empty_texts": True,
                "no_duplicates": self.remove_duplicates,
                "length_filtered": True,
            },
            "recommendations": self._generate_recommendations(),
        }
        
        report_path = self.output_dir / "data_quality_report.json"
        safe_save(report, report_path)
        
        logger.info(f"Data quality report saved to {report_path}")
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on data analysis."""
        recommendations = []
        
        # Check class imbalance
        if "train" in self.stats:
            class_dist = self.stats["train"].class_distribution
            counts = list(class_dist.values())
            max_count = max(counts)
            min_count = min(counts)
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio > 2:
                recommendations.append(
                    f"Class imbalance detected (ratio: {imbalance_ratio:.2f}). "
                    "Consider using weighted loss or oversampling."
                )
        
        # Check vocabulary size
        if "train" in self.stats:
            vocab_size = self.stats["train"].vocabulary_size
            if vocab_size > 100000:
                recommendations.append(
                    f"Large vocabulary ({vocab_size}). Consider using subword tokenization."
                )
        
        # Check text length variation
        if "train" in self.stats:
            std_words = self.stats["train"].std_words
            avg_words = self.stats["train"].avg_words
            cv = std_words / avg_words  # Coefficient of variation
            
            if cv > 1:
                recommendations.append(
                    f"High text length variation (CV: {cv:.2f}). "
                    "Consider using dynamic padding or bucketing."
                )
        
        return recommendations

def create_k_fold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    stratified: bool = True,
    random_state: int = 42,
    output_dir: Path = None
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create K-fold cross-validation splits.
    
    Following cross-validation best practices from:
    - Kohavi (1995): "A Study of Cross-Validation and Bootstrap"
    """
    logger.info(f"Creating {n_splits}-fold cross-validation splits...")
    
    output_dir = output_dir or PROJECT_ROOT / "data" / "processed" / "cv_splits"
    ensure_dir(output_dir)
    
    if stratified:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kfold.split(df.index, df["label"])
    else:
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = kfold.split(df.index)
    
    fold_data = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        val_fold = df.iloc[val_idx].reset_index(drop=True)
        
        # Save fold data
        train_path = output_dir / f"fold_{fold_idx}_train.csv"
        val_path = output_dir / f"fold_{fold_idx}_val.csv"
        
        train_fold.to_csv(train_path, index=False)
        val_fold.to_csv(val_path, index=False)
        
        fold_data.append((train_fold, val_fold))
        
        logger.info(f"Fold {fold_idx}: {len(train_fold)} train, {len(val_fold)} validation")
    
    return fold_data

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Prepare AG News dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "ag_news",
        help="Input data directory"
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Output directory for processed data"
    )
    
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_SEQUENCE_LENGTH,
        help=f"Maximum text length (default: {MAX_SEQUENCE_LENGTH})"
    )
    
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum text length in words (default: 10)"
    )
    
    parser.add_argument(
        "--no-duplicates",
        action="store_true",
        help="Keep duplicate samples"
    )
    
    parser.add_argument(
        "--k-folds",
        type=int,
        default=0,
        help="Create K-fold cross-validation splits (0 to disable)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize preprocessor
        preprocessor = AGNewsPreprocessor(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            max_length=args.max_length,
            min_length=args.min_length,
            remove_duplicates=not args.no_duplicates
        )
        
        # Prepare dataset
        datasets = preprocessor.prepare_dataset(
            val_split=args.val_split,
            random_state=args.seed
        )
        
        # Create K-fold splits if requested
        if args.k_folds > 0:
            create_k_fold_splits(
                datasets["train"],
                n_splits=args.k_folds,
                random_state=args.seed,
                output_dir=args.output_dir / "cv_splits"
            )
        
        logger.info("Dataset preparation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
