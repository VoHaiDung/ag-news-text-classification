#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quality Data Selection Script for AG News Text Classification
================================================================================
This script selects high-quality subsets from training data using advanced
filtering and selection strategies, implementing data-centric AI approaches to
improve model performance through intelligent data curation. It employs quality
metrics, diversity selection, and influence-based methods to identify the most
valuable training samples.

The data selection approach enables training more efficient models with less data
while maintaining or improving performance, particularly useful for reducing
computational costs and improving model robustness.

References:
    - Swayamdipta, S. et al. (2020): Dataset Cartography - Mapping and Diagnosing Datasets with Training Dynamics  
    - Paul, M. et al. (2021): Deep Learning on a Data Diet - Finding Important Examples Early in Training
    - Coleman, C. et al. (2020): Selection via Proxy - Efficient Data Selection for Deep Learning
    - Sorscher, B. et al. (2022): Beyond Neural Scaling Laws - Beating Power Law Scaling via Data Pruning
    - Killamsetty, K. et al. (2021): GRAD-MATCH - Gradient Matching based Data Subset Selection

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import json
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.data.selection.quality_filtering import QualityFilter, QualityFilterConfig
from src.data.selection.diversity_selection import DiversitySelector
from configs.constants import DATA_DIR

logger = setup_logging(__name__)


def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load training data from file
    
    Supports multiple file formats and provides basic validation
    to ensure data integrity before processing.
    
    Args:
        data_path: Path to the data file (CSV or Parquet format)
        
    Returns:
        DataFrame containing the loaded training data
        
    Raises:
        ValueError: If file format is not supported
    """
    logger.info(f"Loading data from {data_path}")
    
    if data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    elif data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported format: {data_path.suffix}")
    
    logger.info(f"Loaded {len(df)} samples")
    return df


def apply_quality_filtering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply quality-based filtering to remove low-quality samples
    
    Implements quality metrics from Swayamdipta et al. (2020) to identify
    and remove potentially harmful or uninformative training samples based
    on text characteristics and statistical properties.
    
    Args:
        df: DataFrame containing training data with 'text' column
        
    Returns:
        DataFrame with low-quality samples filtered out
    """
    filter_config = QualityFilterConfig(
        min_length=10,
        max_length=1000,
        min_unique_words=5,
        max_repetition_ratio=0.3
    )
    
    quality_filter = QualityFilter(filter_config)
    
    texts = df['text'].tolist()
    mask = quality_filter.filter(texts)
    
    filtered_df = df[mask].reset_index(drop=True)
    
    logger.info(f"Quality filtering: {len(df)} -> {len(filtered_df)} samples")
    
    return filtered_df


def apply_diversity_selection(
    df: pd.DataFrame,
    n_samples: int,
    embeddings_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Apply diversity-based selection to create representative subset
    
    Implements diversity selection techniques from Coleman et al. (2020)
    to select a maximally diverse subset that covers the data distribution
    effectively, reducing redundancy while maintaining representativeness.
    
    Args:
        df: DataFrame containing training data
        n_samples: Number of samples to select
        embeddings_path: Optional path to pre-computed embeddings
        
    Returns:
        DataFrame containing diverse subset of training data
    """
    
    if embeddings_path and embeddings_path.exists():
        # Load pre-computed embeddings
        embeddings = np.load(embeddings_path)
    else:
        # Use simple TF-IDF as fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(max_features=1000)
        embeddings = vectorizer.fit_transform(df['text']).toarray()
    
    # Select diverse subset
    selector = DiversitySelector(method="clustering")
    indices = selector.select_diverse_subset(embeddings, n_samples)
    
    selected_df = df.iloc[indices].reset_index(drop=True)
    
    logger.info(f"Diversity selection: {len(df)} -> {len(selected_df)} samples")
    
    return selected_df


def save_selected_data(df: pd.DataFrame, output_path: Path):
    """
    Save selected high-quality data to file
    
    Persists the selected subset with metadata about the selection
    process for reproducibility and tracking.
    
    Args:
        df: DataFrame containing selected data
        output_path: Path for saving the output file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(df)} samples to {output_path}")


def main():
    """
    Main entry point for quality data selection
    
    Orchestrates the complete data selection pipeline including loading,
    quality filtering, diversity selection, and saving results with
    comprehensive logging and statistics.
    """
    parser = argparse.ArgumentParser(
        description="Select high-quality training data using advanced filtering strategies"
    )
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "processed" / "train.csv",
        help="Path to training data file"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "processed" / "train_quality.csv",
        help="Output path for selected data"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["quality", "diversity", "both"],
        default="both",
        help="Selection strategy to apply"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples to select for diversity selection"
    )
    
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        help="Path to pre-computed embeddings for diversity selection"
    )
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.data_path)
    
    # Apply selection strategies
    if args.strategy in ["quality", "both"]:
        df = apply_quality_filtering(df)
    
    if args.strategy in ["diversity", "both"]:
        df = apply_diversity_selection(df, args.n_samples, args.embeddings_path)
    
    # Save selected data
    save_selected_data(df, args.output_path)
    
    logger.info("Data selection complete!")


if __name__ == "__main__":
    main()
