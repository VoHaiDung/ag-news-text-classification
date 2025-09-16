#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Select High-Quality Training Data
==================================

Selects high-quality subset of training data using various strategies.

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.data.selection.quality_filtering import QualityFilter, QualityFilterConfig
from src.data.selection.diversity_selection import DiversitySelector
from configs.constants import DATA_DIR

logger = setup_logging(__name__)

def load_data(data_path: Path) -> pd.DataFrame:
    """Load training data."""
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
    """Apply quality filtering."""
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
    """Apply diversity-based selection."""
    
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
    """Save selected data."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == ".csv":
        df.to_csv(output_path, index=False)
    elif output_path.suffix == ".parquet":
        df.to_parquet(output_path, index=False)
    
    logger.info(f"Saved {len(df)} samples to {output_path}")

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Select quality training data")
    
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DATA_DIR / "processed" / "train.csv",
        help="Path to training data"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "processed" / "train_quality.csv",
        help="Output path"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["quality", "diversity", "both"],
        default="both",
        help="Selection strategy"
    )
    
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10000,
        help="Number of samples to select"
    )
    
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        help="Path to pre-computed embeddings"
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
