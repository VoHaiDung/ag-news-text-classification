#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
External News Corpus Preparation Script for Domain Adaptation
================================================================================
This script prepares external news corpora for domain-adaptive pretraining and
data augmentation, implementing robust processing pipelines to leverage large-scale
unlabeled news data. It enables domain adaptation techniques that improve model
performance on news classification tasks through continued pretraining on in-domain data.

The preparation pipeline handles diverse corpus formats, applies quality filtering,
and optimizes data for efficient pretraining while maintaining domain relevance
and text quality standards.

References:
    - Gururangan, S. et al. (2020): Don't Stop Pretraining - Adapt Language Models to Domains and Tasks
    - Lee, J. et al. (2020): BioBERT - A Pre-trained Biomedical Language Representation Model
    - Alsentzer, E. et al. (2019): Publicly Available Clinical BERT Embeddings
    - Beltagy, I. et al. (2019): SciBERT - A Pretrained Language Model for Scientific Text
    - Chalkidis, I. et al. (2020): LEGAL-BERT - The Muppets straight out of Law School

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
from tqdm import tqdm
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.data.preprocessing.text_cleaner import TextCleaner, CleaningConfig
from src.data.selection.quality_filtering import QualityFilter, QualityFilterConfig
from configs.constants import DATA_DIR

logger = setup_logging(__name__)


def load_news_corpus(corpus_path: Path, max_samples: Optional[int] = None) -> List[str]:
    """
    Load external news corpus from various formats
    
    Implements flexible corpus loading to handle diverse data sources including
    plain text, JSON Lines, and CSV formats commonly used for large-scale text
    corpora. Follows data loading practices from Gururangan et al. (2020).
    
    Args:
        corpus_path: Path to the corpus file
        max_samples: Optional maximum number of samples to load
        
    Returns:
        List of text strings from the corpus
        
    Raises:
        ValueError: If corpus format is not supported
    """
    logger.info(f"Loading news corpus from {corpus_path}")
    
    texts = []
    
    # Handle different formats
    if corpus_path.suffix == ".txt":
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading texts"):
                if max_samples and len(texts) >= max_samples:
                    break
                texts.append(line.strip())
                
    elif corpus_path.suffix == ".jsonl":
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading texts"):
                if max_samples and len(texts) >= max_samples:
                    break
                data = json.loads(line)
                texts.append(data.get('text', ''))
                
    elif corpus_path.suffix == ".csv":
        df = pd.read_csv(corpus_path)
        text_col = 'text' if 'text' in df.columns else df.columns[0]
        texts = df[text_col].tolist()[:max_samples]
    
    logger.info(f"Loaded {len(texts)} texts")
    return texts


def filter_news_texts(texts: List[str]) -> List[str]:
    """
    Apply domain-specific filtering for news texts
    
    Implements quality filtering tailored for news domain following principles
    from domain adaptation literature. Ensures texts meet minimum quality standards
    while preserving domain-relevant characteristics.
    
    Args:
        texts: List of raw text strings
        
    Returns:
        List of filtered text strings meeting quality criteria
    """
    # Quality filtering configuration for news domain
    filter_config = QualityFilterConfig(
        min_length=20,        # News articles should have substantive content
        max_length=2000,      # Filter out extremely long documents
        min_unique_words=10,  # Ensure lexical diversity
        max_repetition_ratio=0.3  # Remove repetitive content
    )
    
    quality_filter = QualityFilter(filter_config)
    mask = quality_filter.filter(texts)
    
    filtered = [text for text, keep in zip(texts, mask) if keep]
    
    logger.info(f"Filtered: {len(texts)} -> {len(filtered)} texts")
    
    return filtered


def clean_texts(texts: List[str]) -> List[str]:
    """
    Clean texts while preserving information for pretraining
    
    Applies minimal cleaning to preserve linguistic features important for
    language model pretraining, following practices from BERT and RoBERTa
    pretraining that showed minimal preprocessing works best.
    
    Args:
        texts: List of texts to clean
        
    Returns:
        List of cleaned texts
    """
    cleaning_config = CleaningConfig(
        lowercase=False,           # Preserve case information
        remove_urls=True,          # Remove URLs as they're not useful for pretraining
        remove_emails=True,        # Remove email addresses for privacy
        normalize_whitespace=True  # Normalize spacing
    )
    
    cleaner = TextCleaner(cleaning_config)
    
    cleaned = []
    for text in tqdm(texts, desc="Cleaning texts"):
        cleaned.append(cleaner.clean(text))
    
    return cleaned


def save_processed_corpus(
    texts: List[str],
    output_path: Path,
    format: str = "jsonl",
    metadata: Optional[Dict[str, Any]] = None
):
    """
    Save processed corpus with metadata for reproducibility
    
    Persists the processed corpus in efficient formats suitable for pretraining
    pipelines, including metadata for tracking data provenance and processing
    parameters following best practices from Gururangan et al. (2020).
    
    Args:
        texts: List of processed text strings
        output_path: Path for saving the output file
        format: Output format (jsonl, txt, or csv)
        metadata: Optional metadata about the corpus
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                data = {
                    'id': i,
                    'text': text,
                    'source': 'external_news',
                    'processed_date': datetime.now().isoformat()
                }
                f.write(json.dumps(data) + '\n')
                
    elif format == "txt":
        with open(output_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n')
                
    elif format == "csv":
        df = pd.DataFrame({'text': texts})
        df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(texts)} texts to {output_path}")
    
    # Save metadata if provided
    if metadata:
        metadata_path = output_path.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


def main():
    """
    Main entry point for external data preparation
    
    Orchestrates the complete external corpus preparation pipeline including
    loading, cleaning, filtering, and saving with comprehensive logging and
    error handling for robust large-scale data processing.
    """
    parser = argparse.ArgumentParser(
        description="Prepare external news corpus for domain-adaptive pretraining"
    )
    
    parser.add_argument(
        "--corpus-path",
        type=Path,
        required=True,
        help="Path to external news corpus file"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "external" / "news_corpus.jsonl",
        help="Output path for processed corpus"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Maximum number of samples to process"
    )
    
    parser.add_argument(
        "--format",
        choices=["jsonl", "txt", "csv"],
        default="jsonl",
        help="Output format for processed corpus"
    )
    
    args = parser.parse_args()
    
    # Load corpus
    texts = load_news_corpus(args.corpus_path, args.max_samples)
    
    # Clean texts
    texts = clean_texts(texts)
    
    # Filter texts
    texts = filter_news_texts(texts)
    
    # Prepare metadata
    metadata = {
        "source_file": str(args.corpus_path),
        "processing_date": datetime.now().isoformat(),
        "original_count": args.max_samples,
        "final_count": len(texts),
        "format": args.format
    }
    
    # Save processed corpus
    save_processed_corpus(texts, args.output_path, args.format, metadata)
    
    logger.info("External data preparation complete!")


if __name__ == "__main__":
    main()
