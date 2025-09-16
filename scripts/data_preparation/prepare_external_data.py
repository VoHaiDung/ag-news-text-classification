#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prepare External Data for Domain Adaptation
============================================

Prepares external news corpora for domain-adaptive pretraining.

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
from typing import List
import json
from tqdm import tqdm

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging_config import setup_logging
from src.data.preprocessing.text_cleaner import TextCleaner, CleaningConfig
from src.data.selection.quality_filtering import QualityFilter, QualityFilterConfig
from configs.constants import DATA_DIR

logger = setup_logging(__name__)

def load_news_corpus(corpus_path: Path, max_samples: int = None) -> List[str]:
    """Load external news corpus."""
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
    """Filter texts for news domain."""
    # Quality filtering
    filter_config = QualityFilterConfig(
        min_length=20,
        max_length=2000,
        min_unique_words=10,
        max_repetition_ratio=0.3
    )
    
    quality_filter = QualityFilter(filter_config)
    mask = quality_filter.filter(texts)
    
    filtered = [text for text, keep in zip(texts, mask) if keep]
    
    logger.info(f"Filtered: {len(texts)} -> {len(filtered)} texts")
    
    return filtered

def clean_texts(texts: List[str]) -> List[str]:
    """Clean texts for pretraining."""
    cleaning_config = CleaningConfig(
        lowercase=False,
        remove_urls=True,
        remove_emails=True,
        normalize_whitespace=True
    )
    
    cleaner = TextCleaner(cleaning_config)
    
    cleaned = []
    for text in tqdm(texts, desc="Cleaning texts"):
        cleaned.append(cleaner.clean(text))
    
    return cleaned

def save_processed_corpus(
    texts: List[str],
    output_path: Path,
    format: str = "jsonl"
):
    """Save processed corpus."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "jsonl":
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, text in enumerate(texts):
                data = {
                    'id': i,
                    'text': text,
                    'source': 'external_news'
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

def main():
    """Main execution."""
    parser = argparse.ArgumentParser(description="Prepare external news data")
    
    parser.add_argument(
        "--corpus-path",
        type=Path,
        required=True,
        help="Path to external corpus"
    )
    
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "external" / "news_corpus.jsonl",
        help="Output path"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Maximum samples to process"
    )
    
    parser.add_argument(
        "--format",
        choices=["jsonl", "txt", "csv"],
        default="jsonl",
        help="Output format"
    )
    
    args = parser.parse_args()
    
    # Load corpus
    texts = load_news_corpus(args.corpus_path, args.max_samples)
    
    # Clean texts
    texts = clean_texts(texts)
    
    # Filter texts
    texts = filter_news_texts(texts)
    
    # Save processed corpus
    save_processed_corpus(texts, args.output_path, args.format)
    
    logger.info("External data preparation complete!")

if __name__ == "__main__":
    main()
