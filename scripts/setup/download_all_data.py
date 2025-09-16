#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Download All Required Data for AG News Classification
======================================================

This script downloads and prepares all necessary datasets including:
- AG News dataset (Zhang et al., 2015)
- External news corpora for domain adaptation
- Pre-trained model weights
- Auxiliary resources

Following data management best practices from:
- Amershi et al. (2019): "Software Engineering for Machine Learning"
- Gebru et al. (2021): "Datasheets for Datasets"

Author: Võ Hải Dũng
License: MIT
"""

import os
import sys
import json
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import shutil
import requests
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io_utils import (
    download_file, extract_archive, compute_file_hash,
    ensure_dir, safe_save, safe_load
)
from src.utils.logging_config import setup_logging
from src.core.exceptions import DataError, DataNotFoundError
from configs.constants import DATA_DIR

# Setup logging
logger = setup_logging(
    name=__name__,
    log_dir=PROJECT_ROOT / "outputs" / "logs" / "setup",
    log_file="download_data.log"
)

# Data sources configuration
@dataclass
class DataSource:
    """
    Data source configuration following Gebru et al. (2021) datasheet format.
    
    References:
        Gebru et al. (2021): "Datasheets for Datasets"
        https://arxiv.org/abs/1803.09010
    """
    name: str
    url: str
    description: str
    size_mb: float
    checksum: Optional[str] = None
    license: str = "Unknown"
    citation: Optional[str] = None
    extract: bool = True
    target_dir: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        if self.target_dir:
            data['target_dir'] = str(self.target_dir)
        return data

# Define data sources
DATA_SOURCES = {
    # Primary dataset
    "ag_news": DataSource(
        name="AG News",
        url="https://github.com/mhjabreel/CharCnn_Keras/raw/master/data/ag_news_csv.tar.gz",
        description="AG News dataset with 4 categories (Zhang et al., 2015)",
        size_mb=11.8,
        checksum="b86b60248a414828709e48c3f15c2e56",
        license="Apache-2.0",
        citation="Zhang et al. (2015): Character-level Convolutional Networks for Text Classification",
        target_dir=DATA_DIR / "raw" / "ag_news"
    ),
    
    # Alternative AG News source (Hugging Face)
    "ag_news_hf": DataSource(
        name="AG News (Hugging Face)",
        url="https://huggingface.co/datasets/ag_news/resolve/main/data.zip",
        description="AG News from Hugging Face datasets",
        size_mb=12.0,
        license="Apache-2.0",
        target_dir=DATA_DIR / "raw" / "ag_news_hf"
    ),
    
    # External news corpora for domain adaptation
    "news_crawl": DataSource(
        name="News Crawl 2023",
        url="https://data.statmt.org/news-crawl/en/news.2023.en.shuffled.deduped.gz",
        description="WMT News Crawl corpus for domain-adaptive pretraining",
        size_mb=5000.0,  # 5GB
        license="CC-BY-SA",
        extract=False,  # Keep compressed
        target_dir=DATA_DIR / "external" / "news_crawl"
    ),
    
    # CC-News subset
    "cc_news": DataSource(
        name="CC-News Sample",
        url="https://huggingface.co/datasets/cc_news/resolve/main/cc_news_sample.tar.gz",
        description="Common Crawl News dataset sample",
        size_mb=1000.0,
        license="CC-BY",
        target_dir=DATA_DIR / "external" / "cc_news"
    ),
    
    # Pre-trained models
    "deberta_v3_xlarge": DataSource(
        name="DeBERTa-v3-xlarge",
        url="https://huggingface.co/microsoft/deberta-v3-xlarge/resolve/main/pytorch_model.bin",
        description="DeBERTa-v3-xlarge pre-trained weights (He et al., 2021)",
        size_mb=1500.0,
        extract=False,
        license="MIT",
        citation="He et al. (2021): DeBERTa",
        target_dir=DATA_DIR / "models" / "pretrained"
    ),
    
    # Word embeddings for classical models
    "glove": DataSource(
        name="GloVe Embeddings",
        url="https://nlp.stanford.edu/data/glove.6B.zip",
        description="GloVe word embeddings (Pennington et al., 2014)",
        size_mb=862.0,
        license="Public Domain",
        citation="Pennington et al. (2014): GloVe",
        target_dir=DATA_DIR / "embeddings"
    ),
    
    # Auxiliary resources
    "stopwords": DataSource(
        name="Stopwords",
        url="https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt",
        description="English stopwords list",
        size_mb=0.01,
        extract=False,
        license="MIT",
        target_dir=DATA_DIR / "resources"
    ),
    
    # Contrast sets for evaluation
    "contrast_sets": DataSource(
        name="AG News Contrast Sets",
        url="https://github.com/allenai/contrast-sets/raw/main/data/ag_news/contrast_sets.json",
        description="Contrast sets for robust evaluation (Gardner et al., 2020)",
        size_mb=2.0,
        extract=False,
        license="Apache-2.0",
        citation="Gardner et al. (2020): Evaluating Models' Local Decision Boundaries",
        target_dir=DATA_DIR / "evaluation"
    ),
}

class DataDownloader:
    """
    Manages data downloading with integrity checks and progress tracking.
    
    Implements best practices from:
    - Amershi et al. (2019): "Software Engineering for Machine Learning"
    """
    
    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        max_workers: int = 4,
        verify_checksums: bool = True,
        force_download: bool = False
    ):
        """
        Initialize data downloader.
        
        Args:
            data_dir: Base directory for data storage
            max_workers: Number of parallel download workers
            verify_checksums: Whether to verify file checksums
            force_download: Force re-download even if files exist
        """
        self.data_dir = Path(data_dir)
        self.max_workers = max_workers
        self.verify_checksums = verify_checksums
        self.force_download = force_download
        
        # Create base directories
        ensure_dir(self.data_dir)
        
        # Track download progress
        self.downloaded = []
        self.failed = []
        
    def download_all(
        self,
        sources: Optional[List[str]] = None,
        parallel: bool = True
    ) -> Dict[str, bool]:
        """
        Download all specified data sources.
        
        Args:
            sources: List of source names to download (None for all)
            parallel: Whether to download in parallel
            
        Returns:
            Dictionary mapping source names to success status
        """
        # Select sources to download
        if sources is None:
            sources = list(DATA_SOURCES.keys())
        
        # Filter valid sources
        valid_sources = [s for s in sources if s in DATA_SOURCES]
        if len(valid_sources) < len(sources):
            invalid = set(sources) - set(valid_sources)
            logger.warning(f"Invalid sources will be skipped: {invalid}")
        
        logger.info(f"Starting download of {len(valid_sources)} data sources")
        
        # Download sources
        results = {}
        if parallel and len(valid_sources) > 1:
            results = self._download_parallel(valid_sources)
        else:
            results = self._download_sequential(valid_sources)
        
        # Generate download report
        self._generate_report(results)
        
        return results
    
    def _download_sequential(self, sources: List[str]) -> Dict[str, bool]:
        """Download sources sequentially."""
        results = {}
        
        for source_name in sources:
            source = DATA_SOURCES[source_name]
            success = self._download_source(source)
            results[source_name] = success
            
        return results
    
    def _download_parallel(self, sources: List[str]) -> Dict[str, bool]:
        """Download sources in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit download tasks
            future_to_source = {
                executor.submit(self._download_source, DATA_SOURCES[name]): name
                for name in sources
            }
            
            # Process completed downloads
            for future in tqdm(
                as_completed(future_to_source),
                total=len(sources),
                desc="Downloading data sources"
            ):
                source_name = future_to_source[future]
                try:
                    success = future.result()
                    results[source_name] = success
                except Exception as e:
                    logger.error(f"Failed to download {source_name}: {e}")
                    results[source_name] = False
                    self.failed.append(source_name)
        
        return results
    
    def _download_source(self, source: DataSource) -> bool:
        """
        Download a single data source.
        
        Args:
            source: Data source configuration
            
        Returns:
            Success status
        """
        try:
            # Setup target directory
            target_dir = source.target_dir or self.data_dir / source.name.lower().replace(" ", "_")
            ensure_dir(target_dir)
            
            # Determine file path
            filename = source.url.split("/")[-1]
            file_path = target_dir / filename
            
            # Check if already downloaded
            if file_path.exists() and not self.force_download:
                logger.info(f"{source.name} already exists, verifying...")
                
                # Verify checksum if provided
                if self.verify_checksums and source.checksum:
                    actual_checksum = compute_file_hash(file_path, algorithm="md5")
                    if actual_checksum != source.checksum:
                        logger.warning(f"Checksum mismatch for {source.name}, re-downloading...")
                    else:
                        logger.info(f"{source.name} verified successfully")
                        self.downloaded.append(source.name)
                        return True
                else:
                    self.downloaded.append(source.name)
                    return True
            
            # Download file
            logger.info(f"Downloading {source.name} ({source.size_mb:.1f} MB)...")
            download_file(
                url=source.url,
                dest_path=file_path,
                show_progress=True
            )
            
            # Verify download
            if self.verify_checksums and source.checksum:
                actual_checksum = compute_file_hash(file_path, algorithm="md5")
                if actual_checksum != source.checksum:
                    raise DataError(f"Checksum verification failed for {source.name}")
            
            # Extract if needed
            if source.extract and file_path.suffix in ['.zip', '.tar', '.gz', '.tgz']:
                logger.info(f"Extracting {source.name}...")
                extract_archive(file_path, extract_to=target_dir)
            
            # Save metadata
            self._save_metadata(source, target_dir)
            
            logger.info(f"Successfully downloaded {source.name}")
            self.downloaded.append(source.name)
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {source.name}: {e}")
            self.failed.append(source.name)
            return False
    
    def _save_metadata(self, source: DataSource, target_dir: Path):
        """
        Save data source metadata following datasheet format.
        
        References:
            Gebru et al. (2021): "Datasheets for Datasets"
        """
        metadata = {
            "name": source.name,
            "description": source.description,
            "url": source.url,
            "size_mb": source.size_mb,
            "license": source.license,
            "citation": source.citation,
            "download_date": datetime.now().isoformat(),
            "checksum": source.checksum,
            "version": "1.0.0",
        }
        
        metadata_path = target_dir / "metadata.json"
        safe_save(metadata, metadata_path)
    
    def _generate_report(self, results: Dict[str, bool]):
        """Generate download report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_sources": len(results),
            "successful": sum(results.values()),
            "failed": len(results) - sum(results.values()),
            "downloaded": self.downloaded,
            "failed_sources": self.failed,
            "results": results,
        }
        
        # Save report
        report_path = self.data_dir / "download_report.json"
        safe_save(report, report_path)
        
        # Log summary
        logger.info("=" * 80)
        logger.info("Download Summary:")
        logger.info(f"  Total sources: {report['total_sources']}")
        logger.info(f"  Successful: {report['successful']}")
        logger.info(f"  Failed: {report['failed']}")
        
        if self.failed:
            logger.warning(f"  Failed sources: {', '.join(self.failed)}")
        
        logger.info(f"Report saved to: {report_path}")
        logger.info("=" * 80)

def verify_data_integrity():
    """
    Verify integrity of downloaded data.
    
    Implements data validation following:
    - Breck et al. (2019): "Data Validation for Machine Learning"
    """
    logger.info("Verifying data integrity...")
    
    issues = []
    
    # Check AG News dataset
    ag_news_dir = DATA_DIR / "raw" / "ag_news"
    if ag_news_dir.exists():
        # Check for required files
        required_files = ["train.csv", "test.csv", "classes.txt"]
        for file in required_files:
            if not (ag_news_dir / file).exists():
                issues.append(f"Missing AG News file: {file}")
    else:
        issues.append("AG News dataset not found")
    
    # Report issues
    if issues:
        logger.warning("Data integrity issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
        return False
    
    logger.info("Data integrity verified successfully")
    return True

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Download all data for AG News classification",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help="Specific sources to download (default: all)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Base directory for data storage"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers"
    )
    
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip checksum verification"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available data sources and exit"
    )
    
    args = parser.parse_args()
    
    # List sources if requested
    if args.list:
        print("\nAvailable data sources:")
        print("=" * 80)
        for name, source in DATA_SOURCES.items():
            print(f"\n{name}:")
            print(f"  Description: {source.description}")
            print(f"  Size: {source.size_mb:.1f} MB")
            print(f"  License: {source.license}")
        return 0
    
    try:
        # Initialize downloader
        downloader = DataDownloader(
            data_dir=args.data_dir,
            max_workers=args.workers,
            verify_checksums=not args.no_verify,
            force_download=args.force
        )
        
        # Download data
        results = downloader.download_all(sources=args.sources)
        
        # Verify integrity
        if all(results.values()):
            verify_data_integrity()
            logger.info("All data downloaded successfully!")
            return 0
        else:
            logger.error("Some downloads failed. Check the report for details.")
            return 1
            
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
