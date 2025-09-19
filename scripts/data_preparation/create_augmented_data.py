"""
Create Augmented Data Script
=============================

This script generates augmented data for AG News classification using various
augmentation techniques following best practices from:
- Shorten et al. (2021): "Text Data Augmentation for Deep Learning"
- Wei & Zou (2019): "EDA: Easy Data Augmentation Techniques"
- Zhang et al. (2018): "mixup: Beyond Empirical Risk Minimization"
- Gardner et al. (2020): "Evaluating Models' Local Decision Boundaries via Contrast Sets"

Author: Võ Hải Dũng
License: MIT
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import hashlib
from collections import defaultdict, Counter

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import custom modules
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_all_seeds
from src.utils.io_utils import save_json, load_json, ensure_dir
from src.core.registry import Registry
from src.data.datasets.ag_news import AGNewsDataset
from configs.config_loader import ConfigLoader
from configs.constants import (
    AG_NEWS_CLASSES,
    MAX_SEQUENCE_LENGTH,
    RANDOM_SEED,
    DATA_DIR
)

# Import augmentation modules
from src.data.augmentation.base_augmenter import BaseAugmenter, CompositeAugmenter
from src.data.augmentation.token_replacement import TokenReplacementAugmenter
from src.data.augmentation.back_translation import BackTranslationAugmenter
from src.data.augmentation.paraphrase import ParaphraseAugmenter
from src.data.augmentation.mixup import MixUpAugmenter
from src.data.augmentation.cutmix import CutMixAugmenter
from src.data.augmentation.adversarial import AdversarialAugmenter
from src.data.augmentation.contrast_set_generator import ContrastSetGenerator

# Setup logging
logger = setup_logging(__name__)


@dataclass
class AugmentationStats:
    """Statistics for augmentation process."""
    
    total_original: int = 0
    total_augmented: int = 0
    augmentation_ratio: float = 0.0
    
    # Per-class statistics
    class_distribution_original: Dict[str, int] = None
    class_distribution_augmented: Dict[str, int] = None
    
    # Per-method statistics
    method_counts: Dict[str, int] = None
    method_success_rates: Dict[str, float] = None
    
    # Quality metrics
    avg_similarity: float = 0.0
    avg_length_ratio: float = 0.0
    label_preservation_rate: float = 0.0
    
    # Performance metrics
    processing_time: float = 0.0
    samples_per_second: float = 0.0
    
    # Cache statistics
    cache_hits: int = 0
    cache_misses: int = 0
    
    def __post_init__(self):
        """Initialize dictionaries if None."""
        if self.class_distribution_original is None:
            self.class_distribution_original = {}
        if self.class_distribution_augmented is None:
            self.class_distribution_augmented = {}
        if self.method_counts is None:
            self.method_counts = {}
        if self.method_success_rates is None:
            self.method_success_rates = {}


class AugmentationPipeline:
    """
    Pipeline for data augmentation following best practices from:
    - Bayer et al. (2022): "A Survey on Data Augmentation for Text Classification"
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        augmentation_methods: Optional[List[str]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize augmentation pipeline.
        
        Args:
            config_path: Path to augmentation configuration
            augmentation_methods: List of augmentation methods to use
            device: Computing device
        """
        self.config = self._load_config(config_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.augmenters = {}
        self.stats = AugmentationStats()
        
        # Initialize specified augmentation methods
        if augmentation_methods:
            self._initialize_augmenters(augmentation_methods)
        
        logger.info(f"Initialized augmentation pipeline on {self.device}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load augmentation configuration."""
        if config_path and Path(config_path).exists():
            config_loader = ConfigLoader()
            return config_loader.load_config(config_path)
        else:
            # Load default configurations
            default_configs = {}
            config_dir = PROJECT_ROOT / "configs" / "data" / "augmentation"
            
            for config_file in config_dir.glob("*.yaml"):
                config_name = config_file.stem
                try:
                    config_loader = ConfigLoader()
                    default_configs[config_name] = config_loader.load_config(config_file)
                except Exception as e:
                    logger.warning(f"Failed to load config {config_file}: {e}")
            
            return default_configs
    
    def _initialize_augmenters(self, methods: List[str]):
        """Initialize specified augmentation methods."""
        for method in methods:
            try:
                if method == "token_replacement":
                    self.augmenters[method] = TokenReplacementAugmenter()
                    
                elif method == "back_translation":
                    self.augmenters[method] = BackTranslationAugmenter(device=self.device)
                    
                elif method == "paraphrase":
                    self.augmenters[method] = ParaphraseAugmenter(device=self.device)
                    
                elif method == "mixup":
                    self.augmenters[method] = MixUpAugmenter(device=self.device)
                    
                elif method == "cutmix":
                    self.augmenters[method] = CutMixAugmenter()
                    
                elif method == "adversarial":
                    self.augmenters[method] = AdversarialAugmenter(device=self.device)
                    
                elif method == "contrast_sets":
                    self.augmenters[method] = ContrastSetGenerator()
                    
                else:
                    logger.warning(f"Unknown augmentation method: {method}")
                    continue
                
                logger.info(f"Initialized {method} augmenter")
                
            except Exception as e:
                logger.error(f"Failed to initialize {method}: {e}")
    
    def augment_dataset(
        self,
        dataset: Union[Dataset, List[Tuple[str, int]]],
        methods: Optional[List[str]] = None,
        num_augmentations: int = 1,
        preserve_original: bool = True,
        balance_classes: bool = False,
        max_samples_per_class: Optional[int] = None
    ) -> Dataset:
        """
        Augment entire dataset.
        
        Args:
            dataset: Input dataset
            methods: Augmentation methods to apply
            num_augmentations: Number of augmentations per sample
            preserve_original: Whether to keep original samples
            balance_classes: Whether to balance class distribution
            max_samples_per_class: Maximum samples per class
            
        Returns:
            Augmented dataset
        """
        start_time = time.time()
        
        # Use all available methods if not specified
        if methods is None:
            methods = list(self.augmenters.keys())
        
        # Convert to standard format
        samples = self._extract_samples(dataset)
        self.stats.total_original = len(samples)
        
        # Calculate class distribution
        self._calculate_class_distribution(samples, "original")
        
        # Determine augmentation strategy per class
        augmentation_plan = self._create_augmentation_plan(
            samples,
            balance_classes,
            max_samples_per_class,
            num_augmentations
        )
        
        # Perform augmentation
        augmented_samples = []
        
        if preserve_original:
            augmented_samples.extend(samples)
        
        # Progress bar
        total_augmentations = sum(plan["num_augmentations"] for plan in augmentation_plan.values())
        pbar = tqdm(total=total_augmentations, desc="Augmenting samples")
        
        for class_id, plan in augmentation_plan.items():
            class_samples = [s for s in samples if s[1] == class_id]
            
            for sample_idx, (text, label) in enumerate(class_samples[:plan["samples_to_augment"]]):
                for aug_idx in range(plan["num_augmentations"]):
                    # Select augmentation method
                    method = self._select_augmentation_method(methods, class_id, aug_idx)
                    
                    # Apply augmentation
                    augmented = self._apply_augmentation(
                        text,
                        label,
                        method,
                        class_samples
                    )
                    
                    # Add augmented samples
                    for aug_text, aug_label in augmented:
                        augmented_samples.append((
                            aug_text,
                            aug_label,
                            {
                                "original_text": text,
                                "original_label": label,
                                "augmentation_method": method,
                                "augmentation_index": aug_idx
                            }
                        ))
                        self.stats.method_counts[method] = self.stats.method_counts.get(method, 0) + 1
                    
                    pbar.update(1)
        
        pbar.close()
        
        # Calculate final statistics
        self.stats.total_augmented = len(augmented_samples) - (len(samples) if preserve_original else 0)
        self.stats.augmentation_ratio = self.stats.total_augmented / max(self.stats.total_original, 1)
        self.stats.processing_time = time.time() - start_time
        self.stats.samples_per_second = self.stats.total_augmented / max(self.stats.processing_time, 1)
        
        # Calculate augmented class distribution
        self._calculate_class_distribution(
            [(s[0], s[1]) for s in augmented_samples],
            "augmented"
        )
        
        # Convert to Dataset format
        augmented_dataset = self._create_dataset(augmented_samples)
        
        return augmented_dataset
    
    def _extract_samples(self, dataset: Union[Dataset, List]) -> List[Tuple[str, int]]:
        """Extract (text, label) pairs from dataset."""
        samples = []
        
        if isinstance(dataset, Dataset):
            for item in dataset:
                text = item.get("text", item.get("sentence", ""))
                label = item.get("label", item.get("labels", 0))
                samples.append((text, label))
        else:
            # Assume list of tuples
            samples = dataset
        
        return samples
    
    def _calculate_class_distribution(
        self,
        samples: List[Tuple[str, int]],
        distribution_type: str
    ):
        """Calculate class distribution."""
        distribution = Counter(label for _, label in samples)
        
        if distribution_type == "original":
            self.stats.class_distribution_original = dict(distribution)
        else:
            self.stats.class_distribution_augmented = dict(distribution)
    
    def _create_augmentation_plan(
        self,
        samples: List[Tuple[str, int]],
        balance_classes: bool,
        max_samples_per_class: Optional[int],
        num_augmentations: int
    ) -> Dict[int, Dict[str, Any]]:
        """
        Create augmentation plan for each class.
        
        Following class balancing strategies from:
        - Buda et al. (2018): "A systematic study of the class imbalance problem"
        """
        class_counts = Counter(label for _, label in samples)
        max_count = max(class_counts.values())
        
        plan = {}
        
        for class_id, count in class_counts.items():
            if balance_classes:
                # Calculate how many augmentations needed to balance
                target_count = min(max_count, max_samples_per_class or max_count)
                augmentations_needed = max(0, target_count - count)
                
                # Distribute augmentations across samples
                if count > 0:
                    augmentations_per_sample = augmentations_needed // count
                    extra_augmentations = augmentations_needed % count
                else:
                    augmentations_per_sample = 0
                    extra_augmentations = 0
                
                plan[class_id] = {
                    "samples_to_augment": count,
                    "num_augmentations": augmentations_per_sample,
                    "extra_augmentations": extra_augmentations
                }
            else:
                # Uniform augmentation
                plan[class_id] = {
                    "samples_to_augment": count,
                    "num_augmentations": num_augmentations,
                    "extra_augmentations": 0
                }
        
        return plan
    
    def _select_augmentation_method(
        self,
        methods: List[str],
        class_id: int,
        aug_idx: int
    ) -> str:
        """
        Select augmentation method based on strategy.
        
        Following selection strategies from:
        - Cubuk et al. (2020): "RandAugment: Practical automated data augmentation"
        """
        # Simple round-robin selection
        # Could be enhanced with more sophisticated strategies
        method_idx = aug_idx % len(methods)
        return methods[method_idx]
    
    def _apply_augmentation(
        self,
        text: str,
        label: int,
        method: str,
        class_samples: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        """Apply specified augmentation method."""
        augmented = []
        
        try:
            augmenter = self.augmenters.get(method)
            
            if not augmenter:
                return [(text, label)]
            
            # Special handling for MixUp and CutMix (need pairs)
            if method in ["mixup", "cutmix"]:
                # Select random sample from same class for mixing
                mix_candidates = [s for s in class_samples if s[0] != text]
                
                if mix_candidates:
                    mix_text, mix_label = random.choice(mix_candidates)
                    result = augmenter.augment_single(
                        text,
                        label,
                        mix_with=mix_text,
                        mix_label=mix_label
                    )
                else:
                    # Fallback to original
                    result = text
            
            # Special handling for contrast sets
            elif method == "contrast_sets":
                result = augmenter.augment_single(text, label)
                
                # Contrast sets return (text, label) tuples
                if isinstance(result, list) and result and isinstance(result[0], tuple):
                    augmented.extend(result)
                    return augmented
            
            # Standard augmentation
            else:
                result = augmenter.augment_single(text, label)
            
            # Handle different return types
            if isinstance(result, tuple):
                # (text, mixed_label) from MixUp
                augmented.append(result)
            elif isinstance(result, list):
                # List of augmented texts
                for aug_text in result:
                    augmented.append((aug_text, label))
            else:
                # Single augmented text
                augmented.append((result, label))
            
        except Exception as e:
            logger.error(f"Augmentation failed for {method}: {e}")
            # Return original on failure
            augmented.append((text, label))
        
        return augmented
    
    def _create_dataset(self, samples: List[Tuple]) -> Dataset:
        """Create HuggingFace Dataset from samples."""
        data_dict = {
            "text": [],
            "label": [],
            "metadata": []
        }
        
        for sample in samples:
            if len(sample) == 2:
                text, label = sample
                metadata = {}
            else:
                text, label, metadata = sample
            
            data_dict["text"].append(text)
            data_dict["label"].append(label)
            data_dict["metadata"].append(metadata)
        
        return Dataset.from_dict(data_dict)
    
    def generate_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate augmentation report.
        
        Following reporting practices from:
        - Dodge et al. (2019): "Show Your Work: Improved Reporting of Experimental Results"
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "statistics": asdict(self.stats),
            "configuration": {
                "device": str(self.device),
                "augmenters": list(self.augmenters.keys())
            },
            "class_balance": {
                "original": self.stats.class_distribution_original,
                "augmented": self.stats.class_distribution_augmented
            },
            "method_performance": {
                method: {
                    "count": count,
                    "percentage": count / max(self.stats.total_augmented, 1) * 100
                }
                for method, count in self.stats.method_counts.items()
            }
        }
        
        if output_path:
            save_json(report, output_path)
            logger.info(f"Report saved to {output_path}")
        
        return report


def main():
    """Main function for data augmentation."""
    parser = argparse.ArgumentParser(
        description="Generate augmented data for AG News classification"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DATA_DIR / "augmented"),
        help="Output directory for augmented data"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to augment"
    )
    
    # Augmentation arguments
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["token_replacement", "back_translation"],
        choices=[
            "token_replacement",
            "back_translation",
            "paraphrase",
            "mixup",
            "cutmix",
            "adversarial",
            "contrast_sets"
        ],
        help="Augmentation methods to apply"
    )
    parser.add_argument(
        "--num_augmentations",
        type=int,
        default=1,
        help="Number of augmentations per sample"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to augmentation configuration file"
    )
    
    # Processing arguments
    parser.add_argument(
        "--balance_classes",
        action="store_true",
        help="Balance class distribution through augmentation"
    )
    parser.add_argument(
        "--max_samples_per_class",
        type=int,
        default=None,
        help="Maximum samples per class after augmentation"
    )
    parser.add_argument(
        "--preserve_original",
        action="store_true",
        default=True,
        help="Keep original samples in augmented dataset"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for augmentation"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=str(PROJECT_ROOT / ".cache" / "augmentation"),
        help="Cache directory for augmentation"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    set_all_seeds(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(args.cache_dir)
    
    logger.info("Starting data augmentation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load dataset
    logger.info(f"Loading {args.split} split from {args.data_dir}")
    
    ag_news = AGNewsDataset(data_dir=args.data_dir)
    dataset = ag_news.load_split(args.split)
    
    if dataset is None:
        # Fallback to HuggingFace datasets
        dataset = load_dataset("ag_news", split=args.split)
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Initialize pipeline
    device = torch.device(args.device)
    pipeline = AugmentationPipeline(
        config_path=args.config,
        augmentation_methods=args.methods,
        device=device
    )
    
    # Perform augmentation
    logger.info(f"Applying augmentation methods: {args.methods}")
    
    augmented_dataset = pipeline.augment_dataset(
        dataset,
        methods=args.methods,
        num_augmentations=args.num_augmentations,
        preserve_original=args.preserve_original,
        balance_classes=args.balance_classes,
        max_samples_per_class=args.max_samples_per_class
    )
    
    logger.info(f"Generated {len(augmented_dataset)} samples after augmentation")
    
    # Save augmented dataset
    output_path = Path(args.output_dir) / f"{args.split}_augmented"
    
    # Save in multiple formats
    # 1. HuggingFace Dataset format
    augmented_dataset.save_to_disk(output_path)
    logger.info(f"Saved augmented dataset to {output_path}")
    
    # 2. CSV format for inspection
    csv_path = output_path.with_suffix(".csv")
    df = pd.DataFrame({
        "text": augmented_dataset["text"],
        "label": augmented_dataset["label"]
    })
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")
    
    # 3. JSON format with metadata
    json_path = output_path.with_suffix(".json")
    json_data = []
    for i in range(len(augmented_dataset)):
        json_data.append({
            "text": augmented_dataset[i]["text"],
            "label": augmented_dataset[i]["label"],
            "metadata": augmented_dataset[i].get("metadata", {})
        })
    save_json(json_data, json_path)
    logger.info(f"Saved JSON to {json_path}")
    
    # Generate report
    report_path = output_path.parent / f"{args.split}_augmentation_report.json"
    report = pipeline.generate_report(str(report_path))
    
    # Print summary
    print("\n" + "=" * 50)
    print("Augmentation Summary")
    print("=" * 50)
    print(f"Original samples: {report['statistics']['total_original']}")
    print(f"Augmented samples: {report['statistics']['total_augmented']}")
    print(f"Augmentation ratio: {report['statistics']['augmentation_ratio']:.2f}x")
    print(f"Processing time: {report['statistics']['processing_time']:.2f} seconds")
    print(f"Samples/second: {report['statistics']['samples_per_second']:.2f}")
    
    print("\nClass Distribution:")
    print("-" * 30)
    for class_id in range(len(AG_NEWS_CLASSES)):
        orig_count = report['class_balance']['original'].get(str(class_id), 0)
        aug_count = report['class_balance']['augmented'].get(str(class_id), 0)
        print(f"{AG_NEWS_CLASSES[class_id]:15} | Original: {orig_count:6} | Augmented: {aug_count:6}")
    
    print("\nAugmentation Methods Used:")
    print("-" * 30)
    for method, stats in report['method_performance'].items():
        print(f"{method:20} | Count: {stats['count']:6} | {stats['percentage']:.1f}%")
    
    print("\nAugmentation completed successfully!")


if __name__ == "__main__":
    main()
