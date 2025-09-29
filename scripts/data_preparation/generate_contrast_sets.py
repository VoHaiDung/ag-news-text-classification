"""
Contrast Set Generation Script for AG News Text Classification
================================================================================
This script generates contrast sets for robust evaluation of text classification
models by creating minimal perturbations that change the expected label. It
implements state-of-the-art techniques for generating counterfactual examples
that test model decision boundaries and robustness.

The contrast set methodology provides a more challenging and realistic evaluation
framework compared to standard test sets, revealing model weaknesses and biases
that may not be apparent in traditional evaluation settings.

References:
    - Gardner et al. (2020): Evaluating Models' Local Decision Boundaries via Contrast Sets
    - Kaushik et al. (2020): Learning the Difference that Makes a Difference with Counterfactually-Augmented Data
    - Ross et al. (2021): Tailor - Generating and Perturbing Text with Semantic Controls
    - Wu et al. (2021): Polyjuice - Generating Counterfactuals for Explaining, Evaluating, and Improving Models
    - Ribeiro et al. (2020): Beyond Accuracy - Behavioral Testing of NLP Models with CheckList

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
from dataclasses import dataclass, asdict, field
import time
from datetime import datetime
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
from collections import defaultdict, Counter
import hashlib

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import custom modules
from src.utils.logging_config import setup_logging
from src.utils.reproducibility import set_all_seeds
from src.utils.io_utils import save_json, load_json, ensure_dir
from src.core.registry import Registry
from src.data.datasets.ag_news import AGNewsDataset, AGNewsConfig
from src.data.augmentation.contrast_set_generator import ContrastSetGenerator, ContrastSetConfig
from configs.config_loader import ConfigLoader
from configs.constants import (
    AG_NEWS_CLASSES,
    AG_NEWS_NUM_CLASSES,
    ID_TO_LABEL,
    LABEL_TO_ID,
    MAX_SEQUENCE_LENGTH,
    DATA_DIR,
    RANDOM_SEEDS
)

# Setup logging
logger = setup_logging(__name__)


@dataclass
class ContrastSetStats:
    """
    Comprehensive statistics for contrast set generation process
    
    This dataclass tracks detailed metrics about the contrast set generation
    including quality measures, perturbation analysis, and validation results
    following the evaluation framework proposed in Gardner et al. (2020).
    """
    
    # Basic statistics
    total_original: int = 0
    total_contrasts: int = 0
    contrast_ratio: float = 0.0
    
    # Per-class statistics
    class_distribution_original: Dict[int, int] = field(default_factory=dict)
    class_distribution_contrasts: Dict[int, int] = field(default_factory=dict)
    
    # Label transition matrix
    label_transitions: Dict[Tuple[int, int], int] = field(default_factory=dict)
    
    # Quality metrics
    avg_edit_distance: float = 0.0
    avg_semantic_similarity: float = 0.0
    label_flip_rate: float = 0.0
    
    # Perturbation statistics
    perturbation_types: Dict[str, int] = field(default_factory=dict)
    avg_perturbations_per_sample: float = 0.0
    
    # Performance metrics
    processing_time: float = 0.0
    samples_per_second: float = 0.0
    
    # Validation statistics
    valid_contrasts: int = 0
    invalid_contrasts: int = 0
    validation_rate: float = 0.0


class ContrastSetPipeline:
    """
    Pipeline for generating and validating contrast sets
    
    This class orchestrates the complete contrast set generation process
    including perturbation generation, model validation, and quality assessment.
    It implements techniques from recent research on counterfactual generation
    and behavioral testing of NLP models.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize the contrast set generation pipeline
        
        Args:
            config_path: Path to contrast set configuration file
            device: Computing device for model inference
            model_name: Pre-trained model name for validation
        """
        self.config = self._load_config(config_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats = ContrastSetStats()
        
        # Initialize contrast set generator
        self.generator = self._initialize_generator()
        
        # Load model for validation if specified
        self.model = None
        self.tokenizer = None
        if model_name:
            self._load_model(model_name)
        
        logger.info(f"Initialized contrast set pipeline on {self.device}")
    
    def _load_config(self, config_path: Optional[str]) -> ContrastSetConfig:
        """
        Load configuration for contrast set generation
        
        Args:
            config_path: Optional path to configuration file
            
        Returns:
            ContrastSetConfig object with generation parameters
        """
        if config_path and Path(config_path).exists():
            config_loader = ConfigLoader()
            config_dict = config_loader.load_config(config_path)
            return ContrastSetConfig(**config_dict.get("contrast", {}))
        else:
            # Load default configuration
            default_config_path = PROJECT_ROOT / "configs" / "data" / "augmentation" / "contrast_sets.yaml"
            if default_config_path.exists():
                config_loader = ConfigLoader()
                config_dict = config_loader.load_config(default_config_path)
                return ContrastSetConfig(**config_dict.get("contrast", {}))
            else:
                return ContrastSetConfig()
    
    def _initialize_generator(self) -> ContrastSetGenerator:
        """
        Initialize the contrast set generator with configured strategy
        
        Returns:
            Configured ContrastSetGenerator instance
        """
        logger.info(f"Initializing contrast set generator with strategy: {self.config.generation_strategy}")
        return ContrastSetGenerator(self.config)
    
    def _load_model(self, model_name: str):
        """
        Load pre-trained model for contrast validation
        
        Loads a transformer model to validate that generated contrasts
        actually change the model's predictions as expected.
        
        Args:
            model_name: HuggingFace model identifier
        """
        try:
            logger.info(f"Loading model {model_name} for validation")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=AG_NEWS_NUM_CLASSES
            ).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            logger.warning("Proceeding without model validation")
    
    def generate_contrast_sets(
        self,
        dataset: Union[Dataset, List[Tuple[str, int]]],
        num_contrasts_per_sample: int = 1,
        validate_with_model: bool = False,
        stratified_sampling: bool = False,
        max_samples: Optional[int] = None
    ) -> Tuple[Dataset, ContrastSetStats]:
        """
        Generate contrast sets for evaluation dataset
        
        Creates minimal perturbations that change the expected label while
        preserving fluency and meaningfulness. Implements the contrast set
        generation protocol from Gardner et al. (2020).
        
        Args:
            dataset: Input dataset to generate contrasts for
            num_contrasts_per_sample: Number of contrasts per original example
            validate_with_model: Whether to validate contrasts with model predictions
            stratified_sampling: Whether to use stratified sampling for class balance
            max_samples: Maximum number of samples to process
            
        Returns:
            Tuple of (contrast_dataset, generation_statistics)
        """
        start_time = time.time()
        
        # Extract samples
        samples = self._extract_samples(dataset, max_samples, stratified_sampling)
        self.stats.total_original = len(samples)
        
        # Calculate original class distribution
        self._calculate_class_distribution(samples, "original")
        
        # Generate contrasts
        contrast_data = []
        
        # Progress bar
        pbar = tqdm(total=len(samples), desc="Generating contrast sets")
        
        for text, label in samples:
            # Generate contrasts for this sample
            contrasts = self.generator.augment_single(text, label)
            
            # Ensure contrasts is a list of tuples
            if not isinstance(contrasts, list):
                contrasts = []
            
            # Process contrasts
            valid_contrasts = []
            for contrast_item in contrasts[:num_contrasts_per_sample]:
                if isinstance(contrast_item, tuple) and len(contrast_item) == 2:
                    contrast_text, contrast_label = contrast_item
                    
                    # Validate contrast if requested
                    if validate_with_model and self.model:
                        if self._validate_contrast(text, contrast_text, label, contrast_label):
                            valid_contrasts.append((contrast_text, contrast_label))
                            self.stats.valid_contrasts += 1
                        else:
                            self.stats.invalid_contrasts += 1
                    else:
                        valid_contrasts.append((contrast_text, contrast_label))
                        self.stats.valid_contrasts += 1
            
            # Add to contrast data
            for contrast_text, contrast_label in valid_contrasts:
                contrast_data.append({
                    "original_text": text,
                    "original_label": label,
                    "contrast_text": contrast_text,
                    "contrast_label": contrast_label,
                    "label_changed": label != contrast_label,
                    "perturbation_type": self._identify_perturbation_type(text, contrast_text),
                    "edit_distance": self._calculate_edit_distance(text, contrast_text)
                })
                
                # Update statistics
                self.stats.total_contrasts += 1
                self.stats.label_transitions[(label, contrast_label)] = \
                    self.stats.label_transitions.get((label, contrast_label), 0) + 1
            
            pbar.update(1)
        
        pbar.close()
        
        # Calculate final statistics
        self._calculate_final_statistics(contrast_data)
        self.stats.processing_time = time.time() - start_time
        self.stats.samples_per_second = self.stats.total_original / max(self.stats.processing_time, 1)
        
        # Create dataset
        contrast_dataset = Dataset.from_list(contrast_data)
        
        return contrast_dataset, self.stats
    
    def _extract_samples(
        self,
        dataset: Union[Dataset, List],
        max_samples: Optional[int],
        stratified: bool
    ) -> List[Tuple[str, int]]:
        """
        Extract samples from dataset with optional stratification
        
        Args:
            dataset: Input dataset
            max_samples: Maximum number of samples to extract
            stratified: Whether to use stratified sampling
            
        Returns:
            List of (text, label) tuples
        """
        samples = []
        
        if isinstance(dataset, Dataset):
            for item in dataset:
                text = item.get("text", item.get("sentence", ""))
                label = item.get("label", item.get("labels", 0))
                if isinstance(label, torch.Tensor):
                    label = label.item()
                samples.append((text, label))
        else:
            samples = dataset
        
        # Apply stratified sampling if requested
        if stratified and max_samples and max_samples < len(samples):
            samples = self._stratified_sample(samples, max_samples)
        elif max_samples and max_samples < len(samples):
            samples = random.sample(samples, max_samples)
        
        return samples
    
    def _stratified_sample(
        self,
        samples: List[Tuple[str, int]],
        n_samples: int
    ) -> List[Tuple[str, int]]:
        """
        Perform stratified sampling to maintain class distribution
        
        Implements Neyman allocation for proportional sampling across classes
        ensuring representative samples from each category.
        
        Args:
            samples: Full list of samples
            n_samples: Number of samples to select
            
        Returns:
            Stratified sample maintaining class proportions
        """
        # Group by class
        class_samples = defaultdict(list)
        for text, label in samples:
            class_samples[label].append((text, label))
        
        # Calculate samples per class
        samples_per_class = n_samples // len(class_samples)
        
        # Sample from each class
        stratified = []
        for class_id, class_data in class_samples.items():
            n = min(samples_per_class, len(class_data))
            stratified.extend(random.sample(class_data, n))
        
        return stratified
    
    def _validate_contrast(
        self,
        original_text: str,
        contrast_text: str,
        original_label: int,
        contrast_label: int
    ) -> bool:
        """
        Validate contrast with model predictions
        
        Checks that the model actually predicts the expected label for
        the contrast example, following validation protocol from
        Kaushik et al. (2020).
        
        Args:
            original_text: Original example text
            contrast_text: Generated contrast text
            original_label: Original label
            contrast_label: Expected contrast label
            
        Returns:
            True if model prediction matches expected contrast label
        """
        if not self.model or not self.tokenizer:
            return True
        
        # Predict on contrast
        inputs = self.tokenizer(
            contrast_text,
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_label = outputs.logits.argmax(dim=-1).item()
        
        # Check if prediction matches expected contrast label
        return predicted_label == contrast_label
    
    def _identify_perturbation_type(self, original: str, contrast: str) -> str:
        """
        Identify the type of perturbation applied
        
        Classifies perturbations into categories following the taxonomy
        from Gardner et al. (2020) for analyzing contrast mechanisms.
        
        Args:
            original: Original text
            contrast: Contrast text
            
        Returns:
            Perturbation type identifier
        """
        original_lower = original.lower()
        contrast_lower = contrast.lower()
        
        # Check for negation
        negation_words = ["not", "no", "never", "neither", "nor", "n't"]
        original_negations = sum(1 for word in negation_words if word in original_lower)
        contrast_negations = sum(1 for word in negation_words if word in contrast_lower)
        
        if original_negations != contrast_negations:
            return "negation"
        
        # Check for number changes
        import re
        original_numbers = re.findall(r'\d+', original)
        contrast_numbers = re.findall(r'\d+', contrast)
        
        if original_numbers != contrast_numbers:
            return "number"
        
        # Check for entity changes
        original_words = set(original_lower.split())
        contrast_words = set(contrast_lower.split())
        
        if len(original_words - contrast_words) > 0 and len(contrast_words - original_words) > 0:
            return "entity"
        
        # Check for temporal changes
        temporal_words = ["today", "yesterday", "tomorrow", "now", "then", "will", "was", "is"]
        original_temporal = any(word in original_lower for word in temporal_words)
        contrast_temporal = any(word in contrast_lower for word in temporal_words)
        
        if original_temporal != contrast_temporal:
            return "temporal"
        
        return "other"
    
    def _calculate_edit_distance(self, text1: str, text2: str) -> int:
        """
        Calculate word-level Levenshtein edit distance
        
        Computes the minimum number of word-level edits required to
        transform one text into another, providing a measure of
        perturbation magnitude.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Word-level edit distance
        """
        words1 = text1.split()
        words2 = text2.split()
        
        # Dynamic programming for edit distance
        m, n = len(words1), len(words2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i-1] == words2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        return dp[m][n]
    
    def _calculate_class_distribution(
        self,
        samples: List[Tuple[str, int]],
        distribution_type: str
    ):
        """
        Calculate and store class distribution statistics
        
        Args:
            samples: List of (text, label) samples
            distribution_type: Type of distribution ('original' or 'contrasts')
        """
        distribution = Counter(label for _, label in samples)
        
        if distribution_type == "original":
            self.stats.class_distribution_original = dict(distribution)
        else:
            self.stats.class_distribution_contrasts = dict(distribution)
    
    def _calculate_final_statistics(self, contrast_data: List[Dict]):
        """
        Calculate comprehensive statistics for the generated contrast set
        
        Computes quality metrics and analysis following the evaluation
        framework from Gardner et al. (2020).
        
        Args:
            contrast_data: List of generated contrast examples
        """
        if not contrast_data:
            return
        
        # Calculate contrast ratio
        self.stats.contrast_ratio = self.stats.total_contrasts / max(self.stats.total_original, 1)
        
        # Calculate label flip rate
        label_flips = sum(1 for c in contrast_data if c["label_changed"])
        self.stats.label_flip_rate = label_flips / max(len(contrast_data), 1)
        
        # Calculate average edit distance
        edit_distances = [c["edit_distance"] for c in contrast_data]
        self.stats.avg_edit_distance = np.mean(edit_distances) if edit_distances else 0.0
        
        # Count perturbation types
        for contrast in contrast_data:
            perturb_type = contrast["perturbation_type"]
            self.stats.perturbation_types[perturb_type] = \
                self.stats.perturbation_types.get(perturb_type, 0) + 1
        
        # Calculate average perturbations per sample
        self.stats.avg_perturbations_per_sample = \
            self.stats.total_contrasts / max(self.stats.total_original, 1)
        
        # Calculate validation rate
        total_attempted = self.stats.valid_contrasts + self.stats.invalid_contrasts
        self.stats.validation_rate = self.stats.valid_contrasts / max(total_attempted, 1)
        
        # Calculate contrast class distribution
        contrast_labels = [c["contrast_label"] for c in contrast_data]
        self.stats.class_distribution_contrasts = dict(Counter(contrast_labels))
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive report of contrast set generation
        
        Creates a detailed report following documentation standards from
        Dodge et al. (2019) for reproducible experimental results.
        
        Returns:
            Dictionary containing full generation report and statistics
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "configuration": {
                "generation_strategy": self.config.generation_strategy,
                "contrast_type": self.config.contrast_type,
                "max_perturbations": self.config.max_perturbations,
                "ensure_label_change": self.config.ensure_label_change
            },
            "statistics": asdict(self.stats),
            "class_distributions": {
                "original": self.stats.class_distribution_original,
                "contrasts": self.stats.class_distribution_contrasts
            },
            "label_transitions": {
                f"{ID_TO_LABEL[src]}->{ID_TO_LABEL[tgt]}": count
                for (src, tgt), count in self.stats.label_transitions.items()
            },
            "perturbation_analysis": {
                "types": self.stats.perturbation_types,
                "avg_edit_distance": self.stats.avg_edit_distance,
                "label_flip_rate": self.stats.label_flip_rate
            },
            "performance": {
                "processing_time": self.stats.processing_time,
                "samples_per_second": self.stats.samples_per_second,
                "contrast_ratio": self.stats.contrast_ratio
            }
        }
        
        return report


def evaluate_model_on_contrast_sets(
    model,
    tokenizer,
    contrast_dataset: Dataset,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluate model performance on contrast sets
    
    Implements the evaluation protocol from Gardner et al. (2020) to assess
    model robustness and consistency on contrast examples.
    
    Args:
        model: Pre-trained classification model
        tokenizer: Tokenizer for the model
        contrast_dataset: Generated contrast set
        device: Computing device
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Metrics
    total_original = 0
    correct_original = 0
    total_contrast = 0
    correct_contrast = 0
    consistency_count = 0
    
    with torch.no_grad():
        for item in tqdm(contrast_dataset, desc="Evaluating on contrast sets"):
            # Predict on original
            orig_inputs = tokenizer(
                item["original_text"],
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt"
            ).to(device)
            
            orig_outputs = model(**orig_inputs)
            orig_pred = orig_outputs.logits.argmax(dim=-1).item()
            
            # Predict on contrast
            contrast_inputs = tokenizer(
                item["contrast_text"],
                truncation=True,
                padding="max_length",
                max_length=MAX_SEQUENCE_LENGTH,
                return_tensors="pt"
            ).to(device)
            
            contrast_outputs = model(**contrast_inputs)
            contrast_pred = contrast_outputs.logits.argmax(dim=-1).item()
            
            # Update metrics
            total_original += 1
            total_contrast += 1
            
            if orig_pred == item["original_label"]:
                correct_original += 1
            
            if contrast_pred == item["contrast_label"]:
                correct_contrast += 1
            
            # Check consistency
            if item["label_changed"]:
                if orig_pred != contrast_pred:
                    consistency_count += 1
            else:
                if orig_pred == contrast_pred:
                    consistency_count += 1
    
    # Calculate metrics
    metrics = {
        "original_accuracy": correct_original / max(total_original, 1),
        "contrast_accuracy": correct_contrast / max(total_contrast, 1),
        "consistency_rate": consistency_count / max(total_original, 1),
        "robustness_gap": (correct_original - correct_contrast) / max(total_original, 1)
    }
    
    return metrics


def main():
    """
    Main entry point for contrast set generation
    
    Orchestrates the complete contrast set generation pipeline including
    data loading, generation, validation, and evaluation with comprehensive
    reporting and statistics.
    """
    parser = argparse.ArgumentParser(
        description="Generate contrast sets for AG News classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate basic contrast sets
  python generate_contrast_sets.py --split test --num_contrasts 1
  
  # Generate with model validation
  python generate_contrast_sets.py --split test --validate --model roberta-base
  
  # Generate stratified sample
  python generate_contrast_sets.py --split test --stratified --max_samples 1000
        """
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(DATA_DIR / "processed"),
        help="Directory containing processed data"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DATA_DIR / "augmented" / "contrast_sets"),
        help="Output directory for contrast sets"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to generate contrasts for"
    )
    
    # Generation arguments
    parser.add_argument(
        "--num_contrasts",
        type=int,
        default=1,
        help="Number of contrasts per sample"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="rule_based",
        choices=["rule_based", "model_based", "human_guided"],
        help="Generation strategy"
    )
    parser.add_argument(
        "--contrast_type",
        type=str,
        default="minimal",
        choices=["minimal", "diverse", "adversarial"],
        help="Type of contrasts to generate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to contrast set configuration file"
    )
    
    # Validation arguments
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate contrasts with model"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name for validation"
    )
    
    # Sampling arguments
    parser.add_argument(
        "--stratified",
        action="store_true",
        help="Use stratified sampling"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model on generated contrast sets"
    )
    
    # Other arguments
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEEDS[0],
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
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
    
    logger.info("Starting contrast set generation")
    logger.info(f"Arguments: {vars(args)}")
    
    # Load dataset
    logger.info(f"Loading {args.split} split from {args.data_dir}")
    
    ag_news_config = AGNewsConfig(data_dir=Path(args.data_dir))
    ag_news = AGNewsDataset(ag_news_config, split=args.split)
    
    # Convert to list of (text, label) tuples
    samples = []
    for i in range(len(ag_news)):
        item = ag_news[i]
        samples.append((item["text"], item["label"]))
    
    logger.info(f"Loaded {len(samples)} samples")
    
    # Initialize pipeline
    device = torch.device(args.device)
    
    # Create custom config if specified
    if args.config:
        pipeline = ContrastSetPipeline(
            config_path=args.config,
            device=device,
            model_name=args.model if args.validate else None
        )
    else:
        # Create config from arguments
        config = ContrastSetConfig(
            generation_strategy=args.strategy,
            contrast_type=args.contrast_type,
            ensure_label_change=True
        )
        
        # Save config to temporary path
        temp_config_path = Path(args.output_dir) / "temp_config.yaml"
        ensure_dir(temp_config_path.parent)
        
        config_dict = {"contrast": asdict(config)}
        with open(temp_config_path, 'w') as f:
            import yaml
            yaml.dump(config_dict, f)
        
        pipeline = ContrastSetPipeline(
            config_path=str(temp_config_path),
            device=device,
            model_name=args.model if args.validate else None
        )
    
    # Generate contrast sets
    logger.info("Generating contrast sets...")
    
    contrast_dataset, stats = pipeline.generate_contrast_sets(
        samples,
        num_contrasts_per_sample=args.num_contrasts,
        validate_with_model=args.validate,
        stratified_sampling=args.stratified,
        max_samples=args.max_samples
    )
    
    logger.info(f"Generated {len(contrast_dataset)} contrast pairs")
    
    # Save contrast dataset
    output_path = Path(args.output_dir) / f"{args.split}_contrast_sets"
    
    # 1. Save as HuggingFace Dataset
    contrast_dataset.save_to_disk(output_path)
    logger.info(f"Saved contrast dataset to {output_path}")
    
    # 2. Save as CSV for inspection
    csv_path = output_path.with_suffix(".csv")
    df = pd.DataFrame(contrast_dataset)
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")
    
    # 3. Save as JSON with full metadata
    json_path = output_path.with_suffix(".json")
    json_data = [dict(item) for item in contrast_dataset]
    save_json(json_data, json_path)
    logger.info(f"Saved JSON to {json_path}")
    
    # Generate and save report
    report = pipeline.generate_report()
    report_path = output_path.parent / f"{args.split}_contrast_report.json"
    save_json(report, report_path)
    logger.info(f"Saved report to {report_path}")
    
    # Evaluate model if requested
    if args.evaluate and args.model:
        logger.info(f"Evaluating model {args.model} on contrast sets...")
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model,
            num_labels=AG_NEWS_NUM_CLASSES
        ).to(device)
        
        eval_metrics = evaluate_model_on_contrast_sets(
            model,
            tokenizer,
            contrast_dataset,
            device
        )
        
        # Add to report
        report["evaluation"] = eval_metrics
        
        # Save updated report
        save_json(report, report_path)
        
        # Print evaluation results
        print("\nModel Evaluation on Contrast Sets:")
        print("-" * 40)
        for metric, value in eval_metrics.items():
            print(f"{metric:20}: {value:.4f}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("Contrast Set Generation Summary")
    print("=" * 50)
    print(f"Original samples: {stats.total_original}")
    print(f"Contrast pairs: {stats.total_contrasts}")
    print(f"Contrast ratio: {stats.contrast_ratio:.2f}")
    print(f"Label flip rate: {stats.label_flip_rate:.2%}")
    print(f"Avg edit distance: {stats.avg_edit_distance:.2f}")
    print(f"Processing time: {stats.processing_time:.2f} seconds")
    print(f"Samples/second: {stats.samples_per_second:.2f}")
    
    print("\nPerturbation Types:")
    print("-" * 30)
    for perturb_type, count in stats.perturbation_types.items():
        percentage = count / max(stats.total_contrasts, 1) * 100
        print(f"{perturb_type:15}: {count:5} ({percentage:.1f}%)")
    
    print("\nLabel Transitions:")
    print("-" * 30)
    for (src, tgt), count in stats.label_transitions.items():
        print(f"{ID_TO_LABEL[src]:10} -> {ID_TO_LABEL[tgt]:10}: {count:5}")
    
    print("\nContrast set generation completed successfully!")


if __name__ == "__main__":
    main()
