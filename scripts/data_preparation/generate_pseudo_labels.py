#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Pseudo-Label Generation for AG News Classification
===================================================

Generates pseudo-labels for unlabeled data using trained models,
following semi-supervised learning approaches from:
- Lee (2013): "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning"
- Xie et al. (2020): "Self-training with Noisy Student improves ImageNet classification"

Author: Võ Hải Dũng
License: MIT
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

from src.utils.logging_config import setup_logging
from src.utils.reproducibility import ensure_reproducibility
from src.data.datasets.external_news import ExternalNewsDataset, ExternalNewsConfig
from configs.constants import AG_NEWS_CLASSES, MAX_SEQUENCE_LENGTH

logger = setup_logging(__name__)

class UnlabeledDataset(Dataset):
    """Dataset for unlabeled texts."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = MAX_SEQUENCE_LENGTH):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "text": text
        }

class PseudoLabelGenerator:
    """
    Generate pseudo-labels for unlabeled data.
    
    Implements pseudo-labeling strategies from:
    - Berthelot et al. (2019): "MixMatch: A Holistic Approach to Semi-Supervised Learning"
    - Sohn et al. (2020): "FixMatch: Simplifying Semi-Supervised Learning"
    """
    
    def __init__(
        self,
        model_path: Path,
        confidence_threshold: float = 0.9,
        temperature: float = 1.0,
        use_ensemble: bool = False,
        device: Optional[str] = None
    ):
        """
        Initialize generator.
        
        Args:
            model_path: Path to trained model
            confidence_threshold: Minimum confidence for pseudo-labeling
            temperature: Temperature for softmax scaling
            use_ensemble: Whether to use model ensemble
            device: Device to use
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        self.use_ensemble = use_ensemble
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Load model(s)
        self._load_models()
        
        # Statistics
        self.stats = {
            "total_processed": 0,
            "high_confidence": 0,
            "low_confidence": 0,
            "class_distribution": {c: 0 for c in AG_NEWS_CLASSES}
        }
    
    def _load_models(self):
        """Load model(s) and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        if self.use_ensemble:
            # Load multiple models for ensemble
            self.models = []
            model_dirs = list(self.model_path.parent.glob("model_*"))
            
            for model_dir in model_dirs[:3]:  # Use up to 3 models
                model = AutoModelForSequenceClassification.from_pretrained(model_dir)
                model.to(self.device)
                model.eval()
                self.models.append(model)
            
            logger.info(f"Loaded {len(self.models)} models for ensemble")
        else:
            # Load single model
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
    
    def generate_pseudo_labels(
        self,
        texts: List[str],
        batch_size: int = 32,
        use_uncertainty_sampling: bool = False
    ) -> pd.DataFrame:
        """
        Generate pseudo-labels for texts.
        
        Args:
            texts: List of unlabeled texts
            batch_size: Batch size for processing
            use_uncertainty_sampling: Whether to use uncertainty sampling
            
        Returns:
            DataFrame with pseudo-labels and confidence scores
        """
        logger.info(f"Generating pseudo-labels for {len(texts)} texts")
        
        # Create dataset and dataloader
        dataset = UnlabeledDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Collect results
        all_texts = []
        all_predictions = []
        all_confidences = []
        all_uncertainties = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating pseudo-labels"):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                texts_batch = batch["text"]
                
                # Get predictions
                if self.use_ensemble:
                    probs = self._ensemble_predict(input_ids, attention_mask)
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    logits = outputs.logits / self.temperature
                    probs = torch.softmax(logits, dim=-1)
                
                # Calculate metrics
                max_probs, predictions = torch.max(probs, dim=-1)
                
                # Calculate uncertainty (entropy)
                uncertainties = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
                
                # Store results
                all_texts.extend(texts_batch)
                all_predictions.extend(predictions.cpu().numpy())
                all_confidences.extend(max_probs.cpu().numpy())
                all_uncertainties.extend(uncertainties.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        # Create DataFrame
        results_df = pd.DataFrame({
            "text": all_texts,
            "pseudo_label": all_predictions,
            "confidence": all_confidences,
            "uncertainty": all_uncertainties
        })
        
        # Add class names
        results_df["predicted_class"] = results_df["pseudo_label"].apply(
            lambda x: AG_NEWS_CLASSES[x]
        )
        
        # Add probabilities for each class
        prob_array = np.array(all_probabilities)
        for i, class_name in enumerate(AG_NEWS_CLASSES):
            results_df[f"prob_{class_name}"] = prob_array[:, i]
        
        # Filter by confidence threshold
        high_confidence_mask = results_df["confidence"] >= self.confidence_threshold
        
        # Update statistics
        self.stats["total_processed"] = len(results_df)
        self.stats["high_confidence"] = high_confidence_mask.sum()
        self.stats["low_confidence"] = (~high_confidence_mask).sum()
        
        for label in results_df[high_confidence_mask]["pseudo_label"].values:
            class_name = AG_NEWS_CLASSES[label]
            self.stats["class_distribution"][class_name] += 1
        
        # Apply uncertainty sampling if requested
        if use_uncertainty_sampling:
            results_df = self._apply_uncertainty_sampling(results_df)
        
        logger.info(f"Generated {len(results_df)} pseudo-labels")
        logger.info(f"High confidence: {self.stats['high_confidence']} "
                   f"({100 * self.stats['high_confidence'] / len(results_df):.1f}%)")
        
        return results_df
    
    def _ensemble_predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get ensemble predictions."""
        all_probs = []
        
        for model in self.models:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits / self.temperature
            probs = torch.softmax(logits, dim=-1)
            all_probs.append(probs)
        
        # Average probabilities
        ensemble_probs = torch.stack(all_probs).mean(dim=0)
        return ensemble_probs
    
    def _apply_uncertainty_sampling(
        self,
        df: pd.DataFrame,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Apply uncertainty sampling to select most informative samples.
        
        Following active learning principles from:
        - Settles (2009): "Active Learning Literature Survey"
        """
        # Sort by uncertainty (high to low)
        df_sorted = df.sort_values("uncertainty", ascending=False)
        
        # Select top-k if specified
        if top_k:
            df_sorted = df_sorted.head(top_k)
        
        # Also include high-confidence samples
        high_conf_df = df[df["confidence"] >= self.confidence_threshold]
        
        # Combine
        result_df = pd.concat([high_conf_df, df_sorted]).drop_duplicates()
        
        logger.info(f"Selected {len(result_df)} samples after uncertainty sampling")
        
        return result_df
    
    def iterative_pseudo_labeling(
        self,
        initial_texts: List[str],
        num_iterations: int = 3,
        batch_size: int = 32
    ) -> pd.DataFrame:
        """
        Iterative pseudo-labeling with self-training.
        
        Following self-training approaches from:
        - Yarowsky (1995): "Unsupervised Word Sense Disambiguation"
        - Riloff et al. (2003): "Learning Dictionaries for Information Extraction"
        """
        all_results = []
        remaining_texts = initial_texts.copy()
        
        for iteration in range(num_iterations):
            logger.info(f"Iteration {iteration + 1}/{num_iterations}")
            
            if not remaining_texts:
                break
            
            # Generate pseudo-labels
            results_df = self.generate_pseudo_labels(
                remaining_texts,
                batch_size=batch_size
            )
            
            # Filter high-confidence samples
            high_conf_df = results_df[results_df["confidence"] >= self.confidence_threshold]
            
            if len(high_conf_df) == 0:
                logger.warning("No high-confidence samples found, stopping iteration")
                break
            
            # Add to results
            all_results.append(high_conf_df)
            
            # Remove labeled texts from remaining
            labeled_texts = set(high_conf_df["text"].values)
            remaining_texts = [t for t in remaining_texts if t not in labeled_texts]
            
            logger.info(f"Labeled {len(high_conf_df)} samples, "
                       f"{len(remaining_texts)} remaining")
            
            # Optionally retrain model here (not implemented)
        
        # Combine all results
        if all_results:
            final_df = pd.concat(all_results, ignore_index=True)
        else:
            final_df = pd.DataFrame()
        
        return final_df
    
    def print_statistics(self):
        """Print pseudo-labeling statistics."""
        print("\n" + "=" * 60)
        print("PSEUDO-LABELING STATISTICS")
        print("=" * 60)
        
        print(f"Total processed: {self.stats['total_processed']}")
        print(f"High confidence: {self.stats['high_confidence']} "
              f"({100 * self.stats['high_confidence'] / max(self.stats['total_processed'], 1):.1f}%)")
        print(f"Low confidence:  {self.stats['low_confidence']} "
              f"({100 * self.stats['low_confidence'] / max(self.stats['total_processed'], 1):.1f}%)")
        
        print("\nClass Distribution (high confidence only):")
        for class_name, count in self.stats['class_distribution'].items():
            percentage = 100 * count / max(self.stats['high_confidence'], 1)
            print(f"  {class_name:<10}: {count:5d} ({percentage:.1f}%)")
        
        print("=" * 60)

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Generate pseudo-labels for unlabeled data"
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to trained model"
    )
    
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Path to unlabeled data file (CSV or TXT)"
    )
    
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("pseudo_labels.csv"),
        help="Output file for pseudo-labeled data"
    )
    
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.9,
        help="Minimum confidence threshold"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing"
    )
    
    parser.add_argument(
        "--use-external",
        action="store_true",
        help="Use external news corpus"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=10000,
        help="Maximum number of samples to process"
    )
    
    parser.add_argument(
        "--iterative",
        action="store_true",
        help="Use iterative pseudo-labeling"
    )
    
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=3,
        help="Number of iterations for iterative labeling"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Ensure reproducibility
    ensure_reproducibility(seed=args.seed)
    
    # Load unlabeled data
    if args.use_external:
        # Use external news corpus
        logger.info("Loading external news corpus")
        config = ExternalNewsConfig(max_samples=args.max_samples)
        dataset = ExternalNewsDataset(config, purpose="pseudo_labeling")
        texts = dataset.texts[:args.max_samples]
    elif args.input_file:
        # Load from file
        logger.info(f"Loading data from {args.input_file}")
        
        if args.input_file.suffix == ".csv":
            df = pd.read_csv(args.input_file)
            texts = df["text"].values.tolist()
        else:
            with open(args.input_file, "r") as f:
                texts = [line.strip() for line in f if line.strip()]
    else:
        # Use sample texts
        logger.info("Using sample texts")
        texts = [
            "The government announced new policies today.",
            "The team won the championship game.",
            "Stock markets rose after earnings report.",
            "Scientists discover new planet in distant galaxy.",
        ] * 100  # Repeat for demonstration
    
    texts = texts[:args.max_samples]
    logger.info(f"Processing {len(texts)} texts")
    
    # Create generator
    generator = PseudoLabelGenerator(
        model_path=args.model_path,
        confidence_threshold=args.confidence_threshold
    )
    
    # Generate pseudo-labels
    if args.iterative:
        results_df = generator.iterative_pseudo_labeling(
            texts,
            num_iterations=args.num_iterations,
            batch_size=args.batch_size
        )
    else:
        results_df = generator.generate_pseudo_labels(
            texts,
            batch_size=args.batch_size
        )
    
    # Filter by confidence
    high_conf_df = results_df[results_df["confidence"] >= args.confidence_threshold]
    
    # Save results
    output_path = args.output_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    high_conf_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(high_conf_df)} pseudo-labeled samples to {output_path}")
    
    # Save all results with confidence scores
    all_results_path = output_path.with_suffix(".all.csv")
    results_df.to_csv(all_results_path, index=False)
    logger.info(f"Saved all {len(results_df)} results to {all_results_path}")
    
    # Print statistics
    generator.print_statistics()
    
    # Save statistics
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(generator.stats, f, indent=2)
    
    logger.info(f"Statistics saved to {stats_path}")

if __name__ == "__main__":
    main()
