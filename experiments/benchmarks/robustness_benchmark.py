"""
Robustness Benchmark for AG News Text Classification
================================================================================
This module implements comprehensive robustness evaluation for classification models,
testing model performance under various perturbations and adversarial conditions.

The benchmark evaluates:
- Adversarial robustness (TextFooler, BERT-Attack)
- Input perturbation resistance
- Out-of-distribution detection
- Contrast set consistency
- Label noise tolerance

References:
    - Jin, D., et al. (2020). Is BERT Really Robust? A Strong Baseline for Natural Language Attack
    - Gardner, M., et al. (2020). Evaluating Models' Local Decision Boundaries via Contrast Sets
    - Hendrycks, D., & Gimpel, K. (2017). A Baseline for Detecting Misclassified and Out-of-Distribution Examples

Author: Võ Hải Dũng
License: MIT
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple, Callable
from pathlib import Path
import json
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

from src.core.registry import Registry
from src.core.factory import Factory
from src.utils.reproducibility import set_seed
from src.data.datasets.ag_news import AGNewsDataset
from src.data.augmentation.adversarial import AdversarialAugmenter
from src.data.augmentation.contrast_set_generator import ContrastSetGenerator
from src.evaluation.metrics.classification_metrics import ClassificationMetrics

logger = logging.getLogger(__name__)


@dataclass
class RobustnessMetrics:
    """Container for robustness evaluation metrics."""
    
    clean_accuracy: float
    perturbed_accuracy: float
    robustness_score: float
    attack_success_rate: float
    confidence_drop: float
    consistency_score: float
    ood_detection_auroc: float
    noise_tolerance: float
    contrast_set_accuracy: float
    prediction_stability: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "clean_accuracy": self.clean_accuracy,
            "perturbed_accuracy": self.perturbed_accuracy,
            "robustness_score": self.robustness_score,
            "attack_success_rate": self.attack_success_rate,
            "confidence_drop": self.confidence_drop,
            "consistency_score": self.consistency_score,
            "ood_detection_auroc": self.ood_detection_auroc,
            "noise_tolerance": self.noise_tolerance,
            "contrast_set_accuracy": self.contrast_set_accuracy,
            "prediction_stability": self.prediction_stability
        }


class RobustnessBenchmark:
    """
    Comprehensive robustness benchmarking for text classification models.
    
    This class evaluates model robustness across multiple dimensions:
    - Adversarial attacks
    - Input perturbations
    - Distribution shifts
    - Contrast sets
    - Noisy labels
    """
    
    def __init__(
        self,
        models: List[str],
        dataset_name: str = "ag_news",
        attack_types: List[str] = ["textfooler", "char_swap", "word_deletion"],
        perturbation_levels: List[float] = [0.1, 0.2, 0.3],
        num_samples: int = 1000,
        device: str = "cuda",
        seed: int = 42
    ):
        """
        Initialize robustness benchmark.
        
        Args:
            models: List of model names to benchmark
            dataset_name: Name of dataset to use
            attack_types: Types of adversarial attacks to test
            perturbation_levels: Levels of perturbation to apply
            num_samples: Number of samples to test
            device: Device to run on
            seed: Random seed for reproducibility
        """
        self.models = models
        self.dataset_name = dataset_name
        self.attack_types = attack_types
        self.perturbation_levels = perturbation_levels
        self.num_samples = num_samples
        self.device = device
        self.seed = seed
        
        self.registry = Registry()
        self.factory = Factory()
        self.metrics_calculator = ClassificationMetrics()
        self.adversarial_augmenter = AdversarialAugmenter()
        self.contrast_generator = ContrastSetGenerator()
        self.results = {}
        
        # Check device availability
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run complete robustness benchmark.
        
        Returns:
            Dictionary containing benchmark results
        """
        logger.info("Starting robustness benchmark")
        logger.info(f"Models: {self.models}")
        logger.info(f"Attack types: {self.attack_types}")
        logger.info(f"Number of samples: {self.num_samples}")
        
        # Load dataset
        dataset = self._load_dataset()
        
        # Select subset for testing
        test_data = self._select_test_subset(dataset)
        
        # Generate contrast sets
        contrast_sets = self._generate_contrast_sets(test_data)
        
        # Benchmark each model
        for model_name in self.models:
            logger.info(f"\nBenchmarking model: {model_name}")
            self.results[model_name] = self._benchmark_model(
                model_name,
                test_data,
                contrast_sets
            )
        
        # Generate comparative analysis
        comparison = self._generate_comparison()
        
        # Calculate overall robustness scores
        robustness_scores = self._calculate_robustness_scores()
        
        # Create summary report
        summary = self._create_summary()
        
        return {
            "model_results": self.results,
            "comparison": comparison,
            "robustness_scores": robustness_scores,
            "summary": summary
        }
    
    def _benchmark_model(
        self,
        model_name: str,
        test_data: Dict[str, Any],
        contrast_sets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Benchmark robustness of a single model.
        
        Args:
            model_name: Name of model to benchmark
            test_data: Test dataset
            contrast_sets: Generated contrast sets
            
        Returns:
            Robustness benchmark results
        """
        # Load model
        model = self._load_model(model_name)
        model = model.to(self.device)
        model.eval()
        
        # Get clean predictions
        clean_predictions, clean_scores = self._get_predictions(model, test_data)
        clean_accuracy = self._calculate_accuracy(
            clean_predictions, 
            test_data["labels"]
        )
        
        logger.info(f"Clean accuracy: {clean_accuracy:.4f}")
        
        # Test adversarial robustness
        adversarial_results = self._test_adversarial_robustness(
            model, test_data, clean_predictions
        )
        
        # Test input perturbations
        perturbation_results = self._test_perturbation_robustness(
            model, test_data, clean_predictions
        )
        
        # Test out-of-distribution detection
        ood_results = self._test_ood_detection(
            model, test_data, clean_scores
        )
        
        # Test contrast set consistency
        contrast_results = self._test_contrast_consistency(
            model, contrast_sets
        )
        
        # Test label noise tolerance
        noise_results = self._test_noise_tolerance(
            model, test_data
        )
        
        # Calculate overall metrics
        overall_metrics = self._calculate_overall_metrics(
            clean_accuracy,
            adversarial_results,
            perturbation_results,
            ood_results,
            contrast_results,
            noise_results
        )
        
        return {
            "metrics": overall_metrics.to_dict(),
            "adversarial": adversarial_results,
            "perturbation": perturbation_results,
            "ood_detection": ood_results,
            "contrast_sets": contrast_results,
            "noise_tolerance": noise_results,
            "clean_accuracy": clean_accuracy
        }
    
    def _test_adversarial_robustness(
        self,
        model: Any,
        test_data: Dict[str, Any],
        clean_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test model robustness against adversarial attacks.
        
        Args:
            model: Model to test
            test_data: Test dataset
            clean_predictions: Clean model predictions
            
        Returns:
            Adversarial robustness results
        """
        results = {}
        
        for attack_type in self.attack_types:
            logger.info(f"Testing {attack_type} attack")
            
            # Generate adversarial examples
            adversarial_texts = self._generate_adversarial_examples(
                test_data["texts"],
                test_data["labels"],
                attack_type,
                model
            )
            
            # Get predictions on adversarial examples
            adv_predictions, adv_scores = self._get_predictions(
                model,
                {"texts": adversarial_texts, "labels": test_data["labels"]}
            )
            
            # Calculate metrics
            adv_accuracy = self._calculate_accuracy(
                adv_predictions,
                test_data["labels"]
            )
            
            # Attack success rate: percentage of samples where prediction changed
            attack_success_rate = np.mean(
                clean_predictions != adv_predictions
            )
            
            # Confidence drop
            clean_confidence = np.max(clean_predictions, axis=-1)
            adv_confidence = np.max(adv_scores, axis=-1)
            confidence_drop = np.mean(clean_confidence - adv_confidence)
            
            results[attack_type] = {
                "accuracy": adv_accuracy,
                "attack_success_rate": attack_success_rate,
                "confidence_drop": confidence_drop,
                "robustness_score": 1.0 - attack_success_rate
            }
        
        return results
    
    def _test_perturbation_robustness(
        self,
        model: Any,
        test_data: Dict[str, Any],
        clean_predictions: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test model robustness against input perturbations.
        
        Args:
            model: Model to test
            test_data: Test dataset
            clean_predictions: Clean model predictions
            
        Returns:
            Perturbation robustness results
        """
        results = {}
        
        perturbation_types = [
            "char_swap",
            "word_shuffle",
            "synonym_replacement",
            "random_insertion",
            "random_deletion"
        ]
        
        for pert_type in perturbation_types:
            logger.info(f"Testing {pert_type} perturbation")
            
            level_results = []
            
            for level in self.perturbation_levels:
                # Apply perturbation
                perturbed_texts = self._apply_perturbation(
                    test_data["texts"],
                    pert_type,
                    level
                )
                
                # Get predictions
                pert_predictions, pert_scores = self._get_predictions(
                    model,
                    {"texts": perturbed_texts, "labels": test_data["labels"]}
                )
                
                # Calculate consistency
                consistency = np.mean(clean_predictions == pert_predictions)
                
                # Calculate accuracy
                accuracy = self._calculate_accuracy(
                    pert_predictions,
                    test_data["labels"]
                )
                
                level_results.append({
                    "level": level,
                    "accuracy": accuracy,
                    "consistency": consistency
                })
            
            results[pert_type] = level_results
        
        return results
    
    def _test_ood_detection(
        self,
        model: Any,
        test_data: Dict[str, Any],
        in_distribution_scores: np.ndarray
    ) -> Dict[str, Any]:
        """
        Test out-of-distribution detection capability.
        
        Args:
            model: Model to test
            test_data: Test dataset
            in_distribution_scores: Scores for in-distribution samples
            
        Returns:
            OOD detection results
        """
        # Generate OOD samples
        ood_samples = self._generate_ood_samples(len(test_data["texts"]))
        
        # Get OOD predictions
        ood_predictions, ood_scores = self._get_predictions(
            model,
            {"texts": ood_samples}
        )
        
        # Calculate uncertainty metrics
        in_dist_uncertainty = self._calculate_uncertainty(in_distribution_scores)
        ood_uncertainty = self._calculate_uncertainty(ood_scores)
        
        # Create labels (1 for in-distribution, 0 for OOD)
        labels = np.concatenate([
            np.ones(len(in_dist_uncertainty)),
            np.zeros(len(ood_uncertainty))
        ])
        
        # Combine uncertainties
        uncertainties = np.concatenate([in_dist_uncertainty, ood_uncertainty])
        
        # Calculate AUROC and AUPRC
        auroc = roc_auc_score(labels, -uncertainties)  # Negative because lower uncertainty means in-distribution
        auprc = average_precision_score(labels, -uncertainties)
        
        return {
            "auroc": auroc,
            "auprc": auprc,
            "in_dist_uncertainty_mean": np.mean(in_dist_uncertainty),
            "ood_uncertainty_mean": np.mean(ood_uncertainty),
            "separation_score": np.mean(ood_uncertainty) - np.mean(in_dist_uncertainty)
        }
    
    def _test_contrast_consistency(
        self,
        model: Any,
        contrast_sets: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test model consistency on contrast sets.
        
        Args:
            model: Model to test
            contrast_sets: Contrast set examples
            
        Returns:
            Contrast consistency results
        """
        original_texts = contrast_sets["original"]
        contrast_texts = contrast_sets["contrast"]
        expected_labels = contrast_sets["expected_labels"]
        
        # Get predictions for original
        orig_predictions, orig_scores = self._get_predictions(
            model,
            {"texts": original_texts}
        )
        
        # Get predictions for contrast
        contrast_predictions, contrast_scores = self._get_predictions(
            model,
            {"texts": contrast_texts}
        )
        
        # Calculate contrast accuracy
        contrast_accuracy = self._calculate_accuracy(
            contrast_predictions,
            expected_labels
        )
        
        # Calculate consistency (predictions should change as expected)
        expected_changes = contrast_sets.get("expected_changes", None)
        if expected_changes is not None:
            consistency = np.mean(
                (orig_predictions != contrast_predictions) == expected_changes
            )
        else:
            consistency = contrast_accuracy
        
        # Calculate confidence stability
        confidence_diff = np.abs(
            np.max(orig_scores, axis=-1) - np.max(contrast_scores, axis=-1)
        )
        confidence_stability = 1.0 - np.mean(confidence_diff)
        
        return {
            "contrast_accuracy": contrast_accuracy,
            "consistency_score": consistency,
            "confidence_stability": confidence_stability,
            "num_samples": len(original_texts)
        }
    
    def _test_noise_tolerance(
        self,
        model: Any,
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Test model tolerance to label noise.
        
        Args:
            model: Model to test
            test_data: Test dataset
            
        Returns:
            Noise tolerance results
        """
        noise_levels = [0.05, 0.1, 0.2, 0.3]
        results = []
        
        for noise_level in noise_levels:
            logger.info(f"Testing with {noise_level*100}% label noise")
            
            # Add label noise
            noisy_labels = self._add_label_noise(
                test_data["labels"],
                noise_level
            )
            
            # Get predictions
            predictions, scores = self._get_predictions(model, test_data)
            
            # Calculate accuracy on noisy labels
            noisy_accuracy = self._calculate_accuracy(
                predictions,
                noisy_labels
            )
            
            # Calculate accuracy on clean labels
            clean_accuracy = self._calculate_accuracy(
                predictions,
                test_data["labels"]
            )
            
            results.append({
                "noise_level": noise_level,
                "noisy_accuracy": noisy_accuracy,
                "clean_accuracy": clean_accuracy,
                "degradation": 1.0 - (noisy_accuracy / max(clean_accuracy, 1e-10))
            })
        
        # Calculate overall noise tolerance score
        degradations = [r["degradation"] for r in results]
        tolerance_score = 1.0 - np.mean(degradations)
        
        return {
            "results_by_level": results,
            "tolerance_score": tolerance_score
        }
    
    def _generate_adversarial_examples(
        self,
        texts: List[str],
        labels: np.ndarray,
        attack_type: str,
        model: Any
    ) -> List[str]:
        """
        Generate adversarial examples.
        
        Args:
            texts: Original texts
            labels: True labels
            attack_type: Type of attack
            model: Target model
            
        Returns:
            Adversarial texts
        """
        adversarial_texts = []
        
        for text, label in zip(texts, labels):
            # Apply attack based on type
            if attack_type == "textfooler":
                adv_text = self._textfooler_attack(text, label, model)
            elif attack_type == "char_swap":
                adv_text = self._char_swap_attack(text)
            elif attack_type == "word_deletion":
                adv_text = self._word_deletion_attack(text)
            else:
                adv_text = text
            
            adversarial_texts.append(adv_text)
        
        return adversarial_texts
    
    def _textfooler_attack(
        self,
        text: str,
        label: int,
        model: Any
    ) -> str:
        """
        Simplified TextFooler attack implementation.
        
        Args:
            text: Input text
            label: True label
            model: Target model
            
        Returns:
            Adversarial text
        """
        words = text.split()
        
        # Try to replace each word
        for i, word in enumerate(words):
            # Get synonyms (simplified)
            synonyms = self._get_synonyms(word)
            
            for synonym in synonyms[:5]:  # Try top 5 synonyms
                # Create candidate
                candidate_words = words.copy()
                candidate_words[i] = synonym
                candidate_text = " ".join(candidate_words)
                
                # Check if it changes prediction
                pred, _ = self._get_predictions(
                    model,
                    {"texts": [candidate_text]}
                )
                
                if pred[0] != label:
                    return candidate_text
        
        return text
    
    def _char_swap_attack(self, text: str, swap_rate: float = 0.1) -> str:
        """
        Character swap attack.
        
        Args:
            text: Input text
            swap_rate: Rate of character swapping
            
        Returns:
            Perturbed text
        """
        chars = list(text)
        num_swaps = int(len(chars) * swap_rate)
        
        for _ in range(num_swaps):
            idx = np.random.randint(0, len(chars) - 1)
            if chars[idx] != ' ' and chars[idx + 1] != ' ':
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        
        return ''.join(chars)
    
    def _word_deletion_attack(self, text: str, delete_rate: float = 0.1) -> str:
        """
        Word deletion attack.
        
        Args:
            text: Input text
            delete_rate: Rate of word deletion
            
        Returns:
            Perturbed text
        """
        words = text.split()
        num_deletions = max(1, int(len(words) * delete_rate))
        
        indices_to_delete = np.random.choice(
            len(words),
            size=min(num_deletions, len(words) - 1),
            replace=False
        )
        
        remaining_words = [
            word for i, word in enumerate(words)
            if i not in indices_to_delete
        ]
        
        return " ".join(remaining_words)
    
    def _apply_perturbation(
        self,
        texts: List[str],
        pert_type: str,
        level: float
    ) -> List[str]:
        """
        Apply perturbation to texts.
        
        Args:
            texts: Input texts
            pert_type: Type of perturbation
            level: Perturbation level
            
        Returns:
            Perturbed texts
        """
        perturbed = []
        
        for text in texts:
            if pert_type == "char_swap":
                perturbed_text = self._char_swap_attack(text, level)
            elif pert_type == "word_shuffle":
                perturbed_text = self._word_shuffle(text, level)
            elif pert_type == "synonym_replacement":
                perturbed_text = self._synonym_replacement(text, level)
            elif pert_type == "random_insertion":
                perturbed_text = self._random_insertion(text, level)
            elif pert_type == "random_deletion":
                perturbed_text = self._word_deletion_attack(text, level)
            else:
                perturbed_text = text
            
            perturbed.append(perturbed_text)
        
        return perturbed
    
    def _word_shuffle(self, text: str, shuffle_rate: float) -> str:
        """Shuffle words in text."""
        words = text.split()
        num_shuffles = int(len(words) * shuffle_rate)
        
        for _ in range(num_shuffles):
            idx1, idx2 = np.random.choice(len(words), size=2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return " ".join(words)
    
    def _synonym_replacement(self, text: str, replace_rate: float) -> str:
        """Replace words with synonyms."""
        words = text.split()
        num_replacements = int(len(words) * replace_rate)
        
        indices = np.random.choice(
            len(words),
            size=min(num_replacements, len(words)),
            replace=False
        )
        
        for idx in indices:
            synonyms = self._get_synonyms(words[idx])
            if synonyms:
                words[idx] = np.random.choice(synonyms)
        
        return " ".join(words)
    
    def _random_insertion(self, text: str, insert_rate: float) -> str:
        """Insert random words."""
        words = text.split()
        num_insertions = int(len(words) * insert_rate)
        
        for _ in range(num_insertions):
            # Get a random word from the text
            random_word = np.random.choice(words)
            # Insert at random position
            insert_pos = np.random.randint(0, len(words) + 1)
            words.insert(insert_pos, random_word)
        
        return " ".join(words)
    
    def _get_synonyms(self, word: str) -> List[str]:
        """
        Get synonyms for a word (simplified).
        
        Args:
            word: Input word
            
        Returns:
            List of synonyms
        """
        # Simplified synonym mapping
        synonym_map = {
            "good": ["great", "excellent", "fine", "nice"],
            "bad": ["poor", "terrible", "awful", "horrible"],
            "big": ["large", "huge", "enormous", "massive"],
            "small": ["tiny", "little", "mini", "compact"],
            # Add more as needed
        }
        
        return synonym_map.get(word.lower(), [])
    
    def _generate_ood_samples(self, num_samples: int) -> List[str]:
        """
        Generate out-of-distribution samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            OOD text samples
        """
        ood_samples = []
        
        # Generate random text
        for _ in range(num_samples // 3):
            # Random characters
            length = np.random.randint(50, 200)
            random_text = ''.join(
                np.random.choice(list('abcdefghijklmnopqrstuvwxyz '), size=length)
            )
            ood_samples.append(random_text)
        
        # Generate from different domain
        for _ in range(num_samples // 3):
            # Scientific text (different from news)
            scientific_terms = [
                "quantum", "photosynthesis", "mitochondria", "algorithm",
                "hypothesis", "theorem", "equation", "molecule"
            ]
            num_words = np.random.randint(10, 30)
            text = ' '.join(np.random.choice(scientific_terms, size=num_words))
            ood_samples.append(text)
        
        # Generate nonsense but grammatical
        for _ in range(num_samples - len(ood_samples)):
            templates = [
                "The {} {} {} the {} {}.",
                "{} {} are {} {} in the {}.",
                "When {} {}, the {} {} {}."
            ]
            template = np.random.choice(templates)
            
            nouns = ["colorless", "green", "ideas", "sleep", "furiously"]
            text = template.format(*np.random.choice(nouns, size=5))
            ood_samples.append(text)
        
        return ood_samples
    
    def _calculate_uncertainty(self, scores: np.ndarray) -> np.ndarray:
        """
        Calculate prediction uncertainty.
        
        Args:
            scores: Prediction scores
            
        Returns:
            Uncertainty values
        """
        # Use entropy as uncertainty measure
        epsilon = 1e-10
        scores = np.clip(scores, epsilon, 1 - epsilon)
        entropy = -np.sum(scores * np.log(scores), axis=-1)
        
        return entropy
    
    def _add_label_noise(
        self,
        labels: np.ndarray,
        noise_level: float
    ) -> np.ndarray:
        """
        Add noise to labels.
        
        Args:
            labels: Original labels
            noise_level: Proportion of labels to corrupt
            
        Returns:
            Noisy labels
        """
        noisy_labels = labels.copy()
        num_classes = len(np.unique(labels))
        num_noisy = int(len(labels) * noise_level)
        
        # Select random indices to corrupt
        noisy_indices = np.random.choice(
            len(labels),
            size=num_noisy,
            replace=False
        )
        
        for idx in noisy_indices:
            # Change to a different random label
            original_label = labels[idx]
            possible_labels = [i for i in range(num_classes) if i != original_label]
            noisy_labels[idx] = np.random.choice(possible_labels)
        
        return noisy_labels
    
    def _calculate_overall_metrics(
        self,
        clean_accuracy: float,
        adversarial_results: Dict[str, Any],
        perturbation_results: Dict[str, Any],
        ood_results: Dict[str, Any],
        contrast_results: Dict[str, Any],
        noise_results: Dict[str, Any]
    ) -> RobustnessMetrics:
        """
        Calculate overall robustness metrics.
        
        Args:
            clean_accuracy: Clean test accuracy
            adversarial_results: Adversarial attack results
            perturbation_results: Perturbation results
            ood_results: OOD detection results
            contrast_results: Contrast set results
            noise_results: Noise tolerance results
            
        Returns:
            Overall robustness metrics
        """
        # Average adversarial accuracy
        adv_accuracies = [
            res["accuracy"] for res in adversarial_results.values()
        ]
        avg_adv_accuracy = np.mean(adv_accuracies)
        
        # Average attack success rate
        attack_success_rates = [
            res["attack_success_rate"] for res in adversarial_results.values()
        ]
        avg_attack_success = np.mean(attack_success_rates)
        
        # Average confidence drop
        confidence_drops = [
            res["confidence_drop"] for res in adversarial_results.values()
        ]
        avg_confidence_drop = np.mean(confidence_drops)
        
        # Perturbation consistency
        consistencies = []
        for pert_type, levels in perturbation_results.items():
            for level_result in levels:
                consistencies.append(level_result["consistency"])
        avg_consistency = np.mean(consistencies)
        
        # Prediction stability (based on perturbations)
        stabilities = []
        for pert_type, levels in perturbation_results.items():
            accuracies = [l["accuracy"] for l in levels]
            stability = 1.0 - np.std(accuracies) / np.mean(accuracies)
            stabilities.append(stability)
        avg_stability = np.mean(stabilities)
        
        # Overall robustness score
        robustness_score = np.mean([
            avg_adv_accuracy / clean_accuracy,
            1.0 - avg_attack_success,
            avg_consistency,
            ood_results["auroc"],
            contrast_results["contrast_accuracy"] / clean_accuracy,
            noise_results["tolerance_score"]
        ])
        
        return RobustnessMetrics(
            clean_accuracy=clean_accuracy,
            perturbed_accuracy=avg_adv_accuracy,
            robustness_score=robustness_score,
            attack_success_rate=avg_attack_success,
            confidence_drop=avg_confidence_drop,
            consistency_score=avg_consistency,
            ood_detection_auroc=ood_results["auroc"],
            noise_tolerance=noise_results["tolerance_score"],
            contrast_set_accuracy=contrast_results["contrast_accuracy"],
            prediction_stability=avg_stability
        )
    
    def _generate_comparison(self) -> Dict[str, Any]:
        """
        Generate model comparison report.
        
        Returns:
            Comparison metrics
        """
        comparison = {
            "most_robust": None,
            "best_adversarial": None,
            "best_ood_detection": None,
            "rankings": {}
        }
        
        if not self.results:
            return comparison
        
        # Collect metrics for comparison
        model_scores = {}
        
        for model_name, results in self.results.items():
            model_scores[model_name] = {
                "robustness_score": results["metrics"]["robustness_score"],
                "adversarial_accuracy": results["metrics"]["perturbed_accuracy"],
                "ood_auroc": results["metrics"]["ood_detection_auroc"]
            }
        
        # Find best performers
        comparison["most_robust"] = max(
            model_scores.items(),
            key=lambda x: x[1]["robustness_score"]
        )[0]
        
        comparison["best_adversarial"] = max(
            model_scores.items(),
            key=lambda x: x[1]["adversarial_accuracy"]
        )[0]
        
        comparison["best_ood_detection"] = max(
            model_scores.items(),
            key=lambda x: x[1]["ood_auroc"]
        )[0]
        
        # Create rankings
        comparison["rankings"]["by_robustness"] = sorted(
            model_scores.keys(),
            key=lambda x: model_scores[x]["robustness_score"],
            reverse=True
        )
        
        comparison["rankings"]["by_adversarial"] = sorted(
            model_scores.keys(),
            key=lambda x: model_scores[x]["adversarial_accuracy"],
            reverse=True
        )
        
        return comparison
    
    def _calculate_robustness_scores(self) -> Dict[str, float]:
        """
        Calculate overall robustness scores for each model.
        
        Returns:
            Robustness scores
        """
        scores = {}
        
        for model_name, results in self.results.items():
            scores[model_name] = results["metrics"]["robustness_score"]
        
        return scores
    
    def _create_summary(self) -> Dict[str, Any]:
        """
        Create benchmark summary.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "num_models": len(self.models),
            "num_samples": self.num_samples,
            "attack_types": self.attack_types,
            "best_model": None,
            "average_robustness": 0.0,
            "key_findings": []
        }
        
        if self.results:
            # Find best model
            robustness_scores = self._calculate_robustness_scores()
            summary["best_model"] = max(
                robustness_scores.items(),
                key=lambda x: x[1]
            )[0]
            
            summary["average_robustness"] = np.mean(
                list(robustness_scores.values())
            )
            
            # Generate key findings
            for model_name, results in self.results.items():
                metrics = results["metrics"]
                
                if metrics["robustness_score"] > 0.8:
                    summary["key_findings"].append(
                        f"{model_name} shows excellent robustness (score: {metrics['robustness_score']:.3f})"
                    )
                
                if metrics["ood_detection_auroc"] > 0.9:
                    summary["key_findings"].append(
                        f"{model_name} has strong OOD detection (AUROC: {metrics['ood_detection_auroc']:.3f})"
                    )
        
        return summary
    
    def _load_dataset(self) -> Dict[str, Any]:
        """Load dataset for benchmarking."""
        dataset = AGNewsDataset()
        return dataset.load_splits()
    
    def _select_test_subset(
        self,
        dataset: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Select subset of test data."""
        indices = np.random.choice(
            len(dataset["test"]["texts"]),
            size=min(self.num_samples, len(dataset["test"]["texts"])),
            replace=False
        )
        
        return {
            "texts": [dataset["test"]["texts"][i] for i in indices],
            "labels": dataset["test"]["labels"][indices]
        }
    
    def _generate_contrast_sets(
        self,
        test_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate contrast sets."""
        return self.contrast_generator.generate(
            test_data["texts"],
            test_data["labels"]
        )
    
    def _load_model(self, model_name: str) -> Any:
        """Load model for benchmarking."""
        return self.factory.create_model(model_name)
    
    def _get_predictions(
        self,
        model: Any,
        data: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get model predictions."""
        model.eval()
        
        # Implementation depends on model interface
        predictions = model.predict(data["texts"])
        scores = model.predict_proba(data["texts"])
        
        return predictions, scores
    
    def _calculate_accuracy(
        self,
        predictions: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Calculate accuracy."""
        return np.mean(predictions == labels)
    
    def save_results(self, filepath: str):
        """
        Save benchmark results to file.
        
        Args:
            filepath: Path to save results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = self._make_serializable(self.results)
        
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays to lists for JSON serialization."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
