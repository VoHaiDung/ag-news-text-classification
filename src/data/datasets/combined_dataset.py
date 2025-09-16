"""
Combined Dataset Module
=======================

Combines multiple datasets for joint training following:
- Conneau et al. (2020): "Unsupervised Cross-lingual Representation Learning at Scale"
- Aghajanyan et al. (2021): "Muppet: Massive Multi-task Representations with Pre-Finetuning"
- Pfeiffer et al. (2021): "AdapterHub: A Framework for Adapting Transformers"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import random
import numpy as np

import torch
from torch.utils.data import Dataset, ConcatDataset, ChainDataset
from transformers import PreTrainedTokenizer

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.constants import AG_NEWS_CLASSES, AG_NEWS_NUM_CLASSES
from src.utils.logging_config import setup_logging
from src.core.exceptions import DataError
from src.data.datasets.ag_news import AGNewsDataset
from src.data.datasets.external_news import ExternalNewsDataset

logger = setup_logging(__name__)

@dataclass
class CombinedDatasetConfig:
    """
    Configuration for combined dataset.
    
    Following multi-dataset training from:
    - Raffel et al. (2020): "Exploring the Limits of Transfer Learning with T5"
    """
    # Dataset components
    use_ag_news: bool = True
    use_external: bool = True
    use_augmented: bool = False
    use_pseudo_labeled: bool = False
    
    # Mixing strategy
    mixing_strategy: str = "proportional"  # proportional, temperature, equal
    mixing_weights: Dict[str, float] = field(default_factory=lambda: {
        "ag_news": 0.7,
        "external": 0.2,
        "augmented": 0.1
    })
    
    # Sampling
    sampling_temperature: float = 1.0
    oversample_minority: bool = False
    undersample_majority: bool = False
    
    # Interleaving
    interleave_datasets: bool = True
    batch_by_dataset: bool = False
    
    # Label handling
    unified_labels: bool = True
    label_mapping: Optional[Dict[str, Dict[int, int]]] = None
    
    # Maximum samples per dataset
    max_samples_per_dataset: Optional[int] = None
    
    # Random seed
    seed: int = 42

class CombinedDataset(Dataset):
    """
    Combined dataset for multi-source training.
    
    Implements dataset mixing strategies from:
    - Arivazhagan et al. (2019): "Massively Multilingual Neural Machine Translation"
    - Wang et al. (2020): "Balancing Training for Multilingual Neural Machine Translation"
    """
    
    def __init__(
        self,
        config: CombinedDatasetConfig,
        datasets: Optional[Dict[str, Dataset]] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        transform: Optional[Any] = None
    ):
        """
        Initialize combined dataset.
        
        Args:
            config: Combined dataset configuration
            datasets: Pre-loaded datasets dictionary
            tokenizer: Optional tokenizer
            transform: Optional transformations
        """
        self.config = config
        self.tokenizer = tokenizer
        self.transform = transform
        self.rng = random.Random(config.seed)
        
        # Load or use provided datasets
        self.datasets = datasets or self._load_datasets()
        
        # Create dataset indices
        self._create_indices()
        
        # Apply sampling strategies
        if config.oversample_minority or config.undersample_majority:
            self._balance_datasets()
        
        logger.info(f"Created combined dataset with {len(self)} samples from {len(self.datasets)} sources")
    
    def _load_datasets(self) -> Dict[str, Dataset]:
        """Load component datasets."""
        datasets = {}
        
        if self.config.use_ag_news:
            from src.data.datasets.ag_news import AGNewsConfig
            ag_config = AGNewsConfig()
            datasets["ag_news"] = AGNewsDataset(
                config=ag_config,
                split="train",
                tokenizer=self.tokenizer
            )
        
        if self.config.use_external:
            from src.data.datasets.external_news import ExternalNewsConfig
            ext_config = ExternalNewsConfig()
            datasets["external"] = ExternalNewsDataset(
                config=ext_config,
                tokenizer=self.tokenizer,
                purpose="augmentation"
            )
        
        if self.config.use_augmented:
            # Load augmented dataset
            aug_path = PROJECT_ROOT / "data" / "augmented"
            if aug_path.exists():
                # Placeholder for augmented dataset loading
                pass
        
        if self.config.use_pseudo_labeled:
            # Load pseudo-labeled dataset
            pseudo_path = PROJECT_ROOT / "data" / "pseudo_labeled"
            if pseudo_path.exists():
                # Placeholder for pseudo-labeled dataset loading
                pass
        
        return datasets
    
    def _create_indices(self):
        """
        Create indices for sampling from datasets.
        
        Following sampling strategies from:
        - Conneau & Lample (2019): "Cross-lingual Language Model Pretraining"
        """
        self.dataset_indices = {}
        self.flat_indices = []
        
        for dataset_name, dataset in self.datasets.items():
            dataset_size = len(dataset)
            
            # Limit samples if configured
            if self.config.max_samples_per_dataset:
                dataset_size = min(dataset_size, self.config.max_samples_per_dataset)
            
            indices = list(range(dataset_size))
            self.dataset_indices[dataset_name] = indices
            
            # Create flat indices based on mixing strategy
            if self.config.mixing_strategy == "proportional":
                weight = self.config.mixing_weights.get(dataset_name, 1.0)
                num_samples = int(dataset_size * weight)
                
                for _ in range(num_samples):
                    idx = self.rng.choice(indices)
                    self.flat_indices.append((dataset_name, idx))
            
            elif self.config.mixing_strategy == "equal":
                # Equal sampling from each dataset
                for idx in indices:
                    self.flat_indices.append((dataset_name, idx))
            
            elif self.config.mixing_strategy == "temperature":
                # Temperature-based sampling
                weight = self.config.mixing_weights.get(dataset_name, 1.0)
                prob = weight ** (1.0 / self.config.sampling_temperature)
                num_samples = int(dataset_size * prob)
                
                for _ in range(num_samples):
                    idx = self.rng.choice(indices)
                    self.flat_indices.append((dataset_name, idx))
        
        # Shuffle if interleaving
        if self.config.interleave_datasets:
            self.rng.shuffle(self.flat_indices)
    
    def _balance_datasets(self):
        """
        Balance datasets through over/undersampling.
        
        Following balancing strategies from:
        - Chawla et al. (2002): "SMOTE: Synthetic Minority Over-sampling Technique"
        """
        # Get class distributions
        class_counts = {}
        
        for dataset_name, dataset in self.datasets.items():
            if hasattr(dataset, 'get_label_distribution'):
                dist = dataset.get_label_distribution()
                for label, count in dist.items():
                    if label not in class_counts:
                        class_counts[label] = 0
                    class_counts[label] += count
        
        if not class_counts:
            return
        
        # Find minority and majority classes
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        
        # Oversample minority
        if self.config.oversample_minority:
            target_count = int((max_count + min_count) / 2)
            
            for class_name, count in class_counts.items():
                if count < target_count:
                    oversample_ratio = target_count / count
                    
                    # Add more samples from this class
                    new_indices = []
                    for dataset_name, idx in self.flat_indices:
                        dataset = self.datasets[dataset_name]
                        
                        if hasattr(dataset, '__getitem__'):
                            item = dataset[idx]
                            if isinstance(item, dict) and 'label' in item:
                                if item['label'] == class_name or str(item['label']) == class_name:
                                    for _ in range(int(oversample_ratio)):
                                        new_indices.append((dataset_name, idx))
                    
                    self.flat_indices.extend(new_indices)
        
        # Undersample majority
        if self.config.undersample_majority:
            target_count = int((max_count + min_count) / 2)
            
            for class_name, count in class_counts.items():
                if count > target_count:
                    keep_ratio = target_count / count
                    
                    # Remove some samples from this class
                    filtered_indices = []
                    for dataset_name, idx in self.flat_indices:
                        dataset = self.datasets[dataset_name]
                        
                        if hasattr(dataset, '__getitem__'):
                            item = dataset[idx]
                            if isinstance(item, dict) and 'label' in item:
                                if item['label'] == class_name or str(item['label']) == class_name:
                                    if self.rng.random() < keep_ratio:
                                        filtered_indices.append((dataset_name, idx))
                                else:
                                    filtered_indices.append((dataset_name, idx))
                    
                    self.flat_indices = filtered_indices
        
        # Re-shuffle
        if self.config.interleave_datasets:
            self.rng.shuffle(self.flat_indices)
    
    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.flat_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with combined data
        """
        dataset_name, dataset_idx = self.flat_indices[idx]
        dataset = self.datasets[dataset_name]
        
        # Get item from specific dataset
        item = dataset[dataset_idx]
        
        # Add dataset identifier
        item["dataset_source"] = dataset_name
        
        # Apply unified label mapping if configured
        if self.config.unified_labels and self.config.label_mapping:
            if dataset_name in self.config.label_mapping and "label" in item:
                mapping = self.config.label_mapping[dataset_name]
                if item["label"] in mapping:
                    item["label"] = mapping[item["label"]]
        
        # Apply transform if provided
        if self.transform:
            item = self.transform(item)
        
        return item
    
    def get_dataset_statistics(self) -> Dict[str, Any]:
        """Get statistics for combined dataset."""
        stats = {
            "total_samples": len(self),
            "num_datasets": len(self.datasets),
            "dataset_names": list(self.datasets.keys()),
            "dataset_sizes": {},
            "dataset_proportions": {},
            "mixing_strategy": self.config.mixing_strategy
        }
        
        # Count samples per dataset
        dataset_counts = {}
        for dataset_name, _ in self.flat_indices:
            if dataset_name not in dataset_counts:
                dataset_counts[dataset_name] = 0
            dataset_counts[dataset_name] += 1
        
        # Calculate proportions
        total = len(self.flat_indices)
        for dataset_name, count in dataset_counts.items():
            stats["dataset_sizes"][dataset_name] = count
            stats["dataset_proportions"][dataset_name] = count / total if total > 0 else 0
        
        # Get individual dataset statistics
        stats["individual_stats"] = {}
        for dataset_name, dataset in self.datasets.items():
            if hasattr(dataset, 'get_statistics'):
                stats["individual_stats"][dataset_name] = dataset.get_statistics()
        
        return stats
    
    def create_subset(
        self,
        fraction: float = 0.1,
        stratified: bool = True
    ) -> 'CombinedDataset':
        """
        Create a subset of the combined dataset.
        
        Args:
            fraction: Fraction of data to include
            stratified: Whether to maintain dataset proportions
            
        Returns:
            New CombinedDataset instance with subset
        """
        n_samples = int(len(self) * fraction)
        
        if stratified:
            # Maintain proportions
            subset_indices = []
            
            for dataset_name in self.datasets.keys():
                dataset_indices = [
                    (name, idx) for name, idx in self.flat_indices 
                    if name == dataset_name
                ]
                
                n_subset = int(len(dataset_indices) * fraction)
                sampled = self.rng.sample(dataset_indices, min(n_subset, len(dataset_indices)))
                subset_indices.extend(sampled)
            
            # Shuffle
            self.rng.shuffle(subset_indices)
        else:
            # Random sampling
            subset_indices = self.rng.sample(self.flat_indices, n_samples)
        
        # Create new dataset with subset
        new_config = self.config
        new_dataset = CombinedDataset(
            config=new_config,
            datasets=self.datasets,
            tokenizer=self.tokenizer,
            transform=self.transform
        )
        new_dataset.flat_indices = subset_indices
        
        return new_dataset

class MultiTaskDataset(CombinedDataset):
    """
    Multi-task dataset for joint training on multiple objectives.
    
    Following multi-task learning from:
    - Liu et al. (2019): "Multi-Task Deep Neural Networks for Natural Language Understanding"
    - Radford et al. (2019): "Language Models are Unsupervised Multitask Learners"
    """
    
    def __init__(
        self,
        config: CombinedDatasetConfig,
        task_configs: Dict[str, Dict[str, Any]],
        tokenizer: Optional[PreTrainedTokenizer] = None
    ):
        """
        Initialize multi-task dataset.
        
        Args:
            config: Combined dataset configuration
            task_configs: Task-specific configurations
            tokenizer: Optional tokenizer
        """
        super().__init__(config, tokenizer=tokenizer)
        self.task_configs = task_configs
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item with task-specific processing."""
        item = super().__getitem__(idx)
        
        # Determine task
        dataset_source = item.get("dataset_source", "default")
        task_config = self.task_configs.get(dataset_source, {})
        
        # Add task identifier
        item["task_id"] = task_config.get("task_id", 0)
        item["task_name"] = task_config.get("task_name", "classification")
        
        # Task-specific processing
        if item["task_name"] == "mlm":
            # Masked language modeling
            item = self._prepare_mlm(item)
        elif item["task_name"] == "nsp":
            # Next sentence prediction
            item = self._prepare_nsp(item)
        
        return item
    
    def _prepare_mlm(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare item for masked language modeling."""
        # Implementation for MLM preparation
        return item
    
    def _prepare_nsp(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare item for next sentence prediction."""
        # Implementation for NSP preparation
        return item
