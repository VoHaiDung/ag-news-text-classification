"""
Sample Data Fixtures for Testing
=================================

This module provides sample data fixtures for testing the AG News Text Classification
framework following best practices from:
- Fowler, M. (2006): "Continuous Integration: Improving Software Quality and Reducing Risk"
- Beck, K. (2003): "Test Driven Development: By Example"
- Martin, R. C. (2008): "Clean Code: A Handbook of Agile Software Craftsmanship"

The fixtures follow the principles of:
1. Deterministic data generation for reproducible tests
2. Representative samples covering edge cases
3. Minimal but sufficient data for comprehensive testing

Mathematical Foundation:
The sample distribution follows a stratified sampling approach where each class
is represented proportionally: P(class_i) = n_i / N, ensuring balanced testing.

Author: Võ Hải Dũng
Email: vohaidung.work@gmail.com
License: MIT
"""

import random
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
from datetime import datetime

# Add project root to path
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from configs.constants import (
    AG_NEWS_CLASSES,
    AG_NEWS_NUM_CLASSES,
    LABEL_TO_ID,
    ID_TO_LABEL,
    MAX_SEQUENCE_LENGTH
)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# Sample Text Data
# ============================================================================

def get_sample_texts() -> Dict[str, List[str]]:
    """
    Generate sample texts for each AG News category.
    
    Following the data characteristics observed in:
    - Zhang et al. (2015): "Character-level Convolutional Networks for Text Classification"
    
    Returns:
        Dictionary mapping category names to lists of sample texts
    """
    sample_texts = {
        "World": [
            "The United Nations Security Council met today to discuss the ongoing crisis in the Middle East region.",
            "European leaders gathered in Brussels for an emergency summit on climate change policies.",
            "China announced new trade agreements with several African nations during the bilateral conference.",
            "The International Court of Justice ruled on the territorial dispute between the two nations.",
            "Diplomatic tensions rise as ambassadors are recalled following the controversial policy announcement."
        ],
        "Sports": [
            "The championship game ended with a dramatic overtime victory for the home team last night.",
            "Olympic athletes begin intensive training camps ahead of the upcoming summer games in Paris.",
            "The tennis star announced retirement after winning a record-breaking 25th Grand Slam title.",
            "Football league officials introduced new regulations to improve player safety during matches.",
            "The basketball team secured their playoff position with a convincing win over rivals."
        ],
        "Business": [
            "Tech giant announces quarterly earnings that exceeded Wall Street expectations by 15 percent.",
            "The Federal Reserve decided to maintain interest rates amid concerns about inflation trends.",
            "Major merger between two pharmaceutical companies creates industry's largest corporation.",
            "Stock markets rallied following positive employment data released by the government today.",
            "Startup company raises $100 million in Series C funding round led by venture capital firms."
        ],
        "Sci/Tech": [
            "Scientists discover a new exoplanet potentially capable of supporting life in nearby star system.",
            "Artificial intelligence breakthrough enables more accurate medical diagnosis predictions.",
            "Quantum computing researchers achieve significant milestone in error correction algorithms.",
            "New smartphone features advanced camera technology with improved low-light performance.",
            "Cybersecurity experts warn about sophisticated new malware targeting financial institutions."
        ]
    }
    return sample_texts

def get_edge_case_texts() -> List[str]:
    """
    Generate edge case texts for robust testing.
    
    Following testing practices from:
    - Beizer, B. (1990): "Software Testing Techniques"
    
    Returns:
        List of edge case texts
    """
    edge_cases = [
        "",  # Empty text
        " ",  # Only whitespace
        "a",  # Single character
        "Word",  # Single word
        "Two words",  # Very short text
        "." * 10,  # Only punctuation
        "123456789",  # Only numbers
        "HTTP://WWW.EXAMPLE.COM",  # URL in uppercase
        "test@example.com is an email",  # Contains email
        "Special chars: @#$%^&*()",  # Special characters
        "\n\n\n",  # Only newlines
        "\t\t\t",  # Only tabs
        "MiXeD cAsE tExT",  # Mixed case
        "Very " + "long " * 500 + "text",  # Very long text
        "Repeated " * 100,  # Repeated words
        "Multi\nline\ntext\nwith\nbreaks",  # Multiline
        "Text with     multiple    spaces",  # Multiple spaces
        "Text with non-ASCII characters",  # Non-ASCII placeholder
        "Text with unicode U+1F600 characters",  # Unicode reference without actual emoji
        "Text\x00with\x00null\x00bytes"  # Contains null bytes
    ]
    return edge_cases

# ============================================================================
# Sample Label Data
# ============================================================================

def get_sample_labels(n_samples: int = 100, balanced: bool = True) -> List[int]:
    """
    Generate sample labels for testing.
    
    Following statistical sampling from:
    - Cochran, W. G. (1977): "Sampling Techniques"
    
    Args:
        n_samples: Number of samples to generate
        balanced: Whether to generate balanced distribution
        
    Returns:
        List of label indices
    """
    if balanced:
        # Generate balanced distribution
        labels_per_class = n_samples // AG_NEWS_NUM_CLASSES
        remainder = n_samples % AG_NEWS_NUM_CLASSES
        
        labels = []
        for i in range(AG_NEWS_NUM_CLASSES):
            count = labels_per_class + (1 if i < remainder else 0)
            labels.extend([i] * count)
        
        random.shuffle(labels)
    else:
        # Generate imbalanced distribution following power law
        # P(class_i) ∝ 1/i^α where α = 1.5
        weights = [1.0 / (i + 1) ** 1.5 for i in range(AG_NEWS_NUM_CLASSES)]
        weights = np.array(weights) / sum(weights)
        labels = np.random.choice(AG_NEWS_NUM_CLASSES, size=n_samples, p=weights).tolist()
    
    return labels

# ============================================================================
# Sample Dataset
# ============================================================================

@dataclass
class SampleDataset:
    """
    Sample dataset for testing.
    
    Following dataset design from:
    - Goodfellow et al. (2016): "Deep Learning", Chapter 8: Optimization for Training
    """
    texts: List[str] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)
    label_names: List[str] = field(default_factory=list)
    split: str = "train"
    
    def __post_init__(self):
        """Initialize label names from labels."""
        if self.labels and not self.label_names:
            self.label_names = [ID_TO_LABEL[label] for label in self.labels]
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index."""
        return {
            'text': self.texts[idx],
            'label': self.labels[idx],
            'label_name': self.label_names[idx],
            'idx': idx
        }

def create_sample_dataset(
    n_samples: int = 100,
    split: str = "train",
    include_edge_cases: bool = False
) -> SampleDataset:
    """
    Create a sample dataset for testing.
    
    Args:
        n_samples: Number of samples
        split: Dataset split name
        include_edge_cases: Whether to include edge cases
        
    Returns:
        SampleDataset instance
    """
    sample_texts_dict = get_sample_texts()
    
    texts = []
    labels = []
    
    if include_edge_cases:
        # Add edge cases
        edge_texts = get_edge_case_texts()
        texts.extend(edge_texts)
        # Assign random labels to edge cases
        labels.extend([random.randint(0, AG_NEWS_NUM_CLASSES - 1) for _ in edge_texts])
    
    # Add regular samples
    remaining = n_samples - len(texts)
    samples_per_class = remaining // AG_NEWS_NUM_CLASSES
    
    for class_idx, class_name in enumerate(AG_NEWS_CLASSES):
        class_texts = sample_texts_dict[class_name]
        for _ in range(samples_per_class):
            # Randomly select and possibly modify text
            text = random.choice(class_texts)
            if random.random() < 0.3:  # 30% chance of modification
                text = modify_text(text)
            texts.append(text)
            labels.append(class_idx)
    
    # Shuffle to avoid ordering bias
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined) if combined else ([], [])
    
    return SampleDataset(
        texts=list(texts),
        labels=list(labels),
        split=split
    )

def modify_text(text: str) -> str:
    """
    Apply random modifications to text for variety.
    
    Following augmentation strategies from:
    - Wei & Zou (2019): "EDA: Easy Data Augmentation Techniques"
    
    Args:
        text: Input text
        
    Returns:
        Modified text
    """
    modifications = [
        lambda t: t.upper(),  # Uppercase
        lambda t: t.lower(),  # Lowercase
        lambda t: t + ".",  # Add punctuation
        lambda t: t.replace(".", ""),  # Remove punctuation
        lambda t: " ".join(t.split()[:10]),  # Truncate
        lambda t: t + " " + t,  # Duplicate
        lambda t: " ".join(reversed(t.split())),  # Reverse words
        lambda t: t.replace(" ", "  "),  # Double spaces
    ]
    
    if random.random() < 0.5:
        modification = random.choice(modifications)
        text = modification(text)
    
    return text

# ============================================================================
# Sample Tokenized Data
# ============================================================================

def get_sample_tokenized_data(
    batch_size: int = 4,
    seq_length: int = 128
) -> Dict[str, torch.Tensor]:
    """
    Generate sample tokenized data for model testing.
    
    Following tokenization patterns from:
    - Devlin et al. (2019): "BERT: Pre-training of Deep Bidirectional Transformers"
    
    Args:
        batch_size: Batch size
        seq_length: Sequence length
        
    Returns:
        Dictionary of tensors
    """
    # Generate random token IDs (vocabulary size ~30000 for BERT-like models)
    vocab_size = 30000
    
    # Input IDs with realistic distribution
    # Most tokens in range [100, 10000], special tokens [0, 100]
    input_ids = torch.randint(100, 10000, (batch_size, seq_length))
    
    # Add special tokens
    input_ids[:, 0] = 101  # [CLS] token
    input_ids[:, -1] = 102  # [SEP] token
    
    # Attention mask (1 for real tokens, 0 for padding)
    # Simulate variable length sequences
    actual_lengths = torch.randint(seq_length // 2, seq_length, (batch_size,))
    attention_mask = torch.zeros(batch_size, seq_length, dtype=torch.long)
    for i, length in enumerate(actual_lengths):
        attention_mask[i, :length] = 1
    
    # Token type IDs (for BERT-style models)
    token_type_ids = torch.zeros(batch_size, seq_length, dtype=torch.long)
    
    # Labels
    labels = torch.randint(0, AG_NEWS_NUM_CLASSES, (batch_size,))
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels
    }

# ============================================================================
# Sample Model Outputs
# ============================================================================

def get_sample_model_outputs(
    batch_size: int = 4,
    num_classes: int = AG_NEWS_NUM_CLASSES,
    include_hidden_states: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Generate sample model outputs for testing.
    
    Following output patterns from:
    - Vaswani et al. (2017): "Attention is All You Need"
    
    Args:
        batch_size: Batch size
        num_classes: Number of classes
        include_hidden_states: Whether to include hidden states
        
    Returns:
        Dictionary of output tensors
    """
    # Logits (unnormalized scores)
    logits = torch.randn(batch_size, num_classes)
    
    # Make one class more likely for each sample
    for i in range(batch_size):
        dominant_class = i % num_classes
        logits[i, dominant_class] += 2.0
    
    # Probabilities (softmax of logits)
    probs = torch.softmax(logits, dim=-1)
    
    # Predictions
    predictions = torch.argmax(logits, dim=-1)
    
    outputs = {
        'logits': logits,
        'probs': probs,
        'predictions': predictions
    }
    
    if include_hidden_states:
        # Hidden states from transformer layers
        hidden_size = 768  # BERT-base size
        num_layers = 12
        seq_length = 128
        
        hidden_states = []
        for _ in range(num_layers):
            layer_output = torch.randn(batch_size, seq_length, hidden_size)
            hidden_states.append(layer_output)
        
        outputs['hidden_states'] = hidden_states
        
        # Attention weights
        num_heads = 12
        attention_weights = []
        for _ in range(num_layers):
            layer_attention = torch.randn(batch_size, num_heads, seq_length, seq_length)
            # Make attention weights sum to 1
            layer_attention = torch.softmax(layer_attention, dim=-1)
            attention_weights.append(layer_attention)
        
        outputs['attentions'] = attention_weights
    
    return outputs

# ============================================================================
# Sample Configuration
# ============================================================================

def get_sample_config() -> Dict[str, Any]:
    """
    Generate sample configuration for testing.
    
    Following configuration patterns from:
    - Paszke et al. (2019): "PyTorch: An Imperative Style, High-Performance Deep Learning Library"
    
    Returns:
        Configuration dictionary
    """
    config = {
        'model': {
            'name': 'deberta-v3-base',
            'num_labels': AG_NEWS_NUM_CLASSES,
            'hidden_size': 768,
            'num_hidden_layers': 12,
            'num_attention_heads': 12,
            'intermediate_size': 3072,
            'hidden_dropout_prob': 0.1,
            'attention_probs_dropout_prob': 0.1,
            'max_position_embeddings': 512,
            'type_vocab_size': 0,
            'initializer_range': 0.02
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 2e-5,
            'num_epochs': 3,
            'warmup_steps': 500,
            'weight_decay': 0.01,
            'adam_epsilon': 1e-8,
            'max_grad_norm': 1.0,
            'fp16': False,
            'gradient_accumulation_steps': 1
        },
        'data': {
            'max_seq_length': MAX_SEQUENCE_LENGTH,
            'pad_to_max_length': True,
            'do_lower_case': False,
            'use_fast_tokenizer': True
        },
        'evaluation': {
            'eval_batch_size': 32,
            'eval_steps': 500,
            'metrics': ['accuracy', 'f1_macro', 'precision', 'recall'],
            'save_best_model': True,
            'early_stopping_patience': 3
        },
        'experiment': {
            'seed': SEED,
            'output_dir': 'outputs/experiments/test',
            'logging_dir': 'outputs/logs/test',
            'save_steps': 1000,
            'save_total_limit': 3,
            'load_best_model_at_end': True
        }
    }
    return config

# ============================================================================
# Sample Augmented Data
# ============================================================================

def get_sample_augmented_data() -> List[Tuple[str, str, int]]:
    """
    Generate sample augmented data for testing.
    
    Following augmentation strategies from:
    - Kobayashi (2018): "Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations"
    
    Returns:
        List of (original_text, augmented_text, label) tuples
    """
    augmented_samples = [
        (
            "The United Nations Security Council met today.",
            "The UN Security Council convened today.",
            0  # World
        ),
        (
            "The championship game ended with a victory.",
            "The championship match concluded with a win.",
            1  # Sports
        ),
        (
            "Tech giant announces quarterly earnings.",
            "Technology company reveals quarterly profits.",
            2  # Business
        ),
        (
            "Scientists discover a new exoplanet.",
            "Researchers find a new planet outside our solar system.",
            3  # Sci/Tech
        ),
        (
            "Stock markets rallied following positive data.",
            "Stock markets following positive data rallied.",  # Word order change
            2  # Business
        ),
        (
            "Olympic athletes begin training camps.",
            "Olympic athletes BEGIN training camps.",  # Case change
            1  # Sports
        )
    ]
    return augmented_samples

# ============================================================================
# Sample Contrast Sets
# ============================================================================

def get_sample_contrast_sets() -> List[Dict[str, Any]]:
    """
    Generate sample contrast sets for testing.
    
    Following contrast set methodology from:
    - Gardner et al. (2020): "Evaluating Models' Local Decision Boundaries via Contrast Sets"
    
    Returns:
        List of contrast set dictionaries
    """
    contrast_sets = [
        {
            'original_text': "The United Nations Security Council met to discuss Middle East peace.",
            'original_label': 0,  # World
            'contrast_text': "The National Basketball Association council met to discuss player trades.",
            'contrast_label': 1,  # Sports
            'perturbation_type': 'entity_replacement'
        },
        {
            'original_text': "Scientists at MIT developed a new artificial intelligence algorithm.",
            'original_label': 3,  # Sci/Tech
            'contrast_text': "Executives at MIT developed a new business strategy algorithm.",
            'contrast_label': 2,  # Business
            'perturbation_type': 'context_shift'
        },
        {
            'original_text': "The company's stock price increased by 10 percent yesterday.",
            'original_label': 2,  # Business
            'contrast_text': "The team's score increased by 10 points yesterday.",
            'contrast_label': 1,  # Sports
            'perturbation_type': 'domain_transfer'
        },
        {
            'original_text': "The football team won the championship after overtime.",
            'original_label': 1,  # Sports
            'contrast_text': "The research team won the Nobel prize after years of work.",
            'contrast_label': 3,  # Sci/Tech
            'perturbation_type': 'semantic_shift'
        }
    ]
    return contrast_sets

# ============================================================================
# Sample Metrics
# ============================================================================

def get_sample_metrics() -> Dict[str, float]:
    """
    Generate sample metrics for testing.
    
    Following metric conventions from:
    - Powers (2011): "Evaluation: From Precision, Recall and F-Measure to ROC"
    
    Returns:
        Dictionary of metric values
    """
    metrics = {
        'accuracy': 0.9234,
        'precision': 0.9156,
        'recall': 0.9189,
        'f1': 0.9172,
        'f1_macro': 0.9145,
        'f1_micro': 0.9234,
        'f1_weighted': 0.9233,
        'matthews_corrcoef': 0.8978,
        'cohen_kappa': 0.8912,
        'roc_auc': 0.9823,
        'pr_auc': 0.9756,
        'loss': 0.2341,
        'perplexity': 1.2634,
        'top_k_accuracy': {
            'k_1': 0.9234,
            'k_3': 0.9912,
            'k_5': 0.9987
        },
        'per_class_metrics': {
            'World': {'precision': 0.92, 'recall': 0.91, 'f1': 0.915},
            'Sports': {'precision': 0.93, 'recall': 0.94, 'f1': 0.935},
            'Business': {'precision': 0.90, 'recall': 0.89, 'f1': 0.895},
            'Sci/Tech': {'precision': 0.91, 'recall': 0.93, 'f1': 0.920}
        }
    }
    return metrics

# ============================================================================
# Mock Data Generators
# ============================================================================

class MockDataLoader:
    """
    Mock DataLoader for testing.
    
    Following testing patterns from:
    - Freeman & Pryce (2009): "Growing Object-Oriented Software, Guided by Tests"
    """
    
    def __init__(
        self,
        dataset: Optional[SampleDataset] = None,
        batch_size: int = 4,
        num_batches: int = 10
    ):
        """
        Initialize mock DataLoader.
        
        Args:
            dataset: Dataset to use
            batch_size: Batch size
            num_batches: Number of batches
        """
        self.dataset = dataset or create_sample_dataset(batch_size * num_batches)
        self.batch_size = batch_size
        self.num_batches = num_batches
        self._current_batch = 0
    
    def __iter__(self):
        """Reset iterator."""
        self._current_batch = 0
        return self
    
    def __next__(self) -> Dict[str, torch.Tensor]:
        """Get next batch."""
        if self._current_batch >= self.num_batches:
            raise StopIteration
        
        batch = get_sample_tokenized_data(self.batch_size)
        self._current_batch += 1
        return batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches

def create_mock_model_checkpoint() -> Dict[str, Any]:
    """
    Create mock model checkpoint for testing.
    
    Returns:
        Dictionary representing model checkpoint
    """
    checkpoint = {
        'epoch': 3,
        'global_step': 1500,
        'model_state_dict': {
            'embeddings.weight': torch.randn(30000, 768),
            'encoder.layer.0.attention.self.query.weight': torch.randn(768, 768),
            'encoder.layer.0.attention.self.key.weight': torch.randn(768, 768),
            'encoder.layer.0.attention.self.value.weight': torch.randn(768, 768),
            'classifier.weight': torch.randn(AG_NEWS_NUM_CLASSES, 768),
            'classifier.bias': torch.randn(AG_NEWS_NUM_CLASSES)
        },
        'optimizer_state_dict': {
            'state': {},
            'param_groups': [{'lr': 2e-5, 'weight_decay': 0.01}]
        },
        'scheduler_state_dict': {
            'last_epoch': 3,
            'base_lrs': [2e-5]
        },
        'best_score': 0.9234,
        'metrics': get_sample_metrics(),
        'config': get_sample_config(),
        'timestamp': datetime.now().isoformat()
    }
    return checkpoint

# ============================================================================
# Data Validation Utilities
# ============================================================================

def validate_sample_data(data: Any, expected_type: str) -> bool:
    """
    Validate sample data format.
    
    Following validation practices from:
    - Meyer (1997): "Object-Oriented Software Construction"
    
    Args:
        data: Data to validate
        expected_type: Expected data type
        
    Returns:
        Boolean indicating validity
    """
    validators = {
        'dataset': lambda d: hasattr(d, '__len__') and hasattr(d, '__getitem__'),
        'dataloader': lambda d: hasattr(d, '__iter__') and hasattr(d, '__next__'),
        'tensor': lambda d: isinstance(d, torch.Tensor),
        'config': lambda d: isinstance(d, dict),
        'metrics': lambda d: isinstance(d, dict) and 'accuracy' in d
    }
    
    validator = validators.get(expected_type)
    return validator(data) if validator else False

# ============================================================================
# Export Functions
# ============================================================================

__all__ = [
    # Text data
    'get_sample_texts',
    'get_edge_case_texts',
    'get_sample_labels',
    
    # Dataset
    'SampleDataset',
    'create_sample_dataset',
    'modify_text',
    
    # Tokenized data
    'get_sample_tokenized_data',
    'get_sample_model_outputs',
    
    # Configuration
    'get_sample_config',
    
    # Augmented data
    'get_sample_augmented_data',
    'get_sample_contrast_sets',
    
    # Metrics
    'get_sample_metrics',
    
    # Mock objects
    'MockDataLoader',
    'create_mock_model_checkpoint',
    
    # Validation
    'validate_sample_data'
]
