"""
Constants and default values for AG News Text Classification Framework.

Centralizes all constants used throughout the project for consistency
and easy configuration management.
"""

from enum import Enum, auto
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# ============================================================================
# Project Information
# ============================================================================

PROJECT_NAME = "ag-news-text-classification"
PROJECT_VERSION = "1.0.0"
PROJECT_AUTHOR = "Võ Hải Dũng"
PROJECT_EMAIL = "vohaidung.work@gmail.com"
PROJECT_URL = "https://github.com/VoHaiDung/ag-news-text-classification"

# ============================================================================
# Dataset Constants
# ============================================================================

# AG News specific
AG_NEWS_CLASSES = ["World", "Sports", "Business", "Sci/Tech"]
AG_NEWS_NUM_CLASSES = 4
AG_NEWS_TRAIN_SIZE = 120000
AG_NEWS_TEST_SIZE = 7600
AG_NEWS_MAX_LENGTH = 512

# Label mappings
LABEL_TO_ID = {label: i for i, label in enumerate(AG_NEWS_CLASSES)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

# Data splits
DEFAULT_SPLIT_RATIOS = {
    "train": 0.8,
    "validation": 0.1,
    "test": 0.1
}

# ============================================================================
# Model Constants
# ============================================================================

# Supported model architectures
SUPPORTED_MODELS = {
    "transformers": [
        "deberta-v3-xlarge",
        "deberta-v3-large",
        "deberta-v3-base",
        "roberta-large",
        "roberta-base",
        "xlnet-large-cased",
        "xlnet-base-cased",
        "electra-large-discriminator",
        "electra-base-discriminator",
        "longformer-base-4096",
        "longformer-large-4096",
        "albert-xxlarge-v2",
        "albert-xlarge-v2",
    ],
    "generative": [
        "gpt2-large",
        "gpt2-medium",
        "t5-large",
        "t5-base",
        "bart-large",
        "bart-base",
    ],
    "efficient": [
        "distilbert-base-uncased",
        "distilroberta-base",
        "tinybert-6l-768d",
        "mobilebert-uncased",
    ],
    "classical": [
        "naive_bayes",
        "svm",
        "logistic_regression",
        "random_forest",
        "xgboost",
        "lightgbm",
        "catboost",
    ]
}

# Pretrained model mappings
PRETRAINED_MODEL_MAPPINGS = {
    "deberta-v3-xlarge": "microsoft/deberta-v3-xlarge",
    "deberta-v3-large": "microsoft/deberta-v3-large",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "roberta-large": "roberta-large",
    "roberta-base": "roberta-base",
    "xlnet-large-cased": "xlnet-large-cased",
    "xlnet-base-cased": "xlnet-base-cased",
    "electra-large-discriminator": "google/electra-large-discriminator",
    "electra-base-discriminator": "google/electra-base-discriminator",
    "longformer-base-4096": "allenai/longformer-base-4096",
    "longformer-large-4096": "allenai/longformer-large-4096",
    "albert-xxlarge-v2": "albert-xxlarge-v2",
    "albert-xlarge-v2": "albert-xlarge-v2",
    "gpt2-large": "gpt2-large",
    "gpt2-medium": "gpt2-medium",
    "t5-large": "t5-large",
    "t5-base": "t5-base",
    "bart-large": "facebook/bart-large",
    "bart-base": "facebook/bart-base",
    "distilbert-base-uncased": "distilbert-base-uncased",
    "distilroberta-base": "distilroberta-base",
    "tinybert-6l-768d": "huawei-noah/TinyBERT_General_6L_768D",
    "mobilebert-uncased": "google/mobilebert-uncased",
}

# Model size configurations
MODEL_SIZES = {
    "small": {
        "hidden_size": 256,
        "num_hidden_layers": 6,
        "num_attention_heads": 4,
        "intermediate_size": 1024,
    },
    "base": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
    "large": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
    },
    "xlarge": {
        "hidden_size": 1536,
        "num_hidden_layers": 48,
        "num_attention_heads": 24,
        "intermediate_size": 6144,
    },
}

# ============================================================================
# Training Constants
# ============================================================================

# Default hyperparameters
DEFAULT_HYPERPARAMETERS = {
    "learning_rate": 2e-5,
    "batch_size": 32,
    "num_epochs": 10,
    "warmup_ratio": 0.1,
    "warmup_steps": 500,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.0,
    "dropout_rate": 0.1,
}

# Learning rate schedulers
SUPPORTED_SCHEDULERS = [
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
    "constant",
    "constant_with_warmup",
    "inverse_sqrt",
    "reduce_on_plateau",
]

# Optimizers
SUPPORTED_OPTIMIZERS = [
    "adam",
    "adamw",
    "sgd",
    "adagrad",
    "adadelta",
    "adamax",
    "nadam",
    "radam",
    "lamb",
    "rmsprop",
]

# Loss functions
SUPPORTED_LOSSES = [
    "cross_entropy",
    "focal_loss",
    "label_smoothing_cross_entropy",
    "dice_loss",
    "f1_loss",
    "weighted_cross_entropy",
]

# ============================================================================
# Evaluation Constants
# ============================================================================

# Metrics
PRIMARY_METRIC = "f1_macro"
SUPPORTED_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "f1_macro",
    "f1_micro",
    "f1_weighted",
    "roc_auc",
    "pr_auc",
    "matthews_corrcoef",
    "cohen_kappa",
    "confusion_matrix",
    "classification_report",
]

# Evaluation settings
EVALUATION_BATCH_SIZE = 64
EVAL_ACCUMULATION_STEPS = 1
COMPUTE_METRICS_INTERVAL = 500

# ============================================================================
# Data Processing Constants
# ============================================================================

# Text preprocessing
MAX_SEQUENCE_LENGTH = 512
TRUNCATION_STRATEGY = "longest_first"  # Options: longest_first, only_first, only_second, do_not_truncate
PADDING_STRATEGY = "max_length"  # Options: max_length, longest, do_not_pad

# Data augmentation
AUGMENTATION_TECHNIQUES = [
    "synonym_replacement",
    "random_insertion",
    "random_swap",
    "random_deletion",
    "back_translation",
    "paraphrase",
    "token_masking",
    "mixup",
    "cutmix",
    "adversarial",
]

# Augmentation parameters
AUGMENTATION_PARAMS = {
    "synonym_replacement": {"probability": 0.1, "num_replacements": 3},
    "random_insertion": {"probability": 0.1, "num_insertions": 3},
    "random_swap": {"probability": 0.1, "num_swaps": 3},
    "random_deletion": {"probability": 0.1, "max_deletions": 3},
    "back_translation": {"languages": ["de", "fr", "es", "zh"]},
    "paraphrase": {"model": "tuner007/pegasus_paraphrase", "num_paraphrases": 2},
    "token_masking": {"mask_probability": 0.15},
    "mixup": {"alpha": 0.2},
    "cutmix": {"alpha": 1.0},
}

# ============================================================================
# Ensemble Constants
# ============================================================================

# Ensemble methods
ENSEMBLE_METHODS = [
    "voting",
    "stacking",
    "blending",
    "bayesian",
    "boosting",
    "snapshot",
]

# Ensemble configurations
ENSEMBLE_CONFIGS = {
    "voting": {
        "voting_type": "soft",  # Options: hard, soft
        "weights": None,  # None for equal weights
    },
    "stacking": {
        "meta_learner": "xgboost",
        "cv_folds": 5,
        "use_probabilities": True,
    },
    "blending": {
        "blend_ratio": 0.2,
        "optimization_metric": "f1_macro",
    },
}

# ============================================================================
# Optimization Constants
# ============================================================================

# Quantization
QUANTIZATION_CONFIGS = {
    "int8": {"bits": 8, "symmetric": True},
    "int4": {"bits": 4, "symmetric": False},
    "dynamic": {"backend": "qnnpack"},
}

# Pruning
PRUNING_CONFIGS = {
    "magnitude": {"sparsity": 0.5, "structured": False},
    "structured": {"sparsity": 0.3, "dimension": 0},
    "lottery_ticket": {"iterations": 5, "reset_weights": True},
}

# LoRA/PEFT
LORA_CONFIGS = {
    "rank": 8,
    "alpha": 16,
    "dropout": 0.1,
    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
}

# ============================================================================
# API Constants
# ============================================================================

# API settings
API_VERSION = "v1"
API_PREFIX = f"/api/{API_VERSION}"
MAX_REQUEST_SIZE = 10 * 1024 * 1024  # 10MB
REQUEST_TIMEOUT = 30  # seconds
RATE_LIMIT = 100  # requests per minute

# Response codes
HTTP_STATUS_CODES = {
    "SUCCESS": 200,
    "CREATED": 201,
    "ACCEPTED": 202,
    "NO_CONTENT": 204,
    "BAD_REQUEST": 400,
    "UNAUTHORIZED": 401,
    "FORBIDDEN": 403,
    "NOT_FOUND": 404,
    "METHOD_NOT_ALLOWED": 405,
    "CONFLICT": 409,
    "UNPROCESSABLE_ENTITY": 422,
    "TOO_MANY_REQUESTS": 429,
    "INTERNAL_SERVER_ERROR": 500,
    "NOT_IMPLEMENTED": 501,
    "SERVICE_UNAVAILABLE": 503,
}

# ============================================================================
# File System Constants
# ============================================================================

# Directory structure
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "outputs" / "models"
LOGS_DIR = ROOT_DIR / "outputs" / "logs"
CACHE_DIR = ROOT_DIR / ".cache"
TEMP_DIR = ROOT_DIR / ".tmp"

# File extensions
SUPPORTED_DATA_FORMATS = [".csv", ".json", ".jsonl", ".txt", ".parquet", ".arrow"]
SUPPORTED_MODEL_FORMATS = [".pt", ".pth", ".bin", ".safetensors", ".onnx"]
SUPPORTED_CONFIG_FORMATS = [".yaml", ".yml", ".json", ".toml"]

# ============================================================================
# Hardware Constants
# ============================================================================

# GPU settings
DEFAULT_GPU_MEMORY_FRACTION = 0.9
CUDA_DEVICE_ORDER = "PCI_BUS_ID"
MIXED_PRECISION_BACKENDS = ["apex", "native"]

# CPU settings
DEFAULT_NUM_WORKERS = 4
DEFAULT_NUM_THREADS = 8

# Memory limits
MAX_MEMORY_GB = 32
CACHE_SIZE_GB = 4

# ============================================================================
# Experiment Constants
# ============================================================================

# Experiment tracking
SUPPORTED_TRACKERS = ["wandb", "mlflow", "tensorboard", "neptune", "comet"]
DEFAULT_TRACKER = "wandb"

# Experiment phases
EXPERIMENT_PHASES = [
    "baseline",
    "exploration",
    "optimization",
    "ablation",
    "production",
]

# Random seeds for reproducibility
RANDOM_SEEDS = [42, 123, 456, 789, 2024]

# ============================================================================
# Research Constants
# ============================================================================

# Research milestones
ACCURACY_MILESTONES = {
    "baseline": 0.85,
    "good": 0.90,
    "very_good": 0.93,
    "excellent": 0.95,
    "sota": 0.96,
}

# Paper references
KEY_PAPERS = {
    "deberta": "https://arxiv.org/abs/2006.03654",
    "roberta": "https://arxiv.org/abs/1907.11692",
    "xlnet": "https://arxiv.org/abs/1906.08237",
    "electra": "https://arxiv.org/abs/2003.10555",
    "longformer": "https://arxiv.org/abs/2004.05150",
}

# ============================================================================
# Prompt Templates
# ============================================================================

# Classification prompts
CLASSIFICATION_PROMPTS = {
    "zero_shot": "Classify the following news article into one of four categories: {classes}.\n\nArticle: {text}\n\nCategory:",
    "few_shot": "Here are some examples of news classification:\n\n{examples}\n\nNow classify this article:\n\nArticle: {text}\n\nCategory:",
    "cot": "Let's classify this news article step by step.\n\nArticle: {text}\n\nFirst, identify the main topic. Then determine which category it belongs to: {classes}.\n\nReasoning:",
    "instruction": "You are a news classifier. Your task is to accurately categorize news articles into one of the following categories: {classes}. Be precise and consider the main theme of the article.\n\nArticle: {text}\n\nCategory:",
}

# ============================================================================
# Error Messages
# ============================================================================

ERROR_MESSAGES = {
    "invalid_input": "Invalid input format. Expected {expected}, got {actual}.",
    "model_not_found": "Model '{model}' not found. Available models: {available}.",
    "config_error": "Configuration error: {message}",
    "training_failed": "Training failed: {message}",
    "evaluation_failed": "Evaluation failed: {message}",
    "api_error": "API error: {message}",
    "data_error": "Data error: {message}",
    "resource_error": "Resource error: {message}",
}

# ============================================================================
# Environment Variables
# ============================================================================

REQUIRED_ENV_VARS = [
    "CUDA_VISIBLE_DEVICES",
    "PYTHONPATH",
    "PROJECT_NAME",
]

OPTIONAL_ENV_VARS = [
    "WANDB_API_KEY",
    "OPENAI_API_KEY",
    "HF_TOKEN",
    "MLFLOW_TRACKING_URI",
    "NEPTUNE_API_TOKEN",
    "COMET_API_KEY",
]

# Export all constants
__all__ = [
    # Project
    "PROJECT_NAME",
    "PROJECT_VERSION",
    "PROJECT_AUTHOR",
    "PROJECT_EMAIL",
    "PROJECT_URL",
    # Dataset
    "AG_NEWS_CLASSES",
    "AG_NEWS_NUM_CLASSES",
    "LABEL_TO_ID",
    "ID_TO_LABEL",
    "DEFAULT_SPLIT_RATIOS",
    # Models
    "SUPPORTED_MODELS",
    "PRETRAINED_MODEL_MAPPINGS",
    "MODEL_SIZES",
    # Training
    "DEFAULT_HYPERPARAMETERS",
    "SUPPORTED_SCHEDULERS",
    "SUPPORTED_OPTIMIZERS",
    "SUPPORTED_LOSSES",
    # Evaluation
    "PRIMARY_METRIC",
    "SUPPORTED_METRICS",
    # Data processing
    "MAX_SEQUENCE_LENGTH",
    "AUGMENTATION_TECHNIQUES",
    "AUGMENTATION_PARAMS",
    # Ensemble
    "ENSEMBLE_METHODS",
    "ENSEMBLE_CONFIGS",
    # Optimization
    "QUANTIZATION_CONFIGS",
    "PRUNING_CONFIGS",
    "LORA_CONFIGS",
    # API
    "API_VERSION",
    "API_PREFIX",
    "HTTP_STATUS_CODES",
    # File system
    "ROOT_DIR",
    "DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    # Research
    "ACCURACY_MILESTONES",
    "KEY_PAPERS",
    # Prompts
    "CLASSIFICATION_PROMPTS",
    # Errors
    "ERROR_MESSAGES",
]
