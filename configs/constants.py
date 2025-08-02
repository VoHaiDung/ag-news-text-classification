"""
Global constants for AG News Text Classification
"""

# Dataset constants
DATASET_NAME = "ag_news"
NUM_CLASSES = 4
CLASS_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business", 
    3: "Sci/Tech"
}

# Model constants
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_NUM_EPOCHS = 3
DEFAULT_WARMUP_RATIO = 0.1
DEFAULT_WEIGHT_DECAY = 0.01

# Training constants
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
EVAL_STEPS = 500
SAVE_STEPS = 1000
LOGGING_STEPS = 100

# Hardware constants
DEFAULT_NUM_WORKERS = 4
MIXED_PRECISION = True
GRADIENT_CHECKPOINTING = True

# Paths
PROJECT_ROOT = "."
DATA_DIR = f"{PROJECT_ROOT}/data"
OUTPUT_DIR = f"{PROJECT_ROOT}/outputs"
CONFIG_DIR = f"{PROJECT_ROOT}/configs"
CACHE_DIR = f"{DATA_DIR}/cache"

# Model names
SUPPORTED_MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large",
    "microsoft/deberta-v3-base",
    "microsoft/deberta-v3-large",
    "microsoft/deberta-v3-xlarge",
    "xlnet-base-cased",
    "xlnet-large-cased",
    "google/electra-base-discriminator",
    "google/electra-large-discriminator",
    "allenai/longformer-base-4096",
    "allenai/longformer-large-4096",
]

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Random seeds
DEFAULT_SEED = 42

# Metrics
PRIMARY_METRIC = "accuracy"
METRICS_FOR_BEST_MODEL = ["accuracy", "f1", "precision", "recall"]

# Early stopping
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001

# Ensemble
MIN_ENSEMBLE_MODELS = 3
MAX_ENSEMBLE_MODELS = 7

# Augmentation
AUGMENTATION_FACTOR = 2  # How many times to augment the dataset
BACK_TRANSLATION_LANGUAGES = ["de", "fr", "es"]
MAX_PARAPHRASES = 2

# DAPT (Domain Adaptive Pre-training)
DAPT_MLM_PROBABILITY = 0.15
DAPT_MAX_STEPS = 100000
DAPT_EVAL_STEPS = 10000

# API
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 4

# Limits
MAX_TEXT_LENGTH = 2048
MAX_BATCH_SIZE = 128
MAX_GRADIENT_ACCUMULATION = 16
