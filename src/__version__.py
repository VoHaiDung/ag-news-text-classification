"""Version information for AG News Text Classification Framework."""

# Version of the ag-news-text-classification package
__version__ = "1.0.0"

# Detailed version information
__version_info__ = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "prerelease": None,  # Options: "alpha", "beta", "rc", None
    "build": None,  # Build metadata
}

def get_version_string() -> str:
    """
    Generate version string from version info.
    
    Returns:
        str: Formatted version string
    """
    version = f"{__version_info__['major']}.{__version_info__['minor']}.{__version_info__['patch']}"
    
    if __version_info__["prerelease"]:
        version += f"-{__version_info__['prerelease']}"
    
    if __version_info__["build"]:
        version += f"+{__version_info__['build']}"
    
    return version

# Research milestones
MILESTONES = {
    "0.1.0": "Initial prototype with classical ML baselines",
    "0.2.0": "Deep learning models (LSTM, CNN) added",
    "0.3.0": "BERT-based models integrated",
    "0.4.0": "Advanced transformers (RoBERTa, XLNet) added",
    "0.5.0": "DeBERTa-v3 implementation",
    "0.6.0": "Ensemble methods implemented",
    "0.7.0": "Domain-adaptive pretraining added",
    "0.8.0": "Prompt-based learning integrated",
    "0.9.0": "GPT-4 distillation implemented",
    "1.0.0": "Production-ready with SOTA performance (96%+ accuracy)",
}

# Compatibility information
COMPATIBLE_VERSIONS = {
    "python": ">=3.8,<3.12",
    "torch": ">=2.0.0,<2.2.0",
    "transformers": ">=4.35.0,<5.0.0",
    "datasets": ">=2.14.0",
}

# API version for backward compatibility
API_VERSION = "v1"

# Model checkpoint versions
MODEL_VERSIONS = {
    "deberta_v3_xlarge": "1.0.0",
    "roberta_large": "1.0.0",
    "xlnet_large": "1.0.0",
    "ensemble_voting": "1.0.0",
    "ensemble_stacking": "1.0.0",
}
