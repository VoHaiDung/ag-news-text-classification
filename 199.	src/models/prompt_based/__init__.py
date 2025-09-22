"""
Prompt-Based Models Package
===========================

This package contains implementations of prompt-based learning methods for
text classification, including soft prompts, instruction models, and template managers.

The prompt-based paradigm reformulates classification as:
1. Cloze-style completion tasks
2. Natural language generation tasks
3. Instruction-following tasks

References:
- Liu et al. (2023): "Pre-train, Prompt, and Predict: A Systematic Survey"
- Schick & Schütze (2021): "Exploiting Cloze Questions for Few Shot Text Classification"
- Lester et al. (2021): "The Power of Scale for Parameter-Efficient Prompt Tuning"

Author: Võ Hải Dũng
License: MIT
"""

import logging
from typing import Dict, Type, Optional, Any

# Import prompt-based models
from src.models.prompt_based.prompt_model import PromptModel
from src.models.prompt_based.soft_prompt import (
    SoftPromptModel,
    SoftPromptConfig,
    PromptEncoder,
    DeepPromptEncoder
)
from src.models.prompt_based.instruction_model import (
    InstructionFollowingModel,
    InstructionModelConfig,
    InstructionTemplate
)
from src.models.prompt_based.template_manager import (
    TemplateManager,
    PromptTemplate,
    Verbalizer,
    TemplateType,
    VerbalizerType
)

from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Version information
__version__ = "1.0.0"

# Package metadata
__author__ = "Võ Hải Dũng"
__email__ = "your.email@example.com"
__description__ = "Prompt-based models for AG News classification"

# Model registry for prompt-based models
PROMPT_MODELS: Dict[str, Type] = {
    "prompt_model": PromptModel,
    "soft_prompt": SoftPromptModel,
    "instruction": InstructionFollowingModel
}

# Configuration registry
PROMPT_CONFIGS: Dict[str, Type] = {
    "soft_prompt": SoftPromptConfig,
    "instruction": InstructionModelConfig
}

# Default configurations
DEFAULT_CONFIGS = {
    "soft_prompt": {
        "prompt_length": 20,
        "init_strategy": "random_normal",
        "reparameterize": True
    },
    "instruction": {
        "model_name": "google/flan-t5-base",
        "instruction_template": "zero_shot",
        "temperature": 0.7
    }
}


def create_prompt_model(
    model_type: str,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Any:
    """
    Factory function to create prompt-based models.
    
    Args:
        model_type: Type of prompt model to create
        config: Configuration dictionary
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Initialized prompt-based model
        
    Raises:
        ValueError: If model type is not recognized
    """
    if model_type not in PROMPT_MODELS:
        raise ValueError(
            f"Unknown prompt model type: {model_type}. "
            f"Available models: {list(PROMPT_MODELS.keys())}"
        )
    
    # Get model class
    model_class = PROMPT_MODELS[model_type]
    
    # Get configuration class if available
    config_class = PROMPT_CONFIGS.get(model_type)
    
    # Create configuration
    if config_class and config is not None:
        if isinstance(config, dict):
            model_config = config_class(**config)
        else:
            model_config = config
    else:
        # Use default configuration
        default_config = DEFAULT_CONFIGS.get(model_type, {})
        if config_class:
            model_config = config_class(**default_config)
        else:
            model_config = None
    
    # Create model
    model = model_class(config=model_config, **kwargs)
    
    logger.info(f"Created {model_type} prompt model")
    return model


def get_available_models() -> Dict[str, str]:
    """
    Get list of available prompt-based models.
    
    Returns:
        Dictionary mapping model names to descriptions
    """
    return {
        "prompt_model": "Base prompt-based model with manual templates",
        "soft_prompt": "Soft prompt tuning with learnable continuous prompts",
        "instruction": "Instruction-following model for zero-shot and few-shot learning"
    }


def get_model_info(model_type: str) -> Dict[str, Any]:
    """
    Get information about a specific prompt model.
    
    Args:
        model_type: Type of prompt model
        
    Returns:
        Dictionary with model information
        
    Raises:
        ValueError: If model type is not recognized
    """
    if model_type not in PROMPT_MODELS:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model_class = PROMPT_MODELS[model_type]
    config_class = PROMPT_CONFIGS.get(model_type)
    
    info = {
        "name": model_type,
        "class": model_class.__name__,
        "module": model_class.__module__,
        "description": model_class.__doc__.split('\n')[0] if model_class.__doc__ else "",
        "has_config": config_class is not None
    }
    
    if config_class:
        info["config_class"] = config_class.__name__
        info["default_config"] = DEFAULT_CONFIGS.get(model_type, {})
    
    return info


# Initialize package-level template manager
_global_template_manager = None


def get_template_manager() -> TemplateManager:
    """
    Get global template manager instance.
    
    Returns:
        Global TemplateManager instance
    """
    global _global_template_manager
    if _global_template_manager is None:
        _global_template_manager = TemplateManager(task="ag_news")
        logger.info("Initialized global template manager")
    return _global_template_manager


def register_template(name: str, template: PromptTemplate):
    """
    Register a new template globally.
    
    Args:
        name: Template name
        template: Template to register
    """
    manager = get_template_manager()
    manager.add_template(name, template)
    logger.info(f"Registered template: {name}")


def register_verbalizer(name: str, verbalizer: Verbalizer):
    """
    Register a new verbalizer globally.
    
    Args:
        name: Verbalizer name
        verbalizer: Verbalizer to register
    """
    manager = get_template_manager()
    manager.add_verbalizer(name, verbalizer)
    logger.info(f"Registered verbalizer: {name}")


# Package exports
__all__ = [
    # Models
    "PromptModel",
    "SoftPromptModel",
    "InstructionFollowingModel",
    
    # Configurations
    "SoftPromptConfig",
    "InstructionModelConfig",
    
    # Template management
    "TemplateManager",
    "PromptTemplate",
    "Verbalizer",
    "TemplateType",
    "VerbalizerType",
    "InstructionTemplate",
    
    # Components
    "PromptEncoder",
    "DeepPromptEncoder",
    
    # Factory functions
    "create_prompt_model",
    "get_available_models",
    "get_model_info",
    
    # Template utilities
    "get_template_manager",
    "register_template",
    "register_verbalizer",
    
    # Constants
    "PROMPT_MODELS",
    "PROMPT_CONFIGS",
    "DEFAULT_CONFIGS"
]

# Log package initialization
logger.info(f"Initialized prompt_based package v{__version__}")
