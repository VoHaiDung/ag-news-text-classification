"""
Configuration module for AG News Text Classification

This module provides centralized configuration management for:
- Model configurations
- Training configurations
- Data configurations
- Experiment configurations
"""

from .config_loader import ConfigLoader
from .constants import *

__all__ = ['ConfigLoader']

# Package metadata
__version__ = '0.1.0'
__author__ = 'AG News SOTA Team'
__description__ = 'Configuration management for AG News classification'
