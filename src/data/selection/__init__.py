"""
Data Selection Module
=====================

Advanced data selection strategies for efficient training.

Author: Võ Hải Dũng
License: MIT
"""

from .quality_filtering import QualityFilter, QualityFilterConfig
from .influence_function import InfluenceFunction
from .gradient_matching import GradientMatching
from .diversity_selection import DiversitySelector

__all__ = [
    "QualityFilter",
    "QualityFilterConfig",
    "InfluenceFunction",
    "GradientMatching",
    "DiversitySelector",
]

__version__ = "1.0.0"
