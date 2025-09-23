"""
Core Services for AG News Text Classification
================================================================================
Core business services implementing the main functionality of the system
including prediction, training, data management, and model lifecycle.

These services follow the Single Responsibility Principle (SRP) and are
designed to be loosely coupled and highly cohesive.

Author: Võ Hải Dũng
License: MIT
"""

from src.services.core.prediction_service import PredictionService
from src.services.core.training_service import TrainingService
from src.services.core.data_service import DataService
from src.services.core.model_management_service import ModelManagementService

__all__ = [
    "PredictionService",
    "TrainingService",
    "DataService",
    "ModelManagementService"
]
