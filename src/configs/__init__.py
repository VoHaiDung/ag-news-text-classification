"""Typed configuration schema for the project.

The schema is implemented as a hierarchy of frozen :mod:`dataclasses`. YAML
configuration files under ``configs/`` are loaded by :func:`load_config` and
validated against this schema, which gives us static type information at the
point of use without resorting to a heavy framework.
"""

from src.configs.schema import (
    DataConfig,
    DeploymentConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
    load_config,
)

__all__ = [
    "DataConfig",
    "DeploymentConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    "ModelConfig",
    "PathConfig",
    "TrainingConfig",
    "load_config",
]
