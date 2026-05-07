"""Dataclass schema for project configurations.

Configuration files are YAML for human readability; this module turns them
into typed objects so that the rest of the code can rely on attribute access
and IDE autocompletion. Unknown YAML keys raise :class:`TypeError`, which
catches typos early.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from src.utils.io_utils import load_yaml


@dataclass(frozen=True)
class PathConfig:
    """File-system locations for inputs and outputs of a single run."""

    data_dir: str = "data"
    output_dir: str = "outputs"
    cache_dir: str = "data/cache"
    checkpoints_dir: str = "outputs/checkpoints"
    figures_dir: str = "outputs/figures"
    metrics_dir: str = "outputs/metrics"


@dataclass(frozen=True)
class DataConfig:
    """Dataset-level configuration."""

    name: str = "ag_news"
    hf_path: str = "ag_news"
    text_column: str = "text"
    label_column: str = "label"
    num_labels: int = 4
    label_names: tuple[str, ...] = ("World", "Sports", "Business", "Sci/Tech")
    train_split: str = "train"
    test_split: str = "test"
    validation_size: float = 0.1
    max_length: int = 256
    language: str = "en"


@dataclass(frozen=True)
class ModelConfig:
    """Model architecture and tokenizer configuration."""

    name: str = "microsoft/deberta-v3-small"
    family: str = "deberta-v3"
    tokenizer_name: str | None = None
    max_length: int = 256
    dropout: float = 0.1
    pooling: str = "cls"


@dataclass(frozen=True)
class TrainingConfig:
    """Optimisation configuration shared by transformer and SetFit pipelines."""

    seed: int = 42
    epochs: int = 3
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    fp16: bool = True
    bf16: bool = False
    optimizer: str = "adamw_torch"
    scheduler: str = "linear"
    early_stopping_patience: int = 2
    save_total_limit: int = 2
    logging_steps: int = 50
    eval_steps: int | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation and explainability configuration."""

    metrics: tuple[str, ...] = ("accuracy", "f1_macro", "f1_weighted")
    compute_calibration: bool = True
    n_calibration_bins: int = 15
    benchmark_latency: bool = True
    latency_warmup: int = 10
    latency_iters: int = 200
    shap_n_samples: int = 25
    lime_n_samples: int = 25


@dataclass(frozen=True)
class DeploymentConfig:
    """Deployment configuration for ONNX export and the Gradio demo."""

    onnx_opset: int = 17
    quantize: bool = True
    quantization_format: str = "QInt8"
    gradio_share: bool = False
    hf_space_repo: str | None = None


@dataclass(frozen=True)
class TrackingConfig:
    """Experiment tracking configuration."""

    backend: str = "none"
    project: str = "ag-news-classification"
    entity: str | None = None
    run_name: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level configuration container.

    Use :func:`load_config` to materialise an instance from a YAML file.
    """

    name: str = "default"
    description: str = ""
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


# When ``from __future__ import annotations`` is in effect the ``type``
# attribute on each :class:`dataclasses.Field` is a string rather than the
# class itself, so we cannot rely on ``is_dataclass(field.type)`` to detect
# nested dataclasses. Instead we resolve the annotations explicitly via
# :func:`typing.get_type_hints`.
def _from_mapping(cls: type, payload: Mapping[str, Any]) -> Any:
    """Recursively build a dataclass from a mapping.

    The function walks the ``cls`` field declarations, descending into nested
    dataclasses where appropriate. Tuple-typed fields receive list inputs and
    convert them transparently so YAML lists are accepted.
    """

    from typing import get_type_hints

    if not is_dataclass(cls):
        return payload
    declared = {f.name: f for f in fields(cls)}
    unknown = set(payload) - declared.keys()
    if unknown:
        raise TypeError(f"Unknown configuration keys for {cls.__name__}: {sorted(unknown)}")
    type_hints = get_type_hints(cls)
    init_kwargs: dict[str, Any] = {}
    for name in declared:
        if name not in payload:
            continue
        value = payload[name]
        annotation = type_hints.get(name)
        if is_dataclass(annotation) and isinstance(value, Mapping):
            init_kwargs[name] = _from_mapping(annotation, value)
        elif isinstance(value, list):
            init_kwargs[name] = tuple(value)
        else:
            init_kwargs[name] = value
    return cls(**init_kwargs)


def load_config(path: Path | str) -> ExperimentConfig:
    """Load a YAML file and return a validated :class:`ExperimentConfig`."""

    payload = load_yaml(path)
    return _from_mapping(ExperimentConfig, payload)
