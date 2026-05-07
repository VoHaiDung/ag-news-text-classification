"""Experiment tracking abstraction.

The same training script must work whether the user has Weights and Biases
configured, prefers MLflow, or wants to disable tracking entirely (for
example during continuous integration). The :class:`ExperimentTracker`
protocol below standardises the small surface that the rest of the codebase
relies on.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Protocol

from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


class ExperimentTracker(Protocol):
    """Minimal tracker interface used by training and evaluation code."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Record hyper-parameters and configuration values."""

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record a set of scalar metrics, optionally tagged with ``step``."""

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        """Persist a file (model checkpoint, plot, table) to the tracker."""

    def finish(self) -> None:
        """Flush pending records and close the underlying client."""


class _NullTracker:
    """No-op tracker used when ``backend == 'none'`` or initialisation fails."""

    def log_params(self, params: Mapping[str, Any]) -> None:
        _logger.debug("NullTracker.log_params(%d entries)", len(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        _logger.debug("NullTracker.log_metrics(%d entries, step=%s)", len(metrics), step)

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        _logger.debug("NullTracker.log_artifact(%s, name=%s)", path, name)

    def finish(self) -> None:
        return None


class _WandbTracker:
    """Adapter around the :mod:`wandb` Python SDK."""

    def __init__(
        self,
        *,
        project: str,
        entity: str | None,
        run_name: str | None,
        config: Mapping[str, Any] | None,
    ) -> None:
        import wandb  # imported lazily to avoid the dependency at import time

        self._wandb = wandb
        self._run = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            config=dict(config) if config else None,
            reinit=True,
        )

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._wandb.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._wandb.log(dict(metrics), step=step)

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        artifact = self._wandb.Artifact(name=name or Path(path).name, type="result")
        artifact.add_file(str(path))
        self._wandb.log_artifact(artifact)

    def finish(self) -> None:
        self._run.finish()


class _MlflowTracker:
    """Adapter around the :mod:`mlflow` Python SDK."""

    def __init__(
        self,
        *,
        experiment_name: str,
        run_name: str | None,
        tracking_uri: str | None,
        config: Mapping[str, Any] | None,
    ) -> None:
        import mlflow

        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self._run = mlflow.start_run(run_name=run_name)
        if config:
            mlflow.log_params(dict(config))

    def log_params(self, params: Mapping[str, Any]) -> None:
        self._mlflow.log_params(dict(params))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._mlflow.log_metrics(dict(metrics), step=step)

    def log_artifact(self, path: Path | str, name: str | None = None) -> None:
        self._mlflow.log_artifact(str(path), artifact_path=name)

    def finish(self) -> None:
        self._mlflow.end_run()


def build_tracker(
    backend: str = "none",
    *,
    project: str | None = None,
    entity: str | None = None,
    run_name: str | None = None,
    experiment_name: str | None = None,
    tracking_uri: str | None = None,
    config: Mapping[str, Any] | None = None,
) -> ExperimentTracker:
    """Factory that returns the requested tracker implementation.

    Parameters
    ----------
    backend:
        One of ``"wandb"``, ``"mlflow"`` or ``"none"`` (case-insensitive).
    project, entity, run_name:
        Forwarded to :func:`wandb.init` when ``backend == "wandb"``.
    experiment_name, tracking_uri:
        Forwarded to MLflow when ``backend == "mlflow"``. ``tracking_uri``
        defaults to the value of the ``MLFLOW_TRACKING_URI`` environment
        variable.
    config:
        Run-level hyper-parameters logged immediately after initialisation.

    Returns
    -------
    ExperimentTracker
        A live tracker instance, or a :class:`_NullTracker` if the requested
        backend is unavailable.
    """

    backend = backend.lower()
    try:
        if backend == "wandb":
            return _WandbTracker(
                project=project or os.environ.get("WANDB_PROJECT", "ag-news-classification"),
                entity=entity or os.environ.get("WANDB_ENTITY"),
                run_name=run_name,
                config=config,
            )
        if backend == "mlflow":
            return _MlflowTracker(
                experiment_name=experiment_name or "ag-news-classification",
                run_name=run_name,
                tracking_uri=tracking_uri or os.environ.get("MLFLOW_TRACKING_URI"),
                config=config,
            )
    except Exception as exc:  # pragma: no cover - depends on optional deps
        _logger.warning("Tracker backend '%s' unavailable (%s); falling back to null.", backend, exc)
    return _NullTracker()
