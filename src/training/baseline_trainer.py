"""Training loop for the classical baselines.

The trainer runs a single ``fit`` call (the classical models have no
notion of epochs in the way transformers do), evaluates on the validation
and test splits, and writes the artefacts that the evaluation pipeline
expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from datasets import Dataset

from src.evaluation.metrics import classification_report_table, compute_metrics
from src.utils.io_utils import ensure_dir, save_json
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


class _BaselineLike(Protocol):
    def fit(self, texts, labels): ...
    def predict(self, texts) -> np.ndarray: ...
    def predict_proba(self, texts) -> np.ndarray: ...
    def save(self, path: Path | str) -> Path: ...


@dataclass
class BaselineRunResult:
    name: str
    metrics: dict[str, float]
    test_predictions: np.ndarray
    test_probabilities: np.ndarray
    artefact_paths: dict[str, Path]


class BaselineTrainer:
    """Generic trainer for classical baselines."""

    def __init__(
        self,
        *,
        text_column: str,
        label_column: str,
        label_names: tuple[str, ...],
        output_dir: Path | str,
    ) -> None:
        self.text_column = text_column
        self.label_column = label_column
        self.label_names = label_names
        self.output_dir = ensure_dir(output_dir)

    def run(
        self,
        name: str,
        model: _BaselineLike,
        train: Dataset,
        validation: Dataset,
        test: Dataset,
    ) -> BaselineRunResult:
        run_dir = ensure_dir(self.output_dir / name)
        _logger.info("Training baseline '%s'", name)
        model.fit(train[self.text_column], train[self.label_column])

        val_proba = model.predict_proba(validation[self.text_column])
        val_pred = np.argmax(val_proba, axis=1)
        val_metrics = compute_metrics(np.asarray(validation[self.label_column]), val_pred)

        test_proba = model.predict_proba(test[self.text_column])
        test_pred = np.argmax(test_proba, axis=1)
        test_metrics = compute_metrics(np.asarray(test[self.label_column]), test_pred)

        all_metrics = {
            **{f"val_{k}": v for k, v in val_metrics.items()},
            **{f"test_{k}": v for k, v in test_metrics.items()},
        }
        save_json(all_metrics, run_dir / "metrics.json")

        report = classification_report_table(
            np.asarray(test[self.label_column]),
            test_pred,
            label_names=self.label_names,
        )
        report.to_csv(run_dir / "classification_report.csv", index=False)

        np.save(run_dir / "test_probabilities.npy", test_proba)
        pd.DataFrame(
            {
                "text": test[self.text_column],
                "label": test[self.label_column],
                "label_name": [self.label_names[i] for i in test[self.label_column]],
                "prediction": test_pred,
                "prediction_name": [self.label_names[i] for i in test_pred],
                "max_probability": test_proba.max(axis=1),
            }
        ).to_csv(run_dir / "test_predictions.csv", index=False)

        model_path = model.save(run_dir / "model.bin")
        _logger.info(
            "Baseline '%s' finished. Test accuracy=%.4f, F1-macro=%.4f",
            name,
            test_metrics["accuracy"],
            test_metrics["f1_macro"],
        )
        return BaselineRunResult(
            name=name,
            metrics=all_metrics,
            test_predictions=test_pred,
            test_probabilities=test_proba,
            artefact_paths={
                "model": model_path,
                "metrics": run_dir / "metrics.json",
                "report": run_dir / "classification_report.csv",
                "predictions": run_dir / "test_predictions.csv",
                "probabilities": run_dir / "test_probabilities.npy",
            },
        )
