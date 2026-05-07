"""Tests for the metric helpers."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.calibration import expected_calibration_error
from src.evaluation.metrics import (
    classification_report_table,
    compute_metrics,
    confusion_matrix_table,
)


def test_compute_metrics_perfect() -> None:
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = y_true.copy()
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["f1_macro"] == pytest.approx(1.0)


def test_compute_metrics_random() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 4, size=200)
    y_pred = rng.integers(0, 4, size=200)
    metrics = compute_metrics(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_classification_report_table_shape() -> None:
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 0])
    report = classification_report_table(
        y_true, y_pred, label_names=("World", "Sports", "Business", "Sci/Tech")
    )
    assert {"precision", "recall", "f1_score", "support", "label"}.issubset(report.columns)


def test_confusion_matrix_normalisation() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    cm = confusion_matrix_table(y_true, y_pred, label_names=("a", "b"), normalize="true")
    assert cm.loc["a", "a"] + cm.loc["a", "b"] == pytest.approx(1.0)
    assert cm.loc["b", "b"] == pytest.approx(1.0)


def test_expected_calibration_error_perfect() -> None:
    probs = np.eye(4)[np.array([0, 1, 2, 3])]
    report = expected_calibration_error(np.array([0, 1, 2, 3]), probs, n_bins=10)
    assert report.ece == pytest.approx(0.0)
