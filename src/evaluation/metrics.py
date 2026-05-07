"""Classification metrics shared across phases.

The helpers in this module return the headline numbers that every WBS phase
needs (accuracy, macro/weighted F1, precision and recall) and a tidy
per-class table compatible with :func:`pandas.DataFrame.to_csv`.

Implementation notes
--------------------
* All inputs are coerced to :class:`numpy.ndarray` so the helpers work on
  Python lists, pandas Series and HuggingFace ``Dataset`` columns.
* ``zero_division=0`` is used everywhere to keep the metrics defined when a
  class is absent from the predictions; this is preferable to NaN values
  that downstream serialisation has to special-case.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, float]:
    """Return the headline classification metrics."""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1_macro": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)
        ),
    }


def classification_report_table(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    label_names: Sequence[str],
) -> pd.DataFrame:
    """Return scikit-learn's classification report as a tidy DataFrame."""

    report = classification_report(
        y_true,
        y_pred,
        target_names=list(label_names),
        output_dict=True,
        zero_division=0,
    )
    rows = []
    for key, value in report.items():
        if not isinstance(value, dict):
            continue
        rows.append(
            {
                "label": key,
                "precision": value.get("precision", float("nan")),
                "recall": value.get("recall", float("nan")),
                "f1_score": value.get("f1-score", float("nan")),
                "support": value.get("support", 0),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix_table(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    *,
    label_names: Sequence[str],
    normalize: str | None = None,
) -> pd.DataFrame:
    """Return the confusion matrix as a labelled DataFrame.

    Parameters
    ----------
    normalize:
        Forwarded to :func:`sklearn.metrics.confusion_matrix`. ``None``
        returns raw counts, ``"true"`` normalises by the true class
        (recall), ``"pred"`` by the predicted class (precision).
    """

    matrix = confusion_matrix(y_true, y_pred, normalize=normalize)
    table = pd.DataFrame(matrix, index=list(label_names), columns=list(label_names))
    table.index.name = "true_label"
    table.columns.name = "predicted_label"
    return table
