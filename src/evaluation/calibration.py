"""Confidence calibration utilities.

Two diagnostics are implemented:

* Expected Calibration Error (Naeini et al., 2015) and the related Maximum
  Calibration Error;
* the reliability diagram that plots predicted confidence against empirical
  accuracy in equal-width bins.

The implementation follows Guo et al. (2017), *On Calibration of Modern
Neural Networks*: predictions are bucketed by confidence, and the gap
between average confidence and average accuracy in each bucket is
weighted by the bucket population.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.io_utils import ensure_dir


@dataclass
class CalibrationReport:
    """Container for calibration diagnostics."""

    ece: float
    mce: float
    n_bins: int
    bin_table: pd.DataFrame  # one row per bin: low, high, count, accuracy, confidence


def expected_calibration_error(
    y_true: Sequence[int] | np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
) -> CalibrationReport:
    """Compute ECE / MCE and return a per-bin summary table."""

    y_true_arr = np.asarray(y_true)
    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    accuracies = (predictions == y_true_arr).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []
    ece = 0.0
    mce = 0.0
    n = len(y_true_arr)
    for low, high in zip(edges[:-1], edges[1:]):
        # The last bin is closed on the right so confidence == 1 is included.
        if high == edges[-1]:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {"low": float(low), "high": float(high), "count": 0, "accuracy": 0.0, "confidence": 0.0}
            )
            continue
        bin_acc = float(accuracies[mask].mean())
        bin_conf = float(confidences[mask].mean())
        gap = abs(bin_conf - bin_acc)
        ece += (count / n) * gap
        mce = max(mce, gap)
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "count": count,
                "accuracy": bin_acc,
                "confidence": bin_conf,
            }
        )
    return CalibrationReport(
        ece=float(ece),
        mce=float(mce),
        n_bins=n_bins,
        bin_table=pd.DataFrame(rows),
    )


def plot_reliability_diagram(
    report: CalibrationReport,
    *,
    output_path: Path | str,
    title: str = "Reliability diagram",
) -> Path:
    """Plot the reliability diagram described by ``report``."""

    output = Path(output_path)
    ensure_dir(output.parent)
    df = report.bin_table.loc[report.bin_table["count"] > 0]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    width = (df["high"] - df["low"]).to_numpy()
    centers = (df["high"] + df["low"]).to_numpy() / 2.0
    ax.bar(
        centers,
        df["accuracy"],
        width=width,
        edgecolor="black",
        alpha=0.7,
        label="Empirical accuracy",
    )
    ax.plot([0.0, 1.0], [0.0, 1.0], "--", color="grey", label="Ideal calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"{title}\nECE={report.ece:.4f}, MCE={report.mce:.4f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
