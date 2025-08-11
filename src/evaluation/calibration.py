# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
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


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    max_iter: int = 200,
) -> float:
    """Learn the scalar temperature T that minimises NLL on ``logits``.

    Implements the temperature-scaling post-hoc calibrator of Guo et al.
    (2017). Logits are divided by a single positive scalar T before the
    softmax; T is optimised on a held-out validation set so the calibration
    fix never sees the test split.

    Parameters
    ----------
    logits:
        ``(n_samples, n_classes)`` array of pre-softmax logits collected
        on the validation set.
    labels:
        ``(n_samples,)`` integer class labels for the same set.
    max_iter:
        Maximum L-BFGS iterations. The objective is convex in
        ``log T`` so convergence is essentially always reached in a
        few dozen steps.

    Returns
    -------
    float
        The fitted temperature. Values above 1 indicate the original
        model was over-confident; values below 1 indicate
        under-confidence.
    """

    import torch
    import torch.nn.functional as F

    logits_t = torch.from_numpy(np.asarray(logits, dtype=np.float32))
    labels_t = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T) keeps T > 0

    optimiser = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        optimiser.zero_grad()
        T = log_T.exp()
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss

    optimiser.step(closure)
    return float(log_T.detach().exp().item())


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return the temperature-scaled softmax probabilities."""

    scaled = np.asarray(logits, dtype=np.float64) / float(temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_isotonic_calibrators(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> list:
    """Fit one isotonic regressor per class (one-vs-rest) on val probs.

    Class-wise isotonic regression (Zadrozny and Elkan, 2002) is a
    non-parametric calibrator. For each class ``k`` it learns a
    monotonic mapping from the predicted probability of class ``k``
    to the empirical frequency of class ``k`` among examples with
    that predicted probability. Unlike temperature scaling, isotonic
    regression has no functional form assumption, so it can correct
    the asymmetric mis-calibration introduced by label smoothing
    during training.

    Parameters
    ----------
    probabilities:
        ``(n_samples, n_classes)`` validation softmax probabilities.
    labels:
        ``(n_samples,)`` integer class labels.

    Returns
    -------
    list of ``sklearn.isotonic.IsotonicRegression``
        One fitted calibrator per class.
    """

    from sklearn.isotonic import IsotonicRegression

    n_classes = probabilities.shape[1]
    calibrators: list = []
    labels_arr = np.asarray(labels)
    for k in range(n_classes):
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        target = (labels_arr == k).astype(np.float64)
        ir.fit(probabilities[:, k], target)
        calibrators.append(ir)
    return calibrators


def apply_isotonic(
    probabilities: np.ndarray,
    calibrators: list,
) -> np.ndarray:
    """Apply per-class isotonic regressors and re-normalise to a simplex."""

    n_classes = len(calibrators)
    calibrated = np.zeros_like(probabilities, dtype=np.float64)
    for k in range(n_classes):
        calibrated[:, k] = calibrators[k].predict(probabilities[:, k])
    # Each row should sum to 1; rows whose components collapse to 0 are
    # mapped to a uniform distribution as a safe fallback.
    row_sum = calibrated.sum(axis=1, keepdims=True)
    calibrated = np.where(row_sum > 0, calibrated, 1.0 / n_classes)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    return calibrated / row_sum


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
