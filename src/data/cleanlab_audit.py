"""Label-noise auditing with Cleanlab.

The audit is run in two passes per WBS task 2.2:

1. *Setup* (task 2.2.1): a probe model produces out-of-fold class
   probabilities that Cleanlab consumes.
2. *Detection* (task 2.2.2): Cleanlab returns a ranked list of suspected
   noisy labels, which is exported as CSV for manual review.

The probe model is intentionally light-weight (TF-IDF + Logistic Regression
on bigrams) so that the audit can be reproduced on a CPU in minutes and is
independent of the transformer training pipeline.

References
----------
Northcutt, C., Jiang, L., and Chuang, I. (2021).
*Confident Learning: Estimating Uncertainty in Dataset Labels.* JAIR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.filter import find_label_issues
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class CleanlabReport:
    """Outcome of a Cleanlab audit."""

    suspect_indices: np.ndarray
    pred_probs: np.ndarray
    suspect_table: pd.DataFrame

    def save(self, directory: Path | str) -> dict[str, Path]:
        """Persist the report under ``directory`` and return the file paths."""

        out_dir = ensure_dir(directory)
        suspect_path = out_dir / "suspect_labels.csv"
        probs_path = out_dir / "out_of_fold_probabilities.npy"
        self.suspect_table.to_csv(suspect_path, index=False)
        np.save(probs_path, self.pred_probs)
        return {"suspects": suspect_path, "probabilities": probs_path}


def _build_probe_pipeline(*, max_features: int, ngram_range: tuple[int, int]) -> Pipeline:
    """Construct the TF-IDF + Logistic Regression probe used for OOF probabilities."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    solver="liblinear",
                    C=1.0,
                    max_iter=1000,
                    n_jobs=None,
                ),
            ),
        ]
    )


def cross_val_predict_proba(
    texts: np.ndarray | list[str],
    labels: np.ndarray | list[int],
    *,
    n_splits: int = 5,
    seed: int = 42,
    max_features: int = 200_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> np.ndarray:
    """Compute out-of-fold class probabilities used as Cleanlab input.

    Stratified ``KFold`` ensures that every class is represented in every
    fold, which prevents Cleanlab from receiving probability vectors with
    missing classes.
    """

    texts = np.asarray(texts)
    labels = np.asarray(labels)
    n_classes = int(np.max(labels)) + 1
    pred_probs = np.zeros((len(texts), n_classes), dtype=np.float64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(texts, labels), start=1):
        _logger.info("Cleanlab probe fold %d/%d", fold, n_splits)
        pipeline = _build_probe_pipeline(max_features=max_features, ngram_range=ngram_range)
        pipeline.fit(texts[train_idx], labels[train_idx])
        pred_probs[valid_idx] = pipeline.predict_proba(texts[valid_idx])
    return pred_probs


def audit(
    texts: np.ndarray | list[str],
    labels: np.ndarray | list[int],
    *,
    label_names: tuple[str, ...] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> CleanlabReport:
    """Run a full Cleanlab audit and return the suspect indices and table."""

    texts_array = np.asarray(texts)
    labels_array = np.asarray(labels)
    pred_probs = cross_val_predict_proba(
        texts_array, labels_array, n_splits=n_splits, seed=seed
    )
    suspects = find_label_issues(
        labels=labels_array,
        pred_probs=pred_probs,
        return_indices_ranked_by="self_confidence",
    )
    predicted = pred_probs.argmax(axis=1)
    suspect_table = pd.DataFrame(
        {
            "index": suspects,
            "given_label": labels_array[suspects],
            "given_label_name": (
                np.asarray(label_names)[labels_array[suspects]] if label_names else None
            ),
            "predicted_label": predicted[suspects],
            "predicted_label_name": (
                np.asarray(label_names)[predicted[suspects]] if label_names else None
            ),
            "self_confidence": pred_probs[suspects, labels_array[suspects]],
            "predicted_probability": pred_probs[suspects, predicted[suspects]],
            "text": texts_array[suspects],
        }
    )
    _logger.info(
        "Cleanlab audit: %d / %d examples flagged as potentially mislabelled.",
        len(suspects),
        len(labels_array),
    )
    return CleanlabReport(
        suspect_indices=suspects,
        pred_probs=pred_probs,
        suspect_table=suspect_table,
    )
