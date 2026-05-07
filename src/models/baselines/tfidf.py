"""TF-IDF + Logistic Regression / Linear SVM baseline.

The class wraps a scikit-learn :class:`Pipeline` so the training and
inference logic is shared with the rest of the codebase. It exposes
``predict_proba`` for both classifiers; for the SVM variant we calibrate
decision values via :class:`sklearn.calibration.CalibratedClassifierCV`,
which is required for the calibration analysis (ECE) in Phase 7.

References
----------
Joachims, T. (1998). *Text categorization with support vector machines*. ECML.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class TfidfConfig:
    """Hyper-parameters for the TF-IDF baseline."""

    classifier: str = "logreg"  # one of {"logreg", "svm"}
    max_features: int = 200_000
    ngram_range: tuple[int, int] = (1, 2)
    sublinear_tf: bool = True
    min_df: int = 2
    max_df: float = 0.95
    lowercase: bool = True
    C: float = 1.0
    max_iter: int = 2000
    seed: int = 42


class TfidfClassifier:
    """TF-IDF + Logistic Regression / Linear SVM."""

    def __init__(self, config: TfidfConfig | None = None) -> None:
        self.config = config or TfidfConfig()
        self.pipeline: Pipeline = self._build_pipeline()
        self._is_fitted = False
        self._classes: np.ndarray | None = None

    # ------------------------------------------------------------------ public API

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> "TfidfClassifier":
        labels_array = np.asarray(list(labels))
        self.pipeline.fit(list(texts), labels_array)
        self._is_fitted = True
        self._classes = self.pipeline.classes_
        return self

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict(list(texts))

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        self._check_fitted()
        return self.pipeline.predict_proba(list(texts))

    def save(self, path: Path | str) -> Path:
        """Persist the fitted pipeline as a single ``joblib`` file."""

        self._check_fitted()
        target = Path(path)
        ensure_dir(target.parent)
        joblib.dump({"config": self.config, "pipeline": self.pipeline}, target)
        _logger.info("Saved TF-IDF baseline to %s", target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "TfidfClassifier":
        bundle = joblib.load(Path(path))
        instance = cls(config=bundle["config"])
        instance.pipeline = bundle["pipeline"]
        instance._is_fitted = True
        instance._classes = instance.pipeline.classes_
        return instance

    # ---------------------------------------------------------------- internals

    def _build_pipeline(self) -> Pipeline:
        vectorizer = TfidfVectorizer(
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            sublinear_tf=self.config.sublinear_tf,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            lowercase=self.config.lowercase,
            strip_accents="unicode",
        )
        if self.config.classifier == "logreg":
            classifier = LogisticRegression(
                solver="liblinear",
                C=self.config.C,
                max_iter=self.config.max_iter,
                random_state=self.config.seed,
            )
        elif self.config.classifier == "svm":
            base = LinearSVC(C=self.config.C, max_iter=self.config.max_iter)
            classifier = CalibratedClassifierCV(base, cv=3, method="sigmoid")
        else:
            raise ValueError(
                f"Unknown classifier '{self.config.classifier}'. Expected 'logreg' or 'svm'."
            )
        return Pipeline(steps=[("tfidf", vectorizer), ("clf", classifier)])

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("TfidfClassifier has not been fitted yet.")
