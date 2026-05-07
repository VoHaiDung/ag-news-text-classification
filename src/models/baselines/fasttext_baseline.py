"""FastText supervised baseline.

FastText (Joulin et al., 2017) trains a linear classifier on the average of
character-aware word embeddings. It is included as a baseline because it is
strong-but-fast on short news headlines and offers a useful contrast with
TF-IDF.

The wrapper exposes a scikit-learn-like API (``fit`` / ``predict`` /
``predict_proba``) and writes inputs in the FastText label format
(``__label__N text``) to a temporary file so the training command can be
issued via the official Python binding.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import fasttext
import numpy as np

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)
_LABEL_PREFIX = "__label__"


@dataclass
class FastTextConfig:
    """FastText supervised hyper-parameters."""

    epochs: int = 8
    learning_rate: float = 0.5
    word_ngrams: int = 2
    dim: int = 100
    min_count: int = 2
    bucket: int = 2_000_000
    loss: str = "softmax"  # softmax, hs, ova
    seed: int = 42


class FastTextClassifier:
    """Thin wrapper around :func:`fasttext.train_supervised`."""

    def __init__(self, config: FastTextConfig | None = None) -> None:
        self.config = config or FastTextConfig()
        self._model: fasttext.FastText._FastText | None = None
        self._classes: np.ndarray | None = None

    # ------------------------------------------------------------------ public API

    def fit(self, texts: Iterable[str], labels: Iterable[int]) -> "FastTextClassifier":
        labels_array = np.asarray(list(labels))
        texts_list = list(texts)
        with tempfile.TemporaryDirectory() as tmp:
            train_path = Path(tmp) / "train.txt"
            self._dump_fasttext_format(train_path, texts_list, labels_array)
            self._model = fasttext.train_supervised(
                input=str(train_path),
                epoch=self.config.epochs,
                lr=self.config.learning_rate,
                wordNgrams=self.config.word_ngrams,
                dim=self.config.dim,
                minCount=self.config.min_count,
                bucket=self.config.bucket,
                loss=self.config.loss,
                seed=self.config.seed,
                verbose=0,
            )
        self._classes = np.unique(labels_array)
        return self

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        self._check_fitted()
        labels, _ = self._model.predict([self._normalise(t) for t in texts], k=1)
        return np.asarray([self._strip_prefix(lbl[0]) for lbl in labels], dtype=int)

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        """Return a dense ``(n_samples, n_classes)`` probability matrix.

        FastText returns the probabilities for the top-``k`` labels. We ask
        for ``k = num_classes`` and reorder the result so column ``i``
        corresponds to label ``i`` regardless of the order FastText emits.
        """

        self._check_fitted()
        n_classes = len(self._classes)
        labels, probs = self._model.predict(
            [self._normalise(t) for t in texts], k=n_classes
        )
        out = np.zeros((len(labels), n_classes), dtype=np.float64)
        for row, (lbl_row, prob_row) in enumerate(zip(labels, probs)):
            for lbl, prob in zip(lbl_row, prob_row):
                out[row, self._strip_prefix(lbl)] = prob
        # Renormalise to guard against rounding errors when ``k < n_classes``.
        row_sums = out.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return out / row_sums

    def save(self, path: Path | str) -> Path:
        self._check_fitted()
        target = Path(path)
        ensure_dir(target.parent)
        self._model.save_model(str(target))
        _logger.info("Saved FastText model to %s", target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "FastTextClassifier":
        instance = cls()
        instance._model = fasttext.load_model(str(path))
        instance._classes = np.array(
            sorted({cls._strip_prefix(lbl) for lbl in instance._model.get_labels()})
        )
        return instance

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _normalise(text: str) -> str:
        return text.replace("\n", " ").strip()

    @staticmethod
    def _strip_prefix(label: str) -> int:
        return int(label.removeprefix(_LABEL_PREFIX))

    def _dump_fasttext_format(
        self, path: Path, texts: list[str], labels: np.ndarray
    ) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for text, label in zip(texts, labels):
                handle.write(f"{_LABEL_PREFIX}{int(label)} {self._normalise(text)}\n")

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("FastTextClassifier has not been fitted yet.")
