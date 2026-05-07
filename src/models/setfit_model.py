"""SetFit few-shot classifier.

SetFit (Tunstall et al., 2022) fine-tunes a sentence-transformer with a
contrastive loss on a small labelled set, then trains a logistic-regression
head on top of the resulting embeddings. The wrapper below accepts a
:class:`datasets.Dataset` so it shares the same data pipeline as the rest
of the project.

References
----------
Tunstall, L. et al. (2022). *Efficient Few-Shot Learning Without Prompts.* EMNLP.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class SetFitTrainingConfig:
    """Hyper-parameters for SetFit training."""

    model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2"
    num_iterations: int = 20
    num_epochs: int = 1
    batch_size: int = 16
    head_learning_rate: float = 1.0e-3
    body_learning_rate: float = 2.0e-5
    seed: int = 42


class SetFitClassifier:
    """Train and evaluate a SetFit model in the AG News context."""

    def __init__(
        self,
        config: SetFitTrainingConfig | None = None,
        *,
        text_column: str = "text",
        label_column: str = "label",
    ) -> None:
        self.config = config or SetFitTrainingConfig()
        self.text_column = text_column
        self.label_column = label_column
        self._model: SetFitModel | None = None

    # ------------------------------------------------------------------ public API

    def fit(self, train: Dataset, validation: Dataset | None = None) -> "SetFitClassifier":
        train = self._prepare(train)
        validation = self._prepare(validation) if validation is not None else None

        self._model = SetFitModel.from_pretrained(self.config.model_name)
        args = TrainingArguments(
            output_dir="setfit_runs",
            num_iterations=self.config.num_iterations,
            num_epochs=self.config.num_epochs,
            batch_size=self.config.batch_size,
            head_learning_rate=self.config.head_learning_rate,
            body_learning_rate=self.config.body_learning_rate,
            seed=self.config.seed,
            report_to="none",
        )
        trainer = Trainer(
            model=self._model,
            args=args,
            train_dataset=train,
            eval_dataset=validation,
            metric="f1",
        )
        trainer.train()
        return self

    def predict(self, texts: Iterable[str]) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._model.predict(list(texts)))

    def predict_proba(self, texts: Iterable[str]) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._model.predict_proba(list(texts)))

    def save(self, path: Path | str) -> Path:
        self._check_fitted()
        target = Path(path)
        ensure_dir(target)
        self._model.save_pretrained(str(target))
        return target

    @classmethod
    def load(cls, path: Path | str) -> "SetFitClassifier":
        instance = cls()
        instance._model = SetFitModel.from_pretrained(str(path))
        return instance

    # ---------------------------------------------------------------- internals

    def _prepare(self, dataset: Dataset) -> Dataset:
        """Rename columns to the SetFit convention (``text`` / ``label``)."""

        rename: dict[str, str] = {}
        if self.text_column != "text":
            rename[self.text_column] = "text"
        if self.label_column != "label":
            rename[self.label_column] = "label"
        return dataset.rename_columns(rename) if rename else dataset

    def _check_fitted(self) -> None:
        if self._model is None:
            raise RuntimeError("SetFitClassifier has not been fitted yet.")
