"""AG News dataset loader.

The loader wraps the Hugging Face ``datasets`` library and applies the same
preprocessing steps that every downstream phase needs:

* materialise the train/test splits;
* derive a stratified validation split from the training data;
* expose label names and integer ids in a consistent order;
* normalise the text column (whitespace collapsing, optional lower-casing).

Returning a :class:`DatasetSplits` named container keeps the train, validation
and test sets together while still allowing each split to be passed to a HF
``Trainer`` directly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset, load_from_disk

from src.configs import DataConfig
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)
_WHITESPACE = re.compile(r"\s+")


@dataclass
class DatasetSplits:
    """Container holding the train, validation and test splits.

    The class deliberately keeps each split as a HF :class:`Dataset` so that
    callers can use the ``map``, ``filter`` and ``with_format`` methods
    directly without needing a custom adapter.
    """

    train: Dataset
    validation: Dataset
    test: Dataset
    label_names: tuple[str, ...]

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    def as_datasetdict(self) -> DatasetDict:
        """Convenience view as a HF :class:`DatasetDict`."""

        return DatasetDict({"train": self.train, "validation": self.validation, "test": self.test})


class AGNewsLoader:
    """Load and split the AG News dataset.

    Parameters
    ----------
    config:
        :class:`DataConfig` describing the dataset path, columns and split
        ratios.
    cache_dir:
        Optional directory passed to ``datasets.load_dataset``. Useful in
        offline environments and CI.
    """

    def __init__(self, config: DataConfig, cache_dir: Path | str | None = None) -> None:
        self.config = config
        self.cache_dir = str(cache_dir) if cache_dir is not None else None

    # ------------------------------------------------------------------ public API

    def load(self, *, normalise_whitespace: bool = True, lower: bool = False) -> DatasetSplits:
        """Load the dataset and return :class:`DatasetSplits`.

        Parameters
        ----------
        normalise_whitespace:
            Replace runs of whitespace with a single space and strip the
            text. The pre-tokenizer of every transformer in the project is
            whitespace-aware, so this normalisation is safe and removes
            artefacts left by the original CSV scrape.
        lower:
            Lower-case the text. Disabled by default because casing is
            informative for proper-noun-heavy news classes (Sports, Sci/Tech).
        """

        ds = self._load_raw()
        ds = self._unify_columns(ds)
        if normalise_whitespace or lower:
            ds = ds.map(
                lambda example: {
                    self.config.text_column: self._clean_text(
                        example[self.config.text_column], lower=lower
                    )
                }
            )
        train_val = self._stratified_split(ds[self.config.train_split])
        test = ds[self.config.test_split]
        label_names = self._resolve_label_names(ds[self.config.train_split])
        _logger.info(
            "Loaded AG News (%s): train=%d, val=%d, test=%d, labels=%s",
            self.config.name,
            len(train_val["train"]),
            len(train_val["validation"]),
            len(test),
            label_names,
        )
        return DatasetSplits(
            train=train_val["train"],
            validation=train_val["validation"],
            test=test,
            label_names=label_names,
        )

    # ---------------------------------------------------------------- internals

    def _load_raw(self) -> DatasetDict:
        """Resolve the dataset either from the HF Hub or from a local folder."""

        path = Path(self.config.hf_path)
        if path.exists():
            _logger.info("Loading local dataset from %s", path)
            ds = load_from_disk(str(path))
            if not isinstance(ds, DatasetDict):
                raise TypeError(f"Expected a DatasetDict at {path}, got {type(ds).__name__}.")
            return ds
        _logger.info("Loading dataset '%s' from the Hugging Face Hub", self.config.hf_path)
        return load_dataset(self.config.hf_path, cache_dir=self.cache_dir)

    def _unify_columns(self, ds: DatasetDict) -> DatasetDict:
        """Rename columns to match :class:`DataConfig` declarations."""

        rename: dict[str, str] = {}
        sample = next(iter(ds.values()))
        if "text" in sample.column_names and self.config.text_column != "text":
            rename["text"] = self.config.text_column
        if "label" in sample.column_names and self.config.label_column != "label":
            rename["label"] = self.config.label_column
        if not rename:
            return ds
        return DatasetDict({name: split.rename_columns(rename) for name, split in ds.items()})

    def _stratified_split(self, train: Dataset) -> DatasetDict:
        """Carve a stratified validation split off the training set."""

        if self.config.validation_size <= 0.0:
            return DatasetDict({"train": train, "validation": train.select(range(0))})
        split = train.train_test_split(
            test_size=self.config.validation_size,
            stratify_by_column=self.config.label_column,
            seed=42,
        )
        return DatasetDict({"train": split["train"], "validation": split["test"]})

    def _resolve_label_names(self, split: Dataset) -> tuple[str, ...]:
        """Return label names from the HF metadata when available."""

        feature = split.features.get(self.config.label_column)
        if isinstance(feature, ClassLabel):
            return tuple(feature.names)
        return tuple(self.config.label_names)

    @staticmethod
    def _clean_text(text: str, *, lower: bool) -> str:
        cleaned = _WHITESPACE.sub(" ", text).strip()
        return cleaned.lower() if lower else cleaned
