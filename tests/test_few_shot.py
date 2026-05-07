"""Tests for the few-shot sampling utility."""

from __future__ import annotations

import numpy as np
import pytest
from datasets import Dataset

from src.training.few_shot import sample_per_class


def _toy_dataset(per_class: int = 50, n_classes: int = 4) -> Dataset:
    rng = np.random.default_rng(0)
    texts = [f"sample {i}" for i in range(per_class * n_classes)]
    labels: list[int] = []
    for label in range(n_classes):
        labels.extend([label] * per_class)
    rng.shuffle(labels)
    return Dataset.from_dict({"text": texts, "label": labels})


def test_sample_per_class_balanced() -> None:
    dataset = _toy_dataset()
    subset = sample_per_class(dataset, label_column="label", samples_per_class=8, seed=42)
    counts = np.bincount(np.asarray(subset["label"]))
    assert (counts == 8).all()


def test_sample_per_class_deterministic() -> None:
    dataset = _toy_dataset()
    a = sample_per_class(dataset, label_column="label", samples_per_class=4, seed=123)
    b = sample_per_class(dataset, label_column="label", samples_per_class=4, seed=123)
    assert a["text"] == b["text"]


def test_sample_per_class_raises_when_too_small() -> None:
    dataset = _toy_dataset(per_class=3)
    with pytest.raises(ValueError):
        sample_per_class(dataset, label_column="label", samples_per_class=8, seed=0)
