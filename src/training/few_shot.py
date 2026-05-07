"""Few-shot training utilities for the SetFit phase.

The functions in this module sample a balanced *N-way K-shot* training set
(``K`` examples per class) and run a SetFit training session. Repeating the
sampling step with multiple seeds yields the data-efficiency learning curve
required by WBS task 6.3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset, concatenate_datasets

from src.evaluation.metrics import compute_metrics
from src.models.setfit_model import SetFitClassifier, SetFitTrainingConfig
from src.utils.io_utils import ensure_dir, save_json
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class FewShotResult:
    """Outcome of a single few-shot training run."""

    samples_per_class: int
    seed: int
    metrics: dict[str, float]


def sample_per_class(
    dataset: Dataset,
    *,
    label_column: str,
    samples_per_class: int,
    seed: int,
) -> Dataset:
    """Return a balanced subset with ``samples_per_class`` examples per label.

    The method draws from each class without replacement using a NumPy
    :class:`Generator` seeded by ``seed`` so repeated runs are deterministic.
    """

    rng = np.random.default_rng(seed)
    labels = np.asarray(dataset[label_column])
    indices: list[int] = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"Class {label} has only {len(candidates)} examples, "
                f"cannot sample {samples_per_class}."
            )
        chosen = rng.choice(candidates, size=samples_per_class, replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return dataset.select(indices)


def run_learning_curve(
    train: Dataset,
    validation: Dataset,
    test: Dataset,
    *,
    text_column: str,
    label_column: str,
    samples: tuple[int, ...] = (8, 16, 32, 64),
    seeds: tuple[int, ...] = (13, 42, 73),
    output_dir: Path | str,
    base_config: SetFitTrainingConfig | None = None,
) -> pd.DataFrame:
    """Run SetFit at every ``(samples_per_class, seed)`` combination.

    Returns a long-form DataFrame with one row per run, suitable for
    plotting with ``seaborn.lineplot``.
    """

    output_dir = ensure_dir(output_dir)
    rows: list[dict[str, float | int]] = []
    for k in samples:
        for seed in seeds:
            run_name = f"setfit_k{k}_seed{seed}"
            _logger.info("Few-shot run: %s", run_name)
            train_subset = sample_per_class(
                train, label_column=label_column, samples_per_class=k, seed=seed
            )
            cfg = (base_config or SetFitTrainingConfig()).__class__(
                **{**(base_config or SetFitTrainingConfig()).__dict__, "seed": seed}
            )
            classifier = SetFitClassifier(
                cfg, text_column=text_column, label_column=label_column
            )
            classifier.fit(train_subset, validation=validation)
            preds = classifier.predict(test[text_column])
            metrics = compute_metrics(np.asarray(test[label_column]), preds)
            run_dir = ensure_dir(output_dir / run_name)
            classifier.save(run_dir / "model")
            save_json(metrics, run_dir / "metrics.json")
            rows.append(
                {
                    "samples_per_class": k,
                    "seed": seed,
                    **metrics,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "learning_curve.csv", index=False)
    return table
