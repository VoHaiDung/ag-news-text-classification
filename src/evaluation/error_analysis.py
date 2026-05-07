"""Error-analysis helpers driven by predictions and metadata.

WBS task 7.6 calls for two complementary views:

* class-level confusion patterns (which classes are confused for which);
* error rate as a function of text length (and, optionally, BERTopic topic).

The functions below return :class:`pandas.DataFrame` objects so that the
output is trivially saved to CSV and rendered as a Markdown table in the
final report.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io_utils import ensure_dir


def errors_by_class(
    predictions: pd.DataFrame,
    *,
    label_column: str = "label_name",
    prediction_column: str = "prediction_name",
    top_k: int = 10,
) -> pd.DataFrame:
    """Return the most frequent (true_label, predicted_label) error pairs."""

    errors = predictions.loc[predictions[label_column] != predictions[prediction_column]]
    pairs = (
        errors.groupby([label_column, prediction_column])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    return pairs.head(top_k)


def errors_by_text_length(
    predictions: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "label",
    prediction_column: str = "prediction",
    bins: Sequence[int] = (0, 20, 40, 60, 80, 120, 200, 10_000),
) -> pd.DataFrame:
    """Bucket predictions by text length and report per-bucket error rates."""

    df = predictions.copy()
    df["length_words"] = df[text_column].str.split().map(len)
    df["correct"] = (df[label_column] == df[prediction_column]).astype(int)
    df["bucket"] = pd.cut(df["length_words"], bins=list(bins), include_lowest=True)
    table = (
        df.groupby("bucket", observed=True)
        .agg(
            count=("correct", "size"),
            accuracy=("correct", "mean"),
            avg_length_words=("length_words", "mean"),
        )
        .reset_index()
    )
    table["error_rate"] = 1.0 - table["accuracy"]
    return table


def hardest_examples(
    predictions: pd.DataFrame,
    *,
    text_column: str = "text",
    label_column: str = "label",
    prediction_column: str = "prediction",
    probability_column: str = "max_probability",
    top_k: int = 50,
) -> pd.DataFrame:
    """Return the most confidently *wrong* predictions.

    These examples are the most informative candidates for SHAP/LIME
    visualisation because the model was both confident and incorrect.
    """

    errors = predictions.loc[predictions[label_column] != predictions[prediction_column]].copy()
    errors = errors.sort_values(probability_column, ascending=False)
    return errors.head(top_k)[
        [text_column, label_column, prediction_column, probability_column]
    ].reset_index(drop=True)


def save_error_analysis(predictions: pd.DataFrame, output_dir: Path | str) -> dict[str, Path]:
    """Convenience wrapper that runs all three analyses and writes CSV files."""

    out_dir = ensure_dir(output_dir)
    artefacts: dict[str, Path] = {}

    pair_table = errors_by_class(predictions)
    pair_path = out_dir / "errors_by_class.csv"
    pair_table.to_csv(pair_path, index=False)
    artefacts["by_class"] = pair_path

    length_table = errors_by_text_length(predictions)
    length_path = out_dir / "errors_by_length.csv"
    length_table.to_csv(length_path, index=False)
    artefacts["by_length"] = length_path

    hardest = hardest_examples(predictions)
    hardest_path = out_dir / "hardest_examples.csv"
    hardest.to_csv(hardest_path, index=False)
    artefacts["hardest"] = hardest_path

    return artefacts
