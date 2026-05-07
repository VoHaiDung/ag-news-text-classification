"""Exploratory Data Analysis primitives.

The functions in this module compute the descriptive statistics required by
WBS section 2.1 (class distribution, text-length statistics, n-gram tables).
They operate on either a :class:`datasets.Dataset` or a plain
:class:`pandas.DataFrame`, and they return :class:`pandas.DataFrame` objects
for easy serialisation and plotting.

Statistical conventions
-----------------------
* Token length is approximated by whitespace-separated words. This is
  deliberately tokenizer-agnostic so the same EDA can be run before any
  decision about the modelling tokenizer has been made.
* All n-gram counts use :class:`sklearn.feature_extraction.text.CountVectorizer`
  with ``stop_words='english'`` to suppress trivial English function words.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import CountVectorizer


def _to_dataframe(
    dataset: Dataset | pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    label_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Coerce ``dataset`` into a pandas DataFrame with named labels."""

    if isinstance(dataset, Dataset):
        df = dataset.to_pandas()
    else:
        df = dataset.copy()
    if label_names is not None:
        mapping = dict(enumerate(label_names))
        df["label_name"] = df[label_column].map(mapping)
    else:
        df["label_name"] = df[label_column].astype(str)
    df["text_length_chars"] = df[text_column].str.len()
    df["text_length_words"] = df[text_column].str.split().map(len)
    return df


def class_distribution(
    dataset: Dataset | pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    label_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a table of class counts, proportions and balance ratio.

    The ``imbalance_ratio`` column reports ``count(class) / count(majority)``,
    which is more informative than raw counts when comparing across splits.
    """

    df = _to_dataframe(
        dataset,
        text_column=text_column,
        label_column=label_column,
        label_names=label_names,
    )
    counts = df["label_name"].value_counts(dropna=False)
    proportions = counts / counts.sum()
    table = pd.DataFrame(
        {
            "count": counts,
            "proportion": proportions,
            "imbalance_ratio": counts / counts.max(),
        }
    )
    table.index.name = "label_name"
    return table.sort_values("count", ascending=False)


def length_statistics(
    dataset: Dataset | pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    label_names: Sequence[str] | None = None,
    quantiles: Sequence[float] = (0.5, 0.9, 0.95, 0.99),
) -> pd.DataFrame:
    """Return descriptive statistics of text length, per class and overall."""

    df = _to_dataframe(
        dataset,
        text_column=text_column,
        label_column=label_column,
        label_names=label_names,
    )
    grouped = df.groupby("label_name")["text_length_words"]
    summary = grouped.agg(["count", "mean", "std", "min", "max"])
    for q in quantiles:
        summary[f"q{int(q * 100)}"] = grouped.quantile(q)
    overall = df["text_length_words"].agg(["count", "mean", "std", "min", "max"]).to_frame().T
    for q in quantiles:
        overall[f"q{int(q * 100)}"] = np.quantile(df["text_length_words"], q)
    overall.index = pd.Index(["__overall__"], name="label_name")
    return pd.concat([summary, overall])


def top_ngrams(
    texts: Iterable[str],
    *,
    ngram_range: tuple[int, int] = (1, 1),
    top_k: int = 30,
    stop_words: str | None = "english",
    min_df: int | float = 5,
) -> pd.DataFrame:
    """Return the ``top_k`` most frequent n-grams.

    The function uses ``CountVectorizer`` rather than a manual counter to
    guarantee identical tokenisation between this exploratory analysis and
    the TF-IDF baseline that consumes it (see ``src.models.baselines.tfidf``).
    """

    vectorizer = CountVectorizer(
        ngram_range=ngram_range,
        stop_words=stop_words,
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(texts)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    vocab = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(-counts)[:top_k]
    return pd.DataFrame({"ngram": vocab[order], "count": counts[order]})


def per_class_ngrams(
    dataset: Dataset | pd.DataFrame,
    *,
    text_column: str,
    label_column: str,
    label_names: Sequence[str],
    ngram_range: tuple[int, int] = (1, 1),
    top_k: int = 20,
    stop_words: str | None = "english",
) -> pd.DataFrame:
    """Concatenate per-class n-gram tables into a single DataFrame.

    The resulting frame has columns ``label_name``, ``ngram`` and ``count``,
    which is convenient for faceted plotting in :mod:`src.data.visualization`.
    """

    df = _to_dataframe(
        dataset,
        text_column=text_column,
        label_column=label_column,
        label_names=label_names,
    )
    parts: list[pd.DataFrame] = []
    for name in label_names:
        subset = df.loc[df["label_name"] == name, text_column]
        if subset.empty:
            continue
        table = top_ngrams(
            subset, ngram_range=ngram_range, top_k=top_k, stop_words=stop_words
        )
        table.insert(0, "label_name", name)
        parts.append(table)
    return pd.concat(parts, ignore_index=True)
