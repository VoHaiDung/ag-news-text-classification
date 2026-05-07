"""Plotting helpers used by the EDA notebook and the Phase 2 script.

Each function writes a single figure to disk and returns the path. The
figures are sized for inclusion in the SIC report template (16:9 aspect
ratio at 150 dpi). Headless execution is supported by setting the
``Agg`` backend at import time.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # safe in headless / CI environments
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from wordcloud import STOPWORDS, WordCloud

from src.utils.io_utils import ensure_dir

sns.set_theme(context="paper", style="whitegrid")
_FIGSIZE = (10, 6)
_DPI = 150


def plot_class_distribution(
    table: pd.DataFrame,
    *,
    output_path: Path | str,
    title: str = "Class distribution",
) -> Path:
    """Plot a horizontal bar chart of class counts and save it to disk."""

    output = Path(output_path)
    ensure_dir(output.parent)
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    ordered = table.sort_values("count", ascending=True)
    ax.barh(ordered.index.astype(str), ordered["count"], color=sns.color_palette("deep"))
    ax.set_xlabel("Number of examples")
    ax.set_ylabel("Class")
    ax.set_title(title)
    for index, value in enumerate(ordered["count"].values):
        ax.text(value, index, f" {int(value):,}", va="center")
    fig.tight_layout()
    fig.savefig(output, dpi=_DPI)
    plt.close(fig)
    return output


def plot_length_histogram(
    dataframe: pd.DataFrame,
    *,
    column: str,
    output_path: Path | str,
    by: str | None = "label_name",
    bins: int = 60,
    clip_quantile: float = 0.99,
    title: str = "Text length distribution",
) -> Path:
    """Plot a histogram of text length, optionally faceted by class."""

    output = Path(output_path)
    ensure_dir(output.parent)
    upper = dataframe[column].quantile(clip_quantile)
    data = dataframe.loc[dataframe[column] <= upper]
    fig, ax = plt.subplots(figsize=_FIGSIZE)
    if by is not None and by in dataframe.columns:
        sns.histplot(data=data, x=column, hue=by, bins=bins, ax=ax, element="step", stat="density")
    else:
        sns.histplot(data=data, x=column, bins=bins, ax=ax, stat="density")
    ax.set_xlabel("Length (words)")
    ax.set_ylabel("Density")
    ax.set_title(f"{title} (clipped at the {clip_quantile:.0%} quantile)")
    fig.tight_layout()
    fig.savefig(output, dpi=_DPI)
    plt.close(fig)
    return output


def plot_word_clouds(
    texts_per_class: dict[str, str],
    *,
    output_path: Path | str,
    extra_stopwords: Sequence[str] = (),
) -> Path:
    """Render one word cloud per class on a single figure.

    Parameters
    ----------
    texts_per_class:
        Mapping ``label_name -> concatenated text``.
    extra_stopwords:
        Optional task-specific stop words to remove (for example, generic
        wire-service tokens such as ``reuters`` and ``afp``).
    """

    output = Path(output_path)
    ensure_dir(output.parent)
    stopwords = set(STOPWORDS) | set(extra_stopwords)
    n = len(texts_per_class)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4.5))
    axes_iter = axes.flatten() if n > 1 else [axes]
    for ax, (label, text) in zip(axes_iter, texts_per_class.items()):
        if not text.strip():
            ax.axis("off")
            continue
        cloud = WordCloud(
            width=1200,
            height=600,
            background_color="white",
            stopwords=stopwords,
            collocations=False,
        ).generate(text)
        ax.imshow(cloud, interpolation="bilinear")
        ax.set_title(label)
        ax.axis("off")
    for ax in axes_iter[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output, dpi=_DPI)
    plt.close(fig)
    return output


def plot_top_ngrams(
    table: pd.DataFrame,
    *,
    output_path: Path | str,
    title: str = "Top n-grams by class",
) -> Path:
    """Plot a faceted bar chart of the top n-grams per class."""

    output = Path(output_path)
    ensure_dir(output.parent)
    classes = table["label_name"].unique()
    cols = min(2, len(classes))
    rows = (len(classes) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7, rows * 4.5), squeeze=False)
    for ax, name in zip(axes.flatten(), classes):
        sub = table[table["label_name"] == name].sort_values("count", ascending=True)
        ax.barh(sub["ngram"], sub["count"], color=sns.color_palette("deep")[0])
        ax.set_title(name)
        ax.set_xlabel("Count")
    for ax in axes.flatten()[len(classes) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=_DPI)
    plt.close(fig)
    return output
