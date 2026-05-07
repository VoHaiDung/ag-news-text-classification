"""Unsupervised topic discovery with BERTopic.

WBS task 2.3 asks for a topic map of the four AG News classes. The aim is
diagnostic rather than predictive: knowing how many semantic clusters each
class actually contains helps interpret the supervised confusion matrix
later in Phase 7.

The default embedding backbone is ``all-MiniLM-L6-v2`` because it strikes a
good balance between speed and quality on a CPU; the function signature lets
the caller substitute a heavier model when GPU is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class TopicReport:
    """Outputs of a BERTopic run."""

    model: BERTopic
    topics: list[int]
    topic_table: pd.DataFrame

    def save(self, directory: Path | str) -> dict[str, Path]:
        """Persist the model and an interactive HTML topic map."""

        out_dir = ensure_dir(directory)
        model_dir = out_dir / "bertopic_model"
        ensure_dir(model_dir)
        self.model.save(str(model_dir), serialization="safetensors")
        table_path = out_dir / "topic_table.csv"
        self.topic_table.to_csv(table_path, index=False)
        try:
            html_path = out_dir / "topic_map.html"
            self.model.visualize_topics().write_html(str(html_path))
        except Exception as exc:  # pragma: no cover - depends on plotly availability
            _logger.warning("Could not render topic map HTML: %s", exc)
            html_path = None
        artefacts = {"model": model_dir, "table": table_path}
        if html_path is not None:
            artefacts["html"] = html_path
        return artefacts


def run_bertopic(
    texts: list[str],
    *,
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size: int = 50,
    nr_topics: int | str | None = "auto",
    seed: int = 42,
) -> TopicReport:
    """Fit BERTopic on ``texts`` and return a :class:`TopicReport`.

    Parameters
    ----------
    texts:
        List of input documents (one per training example).
    embedding_model_name:
        Hugging Face name of the sentence-transformer encoder.
    min_topic_size:
        Lower bound on cluster size; smaller clusters are merged into the
        residual ``-1`` topic.
    nr_topics:
        ``"auto"`` lets BERTopic decide (recommended), an integer caps the
        number of topics, ``None`` disables the reduction step.
    """

    embedder = SentenceTransformer(embedding_model_name)
    model = BERTopic(
        embedding_model=embedder,
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = model.fit_transform(texts)
    table = model.get_topic_info()
    _logger.info("BERTopic discovered %d topics (excluding noise).", (table["Topic"] >= 0).sum())
    return TopicReport(model=model, topics=topics, topic_table=table)
