"""LIME explanations for text classifiers.

LIME (Ribeiro et al., 2016) approximates the local decision boundary of a
classifier with a linear model trained on perturbations of the input. For
text data, the perturbation operator removes individual tokens; the
returned explanation lists the contribution of each token to the predicted
class.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lime.lime_text import LimeTextExplainer as _LimeBackend

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class LimeTextExplainer:
    """Wrapper around :class:`lime.lime_text.LimeTextExplainer`."""

    predict_fn: Callable[[Sequence[str]], np.ndarray]
    label_names: tuple[str, ...]
    num_features: int = 10
    num_samples: int = 1000

    def __post_init__(self) -> None:
        self._backend = _LimeBackend(class_names=list(self.label_names))

    def explain(self, text: str, *, top_labels: int = 1):  # type: ignore[no-untyped-def]
        return self._backend.explain_instance(
            text,
            self.predict_fn,
            num_features=self.num_features,
            num_samples=self.num_samples,
            top_labels=top_labels,
        )


def explain_lime(
    texts: Sequence[str],
    *,
    predict_fn: Callable[[Sequence[str]], np.ndarray],
    label_names: tuple[str, ...],
    output_dir: Path | str,
    num_features: int = 10,
    num_samples: int = 1000,
) -> dict[str, Path]:
    """Generate one HTML explanation per text and return their paths."""

    out_dir = ensure_dir(output_dir)
    explainer = LimeTextExplainer(
        predict_fn=predict_fn,
        label_names=label_names,
        num_features=num_features,
        num_samples=num_samples,
    )
    paths: dict[str, Path] = {}
    for index, text in enumerate(texts):
        explanation = explainer.explain(text)
        html_path = out_dir / f"sample_{index:03d}.html"
        html_path.write_text(explanation.as_html(), encoding="utf-8")
        paths[f"sample_{index:03d}"] = html_path
    _logger.info("Wrote %d LIME explanations to %s", len(paths), out_dir)
    return paths
