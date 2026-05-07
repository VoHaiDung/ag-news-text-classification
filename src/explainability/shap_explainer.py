"""SHAP explanations for text classifiers.

SHAP (Lundberg and Lee, 2017) attributes the prediction to each input token
using Shapley values from cooperative game theory. The wrapper relies on
``shap.Explainer`` with a transformers ``Pipeline`` or a custom callable.
For long-running runs the explainer is constructed once and reused across
multiple texts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import shap

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class ShapTextExplainer:
    """Wrapper around :class:`shap.Explainer` for text classification."""

    predict_fn: Callable[[Sequence[str]], np.ndarray]
    label_names: tuple[str, ...]
    masker: shap.maskers.Text | None = None

    def __post_init__(self) -> None:
        if self.masker is None:
            self.masker = shap.maskers.Text(r"\W+")
        self._explainer = shap.Explainer(
            self.predict_fn,
            self.masker,
            output_names=list(self.label_names),
        )

    def explain(self, texts: Sequence[str]):  # type: ignore[no-untyped-def]
        """Return SHAP ``Explanation`` objects for ``texts``."""

        return self._explainer(list(texts))


def explain_shap(
    texts: Sequence[str],
    *,
    predict_fn: Callable[[Sequence[str]], np.ndarray],
    label_names: tuple[str, ...],
    output_dir: Path | str,
) -> dict[str, Path]:
    """Convenience function that explains ``texts`` and saves HTML plots."""

    out_dir = ensure_dir(output_dir)
    explainer = ShapTextExplainer(predict_fn=predict_fn, label_names=label_names)
    explanations = explainer.explain(texts)
    paths: dict[str, Path] = {}
    for index, explanation in enumerate(explanations):
        html = shap.plots.text(explanation, display=False)
        target = out_dir / f"sample_{index:03d}.html"
        target.write_text(html, encoding="utf-8")
        paths[f"sample_{index:03d}"] = target
    _logger.info("Wrote %d SHAP explanations to %s", len(paths), out_dir)
    return paths
