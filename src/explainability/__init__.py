"""Model explainability: SHAP and LIME wrappers for classification models.

The wrappers in this package take any callable that maps a list of strings
to a probability matrix (``(n_samples, n_classes)``) and return saved
visualisations under ``outputs/xai/``. This abstraction means the same
explainer code works for the TF-IDF baseline, the transformer fine-tune
and the SetFit head.
"""

from src.explainability.lime_explainer import LimeTextExplainer, explain_lime
from src.explainability.shap_explainer import ShapTextExplainer, explain_shap

__all__ = [
    "ShapTextExplainer",
    "explain_shap",
    "LimeTextExplainer",
    "explain_lime",
]
