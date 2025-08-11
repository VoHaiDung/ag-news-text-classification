# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Inference utilities decoupled from training.

Modules under ``src.inference`` host inference-time logic that is not
specific to a single deployment surface (CLI helper or local Gradio
application). The current entry point is the long-document classifier,
which combines a native-context single pass for short inputs and a
sliding-window aggregation pass for inputs that exceed the encoder's
positional embedding budget.
"""

from src.inference.long_doc import (
    LongDocumentClassifier,
    LongDocumentPrediction,
)

__all__ = ["LongDocumentClassifier", "LongDocumentPrediction"]
