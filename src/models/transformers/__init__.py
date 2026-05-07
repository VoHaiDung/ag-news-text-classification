"""Transformer model wrappers.

Each wrapper exposes the same minimal interface used by the training and
evaluation pipelines: it returns a HuggingFace
:class:`PreTrainedModel` together with the matching
:class:`PreTrainedTokenizerBase`.
"""

from src.models.transformers.factory import (
    TransformerBundle,
    build_classification_model,
    build_tokenizer,
)

__all__ = ["TransformerBundle", "build_classification_model", "build_tokenizer"]
