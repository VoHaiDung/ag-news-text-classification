"""Factory functions that materialise a tokenizer and a classification model.

Routing through a single factory means the rest of the codebase does not
need to special-case ``DeBERTa``, ``ModernBERT``, ``mDeBERTa`` or ``XLM-R``:
the differences (slow vs. fast tokenizer, attention dropout naming) are
absorbed here.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from src.configs import DataConfig, ModelConfig
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class TransformerBundle:
    """Pair of tokenizer and classification model."""

    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel


def build_tokenizer(model_cfg: ModelConfig) -> PreTrainedTokenizerBase:
    """Instantiate the tokenizer associated with ``model_cfg``."""

    name = model_cfg.tokenizer_name or model_cfg.name
    _logger.info("Loading tokenizer '%s'", name)
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def build_classification_model(
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
) -> TransformerBundle:
    """Build a classification model and its tokenizer.

    The HF ``AutoConfig`` is overridden with the project's ``num_labels``,
    ``id2label`` and ``label2id`` so that downstream tools (Trainer, ONNX
    export, the Gradio demo) display human-readable labels without further
    plumbing.
    """

    tokenizer = build_tokenizer(model_cfg)
    config = AutoConfig.from_pretrained(
        model_cfg.name,
        num_labels=data_cfg.num_labels,
        id2label={i: name for i, name in enumerate(data_cfg.label_names)},
        label2id={name: i for i, name in enumerate(data_cfg.label_names)},
        hidden_dropout_prob=model_cfg.dropout,
        attention_probs_dropout_prob=model_cfg.dropout,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.name,
        config=config,
    )
    _logger.info(
        "Built classification model '%s' (%d labels, dropout=%.2f).",
        model_cfg.name,
        data_cfg.num_labels,
        model_cfg.dropout,
    )
    return TransformerBundle(tokenizer=tokenizer, model=model)
