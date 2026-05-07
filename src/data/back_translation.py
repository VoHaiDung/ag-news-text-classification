"""Back-translation augmentation (VI -> EN -> VI).

WBS task 5.2 augments the Vietnamese training set by routing each sentence
through an English pivot. The aim is paraphrase-style augmentation that
preserves the topic label while introducing surface variation.

The implementation reuses :class:`OpusMTTranslator` for both legs so the
quality and dependency footprint of the augmentation matches the primary
EN -> VI translation step.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from src.data.translation import OpusMTConfig, OpusMTTranslator
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class BackTranslationConfig:
    """Configuration for the two-leg pivot translation pipeline."""

    forward_model: str = "Helsinki-NLP/opus-mt-vi-en"
    backward_model: str = "Helsinki-NLP/opus-mt-en-vi"
    max_length: int = 512
    batch_size: int = 16
    num_beams: int = 4
    device: str = "cuda"


class BackTranslator:
    """Apply VI -> EN -> VI pivoting to a list of strings."""

    def __init__(self, config: BackTranslationConfig | None = None) -> None:
        self.config = config or BackTranslationConfig()
        common = dict(
            max_length=self.config.max_length,
            batch_size=self.config.batch_size,
            num_beams=self.config.num_beams,
            device=self.config.device,
        )
        self._forward = OpusMTTranslator(
            OpusMTConfig(model_name=self.config.forward_model, **common)
        )
        self._backward = OpusMTTranslator(
            OpusMTConfig(model_name=self.config.backward_model, **common)
        )

    def augment(self, texts: Iterable[str]) -> list[str]:
        """Return back-translated versions of ``texts`` (one per input)."""

        texts = list(texts)
        _logger.info("Back-translating %d examples through English.", len(texts))
        intermediate = self._forward.translate(texts)
        return self._backward.translate(intermediate)
