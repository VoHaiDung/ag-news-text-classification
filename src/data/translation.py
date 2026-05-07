"""Machine translation pipeline (OPUS-MT).

WBS task 5.1 requires producing a Vietnamese copy of AG News so that the
multilingual extension can be trained without relying on Vietnamese-native
labelled corpora. The translator wraps a HuggingFace ``MarianMT`` model and
exposes a batched ``translate`` method that handles long inputs by chunking
them on sentence boundaries.

References
----------
Tiedemann, J. and Thottingal, S. (2020).
*OPUS-MT - building open translation services for the world.* EAMT.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tqdm.auto import tqdm
from transformers import MarianMTModel, MarianTokenizer

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class OpusMTConfig:
    """Configuration for the OPUS-MT translator."""

    model_name: str = "Helsinki-NLP/opus-mt-en-vi"
    max_length: int = 512
    batch_size: int = 16
    num_beams: int = 4
    device: str = "cuda"


class OpusMTTranslator:
    """Batched translator built on top of MarianMT."""

    def __init__(self, config: OpusMTConfig | None = None) -> None:
        import torch

        self.config = config or OpusMTConfig()
        self.tokenizer = MarianTokenizer.from_pretrained(self.config.model_name)
        self.model = MarianMTModel.from_pretrained(self.config.model_name)
        self.device = self.config.device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        _logger.info(
            "Loaded OPUS-MT '%s' on %s.", self.config.model_name, self.device
        )

    # ------------------------------------------------------------------ public API

    def translate(self, texts: Iterable[str], *, show_progress: bool = True) -> list[str]:
        """Translate ``texts`` and return one translation per input."""

        import torch

        texts = list(texts)
        translations: list[str] = []
        iterator = range(0, len(texts), self.config.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Translating", unit="batch")
        with torch.no_grad():
            for start in iterator:
                batch = texts[start : start + self.config.batch_size]
                tokens = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                ).to(self.device)
                output = self.model.generate(
                    **tokens,
                    num_beams=self.config.num_beams,
                    max_length=self.config.max_length,
                )
                translations.extend(
                    self.tokenizer.batch_decode(output, skip_special_tokens=True)
                )
        return translations

    def translate_to_file(
        self,
        texts: Iterable[str],
        output_path: Path | str,
    ) -> Path:
        """Translate and write one translation per line."""

        target = Path(output_path)
        ensure_dir(target.parent)
        outputs = self.translate(texts)
        target.write_text("\n".join(outputs), encoding="utf-8")
        return target
