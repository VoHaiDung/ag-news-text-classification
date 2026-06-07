# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Long-document inference for AG News classifiers.

The strategy follows the academic baseline established by Pappagari et al.
(2019, "Hierarchical Transformers for Long Document Classification") and
later refined by Park et al. (2022, "Efficient Classification of Long
Documents Using Transformers"):

1. If the tokenised input fits inside the encoder's positional embedding
   budget (``model.config.max_position_embeddings``), a single forward
   pass is used. ModernBERT (Warner et al., 2024) extends this budget to
   8192 tokens, which covers more than 99 % of full news articles
   without any chunking.
2. If the input exceeds the budget, the text is split into overlapping
   windows of ``window_size`` tokens with stride ``window_size // 2``,
   each window is encoded independently, and the per-window softmax
   distributions are averaged into a single document-level distribution.
   The argmax of that average is the predicted class.

The implementation is intentionally model-agnostic: it queries the
loaded encoder for its native positional budget and adapts ``window_size``
to ``min(model_limit, requested_max_length)``. This makes the same
helper usable for DeBERTa-v3 (512 tokens), XLM-RoBERTa-large (512),
mDeBERTa-v3 (512) and ModernBERT (8192) without any model-specific code.

References
----------
Pappagari, R., Zelasko, P., Villalba, J., Carmiel, Y., Dehak, N. (2019).
    Hierarchical Transformers for Long Document Classification. ASRU.
Park, H. H., Vyas, Y., Shah, K. (2022). Efficient Classification of Long
    Documents Using Transformers. ACL.
Warner, B., et al. (2024). Smarter, Better, Faster, Longer: A Modern
    Bidirectional Encoder for Fast, Memory Efficient, and Long-Context
    Fine-tuning and Inference. arXiv:2412.13663.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class LongDocumentPrediction:
    """Document-level prediction with chunk-level breakdown for auditing.

    Attributes
    ----------
    label:
        Predicted class name (e.g. ``"Sports"``).
    label_id:
        Integer class id matching ``model.config.id2label``.
    probabilities:
        Document-level softmax distribution averaged over windows; shape
        ``(num_classes,)``.
    num_tokens:
        Total number of subword tokens in the input.
    num_windows:
        Number of windows the document was split into.
    window_probabilities:
        Per-window softmax distributions; shape ``(num_windows, num_classes)``.
    label_names:
        Ordered tuple of class names matching the probability vector.
    """

    label: str
    label_id: int
    probabilities: np.ndarray
    num_tokens: int
    num_windows: int
    window_probabilities: np.ndarray
    label_names: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict:
        """Return a JSON-friendly representation."""

        return {
            "label": self.label,
            "label_id": self.label_id,
            "num_tokens": int(self.num_tokens),
            "num_windows": int(self.num_windows),
            "probabilities": {
                name: float(p) for name, p in zip(self.label_names, self.probabilities)
            },
        }


class LongDocumentClassifier:
    """Sliding-window classifier with native-context fast path.

    Parameters
    ----------
    model_dir:
        Path to a HuggingFace checkpoint directory containing both the
        tokenizer files and the model weights.
    window_size:
        Maximum tokens fed to the encoder per forward pass. Defaults to
        the encoder's native ``max_position_embeddings``; the caller can
        request a shorter window for speed.
    stride:
        Number of overlapping tokens between consecutive windows. The
        default of ``window_size // 2`` follows Pappagari (2019).
    device:
        ``"cpu"`` (default for local inference) or ``"cuda"`` when a GPU
        is available.
    """

    def __init__(
        self,
        model_dir: str | Path,
        window_size: int | None = None,
        stride: int | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.model.to(device)

        native_limit = int(getattr(self.model.config, "max_position_embeddings", 512))
        # XLM-RoBERTa / RoBERTa reserve a padding offset (position ids start
        # at ``padding_idx + 1``), so the usable position count is below the
        # nominal ``max_position_embeddings`` (e.g. 514 nominal but only 512
        # usable). ``tokenizer.model_max_length`` already accounts for this,
        # so prefer it whenever it is a sane value (fast tokenisers report a
        # 1e30 sentinel when no limit is set, which must be ignored).
        tok_max = int(getattr(self.tokenizer, "model_max_length", 0) or 0)
        if 0 < tok_max <= native_limit:
            native_limit = tok_max
        # Reserve room for the special tokens added around every window
        # ([CLS] ... [SEP]); subtracting them here keeps the decorated window
        # within the encoder's positional budget at forward time.
        self._num_special = self.tokenizer.num_special_tokens_to_add(pair=False)
        safe_limit = max(8, native_limit - self._num_special)
        self.window_size = min(window_size or safe_limit, safe_limit)
        self.stride = stride if stride is not None else self.window_size // 2
        if self.stride <= 0 or self.stride >= self.window_size:
            raise ValueError(
                f"Stride must satisfy 0 < stride < window_size "
                f"(got stride={self.stride}, window_size={self.window_size})."
            )

        id2label = self.model.config.id2label
        # ``id2label`` is a dict with string or int keys; sort by integer id
        # to obtain a deterministic class order.
        self.label_names: tuple[str, ...] = tuple(
            id2label[k] for k in sorted(id2label, key=lambda x: int(x))
        )

    def _softmax(self, logits: torch.Tensor) -> np.ndarray:
        """Stable softmax that returns a float32 numpy array."""

        return torch.softmax(logits.float(), dim=-1).cpu().numpy()

    def _forward(self, token_ids: Sequence[int]) -> np.ndarray:
        """Run a single forward pass on a list of subword token ids."""

        input_ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return self._softmax(logits)[0]

    def _window_ids(self, encoded: list[int]) -> list[list[int]]:
        """Split a token-id sequence into overlapping windows."""

        if len(encoded) <= self.window_size:
            return [encoded]
        windows: list[list[int]] = []
        start = 0
        while start < len(encoded):
            end = min(start + self.window_size, len(encoded))
            windows.append(encoded[start:end])
            if end == len(encoded):
                break
            start += self.stride
        return windows

    def classify(self, text: str) -> LongDocumentPrediction:
        """Classify ``text`` and return a chunk-aware prediction.

        For inputs that fit inside ``window_size`` this is equivalent to
        running the underlying HuggingFace pipeline once. For longer
        inputs, the per-window probabilities are mean-pooled to form a
        single document-level distribution.
        """

        encoded = self.tokenizer.encode(text, add_special_tokens=False)
        windows = self._window_ids(encoded)

        # Re-attach the model's start/end tokens around each window so the
        # CLS/EOS embeddings are not lost. Fast tokenisers do not expose
        # ``build_inputs_with_special_tokens``, so the special-token ids
        # are read directly from the tokenizer; models without explicit
        # CLS/SEP (rare for the families used in this project) get the
        # bare window.
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        if cls_id is None or sep_id is None:
            # XLM-R uses ``bos_token_id`` / ``eos_token_id`` instead of CLS/SEP.
            cls_id = self.tokenizer.bos_token_id
            sep_id = self.tokenizer.eos_token_id
        if cls_id is not None and sep_id is not None:
            decorated = [[cls_id] + w + [sep_id] for w in windows]
        else:
            decorated = [list(w) for w in windows]
        # Hard cap each decorated window at the encoder's usable positional
        # budget. ``window_size`` already excludes the special tokens, so the
        # decorated length never exceeds ``window_size + num_special``.
        max_total = self.window_size + self._num_special
        decorated = [w[:max_total] for w in decorated]

        window_probs = np.stack([self._forward(w) for w in decorated], axis=0)
        doc_probs = window_probs.mean(axis=0)
        label_id = int(np.argmax(doc_probs))

        return LongDocumentPrediction(
            label=self.label_names[label_id],
            label_id=label_id,
            probabilities=doc_probs,
            num_tokens=len(encoded),
            num_windows=len(windows),
            window_probabilities=window_probs,
            label_names=self.label_names,
        )
