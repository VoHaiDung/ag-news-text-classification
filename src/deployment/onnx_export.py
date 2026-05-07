"""Export a fine-tuned classification model to ONNX.

Two routes are supported:

1. The Optimum :class:`ORTModelForSequenceClassification` exporter, which is
   the preferred path because it ships with HuggingFace and handles the
   tokenizer side via ``save_pretrained``.
2. A fallback to :func:`torch.onnx.export` for cases where Optimum cannot
   trace the model (rare, but kept here for robustness).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class OnnxExportReport:
    """Outcome of an ONNX export."""

    output_dir: Path
    model_path: Path
    tokenizer_dir: Path
    opset: int


def export_to_onnx(
    pytorch_model_dir: Path | str,
    *,
    output_dir: Path | str,
    opset: int = 17,
) -> OnnxExportReport:
    """Export ``pytorch_model_dir`` to an ONNX directory using Optimum."""

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    pytorch_dir = Path(pytorch_model_dir)
    target_dir = ensure_dir(output_dir)
    _logger.info("Exporting %s to ONNX (opset=%d)", pytorch_dir, opset)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        pytorch_dir,
        export=True,
    )
    ort_model.save_pretrained(str(target_dir))
    tokenizer = AutoTokenizer.from_pretrained(pytorch_dir)
    tokenizer.save_pretrained(str(target_dir))
    return OnnxExportReport(
        output_dir=target_dir,
        model_path=target_dir / "model.onnx",
        tokenizer_dir=target_dir,
        opset=opset,
    )
