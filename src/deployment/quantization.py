"""INT8 quantization of an exported ONNX model.

The quantization is performed with ``optimum.onnxruntime``'s dynamic
quantizer, which yields a sub-200 MB model that meets WBS task 8.2's size
budget without any calibration data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.io_utils import ensure_dir
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


@dataclass
class QuantizationReport:
    """Outcome of an INT8 dynamic quantization."""

    input_dir: Path
    output_dir: Path
    quantized_model_path: Path


def quantize_int8(
    onnx_dir: Path | str,
    *,
    output_dir: Path | str,
) -> QuantizationReport:
    """Apply INT8 dynamic quantization to ``onnx_dir`` and write the result."""

    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    src_dir = Path(onnx_dir)
    dst_dir = ensure_dir(output_dir)
    _logger.info("Quantising %s to INT8 in %s", src_dir, dst_dir)
    quantizer = ORTQuantizer.from_pretrained(src_dir)
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(save_dir=str(dst_dir), quantization_config=qconfig)
    return QuantizationReport(
        input_dir=src_dir,
        output_dir=dst_dir,
        quantized_model_path=dst_dir / "model_quantized.onnx",
    )
