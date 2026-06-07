# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Deployment artefacts: ONNX export, INT8 quantization, Gradio demo."""

from src.deployment.onnx_export import export_to_onnx
from src.deployment.quantization import quantize_int8

__all__ = ["export_to_onnx", "quantize_int8"]
