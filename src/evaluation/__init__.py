"""Evaluation primitives: metrics, calibration, latency and error analysis.

All metric helpers live in :mod:`src.evaluation.metrics`. Calibration
diagnostics (Expected Calibration Error, reliability diagrams) live in
:mod:`src.evaluation.calibration`. Inference-latency benchmarking lives in
:mod:`src.evaluation.latency`. Confusion-matrix-driven and length-bucket
error analyses live in :mod:`src.evaluation.error_analysis`.
"""

from src.evaluation.metrics import classification_report_table, compute_metrics

__all__ = ["compute_metrics", "classification_report_table"]
