"""Inference-latency benchmarking.

The benchmark runs ``warmup`` warm-up iterations to populate caches and
trigger any lazy initialisation, then measures ``iters`` timed iterations
with :func:`time.perf_counter`. Statistics are reported in milliseconds per
sample at the requested batch size.

The same helper benchmarks both PyTorch and ONNX Runtime models because the
caller passes a callable that performs inference; the function itself is
agnostic to the backend.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class LatencyReport:
    """Latency measurement summary."""

    backend: str
    batch_size: int
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    p99_ms: float
    throughput_samples_per_sec: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "backend": self.backend,
            "batch_size": self.batch_size,
            "iterations": self.iterations,
            "mean_ms_per_sample": self.mean_ms,
            "median_ms_per_sample": self.median_ms,
            "p95_ms_per_sample": self.p95_ms,
            "p99_ms_per_sample": self.p99_ms,
            "throughput_samples_per_sec": self.throughput_samples_per_sec,
        }


def benchmark_inference(
    inference_fn: Callable[[Sequence[str]], object],
    sample_texts: Sequence[str],
    *,
    backend: str,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 200,
) -> LatencyReport:
    """Benchmark ``inference_fn`` and return summary statistics.

    Parameters
    ----------
    inference_fn:
        Callable that runs one forward pass on a list of texts.
    sample_texts:
        A pool of strings to draw batches from. Should contain at least
        ``batch_size`` items.
    backend:
        Free-form label written into the report (``"pytorch"``, ``"onnx"``,
        ``"onnx_int8"``).
    batch_size:
        Number of samples per forward pass.
    warmup:
        Untimed iterations executed before measurement.
    iterations:
        Number of timed iterations.
    """

    if len(sample_texts) < batch_size:
        raise ValueError(
            f"Need at least {batch_size} sample texts to fill a batch, got {len(sample_texts)}."
        )
    pool = list(sample_texts)
    rng = np.random.default_rng(0)

    def draw_batch() -> list[str]:
        idx = rng.integers(0, len(pool), size=batch_size)
        return [pool[i] for i in idx]

    for _ in range(warmup):
        inference_fn(draw_batch())

    timings_ms = np.empty(iterations, dtype=np.float64)
    for i in range(iterations):
        batch = draw_batch()
        start = time.perf_counter()
        inference_fn(batch)
        timings_ms[i] = (time.perf_counter() - start) * 1000.0

    per_sample_ms = timings_ms / batch_size
    return LatencyReport(
        backend=backend,
        batch_size=batch_size,
        iterations=iterations,
        mean_ms=float(per_sample_ms.mean()),
        median_ms=float(np.median(per_sample_ms)),
        p95_ms=float(np.percentile(per_sample_ms, 95)),
        p99_ms=float(np.percentile(per_sample_ms, 99)),
        throughput_samples_per_sec=float(batch_size / (per_sample_ms.mean() / 1000.0)),
    )
