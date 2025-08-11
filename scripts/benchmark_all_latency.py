# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Benchmark INT8 inference latency for the full 12-model lineup.

Reads the manifest written by :mod:`scripts.export_all_int8` and runs
:func:`src.evaluation.latency.benchmark_inference` on each INT8 ONNX
model. Used to populate the Pareto-front table in Section 3.5.3 of the
final report.

Usage::

    python -m scripts.benchmark_all_latency \\
        --manifest outputs/deployment_all_int8/manifest.json \\
        --output outputs/evaluation/latency_pareto.csv
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.latency import benchmark_inference

ROOT = Path(__file__).resolve().parents[1]

# Representative AG News test inputs of varying length (drawn once and
# fixed for cross-model comparability).
SAMPLE_TEXTS: tuple[str, ...] = (
    "Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling band of ultra-cynics, are seeing green again.",
    "Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment firm Carlyle Group, which has a reputation for making well-timed and occasionally controversial plays in the defense industry.",
    "Oil and Economy Cloud Stocks' Outlook (Reuters) Reuters - Soaring crude prices plus worries about the economy and the outlook for earnings are expected to hang over the stock market next week.",
    "Iraq Halts Oil Exports from Main Southern Pipeline (Reuters) Reuters - Authorities have halted oil export flows from the main pipeline in southern Iraq after intelligence showed a rebel militia could strike infrastructure.",
    "Stocks End Up, But Near Year Lows (Reuters) Reuters - Stocks ended slightly higher on Friday but stayed near lows for the year as oil prices surged past $46 a barrel.",
    "Money Funds Fell in Latest Week (AP) AP - Assets of the nation's retail money market mutual funds fell by $1.17 billion in the latest week to $849.98 trillion, the Investment Company Institute said Thursday.",
    "Fed minutes show dissent over inflation (USATODAY.com) USATODAY.com - Retail sales bounced back a bit in July, and new claims for jobless benefits fell last week, the government said Thursday.",
    "Safety Net (Forbes.com) Forbes.com - After earning a PhD in Sociology, Danny Bazil Riley started to work as the general manager.",
)


def _cpu_info() -> dict[str, str]:
    """Best-effort CPU identification for the report."""

    info: dict[str, str] = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "machine": platform.machine(),
    }
    try:
        # Linux: /proc/cpuinfo gives a nicer model name than platform.processor()
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except (FileNotFoundError, OSError):
        pass
    try:
        out = subprocess.check_output(["nproc"], text=True).strip()
        info["nproc"] = out
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return info


def _build_inference_fn(int8_dir: Path, *, max_length: int):
    """Return a callable that runs INT8 ONNX inference on a list of texts."""

    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(int8_dir)
    ort_model = ORTModelForSequenceClassification.from_pretrained(
        int8_dir, file_name="model_quantized.onnx"
    )

    def infer(texts):
        enc = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        outputs = ort_model(**enc)
        return outputs.logits

    return infer


def main() -> int:
    parser = argparse.ArgumentParser(
        description="INT8 latency benchmark across the 12-model lineup."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "outputs" / "deployment_all_int8" / "manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "evaluation" / "latency_pareto.csv",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Untimed warm-up iterations per model.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Timed iterations per model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size (O5 target is defined at batch 1).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="If set, only benchmark models whose slug matches one of the given prefixes.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    cpu_info = _cpu_info()
    print("CPU info:", cpu_info)
    print(
        f"Running batch={args.batch_size}, warmup={args.warmup}, "
        f"iterations={args.iterations}, max_length={args.max_length}"
    )

    rows: list[dict] = []
    manifest_root = args.manifest.resolve().parent
    for entry in manifest:
        if args.only and not any(entry["slug"].startswith(prefix) for prefix in args.only):
            continue
        # Resolve the INT8 directory relative to the manifest so the tree
        # can be moved between machines (the manifest stores absolute paths
        # from the host where it was built).
        int8_dir = manifest_root / entry["slug"] / "int8"
        if not (int8_dir / "model_quantized.onnx").exists():
            # Fall back to the manifest's recorded path for legacy layouts.
            fallback = Path(entry["int8_dir"])
            if (fallback / "model_quantized.onnx").exists():
                int8_dir = fallback
            else:
                print(f"[SKIP] {entry['slug']}: INT8 artefact missing at {int8_dir}")
                continue
        print(f"\n=== Benchmarking {entry['slug']} ===")
        infer = _build_inference_fn(int8_dir, max_length=args.max_length)
        report = benchmark_inference(
            infer,
            SAMPLE_TEXTS,
            backend="onnx_int8_cpu",
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        print(
            f"  mean={report.mean_ms:.2f} ms, median={report.median_ms:.2f} ms, "
            f"p95={report.p95_ms:.2f} ms, p99={report.p99_ms:.2f} ms, "
            f"throughput={report.throughput_samples_per_sec:.1f} samples/s"
        )
        rows.append(
            {
                "slug": entry["slug"],
                "language": entry["language"],
                "architecture": entry["architecture"],
                "int8_size_mb": entry.get("int8_size_mb"),
                "test_f1_macro": entry.get("test_f1_macro"),
                "mean_ms": round(report.mean_ms, 2),
                "median_ms": round(report.median_ms, 2),
                "p95_ms": round(report.p95_ms, 2),
                "p99_ms": round(report.p99_ms, 2),
                "throughput_samples_per_sec": round(report.throughput_samples_per_sec, 2),
                "o5_met": bool(report.mean_ms <= 50.0),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)

    # Companion JSON with environment metadata so the report can cite the
    # exact CPU model on which the table was produced.
    meta_path = args.output.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "cpu_info": cpu_info,
                "batch_size": args.batch_size,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "max_length": args.max_length,
                "n_models": len(rows),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote latency table to {args.output} ({len(rows)} models)")
    print(f"Wrote environment metadata to {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
