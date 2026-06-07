# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Batch export of the full 12-model lineup to ONNX INT8.

Used to populate Section 3.5.3 of the final report (latency-accuracy
Pareto front across the four English encoders, four Vietnamese encoders
and four French encoders). Each checkpoint is first exported to FP32
ONNX via :func:`src.deployment.onnx_export.export_to_onnx` and then
dynamically quantised to INT8 via
:func:`src.deployment.quantization.quantize_int8`.

Usage::

    python -m scripts.export_all_int8 \\
        --output-root outputs/deployment_all_int8
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from src.deployment.onnx_export import export_to_onnx
from src.deployment.quantization import quantize_int8

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelEntry:
    """One row of the 12-model lineup."""

    slug: str
    language: str
    architecture: str
    checkpoint: Path
    test_f1_macro: float | None  # populated from saved metrics.json when available


def _checkpoint(rel: str) -> Path:
    return ROOT / rel


# The 12-model lineup (4 EN + 4 VI + 4 FR). F1 values are taken from the
# trained checkpoints' ``metrics.json`` so the final Pareto-front report
# does not need to re-evaluate accuracy.
LINEUP: list[ModelEntry] = [
    # ---------------- English ----------------
    ModelEntry(
        slug="en__deberta_v3_small",
        language="en",
        architecture="DeBERTa-v3-small",
        checkpoint=_checkpoint("outputs/transformers/ag_news_en__deberta_v3_small/best"),
        test_f1_macro=None,
    ),
    ModelEntry(
        slug="en__deberta_v3_base",
        language="en",
        architecture="DeBERTa-v3-base",
        checkpoint=_checkpoint("outputs/transformers/ag_news_en__deberta_v3_base/best"),
        test_f1_macro=None,
    ),
    ModelEntry(
        slug="en__modernbert_base",
        language="en",
        architecture="ModernBERT-base",
        checkpoint=_checkpoint("outputs/transformers/ag_news_en__modernbert_base/best"),
        test_f1_macro=None,
    ),
    ModelEntry(
        slug="en__modernbert_large",
        language="en",
        architecture="ModernBERT-large",
        checkpoint=_checkpoint("outputs/transformers/ag_news_en__modernbert_large/best"),
        test_f1_macro=None,
    ),
    # ---------------- Vietnamese ----------------
    ModelEntry(
        slug="vi__mdeberta_v3__vi_only",
        language="vi",
        architecture="mDeBERTa-v3-base",
        checkpoint=_checkpoint("outputs/multilingual/ag_news_vi__mdeberta_v3__vi_only/best"),
        test_f1_macro=None,
    ),
    ModelEntry(
        slug="vi__mdeberta_v3__vi_with_bt",
        language="vi",
        architecture="mDeBERTa-v3-base + back-translation",
        checkpoint=_checkpoint("outputs/multilingual/ag_news_vi__mdeberta_v3__vi_with_bt/best"),
        test_f1_macro=None,
    ),
    ModelEntry(
        slug="vi__xlm_r_large__vi_only",
        language="vi",
        architecture="XLM-R-large",
        checkpoint=_checkpoint("outputs/multilingual/ag_news_vi__xlm_r_large__vi_only/best"),
        test_f1_macro=None,
    ),
    ModelEntry(
        slug="vi__xlm_r_large__vi_with_bt",
        language="vi",
        architecture="XLM-R-large + back-translation",
        checkpoint=_checkpoint("outputs/multilingual/ag_news_vi__xlm_r_large__vi_with_bt/best"),
        test_f1_macro=None,
    ),
    # ---------------- French ----------------
    ModelEntry(
        slug="fr__mdeberta_v3__fr_only",
        language="fr",
        architecture="mDeBERTa-v3-base",
        checkpoint=_checkpoint("outputs/multilingual/ag_news_fr__mdeberta_v3__fr_only/best"),
        test_f1_macro=None,
    ),
    ModelEntry(
        slug="fr__mdeberta_v3__fr_with_bt",
        language="fr",
        architecture="mDeBERTa-v3-base + back-translation",
        checkpoint=_checkpoint("outputs/multilingual/ag_news_fr__mdeberta_v3__fr_with_bt/best"),
        test_f1_macro=None,
    ),
    ModelEntry(
        slug="fr__xlm_r_large__fr_only",
        language="fr",
        architecture="XLM-R-large",
        checkpoint=_checkpoint("outputs/multilingual/ag_news_fr__xlm_r_large__fr_only/best"),
        test_f1_macro=None,
    ),
    ModelEntry(
        slug="fr__xlm_r_large__fr_with_bt",
        language="fr",
        architecture="XLM-R-large + back-translation",
        checkpoint=_checkpoint("outputs/multilingual/ag_news_fr__xlm_r_large__fr_with_bt/best"),
        test_f1_macro=None,
    ),
]


def _load_test_f1(checkpoint_dir: Path) -> float | None:
    """Try to read the test F1-macro from ``metrics.json`` next to ``best/``."""

    metrics_path = checkpoint_dir.parent / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict) and "test_f1_macro" in data:
        return float(data["test_f1_macro"])
    return None


def _dir_size_mb(path: Path) -> float:
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch ONNX + INT8 export of the full 12-model lineup."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs" / "deployment_all_int8",
        help="Destination root for FP32 / INT8 artefacts (one sub-dir per model).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models whose ``int8`` artefact already exists.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="If set, only export models whose slug matches one of the given prefixes.",
    )
    args = parser.parse_args()

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for entry in LINEUP:
        if args.only and not any(entry.slug.startswith(prefix) for prefix in args.only):
            continue
        if not entry.checkpoint.exists():
            print(f"[SKIP] {entry.slug}: checkpoint not found at {entry.checkpoint}")
            continue

        model_dir = output_root / entry.slug
        onnx_dir = model_dir / "onnx"
        int8_dir = model_dir / "int8"
        int8_model = int8_dir / "model_quantized.onnx"

        if args.skip_existing and int8_model.exists():
            print(f"[SKIP] {entry.slug}: INT8 already exists at {int8_model}")
            int8_size_mb = _dir_size_mb(int8_dir)
        else:
            print(f"\n=== Exporting {entry.slug} ===")
            t0 = time.perf_counter()
            export_to_onnx(entry.checkpoint, output_dir=onnx_dir)
            t_onnx = time.perf_counter() - t0

            t1 = time.perf_counter()
            quantize_int8(onnx_dir, output_dir=int8_dir)
            t_quant = time.perf_counter() - t1
            print(
                f"  ONNX export: {t_onnx:.1f}s, INT8 quantise: {t_quant:.1f}s, "
                f"INT8 size: {_dir_size_mb(int8_dir):.1f} MB"
            )
            int8_size_mb = _dir_size_mb(int8_dir)

        test_f1 = entry.test_f1_macro or _load_test_f1(entry.checkpoint)
        rows.append(
            {
                "slug": entry.slug,
                "language": entry.language,
                "architecture": entry.architecture,
                "checkpoint": str(entry.checkpoint),
                "int8_dir": str(int8_dir),
                "int8_size_mb": round(int8_size_mb, 1),
                "test_f1_macro": test_f1,
            }
        )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nWrote manifest with {len(rows)} entries to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
