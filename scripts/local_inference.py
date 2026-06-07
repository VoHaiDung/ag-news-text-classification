# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Offline inference helper for any AG News checkpoint downloaded to local.

After running ``python -m scripts.sync_from_remote --include-weights`` the
``outputs/`` directory holds a ``best/`` snapshot for every trained model.
This script loads any of them and classifies one or more text snippets on
CPU, so the user can verify every model locally without keeping the rented
cloud GPU instance alive.

Usage::

    # English transformers
    python -m scripts.local_inference --model deberta_v3_small "Apple launches new iPhone."
    python -m scripts.local_inference --model deberta_v3_base  "Apple launches new iPhone."
    python -m scripts.local_inference --model modernbert_base  "Apple launches new iPhone."
    python -m scripts.local_inference --model modernbert_large "Apple launches new iPhone."

    # Vietnamese transformers
    python -m scripts.local_inference --model mdeberta_v3 --family multilingual \\
        "Apple ra mat iPhone moi voi chip M4."
    python -m scripts.local_inference --model xlm_r_large --family multilingual \\
        "Apple ra mat iPhone moi voi chip M4."

    # SetFit (K=64)
    python -m scripts.local_inference --model setfit_64 --family setfit \\
        "Apple launches new iPhone."

    # ONNX INT8 deployment artefact (no torch needed) - twelve are available
    # under outputs/deployment_all_int8/<slug>/int8, where <slug> is e.g.
    # en__deberta_v3_small, vi__xlm_r_large__vi_with_bt, fr__mdeberta_v3__fr_only.
    python -m scripts.local_inference \\
        --onnx outputs/deployment_all_int8/en__deberta_v3_small/int8 \\
        "Apple launches new iPhone."
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = ("World", "Sports", "Business", "Sci/Tech")


def _resolve_checkpoint(family: str, model_name: str) -> Path:
    """Return the directory holding the requested ``best/`` snapshot.

    Run directories on disk carry a dataset prefix (``ag_news_en__`` for the
    English runs, ``ag_news_vi__`` for the Vietnamese runs). The user passes
    only the short model name so this helper expands the prefix and falls
    back to the literal name when no match is found.
    """

    family_to_dir = {
        "transformers": ROOT / "outputs" / "transformers",
        "multilingual": ROOT / "outputs" / "multilingual",
        "setfit": ROOT / "outputs" / "setfit",
    }
    base = family_to_dir.get(family)
    if base is None:
        sys.exit(f"Unknown family '{family}'. Choose from {list(family_to_dir)}.")
    if not base.exists():
        sys.exit(f"Output directory missing: {base}. Run scripts.sync_from_remote first.")

    # Trainer-based models save weights under ``best/``; SetFit stores them
    # at the run root because it does not use the HuggingFace Trainer
    # checkpointing layout.
    matches = [
        sub
        for sub in base.iterdir()
        if sub.is_dir() and (sub.name == model_name or sub.name.endswith(f"__{model_name}"))
    ]
    if not matches:
        # Fall back to substring match so user can disambiguate XLM-R-large
        # variants with names such as ``xlm_r_large__vi_with_bt``.
        matches = [
            sub for sub in base.iterdir() if sub.is_dir() and model_name in sub.name
        ]
    if not matches:
        available = sorted(sub.name for sub in base.iterdir() if sub.is_dir())
        sys.exit(
            f"No run matching '{model_name}' under {base}. "
            f"Available: {available}"
        )
    if len(matches) > 1:
        names = sorted(m.name for m in matches)
        sys.exit(
            f"Ambiguous model name '{model_name}', matches {names}. "
            "Pass the full directory name to disambiguate."
        )
    run_dir = matches[0]
    candidate = run_dir / "best"
    if not candidate.exists():
        candidate = run_dir
    if not candidate.exists():
        sys.exit(f"Checkpoint not found: {candidate}")
    return candidate


def _predict_transformer(checkpoint: Path, texts: list[str]) -> list[tuple[str, dict]]:
    """Run a HuggingFace transformer checkpoint on CPU."""

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    model.eval()
    results: list[tuple[str, dict]] = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", truncation=True, max_length=256, padding=True
            )
            logits = model(**inputs).logits[0].numpy()
            exps = np.exp(logits - logits.max())
            probs = exps / exps.sum()
            pred_idx = int(np.argmax(probs))
            results.append(
                (
                    LABEL_NAMES[pred_idx],
                    {name: float(p) for name, p in zip(LABEL_NAMES, probs)},
                )
            )
    return results


def _predict_setfit(checkpoint: Path, texts: list[str]) -> list[tuple[str, dict]]:
    """Run a SetFit checkpoint on CPU."""

    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(str(checkpoint))
    raw_probs = model.predict_proba(texts).cpu().numpy()
    results: list[tuple[str, dict]] = []
    for probs in raw_probs:
        pred_idx = int(np.argmax(probs))
        results.append(
            (
                LABEL_NAMES[pred_idx],
                {name: float(p) for name, p in zip(LABEL_NAMES, probs)},
            )
        )
    return results


def _predict_onnx(onnx_dir: Path, texts: list[str]) -> list[tuple[str, dict]]:
    """Run the INT8 ONNX deployment artefact on CPU without torch."""

    import onnxruntime as ort
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
    model_file = onnx_dir / "model_quantized.onnx"
    if not model_file.exists():
        model_file = onnx_dir / "model.onnx"
    session = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
    input_names = {inp.name for inp in session.get_inputs()}
    results: list[tuple[str, dict]] = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="np", truncation=True, max_length=256, padding=True
        )
        feeds = {k: v for k, v in inputs.items() if k in input_names}
        logits = session.run(None, feeds)[0][0]
        exps = np.exp(logits - logits.max())
        probs = exps / exps.sum()
        pred_idx = int(np.argmax(probs))
        results.append(
            (
                LABEL_NAMES[pred_idx],
                {name: float(p) for name, p in zip(LABEL_NAMES, probs)},
            )
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Classify text with any local AG News model.")
    parser.add_argument(
        "texts",
        nargs="+",
        help="One or more text snippets to classify.",
    )
    parser.add_argument(
        "--model",
        default="deberta_v3_small",
        help="Model name matching the directory under outputs/<family>/<model>/.",
    )
    parser.add_argument(
        "--family",
        default="transformers",
        choices=("transformers", "multilingual", "setfit"),
        help="Output family containing the model.",
    )
    parser.add_argument(
        "--onnx",
        default=None,
        help="If set, run the ONNX deployment artefact in this directory instead.",
    )
    parser.add_argument(
        "--long-document",
        action="store_true",
        help=(
            "Route the input through the sliding-window long-document classifier "
            "(Pappagari et al., 2019). Required when any input exceeds the encoder's "
            "native context window (e.g. 512 tokens for DeBERTa, 8192 for ModernBERT)."
        ),
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override the per-window token length when --long-document is set.",
    )
    args = parser.parse_args()

    if args.onnx is not None:
        if args.long_document:
            sys.exit("--long-document is not supported on the ONNX deployment artefact yet.")
        onnx_dir = Path(args.onnx)
        if not onnx_dir.is_absolute():
            onnx_dir = ROOT / onnx_dir
        results = _predict_onnx(onnx_dir, args.texts)
        backend = f"ONNX ({onnx_dir.relative_to(ROOT)})"
    else:
        checkpoint = _resolve_checkpoint(args.family, args.model)
        if args.long_document:
            if args.family == "setfit":
                sys.exit("--long-document is only available for transformer encoders.")
            results, backend = _predict_long_document(
                checkpoint, args.texts, args.window_size, args.family, args.model
            )
        elif args.family == "setfit":
            results = _predict_setfit(checkpoint, args.texts)
            backend = f"{args.family}/{args.model}"
        else:
            results = _predict_transformer(checkpoint, args.texts)
            backend = f"{args.family}/{args.model}"

    print(f"Backend: {backend}")
    print("-" * 60)
    for text, (label, probs) in zip(args.texts, results):
        preview = text if len(text) <= 200 else text[:200].rstrip() + "..."
        print(f"Input: {preview}")
        print(f"  Predicted: {label}")
        for name, p in sorted(probs.items(), key=lambda x: -x[1]):
            print(f"    {name:<10} {p:.4f}")
        print()
    return 0


def _predict_long_document(
    checkpoint: Path,
    texts: list[str],
    window_size: int | None,
    family: str,
    model_name: str,
) -> tuple[list[tuple[str, dict]], str]:
    """Classify each text with the sliding-window long-document strategy."""

    from src.inference.long_doc import LongDocumentClassifier

    classifier = LongDocumentClassifier(checkpoint, window_size=window_size)
    results: list[tuple[str, dict]] = []
    for text in texts:
        prediction = classifier.classify(text)
        probs = {
            name: float(p)
            for name, p in zip(prediction.label_names, prediction.probabilities)
        }
        results.append((prediction.label, probs))
        print(
            f"[long-doc] {prediction.num_tokens} tokens -> "
            f"{prediction.num_windows} window(s) of {classifier.window_size} "
            f"tokens (stride {classifier.stride})."
        )
    backend = (
        f"{family}/{model_name} long-document "
        f"(window={classifier.window_size}, stride={classifier.stride})"
    )
    return results, backend


if __name__ == "__main__":
    raise SystemExit(main())
