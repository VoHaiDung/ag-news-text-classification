# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Post-hoc temperature-scaling calibration (Guo et al., 2017).

Fits a single scalar temperature ``T`` on the validation set of a trained
checkpoint, then reports the test-set ECE / MCE before and after
calibration. The technique is a one-line post-processing step that
typically reduces ECE by 50 - 70 % without touching the trained weights,
which is the recommended fix when a model only narrowly misses an ECE
target (objective O4 in this project).

Usage::

    python -m scripts.calibrate_model \\
        --model-dir outputs/transformers/ag_news_en__modernbert_large/best \\
        --output-dir outputs/evaluation/calibrated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

from src.evaluation.calibration import (
    apply_isotonic,
    apply_temperature,
    expected_calibration_error,
    fit_isotonic_calibrators,
    fit_temperature,
    plot_reliability_diagram,
)
from src.utils.io_utils import ensure_dir, save_json

ROOT = Path(__file__).resolve().parents[1]


def _logits_on_split(
    model_dir: Path,
    split: str,
    *,
    batch_size: int,
    max_length: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the model on ``split`` and return ``(logits, labels)`` as numpy."""

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval().to(device)

    dataset = load_dataset("ag_news", split=split)
    labels = np.array(dataset["label"], dtype=np.int64)

    tokenised = dataset.map(
        lambda batch: tokenizer(
            batch["text"], truncation=True, max_length=max_length
        ),
        batched=True,
    )
    tokenised = tokenised.remove_columns(
        [c for c in tokenised.column_names if c not in {"input_ids", "attention_mask"}]
    )
    tokenised.set_format("torch")
    loader = DataLoader(
        tokenised,
        batch_size=batch_size,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )

    all_logits: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            all_logits.append(logits.float().cpu().numpy())
    return np.concatenate(all_logits, axis=0), labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-hoc temperature-scaling calibration.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT
        / "outputs"
        / "transformers"
        / "ag_news_en__modernbert_large"
        / "best",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "evaluation" / "calibrated",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--n-bins", type=int, default=15)
    parser.add_argument(
        "--val-split",
        default="train[90%:]",
        help=(
            "HuggingFace split spec used as the held-out validation slice. "
            "Default matches the 90/10 stratified split used by the trainer."
        ),
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_dir = args.model_dir
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    output_dir = ensure_dir(args.output_dir)

    print("Collecting validation logits...")
    val_logits, val_labels = _logits_on_split(
        model_dir, args.val_split, batch_size=args.batch_size,
        max_length=args.max_length, device=device,
    )
    print(f"  val: {val_logits.shape[0]} examples")

    print("Collecting test logits...")
    test_logits, test_labels = _logits_on_split(
        model_dir, "test", batch_size=args.batch_size,
        max_length=args.max_length, device=device,
    )
    print(f"  test: {test_logits.shape[0]} examples")

    # 1. Baseline (un-calibrated softmax).
    baseline_probs = apply_temperature(test_logits, 1.0)
    baseline = expected_calibration_error(test_labels, baseline_probs, n_bins=args.n_bins)
    print(f"\nBaseline ECE = {baseline.ece:.4f}, MCE = {baseline.mce:.4f}")

    # 2. Temperature scaling (Guo et al., 2017).
    temperature = fit_temperature(val_logits, val_labels)
    temperature_probs = apply_temperature(test_logits, temperature)
    temperature_rep = expected_calibration_error(
        test_labels, temperature_probs, n_bins=args.n_bins
    )
    print(
        f"\nTemperature scaling: T = {temperature:.4f} -> "
        f"ECE = {temperature_rep.ece:.4f}, MCE = {temperature_rep.mce:.4f}"
    )

    # 3. Class-wise isotonic regression (Zadrozny and Elkan, 2002).
    val_probs = apply_temperature(val_logits, 1.0)
    isotonic_calibrators = fit_isotonic_calibrators(val_probs, val_labels)
    isotonic_probs = apply_isotonic(baseline_probs, isotonic_calibrators)
    isotonic_rep = expected_calibration_error(
        test_labels, isotonic_probs, n_bins=args.n_bins
    )
    print(
        f"Isotonic regression: ECE = {isotonic_rep.ece:.4f}, "
        f"MCE = {isotonic_rep.mce:.4f}"
    )

    # 4. Temperature -> Isotonic (apply isotonic on temperature-scaled probs).
    val_probs_t = apply_temperature(val_logits, temperature)
    isotonic_calibrators_t = fit_isotonic_calibrators(val_probs_t, val_labels)
    chained_probs = apply_isotonic(temperature_probs, isotonic_calibrators_t)
    chained_rep = expected_calibration_error(
        test_labels, chained_probs, n_bins=args.n_bins
    )
    print(
        f"Temperature + Isotonic: ECE = {chained_rep.ece:.4f}, "
        f"MCE = {chained_rep.mce:.4f}"
    )

    summary = {
        "model_dir": str(model_dir),
        "n_bins": args.n_bins,
        "n_val": int(val_logits.shape[0]),
        "n_test": int(test_logits.shape[0]),
        "ece_target_o4": 0.03,
        "calibrators": {
            "baseline": {
                "ece": float(baseline.ece),
                "mce": float(baseline.mce),
                "o4_met": bool(baseline.ece <= 0.03),
            },
            "temperature_scaling": {
                "temperature": float(temperature),
                "ece": float(temperature_rep.ece),
                "mce": float(temperature_rep.mce),
                "o4_met": bool(temperature_rep.ece <= 0.03),
            },
            "isotonic_regression": {
                "ece": float(isotonic_rep.ece),
                "mce": float(isotonic_rep.mce),
                "o4_met": bool(isotonic_rep.ece <= 0.03),
            },
            "temperature_then_isotonic": {
                "ece": float(chained_rep.ece),
                "mce": float(chained_rep.mce),
                "o4_met": bool(chained_rep.ece <= 0.03),
            },
        },
    }
    save_json(summary, output_dir / "calibration_summary.json")

    baseline.bin_table.to_csv(output_dir / "bins_baseline.csv", index=False)
    temperature_rep.bin_table.to_csv(output_dir / "bins_temperature.csv", index=False)
    isotonic_rep.bin_table.to_csv(output_dir / "bins_isotonic.csv", index=False)
    chained_rep.bin_table.to_csv(output_dir / "bins_temperature_then_isotonic.csv", index=False)

    plot_reliability_diagram(
        baseline,
        output_path=output_dir / "reliability_baseline.png",
        title="Reliability (uncalibrated)",
    )
    plot_reliability_diagram(
        temperature_rep,
        output_path=output_dir / "reliability_temperature.png",
        title=f"Reliability (temperature scaling, T = {temperature:.3f})",
    )
    plot_reliability_diagram(
        isotonic_rep,
        output_path=output_dir / "reliability_isotonic.png",
        title="Reliability (isotonic regression)",
    )
    plot_reliability_diagram(
        chained_rep,
        output_path=output_dir / "reliability_temperature_then_isotonic.png",
        title=f"Reliability (T then isotonic, T = {temperature:.3f})",
    )

    print(f"\nArtefacts written to {output_dir}")
    best = min(summary["calibrators"].items(), key=lambda kv: kv[1]["ece"])
    print(
        f"Best calibrator: {best[0]} "
        f"(ECE = {best[1]['ece']:.4f}, O4 met = {best[1]['o4_met']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
