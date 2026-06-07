# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Soft-voting ensemble inference on the AG News test split.

Loads N HuggingFace checkpoints, evaluates each on the standard AG News
test set (7,600 examples), averages the softmax distributions across
models, and reports the ensemble F1-macro / accuracy together with the
per-seed numbers needed for the multi-seed mean +/- std reporting style
used in NLP papers (BERT, RoBERTa, DeBERTa).

Used to populate Section 3.3.2.5 of the final report (multi-seed and
R-Drop ablation grid for ModernBERT-large on English AG News).

Usage::

    python -m scripts.ensemble_inference \\
        --models outputs/transformers/ag_news_en__modernbert_large_seed13/best \\
                 outputs/transformers/ag_news_en__modernbert_large/best \\
                 outputs/transformers/ag_news_en__modernbert_large_seed73/best \\
        --output outputs/ensembles/modernbert_large_vanilla_3seed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = ("World", "Sports", "Business", "Sci/Tech")


def _predict_softmax(
    model_dir: Path, dataset, device: str, batch_size: int, max_length: int
) -> np.ndarray:
    """Run a single checkpoint over ``dataset`` and return ``(N, C)`` softmax."""

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval().to(device)

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

    collator = DataCollatorWithPadding(tokenizer)
    loader = DataLoader(tokenised, batch_size=batch_size, collate_fn=collator)

    all_probs: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            probs = torch.softmax(logits.float(), dim=-1)
            all_probs.append(probs.cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Soft-voting ensemble inference on AG News test set."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="One or more checkpoint directories to ensemble.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "ensembles" / "ensemble",
        help="Directory to write ensemble metrics and predictions.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--dataset",
        default="ag_news",
        help="HuggingFace dataset identifier (default: ag_news English test split).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate on.",
    )
    args = parser.parse_args()

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    test_ds = load_dataset(args.dataset, split=args.split)
    labels = np.array(test_ds["label"])

    per_seed_probs: list[np.ndarray] = []
    per_seed_metrics: list[dict] = []

    for model_path in args.models:
        model_dir = Path(model_path)
        if not model_dir.is_absolute():
            model_dir = ROOT / model_dir
        print(f"\nEvaluating {model_dir.name}...")
        probs = _predict_softmax(
            model_dir, test_ds, device, args.batch_size, args.max_length
        )
        preds = probs.argmax(axis=1)
        seed_f1 = f1_score(labels, preds, average="macro")
        seed_acc = accuracy_score(labels, preds)
        per_seed_probs.append(probs)
        per_seed_metrics.append(
            {"model": model_dir.name, "test_f1_macro": float(seed_f1),
             "test_accuracy": float(seed_acc)}
        )
        print(f"  test F1-macro={seed_f1:.4f}, accuracy={seed_acc:.4f}")

    ensemble_probs = np.mean(np.stack(per_seed_probs, axis=0), axis=0)
    ensemble_preds = ensemble_probs.argmax(axis=1)
    ensemble_f1 = f1_score(labels, ensemble_preds, average="macro")
    ensemble_acc = accuracy_score(labels, ensemble_preds)

    seed_f1s = np.array([m["test_f1_macro"] for m in per_seed_metrics])
    summary = {
        "models": [m["model"] for m in per_seed_metrics],
        "per_seed_f1_macro": seed_f1s.tolist(),
        "per_seed_accuracy": [m["test_accuracy"] for m in per_seed_metrics],
        "seed_mean_f1_macro": float(seed_f1s.mean()),
        "seed_std_f1_macro": float(seed_f1s.std(ddof=1)) if len(seed_f1s) > 1 else 0.0,
        "ensemble_f1_macro": float(ensemble_f1),
        "ensemble_accuracy": float(ensemble_acc),
        "n_models": len(args.models),
        "dataset": args.dataset,
        "split": args.split,
    }
    print("\n" + "=" * 60)
    print(
        f"Per-seed F1-macro: "
        f"mean={summary['seed_mean_f1_macro']:.4f} +/- {summary['seed_std_f1_macro']:.4f}"
    )
    print(f"Ensemble (soft voting) F1-macro: {summary['ensemble_f1_macro']:.4f}")
    print(f"Ensemble accuracy: {summary['ensemble_accuracy']:.4f}")

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    report = classification_report(
        labels, ensemble_preds, target_names=LABEL_NAMES, digits=4, output_dict=True
    )
    pd.DataFrame(report).T.to_csv(output_dir / "classification_report.csv")

    cm = confusion_matrix(labels, ensemble_preds)
    pd.DataFrame(cm, index=LABEL_NAMES, columns=LABEL_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )

    pd.DataFrame(
        {
            "label": labels,
            "prediction": ensemble_preds,
            "max_probability": ensemble_probs.max(axis=1),
        }
    ).to_csv(output_dir / "test_predictions.csv", index=False)

    np.save(output_dir / "test_probabilities.npy", ensemble_probs)

    print(f"\nWrote ensemble artefacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
