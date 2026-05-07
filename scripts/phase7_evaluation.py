"""Phase 7 entry point: evaluation and explainability.

Mapped Work Breakdown Structure tasks:

* 7.1.1 Compute accuracy and F1-macro for all models
* 7.1.2 Generate the confusion matrix per model
* 7.2   Measure confidence calibration (ECE) and the reliability diagram
* 7.3   Benchmark inference latency and throughput
* 7.4.1 Set up the SHAP TextExplainer
* 7.4.2 Generate SHAP visualisations for hard examples
* 7.5.1 Set up the LIME text explainer
* 7.5.2 Generate LIME explanations for samples
* 7.6.1 Analyse errors by class (confusion patterns)
* 7.6.2 Analyse errors by text length and topic
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.evaluation.calibration import expected_calibration_error, plot_reliability_diagram
from src.evaluation.error_analysis import save_error_analysis
from src.evaluation.latency import benchmark_inference
from src.evaluation.metrics import classification_report_table, compute_metrics, confusion_matrix_table
from src.explainability.lime_explainer import explain_lime
from src.explainability.shap_explainer import explain_shap
from src.utils import configure_logging, ensure_dir, get_logger, save_json
from src.utils.paths import OUTPUTS_DIR

_logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 7 - evaluation and XAI.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing a HuggingFace classification model checkpoint.",
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        required=True,
        help="CSV produced by Phase 4/5 with columns text, label, label_name, "
        "prediction, prediction_name, max_probability.",
    )
    parser.add_argument(
        "--probabilities-npy",
        type=Path,
        required=True,
        help="NumPy file produced by Phase 4/5 with the test probability matrix.",
    )
    parser.add_argument(
        "--label-names",
        nargs="+",
        default=["World", "Sports", "Business", "Sci/Tech"],
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "evaluation")
    parser.add_argument("--shap-samples", type=int, default=10)
    parser.add_argument("--lime-samples", type=int, default=10)
    parser.add_argument("--latency-iters", type=int, default=200)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _build_predict_fn(model_dir: Path):
    """Construct a callable that maps texts to a probability matrix."""

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        truncation=True,
        max_length=256,
    )

    def predict_proba(texts):
        outputs = pipe(list(texts))
        n_classes = len(outputs[0])
        matrix = np.zeros((len(outputs), n_classes), dtype=np.float64)
        label2id = model.config.label2id
        for row, scores in enumerate(outputs):
            for entry in scores:
                idx = label2id[entry["label"]]
                matrix[row, idx] = entry["score"]
        return matrix

    return predict_proba, pipe


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    output_dir = ensure_dir(args.output_dir)

    predictions = pd.read_csv(args.predictions_csv)
    probabilities = np.load(args.probabilities_npy)
    label_names = tuple(args.label_names)

    # 7.1 - headline metrics ----------------------------------------------------------
    metrics = compute_metrics(predictions["label"], predictions["prediction"])
    save_json(metrics, output_dir / "metrics.json")

    report = classification_report_table(
        predictions["label"], predictions["prediction"], label_names=label_names
    )
    report.to_csv(output_dir / "classification_report.csv", index=False)

    cm = confusion_matrix_table(
        predictions["label"], predictions["prediction"], label_names=label_names
    )
    cm.to_csv(output_dir / "confusion_matrix.csv")

    # 7.2 - calibration ---------------------------------------------------------------
    calibration = expected_calibration_error(predictions["label"], probabilities, n_bins=15)
    save_json(
        {"ece": calibration.ece, "mce": calibration.mce, "n_bins": calibration.n_bins},
        output_dir / "calibration.json",
    )
    calibration.bin_table.to_csv(output_dir / "calibration_bins.csv", index=False)
    plot_reliability_diagram(
        calibration, output_path=output_dir / "reliability_diagram.png"
    )

    # 7.3 - latency benchmark ---------------------------------------------------------
    predict_fn, pipe = _build_predict_fn(args.model_dir)
    latency = benchmark_inference(
        lambda batch: pipe(batch),
        predictions["text"].sample(min(len(predictions), 256), random_state=0).tolist(),
        backend="pytorch",
        batch_size=1,
        warmup=10,
        iterations=args.latency_iters,
    )
    save_json(latency.as_dict(), output_dir / "latency_pytorch.json")

    # 7.4 - SHAP ----------------------------------------------------------------------
    hardest = predictions.sort_values("max_probability", ascending=False).head(args.shap_samples)
    explain_shap(
        hardest["text"].tolist(),
        predict_fn=predict_fn,
        label_names=label_names,
        output_dir=output_dir / "shap",
    )

    # 7.5 - LIME ----------------------------------------------------------------------
    lime_subset = predictions.sample(min(len(predictions), args.lime_samples), random_state=0)
    explain_lime(
        lime_subset["text"].tolist(),
        predict_fn=predict_fn,
        label_names=label_names,
        output_dir=output_dir / "lime",
    )

    # 7.6 - error analysis ------------------------------------------------------------
    save_error_analysis(predictions, output_dir / "errors")

    _logger.info("Phase 7 complete. Artefacts under %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
