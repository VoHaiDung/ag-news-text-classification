"""Phase 3 entry point: classical baselines for AG News.

Mapped Work Breakdown Structure tasks:

* 3.1.1 Build TF-IDF vectoriser pipeline
* 3.1.2 Train Logistic Regression model
* 3.1.3 Train Linear SVM model
* 3.2.1 Prepare FastText data format
* 3.2.2 Train and validate FastText model
* 3.3.1 Run all baselines on the test set
* 3.3.2 Build comparison table and log to W&B

The script trains TF-IDF + Logistic Regression, TF-IDF + Linear SVM, and
FastText, and writes a single ``comparison.csv`` table summarising the test
metrics. When a tracking backend is configured every metric is forwarded
to it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.configs import load_config
from src.data import AGNewsLoader
from src.models.baselines import FastTextClassifier, TfidfClassifier
from src.models.baselines.fasttext_baseline import FastTextConfig
from src.models.baselines.tfidf import TfidfConfig
from src.training.baseline_trainer import BaselineTrainer
from src.utils import (
    build_tracker,
    configure_logging,
    ensure_dir,
    get_logger,
    save_json,
    set_global_seed,
)
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT

_logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 - classical baselines.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "ag_news.yaml",
        help="Dataset configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "baselines",
        help="Directory for baseline artefacts.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["tfidf_logreg", "tfidf_svm", "fasttext"],
        help="Subset of baselines to train.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _build_baselines(seed: int) -> dict[str, object]:
    """Instantiate the three baseline models."""

    return {
        "tfidf_logreg": TfidfClassifier(TfidfConfig(classifier="logreg", seed=seed)),
        "tfidf_svm": TfidfClassifier(TfidfConfig(classifier="svm", seed=seed)),
        "fasttext": FastTextClassifier(FastTextConfig(seed=seed)),
    }


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    set_global_seed(config.training.seed)

    loader = AGNewsLoader(config.data, cache_dir=Path(config.paths.cache_dir))
    splits = loader.load(normalise_whitespace=True, lower=True)

    output_dir = ensure_dir(args.output_dir)
    trainer = BaselineTrainer(
        text_column=config.data.text_column,
        label_column=config.data.label_column,
        label_names=splits.label_names,
        output_dir=output_dir,
    )

    tracker = build_tracker(
        backend=config.tracking.backend,
        project=config.tracking.project,
        entity=config.tracking.entity,
        run_name=config.tracking.run_name or "phase3_baselines",
        config={"phase": "phase3_baselines"},
    )

    candidates = _build_baselines(config.training.seed)
    selected = {name: candidates[name] for name in args.models if name in candidates}
    if not selected:
        raise SystemExit(f"No known baselines among {args.models}.")

    rows: list[dict[str, float | str]] = []
    try:
        for name, model in selected.items():
            result = trainer.run(name, model, splits.train, splits.validation, splits.test)
            tracker.log_metrics({f"{name}/{k}": v for k, v in result.metrics.items()})
            rows.append({"model": name, **result.metrics})
    finally:
        tracker.finish()

    comparison = pd.DataFrame(rows).set_index("model")
    comparison_path = output_dir / "comparison.csv"
    comparison.to_csv(comparison_path)
    save_json(rows, output_dir / "comparison.json")
    _logger.info("Phase 3 complete. Comparison table at %s", comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
