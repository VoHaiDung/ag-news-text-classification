"""Phase 4 entry point: fine-tune transformer models.

Mapped Work Breakdown Structure tasks:

* 4.1.1 Install transformers, datasets, accelerate (validated by Phase 1)
* 4.1.2 Write reusable train.py with the Trainer API (delegated to ``src.training.hf_trainer``)
* 4.2.1 Prepare data loaders and tokenisation
* 4.2.2 Train and save checkpoints (DeBERTa-v3)
* 4.2.3 Evaluate on the validation set
* 4.3.1 Adapt training script for ModernBERT
* 4.3.2 Train and save checkpoints (ModernBERT)
* 4.3.3 Evaluate and compare with DeBERTa
* 4.4.1 Define search space (LR, BS, epochs)
* 4.4.2 Run tuning experiments (Optuna / W&B Sweeps)
* 4.4.3 Select best configuration and retrain final model
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.configs import ExperimentConfig, load_config
from src.data import AGNewsLoader
from src.models.transformers import build_classification_model
from src.training.hf_trainer import TransformerTrainer
from src.training.sweeps import run_optuna_search
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
    parser = argparse.ArgumentParser(description="Phase 4 - transformer fine-tuning.")
    parser.add_argument(
        "--data-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "ag_news.yaml",
    )
    parser.add_argument(
        "--model-configs",
        nargs="+",
        type=Path,
        default=[
            PROJECT_ROOT / "configs" / "models" / "deberta_v3_small.yaml",
            PROJECT_ROOT / "configs" / "models" / "modernbert_base.yaml",
        ],
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "transformers")
    parser.add_argument(
        "--sweep",
        type=Path,
        default=None,
        help="Optional Optuna sweep YAML; runs Phase 4.4 instead of training the listed models.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _merge_configs(data_cfg: ExperimentConfig, model_cfg: ExperimentConfig) -> ExperimentConfig:
    """Combine a data-only YAML with a model-only YAML into one config."""

    return replace(
        data_cfg,
        name=f"{data_cfg.name}__{model_cfg.name}",
        description=model_cfg.description or data_cfg.description,
        model=model_cfg.model,
        training=model_cfg.training,
    )


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    output_dir = ensure_dir(args.output_dir)

    if args.sweep is not None:
        _logger.info("Running Optuna sweep from %s", args.sweep)
        run_optuna_search(args.sweep, output_dir=output_dir / "sweep")
        return 0

    data_cfg = load_config(args.data_config)
    set_global_seed(data_cfg.training.seed)

    loader = AGNewsLoader(data_cfg.data, cache_dir=Path(data_cfg.paths.cache_dir))
    splits = loader.load()

    rows: list[dict[str, float | str]] = []
    for model_yaml in args.model_configs:
        model_cfg = load_config(model_yaml)
        merged = _merge_configs(data_cfg, model_cfg)
        bundle = build_classification_model(merged.model, merged.data)

        tracker = build_tracker(
            backend=merged.tracking.backend,
            project=merged.tracking.project,
            entity=merged.tracking.entity,
            run_name=merged.name,
            config={"phase": "phase4_transformers", "model_config": str(model_yaml)},
        )

        trainer = TransformerTrainer(
            tokenizer=bundle.tokenizer,
            model=bundle.model,
            data_cfg=merged.data,
            training_cfg=merged.training,
            label_names=splits.label_names,
            output_dir=output_dir,
            run_name=merged.name,
            report_to="wandb" if merged.tracking.backend == "wandb" else "none",
        )
        try:
            result = trainer.fit_and_evaluate(splits.train, splits.validation, splits.test)
            tracker.log_metrics(result.metrics)
            rows.append({"model": merged.name, **result.metrics})
        finally:
            tracker.finish()

    comparison = pd.DataFrame(rows).set_index("model")
    comparison_path = output_dir / "comparison.csv"
    comparison.to_csv(comparison_path)
    save_json(rows, output_dir / "comparison.json")
    _logger.info("Phase 4 complete. Comparison table at %s", comparison_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
