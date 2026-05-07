"""Phase 6 entry point: few-shot learning with SetFit.

Mapped Work Breakdown Structure tasks:

* 6.1.1 Install SetFit and sentence-transformers (validated by Phase 1)
* 6.1.2 Write SetFit training script (delegated to ``src.training.few_shot``)
* 6.2.1 Train with 8 and 16 samples per class
* 6.2.2 Train with 32 and 64 samples per class
* 6.2.3 Evaluate all SetFit models
* 6.3.1 Compute accuracy versus sample size
* 6.3.2 Plot learning curve and write the analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.configs import load_config
from src.data import AGNewsLoader
from src.models.setfit_model import SetFitTrainingConfig
from src.training.few_shot import run_learning_curve
from src.utils import configure_logging, ensure_dir, get_logger, set_global_seed
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT

sns.set_theme(context="paper", style="whitegrid")
_logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 6 - few-shot learning with SetFit.")
    parser.add_argument(
        "--data-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "ag_news.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models" / "setfit.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "setfit")
    parser.add_argument(
        "--shots",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Samples-per-class values to train at.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[13, 42, 73],
        help="Random seeds; one SetFit run per (shots, seed).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _plot_curve(table, output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.lineplot(
        data=table,
        x="samples_per_class",
        y="f1_macro",
        marker="o",
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(table["samples_per_class"].unique()))
    ax.set_xticklabels(sorted(table["samples_per_class"].unique()))
    ax.set_xlabel("Samples per class")
    ax.set_ylabel("F1-macro")
    ax.set_title("SetFit data-efficiency learning curve")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    output_dir = ensure_dir(args.output_dir)

    data_cfg = load_config(args.data_config)
    model_cfg = load_config(args.model_config)
    set_global_seed(model_cfg.training.seed)

    loader = AGNewsLoader(data_cfg.data, cache_dir=Path(data_cfg.paths.cache_dir))
    splits = loader.load()

    base_config = SetFitTrainingConfig(
        model_name=model_cfg.model.name,
        num_epochs=model_cfg.training.epochs,
        batch_size=model_cfg.training.batch_size,
        seed=model_cfg.training.seed,
    )
    table = run_learning_curve(
        splits.train,
        splits.validation,
        splits.test,
        text_column=data_cfg.data.text_column,
        label_column=data_cfg.data.label_column,
        samples=tuple(args.shots),
        seeds=tuple(args.seeds),
        output_dir=output_dir,
        base_config=base_config,
    )
    plot_path = output_dir / "learning_curve.png"
    _plot_curve(table, plot_path)
    _logger.info("Phase 6 complete. Curve at %s", plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
