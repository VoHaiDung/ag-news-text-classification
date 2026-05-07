"""Phase 2 entry point: Exploratory Data Analysis.

Mapped Work Breakdown Structure tasks:

* 2.1.1.1 Compute class counts and balance ratio
* 2.1.1.2 Plot text-length histogram per class
* 2.1.2.1 Generate word clouds for each class
* 2.1.2.2 Top n-gram analysis (uni/bi/trigrams)
* 2.2.1   Set up Cleanlab pipeline
* 2.2.2   Run analysis and dump suspected noisy labels (CSV)
* 2.3.1   Train BERTopic model on the dataset
* 2.3.2   Visualise and persist discovered topics

Each sub-task writes one or more artefacts under ``outputs/eda/`` so the
analysis is fully reproducible from disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.configs import load_config
from src.data import AGNewsLoader
from src.data.cleanlab_audit import audit
from src.data.eda import (
    class_distribution,
    length_statistics,
    per_class_ngrams,
    top_ngrams,
)
from src.data.topic_modeling import run_bertopic
from src.data.visualization import (
    plot_class_distribution,
    plot_length_histogram,
    plot_top_ngrams,
    plot_word_clouds,
)
from src.utils import configure_logging, ensure_dir, get_logger, save_json, set_global_seed
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT

_logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 - exploratory data analysis.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "ag_news.yaml",
        help="YAML configuration describing the dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "eda",
        help="Directory that receives all EDA artefacts.",
    )
    parser.add_argument(
        "--topic-sample-size",
        type=int,
        default=20_000,
        help="Number of training examples to sample for BERTopic (full data is slow).",
    )
    parser.add_argument(
        "--skip-cleanlab",
        action="store_true",
        help="Skip the Cleanlab audit (useful when iterating on plots only).",
    )
    parser.add_argument(
        "--skip-bertopic",
        action="store_true",
        help="Skip the BERTopic step.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    set_global_seed(config.training.seed)

    output_dir = ensure_dir(args.output_dir)
    figures_dir = ensure_dir(output_dir / "figures")
    tables_dir = ensure_dir(output_dir / "tables")

    loader = AGNewsLoader(config.data, cache_dir=Path(config.paths.cache_dir))
    splits = loader.load(normalise_whitespace=True, lower=False)
    train_df = splits.train.to_pandas()
    train_df["label_name"] = train_df[config.data.label_column].map(
        dict(enumerate(splits.label_names))
    )

    # 2.1.1 - class distribution and length statistics ---------------------------------
    distribution = class_distribution(
        splits.train,
        text_column=config.data.text_column,
        label_column=config.data.label_column,
        label_names=splits.label_names,
    )
    distribution.to_csv(tables_dir / "class_distribution.csv")
    plot_class_distribution(distribution, output_path=figures_dir / "class_distribution.png")

    length_stats = length_statistics(
        splits.train,
        text_column=config.data.text_column,
        label_column=config.data.label_column,
        label_names=splits.label_names,
    )
    length_stats.to_csv(tables_dir / "length_statistics.csv")
    plot_length_histogram(
        train_df,
        column="text_length_words",
        output_path=figures_dir / "length_histogram.png",
    )

    # 2.1.2 - word clouds and n-gram tables --------------------------------------------
    texts_per_class = {
        name: " ".join(
            train_df.loc[train_df["label_name"] == name, config.data.text_column].tolist()
        )
        for name in splits.label_names
    }
    plot_word_clouds(
        texts_per_class,
        output_path=figures_dir / "word_clouds.png",
        extra_stopwords=("reuters", "ap", "afp", "ldquo", "rdquo"),
    )
    for n_min, n_max, name in [(1, 1, "unigram"), (2, 2, "bigram"), (3, 3, "trigram")]:
        table = per_class_ngrams(
            splits.train,
            text_column=config.data.text_column,
            label_column=config.data.label_column,
            label_names=splits.label_names,
            ngram_range=(n_min, n_max),
            top_k=20,
        )
        table.to_csv(tables_dir / f"top_{name}s.csv", index=False)
        plot_top_ngrams(table, output_path=figures_dir / f"top_{name}s.png", title=f"Top {name}s")

    overall_table = top_ngrams(
        train_df[config.data.text_column], ngram_range=(1, 2), top_k=50
    )
    overall_table.to_csv(tables_dir / "top_ngrams_overall.csv", index=False)

    # 2.2 - Cleanlab audit -------------------------------------------------------------
    if not args.skip_cleanlab:
        report = audit(
            train_df[config.data.text_column].to_numpy(),
            train_df[config.data.label_column].to_numpy(),
            label_names=splits.label_names,
        )
        report.save(output_dir / "cleanlab")

    # 2.3 - BERTopic -------------------------------------------------------------------
    if not args.skip_bertopic:
        if args.topic_sample_size and args.topic_sample_size < len(train_df):
            sample_idx = np.random.RandomState(config.training.seed).choice(
                len(train_df), size=args.topic_sample_size, replace=False
            )
            sample_texts = train_df[config.data.text_column].to_numpy()[sample_idx].tolist()
        else:
            sample_texts = train_df[config.data.text_column].tolist()
        topic_report = run_bertopic(sample_texts, seed=config.training.seed)
        topic_report.save(output_dir / "bertopic")

    summary = {
        "config": str(args.config),
        "num_train": len(splits.train),
        "num_val": len(splits.validation),
        "num_test": len(splits.test),
        "label_names": list(splits.label_names),
        "artefacts_dir": str(output_dir),
    }
    save_json(summary, output_dir / "summary.json")
    _logger.info("Phase 2 EDA complete. Artefacts under %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
