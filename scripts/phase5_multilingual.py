"""Phase 5 entry point: multilingual extension to Vietnamese.

Mapped Work Breakdown Structure tasks:

* 5.1.1 Set up OPUS-MT (Helsinki-NLP) pipeline
* 5.1.2 Batch translate the full dataset and validate
* 5.2.1 VI to EN translation step (back-translation augmentation, leg 1)
* 5.2.2 EN to VI back-translation step (back-translation augmentation, leg 2)
* 5.3.1 Prepare Vietnamese DataLoader
* 5.3.2 Train mDeBERTa with original Vietnamese data
* 5.3.3 Train with back-translation augmentation
* 5.3.4 Evaluate both Vietnamese models
* 5.4.1 Cross-lingual benchmark (English test on the Vietnamese model)
* 5.4.2 Final comparison report (English vs. Vietnamese)
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets

from src.configs import load_config
from src.data import AGNewsLoader
from src.data.back_translation import BackTranslationConfig, BackTranslator
from src.data.translation import OpusMTConfig, OpusMTTranslator
from src.models.transformers import build_classification_model
from src.training.hf_trainer import TransformerTrainer
from src.utils import (
    build_tracker,
    configure_logging,
    ensure_dir,
    get_logger,
    save_json,
    set_global_seed,
)
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT, DATA_PROCESSED

_logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 5 - multilingual extension.")
    parser.add_argument(
        "--source-data-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "ag_news.yaml",
    )
    parser.add_argument(
        "--target-data-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "data" / "ag_news_vi.yaml",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "models" / "mdeberta_v3.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS_DIR / "multilingual")
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["translate", "augment", "train", "evaluate"],
        choices=["translate", "augment", "train", "evaluate"],
        help="Pipeline steps to run (in order).",
    )
    parser.add_argument(
        "--max-train-translate",
        type=int,
        default=None,
        help="Cap on the number of training examples to translate (debug aid).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _translate_split(
    split: Dataset,
    *,
    text_column: str,
    translator: OpusMTTranslator,
    output_column: str,
    cap: int | None,
) -> Dataset:
    texts = split[text_column]
    if cap is not None:
        texts = texts[:cap]
    translations = translator.translate(texts)
    if cap is not None:
        split = split.select(range(cap))
    return split.add_column(output_column, translations)


def step_translate(
    *,
    source_cfg,
    target_dir: Path,
    cap: int | None,
) -> DatasetDict:
    loader = AGNewsLoader(source_cfg.data, cache_dir=Path(source_cfg.paths.cache_dir))
    splits = loader.load(normalise_whitespace=True, lower=False)
    translator = OpusMTTranslator(OpusMTConfig())
    translated = DatasetDict(
        train=_translate_split(
            splits.train,
            text_column=source_cfg.data.text_column,
            translator=translator,
            output_column="text_vi",
            cap=cap,
        ),
        validation=_translate_split(
            splits.validation,
            text_column=source_cfg.data.text_column,
            translator=translator,
            output_column="text_vi",
            cap=None,
        ),
        test=_translate_split(
            splits.test,
            text_column=source_cfg.data.text_column,
            translator=translator,
            output_column="text_vi",
            cap=None,
        ),
    )
    ensure_dir(target_dir)
    translated.save_to_disk(str(target_dir))
    _logger.info("Wrote translated dataset to %s", target_dir)
    return translated


def step_augment(*, dataset: DatasetDict, output_dir: Path) -> DatasetDict:
    augmenter = BackTranslator(BackTranslationConfig())
    augmented_train = dataset["train"].add_column(
        "text_vi_aug", augmenter.augment(dataset["train"]["text_vi"])
    )
    augmented = DatasetDict(
        train=augmented_train,
        validation=dataset["validation"],
        test=dataset["test"],
    )
    ensure_dir(output_dir)
    augmented.save_to_disk(str(output_dir))
    _logger.info("Wrote back-translated dataset to %s", output_dir)
    return augmented


def step_train(
    *,
    source_cfg,
    target_cfg,
    model_yaml: Path,
    augmented: DatasetDict | None,
    output_dir: Path,
) -> dict[str, dict[str, float]]:
    model_cfg = load_config(model_yaml)
    merged = replace(
        target_cfg,
        name=f"{target_cfg.name}__{model_cfg.name}",
        model=model_cfg.model,
        training=model_cfg.training,
    )
    set_global_seed(merged.training.seed)

    loader = AGNewsLoader(merged.data, cache_dir=Path(merged.paths.cache_dir))
    splits = loader.load()

    metrics_by_run: dict[str, dict[str, float]] = {}

    # Run 1: Vietnamese only.
    bundle = build_classification_model(merged.model, merged.data)
    trainer = TransformerTrainer(
        tokenizer=bundle.tokenizer,
        model=bundle.model,
        data_cfg=merged.data,
        training_cfg=merged.training,
        label_names=splits.label_names,
        output_dir=output_dir,
        run_name=f"{merged.name}__vi_only",
    )
    metrics_by_run["vi_only"] = trainer.fit_and_evaluate(
        splits.train, splits.validation, splits.test
    ).metrics

    # Run 2: Vietnamese + back-translation. The augmented column ``text_vi_aug``
    # carries the paraphrased version of each training example; we copy it
    # into ``text_column`` so the result matches the schema of ``splits.train``.
    if augmented is not None and "text_vi_aug" in augmented["train"].column_names:
        text_col = merged.data.text_column
        label_col = merged.data.label_column
        flat_aug = augmented["train"].map(
            lambda example: {text_col: example["text_vi_aug"], label_col: example[label_col]},
            remove_columns=[
                c for c in augmented["train"].column_names if c not in {text_col, label_col}
            ],
        )
        combined = concatenate_datasets([splits.train, flat_aug])
        bundle = build_classification_model(merged.model, merged.data)
        trainer = TransformerTrainer(
            tokenizer=bundle.tokenizer,
            model=bundle.model,
            data_cfg=merged.data,
            training_cfg=merged.training,
            label_names=splits.label_names,
            output_dir=output_dir,
            run_name=f"{merged.name}__vi_with_bt",
        )
        metrics_by_run["vi_with_bt"] = trainer.fit_and_evaluate(
            combined, splits.validation, splits.test
        ).metrics

    return metrics_by_run


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    output_dir = ensure_dir(args.output_dir)

    source_cfg = load_config(args.source_data_config)
    target_cfg = load_config(args.target_data_config)
    set_global_seed(source_cfg.training.seed)

    translated_dir = DATA_PROCESSED / "ag_news_vi"
    augmented_dir = DATA_PROCESSED / "ag_news_vi_aug"
    translated: DatasetDict | None = None
    augmented: DatasetDict | None = None

    if "translate" in args.steps:
        translated = step_translate(
            source_cfg=source_cfg,
            target_dir=translated_dir,
            cap=args.max_train_translate,
        )
    if "augment" in args.steps:
        if translated is None:
            translated = DatasetDict.load_from_disk(str(translated_dir))
        augmented = step_augment(dataset=translated, output_dir=augmented_dir)

    metrics: dict[str, dict[str, float]] = {}
    if "train" in args.steps:
        if augmented is None and augmented_dir.exists():
            augmented = DatasetDict.load_from_disk(str(augmented_dir))
        metrics = step_train(
            source_cfg=source_cfg,
            target_cfg=target_cfg,
            model_yaml=args.model_config,
            augmented=augmented,
            output_dir=output_dir,
        )

    if "evaluate" in args.steps:
        save_json(metrics, output_dir / "comparison.json")
        if metrics:
            comparison = pd.DataFrame(metrics).T
            comparison.to_csv(output_dir / "comparison.csv")
            _logger.info("Phase 5 comparison written to %s", output_dir / "comparison.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
