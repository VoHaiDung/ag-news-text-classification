# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Phase 5 entry point: multilingual extension (Vietnamese and French).

The script is invoked once per target language. With ``--target-data-config
configs/data/ag_news_vi.yaml`` (and the matching OPUS-MT model flags) it
covers the Vietnamese branch (Phase 5A in the WBS); with
``configs/data/ag_news_fr.yaml`` it covers the French branch (Phase 5B).
Output column names, dataset directories and run name suffixes are
derived from the target config so a single code path supports both
languages.

Mapped Work Breakdown Structure tasks (Phase 5A = Vietnamese,
Phase 5B = French; the same code path executes both):

* 5A.1.1 / 5B.1.1 Set up OPUS-MT (Helsinki-NLP) pipeline
* 5A.1.2 / 5B.1.2 Batch translate the full dataset and validate
* 5A.2.1 / 5B.2.1 Target -> EN translation step (back-translation leg 1)
* 5A.2.2 / 5B.2.2 EN -> target back-translation step (leg 2)
* 5A.3.1 / 5B.3.1 Prepare target-language DataLoader
* 5A.3.2 / 5B.3.2 Train mDeBERTa with target-only data
* 5A.3.3            Train mDeBERTa with back-translation augmentation
* 5A.3.4 / 5B.3.3 Scale-up: train XLM-R-large (target-only and target+BT)
* 5A.3.5 / 5B.3.4 Evaluate the 2x2 ablation (encoder x BT)
* 5A.4.1 / 5B.4.1 Cross-lingual benchmark (EN test on target model)
* 5A.4.2            Bilingual comparison report (EN vs. VI)
* 5B.4.2            Tri-lingual comparison report (EN vs. VI vs. FR)
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
    parser = argparse.ArgumentParser(
        description="Phase 5 - multilingual extension (5A Vietnamese / 5B French)."
    )
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
    parser.add_argument(
        "--only-bt",
        action="store_true",
        help=(
            "Skip the language-only Run 1 and train only the back-translation "
            "augmented Run 2. Useful when re-running the phase to fill in a "
            "single missing cell of the 2x2 ablation table."
        ),
    )
    parser.add_argument(
        "--translate-model",
        default="Helsinki-NLP/opus-mt-en-vi",
        help="OPUS-MT model for the primary EN -> target translation step.",
    )
    parser.add_argument(
        "--bt-forward-model",
        default="Helsinki-NLP/opus-mt-vi-en",
        help="OPUS-MT model for the first leg of back-translation (target -> pivot EN).",
    )
    parser.add_argument(
        "--bt-backward-model",
        default="Helsinki-NLP/opus-mt-en-vi",
        help="OPUS-MT model for the second leg of back-translation (pivot EN -> target).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _lang_tag(target_cfg) -> str:
    """Short language tag derived from the target text column (e.g. ``vi`` / ``fr``)."""

    col = target_cfg.data.text_column
    return col.split("_", 1)[1] if col.startswith("text_") else col


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
    target_cfg,
    target_dir: Path,
    cap: int | None,
    translate_model: str,
) -> DatasetDict:
    loader = AGNewsLoader(source_cfg.data, cache_dir=Path(source_cfg.paths.cache_dir))
    splits = loader.load(normalise_whitespace=True, lower=False)
    translator = OpusMTTranslator(OpusMTConfig(model_name=translate_model))
    out_col = target_cfg.data.text_column
    translated = DatasetDict(
        train=_translate_split(
            splits.train,
            text_column=source_cfg.data.text_column,
            translator=translator,
            output_column=out_col,
            cap=cap,
        ),
        validation=_translate_split(
            splits.validation,
            text_column=source_cfg.data.text_column,
            translator=translator,
            output_column=out_col,
            cap=None,
        ),
        test=_translate_split(
            splits.test,
            text_column=source_cfg.data.text_column,
            translator=translator,
            output_column=out_col,
            cap=None,
        ),
    )
    ensure_dir(target_dir)
    translated.save_to_disk(str(target_dir))
    _logger.info("Wrote translated dataset to %s", target_dir)
    return translated


def step_augment(
    *,
    dataset: DatasetDict,
    target_cfg,
    output_dir: Path,
    bt_forward_model: str,
    bt_backward_model: str,
) -> DatasetDict:
    augmenter = BackTranslator(
        BackTranslationConfig(
            forward_model=bt_forward_model,
            backward_model=bt_backward_model,
        )
    )
    text_col = target_cfg.data.text_column
    aug_col = f"{text_col}_aug"
    augmented_train = dataset["train"].add_column(
        aug_col, augmenter.augment(dataset["train"][text_col])
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
    only_bt: bool = False,
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
    lang = _lang_tag(target_cfg)
    aug_col = f"{merged.data.text_column}_aug"

    # Run 1: target language only. Skipped when ``--only-bt`` is set so
    # that a previously completed vi_only / fr_only checkpoint is
    # preserved while only the back-translation cell is (re)trained.
    if not only_bt:
        bundle = build_classification_model(merged.model, merged.data)
        trainer = TransformerTrainer(
            tokenizer=bundle.tokenizer,
            model=bundle.model,
            data_cfg=merged.data,
            training_cfg=merged.training,
            label_names=splits.label_names,
            output_dir=output_dir,
            run_name=f"{merged.name}__{lang}_only",
        )
        metrics_by_run[f"{lang}_only"] = trainer.fit_and_evaluate(
            splits.train, splits.validation, splits.test
        ).metrics

    # Run 2: target language + back-translation. The augmented column
    # ``<text_column>_aug`` carries the paraphrased version of each
    # training example; it is copied into ``text_column`` so the result
    # matches the schema of ``splits.train``.
    if augmented is not None and aug_col in augmented["train"].column_names:
        text_col = merged.data.text_column
        label_col = merged.data.label_column
        flat_aug = augmented["train"].map(
            lambda example: {text_col: example[aug_col], label_col: example[label_col]},
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
            run_name=f"{merged.name}__{lang}_with_bt",
        )
        metrics_by_run[f"{lang}_with_bt"] = trainer.fit_and_evaluate(
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

    # Output directory names are derived from the target dataset name
    # (e.g. ``ag_news_fr`` -> ``data/processed/ag_news_fr`` and
    # ``ag_news_fr_aug``) so the same script supports Vietnamese,
    # French or any other language without code changes.
    dataset_name = target_cfg.data.name
    translated_dir = DATA_PROCESSED / dataset_name
    augmented_dir = DATA_PROCESSED / f"{dataset_name}_aug"
    translated: DatasetDict | None = None
    augmented: DatasetDict | None = None

    if "translate" in args.steps:
        translated = step_translate(
            source_cfg=source_cfg,
            target_cfg=target_cfg,
            target_dir=translated_dir,
            cap=args.max_train_translate,
            translate_model=args.translate_model,
        )
    if "augment" in args.steps:
        if translated is None:
            translated = DatasetDict.load_from_disk(str(translated_dir))
        augmented = step_augment(
            dataset=translated,
            target_cfg=target_cfg,
            output_dir=augmented_dir,
            bt_forward_model=args.bt_forward_model,
            bt_backward_model=args.bt_backward_model,
        )

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
            only_bt=args.only_bt,
        )

    if "evaluate" in args.steps:
        save_json(metrics, output_dir / "comparison.json")
        if metrics:
            comparison = pd.DataFrame(metrics).T
            comparison.to_csv(output_dir / "comparison.csv")
            _logger.info(
                "Phase 5 (%s) comparison written to %s",
                _lang_tag(target_cfg),
                output_dir / "comparison.csv",
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
