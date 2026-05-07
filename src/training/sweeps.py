"""Optuna-driven hyper-parameter search.

The implementation is intentionally framework-light: it reads a YAML
description of the search space, deep-copies the base
:class:`ExperimentConfig` for each trial, overrides the sampled fields, and
calls :class:`TransformerTrainer` with the resulting configuration.

Search-space format
-------------------
Each leaf in ``search_space`` is a mapping with a ``type`` key and the
parameters expected by the matching Optuna suggestion method:

```yaml
search_space:
  training.learning_rate:
    type: loguniform
    low: 5.0e-6
    high: 1.0e-4
  training.batch_size:
    type: categorical
    choices: [16, 32, 64]
```

Supported types: ``categorical``, ``int``, ``uniform``, ``loguniform``.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import optuna
import pandas as pd

from src.configs import ExperimentConfig, load_config
from src.data import AGNewsLoader
from src.models.transformers import build_classification_model
from src.training.hf_trainer import TransformerTrainer
from src.utils.io_utils import ensure_dir, load_yaml, save_json
from src.utils.logging_utils import get_logger

_logger = get_logger(__name__)


def _suggest(trial: optuna.Trial, name: str, spec: Mapping[str, Any]) -> Any:
    """Convert a YAML spec into the corresponding Optuna suggestion call."""

    kind = spec["type"]
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "int":
        return trial.suggest_int(name, spec["low"], spec["high"], step=spec.get("step", 1))
    if kind == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if kind == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    raise ValueError(f"Unsupported search-space type '{kind}' for parameter '{name}'.")


def _override(config: ExperimentConfig, dotted_path: str, value: Any) -> ExperimentConfig:
    """Return a copy of ``config`` with ``dotted_path`` set to ``value``."""

    parts = dotted_path.split(".")
    if len(parts) != 2:
        raise ValueError(f"Search-space keys must use 'section.name' form, got '{dotted_path}'.")
    section, attr = parts
    sub_config = getattr(config, section)
    new_sub = replace(sub_config, **{attr: value})
    return replace(config, **{section: new_sub})


def run_optuna_search(sweep_yaml: Path | str, *, output_dir: Path | str) -> pd.DataFrame:
    """Execute a hyper-parameter search and return a DataFrame with trial results."""

    spec = load_yaml(sweep_yaml)
    base_config = load_config(spec["base_config"])
    search_space: dict[str, Mapping[str, Any]] = spec["search_space"]
    metric = spec.get("metric", "eval_f1_macro")
    direction = spec.get("direction", "maximize")
    n_trials = int(spec.get("n_trials", 20))
    timeout = spec.get("timeout_seconds")
    pruner_name = spec.get("pruner", "median")

    pruner = optuna.pruners.MedianPruner() if pruner_name == "median" else optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=spec.get("study_name", "agnews_sweep"),
        direction=direction,
        pruner=pruner,
    )

    sweep_root = ensure_dir(Path(output_dir))

    def objective(trial: optuna.Trial) -> float:
        suggested = {name: _suggest(trial, name, sp) for name, sp in search_space.items()}
        cfg = copy.deepcopy(base_config)
        for name, value in suggested.items():
            cfg = _override(cfg, name, value)
        bundle = build_classification_model(cfg.model, cfg.data)
        loader = AGNewsLoader(cfg.data)
        splits = loader.load()
        trainer = TransformerTrainer(
            tokenizer=bundle.tokenizer,
            model=bundle.model,
            data_cfg=cfg.data,
            training_cfg=cfg.training,
            label_names=splits.label_names,
            output_dir=sweep_root / f"trial_{trial.number:04d}",
            run_name=f"trial_{trial.number:04d}",
            report_to="none",
        )
        result = trainer.fit_and_evaluate(splits.train, splits.validation, splits.test)
        score = result.metrics.get(metric)
        if score is None:
            raise RuntimeError(f"Metric '{metric}' not produced by the trainer.")
        return float(score)

    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(sweep_root / "trials.csv", index=False)
    save_json(study.best_params, sweep_root / "best_params.json")
    save_json({"value": study.best_value, "trial": study.best_trial.number}, sweep_root / "best_value.json")
    _logger.info(
        "Sweep finished. Best %s=%.4f (trial %d) with params %s",
        metric,
        study.best_value,
        study.best_trial.number,
        study.best_params,
    )
    return trials_df
