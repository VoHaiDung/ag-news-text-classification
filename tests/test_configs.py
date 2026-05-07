"""Tests for the configuration schema."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.configs import ExperimentConfig, load_config

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_base_config_loads() -> None:
    config = load_config(CONFIGS_DIR / "base.yaml")
    assert isinstance(config, ExperimentConfig)
    assert config.data.num_labels == 4
    assert config.data.label_names == ("World", "Sports", "Business", "Sci/Tech")


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_a_real_section: 1\n", encoding="utf-8")
    with pytest.raises(TypeError):
        load_config(bad)


def test_per_model_config_overrides() -> None:
    deberta = load_config(CONFIGS_DIR / "models" / "deberta_v3_small.yaml")
    modernbert = load_config(CONFIGS_DIR / "models" / "modernbert_base.yaml")
    assert deberta.model.family == "deberta-v3"
    assert modernbert.model.family == "modernbert"
    # ModernBERT defaults to bf16 in the project; DeBERTa-v3-small to fp16.
    assert modernbert.training.bf16 is True
    assert deberta.training.fp16 is True
