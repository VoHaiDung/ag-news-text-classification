"""File system helpers used by every phase of the pipeline.

The functions in this module are deliberately thin wrappers around the
standard library and PyYAML; they exist so that the rest of the codebase has
a single, well-tested place to perform path creation and (de)serialisation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path | str) -> Path:
    """Create ``path`` (and any missing parents) and return it as a :class:`Path`.

    The function is idempotent: it does not raise when the directory already
    exists.
    """

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: Path | str) -> dict[str, Any]:
    """Load a YAML file into a Python dictionary.

    A YAML document that resolves to ``None`` (an empty file) is returned as
    an empty dictionary so downstream callers can treat the result uniformly.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_yaml(payload: dict[str, Any], path: Path | str) -> Path:
    """Serialise ``payload`` to ``path`` in YAML form and return the path."""

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)
    return target


def save_json(payload: Any, path: Path | str, *, indent: int = 2) -> Path:
    """Serialise ``payload`` to ``path`` as UTF-8 JSON and return the path."""

    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent, default=str)
    return target


def load_json(path: Path | str) -> Any:
    """Load a JSON file from ``path``."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
