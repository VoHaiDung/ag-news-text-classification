"""Canonical path locations.

Every phase script resolves output directories through this module so that the
on-disk layout matches the documentation in :file:`README.md`. Centralising
the paths also makes it trivial to redirect output during testing.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

# Top-level directories.
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
DATA_DIR: Path = PROJECT_ROOT / "data"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"

# Conventional sub-directories for data lifecycle stages.
DATA_RAW: Path = DATA_DIR / "raw"
DATA_INTERIM: Path = DATA_DIR / "interim"
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_EXTERNAL: Path = DATA_DIR / "external"

# Output sub-directories used across phases.
ARTIFACTS_DIR: Path = OUTPUTS_DIR / "artifacts"
CHECKPOINTS_DIR: Path = OUTPUTS_DIR / "checkpoints"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
LOGS_DIR: Path = OUTPUTS_DIR / "logs"
METRICS_DIR: Path = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"


__all__ = [
    "PROJECT_ROOT",
    "CONFIGS_DIR",
    "DATA_DIR",
    "NOTEBOOKS_DIR",
    "OUTPUTS_DIR",
    "REPORTS_DIR",
    "SCRIPTS_DIR",
    "DATA_RAW",
    "DATA_INTERIM",
    "DATA_PROCESSED",
    "DATA_EXTERNAL",
    "ARTIFACTS_DIR",
    "CHECKPOINTS_DIR",
    "FIGURES_DIR",
    "LOGS_DIR",
    "METRICS_DIR",
    "PREDICTIONS_DIR",
]
