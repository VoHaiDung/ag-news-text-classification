"""Phase 1 entry point: project kickoff and environment validation.

Mapped Work Breakdown Structure tasks:

* 1.1.1.1 Initialise Git repo and branches (main / dev)
* 1.1.1.2 Set up folder structure and ``.gitignore``
* 1.1.2.1 Create virtual environment and install ``requirements.txt``
* 1.1.2.2 Configure Weights and Biases / MLflow workspace
* 1.3.1   Define scope, objectives and success metrics
* 1.3.2   Risk assessment and mitigation strategy

The script performs a *non-destructive* environment check: it reports the
versions of every critical dependency, verifies that GPU and CUDA are
detected when expected, instantiates the configured experiment tracker, and
writes a JSON diagnostic report under ``outputs/diagnostics/``. The output
file is the deliverable that the WBS expects from the kickoff phase.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.configs import ExperimentConfig, load_config
from src.utils import (
    build_tracker,
    configure_logging,
    ensure_dir,
    get_logger,
    save_json,
    set_global_seed,
)
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT

_CRITICAL_PACKAGES = (
    "numpy",
    "pandas",
    "sklearn",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "evaluate",
    "setfit",
    "sentence_transformers",
    "fasttext",
    "cleanlab",
    "bertopic",
    "shap",
    "lime",
    "onnx",
    "onnxruntime",
    "optuna",
    "wandb",
    "mlflow",
    "gradio",
)

_logger = get_logger(__name__)


def _package_version(name: str) -> str | None:
    """Return the installed version of ``name`` or ``None`` if not importable."""

    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return getattr(module, "__version__", "unknown")


def _gpu_report() -> dict[str, Any]:
    """Collect a small CUDA/MPS report from PyTorch."""

    report: dict[str, Any] = {"available": False}
    try:
        import torch

        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["available"] = True
            report["device_count"] = torch.cuda.device_count()
            report["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                }
                for i in range(torch.cuda.device_count())
            ]
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            report["available"] = True
            report["devices"] = [{"name": "Apple MPS"}]
    except ImportError:
        report["torch_version"] = None
    return report


def _build_diagnostics(config: ExperimentConfig) -> dict[str, Any]:
    """Assemble the diagnostic payload that will be written to disk."""

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
        },
        "packages": {name: _package_version(name) for name in _CRITICAL_PACKAGES},
        "gpu": _gpu_report(),
        "config": {
            "name": config.name,
            "model": config.model.name,
            "data": config.data.name,
            "tracking_backend": config.tracking.backend,
            "seed": config.training.seed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 - project kickoff diagnostics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "base.yaml",
        help="YAML configuration file (defaults to configs/base.yaml).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "diagnostics" / "phase1_kickoff.json",
        help="Where to write the diagnostic report.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    seeds = set_global_seed(config.training.seed)
    _logger.info("Applied global seed: %s", seeds)

    diagnostics = _build_diagnostics(config)

    missing = [name for name, ver in diagnostics["packages"].items() if ver is None]
    if missing:
        _logger.warning("Missing packages: %s", ", ".join(missing))
    else:
        _logger.info("All %d critical packages are importable.", len(_CRITICAL_PACKAGES))

    tracker = build_tracker(
        backend=config.tracking.backend,
        project=config.tracking.project,
        entity=config.tracking.entity,
        run_name=config.tracking.run_name or "phase1_kickoff",
        config={"phase": "phase1_kickoff", "config_path": str(args.config)},
    )
    try:
        tracker.log_metrics(
            {"missing_packages": float(len(missing))},
            step=0,
        )
    finally:
        tracker.finish()

    output_path = ensure_dir(args.output.parent) / args.output.name
    save_json(diagnostics, output_path)
    _logger.info("Diagnostic report written to %s", output_path)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
