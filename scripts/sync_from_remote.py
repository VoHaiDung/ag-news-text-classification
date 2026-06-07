# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
"""Synchronise the lightweight result artefacts from a rented cloud GPU host.

The full pipeline produces ~20 GB of model checkpoints; only ~250 MB of that
is needed locally to compile the final report (metrics, figures, predictions,
SHAP/LIME HTML pages, and the deployment-ready ONNX INT8 models). This script
streams a filtered ``tar`` archive over SSH so the download stays small and
does not require ``rsync`` to be present on the local machine.

Usage::

    python -m scripts.sync_from_remote \\
        --host <remote-ip> \\
        --port <ssh-port> \\
        --remote-root /workspace/ag-news-text-classification
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# Top-level paths under the remote project root that the tar stream will
# include. Heavy artefacts are filtered out by the ``--exclude`` flags
# below, so directories such as ``outputs/transformers`` are listed in
# full and the model checkpoints inside them are removed by exclude
# patterns.
INCLUDE_PATHS: list[str] = [
    "outputs/diagnostics",
    "outputs/eda",
    "outputs/baselines",
    "outputs/transformers",
    "outputs/multilingual",
    "outputs/setfit",
    "outputs/evaluation",
    "outputs/deployment_all_int8",
]

# Patterns excluded from the tar stream when the default light sync is used.
# Heavy directories such as ``best/``, ``checkpoints/`` and ``bertopic_model``
# are dropped so that only metrics, predictions, figures and the small INT8
# model travel. Add ``--include-weights`` on the command line to also pull the
# ``best/`` model weights for every trained model (about 7 GB).
EXCLUDE_PATTERNS_LIGHT: list[str] = [
    "outputs/*/*/best",
    "outputs/*/*/checkpoints",
    "outputs/setfit/setfit_*",
    "outputs/eda/bertopic/bertopic_model",
    "outputs/baselines/*/model.bin",
    "outputs/baselines/*/test_probabilities.npy",
    "outputs/eda/cleanlab/out_of_fold_probabilities.npy",
    "outputs/transformers/*/test_probabilities.npy",
    "outputs/multilingual/*/test_probabilities.npy",
    "*.safetensors",
    "*.bin",
]

# Patterns kept when ``--include-weights`` is set. The intermediate
# ``checkpoints/`` directories are still dropped because only the final
# ``best/`` snapshot is needed for inference, and the BERTopic model is
# regeneratable from the EDA notebooks so it stays excluded.
EXCLUDE_PATTERNS_FULL: list[str] = [
    "outputs/*/*/checkpoints",
    "outputs/eda/bertopic/bertopic_model",
    "outputs/baselines/*/test_probabilities.npy",
    "outputs/eda/cleanlab/out_of_fold_probabilities.npy",
    "outputs/transformers/*/test_probabilities.npy",
    "outputs/multilingual/*/test_probabilities.npy",
]


def _build_remote_tar_command(remote_root: str, include_weights: bool) -> str:
    """Return the remote-side shell command that produces the filtered tar stream."""

    patterns = EXCLUDE_PATTERNS_FULL if include_weights else EXCLUDE_PATTERNS_LIGHT
    exclude_args = " ".join(f"--exclude={pattern!r}" for pattern in patterns)
    include_args = " ".join(INCLUDE_PATHS)
    # ``tar`` is told to change to the remote project root, then archive the
    # listed paths to stdout with the excludes applied. Errors on missing
    # paths are suppressed with ``--ignore-failed-read`` so the script
    # continues if a phase is not yet present on disk.
    return (
        f"cd {remote_root!r} && tar --ignore-failed-read {exclude_args} "
        f"-cf - {include_args}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sync remote cloud GPU artefacts to local."
    )
    parser.add_argument("--host", required=True, help="Remote host IP.")
    parser.add_argument("--port", type=int, default=22, help="Remote SSH port.")
    parser.add_argument(
        "--remote-root",
        default="/workspace/ag-news-text-classification",
        help="Remote project root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the SSH and tar commands without executing them.",
    )
    parser.add_argument(
        "--include-weights",
        action="store_true",
        help="Also pull best/ checkpoints for every trained model (about 7 GB).",
    )
    args = parser.parse_args()

    remote_cmd = _build_remote_tar_command(args.remote_root, args.include_weights)
    ssh_cmd = [
        "ssh",
        "-p",
        str(args.port),
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        f"root@{args.host}",
        remote_cmd,
    ]
    tar_cmd = ["tar", "-xf", "-", "-C", str(ROOT)]

    if args.dry_run:
        print("Would run (piped):")
        print("  ", " ".join(ssh_cmd))
        print("  | ", " ".join(tar_cmd))
        return 0

    print(f"Streaming filtered tar from {args.host}:{args.port} ...")
    ssh_proc = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE)
    tar_proc = subprocess.Popen(tar_cmd, stdin=ssh_proc.stdout)
    if ssh_proc.stdout is not None:
        ssh_proc.stdout.close()  # allow ssh to receive SIGPIPE if tar exits
    tar_rc = tar_proc.wait()
    ssh_rc = ssh_proc.wait()

    if tar_rc != 0 or ssh_rc != 0:
        sys.exit(f"Sync failed: tar exit={tar_rc}, ssh exit={ssh_rc}")
    print(f"Done. Artefacts under {ROOT / 'outputs'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
