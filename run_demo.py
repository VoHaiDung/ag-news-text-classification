# ==============================================================================
# Project : AG News Text Classification
# Team    : Aimer PAM
# Author  : Vo Hai Dung
# License : MIT
# ==============================================================================
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#     "torch>=2.2",
#     "transformers>=4.41",
#     "tokenizers>=0.19",
#     "accelerate>=0.30",
#     "sentencepiece>=0.2",
#     "protobuf>=4.21",
#     "gradio>=4.31",
#     "shap>=0.45",
#     "numpy>=1.26",
#     "scipy>=1.11",
#     "scikit-learn>=1.4",
#     "pandas>=2.2",
#     "datasets>=2.18",
#     "onnxruntime>=1.18",
# ]
# ///
"""Zero-setup launcher for the Gradio demo.

The ``# /// script`` block above is PEP 723 inline metadata. ``uv`` reads it,
downloads a pinned CPython 3.12 when the host has none, materialises an
isolated environment holding the listed wheels and runs this file, so the
demo starts identically on a machine carrying Python 3.11, 3.13 or no Python
at all and without disturbing the interpreter already installed there::

    uv run run_demo.py

The interpreter is pinned instead of range-matched because the four
compile-only dependencies of the training pipeline (``fasttext``, ``hdbscan``,
``umap-learn``, ``bertopic``) are deliberately absent: the demo never imports
them, and leaving them out keeps the environment to wheels that resolve on
every host without a C++ toolchain.

Checkpoints are read from the local ``outputs/`` tree, which ``.gitignore``
excludes from version control, so they travel with a folder copy but never
with a ``git clone``. The launcher therefore verifies them up-front and
reports what is missing rather than failing later inside the model loader.

Usage::

    uv run run_demo.py                      # local only, auto-picked checkpoint
    uv run run_demo.py --list-models        # show which checkpoints are present
    uv run run_demo.py --lan                # reachable from the same Wi-Fi
    uv run run_demo.py --share              # public *.gradio.live tunnel
    uv run run_demo.py --model-dir outputs/transformers/ag_news_en__modernbert_base/best
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path

# Anchoring on the file rather than the process working directory is what lets
# the project be copied to another drive and launched from anywhere: every
# checkpoint path below, and MODEL_DIR inside the app, stays relative to it.
REPO_ROOT = Path(__file__).resolve().parent

# Mirrors ``_DEFAULT_MODEL_DIR`` in src/deployment/gradio_app.py; duplicated so
# the presence check can run before importing the app and its heavy stack.
DEFAULT_MODEL_DIR = Path("outputs/transformers/ag_news_en__modernbert_large/best")

# Directories holding the twelve supervised checkpoints exposed by the picker.
CHECKPOINT_ROOTS = (Path("outputs/transformers"), Path("outputs/multilingual"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_demo.py",
        description="Launch the AG News Gradio demo with no prior environment setup.",
    )
    parser.add_argument(
        "--model-dir",
        help="Checkpoint to load at startup; defaults to ModernBERT-large, "
        "falling back to any checkpoint that is present.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Expose a public *.gradio.live tunnel (link expires after 72 hours).",
    )
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Bind 0.0.0.0 so other devices on the same network can connect.",
    )
    parser.add_argument("--port", type=int, help="Port to serve on (default 7860).")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List the checkpoints found under outputs/ and exit.",
    )
    return parser.parse_args()


def _available_checkpoints() -> list[Path]:
    """Return every ``<root>/<model>/best`` directory that holds a usable model."""

    found: list[Path] = []
    for root in CHECKPOINT_ROOTS:
        if not (REPO_ROOT / root).is_dir():
            continue
        for candidate in sorted((REPO_ROOT / root).glob("*/best")):
            if (candidate / "config.json").is_file():
                found.append(candidate.relative_to(REPO_ROOT))
    return found


def _lan_ip() -> str:
    """Best-effort primary IPv4 of this host, used only to print a reachable URL."""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        try:
            # No traffic is sent; connect() on UDP only selects the outbound
            # interface, which is enough to learn the address peers would use.
            probe.connect(("8.8.8.8", 80))
            return str(probe.getsockname()[0])
        except OSError:
            return "127.0.0.1"


def _report_missing_checkpoints() -> None:
    print("No checkpoint found under outputs/.", file=sys.stderr)
    print(file=sys.stderr)
    print(f"Looked in: {REPO_ROOT / CHECKPOINT_ROOTS[0]}", file=sys.stderr)
    print(f"           {REPO_ROOT / CHECKPOINT_ROOTS[1]}", file=sys.stderr)
    print(file=sys.stderr)
    print(
        "outputs/ is listed in .gitignore, so the weights are absent from any\n"
        "git clone or downloaded ZIP. Copy the whole project folder from a\n"
        "machine that has them, or retrain with `make phase4`.",
        file=sys.stderr,
    )


def _resolve_model_dir(requested: str | None, found: list[Path]) -> Path | None:
    """Pick the checkpoint to preload, or ``None`` to let the app use its default."""

    if requested is not None:
        chosen = Path(requested)
        absolute = chosen if chosen.is_absolute() else REPO_ROOT / chosen
        if not (absolute / "config.json").is_file():
            print(f"Checkpoint '{chosen}' does not exist.", file=sys.stderr)
            print("Available checkpoints:", file=sys.stderr)
            for path in found:
                print(f"  {path.as_posix()}", file=sys.stderr)
            raise SystemExit(1)
        return absolute

    if (REPO_ROOT / DEFAULT_MODEL_DIR / "config.json").is_file():
        return None

    # A partial copy may carry some checkpoints but not the default one; start
    # on whatever is available instead of refusing to launch.
    fallback = found[0]
    print(f"Default checkpoint missing; starting on '{fallback.as_posix()}' instead.")
    return REPO_ROOT / fallback


def main() -> int:
    args = _parse_args()

    # The app resolves MODEL_DIR relative to the process working directory.
    os.chdir(REPO_ROOT)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    found = _available_checkpoints()

    if args.list_models:
        if not found:
            _report_missing_checkpoints()
            return 1
        print(f"{len(found)} checkpoint(s) available:")
        for path in found:
            print(f"  {path.as_posix()}")
        return 0

    if not found:
        _report_missing_checkpoints()
        return 1

    model_dir = _resolve_model_dir(args.model_dir, found)
    if model_dir is not None:
        os.environ["MODEL_DIR"] = str(model_dir)

    if args.share:
        os.environ["GRADIO_SHARE"] = "1"
        print("A public link will be printed below; anyone holding it can use the demo.")
    if args.lan:
        # Read by gradio.Blocks.launch() when no server_name argument is given.
        os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
        port = args.port or 7860
        print(f"Reachable on this network at http://{_lan_ip()}:{port}")
    if args.port is not None:
        os.environ["GRADIO_SERVER_PORT"] = str(args.port)

    try:
        from src.deployment.gradio_app import main as launch
    except ModuleNotFoundError as exc:
        print(f"Missing dependency '{exc.name}'.", file=sys.stderr)
        print("Run this file through uv so the environment is built for you:", file=sys.stderr)
        print("  uv run run_demo.py", file=sys.stderr)
        return 1

    launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
