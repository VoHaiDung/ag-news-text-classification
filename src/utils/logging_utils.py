"""Project-wide logging configuration.

A single configuration entry point keeps log formatting consistent across the
phase scripts and the library code, which is a precondition for diagnosing
issues in long-running training runs.
"""

from __future__ import annotations

import logging
import sys
from logging import Logger
from pathlib import Path

_DEFAULT_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Path | str | None = None,
    fmt: str = _DEFAULT_FORMAT,
    datefmt: str = _DEFAULT_DATEFMT,
) -> None:
    """Configure the root logger with stream and (optionally) file handlers.

    Parameters
    ----------
    level:
        Minimum severity to emit. Accepts either an integer (``logging.INFO``)
        or its string name (``"INFO"``).
    log_file:
        Optional path to a file that receives a copy of every log record.
    fmt, datefmt:
        Format strings forwarded to :class:`logging.Formatter`.
    """

    if isinstance(level, str):
        level = logging.getLevelName(level.upper())

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(level)
    # Remove any pre-existing handlers to avoid duplicate output when this
    # function is called multiple times within the same Python process (for
    # example during a Jupyter session).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)


def get_logger(name: str) -> Logger:
    """Return a logger configured to inherit the project-wide settings."""

    return logging.getLogger(name)
