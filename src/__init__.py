"""Top-level package for the AG News text classification capstone project.

The package follows a hybrid layout: domain modules live under ``src.<domain>``
(``src.data``, ``src.models``, ``src.training``, ``src.evaluation``,
``src.explainability``, ``src.deployment``), while phase-level entry points are
collected under ``scripts/`` at the repository root. This keeps the library
reusable across phases while preserving a one-to-one mapping from each phase of
the Work Breakdown Structure to a runnable script.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ag-news-text-classification")
except PackageNotFoundError:  # package is not installed (editable mode without metadata)
    __version__ = "0.1.0"

__all__ = ["__version__"]
