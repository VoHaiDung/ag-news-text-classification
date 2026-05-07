"""Reproducibility helpers.

Deterministic behaviour is essential for a benchmark study where the headline
result is a single F1 score: the same configuration must yield the same number
on a different machine. This module centralises seed management for the
standard library, NumPy and PyTorch, and exposes a single entry point used by
every training and evaluation script.

References
----------
Bouthillier, X., Laurent, C., and Vincent, P. (2019).
*Unreproducible research is reproducible.* ICML.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedingReport:
    """Summary of the seed values that were applied.

    The structure is returned by :func:`set_global_seed` so the caller can log
    the exact values to the experiment tracker without re-reading them from
    environment variables.
    """

    python_seed: int
    numpy_seed: int
    torch_seed: int
    deterministic_torch: bool


def set_global_seed(seed: int = 42, *, deterministic_torch: bool = True) -> SeedingReport:
    """Seed every relevant random number generator.

    Parameters
    ----------
    seed:
        The integer seed used for ``random``, ``numpy.random`` and ``torch``.
    deterministic_torch:
        When ``True`` enable cuDNN deterministic mode. This may slow down
        training on GPU but is required for reproducible benchmark numbers.

    Returns
    -------
    SeedingReport
        Record of the seeds that were set; useful for logging.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        # Torch is an optional dependency for the classical-baselines code path.
        pass

    return SeedingReport(
        python_seed=seed,
        numpy_seed=seed,
        torch_seed=seed,
        deterministic_torch=deterministic_torch,
    )
