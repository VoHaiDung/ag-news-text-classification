"""Command-line entry point.

Exposes the phase scripts via a single ``agnews`` console script declared in
``pyproject.toml``. The implementation defers to each phase's ``main``
function so the same logic is reachable via either ``python -m
scripts.phase2_eda`` or ``agnews phase2-eda``.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import click


def _phase_runner(name: str) -> Callable[..., int]:
    """Lazily import the requested phase's :func:`main` to avoid heavy imports up-front."""

    def runner(args: tuple[str, ...]) -> int:
        # ``argparse`` reads from ``sys.argv``; replace it for the duration of the call.
        argv_backup = sys.argv
        sys.argv = [name, *args]
        try:
            module = __import__(f"scripts.{name}", fromlist=["main"])
            return int(module.main())
        finally:
            sys.argv = argv_backup

    return runner


_PHASES = (
    "phase1_kickoff",
    "phase2_eda",
    "phase3_baselines",
    "phase4_transformers",
    "phase5_multilingual",
    "phase6_setfit",
    "phase7_evaluation",
    "phase8_deployment",
    "phase9_report",
)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def cli() -> None:
    """AG News capstone command-line interface."""


for phase in _PHASES:

    @cli.command(name=phase.replace("_", "-"))
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def _command(ctx: click.Context, args: tuple[str, ...], _phase: str = phase) -> None:
        """Run the corresponding phase script with the given arguments."""

        ctx.exit(_phase_runner(_phase)(args))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
