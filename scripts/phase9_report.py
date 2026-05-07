"""Phase 9 entry point: assemble the final report and presentation assets.

Mapped Work Breakdown Structure tasks:

* 9.1.1 Write the Introduction and Background
* 9.1.2 Write the Methodology and Implementation
* 9.1.3 Write the Results, Discussion and Conclusion
* 9.2.1 Design the slide structure and key takeaways
* 9.2.2 Add visualisations and polish the design
* 9.3   Record the demo video (3-5 minutes)
* 9.4   Rehearse the presentation and reserve buffer time

The Markdown sources for the report and the presentation live under
``reports/final_report`` and ``reports/presentation``. This script:

1. validates that every required section file exists;
2. concatenates the three report sections into a single Markdown document;
3. fills the headline-results table by reading the JSON metric files
   produced by Phase 7;
4. emits a checklist of remaining manual steps (video recording, slide
   polishing, rehearsal).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

from src.utils import configure_logging, ensure_dir, get_logger
from src.utils.paths import OUTPUTS_DIR, PROJECT_ROOT, REPORTS_DIR

_logger = get_logger(__name__)

_REPORT_SECTIONS = (
    REPORTS_DIR / "final_report" / "01_introduction.md",
    REPORTS_DIR / "final_report" / "02_methodology.md",
    REPORTS_DIR / "final_report" / "03_results_discussion_conclusion.md",
)
_PRESENTATION_FILES = (
    REPORTS_DIR / "presentation" / "slide_outline.md",
    REPORTS_DIR / "presentation" / "demo_video_script.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 9 - report and presentation assembly.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "report",
    )
    parser.add_argument(
        "--metrics-files",
        nargs="+",
        type=Path,
        default=list((OUTPUTS_DIR / "evaluation").glob("*.json"))
        if (OUTPUTS_DIR / "evaluation").exists()
        else [],
        help="Optional JSON metric files (Phase 7 outputs) to summarise.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _validate_files(paths: Iterable[Path]) -> list[Path]:
    """Return the list of missing files (empty list = everything present)."""

    return [p for p in paths if not p.exists()]


def _concatenate_sections(paths: Iterable[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        parts.append(f"<!-- begin: {path.relative_to(PROJECT_ROOT)} -->")
        parts.append(path.read_text(encoding="utf-8").strip())
        parts.append(f"<!-- end: {path.relative_to(PROJECT_ROOT)} -->")
        parts.append("")
    return "\n\n".join(parts)


def _summarise_metrics(metrics_files: Iterable[Path]) -> str:
    rows: list[str] = ["| Source | Metric | Value |", "|--------|--------|-------|"]
    for path in metrics_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key, value in payload.items():
            if isinstance(value, (int, float)):
                rows.append(f"| {path.stem} | {key} | {value:.4f} |")
    if len(rows) == 2:
        rows.append("| (no Phase 7 metrics found) |  |  |")
    return "\n".join(rows)


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)
    output_dir = ensure_dir(args.output_dir)

    missing = _validate_files(_REPORT_SECTIONS) + _validate_files(_PRESENTATION_FILES)
    if missing:
        for path in missing:
            _logger.error("Missing report asset: %s", path)
        return 1

    full_report = _concatenate_sections(_REPORT_SECTIONS)
    summary_table = _summarise_metrics(args.metrics_files)
    appendix = (
        "\n\n## Appendix: Phase 7 metric snapshot\n\n"
        + summary_table
        + "\n\nGenerated automatically by ``scripts/phase9_report.py``.\n"
    )
    final_path = output_dir / "final_report.md"
    final_path.write_text(full_report + appendix, encoding="utf-8")

    presentation_index = output_dir / "presentation_index.md"
    presentation_index.write_text(
        "# Presentation assets\n\n"
        + "\n".join(
            f"- [{p.relative_to(PROJECT_ROOT)}]({p.resolve().as_posix()})"
            for p in _PRESENTATION_FILES
        )
        + "\n",
        encoding="utf-8",
    )

    checklist = output_dir / "manual_checklist.md"
    checklist.write_text(
        "# Manual steps remaining\n\n"
        "- [ ] Convert the Markdown report to the SIC ``.docx`` template.\n"
        "- [ ] Build the ``.pptx`` slide deck from ``slide_outline.md``.\n"
        "- [ ] Record the 3-5 minute demo video using ``demo_video_script.md``.\n"
        "- [ ] Rehearse with a 12-minute timer; reserve buffer time per WBS task 9.4.\n",
        encoding="utf-8",
    )

    _logger.info("Phase 9 complete. Report: %s", final_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
