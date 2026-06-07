"""Render slide-ready PNG figures from evaluation CSVs.

Produces two presentation figures used in the capstone defence deck
(reports/presentation/slide_deck_full.md):

  1. Confusion matrix heatmap  (Slide 9)  <- outputs/evaluation/confusion_matrix.csv
  2. Latency-accuracy Pareto front (Slide 10) <- outputs/evaluation/latency_pareto.csv

SHAP and LIME outputs are interactive HTML (outputs/evaluation/{shap,lime}/*.html)
and must be screenshotted by hand from a browser; they are not regenerated here.

Run:  python -m scripts.make_slide_figures
Output: outputs/slide_figures/{confusion_matrix.png, latency_pareto.png}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EVAL = ROOT / "outputs" / "evaluation"
OUT = ROOT / "outputs" / "slide_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Samsung Innovation Campus blue, to match the slide template.
SIC_BLUE = "#1428A0"


def render_confusion_matrix() -> Path:
    """Row-normalised 4x4 confusion matrix heatmap with raw counts."""
    df = pd.read_csv(EVAL / "confusion_matrix.csv", index_col=0)
    # Reorder rows to match column order so the diagonal is visually aligned.
    df = df.reindex(index=df.columns)
    counts = df.to_numpy(dtype=float)
    row_pct = counts / counts.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.4), dpi=200)
    im = ax.imshow(row_pct, cmap="Blues", vmin=0.0, vmax=1.0)

    labels = list(df.columns)
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("Confusion matrix — ModernBERT-large (seed 42)\n"
                 "test set, n = 7,600", fontsize=12, pad=12)

    for i in range(len(labels)):
        for j in range(len(labels)):
            txt = f"{int(counts[i, j]):,}\n{row_pct[i, j]*100:.1f}%"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if row_pct[i, j] > 0.5 else "black",
                    fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised rate", fontsize=10)
    fig.tight_layout()

    path = OUT / "confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_pareto() -> Path:
    """Latency (x) vs F1-macro (y) scatter with the Pareto frontier traced."""
    df = pd.read_csv(EVAL / "latency_pareto.csv", comment="#")

    colour = {"en": SIC_BLUE, "vi": "#E8730C", "fr": "#1B9E4B"}

    fig, ax = plt.subplots(figsize=(7.6, 5.2), dpi=200)
    for lang, sub in df.groupby("language"):
        ax.scatter(sub["mean_ms"], sub["test_f1_macro"], s=70,
                   c=colour.get(lang, "grey"), label=lang.upper(),
                   edgecolors="white", linewidths=0.8, zorder=3)

    # Trace the Pareto frontier: lower latency AND higher F1 is better.
    pts = df.sort_values("mean_ms")
    frontier_x, frontier_y, best_f1 = [], [], -1.0
    for _, r in pts.iterrows():
        if r["test_f1_macro"] > best_f1:
            frontier_x.append(r["mean_ms"])
            frontier_y.append(r["test_f1_macro"])
            best_f1 = r["test_f1_macro"]
    ax.plot(frontier_x, frontier_y, "--", color="grey", lw=1.3,
            zorder=2, label="Pareto frontier")

    # Annotate the deployment pick and the best-accuracy single checkpoint.
    for _, r in df.iterrows():
        if r["architecture"] in ("ModernBERT-base", "ModernBERT-large"):
            ax.annotate(r["architecture"], (r["mean_ms"], r["test_f1_macro"]),
                        textcoords="offset points", xytext=(8, -4),
                        fontsize=8.5, fontweight="bold")

    ax.axvline(50, color="red", ls=":", lw=1.0)
    ax.text(54, ax.get_ylim()[0] + 0.002, "O5 target 50 ms",
            color="red", fontsize=8, rotation=90, va="bottom")

    ax.set_xlabel("Single-stream CPU latency, batch 1 (ms, lower is better)",
                  fontsize=11)
    ax.set_ylabel("Test F1-macro (higher is better)", fontsize=11)
    ax.set_title("Twelve-model INT8 ONNX latency-accuracy Pareto front",
                 fontsize=12, pad=10)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()

    path = OUT / "latency_pareto.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    cm = render_confusion_matrix()
    pf = render_pareto()
    print(f"wrote {cm.relative_to(ROOT)}")
    print(f"wrote {pf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
