"""
Plotting utilities for the trajectory benchmark.
"""

from __future__ import annotations

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Fixed per-method colours, kept consistent across all plots
# ---------------------------------------------------------------------------

_TAB10 = plt.cm.tab10.colors

METHOD_COLORS: dict[str, str] = {
    "TBR":              "#E41A1C",
    "PAGA":             _TAB10[0],
    "DPT":              _TAB10[1],
    "Palantir":         _TAB10[2],
    "ANOVA":            _TAB10[4],
    "Wilcoxon":         _TAB10[5],
    "HiDDEN":           _TAB10[6],
    "MELD":             _TAB10[7],
    "Milo":             _TAB10[8],
    "PseudobulkDESeq2": _TAB10[9],
}


# ---------------------------------------------------------------------------
# Core plot function
# ---------------------------------------------------------------------------

def plot_enrichment_curves(
    enrichment_data: list[tuple[np.ndarray, float]],
    method_names: list[str],
    title: str = "Target enrichment",
    figsize: tuple[int, int] = (6, 4),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot GSEA-style NES curves for multiple methods on a single axis.

    Parameters
    ----------
    enrichment_data : list of (NES_curve, nes) tuples
        One entry per method, in the same order as *method_names*.
    method_names : list[str]
        Method labels corresponding to *enrichment_data*.
    title : str
        Axis title.
    figsize : (width, height)
        Figure size in inches.
    save_path : str, optional
        If given, save the figure as both PNG and SVG.

    Returns
    -------
    matplotlib Figure
    """
    # Sort by NES ascending so TBR (highest) plots last / on top
    order = sorted(range(len(method_names)), key=lambda i: enrichment_data[i][1])

    fig, ax = plt.subplots(figsize=figsize)

    for i in order:
        name = method_names[i]
        curve, nes = enrichment_data[i]
        is_tbr = name == "TBR"
        color = METHOD_COLORS.get(name, "#888888")
        ax.plot(
            np.arange(len(curve)),
            curve,
            color=color,
            linewidth=1.5 if is_tbr else 0.8,
            zorder=3 if is_tbr else 2,
            label=f"{name} (NES={nes:.2f})",
        )

    ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Gene rank")
    ax.set_ylabel("NES")
    ax.legend(loc="upper right", fontsize=5, framealpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        for ext in ("png", "svg"):
            fig.savefig(f"{save_path}.{ext}", dpi=300, bbox_inches="tight")
        print(f"Saved {save_path}.{{png,svg}}")

    return fig


# ---------------------------------------------------------------------------
# Convenience: load pkl and plot
# ---------------------------------------------------------------------------

def plot_from_pkl(
    pkl_path: str,
    title: str | None = None,
    save_path: str | None = None,
    figsize: tuple[int, int] = (6, 4),
) -> plt.Figure:
    """
    Load a ``*_manuscript_data.pkl`` file and plot the enrichment curves.

    Parameters
    ----------
    pkl_path : str
        Path to the pkl file produced by ``evaluate.py``.
    title : str, optional
        Plot title; defaults to the pkl filename stem.
    save_path : str, optional
        Base path (without extension) for PNG/SVG output.
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    ec = data["enrichment_comparison"]
    title = title or os.path.splitext(os.path.basename(pkl_path))[0]

    return plot_enrichment_curves(
        enrichment_data=ec["enrichment_data"],
        method_names=ec["method_names"],
        title=title,
        figsize=figsize,
        save_path=save_path,
    )


# ---------------------------------------------------------------------------
# CLI usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Plot enrichment curves from a manuscript pkl")
    p.add_argument("pkl", help="Path to *_manuscript_data.pkl")
    p.add_argument("--title", default=None)
    p.add_argument("--save", default=None, help="Base path for PNG/SVG output (no extension)")
    args = p.parse_args()

    fig = plot_from_pkl(args.pkl, title=args.title, save_path=args.save)
    plt.show()
