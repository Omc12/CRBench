"""Plotting style for the ASPLOS submission figures.

Differences from the TMLR figure style, all of them forced by the ASPLOS
formatting rules:

* The paper is two-column acmart/sigplan.  ``TEXTWIDTH`` (7.00in) is the
  full ``\textwidth`` for ``figure*``; ``COLWIDTH`` (3.33in) is ``\columnwidth``.
  Both were measured from an actual acmart build, not guessed.
* Every figure is saved at *exactly* its final printed size --- no
  ``bbox_inches="tight"`` --- so that ``\includegraphics[width=\textwidth]``
  applies a scale factor of 1.0 and the point sizes below are the point
  sizes a reader sees.  The CFP asks for figure text that "appears to
  readers as a 9pt font or bigger", and forbids anything under 8pt, so
  8.0 is the floor here and nothing in a figure may go below it.

Data is read from the analysis outputs of the existing study; this module
never recomputes a measurement.
"""
from __future__ import annotations

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ASPLOS = os.path.dirname(HERE)
FIGDIR = os.path.join(ASPLOS, "figures")
# Single source of truth: the per-query records were reduced to these two
# JSON files by the existing pipeline.  We re-render, we never re-measure.
DATADIR = os.path.join(os.path.dirname(ASPLOS), "tmlr", "analysis")

# Okabe-Ito, assigned so that no two adjacent categories share a hue family.
METHOD_COLOR = {
    "KIVI (INT4/INT8)": "#0072B2",   # blue
    "SnapKV": "#D55E00",             # vermillion
    "DKV-Mid": "#CC79A7",            # reddish purple
    "DKV-High": "#8C4A73",           # deeper purple
    "Low-Rank KV": "#009E73",        # bluish green
    "StreamingLLM": "#E69F00",       # orange
    "KV-Merging": "#56B4E9",         # sky blue
    "Dense FP16": "#5A5A5A",         # neutral grey
}

MODEL_COLOR = {
    "Gemma 4 E2B": "#0072B2",
    "Gemma 4 E4B": "#009E73",
    "Qwen2.5 7B": "#D55E00",
}

MODEL_MARKER = {"Gemma 4 E2B": "o", "Gemma 4 E4B": "s", "Qwen2.5 7B": "^"}

SHORT = {
    "KIVI (INT4/INT8)": "KIVI",
    "SnapKV": "SnapKV",
    "DKV-Mid": "DKV-Mid",
    "DKV-High": "DKV-High",
    "Low-Rank KV": "Low-Rank",
    "StreamingLLM": "StreamLLM",
    "KV-Merging": "KV-Merge",
    "Dense FP16": "Dense",
}

INK = "#1A1A1A"
MUTED = "#6E6E6E"
GRID = "#D8D8D8"

# Measured from an acmart [sigplan,10pt] build: \textwidth = 505.89pt,
# \columnwidth = 240.945pt, at 72.27pt per inch.
TEXTWIDTH = 7.00
COLWIDTH = 3.33

# The floor the ASPLOS CFP imposes on text inside a figure.
MIN_PT = 8.0


def use_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 9.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 400,
        # NOTE: deliberately *not* bbox="tight" -- see the module docstring.
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def tidy(ax, grid_axis="y", left=True, bottom=True):
    """Strip the top/right spines and add a restrained grid behind the data."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="-", alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    return ax


def panel_label(ax, text, dx=-0.09, dy=1.10):
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=9.0,
            fontweight="bold", va="top", ha="left", color=INK)


def numbers():
    with open(os.path.join(DATADIR, "paper_numbers.json"), encoding="utf-8") as fh:
        return json.load(fh)


def stats():
    with open(os.path.join(DATADIR, "decoupling_stats.json"), encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, stem):
    """Write the figure at exactly its declared size, after checking it."""
    os.makedirs(FIGDIR, exist_ok=True)
    _assert_min_font(fig, stem)
    _assert_within_canvas(fig, stem)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{stem}.{ext}"))
    w, h = fig.get_size_inches()
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf + .png   ({w:.2f}in x {h:.2f}in)")


def _assert_within_canvas(fig, stem):
    """Fail if any artist extends past the figure canvas.

    We deliberately save without ``bbox_inches="tight"`` so that the figure
    prints at exactly its declared size and the point sizes above are the
    point sizes on paper.  The cost of that choice is that anything outside
    the canvas is silently *cropped* rather than accommodated -- which is
    exactly how a legend below an axes ends up half-missing in the PDF.  This
    catches it at generation time instead of in the proof.
    """
    fig.canvas.draw()
    tight = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    tol = 0.01                                   # inches
    over = {
        "left": -tight.x0, "bottom": -tight.y0,
        "right": tight.x1 - w, "top": tight.y1 - h,
    }
    bad = {k: round(v, 3) for k, v in over.items() if v > tol}
    if bad:
        raise SystemExit(
            f"{stem}: content is cropped -- it extends past the canvas by "
            f"{bad} inches. Increase the figure size or the margin on that "
            f"side; do not rely on a tight bounding box.")


def _assert_min_font(fig, stem):
    """Fail loudly if any text in the figure would print below the CFP floor.

    Because the figure is included at scale 1.0, the point size set here is
    the point size on paper, so this check is exact rather than advisory.
    """
    fig.canvas.draw()  # materialise tick labels before inspecting them
    bad = []
    for t in fig.findobj(mpl.text.Text):
        if not t.get_text():
            continue
        size = t.get_fontsize()
        if size < MIN_PT - 1e-6:
            bad.append((round(size, 2), t.get_text()[:30]))
    if bad:
        uniq = sorted(set(bad))[:8]
        raise SystemExit(
            f"{stem}: {len(bad)} text elements below the {MIN_PT}pt ASPLOS "
            f"floor, e.g. {uniq}")
