"""Shared plotting style for the CRBench manuscript figures.

One place for the palette, typography and axis furniture so every figure in the
paper reads as part of the same system.  Colours are the Okabe-Ito
colour-blind-safe set; the serif stack matches the manuscript body text.
"""
from __future__ import annotations

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(os.path.dirname(HERE), "figures")

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

# text width of the TMLR template, in inches
TEXTWIDTH = 6.5


def use_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.0,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.2,
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
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
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


def panel_label(ax, text, dx=-0.09, dy=1.06):
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left", color=INK)


def numbers():
    with open(os.path.join(HERE, "paper_numbers.json"), encoding="utf-8") as fh:
        return json.load(fh)


def stats():
    with open(os.path.join(HERE, "decoupling_stats.json"), encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, stem):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, f"{stem}.{ext}"))
    plt.close(fig)
    print(f"  wrote figures/{stem}.pdf + .png")
