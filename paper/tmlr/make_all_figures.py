#!/usr/bin/env python3
"""
CRBench publication figure generator — unified blue-to-mint palette.
- All bar charts: solid color fills (no hatching/patterns), small numbers (5.0pt), clean centered SnapKV callout.
- Heatmap: blue-to-mint colormap, no background grid lines cutting through numbers, crisp borders.
- Fig 6 & Fig 7: solid lines and markers, blue-to-mint colors, no distracting decorations.
- Fig 8: simplified, highly intuitive 2-panel sensitivity & formula comparison.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

# ─── Output directory ─────────────────────────────────────────────────────────
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(out_dir, exist_ok=True)

# ─── Color Palette: Categorical & High-Contrast ───────────────────────────────
# 8 Methods — Distinct, vibrant, high-contrast categorical palette (NOT just shades of blue/green!):
METHOD_COLORS = [
    "#1E5AA8",   # 0 KIVI          -> Royal Blue
    "#D32F2F",   # 1 SnapKV        -> Crimson / Coral Red
    "#7B1FA2",   # 2 DKV-Mid       -> Amethyst Purple
    "#C2185B",   # 3 DKV-High      -> Berry / Magenta
    "#212121",   # 4 Dense FP16    -> Dark Slate / Charcoal
    "#E65100",   # 5 Low-Rank KV   -> Warm Amber / Orange
    "#2E7D32",   # 6 StreamLLM     -> Emerald Green
    "#00838F",   # 7 KV-Merge      -> Deep Teal / Cyan
]
METHOD_MARKERS = ['o', 's', 'D', '^', '*', 'p', 'h', 'v']

# 8 methods in gradient order from Deep Blue to Bright Mint (for reference):
BM8 = [
    "#0E4D92",   # 0 Deep Cobalt Blue    -> KIVI
    "#1976D2",   # 1 Royal Blue          -> SnapKV
    "#0288D1",   # 2 Cerulean Blue       -> DKV-Mid
    "#0097A7",   # 3 Cyan Teal           -> DKV-High
    "#00796B",   # 4 Deep Sea Teal       -> Dense FP16
    "#00897B",   # 5 Emerald Teal        -> Low-Rank KV
    "#26A69A",   # 6 Mint Teal           -> StreamLLM
    "#4DB6AC",   # 7 Soft Mint           -> KV-Merge
]

# 3 Foundation Model Palette:
# E2B: Deep Blue | E4B: Teal | Qwen 7B: Mint Green
BM3_FILL = ["#1976D2", "#0097A7", "#38B289"]
BM3_EDGE = ["#0D47A1", "#006064", "#1B5E44"]
BM3_LABEL = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen2.5 7B"]

# General Neutrals
WHITE  = "#FFFFFF"
BLACK  = "#111111"
GRAY_D = "#4F5966"
GRAY_L = "#E2E7ED"
RED_A  = "#C0392B"
PANEL  = "#F5F8FA"

# ─── Global rcParams ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
    "figure.facecolor":    WHITE,
    "axes.facecolor":      WHITE,
    "savefig.facecolor":   WHITE,
    "font.family":         "serif",
    "font.serif":          ["DejaVu Serif", "Times New Roman", "Computer Modern Roman"],
    "font.size":           9.0,
    "mathtext.fontset":    "cm",
    "text.color":          BLACK,
    "axes.titlesize":      9.5,
    "axes.titleweight":    "bold",
    "axes.labelsize":      9.0,
    "axes.labelweight":    "bold",
    "axes.labelcolor":     BLACK,
    "axes.edgecolor":      BLACK,
    "axes.linewidth":      0.85,
    "axes.grid":           True,
    "axes.grid.axis":      "y",
    "axes.axisbelow":      True,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "grid.color":          "#DCE1E7",
    "grid.linewidth":      0.55,
    "grid.linestyle":      "--",
    "xtick.color":         BLACK,
    "ytick.color":         BLACK,
    "xtick.labelsize":     8.2,
    "ytick.labelsize":     8.2,
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    3.0,
    "ytick.major.size":    3.0,
    "xtick.major.width":   0.8,
    "ytick.major.width":   0.8,
    "legend.frameon":      True,
    "legend.edgecolor":    "#BAC2CC",
    "legend.framealpha":   1.0,
    "legend.fancybox":     False,
    "legend.fontsize":     8.0,
    "legend.borderpad":    0.5,
    "legend.labelspacing": 0.3,
    "legend.handlelength": 2.0,
    "legend.handleheight": 0.9,
    "lines.linewidth":     1.8,
    "lines.markersize":    5.5,
    "figure.dpi":          150,
    "savefig.dpi":         300,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.05,
})

methods = ['kivi','snapkv','dkv_mid','dkv_high','dense_fp16',
           'low_rank_kv','streaming_llm','kv_merging']
LABELS  = ['KIVI', 'SnapKV', 'DKV-Mid', 'DKV-High',
           'Dense FP16', 'Low-Rank KV', 'StreamLLM', 'KV-Merge']


# ─── Shared Utilities ─────────────────────────────────────────────────────────
def _box_axes(ax):
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax

def _legend_above(ax, ncol=3, extra=0.22):
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="lower left",
              bbox_to_anchor=(0.0, 1.02, 1.0, extra),
              mode="expand", ncol=ncol,
              frameon=True, edgecolor="#BAC2CC", framealpha=1.0,
              fontsize=7.2, borderpad=0.35,
              handlelength=1.4, handleheight=0.8, borderaxespad=0.)

def _bar_label(ax, rects, fontsize=5.2):
    for r in rects:
        h = r.get_height()
        ax.annotate(f"{h:.1f}",
                    xy=(r.get_x() + r.get_width() / 2, h),
                    xytext=(0, 2.2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=fontsize, color=BLACK,
                    bbox=dict(boxstyle="round,pad=0.10", fc=WHITE, ec="none", alpha=0.88),
                    zorder=6)

def _draw_solid_bars(ax, x, vals_list, W=0.26):
    """Draw solid-fill bars (no hatching) in blue-to-mint scheme."""
    rects = []
    offsets = [-W, 0, W]
    for i, (vals, off) in enumerate(zip(vals_list, offsets)):
        r = ax.bar(x + off, vals, W,
                   label=BM3_LABEL[i],
                   color=BM3_FILL[i],
                   edgecolor=BM3_EDGE[i],
                   linewidth=0.75)
        rects.append(r)
    return rects

def _savefig(fig, stem):
    fig.savefig(os.path.join(out_dir, stem + ".pdf"))
    fig.savefig(os.path.join(out_dir, stem + ".svg"))
    fig.savefig(os.path.join(out_dir, stem + ".png"), dpi=300)
    plt.close(fig)

def _fbox(ax, x, y, w, h, label, sublabel="", fc=WHITE, ec=BLACK, fs=8.5, lw=1.0, title_top=False):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.01,rounding_size=0.04",
                       fc=fc, ec=ec, lw=lw, zorder=4, clip_on=False)
    ax.add_patch(p)
    if sublabel:
        if title_top:
            ax.text(x+w/2, y+h-0.28, label, ha="center", va="center",
                    fontsize=fs, color=BLACK, fontweight="bold", zorder=5)
            ax.text(x+w/2, y+(h-0.4)/2, sublabel, ha="center", va="center",
                    fontsize=fs-1.2, color=BLACK, zorder=5, linespacing=1.25)
        else:
            ax.text(x+w/2, y+h*0.68, label, ha="center", va="center",
                    fontsize=fs, color=BLACK, fontweight="bold", zorder=5, linespacing=1.25)
            ax.text(x+w/2, y+h*0.32, sublabel, ha="center", va="center",
                    fontsize=fs-1.2, color=BLACK, zorder=5, linespacing=1.2)
    else:
        ax.text(x+w/2, y+h/2, label, ha="center", va="center",
                fontsize=fs, color=BLACK, fontweight="bold", zorder=5, linespacing=1.25)

def _farrow(ax, x0, y0, x1, y1, color=BLACK, lw=1.1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=10, connectionstyle="arc3,rad=0"))

def _band(ax, x, y, w, h, fc=PANEL, ec=GRAY_D, lw=0.6, label=""):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.01,rounding_size=0.03",
                       fc=fc, ec=ec, lw=lw, zorder=1, clip_on=False)
    ax.add_patch(p)
    if label:
        ax.text(x+0.08, y+h+0.06, label, ha="left", va="bottom",
                fontsize=7.6, color=GRAY_D, fontweight="bold", zorder=6)


# ══════════════════════════════════════════════════════════════════════════════
# FIG 1 — Framework Architecture Overview (Large Legible Typography)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig1():
    fig, ax = plt.subplots(figsize=(6.4, 2.75), dpi=300)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.5); ax.axis('off')

    def band(x, y, w, h, label=''):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.04',
                           fc='#F8F9FA', ec='#C0C0C0', lw=0.8, zorder=1)
        ax.add_patch(p)
        if label:
            ax.text(x+0.08, y+h+0.06, label, ha='left', va='bottom',
                    fontsize=8.2, color='#2A2A2A', fontweight='bold', zorder=6)

    def fbox(x, y, w, h, title, sub='', fc=WHITE, ec=BLACK, lw=0.9, fs_title=8.2, fs_sub=7.2):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.04',
                           fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(p)
        if sub:
            ax.text(x+w/2, y+h-0.38, title, ha='center', va='center',
                    fontsize=fs_title, color=BLACK, fontweight='bold', zorder=5)
            ax.text(x+w/2, y+(h-0.45)/2, sub, ha='center', va='center',
                    fontsize=fs_sub, color='#1A1A1A', zorder=5, linespacing=1.22)
        else:
            ax.text(x+w/2, y+h/2, title, ha='center', va='center',
                    fontsize=fs_title, color=BLACK, fontweight='bold', zorder=5)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=BLACK, lw=1.1,
                                    mutation_scale=9, connectionstyle='arc3,rad=0'), zorder=10)

    # (1) Input Query
    band(0.08, 0.15, 1.85, 3.8, label='(1) Input Query')
    fbox(0.18, 0.35, 1.65, 3.4, 'Evaluation Unit\n$\\mathcal{U}_i$',
         sub='Query $(x_i, y_i)$\nContext length $L$\nTask class $\\tau_i$\nDense anchor $\\mathcal{R}$',
         fc='#EBF3FB', ec=METHOD_COLORS[0], fs_title=8.4, fs_sub=7.3)

    # (2) Pairwise Execution
    band(2.10, 0.15, 2.50, 3.8, label='(2) Pairwise Execution')
    fbox(2.20, 2.25, 2.30, 1.50, 'Dense FP16 Anchor',
         sub='$\\hat{y}_{\\mathrm{dense}} = \\mathcal{M}(x_i \\mid \\mathcal{R}_{\\mathrm{dense}})$\n$s_{\\mathrm{dense}}$, $M_{\\mathrm{dense}}$, 16 bpt',
         fc='#F0F0F0', ec='#555555', fs_title=8.2, fs_sub=7.1)
    fbox(2.20, 0.35, 2.30, 1.50, 'Candidate $\\mathcal{A}(\\mathcal{B})$',
         sub='Quant / Evict / SVD\n$s_{\\mathrm{method}}$, $M_{\\mathrm{method}}$, $b_{\\mathrm{eff}}$',
         fc='#E6F7FF', ec=METHOD_COLORS[2], fs_title=8.2, fs_sub=7.1)
    arrow(1.83, 3.00, 2.20, 3.00)
    arrow(1.83, 1.10, 2.20, 1.10)

    # (3) Metric Engine
    band(4.78, 0.15, 2.45, 3.8, label='(3) Metric Engine')
    fbox(4.88, 0.35, 2.25, 3.4, 'CRBench Engine',
         sub='Canonical Quality:\n$Q_{\\mathrm{abs}} = 100 \\cdot s_{\\mathrm{method}}$\n\nMemory Savings:\n$R_{\\mathrm{mem}} = 100(1 - \\frac{M_{\\mathrm{m}}}{M_{\\mathrm{d}}})$\n\nEffective Bits:\n$b_{\\mathrm{eff}} = \\frac{8 \\cdot M_{\\mathrm{m}}}{2 N_l N_h d_h L}$',
         fc=WHITE, ec='#444444', fs_title=8.4, fs_sub=7.2)
    arrow(4.50, 3.00, 4.88, 2.70)
    arrow(4.50, 1.10, 4.88, 1.30)

    # (4) Decoupled Tracks
    band(7.40, 0.15, 2.52, 3.8, label='(4) Decoupled Tracks')
    fbox(7.50, 2.25, 2.32, 1.50, 'Part 1: Algorithmic',
         sub='$\\mathcal{S}_{\\mathrm{res}} = 0.70 Q_{\\mathrm{abs}} + 0.30 R_{\\mathrm{mem}}$\nQuality--Memory Fidelity',
         fc='#EAF7EC', ec=METHOD_COLORS[6], fs_title=8.2, fs_sub=7.1)
    fbox(7.50, 0.35, 2.32, 1.50, 'Part 2: Runtime Track',
         sub='$\\mathcal{S}_{\\mathrm{sys}} = 0.70 \\mathcal{S}_{\\mathrm{res}} + 0.30 R_{\\mathrm{sys}}$\nTTFT · Decode · VRAM',
         fc='#FFF5E6', ec=METHOD_COLORS[5], fs_title=8.2, fs_sub=7.1)
    arrow(7.13, 2.70, 7.50, 3.00)
    arrow(7.13, 1.30, 7.50, 1.10)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.03)
    _savefig(fig, 'fig1_framework_overview')
    plt.close(fig)
    print('  fig1 done')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 2 — Query Pipeline (Large Legible Typography)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig2():
    fig, ax = plt.subplots(figsize=(6.4, 2.95), dpi=300)
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis('off')

    def band(x, y, w, h, label=''):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.03',
                           fc='#F8F9FA', ec='#C0C0C0', lw=0.8, zorder=1)
        ax.add_patch(p)
        if label:
            ax.text(x+0.08, y+h+0.05, label, ha='left', va='bottom',
                    fontsize=8.2, color='#2A2A2A', fontweight='bold', zorder=6)

    def fbox(x, y, w, h, title, sub='', fc=WHITE, ec=BLACK, lw=0.9, fs_title=8.0, fs_sub=6.9):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.01,rounding_size=0.03',
                           fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(p)
        if sub:
            ax.text(x+w/2, y+h-0.34, title, ha='center', va='center',
                    fontsize=fs_title, color=BLACK, fontweight='bold', zorder=5)
            ax.text(x+w/2, y+(h-0.42)/2, sub, ha='center', va='center',
                    fontsize=fs_sub, color='#1A1A1A', zorder=5, linespacing=1.20)
        else:
            ax.text(x+w/2, y+h/2, title, ha='center', va='center',
                    fontsize=fs_title, color=BLACK, fontweight='bold', zorder=5)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=BLACK, lw=1.1,
                                    mutation_scale=9, connectionstyle='arc3,rad=0'), zorder=10)

    # Top band: Candidate Execution Pipeline
    band(0.08, 2.35, 9.84, 2.05, label='Candidate Execution Pipeline')
    boxes_top = [
        (0.18, 2.47, 2.25, 1.75, '1. Chunked Prefill',
         'Chunk = 4096 tokens\nShared KV allocation\nZero intermediate spikes', '#F0F0F0', '#555555'),
        (2.68, 2.47, 2.25, 1.75, '2. KV Transform',
         'Quant / Evict / SVD\nApplied post-prefill\nExplicit metadata logged', '#E6F7FF', METHOD_COLORS[2]),
        (5.18, 2.47, 2.25, 1.75, '3. Forward & Decode',
         'Re-forward prompt tail\nAgainst compressed cache\nGreedy decode ($T = 0$)', '#F0F0F0', '#555555'),
        (7.68, 2.47, 2.14, 1.75, '4. Profiling & Log',
         'Task score $s_{i,\\mathrm{method}}$\nTTFT & Decode tok/s\nPeak resident VRAM', '#EAF7EC', METHOD_COLORS[6]),
    ]
    for (x, y, w, h, lbl, sub, fc, ec) in boxes_top:
        fbox(x, y, w, h, lbl, sub, fc=fc, ec=ec)
    arrow(2.43, 3.35, 2.68, 3.35)
    arrow(4.93, 3.35, 5.18, 3.35)
    arrow(7.43, 3.35, 7.68, 3.35)

    # Bottom band: Dense Reference Anchor & Metric Extraction
    band(0.08, 0.10, 9.84, 1.98, label='Dense Reference Anchor & Metric Extraction')
    boxes_bot = [
        (0.18, 0.20, 2.95, 1.70, 'Dense Reference Anchor',
         '$\\hat{y}_{\\mathrm{dense}} = \\mathcal{M}(x_i \\mid \\mathcal{R}_{\\mathrm{dense}})$\n$s_{\\mathrm{dense}} = \\mathrm{Eval}(\\hat{y}_{\\mathrm{dense}}, y_i)$\nReference Memory: $M_{\\mathrm{dense}}$', '#F0F0F0', '#555555'),
        (3.40, 0.20, 3.10, 1.70, 'Canonical Quality & Memory',
         'Part 1 Quality: $Q_{\\mathrm{abs}} = 100 \\cdot s_i$\nSavings: $R_{\\mathrm{mem}} = 100(1 - \\frac{M_{\\mathrm{m}}}{M_{\\mathrm{d}}})$\nDiagnostic: $\\Delta_i = s_{\\mathrm{method}} - s_{\\mathrm{dense}}$', WHITE, '#444444'),
        (6.78, 0.20, 3.04, 1.70, 'Decoupled Track Logging',
         'Part 1: $\\mathcal{S}_{\\mathrm{res}} = 0.70 Q + 0.30 R_{\\mathrm{mem}}$\nPart 2: $\\mathcal{S}_{\\mathrm{sys}} = 0.70 \\mathcal{S}_{\\mathrm{res}} + 0.30 R_{\\mathrm{sys}}$\nDual: $(\\mathcal{S}, C_{\\mathrm{paired}})$ · SUCCESS/OOM', '#EAF7EC', METHOD_COLORS[6]),
    ]
    for (x, y, w, h, lbl, sub, fc, ec) in boxes_bot:
        fbox(x, y, w, h, lbl, sub, fc=fc, ec=ec)
    arrow(3.13, 1.05, 3.40, 1.05)
    arrow(6.50, 1.05, 6.78, 1.05)

    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.03)
    _savefig(fig, 'fig2_query_pipeline')
    plt.close(fig)
    print('  fig2 done')


# ══════════════════════════════════════════════════════════════════════════════
# FIG 3 — Cross-Model S_res Leaderboard (Solid blue-to-mint bars)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig3():
    e2b = [51.2, 51.5, 51.0, 42.5, 32.5, 40.5, 34.8, 28.0]
    e4b = [62.1, 60.8, 59.4, 52.9, 44.8, 52.4, 36.6, 36.4]
    q7b = [61.1, 59.5, 56.3, 48.6, 42.0, 38.6, 44.3, 32.6]

    fig, ax = plt.subplots(figsize=(7.8, 3.45), dpi=300)
    _box_axes(ax)
    x = np.arange(len(methods))
    rects = _draw_solid_bars(ax, x, [e2b, e4b, q7b])
    for r in rects:
        _bar_label(ax, r, fontsize=5.2)

    ax.set_ylabel(r"Resource Score  $\mathcal{S}_{\mathrm{res}}$", labelpad=6)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=20, ha="right")
    ax.set_ylim(0, 82)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.axhline(32.5, linestyle="--", color="#444444", lw=1.1, zorder=1, label="Dense E2B Baseline (32.5)")

    _legend_above(ax, ncol=4, extra=0.20)
    _savefig(fig, "fig3_sres_cross")
    print("  fig3 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 4 — Cross-Model S_sys Leaderboard (Clean centered SnapKV callout)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig4():
    e2b = [64.6, 60.6, 59.1, 52.7, 52.8, 57.4, 54.3, 49.6]
    e4b = [73.4, 48.6, 66.3, 60.0, 61.4, 65.9, 55.6, 55.5]
    q7b = [72.8, 53.7, 54.2, 48.5, 59.4, 54.9, 61.0, 52.8]

    fig, ax = plt.subplots(figsize=(7.8, 3.45), dpi=300)
    _box_axes(ax)
    x = np.arange(len(methods))
    rects = _draw_solid_bars(ax, x, [e2b, e4b, q7b])
    for r in rects:
        _bar_label(ax, r, fontsize=5.3)

    ax.set_ylabel(r"Hardware Serving Score  $\mathcal{S}_{\mathrm{sys}}$", labelpad=6)
    ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=18, ha="right")
    ax.set_ylim(0, 110)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    # Dense E2B reference line in legend:
    ax.axhline(52.8, linestyle="--", color="#555555", lw=1.1, zorder=1, label="Dense E2B Baseline (52.8)")

    # SnapKV E4B penalty callout (cleanly positioned above bar):
    ax.annotate(
        "SnapKV (E4B): $\\phi = 0.20\\times$\n(Prefill TTFT 83.4s Penalty)",
        xy=(1.0, 48.6),
        xytext=(1.4, 88.0),
        fontsize=7.2,
        color=RED_A,
        ha="center",
        va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF2F2", edgecolor=RED_A, lw=0.9),
        arrowprops=dict(arrowstyle="-|>", color=RED_A, lw=1.0, shrinkA=3, shrinkB=4),
        zorder=7
    )

    _legend_above(ax, ncol=4, extra=0.20)
    _savefig(fig, "fig4_ssys_cross")
    print("  fig4 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 5 — Task Heatmap (Rich Multi-Hue Colormap, Seamless Cells — NO white gaps)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig5():
    tasks = ['Single NIAH', 'Multi NIAH', 'RULER KV', 'Var. Track.', 'Multi-Hop QA']
    data = np.array([
        [96.7, 32.2, 65.6,  1.1, 86.7],
        [93.9, 32.2, 61.1,  0.0, 87.8],
        [85.9, 28.9, 51.1,  0.0, 91.1],
        [100.0,31.8, 60.0,  0.0, 86.7],
        [100.0,32.9, 64.5,  0.0, 86.7],
        [50.6, 17.1, 43.3,  0.0, 70.0],
        [50.4, 16.7, 14.5,  0.0, 60.0],
        [36.7, 13.0, 10.0,  0.0, 37.8],
    ])

    fig, ax = plt.subplots(figsize=(7.4, 3.8), dpi=300)
    ax.grid(False)

    # Rich multi-hue colormap: seamless color transitions, zero white gaps!
    im = ax.imshow(data, cmap="YlGnBu", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(tasks, fontsize=8.5, fontweight="bold")
    ax.set_yticklabels(LABELS, fontsize=8.5)
    ax.tick_params(left=False, bottom=False)

    for i in range(len(methods)):
        for j in range(len(tasks)):
            val = data[i, j]
            tc = WHITE if val >= 55 else BLACK
            fw = "bold" if val >= 75 else "normal"
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                    fontsize=8.0, color=tc, fontweight=fw)

    ax.set_xlim(-0.5, len(tasks) - 0.5)
    ax.set_ylim(len(methods) - 0.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_edgecolor(BLACK)
        s.set_linewidth(0.8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
    cbar.set_label("Cross-Model Mean Accuracy (%)", fontsize=8.5, fontweight="bold")
    cbar.ax.tick_params(labelsize=7.8)
    cbar.outline.set_linewidth(0.7)
    cbar.outline.set_edgecolor(BLACK)
    cbar.set_ticks([0, 25, 50, 75, 100])

    _savefig(fig, "fig5_task_heatmap")
    print("  fig5 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 6 — Context Scaling (Solid lines, Filled dots, Distinct Colors)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig6():
    ctx_labels = ['2K', '8K', '32K', '64K', '128K']
    x_pts = np.arange(len(ctx_labels))
    e2b = {
        'kivi':          [53.3, 53.3, 53.3, 46.7, 25.6],
        'snapkv':        [51.7, 53.3, 53.3, 46.7, 28.9],
        'dkv_mid':       [53.3, 46.7, 51.1, 42.2, 30.0],
        'dkv_high':      [53.3, 53.3, 53.3, 46.7, 26.7],
        'dense_fp16':    [53.3, 53.3, 53.3, 46.7, 25.6],
        'low_rank_kv':   [53.3, 40.7, 26.7, 24.5, 11.1],
        'streaming_llm': [31.0, 23.3, 23.3, 23.3, 13.3],
        'kv_merging':    [23.1, 13.8, 11.9, 10.5,  7.1],
    }

    fig, ax = plt.subplots(figsize=(7.8, 3.55), dpi=300)
    _box_axes(ax)

    # Distinct high-contrast colors, solid lines, filled dots on the lines for markers!
    for i, m in enumerate(methods):
        ax.plot(x_pts, e2b[m], label=LABELS[i],
                color=METHOD_COLORS[i],
                linestyle='-',
                marker='o',
                markersize=6.5,
                linewidth=2.0,
                markerfacecolor=METHOD_COLORS[i],
                markeredgecolor=WHITE,
                markeredgewidth=0.9,
                zorder=4)

    ax.axvspan(2.5, 4.4, color=RED_A, alpha=0.06, zorder=0)
    ax.axvline(x=2.5, color=RED_A, lw=0.9, linestyle="--", alpha=0.7)
    ax.text(3.45, 58.0, "Context Cliff (Base Saturation)", ha="center", fontsize=7.2, color=RED_A,
            bbox=dict(boxstyle="round,pad=0.25", fc=WHITE, ec=RED_A, lw=0.7))

    ax.set_ylabel("Mean Task Accuracy (%)", labelpad=6)
    ax.set_xlabel("Context Sequence Horizon (Tokens)", labelpad=5)
    ax.set_xticks(x_pts); ax.set_xticklabels(ctx_labels)
    ax.set_ylim(0, 68)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    _legend_above(ax, ncol=4, extra=0.30)
    _savefig(fig, "fig6_context_scaling")
    print("  fig6 done")


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# FIG 7 — Quality–Memory Pareto Frontier (Distinct Markers & Colors by Method)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig7():
    # Format: (b_eff, acc, method_idx, model_idx)
    # model_idx: 0=E2B, 1=E4B, 2=Qwen7B
    pts = [
        (6.01,46.4,0,0),(6.00,46.8,1,0),(5.46,44.7,2,0),(10.78,46.7,3,0),
        (16.00,46.4,4,0),(6.07,31.2,5,0),(6.00,22.9,6,0),(6.01,13.3,7,0),
        (6.04,62.0,0,1),(6.00,60.0,1,1),(5.00,55.3,2,1),(10.68,61.3,3,1),
        (16.00,64.0,4,1),(6.27,48.8,5,1),(6.00,25.5,6,1),(6.03,25.3,7,1),
        (6.04,60.7,0,2),(6.00,58.2,1,2),(5.84,53.3,2,2),(12.14,59.1,3,2),
        (16.00,60.0,4,2),(6.09,28.6,5,2),(6.00,36.5,6,2),(6.03,19.9,7,2),
    ]

    MODEL_NAMES = ['Gemma 4 E2B', 'Gemma 4 E4B', 'Qwen2.5 7B']
    MODEL_MARKERS = ['o', 's', '^']

    fig, ax = plt.subplots(figsize=(7.8, 3.65), dpi=300)
    _box_axes(ax)

    # Plot points with model-specific marker shape and method-specific color:
    for b, acc, m_idx, mod_idx in pts:
        ax.scatter(b, acc, color=METHOD_COLORS[m_idx],
                   marker=MODEL_MARKERS[mod_idx],
                   s=72,
                   edgecolors=WHITE, linewidths=0.9,
                   alpha=0.95, zorder=5)

    # Clean solid Pareto frontier line across all evaluated configurations
    pareto_x = [5.00, 6.00, 6.04, 10.68, 16.00]
    pareto_y = [55.3, 60.0, 62.0, 61.3,  64.0]
    ax.plot(pareto_x, pareto_y, linestyle="--", color=BLACK,
            linewidth=1.4, label="Pareto Envelope (E4B)", zorder=4)

    # Clean annotations safely positioned inside axes bounds
    ax.annotate("DKV-Mid (5.0 bpt, 55.3%)", xy=(5.00, 55.3), xytext=(4.0, 39.0),
                fontsize=7.2, color=METHOD_COLORS[2], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=METHOD_COLORS[2], lw=0.9))
    ax.annotate("KIVI (6.0 bpt, 62.0%)", xy=(6.04, 62.0), xytext=(7.3, 53.5),
                fontsize=7.2, color=METHOD_COLORS[0], fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=METHOD_COLORS[0], lw=0.9))
    ax.annotate("Dense FP16 (16.0 bpt, 64.0%)", xy=(16.00, 64.0), xytext=(11.5, 71.0),
                fontsize=7.2, color=BLACK, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=0.9))

    ax.set_xlabel(r"Effective KV Footprint  $b_{\mathrm{eff}}^{\mathrm{agg}}$  (Bits per Element, lower is better $\leftarrow$)", labelpad=6)
    ax.set_ylabel(r"Absolute Task Quality $Q_{\mathrm{abs}}$ (%)", labelpad=6)
    ax.set_xlim(3.8, 17.2)
    ax.set_ylim(8, 78)

    # Model Shapes Legend (Top Left)
    model_handles = [
        Line2D([0],[0], marker=MODEL_MARKERS[i], color='w',
               markerfacecolor=GRAY_D, markeredgecolor=WHITE,
               markersize=7.5, label=MODEL_NAMES[i])
        for i in range(len(MODEL_NAMES))
    ]
    pareto_h = Line2D([0],[0], linestyle="--", color=BLACK, lw=1.4, label="Pareto Envelope")

    l_model = ax.legend(handles=model_handles + [pareto_h],
                        loc="upper left", bbox_to_anchor=(0.02, 0.98),
                        ncol=2, fontsize=7.2, edgecolor="#BAC2CC", framealpha=0.95,
                        title=r"$\bf{Model\ (Marker\ Shape)}$", title_fontsize=7.5)
    ax.add_artist(l_model)

    # Method Colors Legend (Lower Right Quadrant)
    method_handles = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=METHOD_COLORS[i], markeredgecolor=WHITE,
               markersize=7.5, label=LABELS[i])
        for i in range(len(LABELS))
    ]

    ax.legend(handles=method_handles,
              loc="lower right", bbox_to_anchor=(0.98, 0.04),
              ncol=2, fontsize=7.2, edgecolor="#BAC2CC", framealpha=0.95,
              title=r"$\bf{Method\ (Marker\ Color)}$", title_fontsize=7.5)

    _savefig(fig, "fig7_pareto_frontier")
    print("  fig7 done")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Sensitivity & Scoring Formula Comparison (Simplified & Crystal Clear)
# ══════════════════════════════════════════════════════════════════════════════
# FIG 8A — Ranking Stability Across α (Standalone with Top Legend)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig8a():
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=300)
    _box_axes(ax)
    ax.grid(True, linestyle="--", alpha=0.35, color="#CCCCCC", zorder=0)

    alphas = np.linspace(0.10, 0.90, 80)
    alpha_dots = np.array([0.15, 0.35, 0.55, 0.70, 0.85])
    Q = {'kivi':62.1,'snapkv':60.8,'dkv_mid':59.4,'dkv_high':52.9,'dense_fp16':44.8,
         'low_rank_kv':52.4,'streaming_llm':36.6,'kv_merging':36.4}
    R = {'kivi':62.2,'snapkv':62.5,'dkv_mid':68.7,'dkv_high':33.3,'dense_fp16':0.0,
         'low_rank_kv':60.8,'streaming_llm':62.5,'kv_merging':62.3}

    for i, m in enumerate(methods):
        scores = [a * Q[m] + (1.0 - a) * R[m] for a in alphas]
        dot_scores = [a * Q[m] + (1.0 - a) * R[m] for a in alpha_dots]
        ax.plot(alphas, scores, label=LABELS[i],
                color=METHOD_COLORS[i], linewidth=1.8, linestyle='-', zorder=4)
        ax.scatter(alpha_dots, dot_scores, color=METHOD_COLORS[i],
                   marker=METHOD_MARKERS[i],
                   s=24, edgecolors=WHITE, linewidths=0.6, zorder=5)

    ax.axvline(0.70, color=RED_A, linestyle="--", linewidth=1.2, zorder=3)
    ax.text(0.71, 3, r"Default $\alpha = 0.70$", color=RED_A, fontsize=7.2, va="bottom", fontweight="bold")

    ax.axvspan(0.40, 0.85, color="#1976D2", alpha=0.07, zorder=1)
    ax.text(0.625, 75, "Stable Ranking Window\n$\\alpha \\in [0.40, 0.85]$",
            ha="center", va="center", fontsize=6.8, color="#0D47A1",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc=WHITE, ec="#1976D2", lw=0.7))

    ax.set_xlabel(r"Quality Preference Weight $\alpha$", labelpad=5)
    ax.set_ylabel(r"Resource Utility $\mathcal{S}_{\mathrm{res}}$", labelpad=5)
    ax.set_xlim(0.08, 0.92)
    ax.set_ylim(0, 85)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    # Place legend ABOVE plot: zero coverage of graph lines!
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.26),
              mode="expand", ncol=4, frameon=True, edgecolor="#BAC2CC", framealpha=1.0,
              fontsize=6.8, borderpad=0.3)

    _savefig(fig, "fig8a_ranking_stability")
    plt.close(fig)
    print("  fig8a done")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8B — Formulation Comparison (Standalone with Top Legend)
# ══════════════════════════════════════════════════════════════════════════════
def make_fig8b():
    fig, ax = plt.subplots(figsize=(4.6, 3.5), dpi=300)
    _box_axes(ax)
    ax.grid(True, linestyle="--", alpha=0.35, color="#CCCCCC", zorder=0)

    Q_arr = np.linspace(0.01, 100, 200)
    R_fix, a = 75.0, 0.70

    s_lin  = a * Q_arr + (1.0 - a) * R_fix
    s_cd   = (Q_arr ** a) * (R_fix ** (1.0 - a))
    s_harm = (a / Q_arr + (1.0 - a) / R_fix) ** (-1)

    ax.plot(Q_arr, s_lin,  label=r"Linear [CRBench]: $\alpha Q + (1-\alpha)R$",
            color="#1E5AA8", lw=2.4, zorder=4)
    ax.plot(Q_arr, s_cd,   label=r"Cobb-Douglas: $Q^\alpha R^{1-\alpha}$",
            color="#D32F2F", lw=1.8, linestyle="--", zorder=4)
    ax.plot(Q_arr, s_harm, label=r"Harmonic: $(\alpha/Q + (1-\alpha)/R)^{-1}$",
            color="#2E7D32", lw=1.8, linestyle="-.", zorder=4)

    ax.axhline(R_fix, color=GRAY_D, lw=0.9, linestyle=":", alpha=0.7, zorder=2)
    ax.text(102, R_fix, "$R=75$", fontsize=7.2, color=GRAY_D, va="center")

    ax.annotate("Linear: Partial credit\neven at low quality",
                xy=(28, s_lin[np.argmin(abs(Q_arr-28))]),
                xytext=(10, 68), fontsize=6.8, color="#1E5AA8", fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color="#1E5AA8", lw=0.85))

    ax.annotate("Harmonic & Cobb-Douglas\ncollapse to zero at $Q=0$",
                xy=(2.0, 2.0), xytext=(38, 14), fontsize=6.8, color="#555555",
                arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))

    ax.set_xlabel("Task Quality Retention $Q$ (%)", labelpad=5)
    ax.set_ylabel(r"Composite Utility Score $\mathcal{S}$", labelpad=5)
    ax.set_xlim(0, 100); ax.set_ylim(0, 105)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    # Place legend ABOVE plot: zero coverage of graph lines!
    h, l = ax.get_legend_handles_labels()
    ax.legend(h, l, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.26),
              ncol=1, frameon=True, edgecolor="#BAC2CC", framealpha=1.0,
              fontsize=6.7, borderpad=0.3)

    _savefig(fig, "fig8b_formulation_comparison")
    plt.close(fig)
    print("  fig8b done")


# ══════════════════════════════════════════════════════════════════════════════
# FIG 8 — Combined Sensitivity & Scoring Formula Comparison
# ══════════════════════════════════════════════════════════════════════════════
def make_fig8():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 3.55), dpi=300)
    for ax in (ax1, ax2):
        _box_axes(ax)
        ax.grid(True, linestyle="--", alpha=0.35, color="#CCCCCC", zorder=0)

    # ── Panel (a): Preference Sensitivity α ∈ [0.10, 0.90] ────────────────────
    alphas = np.linspace(0.10, 0.90, 80)
    alpha_dots = np.array([0.15, 0.35, 0.55, 0.70, 0.85])
    Q = {'kivi':62.1,'snapkv':60.8,'dkv_mid':59.4,'dkv_high':52.9,'dense_fp16':44.8,
         'low_rank_kv':52.4,'streaming_llm':36.6,'kv_merging':36.4}
    R = {'kivi':62.2,'snapkv':62.5,'dkv_mid':68.7,'dkv_high':33.3,'dense_fp16':0.0,
         'low_rank_kv':60.8,'streaming_llm':62.5,'kv_merging':62.3}

    for i, m in enumerate(methods):
        scores = [a * Q[m] + (1.0 - a) * R[m] for a in alphas]
        dot_scores = [a * Q[m] + (1.0 - a) * R[m] for a in alpha_dots]
        ax1.plot(alphas, scores, label=LABELS[i],
                 color=METHOD_COLORS[i], linewidth=1.8, linestyle='-', zorder=4)
        ax1.scatter(alpha_dots, dot_scores, color=METHOD_COLORS[i],
                    marker=METHOD_MARKERS[i],
                    s=24, edgecolors=WHITE, linewidths=0.6, zorder=5)

    ax1.axvline(0.70, color=RED_A, linestyle="--", linewidth=1.2, zorder=3)
    ax1.text(0.71, 3, r"Default $\alpha = 0.70$", color=RED_A, fontsize=7.2, va="bottom", fontweight="bold")

    ax1.axvspan(0.40, 0.85, color="#1976D2", alpha=0.07, zorder=1)
    ax1.text(0.625, 75, "Stable Ranking Window\n$\\alpha \\in [0.40, 0.85]$",
             ha="center", va="center", fontsize=6.8, color="#0D47A1",
             fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.22", fc=WHITE, ec="#1976D2", lw=0.7))

    ax1.set_xlabel(r"Quality Preference Weight $\alpha$", labelpad=5)
    ax1.set_ylabel(r"Resource Score $\mathcal{S}_{\mathrm{res}}$", labelpad=5)
    ax1.set_title(r"(a) Ranking Stability Across $\alpha$", pad=8, fontsize=8.8, fontweight="bold")
    ax1.set_xlim(0.08, 0.92)
    ax1.set_ylim(0, 85)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(20))

    # Place legend ABOVE panel (a) in 4 columns:
    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(h1, l1, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.26),
               mode="expand", ncol=4, frameon=True, edgecolor="#BAC2CC", framealpha=1.0,
               fontsize=6.6, borderpad=0.25)

    # ── Panel (b): Why Linear? Formula Comparison at R = 75% ──────────────────
    Q_arr = np.linspace(0.01, 100, 200)
    R_fix, a = 75.0, 0.70

    s_lin  = a * Q_arr + (1.0 - a) * R_fix
    s_cd   = (Q_arr ** a) * (R_fix ** (1.0 - a))
    s_harm = (a / Q_arr + (1.0 - a) / R_fix) ** (-1)

    ax2.plot(Q_arr, s_lin,  label=r"Linear: $\alpha Q + (1-\alpha)R$",
             color="#1E5AA8", lw=2.4, zorder=4)
    ax2.plot(Q_arr, s_cd,   label=r"Cobb-Douglas: $Q^\alpha R^{1-\alpha}$",
             color="#D32F2F", lw=1.8, linestyle="--", zorder=4)
    ax2.plot(Q_arr, s_harm, label=r"Harmonic: $(\alpha/Q + (1-\alpha)/R)^{-1}$",
             color="#2E7D32", lw=1.8, linestyle="-.", zorder=4)

    ax2.axhline(R_fix, color=GRAY_D, lw=0.9, linestyle=":", alpha=0.7, zorder=2)
    ax2.text(102, R_fix, "$R=75$", fontsize=7.2, color=GRAY_D, va="center")

    ax2.annotate("Linear: Partial credit\neven at low quality",
                 xy=(28, s_lin[np.argmin(abs(Q_arr-28))]),
                 xytext=(10, 68), fontsize=6.8, color="#1E5AA8", fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color="#1E5AA8", lw=0.85))

    ax2.annotate("Harmonic & Cobb-Douglas\ncollapse to zero at $Q=0$",
                 xy=(2.0, 2.0), xytext=(38, 14), fontsize=6.8, color="#555555",
                 arrowprops=dict(arrowstyle="-|>", color="#555555", lw=0.8))

    ax2.set_xlabel("Task Quality Retention $Q$ (%)", labelpad=5)
    ax2.set_ylabel(r"Composite Utility Score $\mathcal{S}$", labelpad=5)
    ax2.set_title(r"(b) Formulation Comparison ($R_{\mathrm{mem}}=75\%$)", pad=8, fontsize=8.8, fontweight="bold")
    ax2.set_xlim(0, 100); ax2.set_ylim(0, 105)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(20))

    # Place legend ABOVE panel (b):
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h2, l2, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.26),
               ncol=1, frameon=True, edgecolor="#BAC2CC", framealpha=1.0,
               fontsize=6.6, borderpad=0.25)

    plt.subplots_adjust(wspace=0.28, left=0.07, right=0.96, top=0.88, bottom=0.15)
    _savefig(fig, "fig8_alpha_sensitivity")
    plt.close(fig)
    print("  fig8 done")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating all CRBench figures (clean blue-to-mint theme)...")
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig7()
    make_fig8()
    make_fig8a()
    make_fig8b()
    print("Done. All figures written to:", out_dir)
