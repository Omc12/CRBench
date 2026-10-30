#!/usr/bin/env python3
"""Query pipeline diagram matching DKV / DeepSeek visual language."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os

BLACK     = "#111111"
BLUE_XL   = "#DDE7F8"
BLUE_D    = "#274F9E"
EMER_L    = "#BFE3D4"
EMERALD   = "#2E9E77"
GRAY_XL   = "#EDEFF2"
WHITE     = "#FFFFFF"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"],
    "font.size": 8.5,
    "text.color": BLACK,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

out_dir = os.path.join(os.path.dirname(__file__), "figures")

fig, ax = plt.subplots(figsize=(7.5, 3.2), dpi=300)
ax.axis('off')

def rbox(x, y, w, h, title, text, fc=GRAY_XL, ec=BLACK):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.025",
                       fc=fc, ec=ec, lw=0.8, zorder=3)
    ax.add_patch(p)
    if title:
        ax.text(x + w/2, y + h - 0.16, title, ha="center", va="center", fontsize=8.0,
                fontweight="bold", color=BLACK, zorder=4)
        ax.text(x + w/2, y + (h - 0.22)/2, text, ha="center", va="center", fontsize=7.0,
                color=BLACK, zorder=4, linespacing=1.25)
    else:
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=7.2,
                color=BLACK, zorder=4, linespacing=1.25)

# Stage 1: Chunked Prefill
rbox(0.02, 0.55, 0.19, 0.38, "1. Chunked Prefill", "Chunk size: 4096\nShared KV Cache\nPrevents alloc spike", fc=GRAY_XL)
ax.annotate("", xy=(0.26, 0.74), xytext=(0.21, 0.74), arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))

# Stage 2: Cache Transform
rbox(0.26, 0.55, 0.21, 0.38, "2. KV Transform", "Transform on resident cache:\nQuantize / Evict / SVD\nState byte accounting", fc=BLUE_XL, ec=BLUE_D)
ax.annotate("", xy=(0.52, 0.74), xytext=(0.47, 0.74), arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))

# Stage 3: Forward Tail
rbox(0.52, 0.55, 0.21, 0.38, "3. Final Step Forward", "Re-forward prompt tail\nagainst transformed cache\nGreedy decode ($T=0$)", fc=GRAY_XL)
ax.annotate("", xy=(0.78, 0.74), xytext=(0.73, 0.74), arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))

# Stage 4: Scoring
rbox(0.78, 0.55, 0.20, 0.38, "4. Evaluation", "Raw score $s_{i,\\mathrm{method}}$\nTTFT / Latency profiling\nPeak resident VRAM", fc=EMER_L, ec=EMERALD)

# Lower Panel: Reference Anchor & Dual Metric Extraction
rbox(0.02, 0.06, 0.96, 0.40, "", "", fc=WHITE, ec=BLACK)
ax.text(0.05, 0.38, "Dense Reference Anchor & Metric Extraction", fontsize=8.5, fontweight="bold", color=BLACK)

# Dense box
rbox(0.05, 0.11, 0.26, 0.23, "Parallel Dense Run", "$\\hat{y}_{i,\\mathrm{dense}} = \\mathcal{M}(x_i \\mid \\mathcal{R}_{\\mathrm{dense}})$\n$s_{i,\\mathrm{dense}} = \\mathrm{Eval}(\\hat{y}_{i,\\mathrm{dense}}, y_i)$", fc=GRAY_XL)
ax.annotate("", xy=(0.35, 0.225), xytext=(0.31, 0.225), arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))

# Normalizer box
rbox(0.35, 0.11, 0.30, 0.23, "Canonical Metrics", "$Q_{\\mathrm{abs},i} = 100 \\cdot s_{\\mathrm{method}}$\n$R_{\\mathrm{mem},i} = 100(1 - \\frac{M_{\\mathrm{method}}}{M_{\\mathrm{dense}}})$", fc=GRAY_XL)
ax.annotate("", xy=(0.69, 0.225), xytext=(0.65, 0.225), arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))

# Output box
rbox(0.69, 0.11, 0.27, 0.23, "Non-Destructive Logging", "$\\mathcal{S}_{\\mathrm{res},i} = 0.70 Q_i + 0.30 R_{\\mathrm{mem},i}$\n$\\mathcal{S}_{\\mathrm{sys},i} = 0.70 Q_i + 0.30 R_{\\mathrm{sys},i}$\nStatus: SUCCESS / OOM", fc=BLUE_XL, ec=BLUE_D)

plt.subplots_adjust(top=0.98, bottom=0.02, left=0.01, right=0.99)
fig.savefig(os.path.join(out_dir, "fig2_query_pipeline.pdf"))
fig.savefig(os.path.join(out_dir, "fig2_query_pipeline.svg"))
plt.close(fig)

print("Figure 2 regenerated cleanly!")
