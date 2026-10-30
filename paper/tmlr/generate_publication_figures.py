#!/usr/bin/env python3
"""CRBench publication figures generator matching the exact DeepSeek-V3 / DKV paper visual system.

Visual Language:
  * Full thin box spines on all four sides (black, 0.8pt);
  * Framed legend with a hairline border (#B9BEC6);
  * Text is always black; colour is reserved for series/bars;
  * Palette = blues (primary) + emerald (secondary) + amber + neutral grays;
  * Subtle dotted y-grid; bold black value labels; generous whitespace;
  * No in-figure title (the LaTeX caption carries the title).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D

# Palette matching DKV style.py
BLACK     = "#111111"
BLUE      = "#3D6FD1"
BLUE_D    = "#274F9E"
BLUE_L    = "#A9C4EF"
BLUE_XL   = "#DDE7F8"
EMERALD   = "#2E9E77"
EMER_L    = "#BFE3D4"
AMBER     = "#D97706"
RED       = "#C0392B"
GRAY      = "#C4C9D0"
GRAY_D    = "#7A828C"
GRAY_XL   = "#EDEFF2"
GRID      = "#E4E7EB"
LEGEND_EC = "#B9BEC6"
WHITE     = "#FFFFFF"
PURPLE    = "#6B5CA5"
TEAL      = "#0891B2"

plt.rcParams.update({
    "figure.facecolor":  WHITE,
    "axes.facecolor":    WHITE,
    "savefig.facecolor": WHITE,
    "font.family":       "DejaVu Sans",
    "font.size":         9.0,
    "mathtext.fontset":  "dejavusans",
    "text.color":        BLACK,
    "axes.titlesize":    9.5,
    "axes.labelsize":    9.0,
    "axes.labelcolor":   BLACK,
    "axes.edgecolor":    BLACK,
    "axes.linewidth":    0.8,
    "axes.grid":         True,
    "axes.grid.axis":    "y",
    "axes.axisbelow":    True,
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "grid.color":        GRID,
    "grid.linewidth":    0.7,
    "grid.linestyle":    (0, (1, 3)),
    "xtick.color":       BLACK,
    "ytick.color":       BLACK,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.size":  3.0,
    "ytick.major.size":  3.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "legend.frameon":    True,
    "legend.edgecolor":  LEGEND_EC,
    "legend.framealpha": 1.0,
    "legend.fancybox":   False,
    "legend.fontsize":   8.0,
    "legend.borderpad":  0.5,
    "legend.labelspacing": 0.35,
    "legend.handlelength": 1.6,
    "lines.linewidth":   1.6,
    "lines.markersize":  4.5,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.03,
})

out_dir = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(out_dir, exist_ok=True)

methods = ['kivi', 'snapkv', 'dkv_mid', 'dkv_high', 'dense_fp16', 'low_rank_kv', 'streaming_llm', 'kv_merging']
method_labels = ['KIVI (INT4)', 'SnapKV', 'DKV-Mid', 'DKV-High', 'Dense FP16', 'Low-Rank KV', 'StreamingLLM', 'KV-Merging']

def box_axes(ax):
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_color(BLACK)
        s.set_linewidth(0.8)
    ax.tick_params(color=BLACK)
    return ax

# ==============================================================================
# FIGURE 1: Framework Overview Architecture
# ==============================================================================
def make_fig1():
    fig, ax = plt.subplots(figsize=(7.4, 2.7), dpi=300)
    ax.axis('off')
    
    def box_chip(x, y, w, h, title, lines, fc=GRAY_XL, ec=BLACK, lw=0.8):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.03",
                           fc=fc, ec=ec, lw=lw, zorder=3)
        ax.add_patch(p)
        ax.text(x + w/2, y + h - 0.18, title, ha="center", va="center",
                fontsize=8.0, fontweight="bold", color=BLACK, zorder=4)
        for i, line in enumerate(lines):
            ax.text(x + w/2, y + h - 0.38 - i*0.17, line, ha="center", va="center",
                    fontsize=7.0, color=BLACK, zorder=4)

    # 1. Input Box
    box_chip(0.02, 0.22, 0.16, 0.56, "Input Query",
             ["Evaluation tuple", "$(x_i, y_i)$", "Context length $L$"], fc=GRAY_XL)
    
    # Arrows
    ax.annotate("", xy=(0.24, 0.65), xytext=(0.18, 0.52),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    ax.annotate("", xy=(0.24, 0.35), xytext=(0.18, 0.48),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    
    # 2. Parallel Evaluation Branches
    box_chip(0.24, 0.50, 0.22, 0.45, "Dense FP16 Baseline",
             ["Full precision ref.", "Score $s_{\mathrm{dense}}$", "$b_{\mathrm{eff}} = 16.0$ bpt"], fc=GRAY_XL)
    box_chip(0.24, 0.05, 0.22, 0.45, "Candidate Adapter",
             ["Quant / Evict / SVD", "Score $s_{\mathrm{method}}$", "$b_{\mathrm{eff}} = 8 M / N_{\mathrm{KV}}$"],
             fc=BLUE_XL, ec=BLUE_D)
    
    # Arrows to Normalization
    ax.annotate("", xy=(0.51, 0.55), xytext=(0.46, 0.68),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    ax.annotate("", xy=(0.51, 0.45), xytext=(0.46, 0.32),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    
    # 3. Canonical scoring engine
    box_chip(0.51, 0.12, 0.24, 0.76, "CRBench Metric Engine",
             ["Absolute quality $Q_{\\mathrm{abs},i}$:",
              "$Q_{\\mathrm{abs},i} = 100 \\cdot s_{\\mathrm{method}}$",
              "Dense run logged pairwise",
              "Memory savings $R_{\\mathrm{mem},i}$:",
              "$R_{\\mathrm{mem},i} = 100(1 - \\frac{M_{\\mathrm{m}}}{M_{\\mathrm{d}}})$"],
             fc=WHITE, ec=BLACK)
    
    # Arrows to Outputs
    ax.annotate("", xy=(0.80, 0.68), xytext=(0.75, 0.58),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    ax.annotate("", xy=(0.80, 0.32), xytext=(0.75, 0.42),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0))
    
    # 4. Two Decoupled Tracks
    box_chip(0.80, 0.50, 0.18, 0.45, "Part 1: S_res",
             ["Quality--memory utility", "$\\mathcal{S}_{\\mathrm{res}} = \\alpha Q_{\\mathrm{abs}} + (1-\\alpha)R$"],
             fc=EMER_L, ec=EMERALD)
    box_chip(0.80, 0.05, 0.18, 0.45, "Part 2: S_sys",
             ["System Serving Utility", "TTFT + Decode + VRAM"],
             fc=BLUE_XL, ec=BLUE_D)
    
    fig.savefig(os.path.join(out_dir, "fig1_framework_overview.pdf"))
    fig.savefig(os.path.join(out_dir, "fig1_framework_overview.svg"))
    plt.close(fig)

# ==============================================================================
# FIGURE 3: Cross-Model Part 1 Score (S_res)
# ==============================================================================
def make_fig3():
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=300)
    box_axes(ax)
    
    e2b_sres = [51.2, 51.5, 51.0, 42.5, 32.5, 40.5, 34.8, 28.0]
    e4b_sres = [62.1, 60.8, 59.4, 52.9, 44.8, 52.4, 36.6, 36.4]
    q7b_sres = [61.1, 59.5, 56.3, 48.6, 42.0, 38.6, 44.3, 32.6]
    
    x = np.arange(len(methods))
    width = 0.26
    
    r1 = ax.bar(x - width, e2b_sres, width, label='Gemma 4 E2B (2K-128K)', color=BLUE, edgecolor=BLACK, linewidth=0.7)
    r2 = ax.bar(x, e4b_sres, width, label='Gemma 4 E4B (2K-32K)', color=EMERALD, edgecolor=BLACK, linewidth=0.7)
    r3 = ax.bar(x + width, q7b_sres, width, label='Qwen2.5 7B (2K-32K)', color=GRAY, edgecolor=BLACK, linewidth=0.7)
    
    ax.set_ylabel('Resource score ($S_{\\mathrm{res}}$)', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=15, ha='right', fontsize=8.0)
    ax.set_ylim(0, 75)
    
    for r in [r1, r2, r3]:
        for bar in r:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width()/2, h),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        va="bottom", fontsize=6.2, color=BLACK)
            
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.15),
              mode='expand', ncol=3, frameon=True, edgecolor=LEGEND_EC, framealpha=1.0)
    
    fig.savefig(os.path.join(out_dir, "fig3_sres_cross.pdf"))
    fig.savefig(os.path.join(out_dir, "fig3_sres_cross.svg"))
    plt.close(fig)

# ==============================================================================
# FIGURE 4: Cross-Model Part 2 Score (S_sys)
# ==============================================================================
def make_fig4():
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=300)
    box_axes(ax)
    
    e2b_ssys = [64.6, 60.6, 59.1, 52.7, 52.8, 57.4, 54.3, 49.6]
    e4b_ssys = [73.4, 48.6, 66.3, 60.0, 61.4, 65.9, 55.6, 55.5]
    q7b_ssys = [72.8, 53.7, 54.2, 48.5, 59.4, 54.9, 61.0, 52.8]
    
    x = np.arange(len(methods))
    width = 0.26
    
    r1 = ax.bar(x - width, e2b_ssys, width, label='Gemma 4 E2B ($S_{\\mathrm{sys}}$)', color=BLUE, edgecolor=BLACK, linewidth=0.7)
    r2 = ax.bar(x, e4b_ssys, width, label='Gemma 4 E4B ($S_{\\mathrm{sys}}$)', color=EMERALD, edgecolor=BLACK, linewidth=0.7)
    r3 = ax.bar(x + width, q7b_ssys, width, label='Qwen2.5 7B ($S_{\\mathrm{sys}}$)', color=GRAY, edgecolor=BLACK, linewidth=0.7)
    
    ax.set_ylabel('System score ($S_{\\mathrm{sys}}$)', fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=15, ha='right', fontsize=8.0)
    ax.set_ylim(0, 85)
    
    for r in [r1, r2, r3]:
        for bar in r:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width()/2, h),
                        textcoords="offset points", xytext=(0, 2), ha="center",
                        va="bottom", fontsize=6.2, color=BLACK)
            
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.15),
              mode='expand', ncol=3, frameon=True, edgecolor=LEGEND_EC, framealpha=1.0)
    
    fig.savefig(os.path.join(out_dir, "fig4_ssys_cross.pdf"))
    fig.savefig(os.path.join(out_dir, "fig4_ssys_cross.svg"))
    plt.close(fig)

# ==============================================================================
# FIGURE 5: Task Heatmap
# ==============================================================================
def make_fig5():
    fig, ax = plt.subplots(figsize=(6.8, 3.2), dpi=300)
    box_axes(ax)
    
    tasks = ['Single NIAH', 'Multi NIAH', 'RULER KV', 'Variable Track.', 'Multi-Hop QA']
    data = np.array([
        [96.7, 31.8, 65.6, 1.1, 86.7],  # KIVI
        [93.9, 32.2, 61.1, 0.0, 87.8],  # SnapKV
        [85.9, 27.4, 51.1, 0.0, 91.1],  # DKV Mid
        [100.0, 31.8, 60.0, 0.0, 86.7], # DKV High
        [100.0, 32.9, 64.5, 0.0, 86.7], # Dense FP16
        [50.6, 17.1, 43.3, 0.0, 70.0],  # Low-Rank KV
        [50.4, 16.7, 14.5, 0.0, 60.0],  # StreamingLLM
        [36.7, 13.0, 10.0, 0.0, 37.8]   # KV Merging
    ])
    
    im = ax.imshow(data, cmap='Blues', aspect='auto', vmin=0, vmax=100)
    
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_yticks(np.arange(len(methods)))
    ax.set_xticklabels(tasks, fontsize=8.2)
    ax.set_yticklabels(method_labels, fontsize=8.2)
    
    for i in range(len(methods)):
        for j in range(len(tasks)):
            val = data[i, j]
            tc = WHITE if val > 55 else BLACK
            ax.text(j, i, f"{val:.1f}%", ha="center", va="center", color=tc, fontsize=7.6)
            
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.03, pad=0.03)
    cbar.set_label('Mean accuracy (%)', fontsize=8.0)
    cbar.ax.tick_params(labelsize=7.5)
    cbar.outline.set_edgecolor(BLACK)
    cbar.outline.set_linewidth(0.8)
    
    fig.savefig(os.path.join(out_dir, "fig5_task_heatmap.pdf"))
    fig.savefig(os.path.join(out_dir, "fig5_task_heatmap.svg"))
    plt.close(fig)

# ==============================================================================
# FIGURE 6: Context Scaling (Gemma 4 E2B 2K-128K)
# ==============================================================================
def make_fig6():
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
    box_axes(ax)
    
    ctx_lengths = [2048, 8192, 32768, 65536, 131072]
    ctx_labels = ['2K', '8K', '32K', '64K', '128K']
    
    e2b_ladder = {
        'kivi': [53.3, 53.3, 53.3, 46.7, 25.6],
        'snapkv': [51.7, 53.3, 53.3, 46.7, 28.9],
        'dkv_mid': [53.3, 46.7, 51.1, 42.2, 30.0],
        'dkv_high': [53.3, 53.3, 53.3, 46.7, 26.7],
        'dense_fp16': [53.3, 53.3, 53.3, 46.7, 25.6],
        'low_rank_kv': [53.3, 40.7, 26.7, 24.5, 11.1],
        'streaming_llm': [31.0, 23.3, 23.3, 23.3, 13.3],
        'kv_merging': [23.1, 13.8, 11.9, 10.5, 7.1]
    }
    
    mcolors = [BLUE, EMERALD, AMBER, RED, BLACK, PURPLE, TEAL, GRAY_D]
    mstyles = ['-', '-', '-', '--', ':', '-.', '-.', ':']
    mmarks  = ['s', '^', 'D', 'o', 'x', 'v', 'p', '*']
    
    for i, m in enumerate(methods):
        ax.plot(range(len(ctx_lengths)), e2b_ladder[m], label=method_labels[i],
                color=mcolors[i], marker=mmarks[i], markersize=4.8, linewidth=1.4,
                linestyle=mstyles[i])
        
    ax.set_ylabel('Mean accuracy (%)', fontsize=8.5)
    ax.set_xlabel('Context length (tokens)', fontsize=8.5)
    ax.set_xticks(range(len(ctx_lengths)))
    ax.set_xticklabels(ctx_labels)
    ax.set_ylim(0, 60)
    
    # Degradation cliff callout
    ax.axvspan(3, 4, color=GRAY_XL, alpha=0.6, zorder=0)
    ax.text(3.5, 52, 'Context cliff\n(Base model saturation)',
            ha='center', va='center', fontsize=7.2, color=RED,
            bbox=dict(boxstyle='round,pad=0.2', facecolor=WHITE, edgecolor=RED, lw=0.6))
    
    ax.legend(loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.18),
              mode='expand', ncol=4, frameon=True, edgecolor=LEGEND_EC, framealpha=1.0)
    
    fig.savefig(os.path.join(out_dir, "fig6_context_scaling.pdf"))
    fig.savefig(os.path.join(out_dir, "fig6_context_scaling.svg"))
    plt.close(fig)

# ==============================================================================
# FIGURE 7: Pareto Frontier
# ==============================================================================
def make_fig7():
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=300)
    box_axes(ax)
    
    points = [
        # (b_eff, acc, method_idx, model_idx)
        (6.01, 46.4, 0, 0), (6.00, 46.8, 1, 0), (5.46, 44.7, 2, 0), (10.78, 46.7, 3, 0),
        (16.00, 46.4, 4, 0), (6.07, 31.2, 5, 0), (6.00, 22.9, 6, 0), (6.01, 13.3, 7, 0),
        (6.04, 62.0, 0, 1), (6.00, 60.0, 1, 1), (5.00, 55.3, 2, 1), (10.68, 61.3, 3, 1),
        (16.00, 64.0, 4, 1), (6.27, 48.8, 5, 1), (6.00, 25.5, 6, 1), (6.03, 25.3, 7, 1),
        (6.04, 60.7, 0, 2), (6.00, 58.2, 1, 2), (5.84, 53.3, 2, 2), (12.14, 59.1, 3, 2),
        (16.00, 60.0, 4, 2), (6.09, 28.6, 5, 2), (6.00, 36.5, 6, 2), (6.03, 19.9, 7, 2)
    ]
    
    mcolors = [BLUE, EMERALD, AMBER, RED, BLACK, PURPLE, TEAL, GRAY_D]
    mshapes = ['s', '^', 'o']
    
    for b_eff, acc, m_idx, mod_idx in points:
        ax.scatter(b_eff, acc, color=mcolors[m_idx], marker=mshapes[mod_idx], s=38,
                   edgecolors=BLACK, linewidths=0.6, alpha=0.9, zorder=4)
        
    pareto_x = [5.00, 6.00, 6.04, 10.68, 16.00]
    pareto_y = [55.3, 60.0, 62.0, 61.3, 64.0]
    ax.plot(pareto_x, pareto_y, linestyle='--', color=BLACK, linewidth=1.2, label='Pareto envelope (E4B)', zorder=3)
    
    ax.annotate('DKV-Mid (5.0 bpt, 55.3%)', xy=(5.00, 55.3), xytext=(4.2, 46),
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.0), fontsize=7.2, color=BLACK)
    ax.annotate('KIVI (6.0 bpt, 62.0%)', xy=(6.04, 62.0), xytext=(6.5, 66),
                arrowprops=dict(arrowstyle="->", color=BLUE, lw=1.0), fontsize=7.2, color=BLACK)
    ax.annotate('Dense (16.0 bpt, 64.0%)', xy=(16.00, 64.0), xytext=(12.0, 66),
                arrowprops=dict(arrowstyle="->", color=BLACK, lw=1.0), fontsize=7.2, color=BLACK)
    
    ax.set_xlabel('Effective bits per element ($b_{\mathrm{eff}}$, lower is better $\\leftarrow$)', fontsize=8.5)
    ax.set_ylabel('Task accuracy $Q_{\mathrm{abs}}$ (%)', fontsize=8.5)
    ax.set_xlim(3.8, 17.5)
    ax.set_ylim(8, 72)
    
    custom_legend = [
        Line2D([0], [0], marker='s', color='w', label='Gemma 4 E2B', markerfacecolor=GRAY_D, markersize=5),
        Line2D([0], [0], marker='^', color='w', label='Gemma 4 E4B', markerfacecolor=GRAY_D, markersize=5),
        Line2D([0], [0], marker='o', color='w', label='Qwen2.5 7B', markerfacecolor=GRAY_D, markersize=5),
        Line2D([0], [0], linestyle='--', color=BLACK, lw=1.2, label='Pareto envelope')
    ]
    ax.legend(handles=custom_legend, loc='lower left', bbox_to_anchor=(0.0, 1.02, 1.0, 0.15),
              mode='expand', ncol=4, frameon=True, edgecolor=LEGEND_EC, framealpha=1.0)
    
    fig.savefig(os.path.join(out_dir, "fig7_pareto_frontier.pdf"))
    fig.savefig(os.path.join(out_dir, "fig7_pareto_frontier.svg"))
    plt.close(fig)

# ==============================================================================
# FIGURE 8: Scoring Formula Sensitivity
# ==============================================================================
def make_fig8():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.9), dpi=300)
    box_axes(ax1); box_axes(ax2)
    
    alphas = np.linspace(0.1, 0.9, 50)
    qwen_Q = {'kivi': 60.7, 'snapkv': 58.2, 'dkv_mid': 53.3, 'dkv_high': 59.1, 'dense_fp16': 60.0, 'low_rank_kv': 28.6, 'streaming_llm': 36.5, 'kv_merging': 19.9}
    qwen_R = {'kivi': 62.2, 'snapkv': 62.5, 'dkv_mid': 63.5, 'dkv_high': 24.1, 'dense_fp16': 0.0, 'low_rank_kv': 62.0, 'streaming_llm': 62.5, 'kv_merging': 62.3}
    mcolors = [BLUE, EMERALD, AMBER, RED, BLACK, PURPLE, TEAL, GRAY_D]
    
    for i, m in enumerate(methods):
        scores = [a * qwen_Q[m] + (1 - a) * qwen_R[m] for a in alphas]
        ax1.plot(alphas, scores, label=method_labels[i], color=mcolors[i], linewidth=1.3)
        
    ax1.axvline(0.70, color=RED, linestyle=':', linewidth=1.2)
    ax1.text(0.71, 14, r'Default $\alpha=0.70$', color=RED, fontsize=7.2)
    ax1.set_xlabel(r'Preference weight $\alpha$', fontsize=8.5)
    ax1.set_ylabel(r'Utility score ($S_{\mathrm{res}}$)', fontsize=8.5)
    ax1.set_title('(a) Ranking invariance over alpha', fontsize=8.5, color=BLACK)
    ax1.set_xlim(0.1, 0.9)
    ax1.set_ylim(10, 70)
    
    Q_vals = np.linspace(0, 100, 100)
    R_fixed = 75.0
    a = 0.70
    
    s_linear = a * Q_vals + (1 - a) * R_fixed
    s_cd = (Q_vals ** a) * (R_fixed ** (1 - a))
    s_harm = [(a / max(0.01, q) + (1 - a) / R_fixed)**(-1) for q in Q_vals]
    s_power = np.sqrt(a * (Q_vals**2) + (1 - a) * (R_fixed**2))
    
    ax2.plot(Q_vals, s_linear, label='F1 Linear (CRBench)', color=BLUE, linewidth=1.6)
    ax2.plot(Q_vals, s_cd, label='F2 Cobb-Douglas', color=AMBER, linewidth=1.1, linestyle='--')
    ax2.plot(Q_vals, s_harm, label='F3 Harmonic Mean', color=EMERALD, linewidth=1.1, linestyle='-.')
    ax2.plot(Q_vals, s_power, label='F4 Power Mean', color=RED, linewidth=1.1, linestyle=':')
    
    ax2.set_xlabel('Quality retention $Q$ (%)', fontsize=8.5)
    ax2.set_ylabel('Calculated score ($S$)', fontsize=8.5)
    ax2.set_title('(b) Candidate formulations at R=75%', fontsize=8.5, color=BLACK)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, 100)
    ax2.legend(loc='lower right', fontsize=7.0, edgecolor=LEGEND_EC)
    
    ax1.legend(loc='lower left', bbox_to_anchor=(0.0, 1.14, 2.15, 0.2),
               mode='expand', ncol=4, frameon=True, edgecolor=LEGEND_EC, framealpha=1.0, fontsize=7.2)
    
    plt.subplots_adjust(top=0.76, bottom=0.15, left=0.08, right=0.98, wspace=0.26)
    fig.savefig(os.path.join(out_dir, "fig8_alpha_sensitivity.pdf"))
    fig.savefig(os.path.join(out_dir, "fig8_alpha_sensitivity.svg"))
    plt.close(fig)

if __name__ == "__main__":
    make_fig1()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig7()
    make_fig8()
    print("Regenerated all publication figures cleanly!")
