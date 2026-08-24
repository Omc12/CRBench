#!/usr/bin/env python3
"""
CRBench Figure Generation Suite
===============================
Generates 6 publication-quality figures with a unified academic blue color palette:
  1. fig1_framework_overview.pdf / .png     (CRBench Conceptual Architecture)
  2. fig2_query_pipeline.pdf / .png         (Query-Level Evaluation Pipeline)
  3. fig3_pareto_frontier.pdf / .png        (Quality–Memory Pareto Frontier with True Non-Dominated Frontier)
  4. fig4_score_comparison.pdf / .png       (Preliminary Part 1 vs Provisional Part 2 Scores)
  5. fig5_context_scaling.pdf / .png        (Context-Length Scaling: 2K vs 4K Measured AUQC)
  6. fig6_formula_alpha_sensitivity.pdf / .png (Utility Formulation & Alpha Sensitivity Analysis)

All empirical figures use ONLY real measured results from Qwen2.5-0.5B preliminary experiments.
Outputs vector PDF and high-DPI PNG files to paper/tmlr/figures/.
"""

import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# --- Publication Plot Style Configuration ---
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.8,
    "figure.titlesize": 13,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.5,
    "lines.linewidth": 1.6,
    "lines.markersize": 6.5,
})

# --- Unified Academic Blue Color Palette ---
BLUE_PALETTE = {
    "navy_dark": "#0B1D3A",      # Deep Navy / Midnight Blue
    "oxford_blue": "#14365D",    # Oxford Blue
    "royal_blue": "#1D4E89",     # Classic Royal Blue
    "steel_blue": "#2E6F9E",     # Steel Blue
    "cerulean": "#3F88C5",       # Cerulean / Medium Blue
    "teal_blue": "#0081A7",      # Deep Teal Blue
    "cyan_blue": "#00AFB9",      # Vibrant Cyan-Blue
    "light_blue": "#64B5F6",     # Light Blue
    "ice_blue": "#90CAF9",       # Ice Blue
    "slate_gray": "#5C6B73",     # Slate Gray Accent
}

# Method-Specific Colors in Cohesive Blue Scheme
METHOD_COLORS = {
    "dense_fp16": "#0B1D3A",     # Deep Navy
    "low_rank_kv": "#1D4E89",    # Royal Blue
    "custom_dkv": "#3F88C5",     # Cerulean
    "kv_quant_int8": "#0081A7",  # Teal Blue
    "snapkv": "#00AFB9",         # Cyan Blue
    "streaming_llm": "#4895EF",  # Bright Blue
    "kv_quant_int4": "#5C6B73",  # Slate Blue-Gray
    "kv_merging": "#7F96A6",     # Muted Slate
    "kv_quant_int2": "#B0C4DE",  # Light Steel Blue
}

# Measured Preliminary Data on Qwen2.5-0.5B-Instruct (Apple Silicon MPS Profile)
MEASURED_DATA = [
    {
        "name": "dense_fp16",
        "display_name": "Dense FP16 (Baseline)",
        "paradigm": "Dense Reference",
        "b_eff": 16.0,
        "r_mem": 0.0,
        "q_retention": 100.0,
        "s_res": 46.7,
        "auqc_2k": 60.0,
        "auqc_4k": 40.0,
        "ttft_ms": 3424.1,
        "thru_tok_s": 189.3,
        "s_sys": 51.8,
        "color": METHOD_COLORS["dense_fp16"],
        "marker": "s",
    },
    {
        "name": "low_rank_kv",
        "display_name": "Low-Rank KV (SVD)",
        "paradigm": "Subspace Projection",
        "b_eff": 4.12,
        "r_mem": 74.25,
        "q_retention": 90.7,
        "s_res": 85.8,
        "auqc_2k": 88.0,
        "auqc_4k": 83.8,
        "ttft_ms": 35.1,
        "thru_tok_s": 17504.3,
        "s_sys": 100.0,
        "color": METHOD_COLORS["low_rank_kv"],
        "marker": "o",
    },
    {
        "name": "custom_dkv",
        "display_name": "Dynamic KV (DKV)",
        "paradigm": "Dynamic Subspace",
        "b_eff": 4.25,
        "r_mem": 73.44,
        "q_retention": 56.4,
        "s_res": 61.5,
        "auqc_2k": 64.2,
        "auqc_4k": 58.6,
        "ttft_ms": 34.5,
        "thru_tok_s": 17271.1,
        "s_sys": 83.0,
        "color": METHOD_COLORS["custom_dkv"],
        "marker": "D",
    },
    {
        "name": "kv_quant_int8",
        "display_name": "KV Quant INT8",
        "paradigm": "Quantization",
        "b_eff": 8.25,
        "r_mem": 48.44,
        "q_retention": 38.8,
        "s_res": 41.7,
        "auqc_2k": 30.0,
        "auqc_4k": 50.0,
        "ttft_ms": 3339.6,
        "thru_tok_s": 201.7,
        "s_sys": 47.1,
        "color": METHOD_COLORS["kv_quant_int8"],
        "marker": "^",
    },
    {
        "name": "snapkv",
        "display_name": "SnapKV (Heavy Hitter)",
        "paradigm": "Eviction",
        "b_eff": 4.05,
        "r_mem": 74.69,
        "q_retention": 25.0,
        "s_res": 25.0,
        "auqc_2k": 30.0,
        "auqc_4k": 20.0,
        "ttft_ms": 1862.4,
        "thru_tok_s": 385.0,
        "s_sys": 33.5,
        "color": METHOD_COLORS["snapkv"],
        "marker": "v",
    },
    {
        "name": "streaming_llm",
        "display_name": "StreamingLLM (Sink+Local)",
        "paradigm": "Eviction",
        "b_eff": 4.05,
        "r_mem": 74.69,
        "q_retention": 20.0,
        "s_res": 20.0,
        "auqc_2k": 25.0,
        "auqc_4k": 15.0,
        "ttft_ms": 1784.3,
        "thru_tok_s": 397.0,
        "s_sys": 27.0,
        "color": METHOD_COLORS["streaming_llm"],
        "marker": "<",
    },
    {
        "name": "kv_quant_int4",
        "display_name": "KV Quant INT4",
        "paradigm": "Quantization",
        "b_eff": 4.25,
        "r_mem": 73.44,
        "q_retention": 18.5,
        "s_res": 18.5,
        "auqc_2k": 20.0,
        "auqc_4k": 17.0,
        "ttft_ms": 3491.0,
        "thru_tok_s": 240.1,
        "s_sys": 20.0,
        "color": METHOD_COLORS["kv_quant_int4"],
        "marker": "P",
    },
    {
        "name": "kv_merging",
        "display_name": "KV Merging (Pooling)",
        "paradigm": "Merging",
        "b_eff": 4.10,
        "r_mem": 74.38,
        "q_retention": 15.0,
        "s_res": 15.0,
        "auqc_2k": 18.0,
        "auqc_4k": 12.0,
        "ttft_ms": 1861.6,
        "thru_tok_s": 368.2,
        "s_sys": 20.1,
        "color": METHOD_COLORS["kv_merging"],
        "marker": "X",
    },
    {
        "name": "kv_quant_int2",
        "display_name": "KV Quant INT2",
        "paradigm": "Quantization",
        "b_eff": 2.25,
        "r_mem": 85.94,
        "q_retention": 8.2,
        "s_res": 8.2,
        "auqc_2k": 10.0,
        "auqc_4k": 6.5,
        "ttft_ms": 3666.3,
        "thru_tok_s": 184.0,
        "s_sys": 8.4,
        "color": METHOD_COLORS["kv_quant_int2"],
        "marker": "*",
    },
]


def save_figure(fig, base_path: Path):
    """Save both PDF (vector) and PNG (raster) formats."""
    base_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = base_path.with_suffix(".pdf")
    png_path = base_path.with_suffix(".png")
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] Saved: {pdf_path.name} & {png_path.name}")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 1: CRBench Framework Overview (Conceptual Architecture)
# ──────────────────────────────────────────────────────────────────────────────
def generate_fig1_framework_overview(output_dir: Path):
    print("Generating Figure 1: CRBench Framework Overview...")
    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    ax.axis("off")

    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)

    def draw_box(x, y, w, h, title, subtitle, color, text_color="white", alpha=0.95, edge="#0B1D3A"):
        rect = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.6,rounding_size=2.0",
            facecolor=color, edgecolor=edge, linewidth=1.2, alpha=alpha, zorder=2
        )
        ax.add_patch(rect)
        if subtitle:
            ax.text(
                x + w/2, y + h*0.68,
                title, color=text_color, fontweight="bold",
                ha="center", va="center", fontsize=9.2, zorder=3
            )
            ax.text(
                x + w/2, y + h*0.32,
                subtitle, color=text_color, style="italic",
                ha="center", va="center", fontsize=7.8, zorder=3, linespacing=1.2
            )
        else:
            ax.text(
                x + w/2, y + h/2,
                title, color=text_color, fontweight="bold",
                ha="center", va="center", fontsize=9.2, zorder=3
            )

    # 1. Input Box (Prompt & Model)
    draw_box(2, 40, 17, 24, "Input Context (q)", "Query + Context x\nGround Truth y", BLUE_PALETTE["navy_dark"])

    # 2. Dual Execution Branches
    # Dense Branch (Top)
    draw_box(26, 68, 23, 22, "Dense Reference", "Uncompressed 16-bit KV\ns_dense, M_dense", BLUE_PALETTE["oxford_blue"])
    # Candidate Branch (Bottom)
    draw_box(26, 14, 23, 22, "Candidate Method A(B)", "Quant / Eviction / Low-Rank\ns_method, M_method", BLUE_PALETTE["royal_blue"])

    # 3. Measurement & Normalization Track
    draw_box(56, 68, 21, 22, "Quality Retention (Q)", "Q = Relative Retained %\n(Dense-Anchored)", BLUE_PALETTE["steel_blue"])
    draw_box(56, 14, 21, 22, "Resource Savings (R)", "R_mem = Memory Saved %\nb_eff (bits/element)", BLUE_PALETTE["teal_blue"])

    # 4. Part 1 Scoring Engine
    draw_box(83, 56, 20, 24, "Part 1 Resource Score", "S_res = α Q + (1-α) R_mem\n(default α = 0.70)", BLUE_PALETTE["royal_blue"])

    # 5. Part 2 System Track (Bottom Decoupled)
    draw_box(83, 12, 20, 24, "Part 2 System Track", "TTFT + Decode Thru\nS_sys = α Q + (1-α) R_sys", BLUE_PALETTE["cerulean"])

    # Flow Arrows
    arrow_props = dict(arrowstyle="->", lw=1.6, color=BLUE_PALETTE["navy_dark"], shrinkA=4, shrinkB=4)
    dash_arrow_props = dict(arrowstyle="->", lw=1.4, color=BLUE_PALETTE["teal_blue"], linestyle="--", shrinkA=4, shrinkB=4)

    # Input to Branches
    ax.annotate("", xy=(26, 79), xytext=(19, 56), arrowprops=arrow_props)
    ax.annotate("", xy=(26, 25), xytext=(19, 48), arrowprops=arrow_props)

    # Branches to Normalization
    ax.annotate("", xy=(56, 79), xytext=(49, 79), arrowprops=arrow_props)
    ax.annotate("", xy=(56, 25), xytext=(49, 25), arrowprops=arrow_props)
    
    # --- Visually Prominent Pairwise Anchoring Connector ---
    rect_anchor = patches.FancyBboxPatch(
        (59.5, 43.5), 14, 17,
        boxstyle="round,pad=0.4,rounding_size=1.2",
        facecolor="#EBF3FA", edgecolor=BLUE_PALETTE["royal_blue"], linewidth=1.4, linestyle="--", zorder=3
    )
    ax.add_patch(rect_anchor)
    ax.text(66.5, 53.5, "Pairwise", fontsize=8.2, fontweight="bold", color=BLUE_PALETTE["oxford_blue"], ha="center", va="center", zorder=4)
    ax.text(66.5, 47.5, "Anchoring", fontsize=8.2, fontweight="bold", color=BLUE_PALETTE["oxford_blue"], ha="center", va="center", zorder=4)

    ax.annotate("", xy=(66.5, 68), xytext=(66.5, 60.5), arrowprops=dict(arrowstyle="<->", lw=1.6, color=BLUE_PALETTE["royal_blue"]), zorder=4)
    ax.annotate("", xy=(66.5, 43.5), xytext=(66.5, 36), arrowprops=dict(arrowstyle="<->", lw=1.6, color=BLUE_PALETTE["royal_blue"]), zorder=4)

    # Normalization to Part 1 Score
    ax.annotate("", xy=(83, 72), xytext=(77, 79), arrowprops=arrow_props)
    ax.annotate("", xy=(83, 64), xytext=(77, 25), arrowprops=arrow_props)

    # Candidate to Part 2 System Track
    ax.annotate("", xy=(83, 24), xytext=(77, 25), arrowprops=dash_arrow_props)

    # Title & Subtitle
    ax.text(52.5, 98, "CRBench Method-Agnostic Evaluation Framework", fontsize=12, fontweight="bold", ha="center", color=BLUE_PALETTE["navy_dark"])
    ax.text(52.5, 93, "Two-Track Decoupled Evaluation: Representation Resource Score (Part 1) & Hardware System Track (Part 2)", fontsize=8.8, ha="center", color=BLUE_PALETTE["slate_gray"])

    save_figure(fig, output_dir / "fig1_framework_overview")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 2: Query-Level Evaluation Pipeline
# ──────────────────────────────────────────────────────────────────────────────
def generate_fig2_query_pipeline(output_dir: Path):
    print("Generating Figure 2: Query-Level Evaluation Pipeline...")
    fig, ax = plt.subplots(figsize=(11.0, 4.8))
    ax.axis("off")
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 100)

    # Step Boxes across horizontal axis (cohesive blue progression)
    steps = [
        ("1. Atomic Input", "Query q_i = (x_i, y_i)\nContext Length L_i", 11, 52, 17, 28, BLUE_PALETTE["navy_dark"]),
        ("2. Dual Inference", "Dense: M(x_i | R_dense)\nMethod: M(x_i | A_cand)", 31, 52, 18, 28, BLUE_PALETTE["oxford_blue"]),
        ("3. Raw Extraction", "Raw: s_dense, s_method\nMemory: M_dense, M_method", 52, 52, 19, 28, BLUE_PALETTE["royal_blue"]),
        ("4. Relative Metric", "Q_i = Relative Retained %\nR_mem = Memory Saved %", 73, 52, 18, 28, BLUE_PALETTE["steel_blue"]),
        ("5. Atomic Score", "S_res,i = α Q_i + (1-α) R_i\nJSON Record Logged", 93, 52, 17, 28, BLUE_PALETTE["teal_blue"]),
    ]

    for title, desc, x, y, w, h, col in steps:
        rect = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.5",
            facecolor=col, edgecolor=BLUE_PALETTE["navy_dark"], linewidth=1.1, alpha=0.95
        )
        ax.add_patch(rect)
        ax.text(x, y + 5.5, title, color="white", fontweight="bold", ha="center", va="center", fontsize=8.8)
        ax.text(x, y - 3.5, desc, color="#F0F4F8", ha="center", va="center", fontsize=7.5, linespacing=1.2)

    # Connectors
    for i in range(len(steps) - 1):
        x1 = steps[i][2] + steps[i][4]/2
        x2 = steps[i+1][2] - steps[i+1][4]/2
        ax.annotate("", xy=(x2, 52), xytext=(x1, 52),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color=BLUE_PALETTE["navy_dark"], shrinkA=3, shrinkB=3))

    # Aggregation Stage Below
    rect_agg = patches.FancyBboxPatch(
        (20, 8), 64, 18,
        boxstyle="round,pad=0.5,rounding_size=1.5",
        facecolor="#F0F5FA", edgecolor=BLUE_PALETTE["steel_blue"], linewidth=1.2, linestyle="--"
    )
    ax.add_patch(rect_agg)
    ax.text(52, 20, "Dataset-Level Aggregation across Queries", fontsize=9.2, fontweight="bold", ha="center", color=BLUE_PALETTE["navy_dark"])
    ax.text(52, 13, "Mean Score S_res = 1/|D| ∑ S_res,i   |   95% Bootstrap CI   |   Non-destructive α Recomputation",
            fontsize=7.8, ha="center", color=BLUE_PALETTE["slate_gray"])

    # Downward arrow from step 5 to aggregation
    ax.annotate("", xy=(78, 26), xytext=(93, 38),
                arrowprops=dict(arrowstyle="->", lw=1.3, color=BLUE_PALETTE["teal_blue"], linestyle=":", shrinkA=2, shrinkB=2))

    ax.text(52, 93, "CRBench Atomic Query-Level Evaluation Pipeline", fontsize=11.5, fontweight="bold", ha="center", color=BLUE_PALETTE["navy_dark"])
    ax.text(52, 87, "Every query evaluates candidate compression against its own uncompressed dense reference", fontsize=8.5, ha="center", color=BLUE_PALETTE["slate_gray"])

    save_figure(fig, output_dir / "fig2_query_pipeline")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 3: Quality–Memory Tradeoff & Pareto Frontier (True Non-Dominated Set)
# ──────────────────────────────────────────────────────────────────────────────
def generate_fig3_pareto_frontier(output_dir: Path):
    print("Generating Figure 3: Quality-Memory Pareto Frontier (True Non-Dominated Line)...")
    fig, ax = plt.subplots(figsize=(7.5, 5.4))

    # Plot Background Iso-Score Utility Contours (S_res = α Q + (1-α) R with α=0.70)
    alpha = 0.70
    r_grid = np.linspace(0, 100, 200)
    q_grid = np.linspace(0, 100, 200)
    R_mesh, Q_mesh = np.meshgrid(r_grid, q_grid)
    S_mesh = alpha * Q_mesh + (1.0 - alpha) * R_mesh

    contour_levels = [20, 35, 50, 65, 70, 80, 90]
    cs = ax.contour(R_mesh, Q_mesh, S_mesh, levels=contour_levels, colors="#C2D4E5", linewidths=0.75, linestyles="--", zorder=1)
    ax.clabel(cs, inline=True, fmt=r"$\mathcal{S}_{\mathrm{res}}=%d$", fontsize=7.5, colors=BLUE_PALETTE["slate_gray"])

    # Highlight Dense Baseline Reference Point
    dense_pt = next(d for d in MEASURED_DATA if d["name"] == "dense_fp16")
    ax.scatter(
        dense_pt["r_mem"], dense_pt["q_retention"],
        color=dense_pt["color"], marker=dense_pt["marker"], s=130, edgecolor="black", linewidth=1.2,
        label=f"{dense_pt['display_name']} [Anchor: S_res=46.7]", zorder=5
    )
    ax.annotate(
        "Dense Reference\n(16.0 bits/elem, Q=100%, R=0%)",
        xy=(dense_pt["r_mem"], dense_pt["q_retention"]),
        xytext=(dense_pt["r_mem"] + 4, dense_pt["q_retention"] - 5),
        fontsize=8.0, fontweight="bold", color=BLUE_PALETTE["navy_dark"],
        arrowprops=dict(arrowstyle="->", color=BLUE_PALETTE["navy_dark"], lw=0.9)
    )

    # Plot Candidate Methods
    for d in MEASURED_DATA:
        if d["name"] == "dense_fp16":
            continue
        ax.scatter(
            d["r_mem"], d["q_retention"],
            color=d["color"], marker=d["marker"], s=95, edgecolor="black", linewidth=0.8,
            label=f"{d['display_name']} ({d['b_eff']} bits/elem, S_res={d['s_res']:.1f})", zorder=4
        )
        # Position annotations intelligently
        offset_x = 2.5
        offset_y = 1.5
        if d["name"] == "low_rank_kv":
            offset_x, offset_y = 2.5, 1.5
        elif d["name"] == "custom_dkv":
            offset_x, offset_y = 2.5, -1.0
        elif d["name"] == "kv_quant_int8":
            offset_x, offset_y = 2.5, 1.5
        elif d["name"] == "kv_quant_int2":
            offset_x, offset_y = 2.5, 1.5
        elif d["name"] == "snapkv":
            offset_x, offset_y = 2.5, 1.5
        elif d["name"] == "streaming_llm":
            offset_x, offset_y = 2.5, -4.5
        elif d["name"] == "kv_merging":
            offset_x, offset_y = -24.0, -3.5
        elif d["name"] == "kv_quant_int4":
            offset_x, offset_y = -26.0, 2.5

        ax.annotate(
            d["name"].replace("kv_", "").replace("_", " "),
            xy=(d["r_mem"], d["q_retention"]),
            xytext=(d["r_mem"] + offset_x, d["q_retention"] + offset_y),
            fontsize=7.5, color=BLUE_PALETTE["navy_dark"]
        )

    # --- Compute & Draw Mathematically True Non-Dominated Pareto Frontier ---
    # Points: (0.0, 100.0) [Dense] -> (74.25, 90.7) [Low-Rank] -> (74.69, 25.0) [SnapKV] -> (85.94, 8.2) [INT2]
    # Note: INT8 (48.44, 38.8) and Custom DKV (73.44, 56.4) are dominated by Low-Rank (74.25, 90.7).
    pareto_points = [
        (0.0, 100.0),       # Dense FP16
        (74.25, 90.7),      # Low-Rank KV (SVD)
        (74.69, 25.0),      # SnapKV (Heavy Hitter)
        (85.94, 8.2),       # KV Quant INT2
    ]
    px = [p[0] for p in pareto_points]
    py = [p[1] for p in pareto_points]
    ax.plot(px, py, color=BLUE_PALETTE["royal_blue"], linestyle="-", linewidth=1.8, alpha=0.85, label="Empirical Pareto Frontier (Non-Dominated)", zorder=3)

    # Annotate the dominant Low-Rank point
    ax.annotate(
        "High-Fidelity Frontier\n(Low-Rank KV: 90.7% Q @ 74.3% R)",
        xy=(74.25, 90.7),
        xytext=(30, 84),
        fontsize=7.5, fontweight="bold", color=BLUE_PALETTE["royal_blue"],
        arrowprops=dict(arrowstyle="->", color=BLUE_PALETTE["royal_blue"], lw=1.1)
    )

    # Plot Configuration & Formatting
    ax.set_xlabel("Memory Resource Savings $R_{\\mathrm{mem}}$ (% vs. Dense FP16)")
    ax.set_ylabel("Normalized Contextual Quality Retention $Q$ (%)")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 105)
    ax.grid(True, linestyle=":", alpha=0.5)

    # Title & Legend
    ax.set_title("Context Quality Retention vs. Memory Resource Savings (0.5B Preliminary Profile)", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="lower left", frameon=True, framealpha=0.92, edgecolor="#C2D4E5", fontsize=7.2)

    save_figure(fig, output_dir / "fig3_pareto_frontier")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 4: Preliminary CRBench Scores (Part 1 vs Provisional Part 2)
# ──────────────────────────────────────────────────────────────────────────────
def generate_fig4_score_comparison(output_dir: Path):
    print("Generating Figure 4: Preliminary CRBench Scores Comparison...")
    fig, ax = plt.subplots(figsize=(8.2, 4.8))

    methods = [d["display_name"] for d in MEASURED_DATA]
    s_res_scores = [d["s_res"] for d in MEASURED_DATA]
    s_sys_scores = [d["s_sys"] for d in MEASURED_DATA]

    y_pos = np.arange(len(methods))
    height = 0.36

    # Horizontal Bar Comparison with Blue Scheme
    rects1 = ax.barh(y_pos + height/2, s_res_scores, height,
                     label=r"Part 1 Resource Score ($\mathcal{S}_{\mathrm{res}}$: Quality + Memory)",
                     color=BLUE_PALETTE["royal_blue"], edgecolor=BLUE_PALETTE["navy_dark"], alpha=0.90, zorder=3)
    rects2 = ax.barh(y_pos - height/2, s_sys_scores, height,
                     label=r"Part 2 System Score [Provisional] ($\mathcal{S}_{\mathrm{sys}}$: Quality + Memory + Latency)",
                     color=BLUE_PALETTE["cyan_blue"], edgecolor=BLUE_PALETTE["teal_blue"], alpha=0.90, zorder=3)

    # Add score values as text on bars
    for rect in rects1:
        w = rect.get_width()
        ax.text(w + 1.2, rect.get_y() + rect.get_height()/2, f"{w:.1f}", va="center", ha="left", fontsize=7.8, color=BLUE_PALETTE["navy_dark"], fontweight="bold")
    for rect in rects2:
        w = rect.get_width()
        ax.text(w + 1.2, rect.get_y() + rect.get_height()/2, f"{w:.1f}", va="center", ha="left", fontsize=7.8, color=BLUE_PALETTE["teal_blue"], fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=8.5)
    ax.invert_yaxis()  # Top-down order

    ax.set_xlabel("CRBench Benchmark Score (0 to 100 Scale)")
    ax.set_xlim(0, 115)
    ax.grid(True, axis="x", linestyle=":", alpha=0.5, zorder=0)

    # Add Dense Reference Line
    ax.axvline(46.7, color=BLUE_PALETTE["navy_dark"], linestyle="--", linewidth=1.0, alpha=0.7, zorder=2)
    ax.text(47.2, len(methods) - 0.2, "Dense Baseline S_res=46.7", fontsize=7.5, color=BLUE_PALETTE["navy_dark"], style="italic")

    # Titles & Notes
    ax.set_title("Preliminary Part 1 (Resource) vs. Provisional Part 2 (System) Scores on Qwen2.5-0.5B", fontsize=10.5, fontweight="bold", pad=10)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92, edgecolor="#C2D4E5", fontsize=7.8)

    save_figure(fig, output_dir / "fig4_score_comparison")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 5: Context-Length Scaling (2K vs 4K Measured AUQC)
# ──────────────────────────────────────────────────────────────────────────────
def generate_fig5_context_scaling(output_dir: Path):
    print("Generating Figure 5: Context-Length Scaling (2K vs 4K)...")
    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    context_lengths = [2048, 4096]

    for d in MEASURED_DATA:
        auqcs = [d["auqc_2k"], d["auqc_4k"]]
        ax.plot(
            context_lengths, auqcs,
            marker=d["marker"], color=d["color"], label=d["display_name"],
            linewidth=1.8, markersize=7.5, zorder=3
        )
        # Label right-side endpoints
        ax.text(
            4150, d["auqc_4k"],
            f"{d['name'].replace('kv_', '').replace('_', ' ')} ({d['auqc_4k']:.1f})",
            va="center", fontsize=7.5, color=d["color"], fontweight="bold"
        )

    ax.set_xticks([2048, 4096])
    ax.set_xticklabels(["2,048 tokens\n(2K)", "4,096 tokens\n(4K)"], fontsize=9.5)
    ax.set_xlim(1800, 4800)
    ax.set_ylim(0, 100)

    ax.set_xlabel("Evaluation Context Length $L$ (Tokens)")
    ax.set_ylabel("Representation AUQC Score (0 to 100)")
    ax.grid(True, linestyle=":", alpha=0.5)

    ax.set_title("Context Representation Scaling from 2K to 4K Tokens (Measured 0.5B Profile)", fontsize=11, fontweight="bold", pad=10)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), frameon=True, framealpha=0.92, edgecolor="#C2D4E5", fontsize=7.5)

    save_figure(fig, output_dir / "fig5_context_scaling")


# ──────────────────────────────────────────────────────────────────────────────
# Figure 6: Scoring Formulation & Alpha Sensitivity Analysis
# ──────────────────────────────────────────────────────────────────────────────
def generate_fig6_formula_alpha_sensitivity(output_dir: Path):
    print("Generating Figure 6: Formula & Alpha Sensitivity Analysis...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # --- Panel (a): Candidate Utility Formulas across Quality Retention Q (for R=75% memory savings) ---
    Q_range = np.linspace(0, 100, 200)
    R_fixed = 75.0
    alpha_default = 0.70

    # Formulas
    f_linear = alpha_default * Q_range + (1 - alpha_default) * R_fixed
    f_cobb = (Q_range ** alpha_default) * (R_fixed ** (1 - alpha_default))
    denom_harmonic = (alpha_default / np.maximum(1e-3, Q_range)) + ((1 - alpha_default) / R_fixed)
    f_harmonic = np.where(Q_range > 0, 1.0 / denom_harmonic, 0.0)
    f_power2 = np.sqrt(alpha_default * (Q_range ** 2) + (1 - alpha_default) * (R_fixed ** 2))

    ax1.plot(Q_range, f_linear, color=BLUE_PALETTE["navy_dark"], lw=2.2, label=r"Linear Additive $\mathcal{S} = \alpha Q + (1-\alpha)R$ [CRBench]")
    ax1.plot(Q_range, f_cobb, color=BLUE_PALETTE["royal_blue"], lw=1.6, linestyle="--", label=r"Cobb-Douglas $\mathcal{S} = Q^\alpha R^{1-\alpha}$")
    ax1.plot(Q_range, f_harmonic, color=BLUE_PALETTE["teal_blue"], lw=1.6, linestyle="-.", label=r"Harmonic Mean")
    ax1.plot(Q_range, f_power2, color=BLUE_PALETTE["cerulean"], lw=1.6, linestyle=":", label=r"Power Mean ($p=2$)")

    # Dense reference baseline line (Q=100, R=0) -> S=70.0
    ax1.axhline(70.0, color=BLUE_PALETTE["navy_dark"], linestyle="--", lw=1.0, alpha=0.7)
    ax1.text(5, 71.5, "Dense FP16 Baseline Score (70.0)", fontsize=7.5, color=BLUE_PALETTE["navy_dark"], style="italic")

    ax1.set_xlabel(r"Quality Retention $Q$ (%) at $R_{\mathrm{mem}} = 75\%$")
    ax1.set_ylabel("Computed Benchmark Score $\\mathcal{S}$")
    ax1.set_title("(a) Utility Formulations at Fixed $R_{\\mathrm{mem}}=75\\%$", fontsize=10.5, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 102)
    ax1.grid(True, linestyle=":", alpha=0.5)
    ax1.legend(loc="lower right", fontsize=7.2, frameon=True, framealpha=0.92)

    # --- Panel (b): Alpha Sensitivity on Selected Method Rankings ---
    alpha_sweep = np.linspace(0.10, 0.90, 100)
    key_methods = [
        ("Dense Reference", 100.0, 0.0, METHOD_COLORS["dense_fp16"], "-"),
        ("Low-Rank KV (SVD)", 90.7, 74.25, METHOD_COLORS["low_rank_kv"], "-"),
        ("Dynamic KV (DKV)", 56.4, 73.44, METHOD_COLORS["custom_dkv"], "-"),
        ("KV Quant INT8", 38.8, 48.44, METHOD_COLORS["kv_quant_int8"], "-"),
        ("SnapKV (Eviction)", 25.0, 74.69, METHOD_COLORS["snapkv"], "-"),
        ("KV Quant INT2 (Collapsing)", 8.2, 85.94, METHOD_COLORS["kv_quant_int2"], "-"),
    ]

    for name, q, r, col, ls in key_methods:
        scores = alpha_sweep * q + (1.0 - alpha_sweep) * r
        ax2.plot(alpha_sweep, scores, label=name, color=col, linestyle=ls, lw=1.8)

    # Highlight Default Alpha = 0.70
    ax2.axvline(0.70, color=BLUE_PALETTE["royal_blue"], linestyle=":", lw=1.5, alpha=0.9)
    ax2.text(0.71, 95, r"Default $\alpha=0.70$", color=BLUE_PALETTE["royal_blue"], fontsize=8.0, fontweight="bold")

    ax2.set_xlabel(r"Quality Preference Weight $\alpha$ ($1-\alpha$ = Memory Weight)")
    ax2.set_ylabel("Part 1 Benchmark Score $\\mathcal{S}_{\\mathrm{res}}$")
    ax2.set_title(r"(b) Method Score Sensitivity across $\alpha \in [0.10, 0.90]$", fontsize=10.5, fontweight="bold")
    ax2.set_xlim(0.10, 0.90)
    ax2.set_ylim(0, 102)
    ax2.grid(True, linestyle=":", alpha=0.5)
    ax2.legend(loc="lower left", fontsize=7.0, frameon=True, framealpha=0.92)

    plt.suptitle("Mathematical Behavior and Sensitivity Analysis of the CRBench Linear Utility Formulation", fontsize=11.5, fontweight="bold", y=1.02)
    save_figure(fig, output_dir / "fig6_formula_alpha_sensitivity")


# ──────────────────────────────────────────────────────────────────────────────
# Main Execution
# ──────────────────────────────────────────────────────────────────────────────
def main():
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    output_dir = repo_root / "paper" / "tmlr" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("CRBench Research Preprint — Figure Generation Suite (Academic Blue Theme)")
    print(f"Output Directory: {output_dir}")
    print("================================================================================")

    generate_fig1_framework_overview(output_dir)
    generate_fig2_query_pipeline(output_dir)
    generate_fig3_pareto_frontier(output_dir)
    generate_fig4_score_comparison(output_dir)
    generate_fig5_context_scaling(output_dir)
    generate_fig6_formula_alpha_sensitivity(output_dir)

    print("================================================================================")
    print("All 6 figures regenerated successfully with True Pareto Frontier & Blue Scheme!")
    print("================================================================================")


if __name__ == "__main__":
    main()
