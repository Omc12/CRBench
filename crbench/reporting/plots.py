"""
Publication-quality plotting engine for CRBench.
Generates vector and high-DPI figures for quality-memory Pareto frontiers, AUQC scaling,
iso-budget retention, and Part 1 vs Part 2 system tradeoffs.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from crbench.scoring.pareto import OperatingPoint, ParetoFrontierResult
from crbench.scoring.resource_score import CRBenchResourceScoreResult
from crbench.scoring.system_score import CRBenchSystemScoreResult

# Set high quality academic publication style
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 1.1,
    "grid.color": "#E5E5E5",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
    "figure.dpi": 300,
})

PALETTE = {
    "dense_fp16": "#1f77b4",
    "dense": "#1f77b4",
    "kv_quant_int8": "#2ca02c",
    "kv_quant_int4": "#d62728",
    "int4": "#d62728",
    "int8": "#2ca02c",
    "snapkv": "#9467bd",
    "streaming_llm": "#8c564b",
    "kv_merging": "#e377c2",
    "low_rank_kv": "#ff7f0e",
    "dkv": "#17becf",
    "custom": "#bcbd22",
}


def get_color(method_name: str) -> str:
    norm = method_name.lower()
    for k, v in PALETTE.items():
        if k in norm:
            return v
    return "#333333"


def plot_quality_vs_memory_pareto(
    all_points_by_method: Dict[str, List[OperatingPoint]],
    context_length: int,
    output_path: Optional[str] = None,
    log_x: bool = True
) -> plt.Figure:
    """
    Plots Normalized Quality (%) vs. Memory Budget (Bits/token or MB) with Pareto Frontiers.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for method, points in all_points_by_method.items():
        pts = [p for p in points if p.context_length == context_length]
        if not pts:
            continue
        
        pts_sorted = sorted(pts, key=lambda x: x.memory_cost)
        x_vals = [p.memory_cost for p in pts_sorted]
        y_vals = [p.quality_score for p in pts_sorted]
        color = get_color(method)

        ax.plot(x_vals, y_vals, marker="o", markersize=7, label=method, color=color, linewidth=2.0, alpha=0.9)

    ax.set_title(f"CRBench Quality–Resource Frontier (Context Length: {context_length:,} tokens)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Effective KV Memory Cost (Bits per Token / MB)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Contextual Quality Retention (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(-2, 105)
    
    if log_x:
        ax.set_xscale("log")

    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="lower right")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_auqc_vs_context_length(
    resource_results: List[CRBenchResourceScoreResult],
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots AUQC Resource Score scaling across context lengths.
    """
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for res in resource_results:
        lengths = sorted(res.context_scores.keys())
        auqcs = [res.context_scores[L].auqc_result.auqc_score for L in lengths]
        color = get_color(res.method_name)

        ax.plot(lengths, auqcs, marker="s", markersize=7, label=f"{res.method_name} (S_res={res.resource_score:.1f})", color=color, linewidth=2.2)

    ax.set_title("CRBench Context Scaling: AUQC Score vs. Context Length", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Context Length (Tokens)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Area Under Quality-Resource Curve (AUQC Score)", fontsize=11, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-2, 105)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_isobudget_comparison(
    resource_results: List[CRBenchResourceScoreResult],
    context_length: int,
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots grouped bar chart of contextual capability retention across standard budgets (2, 4, 8, 16 bpt).
    """
    fig, ax = plt.subplots(figsize=(9, 5.5))

    methods = [r.method_name for r in resource_results if context_length in r.context_scores]
    if not methods:
        return fig

    budgets = [2.0, 4.0, 8.0, 16.0]
    n_methods = len(methods)
    bar_width = 0.8 / max(1, n_methods)
    indices = np.arange(len(budgets))

    for i, res in enumerate(resource_results):
        if context_length not in res.context_scores:
            continue
        iso = res.context_scores[context_length].isobudget_result
        scores = [iso.budget_scores.get(b, 0.0) for b in budgets]
        color = get_color(res.method_name)
        
        pos = indices + (i - (n_methods - 1) / 2) * bar_width
        ax.bar(pos, scores, width=bar_width, label=res.method_name, color=color, edgecolor="#333333", alpha=0.85)

    ax.set_title(f"Iso-Budget Quality Retention at Context Length {context_length:,}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Resource Budget Constraint (Effective Bits Per Token)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Normalized Quality Retained (%)", fontsize=11, fontweight="bold")
    ax.set_xticks(indices)
    ax.set_xticklabels([f"{b:.0f} bpt" for b in budgets], fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(True, axis="y")
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_resource_vs_system_score(
    system_results: List[CRBenchSystemScoreResult],
    output_path: Optional[str] = None
) -> plt.Figure:
    """
    Plots Part 1 Resource Score vs. Part 2 System Score to highlight runtime deployment viability.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for res in system_results:
        color = get_color(res.method_name)
        x = res.resource_score
        y = res.system_score
        
        ax.scatter(x, y, s=160, color=color, edgecolors="#333333", linewidth=1.5, zorder=4)
        ax.annotate(
            res.method_name,
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=9,
            fontweight="bold"
        )

    # Reference diagonal (y = x)
    ax.plot([0, 100], [0, 100], linestyle="--", color="#888888", label="Ideal System Efficiency (S_sys = S_res)", alpha=0.7)

    ax.set_title("CRBench Part 1 (Resource Score) vs. Part 2 (System Score)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("CRBench Resource Score (S_res) [Memory & Quality]", fontsize=11, fontweight="bold")
    ax.set_ylabel("CRBench System Score (S_sys) [Memory, Quality & Latency]", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 105)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="upper left")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    return fig
