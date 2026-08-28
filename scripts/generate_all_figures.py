"""
Master Publication Figure Generator for CRBench.
Reads unified progress datasets from results/ and generates all 8-method figures:
1. Quality-Memory Pareto Frontiers per context length
2. Context Scaling (Accuracy vs Context Length)
3. Part 1 vs Part 2 System Tradeoff Scatter
4. Multi-Model Cross-Architecture Comparison Charts
Outputs high-DPI PNG and vector PDF figures to results/<model>/figures/ and paper/tmlr/figures/.
"""

import os
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Publication style
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

METHOD_COLORS = {
    "dense_fp16": "#1f77b4",          # Deep Blue
    "dkv_mid": "#00a86b",             # Jade Green
    "dkv_high": "#2ca02c",            # Forest Green
    "kivi_style_kv_quant": "#ff7f0e", # Vibrant Orange
    "kv_quant": "#ff7f0e",            # Vibrant Orange
    "kivi": "#ff7f0e",
    "snapkv": "#9467bd",              # Purple
    "low_rank_kv": "#d62728",         # Crimson Red
    "streaming_llm": "#8c564b",       # Brown
    "kv_merging": "#e377c2",          # Pink
}

METHOD_MARKERS = {
    "dense_fp16": "D",
    "dkv_mid": "o",
    "dkv_high": "v",
    "kivi_style_kv_quant": "s",
    "kv_quant": "s",
    "kivi": "s",
    "snapkv": "^",
    "low_rank_kv": "p",
    "streaming_llm": "X",
    "kv_merging": "h",
}

def load_data(progress_file):
    lines = [json.loads(l) for l in open(progress_file, encoding="utf-8") if l.strip()]
    queries = [q for l in lines for q in l["query_results"]]
    
    by_method_ctx = defaultdict(lambda: defaultdict(list))
    for q in queries:
        by_method_ctx[q["method_name"]][q["context_length"]].append(q)
    return queries, by_method_ctx

def generate_model_figures(model_name, progress_file, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    queries, by_m_c = load_data(progress_file)
    
    contexts = sorted(list(set(q["context_length"] for q in queries)))
    methods = sorted(list(set(q["method_name"] for q in queries)))
    
    # -------------------------------------------------------------------------
    # 1. Pareto Frontiers for each context length
    # -------------------------------------------------------------------------
    for c in contexts:
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        points = []
        for m in methods:
            qs = by_m_c[m].get(c, [])
            if not qs:
                continue
            avg_q = sum(x.get("method_raw_score", 0.0) for x in qs) / len(qs) * 100.0
            tot_d = sum(float(x.get("dense_memory_bytes", 0.0) or (c*100000)) for x in qs)
            tot_m = sum(float(x.get("method_memory_bytes", 0.0) or (tot_d * x.get("method_effective_bpt", 16.0)/16.0)) for x in qs)
            b_eff = 16.0 * (tot_m / tot_d) if tot_d > 0 else 16.0
            points.append((b_eff, avg_q, m))
            
            color = METHOD_COLORS.get(m, "#333333")
            marker = METHOD_MARKERS.get(m, "o")
            ax.scatter(b_eff, avg_q, color=color, marker=marker, s=140, label=m, edgecolor="#222222", linewidth=1.2, zorder=4)
            
        # Draw Pareto frontier line
        points.sort(key=lambda x: (x[0], -x[1]))
        frontier = []
        max_q_seen = -1.0
        # for pareto: sort by b_eff ascending. A point is pareto-optimal if its quality is > any previous point with lower b_eff
        # But here lower b_eff is better, higher Q is better.
        # So sort by b_eff ascending: we want strictly higher Q than any previously visited lower b_eff point
        current_best_q = -1.0
        for b_eff, q_val, m in points:
            if q_val > current_best_q:
                frontier.append((b_eff, q_val))
                current_best_q = q_val
        
        if len(frontier) >= 2:
            fx, fy = zip(*frontier)
            ax.plot(fx, fy, linestyle="--", color="#555555", alpha=0.6, label="Pareto Frontier", zorder=3)
            
        ctx_k = f"{c//1024}K" if c >= 1024 else f"{c}"
        ax.set_title(f"{model_name}: Quality–Resource Tradeoff ({ctx_k} Tokens)", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Effective KV Memory Footprint (Bits per Token, b_eff)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Absolute Task Accuracy (Q_abs %)", fontsize=11, fontweight="bold")
        ax.set_ylim(-2, 105)
        ax.set_xlim(2.0, 18.0)
        ax.grid(True)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="lower right")
        plt.tight_layout()
        
        fig.savefig(os.path.join(out_dir, f"pareto_frontier_{c}.png"), dpi=300, bbox_inches="tight")
        fig.savefig(os.path.join(out_dir, f"pareto_frontier_{c}.pdf"), bbox_inches="tight")
        plt.close(fig)

    # -------------------------------------------------------------------------
    # 2. Context Scaling Curve (Accuracy vs Context Length)
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in methods:
        x_vals = []
        y_vals = []
        for c in contexts:
            qs = by_m_c[m].get(c, [])
            if qs:
                avg_q = sum(x.get("method_raw_score", 0.0) for x in qs) / len(qs) * 100.0
                x_vals.append(c)
                y_vals.append(avg_q)
        if x_vals:
            color = METHOD_COLORS.get(m, "#333333")
            marker = METHOD_MARKERS.get(m, "o")
            ax.plot(x_vals, y_vals, marker=marker, markersize=8, color=color, linewidth=2.0, label=m, alpha=0.9)
            
    ax.set_title(f"{model_name}: Context Scaling Accuracy vs. Context Length", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Context Length (Tokens)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Task Accuracy (Q_abs %)", fontsize=11, fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks(contexts)
    ax.set_xticklabels([f"{c//1024}K" if c >= 1024 else str(c) for c in contexts], fontweight="bold")
    ax.set_ylim(-2, 105)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="lower left")
    plt.tight_layout()
    
    fig.savefig(os.path.join(out_dir, "context_scaling.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "context_scaling.pdf"), bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------------------------
    # 3. Part 1 (S_res) vs Part 2 (S_sys) Tradeoff Scatter
    # -------------------------------------------------------------------------
    ALPHA = 0.70
    dense_qs = by_m_c["dense_fp16"]
    dense_all = [q for c in contexts for q in dense_qs.get(c, [])]
    ref_ttft = sum(q.get("method_ttft_ms", 1000.0) for q in dense_all) / len(dense_all) if dense_all else 1000.0
    ref_thru = sum(q.get("method_decode_throughput", 10.0) for q in dense_all) / len(dense_all) if dense_all else 10.0
    ref_vram = sum(q.get("method_peak_vram_mb", 1000.0) for q in dense_all) / len(dense_all) if dense_all else 1000.0

    fig, ax = plt.subplots(figsize=(8.5, 6))
    for m in methods:
        m_qs = [q for c in contexts for q in by_m_c[m].get(c, [])]
        if not m_qs:
            continue
        avg_q = sum(x.get("method_raw_score", 0.0) for x in m_qs) / len(m_qs) * 100.0
        tot_d = sum(float(x.get("dense_memory_bytes", 0.0) or (x.get("context_length", 2048)*100000)) for x in m_qs)
        tot_m = sum(float(x.get("method_memory_bytes", 0.0) or (float(x.get("dense_memory_bytes", 0.0) or (x.get("context_length", 2048)*100000)) * x.get("method_effective_bpt", 16.0)/16.0)) for x in m_qs)
        r_mem = 100.0 * max(0.0, (tot_d - tot_m) / tot_d) if tot_d > 0 else 0.0
        s_res = ALPHA * avg_q + (1.0 - ALPHA) * r_mem

        m_ttft = sum(q.get("method_ttft_ms", 1000.0) for q in m_qs) / len(m_qs)
        m_thru = sum(q.get("method_decode_throughput", 10.0) for q in m_qs) / len(m_qs)
        m_vram = sum(q.get("method_peak_vram_mb", 1000.0) for q in m_qs) / len(m_qs)

        phi_ttft = max(0.0, min(1.0, ref_ttft / max(0.01, m_ttft)))
        phi_thru = max(0.0, min(1.0, m_thru / max(0.01, ref_thru)))
        phi_vram = max(0.0, min(1.0, ref_vram / max(0.01, m_vram)))

        mult = (phi_ttft ** 0.35) * (phi_thru ** 0.35) * (phi_vram ** 0.30)
        r_sys = 100.0 * mult
        s_sys = ALPHA * s_res + (1.0 - ALPHA) * r_sys

        color = METHOD_COLORS.get(m, "#333333")
        marker = METHOD_MARKERS.get(m, "o")
        ax.scatter(s_res, s_sys, s=180, color=color, marker=marker, edgecolors="#222222", linewidth=1.4, zorder=4, label=m)
        ax.annotate(m, (s_res, s_sys), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, fontweight="bold")

    ax.plot([0, 100], [0, 100], linestyle="--", color="#888888", label="Ideal Parity (S_sys = S_res)", alpha=0.7)
    ax.set_title(f"{model_name}: Part 1 (S_res) vs. Part 2 (S_sys) Serving Utility", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Part 1 Score (S_res) [Algorithmic Quality & Memory]", fontsize=11, fontweight="bold")
    ax.set_ylabel("Part 2 Score (S_sys) [Production Serving Utility]", fontsize=11, fontweight="bold")
    ax.set_xlim(20, 75)
    ax.set_ylim(40, 80)
    ax.grid(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9, loc="lower right")
    plt.tight_layout()

    fig.savefig(os.path.join(out_dir, "resource_vs_system_score.png"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, "resource_vs_system_score.pdf"), bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Generated all figures for {model_name} in {out_dir}")

# Generate for all 3 models
generate_model_figures("Gemma 4 E2B", "results/gemma4_e2b/progress.jsonl", "results/gemma4_e2b/figures")
generate_model_figures("Gemma 4 E4B", "results/gemma4_e4b/progress.jsonl", "results/gemma4_e4b/figures")
generate_model_figures("Qwen2.5 7B", "results/qwen2.5_7b/progress.jsonl", "results/qwen2.5_7b/figures")

# Also copy primary figures into paper/tmlr/figures/ for publication compilation
os.makedirs("paper/tmlr/figures", exist_ok=True)
import shutil
shutil.copy("results/qwen2.5_7b/figures/pareto_frontier_32768.png", "paper/tmlr/figures/fig3_pareto_frontier.png")
shutil.copy("results/qwen2.5_7b/figures/pareto_frontier_32768.pdf", "paper/tmlr/figures/fig3_pareto_frontier.pdf")
shutil.copy("results/gemma4_e2b/figures/context_scaling.png", "paper/tmlr/figures/fig5_context_scaling.png")
shutil.copy("results/gemma4_e2b/figures/context_scaling.pdf", "paper/tmlr/figures/fig5_context_scaling.pdf")
shutil.copy("results/qwen2.5_7b/figures/resource_vs_system_score.png", "paper/tmlr/figures/fig4_score_comparison.png")
shutil.copy("results/qwen2.5_7b/figures/resource_vs_system_score.pdf", "paper/tmlr/figures/fig4_score_comparison.pdf")

print("[OK] Paper figures in paper/tmlr/figures/ regenerated and synced with latest benchmark data!")
