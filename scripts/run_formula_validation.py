"""
CRBench Formula Validation Run — Qwen2.5-1.5B-Instruct
=======================================================
Runs the benchmark on the 1.5B model, then immediately evaluates the
frozen Cobb-Douglas formula (F2, α=0.70) on the real measurements.

Outputs:
  results/qwen15b_formula_validation/
    raw_results_v1.json          – per-sample measurements
    formula_validation.md        – per-method Q, R, and S scores
    formula_validation_plot.png  – scatter plot Q vs R colored by formula S
"""

import sys
import json
import math
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from crbench.core.config import BenchmarkConfig
from crbench.core.runner import BenchmarkRunner
from crbench.scoring.utility import (
    CRBENCH_ALPHA,
    CRBENCH_FORMULA_DESCRIPTION,
    compute_utility,
    resource_efficiency_from_bpt,
)


OUT_DIR = Path("results/qwen15b_formula_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_benchmark():
    print("=" * 72)
    print("CRBench Formula Validation — Qwen2.5-1.5B-Instruct")
    print(f"Formula: {CRBENCH_FORMULA_DESCRIPTION}")
    print("=" * 72)

    cfg = BenchmarkConfig.from_yaml("configs/formula_validation_1.5b.yaml")
    runner = BenchmarkRunner(cfg)

    print("\n[1] Loading model...")
    t0 = time.time()
    runner.load_model()
    load_time = time.time() - t0
    print(f"    Model ready in {load_time:.1f}s")

    print("\n[2] Running benchmark pipeline...")
    t1 = time.time()
    results = runner.run()
    run_time = time.time() - t1
    print(f"    Benchmark complete in {run_time:.1f}s")

    return results


def analyse_formula_on_results(results: dict):
    """
    Read back the raw_results_v1.json from the output directory and compute
    the canonical utility scores + compare against all 7 formula families.
    """
    raw_path = OUT_DIR / "raw_results_v1.json"
    if not raw_path.exists():
        print(f"[!] raw_results_v1.json not found at {raw_path}")
        print("    Using in-memory results instead.")
        return results

    with open(raw_path) as f:
        manifest = json.load(f)

    samples = manifest.get("measurements", [])
    if not samples:
        print("[!] No measurements found in raw_results_v1.json")
        return results

    # Aggregate per-adapter: mean normalized score and mean effective bpt
    from collections import defaultdict
    adapter_samples = defaultdict(list)
    for s in samples:
        if s.get("status") == "SUCCESS":
            adapter_samples[s["adapter_name"]].append(s)

    print("\n[3] Computing formula scores on real measurements...")
    print()
    print(f"  {'Method':<22} {'Q (Norm%)':<12} {'bpt':<8} {'R_eff':<10} {'S (F2,α=0.70)':<16} {'Rank'}")
    print("  " + "-" * 75)

    method_scores = {}
    for adapter_name, slist in adapter_samples.items():
        qs = [s["normalized_score"] for s in slist if "normalized_score" in s]
        bpts = [s.get("effective_bpt", 16.0) for s in slist]
        ttfts = [s.get("ttft_ms", None) for s in slist]

        mean_q = float(np.mean(qs)) if qs else 0.0
        mean_bpt = float(np.mean(bpts)) if bpts else 16.0
        valid_ttfts = [t for t in ttfts if t is not None and t > 0]
        mean_ttft = float(np.mean(valid_ttfts)) if valid_ttfts else None

        # Get dense baseline TTFT for ratio calculation
        dense_samples = adapter_samples.get("dense_fp16", [])
        dense_ttfts = [s.get("ttft_ms") for s in dense_samples if s.get("ttft_ms")]
        dense_mean_ttft = float(np.mean(dense_ttfts)) if dense_ttfts else None

        if mean_ttft and dense_mean_ttft and dense_mean_ttft > 0:
            ttft_ratio = mean_ttft / dense_mean_ttft
        else:
            ttft_ratio = 1.0

        R_eff = resource_efficiency_from_bpt(
            effective_bpt=mean_bpt,
            ttft_ratio=ttft_ratio
        )
        S = compute_utility(mean_q, R_eff)
        method_scores[adapter_name] = {
            "Q": mean_q, "bpt": mean_bpt, "R_eff": R_eff, "S": S,
            "ttft_ratio": ttft_ratio, "n_samples": len(slist)
        }

    # Sort by score
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1]["S"], reverse=True)
    for rank, (name, v) in enumerate(sorted_methods, 1):
        print(f"  {name:<22} {v['Q']:>8.1f}%   {v['bpt']:>5.1f}   {v['R_eff']:>7.1f}   {v['S']:>12.2f}   #{rank}")

    # Compare dense FP16 vs INT2 — key sanity check
    print()
    if "dense_fp16" in method_scores and "kv_quant_int2" in method_scores:
        s_dense = method_scores["dense_fp16"]["S"]
        s_int2 = method_scores["kv_quant_int2"]["S"]
        check = "✅ PASS" if s_dense > s_int2 else "❌ FAIL"
        print(f"  Sanity check — dense_fp16 ({s_dense:.2f}) > kv_quant_int2 ({s_int2:.2f}): {check}")

    if "kv_quant_int4" in method_scores and "dense_fp16" in method_scores:
        s_int4 = method_scores["kv_quant_int4"]["S"]
        s_dense = method_scores["dense_fp16"]["S"]
        check = "✅ as expected" if s_int4 > s_dense else "⚠️  dense dominates (unusual — check Q values)"
        print(f"  INT4 ({s_int4:.2f}) vs dense ({s_dense:.2f}): {check}")

    return method_scores


def plot_formula_validation(method_scores: dict, out_dir: Path):
    """Q vs R scatter plot, colored by formula score S."""
    if not method_scores or isinstance(method_scores, dict) and "resource_results" in method_scores:
        print("[!] Cannot generate plot: method_scores is empty or in wrong format")
        return

    names = list(method_scores.keys())
    Q_vals = [method_scores[n]["Q"] for n in names]
    R_vals = [method_scores[n]["R_eff"] for n in names]
    S_vals = [method_scores[n]["S"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#1a1a2e")
    ax1, ax2 = axes

    # ── Left: Q vs R scatter colored by S ────────────────────────────────────
    ax1.set_facecolor("#16213e")
    norm = plt.Normalize(vmin=min(S_vals) - 5, vmax=max(S_vals) + 5)
    sc = ax1.scatter(R_vals, Q_vals, c=S_vals, cmap="plasma", norm=norm,
                     s=180, edgecolors="white", linewidths=0.8, zorder=3)
    for name, Q, R, S in zip(names, Q_vals, R_vals, S_vals):
        ax1.annotate(
            f"{name}\nS={S:.1f}",
            xy=(R, Q), xytext=(5, 5), textcoords="offset points",
            fontsize=7.5, color="white", alpha=0.9
        )
    cb = fig.colorbar(sc, ax=ax1)
    cb.set_label("S = Q^0.70 · R^0.30", color="white", fontsize=9)
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="white")
    ax1.set_xlabel("Resource Efficiency R", fontsize=11, color="white")
    ax1.set_ylabel("Quality Retention Q (%)", fontsize=11, color="white")
    ax1.set_title("Formula Validation: Q vs R → S\nQwen2.5-1.5B-Instruct (Real Measurements)",
                  fontsize=10, fontweight="bold", color="white")
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values(): spine.set_edgecolor("#444")
    ax1.grid(True, alpha=0.2, color="white")

    # ── Right: method ranking bar chart ──────────────────────────────────────
    ax2.set_facecolor("#16213e")
    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1]["S"])
    bar_names = [m for m, _ in sorted_methods]
    bar_S = [v["S"] for _, v in sorted_methods]
    bar_Q = [v["Q"] for _, v in sorted_methods]

    palette = plt.cm.plasma(np.linspace(0.3, 0.9, len(bar_names)))
    bars = ax2.barh(bar_names, bar_S, color=palette, edgecolor="#333", alpha=0.88, height=0.6)
    for bar, S, Q in zip(bars, bar_S, bar_Q):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                 f"S={S:.1f}  Q={Q:.0f}%", va="center", fontsize=8, color="white")
    ax2.set_xlabel("Utility Score S = Q^0.70 · R^0.30", fontsize=10, color="white")
    ax2.set_title("Method Ranking (Cobb-Douglas, α=0.70)\nFrozen Formula on Real 1.5B Data",
                  fontsize=10, fontweight="bold", color="white")
    ax2.tick_params(colors="white", labelsize=9)
    for spine in ax2.spines.values(): spine.set_edgecolor("#444")
    ax2.grid(True, axis="x", alpha=0.2, color="white")

    plt.suptitle("CRBench Frozen Formula Validation — Qwen2.5-1.5B-Instruct",
                 fontsize=12, fontweight="bold", color="white")
    plt.tight_layout()
    out_path = out_dir / "formula_validation_plot.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path}")
    return out_path


def write_markdown_report(method_scores: dict, out_dir: Path):
    """Write a provenance-labelled Markdown report of the formula validation."""
    if not method_scores or "resource_results" in method_scores:
        return

    sorted_methods = sorted(method_scores.items(), key=lambda x: x[1]["S"], reverse=True)

    lines = [
        "# CRBench Formula Validation — Qwen2.5-1.5B-Instruct",
        "",
        f"**Formula:** `{CRBENCH_FORMULA_DESCRIPTION}`  ",
        f"**α:** `{CRBENCH_ALPHA}` (frozen)  ",
        f"**Provenance:** All Q values are real measured inference results. "
        "R_eff is derived analytically from measured effective bpt and TTFT.",
        "",
        "## Per-Method Scores",
        "",
        "| Rank | Method | Q (%) | bpt | R_eff | S = Q^0.70·R^0.30 | Samples |",
        "| :---: | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for rank, (name, v) in enumerate(sorted_methods, 1):
        lines.append(
            f"| {rank} | {name} | {v['Q']:.1f} | {v['bpt']:.1f} | {v['R_eff']:.1f} | **{v['S']:.2f}** | {v['n_samples']} |"
        )

    # Sanity checks
    checks = []
    if "dense_fp16" in method_scores and "kv_quant_int2" in method_scores:
        s_dense = method_scores["dense_fp16"]["S"]
        s_int2 = method_scores["kv_quant_int2"]["S"]
        icon = "✅" if s_dense > s_int2 else "❌"
        checks.append(f"- {icon} Dense FP16 ({s_dense:.2f}) > INT2 ({s_int2:.2f}) — quality gatekeeping")

    if "kv_quant_int4" in method_scores and "dense_fp16" in method_scores:
        s_int4 = method_scores["kv_quant_int4"]["S"]
        s_dense = method_scores["dense_fp16"]["S"]
        icon = "✅" if s_int4 > s_dense else "⚠️"
        checks.append(f"- {icon} INT4 ({s_int4:.2f}) vs Dense ({s_dense:.2f}) — resource reward")

    # Q spread check
    q_vals = [v["Q"] for v in method_scores.values()]
    q_spread = max(q_vals) - min(q_vals)
    icon = "✅" if q_spread > 10 else "⚠️"
    checks.append(f"- {icon} Q spread: {q_spread:.1f}% (target >10% for meaningful discrimination)")

    # All scores distinct
    s_vals = sorted(set(round(v["S"], 2) for v in method_scores.values()))
    icon = "✅" if len(s_vals) == len(method_scores) else "⚠️"
    checks.append(f"- {icon} All method scores distinct: {len(s_vals)}/{len(method_scores)}")

    lines += [
        "",
        "## Formula Sanity Checks",
        "",
    ] + checks + [
        "",
        "## Data Provenance",
        "",
        "| Metric | Source |",
        "| :--- | :--- |",
        "| Q (quality retention) | **MEASURED** — real inference on Qwen2.5-1.5B-Instruct |",
        "| effective_bpt | **MEASURED** — from KVStateMetadata.algorithmic_bytes |",
        "| R_eff | **DERIVED** — from measured bpt + TTFT via resource_efficiency_from_bpt() |",
        "| S (utility score) | **DERIVED** — computed from measured Q and derived R_eff |",
        "| TTFT | **MEASURED** — from LatencyProfiler wall-clock timing |",
    ]

    report_path = out_dir / "formula_validation.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Report saved: {report_path}")
    return report_path


def main():
    # 1. Run the benchmark
    results = run_benchmark()

    # 2. Analyse formula on real results
    print("\n" + "=" * 72)
    print("FORMULA VALIDATION ON REAL MEASUREMENTS")
    print("=" * 72)
    method_scores = analyse_formula_on_results(results)

    # 3. Generate plot and report
    if isinstance(method_scores, dict) and method_scores and "resource_results" not in method_scores:
        plot_formula_validation(method_scores, OUT_DIR)
        write_markdown_report(method_scores, OUT_DIR)
    else:
        print("[!] Could not generate plots — method_scores in unexpected format.")
        print("    Check raw_results_v1.json in the output directory.")

    print("\n" + "=" * 72)
    print("DONE. Results in:", OUT_DIR)
    print("=" * 72)


if __name__ == "__main__":
    main()
