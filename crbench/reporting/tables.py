"""
Markdown and LaTeX table formatting utilities for CRBench results.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from tabulate import tabulate
from crbench.scoring.resource_score import CRBenchResourceScoreResult
from crbench.scoring.system_score import CRBenchSystemScoreResult


def format_resource_scores_table(
    results: List[CRBenchResourceScoreResult],
    format_type: str = "markdown"  # "markdown", "latex", "grid"
) -> str:
    """
    Generates summary table of Part 1 Resource Scores across context lengths and standard budgets.
    """
    # Collect all unique context lengths across all results
    all_lengths = sorted(list({L for res in results for L in res.context_scores.keys()}))
    
    headers = ["Method", "S_res (0-100)"]
    for L in all_lengths:
        headers.append(f"AUQC ({L//1024 if L>=1024 else L}{'K' if L>=1024 else ''})")
    headers.extend(["Q@2bpt", "Q@4bpt", "Q@8bpt", "Q@16bpt"])

    rows = []
    for res in sorted(results, key=lambda x: x.resource_score, reverse=True):
        row = [res.method_name, f"{res.resource_score:.1f}"]
        
        # AUQCs for evaluated context lengths
        for L in all_lengths:
            if L in res.context_scores:
                row.append(f"{res.context_scores[L].auqc_result.auqc_score:.1f}")
            else:
                row.append("-")

        # Iso-budgets at representative length (longest evaluated)
        lengths = sorted(res.context_scores.keys())
        if lengths:
            target_L = lengths[-1]
            iso = res.context_scores[target_L].isobudget_result
            for b in [2.0, 4.0, 8.0, 16.0]:
                val = iso.budget_scores.get(b, 0.0)
                row.append(f"{val:.1f}%")
        else:
            row.extend(["-", "-", "-", "-"])

        rows.append(row)

    if format_type == "latex":
        return tabulate(rows, headers=headers, tablefmt="latex_booktabs")
    else:
        return tabulate(rows, headers=headers, tablefmt="github")


def format_system_scores_table(
    results: List[CRBenchSystemScoreResult],
    format_type: str = "markdown"
) -> str:
    """
    Generates summary table of Part 2 System Scores with runtime metrics.
    """
    headers = ["Method", "S_sys (0-100)", "S_res (Part 1)", "TTFT (ms)", "Decode Thru (tok/s)", "Peak VRAM (MB)", "Multiplier"]
    rows = []

    for res in sorted(results, key=lambda x: x.system_score, reverse=True):
        m = res.runtime_metrics
        row = [
            res.method_name,
            f"{res.system_score:.1f}",
            f"{res.resource_score:.1f}",
            f"{m.mean_ttft_ms:.1f}",
            f"{m.mean_decode_throughput_tok_per_sec:.1f}",
            f"{m.peak_vram_mb:.1f}",
            f"{res.system_utility_multiplier:.2f}x"
        ]
        rows.append(row)

    if format_type == "latex":
        return tabulate(rows, headers=headers, tablefmt="latex_booktabs")
    else:
        return tabulate(rows, headers=headers, tablefmt="github")


def format_provenance_audit_table(format_type: str = "markdown") -> str:
    """
    Generates explicit Scientific Provenance Audit Table for all CRBench metrics.
    """
    headers = ["Metric Name", "Classification", "Measurement / Mathematical Instrument", "Scientific Status"]
    rows = [
        ["Raw Task Quality (Q_m)", "MEASURED", "Autoregressive generation on model with PyTorch forward hooks", "Real Measurement"],
        ["Dense Reference (Q_dense)", "MEASURED", "Dense uncompressed FP16 baseline on identical sample set", "Real Measurement"],
        ["Model-Relative Retention (q_rel)", "DERIVED", "(Q_m - Q_floor) / max(0.05, Q_dense - Q_floor) * 100", "Mathematically Derived"],
        ["Effective Bits/Token (b_eff)", "DERIVED (ANALYTICAL)", "8 * (Algorithmic_Bytes + Scale_Metadata_Bytes) / L", "Exact Architecture Formula"],
        ["AUQC Score", "DERIVED", "Monotonic PCHIP logarithmic spline integral: 1/ln(8) * ∫ Q(e^u) du", "Numerically Integrated"],
        ["Iso-Budget Quality (Q@B)", "DERIVED", "Standardized evaluation at 2, 4, 8, 16 bpt along empirical spline", "Interpolated / Extrapolated"],
        ["Part 1 Resource Score (S_res)", "DERIVED", "Logarithmically context-weighted sum: ∑ w_L AUQC(L)", "Axiomatically Derived"],
        ["TTFT Prefill Latency", "MEASURED", "Synchronized wall-clock prefill duration (ms)", "Real Measurement"],
        ["Decode Throughput", "MEASURED", "Tokens generated per second during autoregressive decode", "Real Measurement"],
        ["Peak VRAM Footprint", "MEASURED / DERIVED", "Effective bpt-scaled buffer + physical device allocator tracking", "Measured & Scaled"],
        ["Part 2 System Score (S_sys)", "DERIVED", "S_res * (phi_ttft * phi_thru * phi_vram)^alpha", "Constrained Utility Function"]
    ]

    if format_type == "latex":
        return tabulate(rows, headers=headers, tablefmt="latex_booktabs")
    else:
        return tabulate(rows, headers=headers, tablefmt="github")
