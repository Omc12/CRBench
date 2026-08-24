"""
CRBench Canonical Utility Formula
===================================
Formula frozen after systematic empirical evaluation over 7 candidate families
and a full α-sweep [0.10, 0.90] with 13 discrete values.

FROZEN SELECTION (do not change without re-running the axiom sweep):
  Formula : F2 Cobb-Douglas Geometric Mean
            S(Q, R) = Q^α · R^(1-α)
  α       : 0.70   (quality-dominant; resource-aware but not resource-overriding)

RATIONALE (see results/formula_analysis/formula_analysis_report.md):
  - 100% Pareto-dominance consistency at α=0.70 (zero ordering violations)
  - Satisfies all monotonicity axioms (Q-monotone and R-monotone)
  - Bounded in [0, 100] for Q, R ∈ [0, 100]
  - Dense FP16 correctly scored above INT2 (Q=55 baseline)
  - Low-quality methods NOT incorrectly rewarded by resource efficiency alone
  - Composite axiom objective: 0.8071 (F3 Harmonic: 0.8297 but less interpretable)
  - F2 has direct economic interpretation (Cobb-Douglas production function)
  - F3 Harmonic is marginally higher on composite score (0.8297 vs 0.8071) due to
    higher rank_stability, but F2 is preferred because:
      (a) Cobb-Douglas is a standard scientifically motivated utility function
      (b) F2 is more commonly interpretable and reviewable
      (c) The 0.5B data is insufficient to definitively prefer F3 over F2
  - F5 (current multiplicative) is REJECTED: phi_raw ∈ [0.88, 1.0] for all methods,
    making α effectively a no-op — rankings are identical at α=0.1 and α=0.9.

This module is the single authoritative source for the benchmark utility formula.
All scoring engines (resource_score.py, system_score.py) MUST import from here.
"""
from __future__ import annotations
import math
from typing import Union


# ──────────────────────────────────────────────────────────────────────────────
# FROZEN PARAMETERS — do not modify without re-running analyze_utility_formulas.py
# ──────────────────────────────────────────────────────────────────────────────

CRBENCH_ALPHA: float = 0.70
"""
Quality weight in the Cobb-Douglas utility function.
Frozen at 0.70 based on systematic axiom compliance sweep.
A higher value (0.7 vs 0.5) reflects that contextual quality retention
is more critical than resource efficiency for general-purpose LLM evaluation.
"""

CRBENCH_FORMULA_NAME: str = "F2_cobb_douglas_geometric"
"""Machine-readable identifier for the frozen formula."""

CRBENCH_FORMULA_DESCRIPTION: str = (
    "S(Q, R) = Q^α · R^(1-α)  [Cobb-Douglas Geometric Mean, α=0.70]"
)
"""Human-readable description for reports and metadata."""


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def compute_utility(
    quality_score: float,
    resource_efficiency: float,
    alpha: float = CRBENCH_ALPHA,
) -> float:
    """
    Compute the CRBench utility score using the frozen Cobb-Douglas formula.

    Parameters
    ----------
    quality_score : float
        Normalised quality retention in [0.0, 100.0].
        100.0 = same quality as the dense baseline on this model/task.
    resource_efficiency : float
        Normalised resource efficiency in [0.0, 100.0].
        100.0 = maximum compression with no latency overhead.
    alpha : float, optional
        Quality weight. Defaults to the frozen global ``CRBENCH_ALPHA = 0.70``.
        Override ONLY for sensitivity analysis; never change in production runs.

    Returns
    -------
    float
        Utility score in [0.0, 100.0].

    Notes
    -----
    The Cobb-Douglas form ensures:
    * Quality-monotone: ∂S/∂Q > 0  for Q, R > 0
    * Resource-monotone: ∂S/∂R > 0 for Q, R > 0
    * S = 0 if either Q = 0 or R = 0 (strict Leontief penalty for zero components)
    * Pareto consistency: if A dominates B (A.Q ≥ B.Q and A.R ≥ B.R), then S(A) ≥ S(B)
    """
    if quality_score <= 0.0 or resource_efficiency <= 0.0:
        return 0.0
    q = float(quality_score)
    r = float(resource_efficiency)
    return (q ** alpha) * (r ** (1.0 - alpha))


def compute_utility_decomposition(
    quality_score: float,
    resource_efficiency: float,
    alpha: float = CRBENCH_ALPHA,
) -> dict:
    """
    Compute utility with a detailed decomposition for reporting.

    Returns
    -------
    dict with keys:
        utility, quality_component, resource_component, alpha, formula
    """
    u = compute_utility(quality_score, resource_efficiency, alpha)
    q = max(0.0, quality_score)
    r = max(0.0, resource_efficiency)
    return {
        "utility": u,
        "quality_component": q ** alpha if q > 0 else 0.0,
        "resource_component": r ** (1.0 - alpha) if r > 0 else 0.0,
        "alpha": alpha,
        "formula": CRBENCH_FORMULA_DESCRIPTION,
        "formula_id": CRBENCH_FORMULA_NAME,
        "provenance": "frozen_empirical_selection_2026_08_24",
    }


def resource_efficiency_from_bpt(
    effective_bpt: float,
    baseline_bpt: float = 16.0,
    ttft_ratio: float = 1.0,
    w_memory: float = 0.60,
    w_latency: float = 0.40,
) -> float:
    """
    Convert effective bits/token + TTFT ratio into a normalised resource efficiency R ∈ [0, 100].

    R = w_mem × (1 - bpt/baseline_bpt) × 100
      + w_lat × max(0, 1 - (ttft - ref_ttft)/ref_ttft) × 100

    Parameters
    ----------
    effective_bpt : float
        Effective bits per KV token for this method.
    baseline_bpt : float
        FP16 baseline (16.0 bits/token by default).
    ttft_ratio : float
        method_ttft / dense_baseline_ttft.  1.0 = same speed, <1 = faster.
    w_memory : float
        Weight on memory compression component (default 0.60).
    w_latency : float
        Weight on TTFT latency component (default 0.40).
    """
    if effective_bpt <= 0:
        effective_bpt = baseline_bpt
    memory_eff = 100.0 * max(0.0, 1.0 - effective_bpt / baseline_bpt)
    latency_eff = 100.0 * max(0.0, min(1.0, 1.0 - (ttft_ratio - 1.0)))
    return w_memory * memory_eff + w_latency * latency_eff
