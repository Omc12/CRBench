"""
CRBench Canonical Utility Formulation & Resource Metrics Engine
===============================================================
Provides principled utility formulations for combining contextual quality retention (Q)
and resource efficiency (R) into single scalar scores for Part 1 (Resource) and Part 2 (System).

RECOMMENDED FORMULATION (from empirical & mathematical re-evaluation):
  Linear Additive Utility:
      S(Q, R) = α · Q + (1 - α) · R
  where:
      Q ∈ [0.0, 100.0]  (Normalized capability retention relative to dense baseline on identical query)
      R ∈ [0.0, 100.0]  (Normalized memory resource savings relative to dense baseline)
      α ∈ [0.10, 0.90]  (Global quality weight; default α = 0.70)

Axiomatic Properties of Linear Formulation:
  - Quality Monotonicity:    ∂S/∂Q = α > 0  (Strictly monotone)
  - Resource Monotonicity:   ∂S/∂R = 1 - α > 0  (Strictly monotone)
  - Boundedness:             For Q, R ∈ [0, 100], S ∈ [0, 100] exactly
  - Dense Reference Score:   For Dense FP16 (Q=100, R=0), S_dense = α · 100.0 (Preserved non-zero!)
  - Pareto Consistency:      100% consistent across all pairwise dominant methods
  - Low-Quality Rejection:   Methods with failing quality (Q < 10%) are penalized below Dense baseline

Candidate formulas supported for comparative evaluation:
  - "linear":              S = αQ + (1-α)R
  - "cobb_douglas":        S = Q^α · R^(1-α)
  - "harmonic":            S = (α/Q + (1-α)/R)^(-1)
  - "power_mean_05":       S = (α·Q^0.5 + (1-α)·R^0.5)^2
  - "power_mean_2":        S = (α·Q^2 + (1-α)·R^2)^0.5
  - "logarithmic":         S = 100 · (α ln(1+Q) + (1-α) ln(1+R)) / ln(101)
  - "gated_linear":        S = (Q/100) · (αQ + (1-α)R)
"""

from __future__ import annotations
import math
from typing import Any, Dict, Optional, Union


# ──────────────────────────────────────────────────────────────────────────────
# Global Default Parameters (Configurable via BenchmarkConfig / CLI)
# ──────────────────────────────────────────────────────────────────────────────

CRBENCH_ALPHA: float = 0.70
"""Default quality weight in the utility function (0.70 = quality-dominant)."""

CRBENCH_FORMULA_NAME: str = "linear"
"""Default canonical utility formula."""

CRBENCH_FORMULA_DESCRIPTION: str = (
    "S(Q, R) = α·Q + (1-α)·R  [Linear Additive Utility, default α=0.70, R ∈ [0, 100]]"
)


# ──────────────────────────────────────────────────────────────────────────────
# Formula Implementations
# ──────────────────────────────────────────────────────────────────────────────

def formula_linear(Q: float, R: float, alpha: float = CRBENCH_ALPHA) -> float:
    """Linear Additive: S = αQ + (1-α)R"""
    q_clamped = max(0.0, min(100.0, float(Q)))
    r_clamped = max(0.0, min(100.0, float(R)))
    return alpha * q_clamped + (1.0 - alpha) * r_clamped


def formula_cobb_douglas(Q: float, R: float, alpha: float = CRBENCH_ALPHA) -> float:
    """Cobb-Douglas Geometric: S = Q^α · R^(1-α)"""
    if Q <= 0.0 or R <= 0.0:
        return 0.0
    q_clamped = max(0.0, min(100.0, float(Q)))
    r_clamped = max(0.0, min(100.0, float(R)))
    return (q_clamped ** alpha) * (r_clamped ** (1.0 - alpha))


def formula_harmonic(Q: float, R: float, alpha: float = CRBENCH_ALPHA) -> float:
    """Harmonic Mean: S = (α/Q + (1-α)/R)^(-1)"""
    if Q <= 0.0 or R <= 0.0:
        return 0.0
    denom = (alpha / max(1e-6, float(Q))) + ((1.0 - alpha) / max(1e-6, float(R)))
    return min(100.0, 1.0 / denom) if denom > 0 else 0.0


def formula_power_mean(Q: float, R: float, alpha: float = CRBENCH_ALPHA, p: float = 2.0) -> float:
    """Power Mean: S = (α·Q^p + (1-α)·R^p)^(1/p)"""
    if p == 0.0:
        return formula_cobb_douglas(Q, R, alpha)
    q_clamped = max(0.0, min(100.0, float(Q)))
    r_clamped = max(0.0, min(100.0, float(R)))
    val = alpha * (q_clamped ** p) + (1.0 - alpha) * (r_clamped ** p)
    return min(100.0, max(0.0, val ** (1.0 / p)))


def formula_logarithmic(Q: float, R: float, alpha: float = CRBENCH_ALPHA) -> float:
    """Log-Utility: S = 100 · (α ln(1+Q) + (1-α) ln(1+R)) / ln(101)"""
    q_clamped = max(0.0, min(100.0, float(Q)))
    r_clamped = max(0.0, min(100.0, float(R)))
    norm = math.log(101.0)
    return 100.0 * (alpha * math.log(1.0 + q_clamped) + (1.0 - alpha) * math.log(1.0 + r_clamped)) / norm


def formula_gated_linear(Q: float, R: float, alpha: float = CRBENCH_ALPHA) -> float:
    """Quality-Gated Linear: S = (Q/100) · (αQ + (1-α)R)"""
    q_clamped = max(0.0, min(100.0, float(Q)))
    r_clamped = max(0.0, min(100.0, float(R)))
    gate = q_clamped / 100.0
    return gate * (alpha * q_clamped + (1.0 - alpha) * r_clamped)


FORMULA_DISPATCH: Dict[str, Any] = {
    "linear": formula_linear,
    "cobb_douglas": formula_cobb_douglas,
    "geometric": formula_cobb_douglas,
    "harmonic": formula_harmonic,
    "power_mean_05": lambda q, r, a: formula_power_mean(q, r, a, p=0.5),
    "power_mean_2": lambda q, r, a: formula_power_mean(q, r, a, p=2.0),
    "logarithmic": formula_logarithmic,
    "gated_linear": formula_gated_linear,
}


# ──────────────────────────────────────────────────────────────────────────────
# Public Scoring API
# ──────────────────────────────────────────────────────────────────────────────

def compute_utility(
    quality_score: float,
    resource_efficiency: float,
    alpha: float = CRBENCH_ALPHA,
    formula: str = "linear",
) -> float:
    """
    Computes scalar CRBench utility score combining Quality Q and Resource R.

    Parameters
    ----------
    quality_score : float
        Normalized capability retention Q ∈ [0.0, 100.0] relative to Dense baseline.
    resource_efficiency : float
        Normalized resource efficiency R ∈ [0.0, 100.0].
    alpha : float
        Quality weight α ∈ [0.0, 1.0]. Defaults to global CRBENCH_ALPHA (0.70).
    formula : str
        Formula identifier ("linear", "cobb_douglas", "harmonic", etc.).

    Returns
    -------
    float
        Utility score in [0.0, 100.0].
    """
    fn = FORMULA_DISPATCH.get(formula.lower(), formula_linear)
    score = fn(quality_score, resource_efficiency, alpha)
    return float(max(0.0, min(100.0, score)))


def compute_utility_decomposition(
    quality_score: float,
    resource_efficiency: float,
    alpha: float = CRBENCH_ALPHA,
    formula: str = "linear",
) -> Dict[str, Any]:
    """
    Computes utility with full mathematical decomposition for reporting.
    """
    u = compute_utility(quality_score, resource_efficiency, alpha, formula)
    q = max(0.0, min(100.0, float(quality_score)))
    r = max(0.0, min(100.0, float(resource_efficiency)))

    return {
        "utility": u,
        "quality_score": q,
        "resource_efficiency": r,
        "alpha": alpha,
        "formula": formula,
        "quality_term": alpha * q if formula == "linear" else (q ** alpha),
        "resource_term": (1.0 - alpha) * r if formula == "linear" else (r ** (1.0 - alpha)),
        "provenance": "measured_query_pair",
    }


def compute_query_resource_efficiency(
    dense_bytes: float,
    method_bytes: float,
    dense_bpt: float = 16.0,
    method_bpt: float = 16.0,
) -> float:
    """
    Computes Part 1 memory resource efficiency R ∈ [0.0, 100.0] for a single query.

    Mathematical Definition:
        R = 100.0 · max(0.0, 1.0 - method_bytes / dense_bytes)

    If bytes are unavailable, falls back to effective bits per token (bpt):
        R = 100.0 · max(0.0, 1.0 - method_bpt / dense_bpt)
    """
    if dense_bytes > 0 and method_bytes > 0:
        savings = 1.0 - (method_bytes / dense_bytes)
    elif dense_bpt > 0:
        savings = 1.0 - (method_bpt / dense_bpt)
    else:
        savings = 0.0

    return float(max(0.0, min(100.0, savings * 100.0)))


def resource_efficiency_from_bpt(
    effective_bpt: float,
    baseline_bpt: float = 16.0,
    ttft_ratio: float = 1.0,
    w_memory: float = 0.60,
    w_latency: float = 0.40,
) -> float:
    """
    Helper converting effective bits/token + TTFT ratio into resource efficiency R ∈ [0, 100].
    """
    if effective_bpt <= 0:
        effective_bpt = baseline_bpt
    memory_eff = 100.0 * max(0.0, 1.0 - effective_bpt / baseline_bpt)
    latency_eff = 100.0 * max(0.0, min(1.0, 1.0 - (ttft_ratio - 1.0)))
    return w_memory * memory_eff + w_latency * latency_eff


def compute_query_system_efficiency(
    dense_bytes: float,
    method_bytes: float,
    dense_ttft_ms: Optional[float] = None,
    method_ttft_ms: Optional[float] = None,
    dense_throughput: Optional[float] = None,
    method_throughput: Optional[float] = None,
    w_memory: float = 0.50,
    w_ttft: float = 0.25,
    w_thru: float = 0.25,
) -> float:
    """
    Computes Part 2 system runtime efficiency R_sys ∈ [0.0, 100.0] for a single query.
    Combines memory savings, TTFT prefill latency ratio, and decode throughput.
    """
    # 1. Memory component
    r_mem = compute_query_resource_efficiency(dense_bytes, method_bytes)

    # 2. TTFT prefill speedup component (1.0 = baseline, >1.0 = faster)
    if dense_ttft_ms and method_ttft_ms and dense_ttft_ms > 0 and method_ttft_ms > 0:
        ttft_speedup = dense_ttft_ms / method_ttft_ms
        r_ttft = max(0.0, min(100.0, 50.0 * ttft_speedup))
    else:
        r_ttft = 50.0

    # 3. Decode throughput component
    if dense_throughput and method_throughput and dense_throughput > 0 and method_throughput > 0:
        thru_ratio = method_throughput / dense_throughput
        r_thru = max(0.0, min(100.0, 50.0 * thru_ratio))
    else:
        r_thru = 50.0

    # Weighted combination
    r_sys = w_memory * r_mem + w_ttft * r_ttft + w_thru * r_thru
    return float(max(0.0, min(100.0, r_sys)))
