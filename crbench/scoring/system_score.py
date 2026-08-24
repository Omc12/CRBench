"""
Part 2 — CRBench System Score (S_sys) Engine.
Extends resource score with runtime metrics (TTFT, throughput, decode latency, peak VRAM)
via the frozen canonical Cobb-Douglas utility formula.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from crbench.scoring.resource_score import CRBenchResourceScoreResult
from crbench.scoring.utility import (
    CRBENCH_ALPHA,
    CRBENCH_FORMULA_DESCRIPTION,
    CRBENCH_FORMULA_NAME,
    compute_utility,
    compute_utility_decomposition,
    resource_efficiency_from_bpt,
)


@dataclass
class SystemRuntimeMetrics:
    """Aggregated runtime efficiency metrics."""
    mean_ttft_ms: float
    mean_prefill_throughput_tok_per_sec: float
    mean_decode_latency_ms_per_tok: float
    mean_decode_throughput_tok_per_sec: float
    peak_vram_mb: float
    latency_jitter_ms: float = 0.0


@dataclass
class CRBenchSystemScoreResult:
    """Comprehensive Part 2 Benchmark System Score result."""
    method_name: str
    system_score: float                  # S_sys in [0.0, 100.0]
    resource_score: float                # Baseline S_res from Part 1
    system_utility_multiplier: float     # R factor used in Cobb-Douglas utility
    ttft_efficiency_factor: float
    throughput_efficiency_factor: float
    vram_efficiency_factor: float
    runtime_metrics: SystemRuntimeMetrics
    breakdown: Dict[str, Any] = field(default_factory=dict)
    # Canonical formula provenance
    utility_formula: str = field(default=CRBENCH_FORMULA_DESCRIPTION, init=False, repr=False)
    utility_alpha: float = field(default=CRBENCH_ALPHA, init=False, repr=False)


class CRBenchSystemScorer:
    """
    Scoring Engine for Part 2 (CRBench System Score).
    Evaluates real-world deployment viability under hardware and latency constraints.

    The system score is computed using the frozen Cobb-Douglas utility formula:
        S_sys = Q^α · R_sys^(1-α)
    where:
        Q      = Part 1 resource score (quality-over-budget efficiency)
        R_sys  = normalised runtime efficiency ∈ [0, 100]
                 derived from TTFT ratio, decode throughput ratio, and VRAM ratio
        α      = CRBENCH_ALPHA = 0.70  (global frozen quality weight)
    """

    def __init__(
        self,
        reference_ttft_ms: float = 500.0,
        reference_decode_throughput_tok_sec: float = 30.0,
        vram_budget_mb: float = 16384.0,  # 16 GB default
    ):
        self.ref_ttft = reference_ttft_ms
        self.ref_decode_thru = reference_decode_throughput_tok_sec
        self.vram_budget = vram_budget_mb

    def _compute_runtime_efficiency(self,
                                    runtime_metrics: SystemRuntimeMetrics,
                                    ref_ttft: float,
                                    ref_thru: float,
                                    v_budget: float) -> tuple:
        """
        Convert raw runtime metrics to per-factor efficiency scores and a composite
        runtime efficiency R_sys ∈ [0, 100].

        Returns
        -------
        (R_sys, phi_ttft, phi_thru, phi_vram)
        """
        # 1. TTFT efficiency [0, 1]: faster prefill → higher score
        ttft_ratio = runtime_metrics.mean_ttft_ms / max(1.0, ref_ttft)
        phi_ttft = max(0.0, min(1.0, 1.0 / max(0.01, ttft_ratio)))

        # 2. Throughput efficiency [0, 1]: faster decoding → higher score
        thru_ratio = runtime_metrics.mean_decode_throughput_tok_per_sec / max(1.0, ref_thru)
        phi_thru = max(0.0, min(1.0, thru_ratio))

        # 3. VRAM efficiency [0, 1]: smaller footprint → higher score
        vram_ratio = runtime_metrics.peak_vram_mb / max(1.0, v_budget)
        phi_vram = max(0.0, min(1.0, 1.0 / max(0.01, vram_ratio)))

        # Composite runtime efficiency: weighted geometric mean of the three factors
        # Weights: TTFT 0.35, Throughput 0.35, VRAM 0.30
        R_sys = 100.0 * (phi_ttft ** 0.35) * (phi_thru ** 0.35) * (phi_vram ** 0.30)
        return R_sys, phi_ttft, phi_thru, phi_vram

    def score_system(
        self,
        part1_result: CRBenchResourceScoreResult,
        runtime_metrics: SystemRuntimeMetrics,
        target_vram_mb: Optional[float] = None,
        reference_ttft_ms: Optional[float] = None,
        reference_decode_throughput_tok_sec: Optional[float] = None
    ) -> CRBenchSystemScoreResult:
        """
        Computes the Part 2 system score using the canonical Cobb-Douglas formula.

        S_sys = Q^α · R_sys^(1-α)
        where Q = part1_result.resource_score and R_sys is derived from runtime metrics.
        α = CRBENCH_ALPHA = 0.70  (frozen globally).
        """
        v_budget = target_vram_mb if target_vram_mb is not None else self.vram_budget
        ref_ttft = reference_ttft_ms if reference_ttft_ms is not None else self.ref_ttft
        ref_thru = (reference_decode_throughput_tok_sec
                    if reference_decode_throughput_tok_sec is not None else self.ref_decode_thru)

        R_sys, phi_ttft, phi_thru, phi_vram = self._compute_runtime_efficiency(
            runtime_metrics, ref_ttft, ref_thru, v_budget
        )

        # Apply canonical utility formula  S = Q^α · R^(1-α)
        s_res = part1_result.resource_score
        decomp = compute_utility_decomposition(
            quality_score=s_res,
            resource_efficiency=R_sys,
        )
        s_sys = max(0.0, min(100.0, decomp["utility"]))

        return CRBenchSystemScoreResult(
            method_name=part1_result.method_name,
            system_score=float(s_sys),
            resource_score=float(s_res),
            system_utility_multiplier=float(R_sys / 100.0),   # normalised R factor
            ttft_efficiency_factor=float(phi_ttft),
            throughput_efficiency_factor=float(phi_thru),
            vram_efficiency_factor=float(phi_vram),
            runtime_metrics=runtime_metrics,
            breakdown={
                "R_sys": R_sys,
                "Q": s_res,
                "alpha": CRBENCH_ALPHA,
                "formula": CRBENCH_FORMULA_DESCRIPTION,
                "phi_ttft": phi_ttft,
                "phi_thru": phi_thru,
                "phi_vram": phi_vram,
                "quality_component": decomp["quality_component"],
                "resource_component": decomp["resource_component"],
                "provenance": "measured_system_metrics",
            }
        )
