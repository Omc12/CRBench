"""
Part 2 — CRBench System Score (S_sys) Engine.
Extends resource score with runtime metrics (TTFT, throughput, decode latency, peak VRAM)
via a principled constrained utility formulation.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np
from crbench.scoring.resource_score import CRBenchResourceScoreResult


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
    system_utility_multiplier: float     # Multiplicative efficiency factor
    ttft_efficiency_factor: float
    throughput_efficiency_factor: float
    vram_efficiency_factor: float
    runtime_metrics: SystemRuntimeMetrics
    breakdown: Dict[str, Any] = field(default_factory=dict)


class CRBenchSystemScorer:
    """
    Scoring Engine for Part 2 (CRBench System Score).
    Evaluates real-world deployment viability under hardware and latency constraints.
    """

    def __init__(
        self,
        reference_ttft_ms: float = 500.0,
        reference_decode_throughput_tok_sec: float = 30.0,
        vram_budget_mb: float = 16384.0,  # 16 GB default
        utility_exponent: float = 0.5
    ):
        self.ref_ttft = reference_ttft_ms
        self.ref_decode_thru = reference_decode_throughput_tok_sec
        self.vram_budget = vram_budget_mb
        self.alpha = utility_exponent

    def score_system(
        self,
        part1_result: CRBenchResourceScoreResult,
        runtime_metrics: SystemRuntimeMetrics,
        target_vram_mb: Optional[float] = None,
        reference_ttft_ms: Optional[float] = None,
        reference_decode_throughput_tok_sec: Optional[float] = None
    ) -> CRBenchSystemScoreResult:
        """
        Computes the Part 2 system score using constrained utility.
        """
        v_budget = target_vram_mb if target_vram_mb is not None else self.vram_budget
        ref_ttft = reference_ttft_ms if reference_ttft_ms is not None else self.ref_ttft
        ref_thru = reference_decode_throughput_tok_sec if reference_decode_throughput_tok_sec is not None else self.ref_decode_thru

        # 1. TTFT efficiency factor (rewards faster prefill up to 1.25x, penalizes slowdown)
        ttft_ratio = runtime_metrics.mean_ttft_ms / max(1.0, ref_ttft)
        if ttft_ratio <= 1.0:
            phi_ttft = min(1.25, 1.0 + (1.0 - ttft_ratio) * 0.35)
        else:
            phi_ttft = 1.0 / (1.0 + (ttft_ratio - 1.0) ** 0.8)

        # 2. Throughput efficiency factor (rewards faster decoding, penalizes slowdown)
        thru_ratio = runtime_metrics.mean_decode_throughput_tok_per_sec / max(1.0, ref_thru)
        phi_thru = min(1.25, max(0.2, thru_ratio ** 0.5))

        # 3. Peak VRAM footprint factor (rewards smaller footprint, penalizes exceeding budget)
        vram_ratio = runtime_metrics.peak_vram_mb / max(1.0, v_budget)
        if vram_ratio <= 1.0:
            phi_vram = min(1.25, 1.0 + (1.0 - vram_ratio) * 0.25)
        else:
            phi_vram = 1.0 / (1.0 + (vram_ratio - 1.0) * 1.5)

        # Composite system utility multiplier
        raw_multiplier = (phi_ttft * phi_thru * phi_vram) ** self.alpha
        system_multiplier = float(np.clip(raw_multiplier, 0.1, 1.35))

        # System score = S_res * system_multiplier
        s_res = part1_result.resource_score
        s_sys = max(0.0, min(100.0, s_res * system_multiplier))

        return CRBenchSystemScoreResult(
            method_name=part1_result.method_name,
            system_score=float(s_sys),
            resource_score=float(s_res),
            system_utility_multiplier=float(system_multiplier),
            ttft_efficiency_factor=float(phi_ttft),
            throughput_efficiency_factor=float(phi_thru),
            vram_efficiency_factor=float(phi_vram),
            runtime_metrics=runtime_metrics,
            breakdown={
                "ttft_ratio": ttft_ratio,
                "thru_ratio": thru_ratio,
                "vram_ratio": vram_ratio,
            }
        )
