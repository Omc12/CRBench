"""
Hypervolume indicator computation for multi-objective quality-resource evaluation.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from crbench.scoring.pareto import OperatingPoint, ParetoAnalyzer


@dataclass
class HypervolumeResult:
    """Hypervolume indicator evaluation."""
    method_name: str
    context_length: int
    hypervolume: float          # Raw hypervolume area
    normalized_score: float     # [0.0, 100.0] normalized hypervolume
    reference_point: Tuple[float, float]  # (Memory_max, Quality_min=0)


class HypervolumeScorer:
    """
    Computes 2D Hypervolume Indicator of the Pareto front formed by a method's operating points.
    Objectives:
    - Quality (maximize in [0, 100])
    - Memory (minimize in [0, B_max])
    Reference point: (B_max, 0.0)
    """

    def __init__(self, default_b_max: float = 16.0):
        self.default_b_max = default_b_max

    def compute_hypervolume(
        self,
        points: List[OperatingPoint],
        b_max: Optional[float] = None
    ) -> HypervolumeResult:
        if not points:
            return HypervolumeResult(
                method_name="empty",
                context_length=0,
                hypervolume=0.0,
                normalized_score=0.0,
                reference_point=(self.default_b_max, 0.0)
            )

        method_name = points[0].method_name
        ctx_len = points[0].context_length

        ref_b = b_max if b_max is not None else max(self.default_b_max, max(p.memory_cost for p in points))
        ref_q = 0.0
        ref_point = (ref_b, ref_q)

        # Get Pareto frontier points
        frontier_res = ParetoAnalyzer.compute_frontier(points, include_latency=False)
        frontier = frontier_res.frontier_points

        # Sort by memory ascending
        frontier = sorted(frontier, key=lambda p: p.memory_cost)

        # Compute 2D Lebesgue measure (hypervolume) dominated by the frontier above ref_point
        # Box decomposition:
        hv = 0.0
        prev_b = 0.0
        # Maximum possible area in bounding box [0, ref_b] x [0, 100]
        max_possible_area = ref_b * 100.0

        for i, pt in enumerate(frontier):
            q = max(0.0, min(100.0, pt.quality_score))
            m = min(ref_b, max(0.0, pt.memory_cost))
            
            # Next boundary is either next point's memory or ref_b
            next_b = frontier[i + 1].memory_cost if i + 1 < len(frontier) else ref_b
            next_b = min(ref_b, next_b)
            
            width = max(0.0, next_b - m)
            height = max(0.0, q - ref_q)
            hv += width * height

        norm_score = max(0.0, min(100.0, (hv / max_possible_area) * 100.0)) if max_possible_area > 0 else 0.0

        return HypervolumeResult(
            method_name=method_name,
            context_length=ctx_len,
            hypervolume=float(hv),
            normalized_score=float(norm_score),
            reference_point=ref_point
        )
