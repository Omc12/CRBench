"""
Pareto frontier analysis and non-dominated sorting engine for CRBench.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class OperatingPoint:
    """A single evaluated configuration point."""
    method_name: str
    context_length: int
    budget_value: float  # e.g., effective bits per token or bytes
    quality_score: float  # [0, 100] normalized or raw
    memory_cost: float   # in MB, GB, or bpt (lower is better)
    latency_ms: float = 0.0 # (lower is better)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParetoFrontierResult:
    """Extracted Pareto frontier and dominated points."""
    frontier_points: List[OperatingPoint]
    dominated_points: List[OperatingPoint]
    all_points: List[OperatingPoint]

    def is_pareto_optimal(self, point: OperatingPoint) -> bool:
        return any(
            p.method_name == point.method_name and
            abs(p.quality_score - point.quality_score) < 1e-5 and
            abs(p.memory_cost - point.memory_cost) < 1e-5
            for p in self.frontier_points
        )


class ParetoAnalyzer:
    """
    Computes 2D and 3D Pareto frontiers and dominance relations.
    Objectives:
    - Quality Score: Maximize
    - Memory Cost: Minimize
    - Latency (Optional): Minimize
    """

    @staticmethod
    def dominates(p1: OperatingPoint, p2: OperatingPoint, include_latency: bool = False) -> bool:
        """Returns True if p1 Pareto-dominates p2."""
        q1, m1 = p1.quality_score, p1.memory_cost
        q2, m2 = p2.quality_score, p2.memory_cost

        if not include_latency:
            # Better or equal in all, strictly better in at least one
            not_worse = (q1 >= q2) and (m1 <= m2)
            strictly_better = (q1 > q2) or (m1 < m2)
            return not_worse and strictly_better
        else:
            l1, l2 = p1.latency_ms, p2.latency_ms
            not_worse = (q1 >= q2) and (m1 <= m2) and (l1 <= l2)
            strictly_better = (q1 > q2) or (m1 < m2) or (l1 < l2)
            return not_worse and strictly_better

    @classmethod
    def compute_frontier(
        cls,
        points: List[OperatingPoint],
        include_latency: bool = False
    ) -> ParetoFrontierResult:
        """
        Extracts non-dominated operating points from a set of candidate points.
        """
        if not points:
            return ParetoFrontierResult([], [], [])

        frontier: List[OperatingPoint] = []
        dominated: List[OperatingPoint] = []

        for p in points:
            is_dom = False
            for other in points:
                if p is other:
                    continue
                if cls.dominates(other, p, include_latency=include_latency):
                    is_dom = True
                    break
            if not is_dom:
                frontier.append(p)
            else:
                dominated.append(p)

        # Sort frontier points by memory_cost ascending (and quality ascending)
        frontier = sorted(frontier, key=lambda x: (x.memory_cost, x.quality_score))

        return ParetoFrontierResult(
            frontier_points=frontier,
            dominated_points=dominated,
            all_points=points
        )
