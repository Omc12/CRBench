"""
Iso-Budget evaluation engine for CRBench.
Evaluates contextual capability retention at standardized resource operating budgets.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator


@dataclass
class IsoBudgetResult:
    """Evaluation at standard discrete budgets."""
    method_name: str
    context_length: int
    budget_unit: str  # "bpt" or "GB"
    budget_scores: Dict[float, float]  # target_budget -> normalized_quality [0.0, 100.0]
    raw_budgets: List[float]
    raw_qualities: List[float]


class IsoBudgetScorer:
    """
    Interpolates normalized contextual quality at fixed, standardized budget points:
    - BPT: [2.0, 4.0, 8.0, 16.0] bits per token
    - KV Memory: [0.5, 1.0, 2.0, 4.0, 8.0, 16.0] GB
    """

    def __init__(self, standard_budgets_bpt: Optional[List[float]] = None):
        self.standard_bpt = standard_budgets_bpt or [2.0, 4.0, 8.0, 16.0]

    def evaluate_isobudgets(
        self,
        budgets: List[float],
        qualities: List[float],
        method_name: str = "method",
        context_length: int = 4096,
        target_budgets: Optional[List[float]] = None,
        budget_unit: str = "bpt"
    ) -> IsoBudgetResult:
        targets = target_budgets if target_budgets is not None else self.standard_bpt
        
        if len(budgets) == 0:
            return IsoBudgetResult(
                method_name=method_name,
                context_length=context_length,
                budget_unit=budget_unit,
                budget_scores={b: 0.0 for b in targets},
                raw_budgets=[],
                raw_qualities=[]
            )

        # Aggregate duplicates (mean quality for identical budgets)
        budget_map = {}
        for b, q in zip(budgets, qualities):
            b_key = round(float(b), 4)
            if b_key not in budget_map:
                budget_map[b_key] = []
            budget_map[b_key].append(float(q))

        unique_budgets = sorted(budget_map.keys())
        unique_qualities = [float(np.mean(budget_map[b])) for b in unique_budgets]

        b_arr = np.array(unique_budgets, dtype=np.float64)
        q_arr = np.array(unique_qualities, dtype=np.float64)

        if len(b_arr) == 1:
            scores = {}
            for t in targets:
                # If target budget is >= evaluated budget, capability is at least that of evaluated budget
                if t >= b_arr[0]:
                    scores[float(t)] = float(q_arr[0])
                else:
                    # If operating at a lower budget than evaluated, estimate degraded fraction
                    scores[float(t)] = max(0.0, float(q_arr[0] * (t / max(1e-4, b_arr[0]))))
        else:
            if len(b_arr) >= 3:
                interpolator = PchipInterpolator(b_arr, q_arr, extrapolate=True)
            else:
                interpolator = interp1d(b_arr, q_arr, kind="linear", fill_value="extrapolate")

            scores = {}
            for t in targets:
                try:
                    val = float(interpolator(t))
                    if np.isnan(val):
                        val = float(q_arr[-1]) if t >= b_arr[-1] else float(q_arr[0])
                except Exception:
                    val = float(q_arr[-1]) if t >= b_arr[-1] else float(q_arr[0])
                
                # Clamp interpolated value to [0.0, 100.0]
                q_est = float(np.clip(val, 0.0, 100.0))
                scores[float(t)] = q_est

        return IsoBudgetResult(
            method_name=method_name,
            context_length=context_length,
            budget_unit=budget_unit,
            budget_scores=scores,
            raw_budgets=list(b_arr),
            raw_qualities=list(q_arr)
        )
