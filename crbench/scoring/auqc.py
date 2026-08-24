"""
Area Under the Quality-Resource Curve (AUQC) scoring engine for CRBench.
Integrates normalized contextual quality across continuous resource budgets.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import math
import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator


@dataclass
class AUQCResult:
    """AUQC integration outcome."""
    method_name: str
    context_length: int
    auqc_score: float                # [0.0, 100.0]
    min_budget: float
    max_budget: float
    is_log_scale: bool
    budget_points: List[float]
    quality_points: List[float]


class AUQCEngine:
    """
    Computes Area Under the Quality-Resource Curve (AUQC).
    Supports logarithmic resource scaling (preferred for exponential compression ranges)
    and linear scaling.
    """

    def __init__(self, log_scale: bool = True, interpolation: str = "pchip"):
        self.log_scale = log_scale
        self.interpolation = interpolation  # "pchip" (shape-preserving monotonic) or "linear"

    def compute_auqc(
        self,
        budgets: List[float],
        qualities: List[float],
        method_name: str = "method",
        context_length: int = 4096,
        min_budget_bound: Optional[float] = None,
        max_budget_bound: Optional[float] = None
    ) -> AUQCResult:
        """
        Integrates normalized quality over the budget range.
        budgets: list of budget points (e.g., effective bits/token or KV MB/GB)
        qualities: list of normalized quality scores in [0.0, 100.0]
        """
        if len(budgets) != len(qualities):
            raise ValueError("Budgets and qualities must have identical length.")
        if len(budgets) < 2:
            single_q = qualities[0] if qualities else 0.0
            return AUQCResult(
                method_name=method_name,
                context_length=context_length,
                auqc_score=float(single_q),
                min_budget=budgets[0] if budgets else 1.0,
                max_budget=budgets[0] if budgets else 1.0,
                is_log_scale=self.log_scale,
                budget_points=budgets,
                quality_points=qualities
            )

        # Aggregate duplicates (mean quality for identical or near-identical budgets)
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

        b_min = min_budget_bound if min_budget_bound is not None else float(b_arr[0])
        b_max = max_budget_bound if max_budget_bound is not None else float(b_arr[-1])

        if b_min <= 0:
            b_min = 1e-3  # Avoid log(0)

        if len(b_arr) == 1:
            return AUQCResult(
                method_name=method_name,
                context_length=context_length,
                auqc_score=float(q_arr[0]),
                min_budget=b_min,
                max_budget=b_max,
                is_log_scale=self.log_scale,
                budget_points=list(b_arr),
                quality_points=list(q_arr)
            )

        if self.log_scale:
            # Transform to log domain
            u_arr = np.log(np.maximum(1e-5, b_arr))
            u_min = math.log(max(1e-5, b_min))
            u_max = math.log(max(1e-5, b_max))

            if abs(u_max - u_min) < 1e-6:
                return AUQCResult(
                    method_name=method_name,
                    context_length=context_length,
                    auqc_score=float(q_arr[0]),
                    min_budget=b_min,
                    max_budget=b_max,
                    is_log_scale=True,
                    budget_points=list(b_arr),
                    quality_points=list(q_arr)
                )

            # Interpolate in log-space
            if len(u_arr) >= 3 and self.interpolation == "pchip":
                # Monotonic cubic interpolation
                interpolator = PchipInterpolator(u_arr, q_arr, extrapolate=True)
            else:
                interpolator = interp1d(u_arr, q_arr, kind="linear", fill_value="extrapolate")

            # Dense integration grid (100 points)
            grid_u = np.linspace(u_min, u_max, 100)
            grid_q = np.clip(interpolator(grid_u), 0.0, 100.0)

            # Trapezoidal integration normalized by span
            try:
                from scipy.integrate import trapezoid
                area = float(trapezoid(grid_q, grid_u))
            except Exception:
                area = float(0.5 * np.sum((grid_q[:-1] + grid_q[1:]) * (grid_u[1:] - grid_u[:-1])))
            normalized_auqc = area / (u_max - u_min)

        else:
            # Linear scale integration
            if abs(b_max - b_min) < 1e-6:
                return AUQCResult(
                    method_name=method_name,
                    context_length=context_length,
                    auqc_score=float(q_arr[0]),
                    min_budget=b_min,
                    max_budget=b_max,
                    is_log_scale=False,
                    budget_points=list(b_arr),
                    quality_points=list(q_arr)
                )

            if len(b_arr) >= 3 and self.interpolation == "pchip":
                interpolator = PchipInterpolator(b_arr, q_arr, extrapolate=True)
            else:
                interpolator = interp1d(b_arr, q_arr, kind="linear", fill_value="extrapolate")

            grid_b = np.linspace(b_min, b_max, 100)
            grid_q = np.clip(interpolator(grid_b), 0.0, 100.0)

            try:
                from scipy.integrate import trapezoid
                area = float(trapezoid(grid_q, grid_b))
            except Exception:
                area = float(0.5 * np.sum((grid_q[:-1] + grid_q[1:]) * (grid_b[1:] - grid_b[:-1])))
            normalized_auqc = area / (b_max - b_min)

        normalized_auqc = float(max(0.0, min(100.0, normalized_auqc)))

        return AUQCResult(
            method_name=method_name,
            context_length=context_length,
            auqc_score=normalized_auqc,
            min_budget=b_min,
            max_budget=b_max,
            is_log_scale=self.log_scale,
            budget_points=list(b_arr),
            quality_points=list(q_arr)
        )
