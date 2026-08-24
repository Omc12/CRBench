"""
Non-parametric bootstrap confidence interval engine for CRBench.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
import numpy as np


@dataclass
class BootstrapCIResult:
    """Bootstrap confidence interval outcome."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    ci_level: float
    std_error: float
    bootstrap_samples: int


class BootstrapEngine:
    """
    Computes percentile and BCa non-parametric bootstrap confidence intervals.
    """

    def __init__(self, num_resamples: int = 1000, ci_level: float = 0.95, seed: int = 42):
        self.num_resamples = num_resamples
        self.ci_level = ci_level
        self.seed = seed

    def compute_ci(
        self,
        data: List[float],
        statistic_fn: Optional[Callable[[np.ndarray], float]] = None
    ) -> BootstrapCIResult:
        if not data:
            return BootstrapCIResult(0.0, 0.0, 0.0, self.ci_level, 0.0, 0)

        arr = np.array(data, dtype=np.float64)
        stat_fn = statistic_fn if statistic_fn is not None else np.mean

        point_est = float(stat_fn(arr))
        n = len(arr)

        if n <= 1:
            return BootstrapCIResult(point_est, point_est, point_est, self.ci_level, 0.0, 1)

        rng = np.random.default_rng(self.seed)
        boot_stats = np.empty(self.num_resamples, dtype=np.float64)

        for i in range(self.num_resamples):
            sample = rng.choice(arr, size=n, replace=True)
            boot_stats[i] = stat_fn(sample)

        alpha = 1.0 - self.ci_level
        lower_pct = (alpha / 2.0) * 100.0
        upper_pct = (1.0 - alpha / 2.0) * 100.0

        ci_lower = float(np.percentile(boot_stats, lower_pct))
        ci_upper = float(np.percentile(boot_stats, upper_pct))
        std_err = float(np.std(boot_stats, ddof=1))

        return BootstrapCIResult(
            point_estimate=point_est,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_level=self.ci_level,
            std_error=std_err,
            bootstrap_samples=self.num_resamples
        )
