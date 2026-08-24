"""
Ranking stability and correlation statistics for CRBench.
Measures Spearman rho and Kendall tau across tasks, context lengths, and ablations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import spearmanr, kendalltau


@dataclass
class RankingStabilityResult:
    """Correlation between rankings across configurations."""
    spearman_rho: float
    spearman_pvalue: float
    kendall_tau: float
    kendall_pvalue: float
    num_items: int


class StabilityEngine:
    """
    Evaluates consistency of method rankings across different tasks and context lengths.
    """

    @staticmethod
    def compute_ranking_correlation(
        scores_1: List[float],
        scores_2: List[float]
    ) -> RankingStabilityResult:
        if len(scores_1) != len(scores_2):
            raise ValueError("Score lists must have identical length.")
        
        n = len(scores_1)
        if n < 2:
            return RankingStabilityResult(1.0, 1.0, 1.0, 1.0, n)

        rho, rho_p = spearmanr(scores_1, scores_2)
        tau, tau_p = kendalltau(scores_1, scores_2)

        return RankingStabilityResult(
            spearman_rho=float(rho) if not np.isnan(rho) else 1.0,
            spearman_pvalue=float(rho_p) if not np.isnan(rho_p) else 1.0,
            kendall_tau=float(tau) if not np.isnan(tau) else 1.0,
            kendall_pvalue=float(tau_p) if not np.isnan(tau_p) else 1.0,
            num_items=n
        )
