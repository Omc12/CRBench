"""
Hypothesis testing and effect size statistics for CRBench.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class PairedComparisonResult:
    """Outcome of a paired statistical comparison."""
    method_a: str
    method_b: str
    mean_diff: float            # Mean(A) - Mean(B)
    p_value: float              # Permutation test two-tailed p-value
    cohens_d: float             # Standardized effect size
    is_statistically_significant: bool
    num_pairs: int


class HypothesisEngine:
    """
    Executes paired permutation tests and computes effect sizes.
    """

    def __init__(self, num_permutations: int = 2000, alpha: float = 0.05, seed: int = 42):
        self.num_permutations = num_permutations
        self.alpha = alpha
        self.seed = seed

    def paired_permutation_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        method_a_name: str = "A",
        method_b_name: str = "B"
    ) -> PairedComparisonResult:
        if len(scores_a) != len(scores_b):
            raise ValueError("scores_a and scores_b must have equal length.")
        
        n = len(scores_a)
        if n == 0:
            return PairedComparisonResult(method_a_name, method_b_name, 0.0, 1.0, 0.0, False, 0)

        diffs = np.array(scores_a, dtype=np.float64) - np.array(scores_b, dtype=np.float64)
        obs_mean_diff = float(np.mean(diffs))

        # Cohen's d for paired samples
        std_diff = np.std(diffs, ddof=1) if n > 1 else 0.0
        cohens_d = float(obs_mean_diff / std_diff) if std_diff > 1e-6 else 0.0

        if n <= 1:
            return PairedComparisonResult(method_a_name, method_b_name, obs_mean_diff, 1.0, cohens_d, False, n)

        # Random sign-flipping permutation test
        rng = np.random.default_rng(self.seed)
        count_extreme = 0

        for _ in range(self.num_permutations):
            signs = rng.choice([-1.0, 1.0], size=n)
            perm_diff = np.mean(diffs * signs)
            if abs(perm_diff) >= abs(obs_mean_diff):
                count_extreme += 1

        p_val = max(1.0 / self.num_permutations, float(count_extreme / self.num_permutations))
        sig = p_val < self.alpha

        return PairedComparisonResult(
            method_a=method_a_name,
            method_b=method_b_name,
            mean_diff=obs_mean_diff,
            p_value=p_val,
            cohens_d=cohens_d,
            is_statistically_significant=sig,
            num_pairs=n
        )
