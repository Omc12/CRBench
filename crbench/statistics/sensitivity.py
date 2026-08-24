"""
Context-Length Weighting Sensitivity and Robustness Analysis Engine.
Analyzes whether benchmark rankings and score distributions change materially
across Logarithmic, Uniform, and Linear context weighting schemes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import spearmanr, kendalltau
from crbench.scoring.pareto import OperatingPoint
from crbench.scoring.resource_score import CRBenchResourceScorer, CRBenchResourceScoreResult


@dataclass
class WeightingSensitivityResult:
    """Sensitivity analysis results across context weighting schemes."""
    method_names: List[str]
    scores_by_scheme: Dict[str, Dict[str, float]]   # scheme -> {method -> S_res}
    ranks_by_scheme: Dict[str, Dict[str, int]]      # scheme -> {method -> rank (1-indexed)}
    spearman_correlations: Dict[Tuple[str, str], float]  # (scheme1, scheme2) -> rho
    kendall_taus: Dict[Tuple[str, str], float]           # (scheme1, scheme2) -> tau
    max_rank_displacements: Dict[Tuple[str, str], int]   # (scheme1, scheme2) -> max |rank1 - rank2|
    summary_text: str = ""


class WeightingSensitivityAnalyzer:
    """
    Evaluates benchmark robustness across three principled context-length weighting formulations:
    1. 'logarithmic': w_L ~ 1 + log2(L / L_min) (CRBench Default)
    2. 'uniform': w_L = 1 / |L|
    3. 'linear': w_L ~ L
    """

    SCHEMES = ["logarithmic", "uniform", "linear"]

    def __init__(self, log_scale_auqc: bool = True):
        self.scorer = CRBenchResourceScorer(log_scale_auqc=log_scale_auqc)

    def analyze(
        self,
        methods_data: Dict[str, Dict[int, List[OperatingPoint]]]
    ) -> WeightingSensitivityResult:
        """
        Computes scores and rank statistics across all schemes.
        methods_data: method_name -> {context_length -> List[OperatingPoint]}
        """
        method_names = list(methods_data.keys())
        scores_by_scheme: Dict[str, Dict[str, float]] = {s: {} for s in self.SCHEMES}
        ranks_by_scheme: Dict[str, Dict[str, int]] = {s: {} for s in self.SCHEMES}

        for scheme in self.SCHEMES:
            scheme_scores = {}
            for method in method_names:
                res = self.scorer.score_method(
                    method_name=method,
                    operating_points_by_context=methods_data[method],
                    weighting_scheme=scheme
                )
                scheme_scores[method] = res.resource_score
            scores_by_scheme[scheme] = scheme_scores

            # Compute ranks (descending order of score, 1 = best)
            sorted_methods = sorted(method_names, key=lambda m: scheme_scores[m], reverse=True)
            ranks_by_scheme[scheme] = {m: rank + 1 for rank, m in enumerate(sorted_methods)}

        # Compute pairwise rank correlations
        spearman_corrs: Dict[Tuple[str, str], float] = {}
        kendall_corrs: Dict[Tuple[str, str], float] = {}
        max_rank_displacements: Dict[Tuple[str, str], int] = {}

        for i, s1 in enumerate(self.SCHEMES):
            for s2 in self.SCHEMES[i + 1:]:
                ranks1 = [ranks_by_scheme[s1][m] for m in method_names]
                ranks2 = [ranks_by_scheme[s2][m] for m in method_names]

                if len(method_names) >= 2:
                    rho, _ = spearmanr(ranks1, ranks2)
                    tau, _ = kendalltau(ranks1, ranks2)
                else:
                    rho, tau = 1.0, 1.0

                displacements = [abs(r1 - r2) for r1, r2 in zip(ranks1, ranks2)]
                max_d = max(displacements) if displacements else 0

                spearman_corrs[(s1, s2)] = float(rho) if not np.isnan(rho) else 1.0
                kendall_corrs[(s1, s2)] = float(tau) if not np.isnan(tau) else 1.0
                max_rank_displacements[(s1, s2)] = int(max_d)

        # Generate summary text
        summary_lines = ["### Weighting Scheme Sensitivity Analysis\n"]
        summary_lines.append("| Scheme Comparison | Spearman rho | Kendall tau | Max Rank Shift |")
        summary_lines.append("|:---|:---:|:---:|:---:|")
        for (s1, s2), rho in spearman_corrs.items():
            tau = kendall_corrs[(s1, s2)]
            max_d = max_rank_displacements[(s1, s2)]
            summary_lines.append(f"| {s1.capitalize()} vs {s2.capitalize()} | {rho:.4f} | {tau:.4f} | {max_d} |")

        return WeightingSensitivityResult(
            method_names=method_names,
            scores_by_scheme=scores_by_scheme,
            ranks_by_scheme=ranks_by_scheme,
            spearman_correlations=spearman_corrs,
            kendall_taus=kendall_corrs,
            max_rank_displacements=max_rank_displacements,
            summary_text="\n".join(summary_lines)
        )
