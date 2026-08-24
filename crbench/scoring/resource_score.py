"""
Part 1 — CRBench Resource Score (S_res) Engine.
Integrates Contextual Quality vs. Memory Resource Consumption across context lengths
using mathematically principled weighting schemes (Logarithmic, Uniform, Linear).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional
import numpy as np
from crbench.scoring.auqc import AUQCEngine, AUQCResult
from crbench.scoring.isobudget import IsoBudgetScorer, IsoBudgetResult
from crbench.scoring.pareto import OperatingPoint, ParetoAnalyzer, ParetoFrontierResult
from crbench.scoring.hypervolume import HypervolumeScorer, HypervolumeResult


@dataclass
class ContextLengthResourceScore:
    """Resource evaluation results for a single context length."""
    context_length: int
    auqc_result: AUQCResult
    isobudget_result: IsoBudgetResult
    pareto_frontier: ParetoFrontierResult
    hypervolume_result: HypervolumeResult


@dataclass
class CRBenchResourceScoreResult:
    """Comprehensive Part 1 Benchmark Resource Score result."""
    method_name: str
    resource_score: float                # Final aggregate S_res in [0.0, 100.0]
    context_scores: Dict[int, ContextLengthResourceScore]
    context_weights: Dict[int, float]
    weighting_scheme: str = "logarithmic"
    summary_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def mean_auqc(self) -> float:
        auqcs = [cs.auqc_result.auqc_score for cs in self.context_scores.values()]
        return float(np.mean(auqcs)) if auqcs else 0.0


class CRBenchResourceScorer:
    """
    Scoring Engine for Part 1 (CRBench Resource Score).
    Evaluates representation efficiency across resource budgets and context lengths.

    Supports configurable Context-Length Weighting schemes:
    - "logarithmic" (Default): w_L proportional to (1 + log2(L / L_min)). Smoothly rewards longer sequences.
    - "uniform": Equal weight 1/|L| across all context lengths.
    - "linear": w_L proportional to L. Heavily weights long contexts.
    """

    def __init__(
        self,
        log_scale_auqc: bool = True,
        weighting_scheme: str = "logarithmic",
        standard_budgets_bpt: Optional[List[float]] = None
    ):
        self.auqc_engine = AUQCEngine(log_scale=log_scale_auqc)
        self.weighting_scheme = weighting_scheme.lower()
        self.isobudget_scorer = IsoBudgetScorer(standard_budgets_bpt=standard_budgets_bpt)
        self.hypervolume_scorer = HypervolumeScorer()

    def compute_context_weights(
        self,
        context_lengths: List[int],
        scheme: Optional[str] = None
    ) -> Dict[int, float]:
        """
        Computes normalized context length weights for a given scheme.
        """
        lengths = sorted(context_lengths)
        if not lengths:
            return {}
        if len(lengths) == 1:
            return {lengths[0]: 1.0}

        target_scheme = (scheme or self.weighting_scheme).lower()
        l_min = lengths[0]

        if target_scheme == "logarithmic":
            raw_weights = {L: 1.0 + math.log2(max(1.0, L / max(1.0, l_min))) for L in lengths}
        elif target_scheme == "linear":
            raw_weights = {L: float(L) for L in lengths}
        elif target_scheme == "uniform":
            raw_weights = {L: 1.0 for L in lengths}
        else:
            raw_weights = {L: 1.0 for L in lengths}

        total_w = sum(raw_weights.values())
        return {L: raw_weights[L] / total_w for L in lengths}

    def score_method(
        self,
        method_name: str,
        operating_points_by_context: Dict[int, List[OperatingPoint]],
        context_weights: Optional[Dict[int, float]] = None,
        weighting_scheme: Optional[str] = None,
        min_budget_bound: Optional[float] = None,
        max_budget_bound: Optional[float] = None
    ) -> CRBenchResourceScoreResult:
        """
        Computes the complete Part 1 score for a method across context lengths.
        """
        context_scores: Dict[int, ContextLengthResourceScore] = {}
        lengths = sorted(operating_points_by_context.keys())

        if not lengths:
            return CRBenchResourceScoreResult(
                method_name=method_name,
                resource_score=0.0,
                context_scores={},
                context_weights={},
                weighting_scheme=weighting_scheme or self.weighting_scheme
            )

        active_scheme = weighting_scheme or self.weighting_scheme

        if context_weights is None:
            normalized_weights = self.compute_context_weights(lengths, active_scheme)
        else:
            total_w = sum(context_weights.get(L, 1.0) for L in lengths)
            normalized_weights = {L: context_weights.get(L, 1.0) / total_w for L in lengths}

        weighted_auqc_sum = 0.0

        for ctx_len in lengths:
            pts = operating_points_by_context[ctx_len]
            budgets = [p.budget_value for p in pts]
            qualities = [p.quality_score for p in pts]

            auqc_res = self.auqc_engine.compute_auqc(
                budgets=budgets,
                qualities=qualities,
                method_name=method_name,
                context_length=ctx_len,
                min_budget_bound=min_budget_bound,
                max_budget_bound=max_budget_bound
            )

            isobudget_res = self.isobudget_scorer.evaluate_isobudgets(
                budgets=budgets,
                qualities=qualities,
                method_name=method_name,
                context_length=ctx_len
            )

            frontier_res = ParetoAnalyzer.compute_frontier(pts, include_latency=False)
            hv_res = self.hypervolume_scorer.compute_hypervolume(pts, b_max=max_budget_bound)

            context_scores[ctx_len] = ContextLengthResourceScore(
                context_length=ctx_len,
                auqc_result=auqc_res,
                isobudget_result=isobudget_res,
                pareto_frontier=frontier_res,
                hypervolume_result=hv_res
            )

            w = normalized_weights[ctx_len]
            weighted_auqc_sum += w * auqc_res.auqc_score

        final_resource_score = max(0.0, min(100.0, weighted_auqc_sum))

        return CRBenchResourceScoreResult(
            method_name=method_name,
            resource_score=float(final_resource_score),
            context_scores=context_scores,
            context_weights=normalized_weights,
            weighting_scheme=active_scheme,
            summary_metrics={
                "num_context_lengths": len(lengths),
                "context_lengths": lengths,
                "weighting_scheme": active_scheme,
                "auqc_by_context": {L: context_scores[L].auqc_result.auqc_score for L in lengths}
            }
        )
