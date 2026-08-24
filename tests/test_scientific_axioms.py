"""
Rigorous Scientific Axiom & Scoring Engine Validation Tests for CRBench.
Validates:
1. Method Discrimination across quality-resource curves.
2. Quality Monotonicity (better quality at same resource never reduces score).
3. Resource Monotonicity (lower memory at same quality never reduces score).
4. Pareto Consistency (Pareto-dominating methods strictly outrank dominated methods).
5. Mathematical Boundedness (scores strictly in [0.0, 100.0]).
6. Model-Relative Normalization & Dynamic Range Gating.
7. AUQC Interpolation Stability.
8. Context-Length Weighting Sensitivity.
"""

import pytest
import numpy as np
from crbench.scoring.pareto import OperatingPoint, ParetoAnalyzer
from crbench.scoring.auqc import AUQCEngine
from crbench.scoring.isobudget import IsoBudgetScorer
from crbench.scoring.normalizer import QualityNormalizer
from crbench.scoring.resource_score import CRBenchResourceScorer
from crbench.scoring.system_score import CRBenchSystemScorer, SystemRuntimeMetrics
from crbench.statistics.sensitivity import WeightingSensitivityAnalyzer


def test_axiom_method_discrimination():
    """Axiom 1: Distinct quality-resource curves produce distinct CRBench scores."""
    scorer = CRBenchResourceScorer(weighting_scheme="logarithmic")

    pts_method_1 = {
        4096: [
            OperatingPoint("M1", 4096, 2.0, 80.0, 2.0),
            OperatingPoint("M1", 4096, 4.0, 90.0, 4.0),
            OperatingPoint("M1", 4096, 8.0, 95.0, 8.0),
            OperatingPoint("M1", 4096, 16.0, 100.0, 16.0),
        ]
    }
    pts_method_2 = {
        4096: [
            OperatingPoint("M2", 4096, 2.0, 20.0, 2.0),
            OperatingPoint("M2", 4096, 4.0, 50.0, 4.0),
            OperatingPoint("M2", 4096, 8.0, 80.0, 8.0),
            OperatingPoint("M2", 4096, 16.0, 100.0, 16.0),
        ]
    }

    res1 = scorer.score_method("M1", pts_method_1)
    res2 = scorer.score_method("M2", pts_method_2)

    assert res1.resource_score != res2.resource_score
    assert res1.resource_score > res2.resource_score


def test_axiom_quality_monotonicity():
    """Axiom 2: Better quality at the same resource budget must never reduce score."""
    auqc_engine = AUQCEngine(log_scale=True)
    budgets = [2.0, 4.0, 8.0, 16.0]

    q_base = [30.0, 60.0, 85.0, 95.0]
    q_improved = [45.0, 70.0, 90.0, 98.0]  # Strictly higher quality at every budget

    auqc_base = auqc_engine.compute_auqc(budgets, q_base).auqc_score
    auqc_improved = auqc_engine.compute_auqc(budgets, q_improved).auqc_score

    assert auqc_improved > auqc_base


def test_axiom_resource_monotonicity():
    """Axiom 3: Lower resource use for the same quality retention must never reduce AUQC score."""
    auqc_engine = AUQCEngine(log_scale=True)

    # Method A reaches 90% quality at 4 bpt, while Method B requires 8 bpt to reach 90% quality
    budgets_a = [2.0, 4.0, 8.0, 16.0]
    qualities_a = [70.0, 90.0, 95.0, 100.0]

    budgets_b = [2.0, 4.0, 8.0, 16.0]
    qualities_b = [50.0, 70.0, 90.0, 100.0]

    auqc_a = auqc_engine.compute_auqc(budgets_a, qualities_a).auqc_score
    auqc_b = auqc_engine.compute_auqc(budgets_b, qualities_b).auqc_score

    assert auqc_a > auqc_b


def test_axiom_pareto_dominance_consistency():
    """Axiom 4: A Pareto-dominating method must strictly outrank the method it dominates."""
    scorer = CRBenchResourceScorer()

    # Method Dominator has higher quality at lower/equal cost
    pts_dominator = {
        2048: [
            OperatingPoint("Dominator", 2048, 2.0, 85.0, 2.0),
            OperatingPoint("Dominator", 2048, 4.0, 95.0, 4.0),
            OperatingPoint("Dominator", 2048, 8.0, 100.0, 8.0),
        ]
    }
    # Method Dominated has lower quality at higher/equal cost
    pts_dominated = {
        2048: [
            OperatingPoint("Dominated", 2048, 2.0, 40.0, 2.0),
            OperatingPoint("Dominated", 2048, 4.0, 60.0, 4.0),
            OperatingPoint("Dominated", 2048, 8.0, 80.0, 8.0),
        ]
    }

    score_dom = scorer.score_method("Dominator", pts_dominator)
    score_sub = scorer.score_method("Dominated", pts_dominated)

    assert score_dom.resource_score > score_sub.resource_score

    # Check Pareto analyzer
    p_dom = OperatingPoint("Dominator", 2048, 4.0, 95.0, 4.0)
    p_sub = OperatingPoint("Dominated", 2048, 8.0, 80.0, 8.0)
    assert ParetoAnalyzer.dominates(p_dom, p_sub)
    assert not ParetoAnalyzer.dominates(p_sub, p_dom)


def test_axiom_score_boundedness():
    """Axiom 5: All CRBench scores must remain strictly bounded within [0.0, 100.0]."""
    scorer = CRBenchResourceScorer()
    sys_scorer = CRBenchSystemScorer()

    # Extreme Zero Points
    pts_zero = {
        2048: [OperatingPoint("Zero", 2048, 2.0, 0.0, 2.0), OperatingPoint("Zero", 2048, 16.0, 0.0, 16.0)],
        4096: [OperatingPoint("Zero", 4096, 2.0, 0.0, 2.0), OperatingPoint("Zero", 4096, 16.0, 0.0, 16.0)],
    }
    res_zero = scorer.score_method("Zero", pts_zero)
    assert res_zero.resource_score == 0.0

    # Extreme Saturated Points
    pts_max = {
        2048: [OperatingPoint("Max", 2048, 2.0, 100.0, 2.0), OperatingPoint("Max", 2048, 16.0, 100.0, 16.0)],
        4096: [OperatingPoint("Max", 4096, 2.0, 100.0, 2.0), OperatingPoint("Max", 4096, 16.0, 100.0, 16.0)],
    }
    res_max = scorer.score_method("Max", pts_max)
    assert pytest.approx(res_max.resource_score, 1e-5) == 100.0

    # System Score Boundedness under extreme multipliers
    metrics_extreme_fast = SystemRuntimeMetrics(
        mean_ttft_ms=1.0,
        mean_prefill_throughput_tok_per_sec=50000.0,
        mean_decode_latency_ms_per_tok=1.0,
        mean_decode_throughput_tok_per_sec=1000.0,
        peak_vram_mb=100.0
    )
    sys_res_max = sys_scorer.score_system(res_max, metrics_extreme_fast)
    assert 0.0 <= sys_res_max.system_score <= 100.0


def test_axiom_model_relative_normalization():
    """Axiom 6: Model-relative normalization preserves scale invariance and enforces dynamic range gating."""
    norm = QualityNormalizer(floor_score=0.0, min_dynamic_range=0.05)

    # 1. Full capability retention
    assert norm.normalize(raw_score=0.75, dense_reference_score=0.75) == 100.0

    # 2. Scale Invariance: A 0.5B model scoring 40%/80% has identical retention to a 70B model scoring 45%/90%
    retention_small = norm.normalize(raw_score=0.40, dense_reference_score=0.80)
    retention_large = norm.normalize(raw_score=0.45, dense_reference_score=0.90)
    assert pytest.approx(retention_small, 1e-5) == retention_large == 50.0

    # 3. Dynamic Range Gating
    # When base model fails task (raw dense is near floor, e.g. 0.02)
    gated_res = norm.normalize_detailed(raw_score=0.02, dense_reference_score=0.02)
    assert gated_res.normalized_quality == 0.0
    assert not gated_res.is_dense_valid


def test_axiom_auqc_interpolation_stability():
    """Axiom 7: AUQC numerical integration is stable across monotonic PCHIP and linear splines."""
    engine_pchip = AUQCEngine(log_scale=True, interpolation="pchip")
    engine_linear = AUQCEngine(log_scale=True, interpolation="linear")

    budgets = [2.0, 4.0, 8.0, 16.0]
    qualities = [35.0, 65.0, 90.0, 100.0]

    score_pchip = engine_pchip.compute_auqc(budgets, qualities).auqc_score
    score_linear = engine_linear.compute_auqc(budgets, qualities).auqc_score

    # Scores should be within 3% of each other (stable integration)
    assert abs(score_pchip - score_linear) < 3.0
    assert 0.0 <= score_pchip <= 100.0
    assert 0.0 <= score_linear <= 100.0


def test_axiom_context_weighting_sensitivity():
    """Axiom 8: Context weighting sensitivity produces high rank correlation across schemes."""
    data = {
        "A": {2048: [OperatingPoint("A", 2048, 4.0, 95.0, 4.0)], 8192: [OperatingPoint("A", 8192, 4.0, 90.0, 4.0)]},
        "B": {2048: [OperatingPoint("B", 2048, 4.0, 75.0, 4.0)], 8192: [OperatingPoint("B", 8192, 4.0, 70.0, 4.0)]},
        "C": {2048: [OperatingPoint("C", 2048, 4.0, 50.0, 4.0)], 8192: [OperatingPoint("C", 8192, 4.0, 40.0, 4.0)]},
    }
    analyzer = WeightingSensitivityAnalyzer()
    res = analyzer.analyze(data)

    rho = res.spearman_correlations[("logarithmic", "uniform")]
    tau = res.kendall_taus[("logarithmic", "uniform")]

    assert rho >= 0.99
    assert tau >= 0.99
