"""
Unit tests for Scoring Engines (Normalizer, Pareto, AUQC, IsoBudget, Hypervolume, Part 1 & 2 Scores, Sensitivity).
"""

import pytest
from crbench.scoring.normalizer import QualityNormalizer, NormalizationResult
from crbench.scoring.pareto import OperatingPoint, ParetoAnalyzer
from crbench.scoring.auqc import AUQCEngine
from crbench.scoring.isobudget import IsoBudgetScorer
from crbench.scoring.hypervolume import HypervolumeScorer
from crbench.scoring.resource_score import CRBenchResourceScorer
from crbench.scoring.system_score import CRBenchSystemScorer, SystemRuntimeMetrics
from crbench.statistics.sensitivity import WeightingSensitivityAnalyzer


def test_quality_normalizer():
    norm = QualityNormalizer(floor_score=0.0, min_dynamic_range=0.05)
    # Dense achieved 80%, candidate achieved 80% -> 100.0% normalized
    assert norm.normalize(raw_score=0.80, dense_reference_score=0.80) == 100.0
    # Candidate achieved 40% -> 50.0% normalized
    assert pytest.approx(norm.normalize(raw_score=0.40, dense_reference_score=0.80), 0.1) == 50.0
    # Boundary clamp test
    assert norm.normalize(raw_score=-0.05, dense_reference_score=0.80) == 0.0

    # Dynamic range gating test: dense reference is near floor (e.g. 0.02)
    # Candidate raw score <= dense -> returns 0.0 (preventing division-by-near-zero explosion)
    res_gated = norm.normalize_detailed(raw_score=0.02, dense_reference_score=0.02)
    assert res_gated.normalized_quality == 0.0
    assert not res_gated.is_dense_valid


def test_context_weighting_schemes():
    scorer = CRBenchResourceScorer(weighting_scheme="logarithmic")
    lengths = [2048, 4096, 8192, 16384]

    # Test logarithmic weights sum to 1.0 and increase monotonically
    w_log = scorer.compute_context_weights(lengths, "logarithmic")
    assert pytest.approx(sum(w_log.values()), 1e-5) == 1.0
    assert w_log[2048] < w_log[4096] < w_log[8192] < w_log[16384]

    # Test uniform weights
    w_uni = scorer.compute_context_weights(lengths, "uniform")
    assert pytest.approx(sum(w_uni.values()), 1e-5) == 1.0
    assert w_uni[2048] == w_uni[16384] == 0.25

    # Test linear weights
    w_lin = scorer.compute_context_weights(lengths, "linear")
    assert pytest.approx(sum(w_lin.values()), 1e-5) == 1.0
    assert w_lin[16384] == 16384 / sum(lengths)


def test_weighting_sensitivity_analyzer():
    methods_data = {
        "dense_fp16": {
            2048: [OperatingPoint("dense_fp16", 2048, 16.0, 100.0, 16.0)],
            8192: [OperatingPoint("dense_fp16", 8192, 16.0, 98.0, 16.0)],
        },
        "kv_quant_int4": {
            2048: [OperatingPoint("kv_quant_int4", 2048, 4.0, 95.0, 4.0)],
            8192: [OperatingPoint("kv_quant_int4", 8192, 4.0, 90.0, 4.0)],
        },
        "snapkv": {
            2048: [OperatingPoint("snapkv", 2048, 4.0, 88.0, 4.0)],
            8192: [OperatingPoint("snapkv", 8192, 4.0, 80.0, 4.0)],
        }
    }
    analyzer = WeightingSensitivityAnalyzer()
    res = analyzer.analyze(methods_data)

    assert "logarithmic" in res.scores_by_scheme
    assert "uniform" in res.scores_by_scheme
    assert "linear" in res.scores_by_scheme

    # Verify rank correlation between logarithmic and uniform is strong
    rho = res.spearman_correlations[("logarithmic", "uniform")]
    assert rho >= 0.90


def test_pareto_analysis():
    p1 = OperatingPoint(method_name="A", context_length=8192, budget_value=4.0, quality_score=90.0, memory_cost=4.0)
    p2 = OperatingPoint(method_name="B", context_length=8192, budget_value=8.0, quality_score=85.0, memory_cost=8.0) # Dominated by A (lower quality, higher cost)
    p3 = OperatingPoint(method_name="C", context_length=8192, budget_value=2.0, quality_score=70.0, memory_cost=2.0) # Non-dominated (lower cost)

    res = ParetoAnalyzer.compute_frontier([p1, p2, p3])
    frontier_names = [p.method_name for p in res.frontier_points]
    assert "A" in frontier_names
    assert "C" in frontier_names
    assert "B" in [p.method_name for p in res.dominated_points]


def test_auqc_engine_monotonicity():
    engine = AUQCEngine(log_scale=True)
    
    budgets = [2.0, 4.0, 8.0, 16.0]
    high_qualities = [80.0, 90.0, 95.0, 100.0]
    low_qualities = [50.0, 60.0, 75.0, 85.0]

    auqc_high = engine.compute_auqc(budgets, high_qualities).auqc_score
    auqc_low = engine.compute_auqc(budgets, low_qualities).auqc_score

    assert 0.0 <= auqc_high <= 100.0
    assert 0.0 <= auqc_low <= 100.0
    assert auqc_high > auqc_low


def test_isobudget_scorer():
    scorer = IsoBudgetScorer(standard_budgets_bpt=[2.0, 4.0, 8.0, 16.0])
    budgets = [2.0, 4.0, 8.0, 16.0]
    qualities = [60.0, 80.0, 95.0, 100.0]
    
    res = scorer.evaluate_isobudgets(budgets, qualities)
    assert res.budget_scores[2.0] == 60.0
    assert res.budget_scores[4.0] == 80.0
    assert res.budget_scores[16.0] == 100.0


def test_hypervolume_scorer():
    scorer = HypervolumeScorer()
    p1 = OperatingPoint(method_name="A", context_length=8192, budget_value=4.0, quality_score=90.0, memory_cost=4.0)
    p2 = OperatingPoint(method_name="A", context_length=8192, budget_value=8.0, quality_score=95.0, memory_cost=8.0)
    
    res = scorer.compute_hypervolume([p1, p2], b_max=16.0)
    assert 0.0 < res.normalized_score <= 100.0


def test_full_scoring_pipeline():
    res_scorer = CRBenchResourceScorer(weighting_scheme="logarithmic")
    points = {
        8192: [
            OperatingPoint(method_name="int4", context_length=8192, budget_value=4.0, quality_score=85.0, memory_cost=4.0),
            OperatingPoint(method_name="int4", context_length=8192, budget_value=8.0, quality_score=95.0, memory_cost=8.0),
        ]
    }
    part1_res = res_scorer.score_method("int4", points)
    assert 0.0 <= part1_res.resource_score <= 100.0

    sys_scorer = CRBenchSystemScorer()
    sys_metrics = SystemRuntimeMetrics(
        mean_ttft_ms=300.0,
        mean_prefill_throughput_tok_per_sec=1500.0,
        mean_decode_latency_ms_per_tok=25.0,
        mean_decode_throughput_tok_per_sec=40.0,
        peak_vram_mb=4096.0
    )
    part2_res = sys_scorer.score_system(part1_res, sys_metrics)
    assert 0.0 <= part2_res.system_score <= 100.0


def test_method_differentiation_auqc_and_resource_score():
    """
    Proves that two methods with different quality-memory tradeoff curves produce
    strictly distinct AUQC and Part 1 Resource Scores (S_res).
    """
    res_scorer = CRBenchResourceScorer(weighting_scheme="logarithmic")

    # Method A: High efficiency (retains high quality even at 2 and 4 bpt)
    # e.g., DKV / SnapKV
    pts_method_a = {
        4096: [
            OperatingPoint(method_name="Method_A", context_length=4096, budget_value=2.0, quality_score=85.0, memory_cost=2.0),
            OperatingPoint(method_name="Method_A", context_length=4096, budget_value=4.0, quality_score=92.0, memory_cost=4.0),
            OperatingPoint(method_name="Method_A", context_length=4096, budget_value=8.0, quality_score=98.0, memory_cost=8.0),
            OperatingPoint(method_name="Method_A", context_length=4096, budget_value=16.0, quality_score=100.0, memory_cost=16.0),
        ]
    }

    # Method B: Low efficiency (suffers sharp degradation below 8 bpt)
    # e.g., naive truncation / INT2
    pts_method_b = {
        4096: [
            OperatingPoint(method_name="Method_B", context_length=4096, budget_value=2.0, quality_score=20.0, memory_cost=2.0),
            OperatingPoint(method_name="Method_B", context_length=4096, budget_value=4.0, quality_score=45.0, memory_cost=4.0),
            OperatingPoint(method_name="Method_B", context_length=4096, budget_value=8.0, quality_score=85.0, memory_cost=8.0),
            OperatingPoint(method_name="Method_B", context_length=4096, budget_value=16.0, quality_score=100.0, memory_cost=16.0),
        ]
    }

    score_a = res_scorer.score_method("Method_A", pts_method_a)
    score_b = res_scorer.score_method("Method_B", pts_method_b)

    # Assert AUQC and S_res are strictly different and Method A is ranked higher
    assert score_a.resource_score != score_b.resource_score
    assert score_a.resource_score > score_b.resource_score + 15.0  # Significant separation
    assert score_a.context_scores[4096].auqc_result.auqc_score > score_b.context_scores[4096].auqc_result.auqc_score


def test_system_score_hardware_sensitivity():
    """
    Proves that Part 2 System Score responds dynamically to hardware runtime differences
    (TTFT, throughput, peak VRAM) even when Part 1 Resource scores are identical.
    """
    res_scorer = CRBenchResourceScorer()
    dummy_pts = {
        4096: [
            OperatingPoint(method_name="M", context_length=4096, budget_value=4.0, quality_score=80.0, memory_cost=4.0),
            OperatingPoint(method_name="M", context_length=4096, budget_value=8.0, quality_score=90.0, memory_cost=8.0),
        ]
    }
    part1_res = res_scorer.score_method("M", dummy_pts)

    sys_scorer = CRBenchSystemScorer()

    # Fast Profile (Low TTFT, high throughput, small VRAM)
    metrics_fast = SystemRuntimeMetrics(
        mean_ttft_ms=500.0,
        mean_prefill_throughput_tok_per_sec=2000.0,
        mean_decode_latency_ms_per_tok=10.0,
        mean_decode_throughput_tok_per_sec=100.0,
        peak_vram_mb=2048.0
    )

    # Slow Profile (High TTFT latency penalty, lower throughput, high VRAM)
    metrics_slow = SystemRuntimeMetrics(
        mean_ttft_ms=5000.0,
        mean_prefill_throughput_tok_per_sec=400.0,
        mean_decode_latency_ms_per_tok=50.0,
        mean_decode_throughput_tok_per_sec=20.0,
        peak_vram_mb=16384.0
    )

    score_fast = sys_scorer.score_system(part1_res, metrics_fast, reference_ttft_ms=1000.0, reference_decode_throughput_tok_sec=30.0)
    score_slow = sys_scorer.score_system(part1_res, metrics_slow, reference_ttft_ms=1000.0, reference_decode_throughput_tok_sec=30.0)

    assert score_fast.system_score != score_slow.system_score
    assert score_fast.system_utility_multiplier > score_slow.system_utility_multiplier
    assert score_fast.system_score > score_slow.system_score

