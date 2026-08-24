"""
Unit tests for the CRBench utility formulations.
Tests verify the default Linear formulation (α=0.70) and candidate families
(Cobb-Douglas, Harmonic, Power Mean) across all mathematical and scientific axioms.
"""
import math
import pytest
import numpy as np
from crbench.scoring.utility import (
    CRBENCH_ALPHA,
    CRBENCH_FORMULA_NAME,
    compute_utility,
    compute_utility_decomposition,
    resource_efficiency_from_bpt,
    formula_linear,
    formula_cobb_douglas,
    formula_harmonic,
    formula_power_mean,
)


# ─── Parametric Monotonicity ──────────────────────────────────────────────────

class TestQualityMonotonicity:
    """Axiom: S must be strictly increasing in Q for any fixed R > 0."""

    @pytest.mark.parametrize("R", [20.0, 50.0, 80.0, 100.0])
    @pytest.mark.parametrize("formula", ["linear", "cobb_douglas", "harmonic", "power_mean_2"])
    def test_monotone_increasing_in_Q(self, R, formula):
        Q_values = [10.0, 30.0, 50.0, 70.0, 90.0, 100.0]
        scores = [compute_utility(Q, R, alpha=0.70, formula=formula) for Q in Q_values]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Monotonicity violated in {formula} at R={R}: "
                f"S(Q={Q_values[i]})={scores[i]:.4f} >= S(Q={Q_values[i+1]})={scores[i+1]:.4f}"
            )


class TestResourceMonotonicity:
    """Axiom: S must be strictly increasing in R for any fixed Q > 0."""

    @pytest.mark.parametrize("Q", [20.0, 50.0, 80.0, 100.0])
    @pytest.mark.parametrize("formula", ["linear", "cobb_douglas", "harmonic", "power_mean_2"])
    def test_monotone_increasing_in_R(self, Q, formula):
        R_values = [10.0, 30.0, 50.0, 70.0, 90.0, 100.0]
        scores = [compute_utility(Q, R, alpha=0.70, formula=formula) for R in R_values]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Resource monotonicity violated in {formula} at Q={Q}: "
                f"S(R={R_values[i]})={scores[i]:.4f} >= S(R={R_values[i+1]})={scores[i+1]:.4f}"
            )


# ─── Boundedness ──────────────────────────────────────────────────────────────

class TestBoundedness:
    """Axiom: all scores must stay in [0.0, 100.0]."""

    def test_both_zero_returns_zero(self):
        assert compute_utility(0.0, 0.0, formula="linear") == 0.0
        assert compute_utility(0.0, 0.0, formula="cobb_douglas") == 0.0

    def test_max_inputs_bounded(self):
        s = compute_utility(100.0, 100.0, formula="linear")
        assert abs(s - 100.0) < 1e-9

    @pytest.mark.parametrize("formula", ["linear", "cobb_douglas", "harmonic", "power_mean_2"])
    @pytest.mark.parametrize("Q,R", [
        (100.0, 100.0), (0.0, 100.0), (100.0, 0.0),
        (50.0, 50.0), (1.0, 99.0), (99.0, 1.0),
    ])
    def test_bounded_grid(self, formula, Q, R):
        s = compute_utility(Q, R, alpha=0.70, formula=formula)
        assert 0.0 <= s <= 100.0 + 1e-9, f"{formula}: S({Q},{R})={s} out of [0,100]"


# ─── Pareto Consistency ───────────────────────────────────────────────────────

class TestParetoConsistency:
    """
    Axiom: if method A weakly dominates method B in both Q and R (strict in at least one),
    then S(A) > S(B).
    """

    @pytest.mark.parametrize("formula", ["linear", "cobb_douglas", "harmonic", "power_mean_2"])
    def test_pareto_dominant_method_scores_higher(self, formula):
        s_a = compute_utility(90.0, 80.0, alpha=0.70, formula=formula)
        s_b = compute_utility(70.0, 60.0, alpha=0.70, formula=formula)
        assert s_a > s_b, f"{formula}: Pareto violation: A({s_a:.4f}) <= B({s_b:.4f})"

    def test_quality_dominant_method_correctly_ordered(self):
        # A has much higher Q (95%), lower R (40%) vs INT2-like (55% Q, 88% R)
        s_high_q = compute_utility(95.0, 40.0, alpha=0.70, formula="linear")  # 0.7(95) + 0.3(40) = 66.5 + 12 = 78.5
        s_low_q  = compute_utility(55.0, 88.0, alpha=0.70, formula="linear")  # 0.7(55) + 0.3(88) = 38.5 + 26.4 = 64.9
        assert s_high_q > s_low_q

    def test_int2_does_not_outscore_dense_in_linear(self):
        """Dense FP16 (Q=100, R=0) must score higher than failing INT2 (Q=50, R=87.5) under α=0.70."""
        s_dense = compute_utility(100.0, 0.0, alpha=0.70, formula="linear")   # 70.0
        s_int2  = compute_utility(50.0, 87.5, alpha=0.70, formula="linear")   # 0.7(50) + 0.3(87.5) = 35 + 26.25 = 61.25
        assert s_dense > s_int2

    def test_int4_can_outscore_dense(self):
        """INT4 (near-lossless Q=95% + 75% savings) is legitimately allowed to beat Dense."""
        s_int4  = compute_utility(95.0, 75.0, alpha=0.70, formula="linear")  # 0.7(95) + 0.3(75) = 89.0
        s_dense = compute_utility(100.0, 0.0, alpha=0.70, formula="linear")  # 70.0
        assert s_int4 > s_dense


# ─── Low Quality Rejection ───────────────────────────────────────────────────

class TestLowQualityNotRewarded:
    """
    A method with very low quality must not outscore one with high quality,
    even if the low-quality method has maximum compression.
    """

    def test_q5_does_not_beat_q90(self):
        s_low = compute_utility(5.0, 100.0, alpha=0.70, formula="linear")   # 0.7(5) + 0.3(100) = 33.5
        s_high = compute_utility(90.0, 50.0, alpha=0.70, formula="linear")  # 0.7(90) + 0.3(50) = 78.0
        assert s_low < s_high

    def test_q10_does_not_beat_dense(self):
        s_low = compute_utility(10.0, 100.0, alpha=0.70, formula="linear")  # 0.7(10) + 0.3(100) = 37.0
        s_dense = compute_utility(100.0, 0.0, alpha=0.70, formula="linear") # 70.0
        assert s_low < s_dense


# ─── Alpha Configuration ─────────────────────────────────────────────────────

class TestAlphaConfigurable:
    def test_default_alpha_value(self):
        assert CRBENCH_ALPHA == 0.70

    def test_formula_id(self):
        assert CRBENCH_FORMULA_NAME == "linear"


# ─── Decomposition Correctness ────────────────────────────────────────────────

class TestUtilityDecomposition:
    def test_linear_decomposition(self):
        Q, R = 80.0, 60.0
        direct = compute_utility(Q, R, alpha=0.70, formula="linear")
        decomp = compute_utility_decomposition(Q, R, alpha=0.70, formula="linear")
        assert abs(decomp["utility"] - direct) < 1e-9
        assert abs((decomp["quality_term"] + decomp["resource_term"]) - direct) < 1e-9

    def test_decomposition_metadata(self):
        decomp = compute_utility_decomposition(80.0, 50.0, alpha=0.70, formula="linear")
        assert decomp["alpha"] == 0.70
        assert decomp["formula"] == "linear"
        assert decomp["provenance"] == "measured_query_pair"


# ─── resource_efficiency_from_bpt ─────────────────────────────────────────────

class TestResourceEfficiencyFromBPT:
    def test_dense_fp16_gives_zero_memory_component(self):
        R = resource_efficiency_from_bpt(effective_bpt=16.0, ttft_ratio=1.0)
        assert abs(R - 40.0) < 0.1

    def test_int4_gives_positive_memory_efficiency(self):
        R = resource_efficiency_from_bpt(effective_bpt=4.0, ttft_ratio=1.0)
        assert abs(R - 85.0) < 0.1

    def test_slower_ttft_reduces_efficiency(self):
        R_fast = resource_efficiency_from_bpt(effective_bpt=4.0, ttft_ratio=0.5)
        R_slow = resource_efficiency_from_bpt(effective_bpt=4.0, ttft_ratio=2.0)
        assert R_fast > R_slow


# ─── Differentiation Across Formulas ──────────────────────────────────────────

class TestDistinctMethods:
    def test_all_candidate_formulas_produce_distinct_values(self):
        methods = [
            (100.0, 0.0),    # dense_fp16
            (97.0, 50.0),    # kv_quant_int8
            (88.0, 75.0),    # kv_quant_int4
            (55.0, 87.5),    # kv_quant_int2
            (83.0, 75.0),    # snapkv
            (72.0, 87.5),    # streaming_llm
            (65.0, 62.5),    # kv_merging
            (91.0, 75.0),    # low_rank_kv
            (78.0, 75.0),    # custom_dkv
        ]
        scores = [compute_utility(Q, R, alpha=0.70, formula="linear") for Q, R in methods]
        unique_scores = set(round(s, 4) for s in scores)
        assert len(unique_scores) == len(methods)
