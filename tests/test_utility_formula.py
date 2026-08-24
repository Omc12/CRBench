"""
Unit tests for the CRBench canonical utility formula.
Tests verify the frozen Cobb-Douglas formula (α=0.70) satisfies all scientific axioms.
"""
import math
import pytest
from crbench.scoring.utility import (
    CRBENCH_ALPHA,
    CRBENCH_FORMULA_NAME,
    compute_utility,
    compute_utility_decomposition,
    resource_efficiency_from_bpt,
)


# ─── Parametric monotonicity ──────────────────────────────────────────────────

class TestQualityMonotonicity:
    """Axiom: S must be strictly increasing in Q for any fixed R > 0."""

    @pytest.mark.parametrize("R", [20.0, 50.0, 80.0, 100.0])
    def test_monotone_increasing_in_Q(self, R):
        Q_values = [10.0, 30.0, 50.0, 70.0, 90.0, 100.0]
        scores = [compute_utility(Q, R) for Q in Q_values]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Monotonicity violated at R={R}: "
                f"S(Q={Q_values[i]})={scores[i]:.4f} >= S(Q={Q_values[i+1]})={scores[i+1]:.4f}"
            )


class TestResourceMonotonicity:
    """Axiom: S must be strictly increasing in R for any fixed Q > 0."""

    @pytest.mark.parametrize("Q", [20.0, 50.0, 80.0, 100.0])
    def test_monotone_increasing_in_R(self, Q):
        R_values = [10.0, 30.0, 50.0, 70.0, 90.0, 100.0]
        scores = [compute_utility(Q, R) for R in R_values]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1], (
                f"Resource monotonicity violated at Q={Q}: "
                f"S(R={R_values[i]})={scores[i]:.4f} >= S(R={R_values[i+1]})={scores[i+1]:.4f}"
            )


# ─── Boundedness ──────────────────────────────────────────────────────────────

class TestBoundedness:
    """Axiom: all scores must stay in [0.0, 100.0]."""

    def test_zero_quality_returns_zero(self):
        assert compute_utility(0.0, 80.0) == 0.0

    def test_zero_resource_returns_zero(self):
        assert compute_utility(80.0, 0.0) == 0.0

    def test_both_zero_returns_zero(self):
        assert compute_utility(0.0, 0.0) == 0.0

    def test_max_inputs_bounded(self):
        s = compute_utility(100.0, 100.0)
        assert s <= 100.0 + 1e-9, f"S(100, 100)={s} exceeds 100"

    @pytest.mark.parametrize("Q,R", [
        (100.0, 100.0), (0.0, 100.0), (100.0, 0.0),
        (50.0, 50.0), (1.0, 99.0), (99.0, 1.0),
    ])
    def test_bounded_grid(self, Q, R):
        s = compute_utility(Q, R)
        assert 0.0 <= s <= 100.0 + 1e-9, f"S({Q},{R})={s} out of [0,100]"


# ─── Pareto consistency ───────────────────────────────────────────────────────

class TestParetoConsistency:
    """
    Axiom: if method A weakly dominates method B in both Q and R (strict in at least one),
    then S(A) > S(B).
    """

    def test_pareto_dominant_method_scores_higher(self):
        # A dominates B: higher Q AND higher R
        s_a = compute_utility(90.0, 80.0)
        s_b = compute_utility(70.0, 60.0)
        assert s_a > s_b, f"Pareto violation: A({s_a:.4f}) <= B({s_b:.4f})"

    def test_quality_dominant_method_correctly_ordered(self):
        # A has much higher Q, slightly lower R — still Pareto-non-comparable, but quality
        # weighted at 0.70 should give A the higher score for large enough Q gap
        s_high_q = compute_utility(95.0, 40.0)
        s_low_q  = compute_utility(55.0, 88.0)  # INT2-like method
        assert s_high_q > s_low_q, (
            f"High quality ({s_high_q:.2f}) should outscore INT2-like low quality ({s_low_q:.2f}) at α=0.70"
        )

    def test_int2_does_not_outscore_dense(self):
        """Dense FP16 (Q=100) must score higher than INT2 (Q=55) despite INT2's resource advantage."""
        s_dense = compute_utility(100.0, 40.0)   # Dense: perfect quality, poor resource
        s_int2  = compute_utility(55.0, 88.0)    # INT2: poor quality, great resource
        assert s_dense > s_int2, (
            f"Dense FP16 ({s_dense:.2f}) must outscore INT2 ({s_int2:.2f}) at α=0.70"
        )

    def test_int4_can_outscore_dense(self):
        """INT4 (near-lossless quality + better resource) is legitimately allowed to beat dense."""
        s_int4  = compute_utility(88.0, 82.0)
        s_dense = compute_utility(100.0, 40.0)
        assert s_int4 > s_dense, (
            f"INT4 ({s_int4:.2f}) should outscore dense ({s_dense:.2f}) — "
            f"good quality AND much better resource efficiency"
        )


# ─── Low quality not rewarded ─────────────────────────────────────────────────

class TestLowQualityNotRewarded:
    """
    A method with very low quality must not outscore one with high quality,
    even if the low-quality method has perfect resource efficiency.
    """

    def test_q5_does_not_beat_q90(self):
        s_low = compute_utility(5.0, 100.0)
        s_high = compute_utility(90.0, 50.0)
        assert s_low < s_high, (
            f"Low-quality method ({s_low:.2f}) must not outscore high-quality ({s_high:.2f})"
        )

    def test_q10_does_not_beat_q80_at_any_resource(self):
        for R_low in [80.0, 95.0, 100.0]:
            for R_high in [30.0, 50.0]:
                if R_low > R_high:
                    s_low  = compute_utility(10.0, R_low)
                    s_high = compute_utility(80.0, R_high)
                    assert s_low < s_high, (
                        f"Q=10,R={R_low} ({s_low:.2f}) beat Q=80,R={R_high} ({s_high:.2f})"
                    )


# ─── Alpha sensitivity ────────────────────────────────────────────────────────

class TestAlphaFrozen:
    """Verify that the frozen alpha equals the scientifically selected value."""

    def test_frozen_alpha_value(self):
        assert CRBENCH_ALPHA == 0.70, (
            f"CRBENCH_ALPHA must be frozen at 0.70; found {CRBENCH_ALPHA}"
        )

    def test_formula_id(self):
        assert CRBENCH_FORMULA_NAME == "F2_cobb_douglas_geometric"


# ─── Decomposition correctness ────────────────────────────────────────────────

class TestUtilityDecomposition:
    def test_decomposition_utility_matches_compute_utility(self):
        Q, R = 80.0, 60.0
        direct = compute_utility(Q, R)
        decomp = compute_utility_decomposition(Q, R)
        assert abs(decomp["utility"] - direct) < 1e-9

    def test_decomposition_components_product(self):
        Q, R = 75.0, 65.0
        decomp = compute_utility_decomposition(Q, R)
        reconstructed = decomp["quality_component"] * decomp["resource_component"]
        assert abs(reconstructed - decomp["utility"]) < 1e-9

    def test_decomposition_provenance_field(self):
        decomp = compute_utility_decomposition(80.0, 50.0)
        assert decomp["provenance"] == "frozen_empirical_selection_2026_08_24"

    def test_alpha_in_decomposition(self):
        decomp = compute_utility_decomposition(80.0, 60.0)
        assert decomp["alpha"] == CRBENCH_ALPHA


# ─── resource_efficiency_from_bpt ─────────────────────────────────────────────

class TestResourceEfficiencyFromBPT:
    def test_dense_fp16_gives_zero_memory_component(self):
        # bpt = 16 = baseline → memory efficiency = 0%
        R = resource_efficiency_from_bpt(effective_bpt=16.0, ttft_ratio=1.0)
        # pure memory term = 0, pure latency term = 1.0 → weighted: 0.60*0 + 0.40*100 = 40
        assert abs(R - 40.0) < 0.1, f"Expected ~40.0, got {R}"

    def test_int4_gives_positive_memory_efficiency(self):
        R = resource_efficiency_from_bpt(effective_bpt=4.0, ttft_ratio=1.0)
        # memory eff = (1-4/16)*100 = 75%, latency = 100% → 0.60*75 + 0.40*100 = 45+40 = 85
        assert abs(R - 85.0) < 0.1, f"Expected ~85.0, got {R}"

    def test_slower_ttft_reduces_efficiency(self):
        R_fast = resource_efficiency_from_bpt(effective_bpt=4.0, ttft_ratio=0.5)
        R_slow = resource_efficiency_from_bpt(effective_bpt=4.0, ttft_ratio=2.0)
        assert R_fast > R_slow, "Faster TTFT should give higher resource efficiency"

    def test_non_negative(self):
        R = resource_efficiency_from_bpt(effective_bpt=16.0, ttft_ratio=5.0)
        assert R >= 0.0


# ─── Two distinct methods produce different scores ────────────────────────────

class TestDistinctMethods:
    """Regression test: two genuinely different methods must produce different scores."""

    def test_snapkv_vs_streaming_llm_different_scores(self):
        s_snap = compute_utility(83.0, 85.0)
        s_stream = compute_utility(72.0, 92.5)
        assert s_snap != s_stream, (
            f"SnapKV and StreamingLLM have identical scores {s_snap:.4f} — "
            f"quality difference (83 vs 72) must create score difference"
        )

    def test_dense_vs_int8_different_scores(self):
        s_dense = compute_utility(100.0, 40.0)
        s_int8 = compute_utility(97.0, 69.1)
        assert s_dense != s_int8

    def test_all_methods_produce_distinct_scores(self):
        # All 9 methods from the benchmark should have unique scores
        methods = [
            (100.0, 40.0),   # dense_fp16
            (97.0, 69.1),    # kv_quant_int8
            (88.0, 82.0),    # kv_quant_int4
            (55.0, 88.1),    # kv_quant_int2
            (83.0, 85.0),    # snapkv
            (72.0, 92.5),    # streaming_llm
            (65.0, 85.0),    # kv_merging
            (91.0, 82.1),    # low_rank_kv
            (78.0, 81.2),    # custom_dkv
        ]
        scores = [compute_utility(Q, R) for Q, R in methods]
        unique_scores = set(round(s, 6) for s in scores)
        assert len(unique_scores) == len(methods), (
            f"Expected {len(methods)} unique scores, got {len(unique_scores)}: {scores}"
        )
