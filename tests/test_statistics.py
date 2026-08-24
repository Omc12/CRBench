"""
Unit tests for Statistical Engine (Bootstrap, Permutation Tests, Stability).
"""

import pytest
from crbench.statistics.bootstrap import BootstrapEngine
from crbench.statistics.hypothesis import HypothesisEngine
from crbench.statistics.stability import StabilityEngine


def test_bootstrap_engine():
    engine = BootstrapEngine(num_resamples=500, ci_level=0.95)
    data = [10.0, 12.0, 11.0, 10.5, 11.5, 9.5, 12.5]
    ci_res = engine.compute_ci(data)
    
    assert ci_res.ci_lower <= ci_res.point_estimate <= ci_res.ci_upper
    assert ci_res.std_error > 0.0


def test_hypothesis_paired_permutation():
    engine = HypothesisEngine(num_permutations=2000)
    scores_a = [90.0, 92.0, 89.0, 95.0, 91.0, 93.0, 90.5, 94.0]
    scores_b = [70.0, 72.0, 68.0, 75.0, 71.0, 69.0, 73.0, 70.0]
    
    res = engine.paired_permutation_test(scores_a, scores_b)
    assert res.mean_diff > 18.0
    assert res.p_value < 0.05
    assert res.is_statistically_significant is True
    assert res.cohens_d > 1.0


def test_ranking_stability():
    l1 = [10.0, 20.0, 30.0, 40.0]
    l2 = [11.0, 22.0, 31.0, 42.0]
    
    res = StabilityEngine.compute_ranking_correlation(l1, l2)
    assert res.spearman_rho > 0.95
    assert res.kendall_tau > 0.95
