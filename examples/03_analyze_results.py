"""
Example 03: Statistical analysis, bootstrap confidence intervals, and hypothesis testing.
"""

from crbench.statistics.bootstrap import BootstrapEngine
from crbench.statistics.hypothesis import HypothesisEngine
from crbench.statistics.stability import StabilityEngine


def main():
    print("=== CRBench Statistical Analysis Demo ===")
    
    # 1. Bootstrap Confidence Intervals for AUQC
    int4_sample_scores = [81.5, 83.2, 79.8, 82.1, 80.4, 84.0, 78.9, 82.7, 81.1, 83.5]
    snapkv_sample_scores = [76.2, 74.8, 77.1, 75.0, 73.5, 78.2, 76.9, 75.4, 74.1, 77.0]

    boot_engine = BootstrapEngine(num_resamples=2000, ci_level=0.95)
    ci_int4 = boot_engine.compute_ci(int4_sample_scores)
    ci_snapkv = boot_engine.compute_ci(snapkv_sample_scores)

    print(f"INT4 AUQC:   {ci_int4.point_estimate:.2f} (95% CI: [{ci_int4.ci_lower:.2f}, {ci_int4.ci_upper:.2f}])")
    print(f"SnapKV AUQC: {ci_snapkv.point_estimate:.2f} (95% CI: [{ci_snapkv.ci_lower:.2f}, {ci_snapkv.ci_upper:.2f}])")

    # 2. Paired Permutation Test & Cohen's d
    hyp_engine = HypothesisEngine(num_permutations=5000)
    test_res = hyp_engine.paired_permutation_test(
        scores_a=int4_sample_scores,
        scores_b=snapkv_sample_scores,
        method_a_name="INT4_Quant",
        method_b_name="SnapKV_0.25"
    )

    print(f"\n--- Paired Comparison: {test_res.method_a} vs {test_res.method_b} ---")
    print(f"Mean Difference: {test_res.mean_diff:+.2f}")
    print(f"Cohen's d:       {test_res.cohens_d:.3f}")
    print(f"Permutation p:   {test_res.p_value:.4f}")
    print(f"Significant:     {test_res.is_statistically_significant}")

    # 3. Ranking Stability across Context Lengths (Spearman rho & Kendall tau)
    rankings_8k = [95.0, 88.0, 82.0, 75.0, 70.0]
    rankings_32k = [94.0, 86.5, 80.5, 74.0, 68.5]
    stab_res = StabilityEngine.compute_ranking_correlation(rankings_8k, rankings_32k)

    print(f"\n--- Ranking Stability (8K vs 32K Context) ---")
    print(f"Spearman's rho: {stab_res.spearman_rho:.3f} (p = {stab_res.spearman_pvalue:.4f})")
    print(f"Kendall's tau:  {stab_res.kendall_tau:.3f} (p = {stab_res.kendall_pvalue:.4f})")


if __name__ == "__main__":
    main()
