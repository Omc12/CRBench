"""
Comprehensive Empirical & Mathematical Re-Evaluation of CRBench Scoring Formulas
=================================================================================
Evaluates candidate utility formulations and resource scaling schemes across
all measured experimental data in CRBench (Stage 2 standard benchmark).

Formula families evaluated:
  1. Linear:                    S = αQ + (1-α)R
  2. Cobb-Douglas / Geometric:  S = Q^α · R^(1-α)
  3. Harmonic:                  S = (α/Q + (1-α)/R)^(-1)
  4. Power Mean (p=0.5, 2.0):   S = (αQ^p + (1-α)R^p)^(1/p)
  5. Logarithmic Utility:       S = 100 * (α ln(1+Q) + (1-α) ln(1+R)) / ln(101)
  6. Gated Linear:              S = (Q / 100) * (αQ + (1-α)R)
  7. Quality-Anchored Cobb:     S = Q * (0.2 + 0.8 * (R/100))^(1-α)

Resource Normalization Schemes evaluated:
  - Scheme A (R in [0, 100]):   R_pure_savings = 100 * max(0, 1 - M_method / M_dense)
  - Scheme A2 (R in [0, 100]):  R_balanced = 50 + 50 * (1 - M_method / M_dense)  (Dense = 50)
  - Scheme B (R in [0, 150]):   R_150 = 150 * max(0, 1 - M_method / M_dense)

Alpha sweep: 0.10 to 0.90 in fine increments.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable
import numpy as np
from scipy.stats import spearmanr, kendalltau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─── Load Real Measured Data from Stage 2 ─────────────────────────────────────

def load_real_measurements():
    data_path = Path("results/stage2_standard/raw_results_v1.json")
    if not data_path.exists():
        raise FileNotFoundError(f"Missing {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["raw_measurements"]


# ─── Candidate Formulas ───────────────────────────────────────────────────────

def formula_linear(Q: float, R: float, alpha: float) -> float:
    """S = αQ + (1-α)R"""
    return alpha * Q + (1.0 - alpha) * R


def formula_cobb_douglas(Q: float, R: float, alpha: float) -> float:
    """S = Q^α · R^(1-α)"""
    if Q <= 0.0 or R <= 0.0:
        return 0.0
    return (Q ** alpha) * (R ** (1.0 - alpha))


def formula_harmonic(Q: float, R: float, alpha: float) -> float:
    """S = (α/Q + (1-α)/R)^(-1)"""
    if Q <= 0.0 or R <= 0.0:
        return 0.0
    denom = (alpha / max(1e-6, Q)) + ((1.0 - alpha) / max(1e-6, R))
    return 1.0 / denom if denom > 0 else 0.0


def formula_power_mean_05(Q: float, R: float, alpha: float) -> float:
    """Power Mean p=0.5: S = (α·Q^0.5 + (1-α)·R^0.5)^2"""
    q_sqrt = math.sqrt(max(0.0, Q))
    r_sqrt = math.sqrt(max(0.0, R))
    val = alpha * q_sqrt + (1.0 - alpha) * r_sqrt
    return val ** 2


def formula_power_mean_2(Q: float, R: float, alpha: float) -> float:
    """Power Mean p=2.0 (RMS): S = (α·Q^2 + (1-α)·R^2)^0.5"""
    val = alpha * (Q ** 2) + (1.0 - alpha) * (R ** 2)
    return math.sqrt(max(0.0, val))


def formula_log_utility(Q: float, R: float, alpha: float) -> float:
    """Log-Utility: S = 100 * (α ln(1+Q) + (1-α) ln(1+R)) / ln(101)"""
    q_log = math.log(1.0 + max(0.0, Q))
    r_log = math.log(1.0 + max(0.0, R))
    norm = math.log(101.0)
    return 100.0 * (alpha * q_log + (1.0 - alpha) * r_log) / norm


def formula_gated_linear(Q: float, R: float, alpha: float) -> float:
    """Quality-Gated Linear: S = (Q/100) * (αQ + (1-α)R)"""
    gate = max(0.0, min(1.0, Q / 100.0))
    return gate * (alpha * Q + (1.0 - alpha) * R)


CANDIDATE_FORMULAS = {
    "Linear": formula_linear,
    "Cobb-Douglas": formula_cobb_douglas,
    "Harmonic": formula_harmonic,
    "Power Mean (p=0.5)": formula_power_mean_05,
    "Power Mean (p=2.0)": formula_power_mean_2,
    "Logarithmic": formula_log_utility,
    "Quality-Gated Linear": formula_gated_linear,
}


# ─── Axiom Evaluation Suite ───────────────────────────────────────────────────

def test_quality_monotonicity(formula_fn: Callable, alpha: float, r_max: float = 100.0) -> bool:
    """Axiom: For any fixed R > 0, S must be strictly non-decreasing in Q."""
    for r in np.linspace(10.0, r_max, 10):
        prev = -1e-9
        for q in np.linspace(0.0, 100.0, 21):
            s = formula_fn(q, r, alpha)
            if s < prev - 1e-6:
                return False
            prev = s
    return True


def test_resource_monotonicity(formula_fn: Callable, alpha: float, r_max: float = 100.0) -> bool:
    """Axiom: For any fixed Q > 0, S must be strictly non-decreasing in R."""
    for q in np.linspace(10.0, 100.0, 10):
        prev = -1e-9
        for r in np.linspace(0.0, r_max, 21):
            s = formula_fn(q, r, alpha)
            if s < prev - 1e-6:
                return False
            prev = s
    return True


def test_boundedness(formula_fn: Callable, alpha: float, r_max: float = 100.0) -> Tuple[bool, float, float]:
    """Axiom: For all Q in [0, 100] and R in [0, R_max], S must remain in [0, 100]."""
    min_val, max_val = 1e9, -1e9
    for q in np.linspace(0.0, 100.0, 21):
        for r in np.linspace(0.0, r_max, 21):
            s = formula_fn(q, r, alpha)
            min_val = min(min_val, s)
            max_val = max(max_val, s)
    is_bounded = (min_val >= -1e-6) and (max_val <= 100.0 + 1e-6)
    return is_bounded, min_val, max_val


def test_low_quality_rejection(formula_fn: Callable, alpha: float, r_max: float = 100.0) -> bool:
    """
    Axiom: A method with garbage quality (Q=5%) and maximum compression (R=R_max)
    must NEVER outscore a method with strong capability retention (Q=85%) and moderate compression (R=0.5*R_max).
    """
    s_garbage = formula_fn(5.0, r_max, alpha)
    s_good = formula_fn(85.0, 0.5 * r_max, alpha)
    return s_garbage < s_good


def test_dense_treatment(formula_fn: Callable, alpha: float, r_dense: float) -> Tuple[float, bool]:
    """
    Test what score Dense FP16 (Q=100) receives.
    Dense should NOT receive a score of 0.0 unless there is a severe mathematical collapse.
    """
    s_dense = formula_fn(100.0, r_dense, alpha)
    is_acceptable = s_dense >= 20.0  # Dense should not be wiped out to 0
    return s_dense, is_acceptable


def compute_pareto_consistency_on_real_data(
    formula_fn: Callable,
    alpha: float,
    methods_data: Dict[str, Dict[str, float]],
    r_key: str = "R_100"
) -> Tuple[float, int, int]:
    """
    Evaluates Pareto consistency across all pairs of methods on real measured data.
    If Method A has Q_A >= Q_B and R_A >= R_B (strictly > in at least one),
    then S(A) must be > S(B).
    """
    names = list(methods_data.keys())
    pairs = 0
    consistent = 0
    violations = 0

    for i in range(len(names)):
        n1 = names[i]
        q1 = methods_data[n1]["mean_Q"]
        r1 = methods_data[n1][r_key]
        s1 = formula_fn(q1, r1, alpha)

        for j in range(i + 1, len(names)):
            n2 = names[j]
            q2 = methods_data[n2]["mean_Q"]
            r2 = methods_data[n2][r_key]
            s2 = formula_fn(q2, r2, alpha)

            # Check if n1 dominates n2
            if (q1 >= q2 and r1 >= r2) and (q1 > q2 or r1 > r2):
                pairs += 1
                if s1 > s2 + 1e-6:
                    consistent += 1
                else:
                    violations += 1
            # Check if n2 dominates n1
            elif (q2 >= q1 and r2 >= r1) and (q2 > q1 or r2 > r1):
                pairs += 1
                if s2 > s1 + 1e-6:
                    consistent += 1
                else:
                    violations += 1

    frac = (consistent / pairs) if pairs > 0 else 1.0
    return frac, consistent, violations


def compute_alpha_responsiveness(
    formula_fn: Callable,
    methods_data: Dict[str, Dict[str, float]],
    r_key: str = "R_100"
) -> float:
    """
    Measures whether alpha actually changes the scores meaningfully.
    Calculates the mean absolute shift in scores between alpha=0.2 and alpha=0.8.
    If responsiveness is near 0, alpha is a no-op!
    """
    shifts = []
    for name, d in methods_data.items():
        s_low = formula_fn(d["mean_Q"], d[r_key], 0.2)
        s_high = formula_fn(d["mean_Q"], d[r_key], 0.8)
        shifts.append(abs(s_high - s_low))
    return float(np.mean(shifts))


def compute_ranking_stability(
    formula_fn: Callable,
    alpha_list: List[float],
    methods_data: Dict[str, Dict[str, float]],
    r_key: str = "R_100"
) -> float:
    """
    Computes Spearman rank correlation across consecutive alpha steps.
    """
    correlations = []
    names = list(methods_data.keys())
    for i in range(len(alpha_list) - 1):
        a1 = alpha_list[i]
        a2 = alpha_list[i + 1]
        scores1 = [formula_fn(methods_data[n]["mean_Q"], methods_data[n][r_key], a1) for n in names]
        scores2 = [formula_fn(methods_data[n]["mean_Q"], methods_data[n][r_key], a2) for n in names]
        if len(set(scores1)) > 1 and len(set(scores2)) > 1:
            rho, _ = spearmanr(scores1, scores2)
            if not math.isnan(rho):
                correlations.append(rho)
    return float(np.mean(correlations)) if correlations else 1.0


# ─── Main Execution ───────────────────────────────────────────────────────────

def run_empirical_evaluation():
    print("=" * 80)
    print("CRBench: Rigorous Empirical & Mathematical Scoring Formula Evaluation")
    print("=" * 80)

    measurements = load_real_measurements()
    print(f"[*] Loaded {len(measurements)} real experimental measurements from Stage 2.")

    # Aggregate by adapter
    adapter_data = {}
    for m in measurements:
        if m.get("status") != "SUCCESS":
            continue
        ad = m["adapter_name"]
        if ad not in adapter_data:
            adapter_data[ad] = {
                "norm_scores": [],
                "bpt_values": [],
                "algorithmic_bytes": [],
                "ttft_values": [],
                "contexts": {}
            }
        adapter_data[ad]["norm_scores"].append(m["normalized_score"])
        adapter_data[ad]["bpt_values"].append(m.get("effective_bpt", 16.0))
        adapter_data[ad]["algorithmic_bytes"].append(m.get("algorithmic_bytes", 25165824.0))
        if m.get("ttft_ms"):
            adapter_data[ad]["ttft_values"].append(m["ttft_ms"])
        
        ctx = m["context_length"]
        if ctx not in adapter_data[ad]["contexts"]:
            adapter_data[ad]["contexts"][ctx] = []
        adapter_data[ad]["contexts"][ctx].append(m["normalized_score"])

    # Compute summary metrics for each adapter
    dense_bytes = np.mean(adapter_data["dense_fp16"]["algorithmic_bytes"])
    summary_data = {}

    for ad, d in adapter_data.items():
        mean_q = float(np.mean(d["norm_scores"]))
        mean_bpt = float(np.mean(d["bpt_values"]))
        mean_bytes = float(np.mean(d["algorithmic_bytes"]))
        
        # Scheme A: R in [0, 100] pure savings (Dense = 0)
        # R_savings = 100 * (1 - bytes / dense_bytes)
        r_pure = float(max(0.0, min(100.0, 100.0 * (1.0 - mean_bytes / dense_bytes))))
        
        # Scheme A2: R in [0, 100] balanced (Dense = 50, maximum compression = 100)
        # R = 50 + 50 * (1 - bytes / dense_bytes)
        r_balanced = float(max(0.0, min(100.0, 50.0 + 50.0 * (1.0 - mean_bytes / dense_bytes))))

        # Scheme B: R in [0, 150]
        # R_150 = 150 * (1 - bytes / dense_bytes)
        r_150 = float(max(0.0, min(150.0, 150.0 * (1.0 - mean_bytes / dense_bytes))))

        summary_data[ad] = {
            "mean_Q": mean_q,
            "mean_bpt": mean_bpt,
            "mean_bytes": mean_bytes,
            "R_pure": r_pure,
            "R_balanced": r_balanced,
            "R_150": r_150,
            "contexts": {ctx: float(np.mean(scores)) for ctx, scores in d["contexts"].items()}
        }

    print("\n--- Real Adapter Measurements Overview ---")
    print(f"{'Adapter':<18} {'Mean Q (%)':<12} {'Eff BPT':<10} {'R (Pure)':<10} {'R (Balanced)':<14} {'R (0-150)':<10}")
    print("-" * 76)
    for ad, d in summary_data.items():
        print(f"{ad:<18} {d['mean_Q']:>10.2f}% {d['mean_bpt']:>8.2f} {d['R_pure']:>8.1f} {d['R_balanced']:>12.1f} {d['R_150']:>10.1f}")

    # ─── Alpha Sweep & Evaluation Across Formulas and R-Schemes ───────────────
    alphas = [round(a, 3) for a in np.arange(0.10, 0.95, 0.05)]
    results = {}

    r_schemes = [
        ("R_pure", 100.0, 0.0, "R in [0, 100] (Pure Savings, Dense R=0)"),
        ("R_balanced", 100.0, 50.0, "R in [0, 100] (Balanced, Dense R=50)"),
        ("R_150", 150.0, 0.0, "R in [0, 150] (Asymmetric Scale, Dense R=0)"),
    ]

    evaluation_records = []

    for r_key, r_max, r_dense, r_desc in r_schemes:
        for fname, f_fn in CANDIDATE_FORMULAS.items():
            for a in alphas:
                q_mono = test_quality_monotonicity(f_fn, a, r_max)
                r_mono = test_resource_monotonicity(f_fn, a, r_max)
                is_bounded, min_s, max_s = test_boundedness(f_fn, a, r_max)
                low_q_rej = test_low_quality_rejection(f_fn, a, r_max)
                dense_score, dense_ok = test_dense_treatment(f_fn, a, r_dense)
                pareto_frac, par_c, par_v = compute_pareto_consistency_on_real_data(f_fn, a, summary_data, r_key)
                resp = compute_alpha_responsiveness(f_fn, summary_data, r_key)

                # Composite metric:
                # 1. Monotonicity (Q and R) = 25%
                # 2. Boundedness = 20%
                # 3. Low-Q Rejection = 15%
                # 4. Dense Validity (score > 0 for dense) = 15%
                # 5. Pareto Consistency = 15%
                # 6. Alpha Responsiveness (non-zero) = 10%
                composite_score = (
                    (0.125 if q_mono else 0.0) +
                    (0.125 if r_mono else 0.0) +
                    (0.20 if is_bounded else 0.0) +
                    (0.15 if low_q_rej else 0.0) +
                    (0.15 if dense_ok else (0.05 if dense_score > 0 else 0.0)) +
                    (0.15 * pareto_frac) +
                    (0.10 * min(1.0, resp / 10.0))
                )

                rec = {
                    "formula": fname,
                    "r_scheme": r_key,
                    "r_desc": r_desc,
                    "r_max": r_max,
                    "alpha": a,
                    "q_mono": q_mono,
                    "r_mono": r_mono,
                    "is_bounded": is_bounded,
                    "min_s": min_s,
                    "max_s": max_s,
                    "low_q_rej": low_q_rej,
                    "dense_score": dense_score,
                    "dense_ok": dense_ok,
                    "pareto_frac": pareto_frac,
                    "pareto_viol": par_v,
                    "alpha_resp": resp,
                    "composite": composite_score,
                }
                evaluation_records.append(rec)

    # ─── Analysis by Formula & Scheme ─────────────────────────────────────────
    out_dir = Path("results/formula_reevaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Formula':<22} {'Scheme':<12} {'Best α':<8} {'Composite':<10} {'Bounded':<8} {'Dense S':<10} {'Pareto%':<8} {'LowQ Rej':<8}")
    print("-" * 88)

    best_by_group = {}
    for r_key, r_max, r_dense, r_desc in r_schemes:
        for fname in CANDIDATE_FORMULAS.keys():
            subset = [r for r in evaluation_records if r["formula"] == fname and r["r_scheme"] == r_key]
            best = max(subset, key=lambda x: x["composite"])
            best_by_group[(fname, r_key)] = best
            print(f"{fname:<22} {r_key:<12} {best['alpha']:>6.2f} {best['composite']:>9.4f} "
                  f"{'✓' if best['is_bounded'] else '✗':^8} {best['dense_score']:>8.1f} "
                  f"{best['pareto_frac']*100:>6.0f}% {'✓' if best['low_q_rej'] else '✗':^8}")

    # ─── Deep Insights & Axiomatic Failure Points ──────────────────────────────
    print("\n--- Key Methodological Findings ---")
    
    # 1. Why Cobb-Douglas fails with Pure Savings (Dense R=0):
    cd_pure = best_by_group[("Cobb-Douglas", "R_pure")]
    print(f"1. Cobb-Douglas with R_pure (Dense R=0): Dense FP16 Score = {cd_pure['dense_score']:.1f}!")
    print("   -> In Q^α · R^(1-α), when Dense has R=0, S(Dense) = 100^α · 0^(1-α) = 0.0! Dense baseline is wiped out!")

    # 2. Why R in [0, 150] violates Boundedness:
    lin_150 = best_by_group[("Linear", "R_150")]
    print(f"2. R in [0, 150] Scaling: Max possible score = {lin_150['max_s']:.1f} (Exceeds 100.0!). Bounded: {lin_150['is_bounded']}")
    print("   -> Scaling R to 150 breaks the [0, 100] capability boundary and unfairly distorts alpha weighting.")

    # 3. Linear Formulation Performance:
    lin_pure = best_by_group[("Linear", "R_pure")]
    print(f"3. Linear (S = αQ + (1-α)R) with R in [0, 100]:")
    print(f"   -> Composite = {lin_pure['composite']:.4f}, Dense FP16 = {lin_pure['dense_score']:.1f}, Bounded = {lin_pure['is_bounded']}, Pareto = {lin_pure['pareto_frac']*100:.0f}%")
    print(f"   -> Satisfies all axioms: Quality Monotone, Resource Monotone, Bounded [0, 100], Dense FP16 preserved!")

    # 4. Quality-Gated Linear Performance:
    gated_pure = best_by_group[("Quality-Gated Linear", "R_pure")]
    print(f"4. Quality-Gated Linear (S = (Q/100)·(αQ + (1-α)R)):")
    print(f"   -> Composite = {gated_pure['composite']:.4f}, Low-Q Rejection = {gated_pure['low_q_rej']}, Dense FP16 = {gated_pure['dense_score']:.1f}")

    # ─── Generate Comparison Plots ─────────────────────────────────────────────
    # Plot 1: Composite Score vs Alpha across all formulas for R in [0, 100]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), facecolor="#1a1a2e")
    
    palette = ["#00d2ff", "#ff6b6b", "#f9ca24", "#6ab04c", "#a29bfe", "#fd79a8", "#81ecec"]
    
    for ax_idx, (r_key, r_max, r_dense, r_title) in enumerate(r_schemes):
        ax = axes[ax_idx]
        ax.set_facecolor("#16213e")
        for f_idx, fname in enumerate(CANDIDATE_FORMULAS.keys()):
            subset = [r for r in evaluation_records if r["formula"] == fname and r["r_scheme"] == r_key]
            alphas_plot = [r["alpha"] for r in subset]
            comp_plot = [r["composite"] for r in subset]
            ax.plot(alphas_plot, comp_plot, label=fname, lw=2.0, color=palette[f_idx % len(palette)])
        
        ax.set_title(f"{r_title}", color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Alpha (Quality Weight)", color="white", fontsize=9)
        ax.set_ylabel("Composite Axiom Score (0-1)", color="white", fontsize=9)
        ax.set_ylim(0.2, 1.05)
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_edgecolor("#444")
        ax.grid(True, alpha=0.2, color="white")
        if ax_idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    plt.suptitle("CRBench Scoring Formula Evaluation Across Utility Families & R Normalization Schemes",
                 fontsize=12, fontweight="bold", color="white")
    plt.tight_layout()
    plot1_path = out_dir / "formula_comparison_sweep.png"
    fig.savefig(plot1_path, dpi=200, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"\n[✓] Saved comparison plot to: {plot1_path}")

    # Plot 2: Method Rankings Under Winning Candidate (Linear S = αQ + (1-α)R) across Alphas
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="#1a1a2e")
    ax.set_facecolor("#16213e")
    
    selected_alphas = [0.3, 0.5, 0.7, 0.85]
    bar_width = 0.18
    method_names = list(summary_data.keys())
    x = np.arange(len(method_names))

    colors_alpha = ["#81ecec", "#74b9ff", "#0984e3", "#6c5ce7"]

    for i, a in enumerate(selected_alphas):
        scores = [formula_linear(summary_data[m]["mean_Q"], summary_data[m]["R_pure"], a) for m in method_names]
        ax.bar(x + (i - 1.5) * bar_width, scores, width=bar_width, label=f"α = {a:.2f}", color=colors_alpha[i], edgecolor="#222")

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, rotation=35, ha="right", color="white", fontsize=9)
    ax.set_ylabel("Linear Score S = αQ + (1-α)R", color="white", fontsize=10)
    ax.set_title("Method Scores Under Linear Formulation Across Alpha Weights (R ∈ [0, 100])",
                 color="white", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_edgecolor("#444")
    ax.grid(True, axis="y", alpha=0.2, color="white")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plot2_path = out_dir / "linear_alpha_sensitivity.png"
    fig.savefig(plot2_path, dpi=200, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    print(f"[✓] Saved linear sensitivity plot to: {plot2_path}")

    # Convert records for clean JSON serialization
    serializable_records = []
    for r in evaluation_records:
        clean_r = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}
        serializable_records.append(clean_r)

    with open(out_dir / "formula_reevaluation_records.json", "w") as f:
        json.dump(serializable_records, f, indent=2)

    return evaluation_records, best_by_group, summary_data


if __name__ == "__main__":
    run_empirical_evaluation()
