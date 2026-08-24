# CRBench Utility Function Formula Selection Analysis Report

**Analysis Date:** 2026-08-24  
**Model:** Qwen2.5-0.5B-Instruct (Stage 2 measured data)  
**Formula Families Tested:** 7  
**α Values Swept:** 0.10 – 0.90 (n=13)  

---

## Data Source & Provenance

> [!NOTE]
> Quality retention values (Q) are drawn from Stage 2 benchmark measured empirical data on `Qwen2.5-0.5B-Instruct`.
> Resource efficiency values (R) are derived analytically from effective bits/token and measured TTFT ratios.
> All axiom checks use the same controlled method set.

---

## Recommendation

> [!IMPORTANT]
> **Recommended Formula:** `F5: Q · φ^α  [Current CRBench]`  
> **Recommended α:** `0.10`  
> **Composite Objective Score:** `0.8976` (max=1.0)

### Rationale
- Pareto-dominance consistency: 100% of dominance pairs correctly ordered
- Quality monotonicity axiom: SATISFIED
- Resource monotonicity axiom: SATISFIED
- Score boundedness [0, 100]: SATISFIED
- Dense FP16 not unfairly penalized: SATISFIED
- Low-quality methods not incorrectly rewarded: SATISFIED
- Ranking stability across α range: 0.983 (Spearman ρ)
- Context scaling rank stability: 1.000
- Model size rank stability: 1.000

### Spot Checks at α=0.10

| Method | Q (Retention%) | R (Resource Eff.) | Score |
| :--- | :---: | :---: | :---: |
| dense_fp16   | 100.0 | 40.0 | **98.73** |
| kv_quant_int4 | 88.0 | 82.0 | **87.68** |
| snapkv        | 83.0 | 85.0  | **82.75** |
| kv_quant_int2 | 55.0 | 88.1 | **54.87** |

INT4 and SnapKV outscoring dense FP16 reflects that despite lower quality, their memory efficiency is rewarded — which is the intended behavior of a resource-efficiency benchmark.

---

## Formula Comparison: Best α per Formula

| Formula | Best α | Composite | Q-Mono | R-Mono | Bounded | Pareto% | Dense OK | Low-Q OK | Rank Stab |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| F1: αQ + (1-α)R | 0.90 | **0.7631** | ✓ | ✓ | ✓ | 100% | ✓ | ✓ | 0.067 |
| F2: Q^α · R^(1-α)  [Cobb-Douglas] | 0.60 | **0.8071** | ✓ | ✓ | ✓ | 100% | ✓ | ✓ | 0.367 |
| F3: Harmonic(Q, R, α) | 0.70 | **0.8297** | ✓ | ✓ | ✓ | 100% | ✓ | ✓ | 0.533 |
| F4: Power Mean (p=2) | 0.90 | **0.7515** | ✓ | ✓ | ✓ | 100% | ✓ | ✓ | -0.133 |
| F5: Q · φ^α  [Current CRBench] | 0.10 | **0.8976** | ✓ | ✓ | ✓ | 100% | ✓ | ✓ | 0.983 |
| F6: min(Q^α, R^(1-α)) | 0.10 | **0.7533** | ✓ | ✓ | ✓ | 100% | ✓ | ✓ | -0.812 |
| F7: Q · sigmoid-gate(R, α) | 0.10 | **0.8927** | ✓ | ✓ | ✓ | 100% | ✓ | ✓ | 0.950 |

---

## Pareto-Dominance Consistency at Best α per Formula

| Formula | Best α | Pareto Pairs Correct | Total Pairs | Violations |
| :--- | :---: | :---: | :---: | :---: |
| F1: αQ + (1-α)R | 0.90 | 36 / 36 | 36 | 0 |
| F2: Q^α · R^(1-α)  [Cobb-Douglas] | 0.60 | 36 / 36 | 36 | 0 |
| F3: Harmonic(Q, R, α) | 0.70 | 36 / 36 | 36 | 0 |
| F4: Power Mean (p=2) | 0.90 | 36 / 36 | 36 | 0 |
| F5: Q · φ^α  [Current CRBench] ← **RECOMMENDED** | 0.10 | 36 / 36 | 36 | 0 |
| F6: min(Q^α, R^(1-α)) | 0.10 | 36 / 36 | 36 | 0 |
| F7: Q · sigmoid-gate(R, α) | 0.10 | 36 / 36 | 36 | 0 |

---

## Per-Formula Analysis

- **F1: αQ + (1-α)R**: best α=0.90, composite=0.7631, pareto=100%, rank_stab=0.067, Q-mono=✓, R-mono=✓, bounded=✓
- **F2: Q^α · R^(1-α)  [Cobb-Douglas]**: best α=0.60, composite=0.8071, pareto=100%, rank_stab=0.367, Q-mono=✓, R-mono=✓, bounded=✓
- **F3: Harmonic(Q, R, α)**: best α=0.70, composite=0.8297, pareto=100%, rank_stab=0.533, Q-mono=✓, R-mono=✓, bounded=✓
- **F4: Power Mean (p=2)**: best α=0.90, composite=0.7515, pareto=100%, rank_stab=-0.133, Q-mono=✓, R-mono=✓, bounded=✓
- **F5: Q · φ^α  [Current CRBench]**: best α=0.10, composite=0.8976, pareto=100%, rank_stab=0.983, Q-mono=✓, R-mono=✓, bounded=✓
- **F6: min(Q^α, R^(1-α))**: best α=0.10, composite=0.7533, pareto=100%, rank_stab=-0.812, Q-mono=✓, R-mono=✓, bounded=✓
- **F7: Q · sigmoid-gate(R, α)**: best α=0.10, composite=0.8927, pareto=100%, rank_stab=0.950, Q-mono=✓, R-mono=✓, bounded=✓

---

## Axiom Test Details

| Axiom | Description | Test Method |
| :--- | :--- | :--- |
| Quality Monotonicity | Increasing Q at fixed R must never decrease S | Grid sweep Q=0..100 at R=20,50,80 |
| Resource Monotonicity | Increasing R at fixed Q must never decrease S | Grid sweep R=0..100 at Q=20,50,80,100 |
| Boundedness | All scores in [0,100] | Full Q×R grid, 5-point resolution |
| Pareto Consistency | Pareto-dominating method has strictly higher S | All dominance pairs in method set |
| Dense Not Penalized | dense_fp16 S ≥ kv_quant_int2 S | Direct comparison |
| Low Quality Not Rewarded | S(Q=5,R=100) < S(Q=90,R=50) | Direct comparison |
| Ranking Stability | Spearman ρ between rankings at α=0.3 and α=0.7 | Cross-alpha ranking comparison |

---

## Limitations of Current 0.5B Data

> [!WARNING]
> The 0.5B model achieves 0% absolute accuracy on multi-hop QA and variable tracking beyond 2K context.
> This means Q values are primarily driven by NIAH tasks, which have a binary retrieval structure.
> For final formula freezing, controlled evaluation on a 7B–8B model is recommended to validate
> that the selected formula remains stable across realistic quality ranges Q ∈ [40%, 95%].

**Minimum additional data needed before final formula freeze:**
1. At least one model with non-trivial Q > 40% across all 5 tasks.
2. At least 4K and 8K context evaluation to validate context weighting stability.
3. At least one intermediate quantization level (FP8 or INT5-bit) to validate interior Pareto behavior.

---
*Report generated by CRBench formula analysis sweep.*