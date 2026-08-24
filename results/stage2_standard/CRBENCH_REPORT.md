# CRBench Benchmark Evaluation Report: crbench_stage2_standard

**Model Evaluated:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Evaluation Tracks:** Part 1 (Resource Score) & Part 2 (System Score)  

---

## 1. Executive Summary

CRBench evaluates long-context representation efficiency by characterizing the quality–resource tradeoff curve under strict resource constraints.

### Part 1: CRBench Resource Scores (S_res)

| Method        |   S_res (0-100) |   AUQC (2K) |   AUQC (4K) | Q@2bpt   | Q@4bpt   | Q@8bpt   | Q@16bpt   |
|---------------|-----------------|-------------|-------------|----------|----------|----------|-----------|
| dense_fp16    |            46.7 |          60 |          40 | 5.0%     | 10.0%    | 20.0%    | 40.0%     |
| kv_quant_int8 |            41.7 |          45 |          40 | 9.7%     | 19.4%    | 38.8%    | 40.0%     |
| kv_quant_int4 |             0   |           0 |           0 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |
| kv_quant_int2 |             0   |           0 |           0 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |
| snapkv        |             0   |           0 |           0 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |
| streaming_llm |             0   |           0 |           0 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |
| kv_merging    |             0   |           0 |           0 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |

### Part 2: CRBench System Scores (S_sys)

| Method        |   S_sys (0-100) |   S_res (Part 1) |   TTFT (ms) |   Decode Thru (tok/s) |   Peak VRAM (MB) | Multiplier   |
|---------------|-----------------|------------------|-------------|-----------------------|------------------|--------------|
| dense_fp16    |            51.8 |             46.7 |      3269.8 |                 189.2 |           1024   | 1.11x        |
| kv_quant_int8 |            47.3 |             41.7 |      3159.4 |                 198.3 |            528   | 1.13x        |
| kv_quant_int4 |             0   |              0   |      3420.2 |                 177.3 |            272   | 1.05x        |
| kv_quant_int2 |             0   |              0   |      3530.7 |                 171.9 |            144   | 1.02x        |
| snapkv        |             0   |              0   |      1797.5 |                 372.2 |            298.8 | 1.34x        |
| streaming_llm |             0   |              0   |      1725.8 |                 399.5 |            298.8 | 1.35x        |
| kv_merging    |             0   |              0   |      1774   |                 375.5 |            384.1 | 1.34x        |

### Context-Length Weighting Sensitivity Analysis

### Weighting Scheme Sensitivity Analysis

| Scheme Comparison | Spearman rho | Kendall tau | Max Rank Shift |
|:---|:---:|:---:|:---:|
| Logarithmic vs Uniform | 1.0000 | 1.0000 | 0 |
| Logarithmic vs Linear | 1.0000 | 1.0000 | 0 |
| Uniform vs Linear | 1.0000 | 1.0000 | 0 |

## 2. Benchmark Visualizations

### Pareto Frontier (2,048 tokens)
![Pareto Frontier (2,048 tokens)](results/stage2_standard/figures/pareto_frontier_2048.png)

### Iso-Budget Retention (2,048 tokens)
![Iso-Budget Retention (2,048 tokens)](results/stage2_standard/figures/isobudget_comparison_2048.png)

### Pareto Frontier (4,096 tokens)
![Pareto Frontier (4,096 tokens)](results/stage2_standard/figures/pareto_frontier_4096.png)

### Iso-Budget Retention (4,096 tokens)
![Iso-Budget Retention (4,096 tokens)](results/stage2_standard/figures/isobudget_comparison_4096.png)

### AUQC Context Scaling
![AUQC Context Scaling](results/stage2_standard/figures/auqc_context_scaling.png)

### Part 1 vs Part 2 System Tradeoff
![Part 1 vs Part 2 System Tradeoff](results/stage2_standard/figures/resource_vs_system_score.png)

## 3. Scientific Metric Provenance Audit

| Metric Name                      | Classification       | Measurement / Mathematical Instrument                              | Scientific Status            |
|----------------------------------|----------------------|--------------------------------------------------------------------|------------------------------|
| Raw Task Quality (Q_m)           | MEASURED             | Autoregressive generation on model with PyTorch forward hooks      | Real Measurement             |
| Dense Reference (Q_dense)        | MEASURED             | Dense uncompressed FP16 baseline on identical sample set           | Real Measurement             |
| Model-Relative Retention (q_rel) | DERIVED              | (Q_m - Q_floor) / max(0.05, Q_dense - Q_floor) * 100               | Mathematically Derived       |
| Effective Bits/Token (b_eff)     | DERIVED (ANALYTICAL) | 8 * (Algorithmic_Bytes + Scale_Metadata_Bytes) / L                 | Exact Architecture Formula   |
| AUQC Score                       | DERIVED              | Monotonic PCHIP logarithmic spline integral: 1/ln(8) * ∫ Q(e^u) du | Numerically Integrated       |
| Iso-Budget Quality (Q@B)         | DERIVED              | Standardized evaluation at 2, 4, 8, 16 bpt along empirical spline  | Interpolated / Extrapolated  |
| Part 1 Resource Score (S_res)    | DERIVED              | Logarithmically context-weighted sum: ∑ w_L AUQC(L)                | Axiomatically Derived        |
| TTFT Prefill Latency             | MEASURED             | Synchronized wall-clock prefill duration (ms)                      | Real Measurement             |
| Decode Throughput                | MEASURED             | Tokens generated per second during autoregressive decode           | Real Measurement             |
| Peak VRAM Footprint              | MEASURED / DERIVED   | Effective bpt-scaled buffer + physical device allocator tracking   | Measured & Scaled            |
| Part 2 System Score (S_sys)      | DERIVED              | S_res * (phi_ttft * phi_thru * phi_vram)^alpha                     | Constrained Utility Function |

## 4. Methodological Notes & Axioms

1. **Model-Relative Normalization**: Quality measures the exact fraction of the base model's uncompressed Dense FP16 capability retained (q_rel = (Q_m - Q_floor) / max(0.05, Q_dense - Q_floor)).
2. **Primary Resource Axis**: Effective Bits-Per-Token (b_eff in [2.0, 16.0] bpt), unifying quantization, eviction, merging, and low-rank state sizes.
3. **AUQC Integration**: Integrated in the logarithmic domain ln(b_eff) giving equal weight to successive 2x compression factors.
4. **Logarithmic Context Weighting**: Context lengths are weighted proportionally to 1 + log2(L / L_min) to reflect sequence scaling difficulty.
5. **No Arbitrary Weights**: Part 1 evaluates pure representation fidelity without runtime metrics; Part 2 models system utility via a multiplicative constrained utility function.

---
*Report generated automatically by CRBench v0.1.0.*