# CRBench Benchmark Evaluation Report: crbench_stage1_quick

**Model Evaluated:** `Qwen/Qwen2.5-0.5B-Instruct`  
**Evaluation Tracks:** Part 1 (Resource Score) & Part 2 (System Score)  

---

## 1. Executive Summary

CRBench evaluates long-context representation efficiency by characterizing the quality–resource tradeoff curve under strict resource constraints.

### Part 1: CRBench Resource Scores (S_res)

| Method        |   S_res (0-100) |   AUQC (2K) |   AUQC (4K) | Q@2bpt   | Q@4bpt   | Q@8bpt   | Q@16bpt   |
|---------------|-----------------|-------------|-------------|----------|----------|----------|-----------|
| dense_fp16    |             100 |         100 |         100 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |
| kv_quant_int8 |             100 |         100 |         100 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |
| kv_quant_int4 |             100 |         100 |         100 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |
| snapkv_0.25   |             100 |         100 |         100 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |
| custom_dkv    |             100 |         100 |         100 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |

### Part 2: CRBench System Scores (S_sys)

| Method        |   S_sys (0-100) |   S_res (Part 1) |   TTFT (ms) |   Decode Thru (tok/s) |   Peak VRAM (MB) | Multiplier   |
|---------------|-----------------|------------------|-------------|-----------------------|------------------|--------------|
| custom_dkv    |            38.9 |              100 |      6453.9 |                 141   |             1024 | 0.39x        |
| snapkv_0.25   |            37.9 |              100 |      6906   |                 132.6 |             1024 | 0.38x        |
| dense_fp16    |            36.8 |              100 |      7463.2 |                 131.1 |             1024 | 0.37x        |
| kv_quant_int4 |            34.2 |              100 |      9013.8 |                 123.2 |             1024 | 0.34x        |
| kv_quant_int8 |            23.7 |              100 |     23222.8 |                 113.2 |             1024 | 0.24x        |

### Context-Length Weighting Sensitivity Analysis

### Weighting Scheme Sensitivity Analysis

| Scheme Comparison | Spearman rho | Kendall tau | Max Rank Shift |
|:---|:---:|:---:|:---:|
| Logarithmic vs Uniform | 1.0000 | 1.0000 | 0 |
| Logarithmic vs Linear | 1.0000 | 1.0000 | 0 |
| Uniform vs Linear | 1.0000 | 1.0000 | 0 |

## 2. Benchmark Visualizations

### Pareto Frontier (2,048 tokens)
![Pareto Frontier (2,048 tokens)](results/stage1_quick/figures/pareto_frontier_2048.png)

### Iso-Budget Retention (2,048 tokens)
![Iso-Budget Retention (2,048 tokens)](results/stage1_quick/figures/isobudget_comparison_2048.png)

### Pareto Frontier (4,096 tokens)
![Pareto Frontier (4,096 tokens)](results/stage1_quick/figures/pareto_frontier_4096.png)

### Iso-Budget Retention (4,096 tokens)
![Iso-Budget Retention (4,096 tokens)](results/stage1_quick/figures/isobudget_comparison_4096.png)

### AUQC Context Scaling
![AUQC Context Scaling](results/stage1_quick/figures/auqc_context_scaling.png)

### Part 1 vs Part 2 System Tradeoff
![Part 1 vs Part 2 System Tradeoff](results/stage1_quick/figures/resource_vs_system_score.png)

## 3. Methodological Notes & Axioms

1. **Model-Relative Normalization**: Quality measures the exact fraction of the base model's uncompressed Dense FP16 capability retained (q_rel = (Q_m - Q_floor) / max(0.05, Q_dense - Q_floor)).
2. **Primary Resource Axis**: Effective Bits-Per-Token (b_eff in [2.0, 16.0] bpt), unifying quantization, eviction, merging, and low-rank state sizes.
3. **AUQC Integration**: Integrated in the logarithmic domain ln(b_eff) giving equal weight to successive 2x compression factors.
4. **Logarithmic Context Weighting**: Context lengths are weighted proportionally to 1 + log2(L / L_min) to reflect sequence scaling difficulty.
5. **No Arbitrary Weights**: Part 1 evaluates pure representation fidelity without runtime metrics; Part 2 models system utility via a multiplicative constrained utility function.

---
*Report generated automatically by CRBench v0.1.0.*