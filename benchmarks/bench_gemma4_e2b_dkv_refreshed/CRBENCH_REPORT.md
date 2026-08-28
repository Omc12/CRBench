# CRBench Benchmark Evaluation Report: crbench_gemma4_e2b_dkv_refreshed

**Model Evaluated:** `google/gemma-4-e2b-it`  
**Evaluation Tracks:** Part 1 (Resource Score) & Part 2 (System Score)  

---

## 1. Executive Summary

CRBench evaluates long-context representation efficiency by characterizing the quality–resource tradeoff curve under strict resource constraints.

### Part 1: CRBench Resource Scores (S_res)

| Method     |   S_res (0-100) |   AUQC (2K) |   AUQC (8K) |   AUQC (32K) |   AUQC (64K) |   AUQC (128K) | Q@2bpt   | Q@4bpt   | Q@8bpt   | Q@16bpt   |
|------------|-----------------|-------------|-------------|--------------|--------------|---------------|----------|----------|----------|-----------|
| dkv_high   |            66.4 |       100   |        97.4 |         95.3 |         94.5 |           3.5 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |
| dense_fp16 |            61.8 |        80   |        80   |         80   |         60   |          40   | 5.0%     | 10.0%    | 20.0%    | 40.0%     |
| dkv_mid    |            38.3 |        89.3 |        72.9 |         71.5 |         28.2 |           1.1 | 0.0%     | 0.0%     | 0.0%     | 0.0%      |

### Part 2: CRBench System Scores (S_sys)

| Method     |   S_sys (0-100) |   S_res (Part 1) |   TTFT (ms) |   Decode Thru (tok/s) |   Peak VRAM (MB) | Multiplier   |
|------------|-----------------|------------------|-------------|-----------------------|------------------|--------------|
| dense_fp16 |            73.3 |             61.8 |     13802.8 |                  14.2 |           1267.3 | 1.00x        |
| dkv_high   |            71.1 |             66.4 |     23402.4 |                  13.8 |           1268.3 | 0.82x        |
| dkv_mid    |            51.9 |             38.3 |     22257.1 |                  13.8 |           1266.9 | 0.84x        |

### Context-Length Weighting Sensitivity Analysis

### Weighting Scheme Sensitivity Analysis

| Scheme Comparison | Spearman rho | Kendall tau | Max Rank Shift |
|:---|:---:|:---:|:---:|
| Logarithmic vs Uniform | 1.0000 | 1.0000 | 0 |
| Logarithmic vs Linear | 0.5000 | 0.3333 | 1 |
| Uniform vs Linear | 0.5000 | 0.3333 | 1 |

## 2. Benchmark Visualizations

### Pareto Frontier (2,048 tokens)
![Pareto Frontier (2,048 tokens)](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\pareto_frontier_2048.png)

### Iso-Budget Retention (2,048 tokens)
![Iso-Budget Retention (2,048 tokens)](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\isobudget_comparison_2048.png)

### Pareto Frontier (8,192 tokens)
![Pareto Frontier (8,192 tokens)](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\pareto_frontier_8192.png)

### Iso-Budget Retention (8,192 tokens)
![Iso-Budget Retention (8,192 tokens)](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\isobudget_comparison_8192.png)

### Pareto Frontier (32,768 tokens)
![Pareto Frontier (32,768 tokens)](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\pareto_frontier_32768.png)

### Iso-Budget Retention (32,768 tokens)
![Iso-Budget Retention (32,768 tokens)](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\isobudget_comparison_32768.png)

### Pareto Frontier (65,536 tokens)
![Pareto Frontier (65,536 tokens)](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\pareto_frontier_65536.png)

### Iso-Budget Retention (65,536 tokens)
![Iso-Budget Retention (65,536 tokens)](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\isobudget_comparison_65536.png)

### Pareto Frontier (131,072 tokens)
![Pareto Frontier (131,072 tokens)](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\pareto_frontier_131072.png)

### Iso-Budget Retention (131,072 tokens)
![Iso-Budget Retention (131,072 tokens)](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\isobudget_comparison_131072.png)

### AUQC Context Scaling
![AUQC Context Scaling](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\auqc_context_scaling.png)

### Part 1 vs Part 2 System Tradeoff
![Part 1 vs Part 2 System Tradeoff](benchmarks\bench_gemma4_e2b_dkv_refreshed\figures\resource_vs_system_score.png)

## 3. Scientific Metric Provenance Audit

| Metric Name                      | Classification       | Measurement / Mathematical Instrument                                               | Scientific Status            |
|----------------------------------|----------------------|-------------------------------------------------------------------------------------|------------------------------|
| Raw Task Quality (Q_m)           | MEASURED             | Greedy generation after the method's transform is applied to the resident KV cache  | Real Measurement             |
| Dense Reference (Q_dense)        | MEASURED             | Uncompressed baseline run on the identical query at every context length            | Real Measurement             |
| Model-Relative Retention (q_rel) | DERIVED              | (Q_m - Q_floor) / max(0.05, Q_dense - Q_floor) * 100                                | Mathematically Derived       |
| Effective Bits/Token (b_eff)     | DERIVED (ANALYTICAL) | 8 * (Algorithmic_Bytes + Metadata_Bytes) / L, from the model's KV geometry          | Exact Architecture Formula   |
| Resident KV Bytes                | MEASURED             | Storage of the cache tensors after the method's transform, summed per layer         | Real Measurement             |
| AUQC Score                       | DERIVED              | Monotonic PCHIP logarithmic spline integral: 1/ln(8) * ∫ Q(e^u) du                  | Numerically Integrated       |
| Iso-Budget Quality (Q@B)         | DERIVED              | Standardized evaluation at 2, 4, 8, 16 bpt along empirical spline                   | Interpolated / Extrapolated  |
| Part 1 Resource Score (S_res)    | DERIVED              | Logarithmically context-weighted sum: ∑ w_L AUQC(L)                                 | Axiomatically Derived        |
| TTFT Prefill Latency             | MEASURED             | Device-synchronised wall clock from first prefill chunk to first emitted token (ms) | Real Measurement             |
| Decode Throughput                | MEASURED             | Per-token intervals, each device-synchronised, over the decode stage only           | Real Measurement             |
| Latency Jitter                   | MEASURED             | Standard deviation of the observed inter-token intervals                            | Real Measurement             |
| Peak VRAM Footprint              | MEASURED             | torch.cuda.max_memory_allocated over the query, minus the resident weight baseline  | Real Measurement             |
| Part 2 System Score (S_sys)      | DERIVED              | S_res * (phi_ttft * phi_thru * phi_vram)^alpha                                      | Constrained Utility Function |

## 4. Methodological Notes & Axioms

1. **Model-Relative Normalization**: Quality measures the exact fraction of the base model's uncompressed Dense FP16 capability retained (q_rel = (Q_m - Q_floor) / max(0.05, Q_dense - Q_floor)).
2. **Primary Resource Axis**: Effective Bits-Per-Token (b_eff in [2.0, 16.0] bpt), unifying quantization, eviction, merging, and low-rank state sizes.
3. **AUQC Integration**: Integrated in the logarithmic domain ln(b_eff) giving equal weight to successive 2x compression factors.
4. **Logarithmic Context Weighting**: Context lengths are weighted proportionally to 1 + log2(L / L_min) to reflect sequence scaling difficulty.
5. **No Arbitrary Weights**: Part 1 evaluates pure representation fidelity without runtime metrics; Part 2 models system utility via a multiplicative constrained utility function.

---
*Report generated automatically by CRBench v0.1.0.*