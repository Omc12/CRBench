# CRBench Benchmark Evaluation Report: crbench_gemma4_e4b_budget

**Model Evaluated:** `google/gemma-4-e4b-it`  
**Evaluation Tracks:** Part 1 (Resource Score) & Part 2 (System Score)  

---

## 1. Executive Summary

CRBench evaluates long-context representation efficiency by characterizing the quality–resource tradeoff curve under strict resource constraints.

### Part 1: CRBench Resource Scores (S_res)

| Method              |   S_res (0-100) |   AUQC (2K) |   AUQC (4K) |   AUQC (8K) |   AUQC (16K) |   AUQC (32K) | Q@2bpt   | Q@4bpt   | Q@8bpt   | Q@16bpt   |
|---------------------|-----------------|-------------|-------------|-------------|--------------|--------------|----------|----------|----------|-----------|
| dkv_high            |            99.9 |       100   |       100   |        99.9 |         99.9 |         99.9 | 100.0%   | 100.0%   | 100.0%   | 83.1%     |
| kivi_style_kv_quant |            86.7 |        81.4 |        88.2 |        84.8 |         87.9 |         87.4 | 100.0%   | 100.0%   | 95.5%    | 100.0%    |
| dense_fp16          |            80   |        80   |        80   |        80   |         80   |         80   | 10.0%    | 20.0%    | 40.0%    | 80.0%     |
| snapkv              |            80   |        80   |        80   |        80   |         80   |         80   | 80.0%    | 80.0%    | 80.0%    | 80.0%     |
| dkv_mid             |            72.1 |        88.8 |        73.3 |        71.5 |         70.5 |         69.9 | 100.0%   | 100.0%   | 0.0%     | 0.0%      |
| low_rank_kv         |            70   |        80   |        80   |        80   |         42.5 |         80   | 80.0%    | 80.0%    | 80.0%    | 80.0%     |
| kv_merging          |            40.9 |        36.2 |        36.8 |        37.1 |         43.5 |         43.6 | 19.8%    | 39.8%    | 79.8%    | 100.0%    |
| streaming_llm       |            35.1 |        31.2 |        60   |        31.2 |         31.2 |         31.2 | 0.0%     | 20.0%    | 59.9%    | 100.0%    |

### Part 2: CRBench System Scores (S_sys)

| Method              |   S_sys (0-100) |   S_res (Part 1) |   TTFT (ms) |   Decode Thru (tok/s) |   Peak VRAM (MB) | Multiplier   |
|---------------------|-----------------|------------------|-------------|-----------------------|------------------|--------------|
| dkv_high            |            94.3 |             99.9 |      5161.4 |                  13.2 |            698.4 | 0.81x        |
| kivi_style_kv_quant |            90.6 |             86.7 |      2841.3 |                  12.8 |            698.4 | 1.00x        |
| dense_fp16          |            86   |             80   |      2843.2 |                  12.9 |            698.2 | 1.00x        |
| low_rank_kv         |            77.8 |             70   |      3179.4 |                  13.1 |            698.4 | 0.96x        |
| dkv_mid             |            74.8 |             72.1 |      5185.8 |                  13.2 |            698.4 | 0.81x        |
| snapkv              |            64.6 |             80   |     83382.7 |                  10.6 |           2472.1 | 0.29x        |
| kv_merging          |            58.4 |             40.9 |      2912.8 |                  13.5 |            698.4 | 0.99x        |
| streaming_llm       |            54.3 |             35.1 |      2915.8 |                  13.6 |            698.3 | 0.99x        |

### Context-Length Weighting Sensitivity Analysis

### Weighting Scheme Sensitivity Analysis

| Scheme Comparison | Spearman rho | Kendall tau | Max Rank Shift |
|:---|:---:|:---:|:---:|
| Logarithmic vs Uniform | 1.0000 | 1.0000 | 0 |
| Logarithmic vs Linear | 0.9286 | 0.8571 | 2 |
| Uniform vs Linear | 0.9286 | 0.8571 | 2 |

## 2. Benchmark Visualizations

### Pareto Frontier (2,048 tokens)
![Pareto Frontier (2,048 tokens)](benchmarks\bench_gemma4_e4b_budget\figures\pareto_frontier_2048.png)

### Iso-Budget Retention (2,048 tokens)
![Iso-Budget Retention (2,048 tokens)](benchmarks\bench_gemma4_e4b_budget\figures\isobudget_comparison_2048.png)

### Pareto Frontier (4,096 tokens)
![Pareto Frontier (4,096 tokens)](benchmarks\bench_gemma4_e4b_budget\figures\pareto_frontier_4096.png)

### Iso-Budget Retention (4,096 tokens)
![Iso-Budget Retention (4,096 tokens)](benchmarks\bench_gemma4_e4b_budget\figures\isobudget_comparison_4096.png)

### Pareto Frontier (8,192 tokens)
![Pareto Frontier (8,192 tokens)](benchmarks\bench_gemma4_e4b_budget\figures\pareto_frontier_8192.png)

### Iso-Budget Retention (8,192 tokens)
![Iso-Budget Retention (8,192 tokens)](benchmarks\bench_gemma4_e4b_budget\figures\isobudget_comparison_8192.png)

### Pareto Frontier (16,384 tokens)
![Pareto Frontier (16,384 tokens)](benchmarks\bench_gemma4_e4b_budget\figures\pareto_frontier_16384.png)

### Iso-Budget Retention (16,384 tokens)
![Iso-Budget Retention (16,384 tokens)](benchmarks\bench_gemma4_e4b_budget\figures\isobudget_comparison_16384.png)

### Pareto Frontier (32,768 tokens)
![Pareto Frontier (32,768 tokens)](benchmarks\bench_gemma4_e4b_budget\figures\pareto_frontier_32768.png)

### Iso-Budget Retention (32,768 tokens)
![Iso-Budget Retention (32,768 tokens)](benchmarks\bench_gemma4_e4b_budget\figures\isobudget_comparison_32768.png)

### AUQC Context Scaling
![AUQC Context Scaling](benchmarks\bench_gemma4_e4b_budget\figures\auqc_context_scaling.png)

### Part 1 vs Part 2 System Tradeoff
![Part 1 vs Part 2 System Tradeoff](benchmarks\bench_gemma4_e4b_budget\figures\resource_vs_system_score.png)

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