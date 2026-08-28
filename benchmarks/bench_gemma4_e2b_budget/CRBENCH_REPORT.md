# CRBench Benchmark Evaluation Report: crbench_gemma4_e2b_budget

**Model Evaluated:** `google/gemma-4-e2b-it`  
**Evaluation Tracks:** Part 1 (Resource Score) & Part 2 (System Score)  

---

## 1. Executive Summary

CRBench evaluates long-context representation efficiency by characterizing the quality–resource tradeoff curve under strict resource constraints.

### Part 1: CRBench Resource Scores (S_res)

| Method              |   S_res (0-100) |   AUQC (2K) |   AUQC (8K) |   AUQC (32K) |   AUQC (64K) |   AUQC (128K) | Q@2bpt   | Q@4bpt   | Q@8bpt   | Q@16bpt   |
|---------------------|-----------------|-------------|-------------|--------------|--------------|---------------|----------|----------|----------|-----------|
| snapkv              |            68   |        76.7 |        80   |         80   |         60   |          60.1 | 70.0%    | 60.0%    | 40.1%    | 0.2%      |
| dkv_high            |            62.5 |       100   |        99.9 |         99.9 |         77.4 |           1.6 | 0.0%     | 0.0%     | 0.0%     | 97.6%     |
| dense_fp16          |            61.8 |        80   |        80   |         80   |         60   |          40   | 5.0%     | 10.0%    | 20.0%    | 40.0%     |
| kivi_style_kv_quant |            54.7 |        73.8 |        83.7 |        100   |         23.2 |          34.3 | 0.0%     | 0.0%     | 95.7%    | 100.0%    |
| dkv_mid             |            47.6 |        88.8 |         1.5 |         72.8 |         98.3 |           0   | 0.0%     | 0.0%     | 0.0%     | 0.0%      |
| low_rank_kv         |            31.1 |        80   |        41.9 |         30.6 |         30.9 |          19.9 | 0.0%     | 0.0%     | 39.7%    | 100.0%    |
| streaming_llm       |            23.8 |        30.7 |        26.7 |         26.7 |         26.7 |          16.8 | 0.0%     | 6.6%     | 33.2%    | 86.5%     |
| kv_merging          |            12.5 |        32.8 |        18.2 |         18.2 |          8.5 |           6.6 | 0.0%     | 0.9%     | 13.2%    | 37.9%     |

### Part 2: CRBench System Scores (S_sys)

| Method              |   S_sys (0-100) |   S_res (Part 1) |   TTFT (ms) |   Decode Thru (tok/s) |   Peak VRAM (MB) | Multiplier   |
|---------------------|-----------------|------------------|-------------|-----------------------|------------------|--------------|
| snapkv              |            77.2 |             68   |     13037.9 |                  12.9 |           2163.9 | 0.99x        |
| dense_fp16          |            73.3 |             61.8 |     12559.3 |                  12.6 |            912.6 | 1.00x        |
| kivi_style_kv_quant |            67.8 |             54.7 |     12508.4 |                  11.9 |            913.6 | 0.98x        |
| dkv_high            |            64.8 |             62.5 |     34008   |                  12.4 |            913.4 | 0.70x        |
| dkv_mid             |            55.3 |             47.6 |     30551.7 |                  12.6 |            913.1 | 0.73x        |
| low_rank_kv         |            51.3 |             31.1 |     12798.2 |                  12.3 |            913.1 | 0.98x        |
| streaming_llm       |            46.6 |             23.8 |     12579.5 |                  13.7 |            913.4 | 1.00x        |
| kv_merging          |            38.7 |             12.5 |     12589.7 |                  13.7 |            914.1 | 1.00x        |

### Context-Length Weighting Sensitivity Analysis

### Weighting Scheme Sensitivity Analysis

| Scheme Comparison | Spearman rho | Kendall tau | Max Rank Shift |
|:---|:---:|:---:|:---:|
| Logarithmic vs Uniform | 0.9524 | 0.8571 | 1 |
| Logarithmic vs Linear | 0.9762 | 0.9286 | 1 |
| Uniform vs Linear | 0.9048 | 0.7857 | 2 |

## 2. Benchmark Visualizations

### Pareto Frontier (2,048 tokens)
![Pareto Frontier (2,048 tokens)](benchmarks\bench_gemma4_e2b_budget\figures\pareto_frontier_2048.png)

### Iso-Budget Retention (2,048 tokens)
![Iso-Budget Retention (2,048 tokens)](benchmarks\bench_gemma4_e2b_budget\figures\isobudget_comparison_2048.png)

### Pareto Frontier (8,192 tokens)
![Pareto Frontier (8,192 tokens)](benchmarks\bench_gemma4_e2b_budget\figures\pareto_frontier_8192.png)

### Iso-Budget Retention (8,192 tokens)
![Iso-Budget Retention (8,192 tokens)](benchmarks\bench_gemma4_e2b_budget\figures\isobudget_comparison_8192.png)

### Pareto Frontier (32,768 tokens)
![Pareto Frontier (32,768 tokens)](benchmarks\bench_gemma4_e2b_budget\figures\pareto_frontier_32768.png)

### Iso-Budget Retention (32,768 tokens)
![Iso-Budget Retention (32,768 tokens)](benchmarks\bench_gemma4_e2b_budget\figures\isobudget_comparison_32768.png)

### Pareto Frontier (65,536 tokens)
![Pareto Frontier (65,536 tokens)](benchmarks\bench_gemma4_e2b_budget\figures\pareto_frontier_65536.png)

### Iso-Budget Retention (65,536 tokens)
![Iso-Budget Retention (65,536 tokens)](benchmarks\bench_gemma4_e2b_budget\figures\isobudget_comparison_65536.png)

### Pareto Frontier (131,072 tokens)
![Pareto Frontier (131,072 tokens)](benchmarks\bench_gemma4_e2b_budget\figures\pareto_frontier_131072.png)

### Iso-Budget Retention (131,072 tokens)
![Iso-Budget Retention (131,072 tokens)](benchmarks\bench_gemma4_e2b_budget\figures\isobudget_comparison_131072.png)

### AUQC Context Scaling
![AUQC Context Scaling](benchmarks\bench_gemma4_e2b_budget\figures\auqc_context_scaling.png)

### Part 1 vs Part 2 System Tradeoff
![Part 1 vs Part 2 System Tradeoff](benchmarks\bench_gemma4_e2b_budget\figures\resource_vs_system_score.png)

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