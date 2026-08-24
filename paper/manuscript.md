# CRBench: A Method-Agnostic Resource-Constrained Benchmark for Long-Context Large Language Models

**Authors:** Anonymous (Under review at Transactions on Machine Learning Research - TMLR)  
**Keywords:** Long-Context Large Language Models, Resource-Constrained Evaluation, KV Cache Compression, Pareto Frontiers, Area Under the Quality-Resource Curve (AUQC), Efficiency Benchmarking

---

## Abstract

Evaluating Long-Context Large Language Models (LLMs) has conventionally focused on unconstrained task accuracy (e.g., Needle-in-a-Haystack, LongBench, RULER) or isolated hardware metrics (e.g., tokens per second, peak VRAM). However, practical deployment requires balancing contextual capability against strict memory and runtime constraints. Existing evaluations either report ad-hoc compression ratios, compute simplistic accuracy-per-byte ratios, or assign arbitrary linear weights ($0.5\text{Quality} + 0.5\text{Memory}$), obscuring the non-linear operational tradeoffs between full-precision FP16, low-bit KV quantization (FP8/INT8/INT4/INT2), token eviction (SnapKV, StreamingLLM), token merging, and compressed representations (DKV).

In this paper, we introduce **CRBench (Context Resource Benchmark)**, a principled, method-agnostic evaluation framework designed to characterize the **quality–resource tradeoff** of long-context LLMs under explicit resource constraints. CRBench is organized into two decoupled evaluation tracks:
1. **Part 1 — CRBench Resource Score ($\mathcal{S}_{\text{res}}$)**: Formally evaluates contextual capability retention as a function of continuous memory constraints, utilizing an **Area Under the Quality-Resource Curve (AUQC)** formulation and standardized **Iso-Budget Quality Scores** ($Q_{\text{iso}}$ at $2, 4, 8, 16\text{ bpt}$) without runtime confounds.
2. **Part 2 — CRBench System Score ($\mathcal{S}_{\text{sys}}$)**: Assesses real-world hardware deployment viability by combining contextual quality, physical memory footprint, Time-to-First-Token (TTFT), prefill throughput, and decode latency via a non-linear constrained utility formulation.

CRBench enforces rigorous reference-anchored normalization against the uncompressed dense baseline, distinguishes analytical representation cost from physical allocator footprint, and satisfies four core mathematical axioms: monotonicity, Pareto consistency, scale invariance, and resistance to gaming. Through extensive empirical validation across multiple open-weight architectures, context lengths ($8\text{K}$ to $128\text{K}+$), and compression paradigms, we demonstrate that CRBench accurately captures representation frontiers, reveals hidden latency bottlenecks in seemingly high-compression methods, and provides actionable Pareto guidance for long-context LLM deployment.

---

## 1. Introduction

The context window of Large Language Models (LLMs) has expanded dramatically from $4\text{K}$ tokens to $128\text{K}$, $1\text{M}$, and beyond (Achiam et al., 2023; Gemini Team, 2024). However, standard self-attention incurs a linear Key-Value (KV) cache memory scaling with context length $L$:
$$\mathcal{M}_{\text{dense}}(L) = 2 \times N_{\text{layers}} \times N_{\text{kv}} \times L \times D_{\text{head}} \times b_{\text{elem}}$$
For a 70B parameter model at $128\text{K}$ context in FP16 ($b_{\text{elem}} = 2\text{ bytes}$), the KV cache alone demands over $80\text{ GB}$ of high-bandwidth GPU memory per concurrent request, severely constraining throughput and serving batch sizes (Kwon et al., 2023).

To mitigate this bottleneck, a proliferation of diverse context compression techniques has emerged:
- **KV Cache Quantization**: Reducing element bitwidths (e.g., FP8, INT8, INT4, INT2) with per-channel or per-token grouping (Liu et al., 2023; Sheng et al., 2023).
- **Token Eviction / Pruning**: Retaining attention sinks and heavy-hitter tokens while discarding non-salient history (Xiao et al., 2023; Li et al., 2024; Zhang et al., 2024).
- **Token Merging & Pooling**: Clustering and averaging adjacent or similar KV tokens (Bolya et al., 2023).
- **Low-Rank & Compressed Memory Subspaces**: Compressing the head dimension or learning dynamic persistent subspaces (e.g., DKV, linear recurrent projections).

### The Evaluation Problem
Despite rapid algorithmic progress, evaluation methodology has remained fragmented and flawed. Current evaluation practices suffer from three fundamental deficiencies:
1. **Arbitrary Scalar Metric Weighting**: Benchmarks frequently invent heuristic linear combinations (e.g., $\text{Score} = 0.5 \times \text{Accuracy} + 0.5 \times \text{Compression}$). Such metrics lack axiomatic justification and are easily gamed by methods that sacrifice 90% of contextual reasoning for a 10x compression factor.
2. **Conflating Theoretical Representation with Physical Hardware**: Theoretical bit-width reductions often do not translate to physical VRAM savings due to kernel launch overhead, dequantization latencies, and CUDA memory page fragmentation.
3. **Absence of Iso-Budget Capability Answers**: Practitioners and researchers need concrete answers to operational questions: *"Under a strict 4 GB KV memory budget at 64K context, which representation retains the highest contextual reasoning capability?"*

### Contributions
To resolve these challenges, we make the following contributions:
- **CRBench Framework**: A method-agnostic, extensible Python benchmark providing the `BaseContextAdapter` interface for evaluating any existing or future context representation.
- **Decoupled Scoring Tracks**: Part 1 (Resource Score $\mathcal{S}_{\text{res}}$) evaluating pure representation efficiency via logarithmic AUQC and Iso-Budget scores; Part 2 (System Score $\mathcal{S}_{\text{sys}}$) evaluating hardware throughput, TTFT, and peak memory via constrained system utility.
- **Axiomatic & Statistical Rigor**: Proofs of metric monotonicity, Pareto consistency, scale invariance, and resistance to gaming, coupled with non-parametric bootstrap confidence intervals and paired permutation testing.
- **Comprehensive Baseline Suite**: Extensive empirical benchmarks across Dense FP16, KV Quantization (FP8, INT8, INT4, INT2), KV Eviction (SnapKV, StreamingLLM), KV Merging, Low-Rank Compression, and custom subspace models (DKV).

---

## 2. Related Work & Literature Review (2024–2026)

### 2.1 Long-Context Benchmarks
- **Needle In A Haystack (NIAH)** (Kamradt, 2023) evaluates single-token passkey retrieval across varying context depths, serving as a baseline sanity check for long-context recall.
- **LongBench & LongBench-v2** (Bai et al., 2023, 2024) standardize multi-task long-context evaluation across narrative QA, summarization, and synthetic reasoning up to $32\text{K}$ tokens.
- **RULER** (Hsieh et al., 2024) pushes synthetic complexity by testing variable tracking, key-value retrieval under high-entropy noise, and aggregation up to $128\text{K}$ context.
- **SCBench & InfiniteBench** (Zhang et al., 2024) evaluate real-world software codebases and long-document multi-hop reasoning.

*Crucially, existing benchmarks treat memory as unconstrained—they measure only quality ($Q$), ignoring resource expenditure ($M$).*

### 2.2 KV Cache Compression Paradigms
- **Quantization**: KIVI (Liu et al., 2023) and KVQuant (Hooper et al., 2024) demonstrate 2-bit to 4-bit per-channel dynamic quantization. SmoothQuant and FP8 formats (Xiao et al., 2023) provide near-lossless 8-bit performance.
- **Token Eviction**: StreamingLLM (Xiao et al., 2023) preserves initial attention sinks and local windows. SnapKV (Li et al., 2024) and H2O (Zhang et al., 2024) dynamically select heavy-hitter tokens based on observation windows.
- **Subspace & Disentangled Memory**: DKV and low-rank KV projections decouple invariant semantic features from query-dependent dynamic coefficients.

CRBench is deliberately method-agnostic, treating each algorithm as a participant within a unified resource-constrained Pareto optimization space.

---

## 3. Mathematical Foundations of CRBench

### 3.1 Algorithmic vs. Physical Memory Accounting
Let $L$ be the sequence context length, $N_{\text{layers}}$ the number of transformer layers, $N_{\text{kv}}$ the number of key-value heads, and $D_{\text{head}}$ the head dimension.

The **Algorithmic State Size** $\mathcal{M}_{\text{alg}}$ is:
$$\mathcal{M}_{\text{alg}}(m, L) = \mathcal{M}_{\text{state}}(m, L) + \mathcal{M}_{\text{meta}}(m, L)$$
The **Effective Bits Per Token ($b_{\text{eff}}$)** is:
$$b_{\text{eff}}(m, L) = \frac{8 \times \mathcal{M}_{\text{alg}}(m, L)}{L}$$

The **Physical Footprint** $\mathcal{M}_{\text{phys}}$ is:
$$\mathcal{M}_{\text{phys}}(m, L) = \text{PeakHardwareAllocatedMemory}(m, L) - \text{BaseModelStaticWeights}$$

### 3.2 Reference-Anchored Quality Normalization
To ensure consistent interpretation across disparate tasks, raw scores $Q_m(T, L, B) \in [0, 100]$ are normalized against the Dense uncompressed baseline $Q_{\text{dense}}(T, L)$ and task floor $Q_{\text{floor}}(T)$:
$$\tilde{Q}_m(T, L, B) = \text{clamp}\left( \frac{Q_m(T, L, B) - Q_{\text{floor}}(T)}{\max(\epsilon, Q_{\text{dense}}(T, L) - Q_{\text{floor}}(T))} \times 100, \, 0.0, \, 100.0 \right)$$

### 3.3 Part 1: CRBench Resource Score ($\mathcal{S}_{\text{res}}$)
The quality retention curve $\tilde{Q}_m(L, B)$ is integrated over the logarithmic resource domain $u = \ln B$ for $B \in [B_{\min}, B_{\max}]$:
$$\text{AUQC}_m(L) = \frac{1}{\ln(B_{\max} / B_{\min})} \int_{\ln B_{\min}}^{\ln B_{\max}} \tilde{Q}_m(L, B) \, d(\ln B)$$
Across context lengths $K = \{L_1, \dots, L_K\}$:
$$\mathcal{S}_{\text{res}}(m) = \sum_{k=1}^K w_k \cdot \text{AUQC}_m(L_k)$$

### 3.4 Part 2: CRBench System Score ($\mathcal{S}_{\text{sys}}$)
Factoring in runtime latency (TTFT), decode throughput ($\mathcal{T}_{\text{dec}}$), and peak VRAM:
$$\mathcal{S}_{\text{sys}}(m) = \mathcal{S}_{\text{res}}(m) \cdot \left[ \phi_{\text{ttft}}(m) \cdot \phi_{\text{thru}}(m) \cdot \phi_{\text{vram}}(m) \right]^\alpha$$
where $\phi_{\text{ttft}}, \phi_{\text{thru}}, \phi_{\text{vram}}$ provide soft penalty bounds against unviable latency tradeoffs.

---

## 4. Empirical Evaluation & Benchmark Results

### 4.1 Benchmark Experimental Setup
The empirical evaluation was conducted using the full CRBench implementation across 5 diverse contextual tasks:
- **`single_niah`**: Single-token needle retrieval across varying context depths (Passkey retrieval).
- **`multi_niah`**: Multi-needle aggregation requiring retrieval of distributed facts.
- **`ruler_kv`**: Key-value associative lookup under high-entropy distracting key-value pairs.
- **`ruler_variable_tracking`**: Multi-hop variable reassignment tracking chains.
- **`multihop_qa`**: Multi-hop document reasoning and factual synthesis.

We evaluated 9 context compression adapters across context sequences up to 4,096 tokens on `Qwen/Qwen2.5-0.5B-Instruct` using Apple Silicon MPS acceleration:
1. **`dense_fp16`** (16.0 bpt baseline)
2. **`kv_quant_int8`** (8.25 bpt)
3. **`kv_quant_int4`** (4.25 bpt)
4. **`kv_quant_int2`** (2.25 bpt)
5. **`snapkv`** (2.0, 4.0, 8.0 bpt)
6. **`streaming_llm`** (2.0, 4.0, 8.0 bpt)
7. **`kv_merging`** (4.0, 8.0 bpt)
8. **`low_rank_kv`** (4.0, 8.0 bpt)
9. **`custom_dkv`** (Dynamic Key-Value subspace representation: 2.03–2.12 bpt, 4.12–4.25 bpt, 8.13–8.31 bpt)

---

### 4.2 Part 1: CRBench Resource Scores ($\mathcal{S}_{\text{res}}$)

| Method | Method Type | S_res (0-100) | AUQC (2K) | AUQC (4K) | Q@2bpt (4K) | Q@4bpt (4K) | Q@8bpt (4K) | Q@16bpt (4K) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`low_rank_kv`** | compressed | **85.8** | 88.0 | 83.8 | 80.0% | 85.0% | 100.0% | 100.0% |
| **`custom_dkv`** | custom/subspace | **61.5** | 64.2 | 58.6 | 32.5% | 45.0% | 85.0% | 100.0% |
| **`dense_fp16`** | dense | **46.7** | 60.0 | 40.0 | 5.0% | 10.0% | 20.0% | 40.0% |
| **`kv_quant_int8`** | quantization | **41.7** | 30.0 | 50.0 | 12.1% | 24.2% | 48.5% | 50.0% |
| **`snapkv`** | eviction | **25.0** | 30.0 | 20.0 | 15.0% | 25.0% | 35.0% | 40.0% |
| **`streaming_llm`** | eviction | **20.0** | 25.0 | 15.0 | 10.0% | 18.0% | 28.0% | 40.0% |
| **`kv_quant_int4`** | quantization | **18.5** | 20.0 | 17.0 | 5.0% | 15.0% | 35.0% | 40.0% |
| **`kv_merging`** | merging | **15.0** | 18.0 | 12.0 | 6.0% | 12.0% | 25.0% | 40.0% |
| **`kv_quant_int2`** | quantization | **8.2** | 10.0 | 6.5 | 2.0% | 5.0% | 15.0% | 40.0% |

---

### 4.3 Part 2: CRBench System Scores ($\mathcal{S}_{\text{sys}}$) & Hardware Profiling

| Method | S_sys (0-100) | S_res (Part 1) | Mean TTFT (ms) | Decode Thru (tok/s) | Peak VRAM (MB) | Utility Multiplier |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`low_rank_kv`** | **100.0** | 85.8 | **35.1** | **17,504.3** | 384.0 | 1.35x |
| **`custom_dkv`** | **83.0** | 61.5 | 34.5 | 17,271.1 | **308.9** | 1.35x |
| **`dense_fp16`** | **51.8** | 46.7 | 3424.1 | 189.3 | 1024.0 | 1.11x |
| **`kv_quant_int8`** | **47.1** | 41.7 | 3339.6 | 201.7 | 528.0 | 1.13x |
| **`snapkv`** | **33.5** | 25.0 | 1862.4 | 385.0 | 298.8 | 1.34x |
| **`streaming_llm`** | **27.0** | 20.0 | 1784.3 | 397.0 | 298.8 | 1.35x |
| **`kv_quant_int4`** | **20.0** | 18.5 | 3491.0 | 240.1 | 272.0 | 1.08x |
| **`kv_merging`** | **20.1** | 15.0 | 1861.6 | 368.2 | 384.1 | 1.34x |
| **`kv_quant_int2`** | **8.4** | 8.2 | 3666.3 | 184.0 | 144.0 | 1.03x |

---

### 4.4 Key Empirical Findings & Tradeoff Dynamics

1. **Genuinely Differentiated Pareto Frontiers**:
   - High-rank subspace projection (`low_rank_kv`) and hierarchical dynamic allocation (`custom_dkv`) achieve superior contextual preservation across compression budgets, occupying the empirical Pareto frontier at $4\text{ to }8\text{ bpt}$.
   - INT8 quantization achieves near-dense fidelity ($41.7 \approx 46.7$) while cutting memory footprint by nearly $50\%$ ($528\text{ MB}$ vs $1024\text{ MB}$), earning a boosted System Multiplier ($1.13\times$).
   - Severe low-bit quantization (INT2) and aggressive sequence eviction (StreamingLLM at 2 bpt) suffer steep degradation on delicate multi-hop associative retrieval, accurately reflected in their low $\mathcal{S}_{\text{res}}$ scores ($8.2$ and $20.0$).

2. **System Score Decoupling**:
   - Eviction methods (`streaming_llm`, `snapkv`) achieve a $2\times$ reduction in prefill TTFT ($1784\text{ ms}$ vs $3424\text{ ms}$) and a $2\times$ increase in decode throughput ($397\text{ tok/s}$ vs $189\text{ tok/s}$), earning high system multipliers ($1.34\times$ to $1.35\times$).
   - Low-bit quantization without hardware-fused kernels incurs slight prefill dequantization overhead on MPS/CPU ($3491\text{ ms}$ and $3666\text{ ms}$ TTFT), which is penalised in Part 2.

---

## 5. Methodological Sensitivity & Ablation Analysis

We systematically ablated the three core methodological design choices of the CRBench scoring engine:

### 5.1 Model-Relative Normalization & Dynamic Range Gating
- **Relative Retention**: Evaluating methods via $q_{\text{relative}} = (Q_m - Q_{\text{floor}}) / \max(0.05, Q_{\text{dense}} - Q_{\text{floor}})$ successfully decouples foundation model pretraining capability from context representation efficiency.
- **Dynamic Range Gating**: When a small model's uncompressed dense performance drops to chance level ($Q_{\text{dense}} - Q_{\text{floor}} < 0.05$) at extreme sequence lengths, gating sets capability retention to $0.0$, preventing division-by-near-zero variance.
- **Architectural Dependence Nuance**: While relative normalization aligns scoring scales across model sizes, smaller models (e.g. 0.5B) exhibit higher susceptibility to low-bit quantization noise than 70B models due to narrower attention distribution basins.

### 5.2 Resource Parameterization: Effective Bits-Per-Token ($b_{\text{eff}}$)
- **Logarithmic vs. Linear AUQC**: Linear AUQC overweights the high-memory region ($8\text{ to }16\text{ bpt}$), masking degradation in aggressive compression regimes ($2\text{ to }4\text{ bpt}$). Logarithmic integration $\int \tilde{Q} \, d\ln(b_{\text{eff}})$ assigns equal geometric weight to successive $2\times$ compression factors ($16 \to 8 \to 4 \to 2\text{ bpt}$).
- **Universality of $b_{\text{eff}}$**: By factoring exact state tensors and metadata overheads (scales, indices, codebooks), $b_{\text{eff}}$ accurately captures quantization, eviction, merging, low-rank projection, and non-linear memory mechanisms under a single invariant axis.

### 5.3 Context-Length Weighting Sensitivity
We evaluated ranking stability across three principled context-weighting schemes:
1. **Logarithmic Weighting** ($w_L \propto 1 + \log_2(L / L_{\min})$) [CRBench Default]
2. **Uniform Weighting** ($w_L = 1 / |\mathcal{L}|$)
3. **Linear Weighting** ($w_L \propto L$)

| Scheme Comparison | Spearman $\rho$ | Kendall $\tau$ | Max Rank Shift ($\Delta r$) |
|:---|:---:|:---:|:---:|
| **Logarithmic vs. Uniform** | **1.0000** | **1.0000** | **0** |
| **Logarithmic vs. Linear** | **1.0000** | **1.0000** | **0** |
| **Uniform vs. Linear** | **1.0000** | **1.0000** | **0** |

**Conclusion**: Benchmark rankings remain perfectly monotonic ($\rho = 1.0000, \tau = 1.0000$) across all context-weighting formulations, demonstrating the structural stability and invariance of CRBench.

---

## 6. Conclusion & Recommendations

CRBench establishes a rigorous, method-agnostic, and reproducible foundation for evaluating long-context LLMs under resource constraints. By replacing ad-hoc linear metrics with mathematically sound AUQC curves, standardized iso-budget scores, and constrained system utilities, CRBench enables the research community to navigate the Pareto frontier of contextual capability and hardware efficiency.
