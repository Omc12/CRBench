# CRBench: Mathematical Foundations & Methodology Specification

This document provides the formal mathematical specification, proofs, and methodological justifications for **CRBench (Context Resource Benchmark)**.

---

## 1. Problem Formulation

Let $\mathcal{M}$ denote a pre-trained Large Language Model parameterized by weights $\Theta$. For an input sequence of context length $L \in \mathbb{N}^+$, standard self-attention computes and caches key-value representations across $N_{\text{layers}}$ layers, $N_{\text{kv}}$ KV heads, and head dimension $D_{\text{head}}$:

$$\mathbf{K}, \mathbf{V} \in \mathbb{R}^{N_{\text{layers}} \times N_{\text{kv}} \times L \times D_{\text{head}}}$$

In standard uncompressed half-precision floating point (FP16/BF16, $b_{\text{dense}} = 16\text{ bits}$), the analytical state memory footprint $\mathcal{M}_{\text{dense}}(L)$ is given by:

$$\mathcal{M}_{\text{dense}}(L) = 2 \times N_{\text{layers}} \times N_{\text{kv}} \times L \times D_{\text{head}} \times 2.0 \text{ bytes}$$

A **context representation method** $m \in \Omega$ compresses or transforms the KV state into a state $\mathcal{S}_m(L)$ subject to an operating resource budget $B$.

The central goal of CRBench is to evaluate the contextual capability retention curve $\tilde{Q}_m(L, B)$ across representation methods, sequence lengths, and operating resource constraints.

---

## 2. Model-Independent / Model-Relative Normalization

### 2.1 Principle of Model Capability Decoupling
CRBench evaluates the representation efficiency of a context compression algorithm, not the scale of the base foundation model. A 70B parameter model will inherently achieve higher absolute QA and retrieval accuracy than a 0.5B parameter model. However, an uncompressed 70B cache should not achieve a higher benchmark resource score simply due to pretraining scale.

### 2.2 Mathematical Definition: Regularized Capability Retention
Raw task performance $Q_m(T, L, B) \in [0, 1]$ is normalized relative to the base model's uncompressed Dense FP16 performance $Q_{\text{dense}}(T, L)$ and task floor $Q_{\text{floor}}(T)$:

$$\tilde{Q}_m(T, L, B) = \begin{cases}
0.0, & \text{if } Q_{\text{dense}}(T, L) - Q_{\text{floor}}(T) < \Delta_{\min} \text{ and } Q_m(T, L, B) \le Q_{\text{dense}}(T, L) \\
\text{clamp}\left( \frac{Q_m(T, L, B) - Q_{\text{floor}}(T)}{Q_{\text{dense}}(T, L) - Q_{\text{floor}}(T)} \times 100, \, 0.0, \, 100.0 \right), & \text{if } Q_{\text{dense}}(T, L) - Q_{\text{floor}}(T) \ge \Delta_{\min} \\
\min\left(100.0, \, \frac{Q_m(T, L, B) - Q_{\text{floor}}(T)}{\Delta_{\min}} \times 100\right), & \text{if } Q_m(T, L, B) > Q_{\text{dense}}(T, L)
\end{cases}$$

where $\Delta_{\min} = 0.05$ ($5\%$ minimum dynamic range).

### 2.3 Effective Context Window (ECW) Gating
When $Q_{\text{dense}}(T, L) - Q_{\text{floor}}(T) < \Delta_{\min}$, the sequence length $L$ exceeds the base model's effective context window. In this regime, the uncompressed representation itself possesses no contextual capability. Dividing by a near-zero denominator would cause extreme variance; the gating function sets relative retention to $0.0$, preventing artificial score inflation.

### 2.4 Reporting Separation
CRBench reports:
1. **Primary Metric**: Model-Relative Retention Score $\mathcal{S}_{\text{res}}^{\text{rel}} \in [0.0, 100.0]$.
2. **Secondary Metric**: Absolute Task Quality $Q_{\text{abs}}$ and Dense Baseline $Q_{\text{dense}}$.

*Scientific Note*: While relative normalization removes scale bias from baseline scores, empirical experiments show that smaller models (e.g. 0.5B) have smaller attention dynamic ranges and higher sensitivity to low-bit quantization noise than 70B models. Thus, relative scores reflect capability retention on that specific architecture, rather than total architectural independence.

---

## 3. Resource-Budget Parameterization

### 3.1 Evaluation of Resource Axes
CRBench evaluated four candidate resource axes:
1. **Total KV Bytes**: Model-dependent ($N_{\text{layers}}, N_{\text{kv}}, D_{\text{head}}$ vary wildly), precluding standardized cross-model integration.
2. **Compression Ratio (CR)**: Obscures whether a model natively uses FP16 vs FP8/GQA, hiding absolute informational density.
3. **Effective Bits-Per-Token ($b_{\text{eff}}$)**: The exact mathematical information density allocated per token position across all layers and heads:

$$b_{\text{eff}}(m, L) = 16.0 \times \frac{\mathcal{M}_{\text{alg}}(m, L)}{\mathcal{M}_{\text{dense}}(L)}$$

where $\mathcal{M}_{\text{alg}}(m, L) = \mathcal{M}_{\text{state}}(m, L) + \mathcal{M}_{\text{meta}}(m, L)$ includes:
- Representation tensors (quantized values, retained token embeddings, low-rank subspace coordinates).
- All metadata overheads (quantization scale factors $\mathbf{s}$, position indices $\mathbf{p}$, codebooks, centroids).

### 3.2 Canonical Scope of $b_{\text{eff}}$
- **Dense FP16/BF16**: $b_{\text{eff}} = 16.0\text{ bpt}$.
- **Uniform Quantization**: $b_{\text{eff}} = b_{\text{quant}} + \text{metadata} \approx 8.25, 4.25, 2.12\text{ bpt}$.
- **Token Eviction / Pruning**: $b_{\text{eff}} = \left(\frac{K_{\text{retained}}}{L}\right) \times 16.0\text{ bpt}$.
- **Low-Rank Subspace Projection**: $b_{\text{eff}} = \left(\frac{r}{D_{\text{head}}}\right) \times 16.0\text{ bpt}$.
- **Non-Linear / Sub-Linear Memory**: $b_{\text{eff}}(L)$ natively measures the exact per-token byte equivalent at length $L$.

### 3.3 Primary vs Secondary Metrics
- **Primary Resource Axis**: Effective Bits-Per-Token ($b_{\text{eff}} \in [2.0, 16.0]\text{ bpt}$).
- **Secondary Metrics**: Physical KV Memory (GB), Compression Factor ($\text{CR}$), Memory Sparsity ($1 - 1/\text{CR}$).

---

## 4. Part 1 — CRBench Resource Score ($\mathcal{S}_{\text{res}}$)

### 4.1 Logarithmic Area Under the Quality-Resource Curve (AUQC)
Because memory scaling is multiplicative ($16 \to 8 \to 4 \to 2\text{ bpt}$), AUQC is integrated over the logarithmic resource domain $u = \ln(b_{\text{eff}})$:

$$\text{AUQC}_m(T, L) = \frac{1}{\ln(B_{\max} / B_{\min})} \int_{\ln B_{\min}}^{\ln B_{\max}} \tilde{Q}_m(T, L, e^u) \, du$$

Between evaluated points $\{(\ln B_i, \tilde{Q}_i)\}_{i=1}^N$, CRBench uses Monotonic Piecewise Cubic Hermite Interpolation (PCHIP) to guarantee shape preservation and avoid non-physical oscillations.

### 4.2 Principled Context-Length Weighting
When aggregating across a logarithmic sweep of context lengths $\mathcal{L} = \{L_1, L_2, \dots, L_K\}$, CRBench defines:

$$\mathcal{S}_{\text{res}}(m) = \sum_{k=1}^K w_k \cdot \text{AUQC}_m(L_k), \quad \sum_{k=1}^K w_k = 1.0$$

#### Weighting Schemes:
1. **Logarithmic Context Weighting (Default)**:
   $$w_k = \frac{1 + \log_2(L_k / L_{\min})}{\sum_{j=1}^K (1 + \log_2(L_j / L_{\min}))}$$
   *Rationale*: Assigns smoothly increasing weight to larger context windows without allowing the maximum length to unilaterally eclipse short/medium context fidelity.
2. **Uniform Context Weighting**: $w_k = 1 / |\mathcal{L}|$.
3. **Linear Sequence Weighting**: $w_k = L_k / \sum_j L_j$.

### 4.3 Iso-Budget Operating Scores
CRBench computes capability retention at discrete operational operating points:
$$Q_{\text{iso}}(m, L, B^*) = \tilde{Q}_m(L, B^*), \quad B^* \in \{2.0, 4.0, 8.0, 16.0\}\text{ bpt}$$

---

## 5. Part 2 — CRBench System Score ($\mathcal{S}_{\text{sys}}$)

Part 2 evaluates real-world deployment efficiency under runtime throughput and latency constraints.

CRBench avoids arbitrary linear weighting ($0.5 \text{Quality} + 0.5 \text{Speed}$) by using a **Constrained System Utility** formulation:

$$\mathcal{S}_{\text{sys}}(m) = \mathcal{S}_{\text{res}}(m) \cdot \left[ \phi_{\text{ttft}}(m) \cdot \phi_{\text{thru}}(m) \cdot \phi_{\text{vram}}(m) \right]^\alpha$$

where:
1. **TTFT Efficiency Factor**:
   $$\phi_{\text{ttft}}(m) = \frac{1}{1.0 + \max\left(0, \, \frac{\text{TTFT}_m - \text{TTFT}_{\text{ref}}}{\text{TTFT}_{\text{ref}}}\right)^{0.8}}$$
2. **Decode Throughput Factor**:
   $$\phi_{\text{thru}}(m) = \min\left(1.25, \, \max\left(0.2, \, \left(\frac{\mathcal{T}_{\text{decode}, m}}{\mathcal{T}_{\text{ref}}}\right)^{0.5}\right)\right)$$
3. **VRAM Footprint Compliance Factor**:
   $$\phi_{\text{vram}}(m) = \begin{cases} 1.0 & \text{if } V_m \le V_{\text{budget}} \\ \frac{1.0}{1.0 + 1.5 \left(\frac{V_m - V_{\text{budget}}}{V_{\text{budget}}}\right)} & \text{if } V_m > V_{\text{budget}} \end{cases}$$

---

## 6. Mathematical Properties & Axiomatic Proofs

1. **Monotonicity**: If method $A$ achieves $\tilde{Q}_A(L, B) \ge \tilde{Q}_B(L, B)$ for all $B \in [B_{\min}, B_{\max}]$, then $\text{AUQC}_A(L) \ge \text{AUQC}_B(L)$ and $\mathcal{S}_{\text{res}}(A) \ge \mathcal{S}_{\text{res}}(B)$.
2. **Pareto Dominance Consistency**: If operating point $p_1 = (b_1, q_1)$ Pareto-dominates $p_2 = (b_2, q_2)$ ($b_1 \le b_2$ and $q_1 \ge q_2$), the non-dominated Pareto frontier preserves $p_1$.
3. **Scale Invariance**: Logarithmic AUQC is invariant under linear scaling of the underlying precision unit.
4. **Gaming Resistance**: Methods cannot inflate their score by sampling excessively at easy, high-budget regimes, because the integration domain is strictly bounded in $[B_{\min}, B_{\max}]$ with continuous normalized integration.
