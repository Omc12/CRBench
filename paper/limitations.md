# CRBench: Scientific Limitations & Scope

In accordance with TMLR standards of scientific honesty and rigor, this document details the boundaries, assumptions, and limitations of the CRBench benchmark.

---

## 1. Scope & Task Boundaries

1. **Synthetic vs. Open-Ended Long-Context Reasoning**:
   While CRBench integrates standard long-context tasks (Needle-in-a-Haystack, RULER Multi-Hop Tracing, Key-Value Retrieval, and Narrative QA), synthetic retrieval tasks do not capture the full complexity of ambiguous multi-document synthesis or nuanced creative generation.
2. **Context Length Truncation Bounds**:
   CRBench supports context sweeps up to 128K+ tokens. However, evaluating extremely large contexts (>256K) on resource-constrained commodity hardware (e.g. 8GB Apple Silicon or single 24GB GPUs) requires simulation or distributed clusters.

---

## 2. Memory Accounting Assumptions

1. **Analytical vs. Framework Cache Allocator Overhead**:
   PyTorch and CUDA memory allocators utilize memory caching pools that allocate blocks in discrete powers-of-two chunks. Algorithmic representation cost computes exact analytical state sizes ($2 \times L \times N_{\text{kv}} \times D \times b_{\text{eff}} / 8$), which may differ slightly from framework-level memory fragmentation.
2. **Dynamic Sparsity and Dynamic Token Lengths**:
   For methods with input-dependent token retention (e.g. adaptive Heavy-Hitter eviction), the memory footprint varies dynamically per sample. CRBench reports mean algorithmic bytes across evaluation batches.

---

## 3. Runtime & Hardware Dependences in Part 2

1. **Kernel Optimization Disparity**:
   Standard Dense FP16 and FlashAttention-2 benefit from years of hand-tuned CUDA/Triton engineering. Newer research methods (such as custom low-rank representations or non-uniform quantization) may experience slower runtime not due to algorithmic complexity, but due to unoptimized reference implementations.
2. **Hardware Architecture Specificity**:
   Latency metrics (TTFT, decode throughput) measured on Apple Silicon MPS or NVIDIA Ampere/Hopper GPUs exhibit differing compute-to-memory bandwidth ratios. For this reason, Part 1 (Resource Score) is kept strictly decoupled from Part 2 (System Score).
