# CRBench Reproducibility & Protocol Guide

This guide details the exact environment setup, hardware recommendations, benchmark commands, score recomputation protocol, and result schema to ensure complete reproducibility of CRBench evaluations.

---

## 1. Environment Setup

### Prerequisites
- Python >= 3.9
- PyTorch >= 2.0.0 (with CUDA 12.x for NVIDIA GPUs, or MPS on Apple Silicon)
- Hugging Face `transformers` >= 5.0 (the cache API used here is 5.x), `accelerate` >= 0.28.0
- `bitsandbytes` >= 0.45 for NF4 4-bit weights
- The vendored upstream repositories under `third_party/` (`SnapKV`, `streaming-llm`,
  `Differential-KV`). A missing checkout makes the corresponding adapter report
  `UNSUPPORTED` rather than silently substituting an approximation.

### Clean Installation
```bash
# Clone repository
git clone https://github.com/Omc12/CRBench.git
cd CRBench

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install CRBench in editable mode with development tools
pip install -e ".[dev]"

# Verify installation with test suite
pytest tests/ -v
```

---

## 2. Hardware Requirements & Recommendations

What a context length costs is set almost entirely by the KV cache, and that is
a property of the model's attention geometry rather than its parameter count.
Per token, a dense fp16 cache needs `2 x layers x kv_heads x head_dim x 2` bytes:

| Model | Layers | KV heads | head_dim | KV per token | 32K | 64K | 128K |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-3B-Instruct | 36 | 2 | 128 | 36 KiB | 1.12 GiB | 2.25 GiB | 4.50 GiB |
| Qwen2.5-7B-Instruct | 28 | 4 | 128 | 56 KiB | 1.75 GiB | 3.50 GiB | 7.00 GiB |

Add NF4 weights (measured: 1.92 GiB for the 3B, 5.18 GiB for the 7B) and the
prefill working set, and the reachable context on a given card follows. Measured
peak allocation on an RTX 4070 SUPER (11.99 GiB usable), prefill chunk 4096:

| Model | 32K | 64K | 128K |
| :--- | :---: | :---: | :---: |
| Qwen2.5-3B NF4 | 4.58 GiB | 5.81 GiB | 9.69 GiB |
| Qwen2.5-7B NF4 | 9.75 GiB | **11.57 GiB — spills** | does not fit |

The 7B's 64K entry is the one to be careful with. It does not raise an
`OutOfMemoryError`; on Windows it spills into WDDM shared host memory and keeps
running, with decode falling from 251 ms/token to 6–9 s/token. A run that
tolerates that silently will report Part 2 latency figures that measure PCIe
paging rather than the method under test. Neither `torch.cuda.empty_cache()` nor
`garbage_collection_threshold` helps, because the excess is live tensors and not
allocator fragmentation, and `expandable_segments` is unavailable on Windows.

| Model Class | Reachable context on 12 GiB (NF4) | Notes |
| :--- | :--- | :--- |
| **0.5B – 1.5B** | 2K – 32K, or CPU / Apple Silicon | Fits everywhere; useful for pipeline checks |
| **3B** (Qwen2.5-3B) | 2K – 128K | Carries the long end of the published sweep |
| **7B** (Qwen2.5-7B) | 2K – 32K | 64K spills; 128K needs ~24 GB |

### Positional scaling

Qwen2.5 ships `max_position_embeddings = 32768`. Contexts beyond that require
YaRN, and a score collected past the native window without it measures
positional extrapolation failure rather than the KV representation under test.
Set it explicitly, and keep native and scaled lengths in separate runs so the
two regimes stay distinguishable:

```yaml
model:
  max_model_len: 131072
  rope_scaling:
    rope_type: "yarn"
    factor: 4.0
    original_max_position_embeddings: 32768
```

The effective RoPE parameters actually handed to the model are recorded in the
result manifest under `model_config.rope_scaling`.

---

## 3. Benchmark Execution Commands

### A. Atomic Query Evaluation (Single Prompt Primitive)
Evaluates ONE Model on ONE Context/Query against its own Dense reference:
```bash
crbench evaluate \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --query "What is the key for user Zurich?" \
  --context "User Zurich key is 928374." \
  --ground-truth "928374" \
  --method "kv_quant_int4" \
  --budget 4.0 \
  --dense
```

### B. Dataset Evaluation (Query-Level Aggregation)
Evaluates a dataset across context lengths, aggregating individual query scores with 95% bootstrap confidence intervals:
```bash
crbench evaluate-dataset \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --dataset "single_niah" \
  --method "snapkv" \
  --budget 4.0 \
  --context-lengths 2048 --context-lengths 4096 \
  --samples 5 \
  --output-dir "results/dataset_snapkv"
```

### C. Standard Benchmark Suite
Executes the comprehensive suite across all standard tasks, context lengths, and adapter paradigms:
```bash
crbench run --config configs/standard_benchmark.yaml
```

### D. The published 2K-128K quantized sweep

Three runs, because no single (model, RoPE regime) pairing covers the range on a
12 GiB card. Run them **one at a time**: two sweeps sharing a GPU will each spill.

```bash
crbench run --config configs/bench_7b_native.yaml
crbench run --config configs/bench_3b_native.yaml
crbench run --config configs/bench_3b_yarn.yaml
```

Then join them, keeping the model and RoPE regime attached to every row:

```bash
python scripts/aggregate_runs.py results/bench_7b_native results/bench_3b_native results/bench_3b_yarn_long -o results/COMBINED_REPORT.md
```

---

## 4. Measurement Protocol

### Where each method comes from

Published methods are driven from their authors' repositories, vendored under
`third_party/`. Nothing in `crbench/adapters/` reimplements a published
algorithm, and the exact upstream commit is written into every result manifest
under `method_provenance`.

| Adapter | Source | Entry point |
| :--- | :--- | :--- |
| `snapkv` | [FasterDecoding/SnapKV](https://github.com/FasterDecoding/SnapKV) | `SnapKVCluster.update_kv` |
| `streaming_llm` | [mit-han-lab/streaming-llm](https://github.com/mit-han-lab/streaming-llm) | `StartRecentKVCache` |
| `dkv` | [Omc12/Differential-KV](https://github.com/Omc12/Differential-KV) | `compress_lowrank` |
| `kv_quant` | CRBench, following KIVI's grouping | keys per-channel, values per-token |
| `low_rank_kv` | CRBench | per-head mean-centred truncated SVD |
| `kv_merging` | CRBench | temporal mean-pooling |

Two adaptations are needed, and both are recorded in the manifest alongside the
commit:

- **transformers version.** Upstream SnapKV's monkeypatch targets
  `transformers==4.37` and Llama/Mistral/Mixtral. CRBench runs transformers 5.x
  on Qwen2, so the algorithm object is driven directly rather than through
  `replace_llama()`.
- **Grouped-query attention.** Upstream stores its compressed cache with KV
  repeated to the full query-head count. On Qwen2.5-7B's 28 query heads over 4
  KV heads that would multiply the retained cache sevenfold and measure an
  implementation artifact instead of the algorithm, so attention scores are
  pooled across each group and selection is per KV head.
  `tests/test_upstream_fidelity.py::test_snapkv_matches_upstream_when_no_gqa`
  asserts this reduces to upstream bit-for-bit when there is one query head per
  KV head.

### How a query is executed

Every method is a **transform on the resident KV cache**, because that is what
these methods physically are. Deleting input tokens before the forward pass is a
different intervention: it changes the positions the model sees, changes prefill
cost, and denies an attention-based selector the signal it is defined in terms
of.

1. The prompt is prefilled in `prefill_chunk_size` chunks sharing one cache, so
   peak activation memory tracks the chunk size rather than the context length.
   An `OOM` in the results therefore means the KV representation did not fit --
   a real result -- rather than that the prompt was fed too greedily. Chunked
   prefill is bit-exact against a single-shot run on natural text.
2. Streaming methods (KV quantization) compress what is *written*, so later
   chunks and decoded tokens attend to the compressed history.
3. One-shot prompt compressors (SnapKV, StreamingLLM, merging, low-rank, DKV)
   run once on the resident prompt cache.
4. When a transform ran, the final prompt position is re-forwarded against the
   transformed cache before the first token is emitted. The logits produced
   during prefill attended to the *uncompressed* history; scoring them would
   credit the method with information its representation no longer holds.
5. Decoding is greedy. CRBench compares a method against the dense baseline on
   the identical query, and sampling noise is indistinguishable from a quality
   difference.

### The dense anchor

The dense baseline is run for real at **every** context length, on the same
samples, and every quality figure is retention relative to it. If the dense
reference cannot run at some length, that length is skipped and recorded rather
than imputed: a relative score with no reference is not a score. Substituting a
fixed 100% ceiling for the dense run above some length silently asserts that the
model answers perfectly there, which is exactly the claim a long-context
benchmark exists to test.

### Cache storage

Prefill and decode want opposite layouts, and both are used:

- Prefill grows a contiguous `DynamicCache`. A prefix view of a preallocated
  buffer has a head-dimension stride that does not match its logical shape, and
  PyTorch's SDPA falls back to its math backend for such inputs, materialising
  the full attention matrix. Measured on Qwen2.5-7B NF4 at 32768 tokens: 13.03
  GiB peak and 435 s of prefill, against 9.75 GiB and 16 s.
- Decode moves to preallocated storage written in place. `DynamicCache.update`
  is `torch.cat`, so it reallocates and copies the entire cache on every token;
  at 65536 tokens that cycle fragmented the allocator into host memory at 10.0 s
  per token.

---

## 5. Non-Destructive Score Recomputation

CRBench saves raw token predictions, ground truths, prefill/decode latencies, and device memory allocations into a versioned JSON manifest (`raw_results_v1.json`) before computing benchmark scores.

Researchers can recompute all scores under alternative $\alpha$ weights or utility formulations instantly without re-running models:

```bash
crbench recompute \
  --raw-file "results/standard_benchmark/raw_results_v1.json" \
  --alpha 0.70 \
  --formula linear
```

---

## 6. Raw Result Schema Specification (v2.0.0)

Output manifests conform to JSON Schema `2.0.0`:

```json
{
  "schema_version": "2.0.0",
  "benchmark_name": "crbench_standard",
  "timestamp": "2026-08-24 10:00:00",
  "environment": {
    "python_version": "3.12.x",
    "pytorch_version": "2.4.0",
    "device": "cuda:0",
    "os": "Linux-x86_64"
  },
  "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
  "scoring_config": {
    "utility_formula": "linear",
    "utility_alpha": 0.70,
    "resource_normalization_max": 100.0,
    "enable_part2": true
  },
  "query_results": [
    {
      "query_id": "single_niah_2048_0",
      "task_name": "single_niah",
      "context_length": 2048,
      "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
      "method_name": "kv_quant_int4",
      "budget_spec": 4.0,
      "dense_raw_score": 1.0,
      "method_raw_score": 1.0,
      "task_floor": 0.0,
      "normalized_quality": 100.0,
      "quality_retained_pct": 100.0,
      "dense_memory_bytes": 10485760.0,
      "method_memory_bytes": 2621440.0,
      "dense_effective_bpt": 16.0,
      "method_effective_bpt": 4.0,
      "resource_efficiency": 75.0,
      "part1_score": 92.5,
      "status": "SUCCESS"
    }
  ],
  "dataset_aggregates": [
    {
      "dataset_name": "crbench_standard",
      "method_name": "kv_quant_int4",
      "total_queries": 15,
      "successful_queries": 15,
      "failed_queries": 0,
      "mean_part1_score": 89.4,
      "median_part1_score": 90.0,
      "ci95_part1_score": [86.2, 92.1],
      "mean_normalized_quality": 95.2,
      "mean_resource_efficiency": 75.0
    }
  ]
}
```

---

## 7. Metric Provenance Classification

| Metric Name | Status | Method / Instrument |
| :--- | :--- | :--- |
| **Raw Task Score ($s_i$)** | `MEASURED` | Autoregressive generation evaluated against task ground truth |
| **Dense Reference ($s_{i,\text{dense}}$)** | `MEASURED` | Uncompressed Dense FP16 run on the exact identical prompt |
| **Normalized Retention ($Q_i$)** | `DERIVED` | Pairwise relative capability: $(s_m - s_{\text{floor}}) / \max(0.05, s_{\text{dense}} - s_{\text{floor}}) \times 100$ |
| **Effective Bits/Token ($b_{\text{eff}}$)** | `DERIVED (ANALYTICAL)` | Exact state formula including scale and index metadata |
| **Resource Efficiency ($R_i$)** | `DERIVED` | Memory savings: $100 \times \max(0, 1 - M_{\text{method}} / M_{\text{dense}})$ |
| **Part 1 Resource Score ($S_{1}$)** | `DERIVED` | Linear utility: $\alpha Q + (1-\alpha) R$ |
| **Resident KV bytes** | `MEASURED` | Storage of the cache tensors after the transform, summed per layer |
| **TTFT** | `MEASURED` | Device-synchronised wall clock, first prefill chunk to first emitted token |
| **Decode throughput** | `MEASURED` | Per-token intervals, each device-synchronised, over the decode stage only |
| **Latency jitter** | `MEASURED` | Standard deviation of the observed inter-token intervals |
| **Peak VRAM** | `MEASURED` | `max_memory_allocated` over the query, minus the resident weight baseline |
| **Part 2 System Score ($S_{2}$)** | `DERIVED` | System efficiency: $\alpha Q + (1-\alpha) R_{\text{sys}}$ |

Prefill and decode are timed as separate device-synchronised stages, and jitter
is the spread of intervals that were actually observed. `MEASURED` here means an
instrument read the value; anything reconstructed from a formula is labelled
`DERIVED`, including cases where a plausible-looking number could have been
produced by scaling a different metric.
