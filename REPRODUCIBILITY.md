# CRBench Reproducibility & Protocol Guide

This guide details the exact environment setup, hardware recommendations, benchmark commands, score recomputation protocol, and result schema to ensure complete reproducibility of CRBench evaluations.

---

## 1. Environment Setup

### Prerequisites
- Python >= 3.9
- PyTorch >= 2.0.0 (with CUDA 12.x for NVIDIA GPUs, or MPS on Apple Silicon)
- Hugging Face `transformers` >= 4.40.0, `accelerate` >= 0.28.0

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

| Model Class | Target Context | Recommended Device | Minimum VRAM / RAM |
| :--- | :---: | :--- | :--- |
| **0.5B – 1.5B** (e.g. Qwen2.5-1.5B) | 2K – 8K | Apple Silicon (M-series), CPU, or GPU | 6 GB RAM / VRAM |
| **3B – 4B** (e.g. Qwen2.5-3B, Llama-3.2-3B) | 8K – 32K | RTX 4070 Super (12GB) / RTX 3090 / A5000 | 12 – 16 GB VRAM |
| **7B – 8B** (e.g. Llama-3.1-8B, Qwen2.5-7B) | 8K – 32K | NVIDIA A100 (40GB/80GB) / H100 | 24 – 40 GB VRAM |

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

---

## 4. Non-Destructive Score Recomputation

CRBench saves raw token predictions, ground truths, prefill/decode latencies, and device memory allocations into a versioned JSON manifest (`raw_results_v1.json`) before computing benchmark scores.

Researchers can recompute all scores under alternative $\alpha$ weights or utility formulations instantly without re-running models:

```bash
crbench recompute \
  --raw-file "results/standard_benchmark/raw_results_v1.json" \
  --alpha 0.70 \
  --formula linear
```

---

## 5. Raw Result Schema Specification (v2.0.0)

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

## 6. Metric Provenance Classification

| Metric Name | Status | Method / Instrument |
| :--- | :--- | :--- |
| **Raw Task Score ($s_i$)** | `MEASURED` | Autoregressive generation evaluated against task ground truth |
| **Dense Reference ($s_{i,\text{dense}}$)** | `MEASURED` | Uncompressed Dense FP16 run on the exact identical prompt |
| **Normalized Retention ($Q_i$)** | `DERIVED` | Pairwise relative capability: $(s_m - s_{\text{floor}}) / \max(0.05, s_{\text{dense}} - s_{\text{floor}}) \times 100$ |
| **Effective Bits/Token ($b_{\text{eff}}$)** | `DERIVED (ANALYTICAL)` | Exact state formula including scale and index metadata |
| **Resource Efficiency ($R_i$)** | `DERIVED` | Memory savings: $100 \times \max(0, 1 - M_{\text{method}} / M_{\text{dense}})$ |
| **Part 1 Resource Score ($S_{1}$)** | `DERIVED` | Linear utility: $\alpha Q + (1-\alpha) R$ |
| **TTFT & Throughput** | `MEASURED` | Synchronized GPU wall-clock profiler timing |
| **Part 2 System Score ($S_{2}$)** | `DERIVED` | System efficiency: $\alpha Q + (1-\alpha) R_{\text{sys}}$ |
