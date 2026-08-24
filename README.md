# CRBench — Context Resource Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/Tests-142%20passed-success.svg)](tests/)

**CRBench (Context Resource Benchmark)** is a principled, method-agnostic research benchmark designed to evaluate and characterize the **quality–resource tradeoff** of long-context Large Language Models (LLMs) under explicit memory and runtime resource constraints.

As context lengths scale from 8K to 128K+ tokens, the key-value (KV) cache becomes the primary memory bottleneck during LLM inference. CRBench provides a unified, mathematically grounded framework to answer:
- *"Under a 4 GB KV memory budget at 64K context, which compression method retains the highest capability?"*
- *"At 4 bits per token, does KV Quantization, KV Eviction (e.g. SnapKV), or KV Merging dominate the Pareto frontier?"*
- *"What is the real system throughput penalty (TTFT, decode tokens/s) of a compressed KV method?"*

---

## Key Methodological Principles

### 1. Atomic Query-Level Evaluation
The fundamental evaluation unit in CRBench is:
$$\langle \text{Model}, \text{Query / Context}, \text{Dense Baseline}, \text{User Method} \rangle$$
Each prompt is evaluated under both the uncompressed reference (Dense FP16/BF16) and the candidate method, capturing pairwise quality retention and memory savings.

### 2. Model-Relative Normalization
Quality retention measures the fraction of the base model's own uncompressed capability retained on each specific prompt:
$$Q_i = 100 \cdot \frac{s_{i,\text{method}} - s_{\text{floor}}}{\max(\Delta_{\min}, s_{i,\text{dense}} - s_{\text{floor}})}$$
Smaller models (e.g. 0.5B – 1.5B) are scored fairly on contextual retention without being penalized for lower base model reasoning capacity.

### 3. Linear Additive Utility Formulation
$$\mathcal{S}_{\text{res}} = \alpha \cdot Q + (1 - \alpha) \cdot R_{\text{mem}}$$
where $Q \in [0, 100]$ is capability retention and $R_{\text{mem}} = 100 \cdot \max\left(0, 1 - \frac{M_{\text{method}}}{M_{\text{dense}}}\right) \in [0, 100]$ is percentage memory savings.
- **Dense FP16 reference** ($Q=100, R=0$) scores $70.0$ (at default $\alpha=0.70$).
- **High-retention INT4** ($Q=95, R=75$) scores $89.0$ (legitimately outscores Dense).
- **Failing INT2** ($Q=5, R=95$) scores $32.0$ (strictly penalized below Dense).
- $\alpha$ is configurable across all CLI tools and Python APIs.

### 4. Strict Separation: Part 1 vs. Part 2
- **Part 1 — Resource Score ($\mathcal{S}_{\text{res}}$)**: Evaluates quality retention vs. algorithmic/physical KV memory savings only. No runtime latency enters Part 1.
- **Part 2 — System Score ($\mathcal{S}_{\text{sys}}$)**: Combines quality retention with end-to-end system deployment efficiency ($R_{\text{sys}}$ incorporating prefill TTFT and decode throughput).

---

## Supported Methods & Paradigms

| Paradigm | Adapters Included | Key Mechanism |
| :--- | :--- | :--- |
| **Uncompressed Baseline** | `dense_fp16`, `dense_bf16` | Standard full-precision KV state (Reference Ceiling) |
| **KV Quantization** | `kv_quant_int8`, `kv_quant_int4`, `kv_quant_int2` | Dynamic grouped / channel quantization with outlier preservation |
| **KV Eviction / Pruning** | `snapkv`, `streaming_llm`, `h2o` | Attention sinks + sliding window + heavy-hitter token selection |
| **KV Merging / Clustering** | `kv_merging` | Temporal/semantic token centroid pooling and clustering |
| **Low-Rank State** | `low_rank_kv` | Spectral/linear head-dimension subspace reduction |
| **Factorized Representation** | `custom_dkv` | Shared persistent subspace + sparse dynamic coefficients |
| **Custom Researcher Methods** | `BaseContextAdapter` | Easily integrate any novel KV representation in < 50 lines |

---

## Installation

```bash
# Clone repository
git clone https://github.com/Omc12/CRBench.git
cd CRBench

# Setup Python virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with development dependencies
pip install -e ".[dev]"

# Verify installation with test suite
pytest tests/ -v
```

---

## Quickstart

### 1. Evaluate a Single Query (Atomic CLI Primitive)
```bash
crbench evaluate \
  --model "Qwen/Qwen2.5-1.5B-Instruct" \
  --query "What is the secret passkey?" \
  --context "The secret passkey is 987123." \
  --ground-truth "987123" \
  --method "kv_quant_int4" \
  --budget 4.0 \
  --dense
```

Output:
```text
========================================================================
CRBench Query Evaluation Summary
========================================================================
Query ID:             cli_query_001
Task:                 cli_evaluation_task (Context: 512 tokens)
Model:                Qwen/Qwen2.5-1.5B-Instruct
Method:               kv_quant_int4 (Budget: 4.0)
Status:               SUCCESS
------------------------------------------------------------------------
Quality Metrics:
  Dense raw score:     1.000
  Method raw score:    1.000
  Quality retained:    100.0%
Resource Metrics:
  Dense memory:        0.500 GB (16.0 bpt)
  Method memory:       0.125 GB (4.0 bpt)
  Resource efficiency: 75.0% savings
Benchmark Score:
  CRBench Part 1 score: 92.50 (Formula: linear, α=0.70)
========================================================================
```

### 2. Evaluate a Dataset with Query-Level Aggregation
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

### 3. Non-Destructive Score Recomputation
CRBench saves full, versioned raw measurement manifests (`raw_results_v1.json`) containing prompt predictions, ground truths, and memory allocations before computing scores.

Recompute scores under different $\alpha$ values without re-running models:
```bash
crbench recompute \
  --raw-file "results/dataset_snapkv/raw_results_v1.json" \
  --alpha 0.70 \
  --formula linear
```

### 4. Run Standard Benchmark Configuration
```bash
crbench run --config configs/quickstart.yaml
```

### 5. Generate Markdown / Publication Reports
```bash
crbench report --results-dir results/quickstart
```

### 6. Paired Statistical Hypothesis Testing
```bash
crbench compare "Method_A" "Method_B" \
  -a 88.5 -a 89.0 -a 87.5 -a 90.0 \
  -b 72.0 -b 74.0 -b 71.5 -b 73.0
```

---

## Implementing a Custom Method (< 50 Lines)

To evaluate a novel context representation, inherit from `BaseContextAdapter`:

```python
import torch
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.registry import Registry

@Registry.register_adapter("my_custom_kv")
class MyCustomKVAdapter(BaseContextAdapter):
    @property
    def method_type(self) -> str:
        return "custom"

    def forward_or_generate(self, input_ids: torch.Tensor, attention_mask=None, max_new_tokens=32, **kwargs):
        # Attach your custom KV kernel, hook, or compression mechanism
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens
        )

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        # Report exact analytical bytes and metadata overheads
        bytes_stored = 2 * 32 * 32 * 128 * context_length * 0.5  # 4-bit KV storage
        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=4.25,
            total_tokens_stored=context_length,
            context_length=context_length,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            algorithmic_bytes=bytes_stored,
            metadata_overhead_bytes=bytes_stored * 0.05
        )
```

---

## Hardware Backend Support & Safety

- **CUDA (Primary)**: Full GPU acceleration, synchronized latency profiling, and peak VRAM tracking.
- **Apple Silicon (MPS)**: Supported for lightweight development and prototyping with unified memory tracking.
- **CPU (Fallback)**: Graceful fallback for functional and algorithmic verification.
- **Failure Transparency**: OOM and runtime errors produce explicit status codes (`OOM`, `UNSUPPORTED`, `RUNTIME_ERROR`) and never silently score 0.

---

## Project Structure

```text
CRBench/
├── configs/                  # Benchmark configurations (quickstart, standard, cluster_8b)
├── crbench/
│   ├── adapters/             # KV cache adapters (Quantized, Eviction, Merging, Low-Rank, DKV)
│   ├── core/                 # Runner, query evaluation, registry, budget, backend
│   ├── profiler/             # Latency (TTFT/throughput) and memory profilers
│   ├── reporting/            # Automated reports, tables, and Pareto visualization
│   ├── scoring/              # Utility formulas, normalizer, AUQC, hypervolume
│   ├── statistics/           # Bootstrap CIs, paired permutation tests, stability
│   ├── tasks/                # NIAH, RULER, Multihop QA, LongBench tasks
│   ├── cli.py                # Command-line interface
│   └── __init__.py           # Public API
├── examples/                 # Python examples (quickstart, custom adapter, statistics)
├── scripts/                  # Cluster and automation scripts
├── tests/                    # 142 automated unit and integration tests
├── pyproject.toml            # Package configuration
├── README.md                 # Researcher onboarding guide
└── REPRODUCIBILITY.md        # Protocol and schema specification
```

---

## Planned Submission & Citation

CRBench is an active research benchmark planned for submission to **TMLR (Transactions on Machine Learning Research)**. Formal citation details will be provided upon preprint release.

```bibtex
@misc{crbench2026,
  title={CRBench: Context Resource Benchmark for Long-Context Large Language Models},
  author={CRBench Authors},
  year={2026},
  howpublished={\url{https://github.com/Omc12/CRBench}}
}
```

---

## License

CRBench is licensed under the [MIT License](LICENSE).
