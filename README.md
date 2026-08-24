# CRBench — Context Resource Benchmark

[![Status: Research Preview](https://img.shields.io/badge/Status-Research%20Preview%20v0.2.0-orange.svg)](https://github.com/Omc12/CRBench)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

[Documentation](REPRODUCIBILITY.md) &bull; [Paper](#research-target-tmlr) &bull; [Examples](examples/) &bull; [Custom Adapter](examples/02_custom_adapter.py) &bull; [Test Suite](tests/) &bull; [Issues](https://github.com/Omc12/CRBench/issues)

---

### **CRBench measures how much contextual capability an LLM retains for the memory it uses.**

$$\text{Dense Reference} \longrightarrow \text{Candidate Method} \longrightarrow \text{Quality } Q + \text{Memory Savings } R \longrightarrow \text{CRBench Score } \mathcal{S}$$

$$\mathcal{S}_{\text{res}} = \alpha \cdot Q + (1 - \alpha) \cdot R_{\text{mem}} \quad (\text{default } \alpha = 0.70)$$

> **$Q$ measures contextual capability retained relative to the same model's dense reference; $R_{\text{mem}}$ is percentage memory savings relative to that reference.**

> **All scores are computed per query against the same model’s dense reference, then aggregated across evaluation sets.**

> **Status: Research Preview (v0.2.0).** Core implementation and query-level architecture are complete; broader empirical evaluation across larger models is ongoing.

---

## Why CRBench?

Existing long-context benchmarks primarily measure downstream task accuracy in isolation, while KV-cache compression papers often report compression ratios and task accuracies separately without a unified metric. 

**CRBench combines contextual capability retention and memory efficiency into a standardized, resource-aware score.**

> **CRBench does not rank models by absolute capability; it measures capability retained relative to each model's uncompressed dense baseline.**

---

## How the Score Works (in 10 Seconds)

CRBench evaluates each prompt pairwise against the model's own uncompressed baseline:

| Metric | Dense Reference | High-retention 4-bit | Low-quality 2-bit |
| :--- | :---: | :---: | :---: |
| **Quality retention ($Q$)** | 100% | 94% | 5% |
| **Memory savings ($R$)** | 0% | 75% | 87.5% |
| **CRBench Part 1 Score ($\mathcal{S}_{\text{res}}$)** | **70.0** | **88.3** *(higher resource utility)* | **29.8** *(penalized)* |

- **Part 1 (Resource Score)**: Evaluates quality retention vs. memory efficiency ($R_{\text{mem}}$). A method with 94% retention and 75% memory savings achieves higher resource utility than the uncompressed baseline.
- **Part 2 (System Score)**: Combines quality retention with runtime efficiency ($R_{\text{sys}}$ incorporating prefill TTFT speedup and decode throughput).

---

## Installation

```bash
git clone https://github.com/Omc12/CRBench.git
cd CRBench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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

### 2. Evaluate a Dataset (Query Aggregation)
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
Recompute scores from saved raw results under alternative $\alpha$ weights without re-running model inference:
```bash
crbench recompute \
  --raw-file "results/dataset_snapkv/raw_results_v1.json" \
  --alpha 0.70 \
  --formula linear
```

---

## Integrating a Custom Method (< 50 Lines)

Researchers can evaluate any novel KV representation by implementing `BaseContextAdapter`:

```python
import torch
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.registry import Registry

@Registry.register_adapter("my_custom_kv_method")
class MyCustomKVMethod(BaseContextAdapter):
    """Interface skeleton for custom KV representation and compression methods."""

    @property
    def method_type(self) -> str:
        return "custom"  # e.g., 'quantized', 'eviction', 'merging', 'custom'

    def apply_budget(self, budget, context_length: int) -> None:
        """Apply resource budget target (e.g. bits-per-token or retention ratio)."""
        super().apply_budget(budget, context_length)
        # Configure internal adapter parameters based on budget

    def forward_or_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        max_new_tokens: int = 32,
        **kwargs
    ) -> torch.Tensor:
        """Execute autoregressive generation under custom KV cache state."""
        # Attach custom KV kernel, attention mechanism, or forward hooks, then generate:
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **kwargs
        )

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        """Report theoretical KV tensor storage bytes and metadata overheads."""
        # Use actual model configuration dimensions:
        num_layers = getattr(self.model.config, "num_hidden_layers", 32)
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", 32)
        head_dim = getattr(self.model.config, "head_dim", 128)

        total_elements = 2 * num_layers * num_kv_heads * head_dim * context_length
        algorithmic_bytes = total_elements * 2.0 * 0.5  # 4-bit representation payload
        metadata_bytes = (total_elements / 64.0) * 2.0  # Scales, codebooks, or index overhead
        effective_bpe = (algorithmic_bytes + metadata_bytes) * 8.0 / max(1, total_elements)

        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=effective_bpe,
            total_tokens_stored=int(context_length * 0.5),
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            algorithmic_bytes=algorithmic_bytes,
            metadata_overhead_bytes=metadata_bytes
        )
```

---

## Supported Methods & Tasks

### Supported Memory Methods
- **KV Quantization**: FP8, INT8, INT4, INT2 (grouped / per-channel dynamic scaling with outlier protection)
- **KV Eviction / Pruning**: SnapKV, StreamingLLM, H2O (attention sinks, sliding window, heavy-hitter selection)
- **KV Merging / Clustering**: Temporal and semantic token centroid pooling
- **Low-Rank State**: Spectral and linear head-dimension subspace projection
- **Factorized KV**: Shared persistent subspace + sparse dynamic coefficients (`custom_dkv`)
- **Custom Adapters**: Extensible via `BaseContextAdapter`

### Evaluation Tasks
- **Needle-In-A-Haystack (NIAH)**: Single-target and multi-target associative retrieval
- **RULER Benchmark**: Multi-variable tracking and key-value pair aggregation
- **Multi-Hop QA**: Contextual cross-document reasoning
- **LongBench Tasks**: Extended context comprehension

---

## Hardware Support & Error Transparency

- **CUDA (Primary)**: Full GPU acceleration, synchronized latency profiling, and peak VRAM tracking.
- **Apple Silicon (MPS) & CPU**: Graceful fallback for lightweight testing and prototyping.
- **Explicit Failure Statuses**: OOM and runtime errors produce explicit status codes (`OOM`, `UNSUPPORTED`, `RUNTIME_ERROR`) and never silently score 0.

---

## Research Target: TMLR

CRBench is an active research project targeting **TMLR (Transactions on Machine Learning Research)**. Citation details will be updated upon preprint release.

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
