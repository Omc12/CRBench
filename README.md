# CRBench — Context Resource Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passing](https://img.shields.io/badge/Tests-142%20passed-success.svg)](tests/)
[![Target: TMLR](https://img.shields.io/badge/Target-TMLR-darkgreen.svg)](https://jmlr.org/tmlr/)

**CRBench (Context Resource Benchmark)** is a principled, method-agnostic research benchmark designed to characterize the **quality–resource tradeoff** of long-context Large Language Models (LLMs) under explicit memory and runtime resource constraints.

CRBench answers practical questions for researchers and practitioners:
- *"Under a 4 GB KV memory budget at 64K context, which representation retains the highest contextual capability?"*
- *"At 4 bits per token, does KV Quantization, KV Eviction (SnapKV), or KV Merging dominate the Pareto frontier?"*
- *"What is the real system throughput penalty (TTFT, decode tokens/s) of a compressed KV method?"*

---

## Key Methodological Principles

1. **Atomic Query-Level Evaluation**:
   The fundamental unit of evaluation is **(1 Model + 1 Query/Context + Dense Reference + User Method)**. Quality retention and memory savings are computed pairwise on the identical prompt.
2. **Model-Relative Normalization**:
   Quality retention measures the exact fraction of the base model's uncompressed Dense FP16 capability retained on that specific prompt:
   $$Q_i = 100 \cdot \frac{s_{i,\text{method}} - s_{\text{floor}}}{\max(\Delta_{\min}, s_{i,\text{dense}} - s_{\text{floor}})}$$
   Smaller models (e.g. 0.5B) are scored fairly on retention without being penalized for lower base model accuracy.
3. **Linear Additive Utility Formulation**:
   $$S_{\text{res}} = \alpha \cdot Q + (1 - \alpha) \cdot R_{\text{mem}}$$
   where $Q \in [0, 100]$ is capability retention and $R_{\text{mem}} = 100 \cdot \max\left(0, 1 - \frac{M_{\text{method}}}{M_{\text{dense}}}\right) \in [0, 100]$ is memory resource savings.
   - Dense FP16 ($Q=100, R=0$) is preserved at $S = 70.0$ (for default $\alpha=0.70$).
   - INT4 ($Q=95, R=75$) scores $89.0$ (legitimately beats Dense).
   - Failing INT2 ($Q=5, R=95$) scores $32.0$ (strictly penalized below Dense).
4. **Strict Separation of Part 1 and Part 2**:
   - **Part 1 (Resource Score)**: Quality + Memory only. No runtime metrics enter Part 1.
   - **Part 2 (System Score)**: Quality + System efficiency ($R_{\text{sys}}$ combining memory savings, TTFT prefill speedup, and decode throughput).
5. **Method-Agnostic Adapter API (`BaseContextAdapter`)**:
   Standardized plug-in interface for quantization, eviction (SnapKV, StreamingLLM, H2O), token merging, low-rank state, DKV, and custom researcher kernels.

---

## Installation

```bash
git clone https://github.com/Omc12/CRBench.git
cd CRBench
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Command-Line Interface (CLI)

### 1. Evaluate a Single Query (Atomic Primitive)
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

### 2. Evaluate a Dataset (Query-Level Aggregation)
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

### 3. Recompute Scores Non-Destructively from Raw JSON
```bash
crbench recompute \
  --raw-file "results/dataset_snapkv/raw_results_v1.json" \
  --alpha 0.70 \
  --formula linear
```

### 4. Execute Full Benchmark Suite
```bash
crbench run --config configs/stage1_quick.yaml
```

### 5. Generate Publication Report
```bash
crbench report --results-dir results/stage1_quick
```

### 6. Statistical Comparison Between Two Methods
```bash
crbench compare "Method_A" "Method_B" \
  -a 88.5 -a 89.0 -a 87.5 -a 90.0 \
  -b 72.0 -b 74.0 -b 71.5 -b 73.0
```

---

## Integrating a Custom Method (< 50 Lines)

To evaluate a novel KV representation, subclass `BaseContextAdapter`:

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
        # Report analytical bytes and metadata overheads
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

## Execution Status Codes & Failure Transparency

CRBench never replaces failed executions with silent zeros. Every query records an explicit status:

| Status Code | Meaning |
| :--- | :--- |
| `SUCCESS` | Successful autoregressive execution |
| `OOM` | Device Out-Of-Memory error caught and VRAM cleared |
| `UNSUPPORTED` | Context length or precision unsupported by hardware/kernel |
| `RUNTIME_ERROR` | Runtime failure surfaced transparently with error message |
| `INVALID_CONFIG` | Malformed budget or parameter configuration |

---

## Test Suite

The test suite contains 142 automated tests covering query-level evaluation, scientific axioms, score recomputation, memory accounting, adapter transformations, backend capability queries, and CLI commands:

```bash
source .venv/bin/activate
pytest tests/ -v
```

---

## License

CRBench is licensed under the MIT License. See [LICENSE](LICENSE) for details.
