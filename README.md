# CRBench — Context Resource Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TMLR Benchmark](https://img.shields.io/badge/Target-TMLR-darkgreen.svg)](https://jmlr.org/tmlr/)

**CRBench (Context Resource Benchmark)** is a principled, method-agnostic research benchmark designed to characterize the **quality–resource tradeoff** of long-context Large Language Models (LLMs) under explicit memory and runtime resource constraints.

CRBench answers practical questions for researchers and practitioners:
- *"Under a 4 GB KV memory budget at 64K context, which representation retains the highest contextual capability?"*
- *"At 4 bits per token, does KV Quantization, KV Eviction (SnapKV), or KV Merging dominate the Pareto frontier?"*
- *"What is the real system throughput penalty (TTFT, decode tokens/s) of a compressed KV method?"*

---

## Key Features

1. **Method-Agnostic Adapter Pattern (`BaseContextAdapter`)**:
   - Easily benchmark FP16/BF16, FP8, INT8/INT4/INT2 quantization, KV eviction (SnapKV, StreamingLLM, H2O), token merging, low-rank compressed state, DKV, and custom memory representations.
2. **Two Independent Evaluation Tracks**:
   - **Part 1 — CRBench Resource Score ($\mathcal{S}_{\text{res}} \in [0, 100]$)**: Pure quality vs. memory representation efficiency without runtime metrics. Features Area Under the Quality-Resource Curve (AUQC), Iso-Budget Scores, 2D Hypervolume, and Pareto Frontiers.
   - **Part 2 — CRBench System Score ($\mathcal{S}_{\text{sys}} \in [0, 100]$)**: Real-world deployment utility combining task quality, physical VRAM, Time-to-First-Token (TTFT), prefill throughput, and decode latency via constrained utility.
3. **Rigorous Quality Normalization & Non-Arbitrary Scoring**:
   - Anchored against uncompressed Dense FP16 ($Q_{\text{dense}}(L)$) and task floor ($Q_{\text{floor}}$).
   - Satisfies mathematical axioms: Monotonicity, Pareto consistency, Scale invariance, and Resistance to gaming.
4. **Distinction between Algorithmic & Physical Memory**:
   - Explicitly computes analytical representation bits ($b_{\text{eff}}$) and measures actual peak GPU/MPS/CPU memory.
5. **Statistical Rigor**:
   - Bootstrap 95% Confidence Intervals, paired permutation hypothesis tests, Cohen's $d$, and Spearman/Kendall ranking stability.

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

## Quickstart

### Run Benchmark via CLI
```bash
# Execute quick stage 1 benchmark
crbench run --config configs/stage1_quick.yaml

# Execute standard benchmark across all tasks and context lengths
crbench run --config configs/stage2_standard.yaml
```

### Run Benchmark in Python
```python
from crbench.core.config import BenchmarkConfig
from crbench.core.runner import BenchmarkRunner

config = BenchmarkConfig.from_yaml("configs/stage1_quick.yaml")
runner = BenchmarkRunner(config)
runner.load_model()
results = runner.run()
```

---

## Integrating a Custom Method (< 50 Lines)

Researchers can evaluate any novel context representation by implementing `BaseContextAdapter`:

```python
import torch
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.registry import Registry

@Registry.register_adapter("my_custom_kv")
class MyCustomKVAdapter(BaseContextAdapter):
    @property
    def method_type(self) -> str:
        return "custom"

    def forward_or_generate(self, input_ids, attention_mask=None, max_new_tokens=32, **kwargs):
        # Attach your custom KV kernel or compressed inference mechanism
        return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        # Report exact analytical bytes and effective bits per token
        bytes_stored = 2 * 32 * 32 * 128 * context_length * 0.5  # Example 4-bit representation
        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=4.0,
            total_tokens_stored=context_length,
            context_length=context_length,
            num_layers=32,
            num_kv_heads=32,
            head_dim=128,
            algorithmic_bytes=bytes_stored
        )
```

---

## Paper & Research Specification

Detailed documentation and manuscript draft for the TMLR submission are located in:
- `paper/manuscript.md`: Full research paper draft.
- `paper/methodology.md`: Mathematical foundations of AUQC and Constrained System Utility.
- `paper/limitations.md`: Scope, boundaries, and hardware considerations.
- `paper/reproducibility.md`: Reproducibility checklist and environment specs.

---

## License
MIT License.
