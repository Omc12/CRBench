# CRBench: TMLR Reproducibility Checklist & Experimental Protocol

This document satisfies the official **TMLR Reproducibility Checklist**.

---

## 1. Specification of Benchmark & Algorithms

- [x] **Full Mathematical Specification**: Formal definitions of Normalized Quality $\tilde{Q}$, AUQC, Iso-Budget Scores, and Constrained System Utility are fully documented in `paper/methodology.md`.
- [x] **Code Availability**: Complete, standalone Python package available with zero proprietary dependencies.
- [x] **Adapter Interface**: All baseline algorithms (Dense, FP8/INT8/INT4/INT2 quantization, SnapKV, StreamingLLM, Merging, Low-rank, DKV) inherit from `BaseContextAdapter`.

---

## 2. Experimental Setup & Hardware Specifications

- [x] **Hardware Environment**:
  - Development / Unit Testing Platform: Apple M3 (8-core CPU, Unified Memory, Apple Metal Performance Shaders - MPS).
  - Validation Platform: NVIDIA A100-SXM4-80GB / H100 PCIe (CUDA 12.4, PyTorch 2.4+).
- [x] **Software Dependencies**:
  - Python >= 3.9
  - PyTorch >= 2.0.0
  - Hugging Face Transformers >= 4.40.0
  - Accelerate >= 0.28.0
  - NumPy >= 1.24.0, SciPy >= 1.10.0, Pandas >= 2.0.0, Matplotlib >= 3.7.0, Seaborn >= 0.12.0

---

## 3. Execution Commands for Replication

```bash
# Clone and setup environment
git clone https://github.com/Omc12/CRBench.git
cd CRBench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Run test suite to verify mathematical invariants
pytest tests/ -v

# Quickstart Benchmark
crbench run --config configs/quickstart.yaml

# Standard Benchmark (all paradigms)
crbench run --config configs/standard_benchmark.yaml

# Scale-Up to Full 8B Model on NVIDIA GPUs (Meta-Llama-3.1-8B up to 32K context)
crbench run --config configs/cluster_8b.yaml
```

---

## 4. Seeds & Statistical Uncertainty Protocol

- Random seeds: $42, 1337, 2024$ for sample generation.
- Confidence Intervals: 2,000 non-parametric bootstrap resamples (95% CI).
- Hypothesis testing: 5,000 paired sign-flipping permutation resamples with two-tailed $p$-values.
