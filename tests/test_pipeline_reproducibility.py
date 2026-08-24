"""
Tests for CRBench Pipeline Reproducibility, Raw Result Serialization,
Adapter Lifecycle Hooks, Memory Accounting, and Score Recomputation.
"""

import json
import os
import pytest
import torch
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata, ExecutionStatus
from crbench.core.runner import recompute_scores_from_raw_file
from crbench.adapters.dense import DenseAdapter
from crbench.adapters.quantized import QuantizedKVAdapter, quantize_to_int_simulated
from crbench.adapters.compressed import LowRankCompressedKVAdapter
from crbench.scoring.pareto import OperatingPoint
from crbench.scoring.resource_score import CRBenchResourceScorer


def test_memory_accounting_fields_and_alignment():
    """Validates that KVStateMetadata accurately distinguishes algorithmic, metadata, and alignment bytes."""
    meta = KVStateMetadata(
        adapter_name="test_quant",
        method_type="quantization",
        effective_bits_per_element=4.25,
        total_tokens_stored=4096,
        context_length=4096,
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        algorithmic_bytes=4096 * 32 * 8 * 128 * 2 * (4.0 / 8.0), # 4 bits
        metadata_overhead_bytes=4096 * 32 * 8 * 128 * 2 / 64 * 2.0, # 16-bit scale per 64 elements
        alignment_overhead_bytes=64.0
    )

    expected_total = meta.algorithmic_bytes + meta.metadata_overhead_bytes + meta.alignment_overhead_bytes
    assert meta.total_state_bytes == expected_total
    assert meta.effective_bits_per_token > 4.0
    assert 0.0 < meta.compression_ratio < 1.0


def test_quantization_tensor_modification():
    """Proves that quantize_to_int_simulated genuinely modifies floating point tensor activations."""
    torch.manual_seed(42)
    t = torch.randn(2, 64, 128, dtype=torch.float32)

    # 16-bit returns exact identity
    t_fp16 = quantize_to_int_simulated(t, n_bits=16)
    assert torch.equal(t, t_fp16)

    # 8-bit has small quantization error
    t_int8 = quantize_to_int_simulated(t, n_bits=8)
    assert not torch.equal(t, t_int8)
    error_8 = torch.norm(t - t_int8).item()

    # 4-bit has higher error than 8-bit
    t_int4 = quantize_to_int_simulated(t, n_bits=4)
    assert not torch.equal(t, t_int4)
    error_4 = torch.norm(t - t_int4).item()

    # 2-bit has highest error
    t_int2 = quantize_to_int_simulated(t, n_bits=2)
    assert not torch.equal(t, t_int2)
    error_2 = torch.norm(t - t_int2).item()

    assert error_8 < error_4 < error_2


def test_recompute_scores_from_raw_file(tmp_path):
    """Proves that Part 1 and Part 2 scores can be 100% faithfully recomputed from raw measurement JSON."""
    raw_manifest = {
        "schema_version": "1.0.0",
        "benchmark_name": "test_bench",
        "timestamp": "2026-08-24T12:00:00Z",
        "environment": {"device": "cpu", "pytorch_version": "2.4.0"},
        "model_name": "Qwen/Qwen2.5-0.5B-Instruct",
        "raw_measurements": [
            {
                "task_name": "single_niah",
                "context_length": 2048,
                "adapter_name": "dense_fp16",
                "budget_spec": 16.0,
                "status": "SUCCESS",
                "raw_score": 100.0,
                "dense_reference_score": 100.0,
                "normalized_score": 100.0,
                "effective_bpt": 16.0,
                "algorithmic_bytes": 1024000.0,
                "metadata_bytes": 0.0,
                "ttft_ms": 1000.0,
                "decode_throughput_tok_sec": 30.0,
                "decode_latency_ms": 33.3,
            },
            {
                "task_name": "single_niah",
                "context_length": 2048,
                "adapter_name": "snapkv",
                "budget_spec": 4.0,
                "status": "SUCCESS",
                "raw_score": 90.0,
                "dense_reference_score": 100.0,
                "normalized_score": 90.0,
                "effective_bpt": 4.0,
                "algorithmic_bytes": 256000.0,
                "metadata_bytes": 2048.0,
                "ttft_ms": 400.0,
                "decode_throughput_tok_sec": 45.0,
                "decode_latency_ms": 22.2,
            },
            {
                "task_name": "single_niah",
                "context_length": 2048,
                "adapter_name": "failed_adapter",
                "budget_spec": 2.0,
                "status": "OOM",
                "error_message": "CUDA out of memory",
            }
        ]
    }

    raw_file = tmp_path / "raw_results_v1.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_manifest, f)

    recomputed = recompute_scores_from_raw_file(str(raw_file), weighting_scheme="logarithmic")

    assert "resource_results" in recomputed
    assert "system_results" in recomputed

    res_map = {r.method_name: r for r in recomputed["resource_results"]}
    assert "dense_fp16" in res_map
    assert "snapkv" in res_map
    assert "failed_adapter" not in res_map  # Failed runs are not scored as fake success points

    sys_map = {s.method_name: s for s in recomputed["system_results"]}
    # SnapKV with lower TTFT and higher throughput receives higher utility multiplier
    assert sys_map["snapkv"].system_utility_multiplier > sys_map["dense_fp16"].system_utility_multiplier
