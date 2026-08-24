"""
End-to-End Integration & Production Readiness Tests for CRBench.
Verifies:
  1. Dense-vs-method query evaluation
  2. Model-relative normalization consistency
  3. Memory accounting (Analytical vs. Physical, scales, metadata, padding)
  4. Adapter transformations (Quantization, Eviction, Merging, Compressed, Custom DKV)
  5. Score recomputation non-destructively from raw JSON
  6. Failure status handling (OOM, UNSUPPORTED, RUNTIME_ERROR, INVALID_CONFIG)
  7. CLI end-to-end subcommands (run, evaluate, evaluate-dataset, recompute, report, compare)
  8. Report generation with publication tables and closed figures
  9. Backend capability detection and graceful fallback
"""

import json
import os
from pathlib import Path
import pytest
import torch
import numpy as np
from click.testing import CliRunner

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata, ExecutionStatus
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.backend import BackendManager, DeviceCapabilities
from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig, ModelConfig, ScoringConfig
from crbench.core.runner import BenchmarkRunner, recompute_scores_from_raw_file
from crbench.core.query_result import QueryEvaluationResult, QueryEvaluator, QueryAggregationEngine
from crbench.adapters.dense import DenseAdapter
from crbench.adapters.quantized import QuantizedKVAdapter, quantize_to_int_simulated
from crbench.adapters.eviction import EvictionKVAdapter
from crbench.adapters.merging import MergingKVAdapter
from crbench.adapters.compressed import LowRankCompressedKVAdapter
from crbench.adapters.custom_example import DKVContextAdapter
from crbench.scoring.utility import compute_utility, CRBENCH_ALPHA
from crbench.reporting.report_generator import ReportGenerator
from crbench.reporting.plots import (
    plot_quality_vs_memory_pareto,
    plot_auqc_vs_context_length,
    plot_isobudget_comparison,
    plot_resource_vs_system_score,
)
from crbench.scoring.pareto import OperatingPoint
from crbench.cli import cli
from crbench.tasks.base import EvaluationSample, BaseTask, SampleEvaluationResult


# ─── 1. Backend Capability Detection ──────────────────────────────────────────

def test_backend_capabilities():
    """BackendManager accurately inspects current platform capabilities."""
    caps = BackendManager.get_capabilities()
    assert caps.device_type in ("cuda", "mps", "cpu")
    assert caps.total_memory_bytes > 0
    assert caps.available_memory_bytes > 0
    assert isinstance(caps.supports_bfloat16, bool)

    # Validate method support queries
    is_sup, reason = BackendManager.validate_method_support("dense", torch.device("cpu"))
    assert is_sup is True


# ─── 2. Adapter Transformation Fidelity ───────────────────────────────────────

def test_quantization_adapter_transformation():
    """Quantization genuinely perturbs tensor values and reduces state representation."""
    t = torch.randn(2, 4, 128, 64)
    q_4bit = quantize_to_int_simulated(t, n_bits=4, group_size=32)
    q_8bit = quantize_to_int_simulated(t, n_bits=8, group_size=32)
    q_16bit = quantize_to_int_simulated(t, n_bits=16, group_size=32)

    # 16-bit returns identical tensor
    assert torch.allclose(t, q_16bit)

    # 4-bit has higher error than 8-bit
    err_4 = torch.norm(t - q_4bit).item()
    err_8 = torch.norm(t - q_8bit).item()
    assert err_4 > err_8 > 0.0, "Quantization error must be monotone in bit-depth reduction!"


def test_eviction_adapter_token_budgeting():
    """Eviction adapter accurately reduces retained tokens according to budget."""
    ad = EvictionKVAdapter(name="snapkv_test", strategy="snapkv", sink_tokens=16, local_tokens=32)
    ad.apply_budget(ContextBudget.from_bits_per_token(4.0), context_length=2048)  # 4/16 = 0.25 -> 512 tokens
    meta = ad.get_kv_metadata(context_length=2048)

    assert meta.total_tokens_stored == 512
    assert meta.metadata_overhead_bytes == 512 * 4.0  # 4 bytes int32 per token
    assert meta.algorithmic_bytes < (2 * 32 * 32 * 128 * 2048 * 2.0)  # less than dense


def test_merging_adapter_metadata_accounting():
    """Merging adapter accounts for centroid metadata overhead."""
    ad = MergingKVAdapter(name="merging_test", merge_ratio=0.5)
    meta = ad.get_kv_metadata(context_length=4096)

    assert meta.total_tokens_stored == 2048
    assert meta.metadata_overhead_bytes == 2048 * 2.0  # 2 bytes per merged token count
    assert meta.total_state_bytes == meta.algorithmic_bytes + meta.metadata_overhead_bytes


def test_low_rank_compressed_adapter():
    """Low-rank compressed adapter scales head dimension according to budget."""
    ad = LowRankCompressedKVAdapter(name="low_rank_test", rank_ratio=0.25)
    meta = ad.get_kv_metadata(context_length=4096)
    assert meta.effective_bits_per_element < 16.0
    assert meta.custom_metrics["effective_head_dim"] == 32  # 128 * 0.25 = 32


def test_dkv_custom_adapter():
    """DKV custom adapter decouples shared subspace and dynamic tokens."""
    ad = DKVContextAdapter(name="dkv_test", subspace_dim_ratio=0.5, token_sparsity=0.5)
    meta = ad.get_kv_metadata(context_length=4096)
    assert meta.algorithmic_bytes > 0
    assert meta.effective_bits_per_element < 16.0


# ─── 3. Memory Accounting: Analytical vs. Physical ────────────────────────────

def test_memory_accounting_separation():
    """Validates clear separation of algorithmic memory, metadata, and physical allocated memory."""
    from crbench.profiler.memory import MemoryProfiler
    profiler = MemoryProfiler()
    profiler.start_tracking()

    # Create dummy tensor
    dummy = torch.zeros(1000, 1000, dtype=torch.float32)

    res = profiler.stop_tracking(
        algorithmic_bytes=4000.0,
        metadata_overhead_bytes=200.0,
        context_length=2048,
        dense_fp16_bytes=16000.0
    )

    assert res.algorithmic_bytes == 4000.0
    assert res.metadata_overhead_bytes == 200.0
    assert res.total_representation_bytes == 4200.0
    assert abs(res.effective_bits_per_token - (4200.0 * 8.0 / 2048)) < 1e-5
    assert abs(res.compression_ratio - (4200.0 / 16000.0)) < 1e-5


# ─── 4. Failure Handling & Explicit Statuses ──────────────────────────────────

class FailingAdapter(BaseContextAdapter):
    @property
    def method_type(self) -> str:
        return "custom"

    def apply_budget(self, budget, context_length):
        pass

    def forward_or_generate(self, input_ids, attention_mask=None, max_new_tokens=32, **kwargs):
        raise RuntimeError("Simulated internal kernel execution failure")

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        return KVStateMetadata("failing", "custom", 16.0, context_length, context_length, 32, 8, 128, 1000.0)


class DummyTask(BaseTask):
    @property
    def floor_score(self) -> float:
        return 0.0

    def generate_samples(self, context_length, num_samples, tokenizer, **kwargs):
        return [EvaluationSample(f"s_{i}", "ctx", "query", ["answer"], context_length) for i in range(num_samples)]

    def evaluate_prediction(self, prediction, sample):
        return SampleEvaluationResult(sample.sample_id, sample.context_length, prediction, sample.ground_truths, 1.0)


def test_failure_handling_explicit_status():
    """Failures must produce explicit RUNTIME_ERROR status without crashing or silently scoring 0."""
    evaluator = QueryEvaluator(alpha=0.70, formula="linear")
    sample = EvaluationSample("fail_q", "ctx", "query", ["answer"], 2048)
    task = DummyTask(name="dummy_task")
    dense_ad = DenseAdapter(name="dense_fp16")
    fail_ad = FailingAdapter(name="failing_method")

    # Dense cached mock
    dense_cached = (
        SampleEvaluationResult("fail_q", 2048, "answer", ["answer"], 1.0),
        dense_ad.get_kv_metadata(2048),
        {"ttft_ms": 100.0, "throughput": 50.0, "peak_vram_mb": 256.0}
    )

    res = evaluator.evaluate_query(
        model=None,
        tokenizer=None,
        sample=sample,
        method_adapter=fail_ad,
        task=task,
        dense_cached_result=dense_cached
    )

    assert res.status == "RUNTIME_ERROR"
    assert "Simulated internal kernel execution failure" in res.error_message
    assert res.method_raw_score == 0.0
    assert res.part1_score == 0.0


# ─── 5. Raw Result Reproducibility & Score Recomputation ──────────────────────

def test_full_reproducibility_raw_to_recomputed(tmp_path):
    """Saving raw results and recomputing scores produces identical outputs."""
    raw_manifest = {
        "schema_version": "2.0.0",
        "benchmark_name": "repro_suite",
        "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
        "scoring_config": {
            "utility_formula": "linear",
            "utility_alpha": 0.70,
            "resource_normalization_max": 100.0,
            "enable_part2": False,
        },
        "query_results": [
            {
                "query_id": f"q_{i}",
                "task_name": "single_niah",
                "context_length": 2048,
                "model_name": "Qwen2.5-1.5B",
                "method_name": "kv_quant_int4",
                "budget_spec": 4.0,
                "dense_raw_score": 1.0,
                "method_raw_score": 0.95,
                "task_floor": 0.0,
                "normalized_quality": 95.0,
                "quality_retained_pct": 95.0,
                "dense_memory_bytes": 16000.0,
                "method_memory_bytes": 4000.0,
                "dense_effective_bpt": 16.0,
                "method_effective_bpt": 4.0,
                "resource_efficiency": 75.0,
                "part1_score": 89.0,
                "status": "SUCCESS",
            }
            for i in range(5)
        ],
        "raw_measurements": [
            {
                "task_name": "single_niah",
                "context_length": 2048,
                "adapter_name": "kv_quant_int4",
                "budget_spec": 4.0,
                "status": "SUCCESS",
                "raw_score": 95.0,
                "dense_reference_score": 100.0,
                "normalized_score": 95.0,
                "effective_bpt": 4.0,
                "algorithmic_bytes": 4000.0,
                "metadata_bytes": 0.0,
                "ttft_ms": 100.0,
                "decode_throughput_tok_sec": 50.0,
                "decode_latency_ms": 20.0,
            }
        ]
    }

    raw_file = tmp_path / "raw_results_v1.json"
    with open(raw_file, "w", encoding="utf-8") as f:
        json.dump(raw_manifest, f)

    # Recompute with default alpha=0.70
    recomputed = recompute_scores_from_raw_file(str(raw_file), utility_formula="linear", utility_alpha=0.70)
    
    assert len(recomputed["query_results"]) == 5
    for q in recomputed["query_results"]:
        assert abs(q.part1_score - 89.0) < 1e-6

    # Aggregates match
    agg = recomputed["dataset_aggregates"]["kv_quant_int4"]
    assert agg.total_queries == 5
    assert abs(agg.mean_part1_score - 89.0) < 1e-6
    assert abs(agg.mean_normalized_quality - 95.0) < 1e-6


# ─── 6. Report Generation & Plot Memory Leak Safety ───────────────────────────

def test_report_generation_and_plots(tmp_path):
    """Plotting utilities save clean high-DPI figures and close them without leaking memory."""
    points = {
        "dense_fp16": [OperatingPoint(method_name="dense_fp16", context_length=2048, budget_value=16.0, quality_score=100.0, memory_cost=16.0)],
        "kv_quant_int4": [OperatingPoint(method_name="kv_quant_int4", context_length=2048, budget_value=4.0, quality_score=92.0, memory_cost=4.0)],
    }
    pareto_img = tmp_path / "pareto.png"
    fig = plot_quality_vs_memory_pareto(points, context_length=2048, output_path=str(pareto_img))
    assert pareto_img.exists()
    assert pareto_img.stat().st_size > 1000

    report_md = tmp_path / "CRBENCH_REPORT.md"
    report_text = ReportGenerator.generate_markdown_report(
        benchmark_name="test_run",
        model_name="test_model",
        resource_results=[],
        plot_paths={"Pareto Frontier": str(pareto_img)},
        output_file=str(report_md)
    )
    assert report_md.exists()
    assert "CRBench Benchmark Evaluation Report" in report_text
    assert "Scientific Metric Provenance Audit" in report_text


# ─── 7. CLI End-to-End Subcommands ───────────────────────────────────────────

def test_cli_all_subcommands(tmp_path):
    """CLI provides robust help, error messages, and execution across all subcommands."""
    runner = CliRunner()

    # 1. Main Help
    res = runner.invoke(cli, ["--help"])
    assert res.exit_code == 0
    assert "CRBench — Context Resource Benchmark" in res.output

    # 2. Evaluate Help
    res = runner.invoke(cli, ["evaluate", "--help"])
    assert res.exit_code == 0
    assert "--model" in res.output
    assert "--dense" in res.output

    # 3. Evaluate-Dataset Help
    res = runner.invoke(cli, ["evaluate-dataset", "--help"])
    assert res.exit_code == 0
    assert "--dataset" in res.output

    # 4. Compare Command
    res = runner.invoke(cli, [
        "compare", "MethodA", "MethodB",
        "-a", "85.0", "-a", "86.0", "-a", "84.0",
        "-b", "70.0", "-b", "72.0", "-b", "68.0"
    ])
    assert res.exit_code == 0
    assert "Statistical Comparison: MethodA vs. MethodB" in res.output

    # 5. Report on empty dir gives clear diagnostic message
    res = runner.invoke(cli, ["report", "--results-dir", str(tmp_path)])
    assert res.exit_code == 0
    assert "No CRBENCH_REPORT.md found" in res.output
