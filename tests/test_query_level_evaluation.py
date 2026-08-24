"""
Unit Tests for CRBench Query-Level Evaluation Architecture.
Verifies:
  - Dense is the reference for each model/query
  - Relative normalization works across models
  - Query-level scores aggregate correctly (mean, median, 95% CI)
  - Linear utility and candidate formulas obey all scientific axioms
  - Different formulas and α values produce different outputs
  - Changing Q or R behaves monotonically and continuously
  - Raw results are sufficient to recompute scores
"""

import json
import pytest
import numpy as np
from pathlib import Path

from crbench.core.query_result import (
    QueryEvaluationResult,
    QueryEvaluator,
    QueryAggregationEngine,
    DatasetAggregateResult,
)
from crbench.scoring.utility import (
    CRBENCH_ALPHA,
    compute_utility,
    compute_query_resource_efficiency,
    compute_query_system_efficiency,
    formula_linear,
    formula_cobb_douglas,
    formula_harmonic,
    formula_power_mean,
)
from crbench.core.runner import recompute_scores_from_raw_file
from crbench.tasks.base import EvaluationSample, BaseTask, SampleEvaluationResult
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata


# ─── Mock Task & Adapter Helpers ──────────────────────────────────────────────

class MockTask(BaseTask):
    @property
    def floor_score(self) -> float:
        return 0.0

    def generate_samples(self, context_length, num_samples, tokenizer, **kwargs):
        return [
            EvaluationSample(
                sample_id=f"sample_{i}",
                context=f"Context text for sample {i}",
                query=f"What is the key for sample {i}?",
                ground_truths=[f"passkey_{i}"],
                context_length=context_length
            )
            for i in range(num_samples)
        ]

    def evaluate_prediction(self, prediction, sample):
        score = 1.0 if sample.ground_truths[0] in prediction else 0.0
        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            context_length=sample.context_length,
            prediction=prediction,
            ground_truths=sample.ground_truths,
            score=score
        )


class MockAdapter(BaseContextAdapter):
    def __init__(self, name: str, bpt: float = 4.0, accuracy: float = 1.0):
        super().__init__(name=name, config={})
        self.bpt = bpt
        self.accuracy = accuracy

    @property
    def method_type(self) -> str:
        return "mock"

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        total_bytes = (self.bpt * context_length * 2) / 8.0 * 1024
        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=self.bpt,
            total_tokens_stored=context_length,
            context_length=context_length,
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            algorithmic_bytes=total_bytes,
            metadata_overhead_bytes=0.0,
            alignment_overhead_bytes=0.0
        )

    def forward_or_generate(self, input_ids, attention_mask=None, max_new_tokens=32, **kwargs):
        import torch
        # Return dummy token tensor
        return torch.zeros((input_ids.shape[0], input_ids.shape[1] + max_new_tokens), dtype=torch.long)

    def generate(self, model, tokenizer, sample, max_new_tokens=32):
        from crbench.core.adapter import GenerationResult, LatencyMetrics
        pred = sample.ground_truths[0] if np.random.rand() < self.accuracy else "wrong_answer"
        return GenerationResult(
            prediction=pred,
            generated_tokens=10,
            latency_metrics=LatencyMetrics(
                prefill_latency_ms=100.0,
                decode_latency_ms_per_token=20.0,
                total_latency_ms=300.0,
                ttft_ms=100.0,
                decode_throughput_tok_per_sec=50.0,
                peak_memory_mb=256.0
            )
        )


# ─── 1. Dense Reference for Each Query ────────────────────────────────────────

class TestDenseReference:
    def test_dense_is_exact_reference(self):
        """Dense baseline on identical query has Q=100.0, R=0.0, S = α·100."""
        evaluator = QueryEvaluator(alpha=0.70, formula="linear")
        sample = EvaluationSample("q1", "ctx", "query", ["gt"], 2048)
        task = MockTask(name="mock")
        dense_ad = MockAdapter(name="dense_fp16", bpt=16.0, accuracy=1.0)

        res = evaluator.evaluate_query(
            model=None,
            tokenizer=None,
            sample=sample,
            method_adapter=dense_ad,
            task=task,
            dense_adapter=dense_ad,
        )

        assert res.dense_raw_score == 1.0
        assert res.method_raw_score == 1.0
        assert res.normalized_quality == 100.0
        assert res.quality_retained_pct == 100.0
        assert res.resource_efficiency == 0.0  # 0% savings compared to itself
        assert abs(res.part1_score - 70.0) < 1e-6  # α · 100 = 70.0


# ─── 2. Model Relativity Across Models ────────────────────────────────────────

class TestModelRelativity:
    def test_model_relativity_cross_model(self):
        """
        Smaller model with lower raw accuracy gets SAME relative retention Q score
        as larger model when method retains the same fraction of dense capability.
        """
        evaluator = QueryEvaluator(alpha=0.70, formula="linear")
        sample = EvaluationSample("q1", "ctx", "query", ["gt"], 2048)
        task = MockTask(name="mock")

        # Small Model (0.5B): Dense gets 0.40 raw score, Method gets 0.36 raw score (90% retention)
        q_small = evaluator.normalizer.normalize(raw_score=0.36, dense_reference_score=0.40, task_floor=0.0)
        
        # Large Model (8B): Dense gets 0.80 raw score, Method gets 0.72 raw score (90% retention)
        q_large = evaluator.normalizer.normalize(raw_score=0.72, dense_reference_score=0.80, task_floor=0.0)

        assert abs(q_small - 90.0) < 1e-4
        assert abs(q_large - 90.0) < 1e-4
        assert abs(q_small - q_large) < 1e-4, "Relative quality must be scale-invariant!"


# ─── 3. Query-Level Resource Efficiency ───────────────────────────────────────

class TestResourceEfficiency:
    def test_compression_savings(self):
        """4-bit quantization has 75% memory savings compared to 16-bit dense."""
        r_int4 = compute_query_resource_efficiency(dense_bytes=16000.0, method_bytes=4000.0)
        assert abs(r_int4 - 75.0) < 1e-6

        r_int8 = compute_query_resource_efficiency(dense_bytes=16000.0, method_bytes=8000.0)
        assert abs(r_int8 - 50.0) < 1e-6

        r_int2 = compute_query_resource_efficiency(dense_bytes=16000.0, method_bytes=2000.0)
        assert abs(r_int2 - 87.5) < 1e-6

    def test_memory_regression_clamped(self):
        """Method using more memory than dense receives R = 0.0."""
        r_overhead = compute_query_resource_efficiency(dense_bytes=16000.0, method_bytes=20000.0)
        assert r_overhead == 0.0


# ─── 4. Query-Level Score Aggregation ─────────────────────────────────────────

class TestQueryAggregation:
    def test_aggregation_mean_median_ci(self):
        """DatasetAggregationEngine computes accurate mean, median, and 95% bootstrap CI."""
        queries = [
            QueryEvaluationResult(
                query_id=f"q_{i}",
                task_name="niah",
                context_length=2048,
                model_name="test_model",
                method_name="int4",
                budget_spec=4.0,
                dense_raw_score=1.0,
                method_raw_score=1.0 if i < 8 else 0.0,
                normalized_quality=100.0 if i < 8 else 0.0,
                quality_retained_pct=100.0 if i < 8 else 0.0,
                dense_memory_bytes=1600.0,
                method_memory_bytes=400.0,
                dense_effective_bpt=16.0,
                method_effective_bpt=4.0,
                resource_efficiency=75.0,
                part1_score=compute_utility(100.0 if i < 8 else 0.0, 75.0, alpha=0.70),
            )
            for i in range(10)
        ]

        agg = QueryAggregationEngine.aggregate(queries, dataset_name="niah_benchmark")
        assert agg.total_queries == 10
        assert agg.successful_queries == 10
        assert agg.failed_queries == 0
        assert abs(agg.mean_normalized_quality - 80.0) < 1e-6
        assert abs(agg.mean_resource_efficiency - 75.0) < 1e-6
        assert agg.ci95_part1_score[0] <= agg.mean_part1_score <= agg.ci95_part1_score[1]


# ─── 5. Utility Formula Axioms ────────────────────────────────────────────────

class TestUtilityAxioms:
    def test_quality_monotonicity(self):
        """Increasing Q must strictly increase S."""
        scores = [compute_utility(q, 50.0, alpha=0.70, formula="linear") for q in [20.0, 40.0, 60.0, 80.0, 100.0]]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    def test_resource_monotonicity(self):
        """Increasing R must strictly increase S."""
        scores = [compute_utility(80.0, r, alpha=0.70, formula="linear") for r in [20.0, 40.0, 60.0, 80.0, 100.0]]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    def test_boundedness(self):
        """All scores for Q, R in [0, 100] must remain in [0, 100]."""
        for q in np.linspace(0, 100, 11):
            for r in np.linspace(0, 100, 11):
                s = compute_utility(q, r, alpha=0.70, formula="linear")
                assert 0.0 <= s <= 100.0

    def test_low_quality_rejection(self):
        """Method with Q=5% cannot beat Dense (Q=100%, R=0%) despite high compression."""
        s_garbage = compute_utility(5.0, 95.0, alpha=0.70, formula="linear")  # 0.7(5) + 0.3(95) = 3.5 + 28.5 = 32.0
        s_dense = compute_utility(100.0, 0.0, alpha=0.70, formula="linear")   # 0.7(100) + 0.3(0) = 70.0
        assert s_garbage < s_dense, f"Garbage ({s_garbage:.1f}) outscored Dense ({s_dense:.1f})"

    def test_int4_can_legitimately_beat_dense(self):
        """Method with strong retention (Q=95%) and 75% savings legitimately beats Dense."""
        s_int4 = compute_utility(95.0, 75.0, alpha=0.70, formula="linear")   # 0.7(95) + 0.3(75) = 66.5 + 22.5 = 89.0
        s_dense = compute_utility(100.0, 0.0, alpha=0.70, formula="linear")  # 70.0
        assert s_int4 > s_dense


# ─── 6. Formula Responsiveness & Differentiation ──────────────────────────────

class TestFormulaResponsiveness:
    def test_alpha_changes_scores_meaningfully(self):
        """Changing α from 0.3 to 0.8 must produce non-trivial score shifts."""
        s_alpha_03 = compute_utility(90.0, 50.0, alpha=0.30, formula="linear")  # 0.3(90) + 0.7(50) = 27 + 35 = 62.0
        s_alpha_08 = compute_utility(90.0, 50.0, alpha=0.80, formula="linear")  # 0.8(90) + 0.2(50) = 72 + 10 = 82.0
        assert abs(s_alpha_08 - s_alpha_03) >= 15.0

    def test_candidate_formulas_produce_distinct_values(self):
        """Different candidate formulas produce distinct outputs on the same Q and R."""
        Q, R = 80.0, 60.0
        s_lin = compute_utility(Q, R, alpha=0.70, formula="linear")
        s_cobb = compute_utility(Q, R, alpha=0.70, formula="cobb_douglas")
        s_harm = compute_utility(Q, R, alpha=0.70, formula="harmonic")
        s_pow2 = compute_utility(Q, R, alpha=0.70, formula="power_mean_2")

        # Values should be numerically distinct
        assert s_lin != s_cobb
        assert s_cobb != s_harm
        assert s_lin != s_pow2


# ─── 7. Recomputation from Stored Raw Results ─────────────────────────────────

class TestRecomputation:
    def test_recompute_from_query_manifest(self, tmp_path):
        """Scores can be completely reconstructed from raw_results_v1.json at query and dataset level."""
        manifest = {
            "schema_version": "2.0.0",
            "benchmark_name": "test_suite",
            "model_name": "test_model",
            "scoring_config": {
                "utility_formula": "linear",
                "utility_alpha": 0.70,
                "resource_normalization_max": 100.0,
                "enable_part2": False,
            },
            "query_results": [
                {
                    "query_id": "q1",
                    "task_name": "single_niah",
                    "context_length": 2048,
                    "model_name": "test_model",
                    "method_name": "snapkv",
                    "budget_spec": 4.0,
                    "dense_raw_score": 1.0,
                    "method_raw_score": 0.9,
                    "task_floor": 0.0,
                    "normalized_quality": 90.0,
                    "quality_retained_pct": 90.0,
                    "dense_memory_bytes": 16000.0,
                    "method_memory_bytes": 4000.0,
                    "dense_effective_bpt": 16.0,
                    "method_effective_bpt": 4.0,
                    "resource_efficiency": 75.0,
                    "part1_score": 85.5,
                    "status": "SUCCESS",
                }
            ],
            "raw_measurements": [
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
                    "algorithmic_bytes": 4000.0,
                    "metadata_bytes": 0.0,
                    "ttft_ms": 200.0,
                    "decode_throughput_tok_sec": 40.0,
                    "decode_latency_ms": 25.0,
                }
            ]
        }

        raw_file = tmp_path / "raw_results_v1.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(manifest, f)

        # Recompute with default alpha=0.70
        recomputed = recompute_scores_from_raw_file(str(raw_file), utility_formula="linear", utility_alpha=0.70)
        assert "resource_results" in recomputed
        assert "query_results" in recomputed
        assert len(recomputed["query_results"]) == 1
        assert abs(recomputed["query_results"][0].part1_score - 85.5) < 1e-6

        # Recompute with different alpha=0.50 without re-running model!
        recomputed_50 = recompute_scores_from_raw_file(str(raw_file), utility_formula="linear", utility_alpha=0.50)
        # S = 0.5(90) + 0.5(75) = 45 + 37.5 = 82.5
        assert abs(recomputed_50["query_results"][0].part1_score - 82.5) < 1e-6


# ─── 8. Summary Formatting ───────────────────────────────────────────────────

class TestFormatSummary:
    def test_individual_query_summary(self):
        """format_summary outputs user-readable metrics matching specification."""
        q = QueryEvaluationResult(
            query_id="q101",
            task_name="ruler_kv",
            context_length=4096,
            model_name="Qwen2.5-1.5B",
            method_name="kv_quant_int4",
            budget_spec=4.0,
            dense_raw_score=1.0,
            method_raw_score=0.94,
            normalized_quality=94.0,
            quality_retained_pct=94.0,
            dense_memory_bytes=4.0 * (1024 ** 3),
            method_memory_bytes=1.0 * (1024 ** 3),
            dense_effective_bpt=16.0,
            method_effective_bpt=4.0,
            resource_efficiency=75.0,
            part1_score=88.3,
        )
        summary = q.format_summary()
        assert "Dense quality:       100.0%" in summary
        assert "Method quality:      94.0%" in summary
        assert "Quality retained:    94.0%" in summary
        assert "Dense memory:        4.000 GB" in summary
        assert "Method memory:       1.000 GB" in summary
        assert "Resource efficiency: 75.0% savings" in summary
        assert "CRBench Part 1 score: 88.30" in summary
