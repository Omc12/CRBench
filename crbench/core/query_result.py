"""
Query-Level Evaluation Architecture for CRBench.
================================================
Defines the atomic benchmarking unit:
  ONE MODEL + ONE QUERY/CONTEXT + DENSE REFERENCE + USER METHOD

Provides:
  - QueryEvaluationResult: Atomic measurement dataclass
  - QueryEvaluator: Single-query evaluation engine
  - QueryAggregationEngine: Dataset and benchmark aggregation
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import time
import math
import numpy as np
import torch

from crbench.tasks.base import BaseTask, EvaluationSample, SampleEvaluationResult
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata, ExecutionStatus
from crbench.adapters.dense import DenseAdapter
from crbench.scoring.normalizer import QualityNormalizer
from crbench.scoring.utility import (
    CRBENCH_ALPHA,
    CRBENCH_FORMULA_NAME,
    compute_utility,
    compute_query_resource_efficiency,
    compute_query_system_efficiency,
)


@dataclass
class QueryEvaluationResult:
    """
    Atomic benchmark result for ONE QUERY/CONTEXT.
    Compares user method directly against dense FP16 reference on the identical model & query.
    """
    query_id: str
    task_name: str
    context_length: int
    model_name: str
    method_name: str
    budget_spec: Any

    # Raw quality metrics
    dense_raw_score: float               # Raw uncompressed task score [0.0, 1.0] or [0.0, 100.0]
    method_raw_score: float              # Raw compressed method task score
    task_floor: float = 0.0              # Chance/floor performance

    # Normalized relative quality
    normalized_quality: float = 100.0    # Q ∈ [0.0, 100.0] (Dense = 100.0)
    quality_retained_pct: float = 100.0  # Percentage of dense capability retained

    # Memory & resource consumption
    dense_memory_bytes: float = 0.0      # Dense KV cache bytes
    method_memory_bytes: float = 0.0     # Method KV cache bytes
    dense_effective_bpt: float = 16.0    # Bits per KV element (16.0 for FP16)
    method_effective_bpt: float = 16.0   # Method effective bits per token
    resource_efficiency: float = 0.0     # R ∈ [0.0, 100.0] (% memory savings vs dense)

    # CRBench Part 1 Score
    part1_score: float = 0.0             # Utility(Q, R) (Quality + Memory only)

    # Part 2 Runtime metrics (Optional)
    dense_ttft_ms: Optional[float] = None
    method_ttft_ms: Optional[float] = None
    dense_decode_throughput: Optional[float] = None
    method_decode_throughput: Optional[float] = None
    dense_peak_vram_mb: Optional[float] = None
    method_peak_vram_mb: Optional[float] = None
    system_runtime_efficiency: Optional[float] = None  # R_sys ∈ [0.0, 100.0]
    part2_score: Optional[float] = None                # Utility(Q, R_sys) (Quality + Memory + Runtime)

    # Status & Predictions
    status: str = "SUCCESS"              # "SUCCESS", "OOM", "RUNTIME_ERROR", "UNSUPPORTED"
    error_message: Optional[str] = None
    dense_prediction: str = ""
    method_prediction: str = ""
    ground_truths: List[str] = field(default_factory=list)

    # Scoring configuration metadata
    formula_name: str = "linear"
    alpha: float = CRBENCH_ALPHA
    provenance: str = "measured_query_pair"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def format_summary(self) -> str:
        """Format clean user-readable summary for this individual query."""
        dense_mem_gb = self.dense_memory_bytes / (1024 ** 3) if self.dense_memory_bytes > 0 else (self.dense_effective_bpt * self.context_length * 2) / (1024 ** 3)
        method_mem_gb = self.method_memory_bytes / (1024 ** 3) if self.method_memory_bytes > 0 else (self.method_effective_bpt * self.context_length * 2) / (1024 ** 3)

        lines = [
            f"Query ID: {self.query_id} (Task: {self.task_name}, Context: {self.context_length:,} tokens)",
            f"Model: {self.model_name} | Method: {self.method_name} (Budget: {self.budget_spec})",
            f"  Dense quality:       {self.dense_raw_score * 100.0:.1f}%",
            f"  Method quality:      {self.method_raw_score * 100.0:.1f}%",
            f"  Quality retained:    {self.quality_retained_pct:.1f}%",
            f"  Dense memory:        {dense_mem_gb:.3f} GB ({self.dense_effective_bpt:.1f} bpt)",
            f"  Method memory:       {method_mem_gb:.3f} GB ({self.method_effective_bpt:.1f} bpt)",
            f"  Resource efficiency: {self.resource_efficiency:.1f}% savings",
            f"  CRBench Part 1 score: {self.part1_score:.2f} (Formula: {self.formula_name}, α={self.alpha:.2f})",
        ]
        if self.part2_score is not None:
            lines.append(f"  CRBench Part 2 score: {self.part2_score:.2f} (Runtime eff: {self.system_runtime_efficiency:.1f}%)")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> QueryEvaluationResult:
        return cls(**data)


class QueryEvaluator:
    """
    Evaluates a single query/context instance under Dense baseline and User Method.
    """

    def __init__(
        self,
        normalizer: Optional[QualityNormalizer] = None,
        alpha: float = CRBENCH_ALPHA,
        formula: str = "linear",
        enable_part2: bool = False,
    ):
        self.normalizer = normalizer or QualityNormalizer()
        self.alpha = alpha
        self.formula = formula
        self.enable_part2 = enable_part2

    def _generate_prediction(
        self,
        adapter: BaseContextAdapter,
        model: Any,
        tokenizer: Any,
        sample: EvaluationSample,
        max_new_tokens: int = 32
    ) -> Tuple[str, Dict[str, float]]:
        t0 = time.time()
        if model is None or tokenizer is None:
            if hasattr(adapter, "forward_or_generate"):
                dummy_ids = torch.zeros((1, 10), dtype=torch.long)
                adapter.forward_or_generate(dummy_ids, max_new_tokens=max_new_tokens)
            if hasattr(adapter, "generate_prediction"):
                pred = adapter.generate_prediction(sample)
            elif hasattr(adapter, "accuracy") and getattr(adapter, "accuracy", 1.0) < 1.0:
                pred = sample.ground_truths[0] if getattr(adapter, "accuracy", 1.0) > 0.5 else "wrong_answer"
            else:
                pred = sample.ground_truths[0] if sample.ground_truths else "sample_answer"
            runtime = {"ttft_ms": 10.0, "throughput": 50.0, "peak_vram_mb": 256.0}
            return pred, runtime

        if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": sample.full_prompt}],
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = sample.full_prompt

        dev = getattr(model, "device", torch.device("cpu"))
        inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=min(sample.context_length, 8192)).to(dev)
        out_tokens = adapter.forward_or_generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens
        )
        elapsed = time.time() - t0
        gen_text = tokenizer.decode(out_tokens[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        runtime = {
            "ttft_ms": elapsed * 500.0,
            "throughput": max_new_tokens / max(1e-3, elapsed),
            "peak_vram_mb": 0.0,
        }
        return gen_text, runtime

    def evaluate_query(
        self,
        model: Any,
        tokenizer: Any,
        sample: EvaluationSample,
        method_adapter: BaseContextAdapter,
        task: BaseTask,
        dense_adapter: Optional[BaseContextAdapter] = None,
        dense_cached_result: Optional[Tuple[SampleEvaluationResult, KVStateMetadata, Optional[Dict[str, float]]]] = None,
        device: Optional[torch.device] = None,
    ) -> QueryEvaluationResult:
        """
        Executes atomic evaluation for ONE QUERY/CONTEXT.
        """
        dev = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        ctx_len = sample.context_length
        model_name = getattr(getattr(model, "config", None), "_name_or_path", "unknown_model")

        # 1. Obtain Dense Baseline on this exact query
        if dense_cached_result is not None:
            dense_eval_res, dense_kv_meta, dense_runtime = dense_cached_result
        else:
            dense_ad = dense_adapter or DenseAdapter(name="dense_fp16_ref")
            if model is not None:
                dense_ad.prepare_model(model, tokenizer)
            
            dense_pred, dense_runtime = self._generate_prediction(dense_ad, model, tokenizer, sample, max_new_tokens=32)
            dense_eval_res = task.evaluate_prediction(dense_pred, sample)
            dense_kv_meta = dense_ad.get_kv_metadata(ctx_len)

        dense_raw = float(dense_eval_res.score)
        dense_bytes = float(dense_kv_meta.algorithmic_bytes) if dense_kv_meta.algorithmic_bytes > 0 else float(dense_kv_meta.effective_bits_per_element * ctx_len * 2)

        # 2. Check environment support for method
        is_supported, support_reason = method_adapter.validate_environment(dev)
        if not is_supported:
            return QueryEvaluationResult(
                query_id=sample.sample_id,
                task_name=task.name,
                context_length=ctx_len,
                model_name=model_name,
                method_name=method_adapter.name,
                budget_spec=getattr(method_adapter, "budget", "default"),
                dense_raw_score=dense_raw,
                method_raw_score=0.0,
                task_floor=task.floor_score,
                normalized_quality=0.0,
                quality_retained_pct=0.0,
                dense_memory_bytes=dense_bytes,
                method_memory_bytes=dense_bytes,
                dense_effective_bpt=dense_kv_meta.effective_bits_per_element,
                method_effective_bpt=dense_kv_meta.effective_bits_per_element,
                resource_efficiency=0.0,
                part1_score=0.0,
                status="UNSUPPORTED",
                error_message=support_reason,
                dense_prediction=dense_eval_res.prediction,
                ground_truths=sample.ground_truths,
                formula_name=self.formula,
                alpha=self.alpha,
            )

        # 3. Execute User Method on the identical query
        try:
            if model is not None:
                method_adapter.prepare_model(model, tokenizer)

            method_pred, method_runtime = self._generate_prediction(method_adapter, model, tokenizer, sample, max_new_tokens=32)
            method_eval_res = task.evaluate_prediction(method_pred, sample)
            method_kv_meta = method_adapter.get_kv_metadata(ctx_len)

            method_raw = float(method_eval_res.score)
            method_bytes = float(method_kv_meta.algorithmic_bytes) if method_kv_meta.algorithmic_bytes > 0 else float(method_kv_meta.effective_bits_per_element * ctx_len * 2)
            method_bpt = float(method_kv_meta.effective_bits_per_element)
            dense_bpt = float(dense_kv_meta.effective_bits_per_element)

            # 4. Normalize quality relative to Dense reference on this query
            norm_q = self.normalizer.normalize(
                raw_score=method_raw,
                dense_reference_score=dense_raw,
                task_floor=task.floor_score,
            )
            retained_pct = norm_q

            # 5. Compute Resource Efficiency relative to Dense reference
            # R in [0, 100]: % memory savings
            resource_eff = compute_query_resource_efficiency(
                dense_bytes=dense_bytes,
                method_bytes=method_bytes,
                dense_bpt=dense_bpt,
                method_bpt=method_bpt,
            )

            # 6. Compute CRBench Part 1 Score (Quality + Memory only)
            s_part1 = compute_utility(
                quality_score=norm_q,
                resource_efficiency=resource_eff,
                alpha=self.alpha,
                formula=self.formula,
            )

            # 7. Optional Part 2 Computation
            s_part2 = None
            sys_eff = None
            method_ttft = None
            method_thru = None
            method_vram = None

            if self.enable_part2:
                method_ttft = method_runtime.get("ttft_ms", 100.0)
                method_thru = method_runtime.get("throughput", 30.0)
                method_vram = method_runtime.get("peak_vram_mb", 0.0)

                sys_eff = compute_query_system_efficiency(
                    dense_bytes=dense_bytes,
                    method_bytes=method_bytes,
                    dense_ttft_ms=dense_runtime.get("ttft_ms", 500.0),
                    method_ttft_ms=method_ttft,
                    dense_throughput=dense_runtime.get("throughput", 30.0),
                    method_throughput=method_thru,
                )
                s_part2 = compute_utility(
                    quality_score=norm_q,
                    resource_efficiency=sys_eff,
                    alpha=self.alpha,
                    formula=self.formula,
                )

            return QueryEvaluationResult(
                query_id=sample.sample_id,
                task_name=task.name,
                context_length=ctx_len,
                model_name=model_name,
                method_name=method_adapter.name,
                budget_spec=getattr(method_adapter, "budget", "default"),
                dense_raw_score=dense_raw,
                method_raw_score=method_raw,
                task_floor=task.floor_score,
                normalized_quality=float(norm_q),
                quality_retained_pct=float(retained_pct),
                dense_memory_bytes=dense_bytes,
                method_memory_bytes=method_bytes,
                dense_effective_bpt=dense_bpt,
                method_effective_bpt=method_bpt,
                resource_efficiency=float(resource_eff),
                part1_score=float(s_part1),
                dense_ttft_ms=dense_runtime.get("ttft_ms"),
                method_ttft_ms=method_ttft,
                dense_decode_throughput=dense_runtime.get("throughput"),
                method_decode_throughput=method_thru,
                dense_peak_vram_mb=dense_runtime.get("peak_vram_mb"),
                method_peak_vram_mb=method_vram,
                system_runtime_efficiency=sys_eff,
                part2_score=s_part2,
                status="SUCCESS",
                dense_prediction=dense_eval_res.prediction if hasattr(dense_eval_res, "prediction") else str(dense_pred),
                method_prediction=str(method_pred),
                ground_truths=sample.ground_truths,
                formula_name=self.formula,
                alpha=self.alpha,
            )

        except torch.cuda.OutOfMemoryError as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return QueryEvaluationResult(
                query_id=sample.sample_id,
                task_name=task.name,
                context_length=ctx_len,
                model_name=model_name,
                method_name=method_adapter.name,
                budget_spec=getattr(method_adapter, "budget", "default"),
                dense_raw_score=dense_raw,
                method_raw_score=0.0,
                task_floor=task.floor_score,
                normalized_quality=0.0,
                quality_retained_pct=0.0,
                dense_memory_bytes=dense_bytes,
                method_memory_bytes=dense_bytes,
                dense_effective_bpt=dense_kv_meta.effective_bits_per_element,
                method_effective_bpt=dense_kv_meta.effective_bits_per_element,
                resource_efficiency=0.0,
                part1_score=0.0,
                status="OOM",
                error_message=str(e),
                dense_prediction=dense_eval_res.prediction,
                ground_truths=sample.ground_truths,
                formula_name=self.formula,
                alpha=self.alpha,
            )
        except Exception as e:
            return QueryEvaluationResult(
                query_id=sample.sample_id,
                task_name=task.name,
                context_length=ctx_len,
                model_name=model_name,
                method_name=method_adapter.name,
                budget_spec=getattr(method_adapter, "budget", "default"),
                dense_raw_score=dense_raw,
                method_raw_score=0.0,
                task_floor=task.floor_score,
                normalized_quality=0.0,
                quality_retained_pct=0.0,
                dense_memory_bytes=dense_bytes,
                method_memory_bytes=dense_bytes,
                dense_effective_bpt=dense_kv_meta.effective_bits_per_element,
                method_effective_bpt=dense_kv_meta.effective_bits_per_element,
                resource_efficiency=0.0,
                part1_score=0.0,
                status="RUNTIME_ERROR",
                error_message=str(e),
                dense_prediction=dense_eval_res.prediction,
                ground_truths=sample.ground_truths,
                formula_name=self.formula,
                alpha=self.alpha,
            )


@dataclass
class DatasetAggregateResult:
    """
    Aggregated benchmark results across query evaluation instances.
    """
    dataset_name: str
    model_name: str
    method_name: str
    total_queries: int
    successful_queries: int
    failed_queries: int

    # Aggregate Part 1 scores
    mean_part1_score: float
    median_part1_score: float
    std_part1_score: float
    ci95_part1_score: Tuple[float, float]

    # Aggregate Quality & Resource
    mean_normalized_quality: float
    median_normalized_quality: float
    mean_resource_efficiency: float
    median_resource_efficiency: float
    mean_effective_bpt: float

    # Aggregate Part 2 scores (if enabled)
    mean_part2_score: Optional[float] = None
    median_part2_score: Optional[float] = None
    ci95_part2_score: Optional[Tuple[float, float]] = None

    # Breakdowns
    task_breakdowns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    context_length_breakdowns: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Underlying atomic measurements
    query_results: List[QueryEvaluationResult] = field(default_factory=list)

    formula_name: str = "linear"
    alpha: float = CRBENCH_ALPHA

    def format_table(self) -> str:
        """Format markdown table for reporting."""
        lines = [
            f"## Aggregate Benchmark Evaluation: {self.method_name}",
            f"**Model:** {self.model_name} | **Dataset/Suite:** {self.dataset_name}",
            f"**Formula:** `{self.formula_name}` (α = {self.alpha:.2f}) | **Queries:** {self.successful_queries}/{self.total_queries} successful",
            "",
            "| Metric | Mean | Median | 95% CI |",
            "| :--- | :---: | :---: | :---: |",
            f"| **CRBench Part 1 Score** | **{self.mean_part1_score:.2f}** | {self.median_part1_score:.2f} | [{self.ci95_part1_score[0]:.2f}, {self.ci95_part1_score[1]:.2f}] |",
            f"| Normalized Quality Q | {self.mean_normalized_quality:.1f}% | {self.median_normalized_quality:.1f}% | — |",
            f"| Resource Efficiency R | {self.mean_resource_efficiency:.1f}% | {self.median_resource_efficiency:.1f}% | — |",
            f"| Effective Memory (bpt) | {self.mean_effective_bpt:.2f} bpt | — | — |",
        ]
        if self.mean_part2_score is not None:
            ci2 = self.ci95_part2_score or (0.0, 0.0)
            lines.append(f"| **CRBench Part 2 Score** | **{self.mean_part2_score:.2f}** | {self.median_part2_score:.2f} | [{ci2[0]:.2f}, {ci2[1]:.2f}] |")

        if self.context_length_breakdowns:
            lines += [
                "",
                "### Context Length Breakdown",
                "| Context Length | Part 1 Score | Mean Quality Q | Resource Eff R |",
                "| :---: | :---: | :---: | :---: |"
            ]
            for ctx, d in sorted(self.context_length_breakdowns.items()):
                lines.append(f"| {ctx:,} | **{d.get('part1_score', 0.0):.2f}** | {d.get('normalized_quality', 0.0):.1f}% | {d.get('resource_efficiency', 0.0):.1f}% |")

        return "\n".join(lines)


class QueryAggregationEngine:
    """
    Aggregates collections of QueryEvaluationResult objects into dataset-level statistics.
    """

    @staticmethod
    def aggregate(
        query_results: List[QueryEvaluationResult],
        dataset_name: str = "custom_dataset",
        bootstrap_samples: int = 1000,
        ci_level: float = 0.95,
    ) -> DatasetAggregateResult:
        """
        Aggregates query-level results into dataset metrics with bootstrap confidence intervals.
        """
        if not query_results:
            return DatasetAggregateResult(
                dataset_name=dataset_name,
                model_name="unknown",
                method_name="unknown",
                total_queries=0,
                successful_queries=0,
                failed_queries=0,
                mean_part1_score=0.0,
                median_part1_score=0.0,
                std_part1_score=0.0,
                ci95_part1_score=(0.0, 0.0),
                mean_normalized_quality=0.0,
                median_normalized_quality=0.0,
                mean_resource_efficiency=0.0,
                median_resource_efficiency=0.0,
                mean_effective_bpt=16.0,
                query_results=[],
            )

        model_name = query_results[0].model_name
        method_name = query_results[0].method_name
        formula_name = query_results[0].formula_name
        alpha = query_results[0].alpha

        successful = [q for q in query_results if q.status == "SUCCESS"]
        total_count = len(query_results)
        success_count = len(successful)
        failed_count = total_count - success_count

        if not successful:
            return DatasetAggregateResult(
                dataset_name=dataset_name,
                model_name=model_name,
                method_name=method_name,
                total_queries=total_count,
                successful_queries=0,
                failed_queries=failed_count,
                mean_part1_score=0.0,
                median_part1_score=0.0,
                std_part1_score=0.0,
                ci95_part1_score=(0.0, 0.0),
                mean_normalized_quality=0.0,
                median_normalized_quality=0.0,
                mean_resource_efficiency=0.0,
                median_resource_efficiency=0.0,
                mean_effective_bpt=16.0,
                query_results=query_results,
                formula_name=formula_name,
                alpha=alpha,
            )

        p1_scores = np.array([q.part1_score for q in successful])
        q_scores = np.array([q.normalized_quality for q in successful])
        r_scores = np.array([q.resource_efficiency for q in successful])
        bpt_scores = np.array([q.method_effective_bpt for q in successful])

        mean_p1 = float(np.mean(p1_scores))
        median_p1 = float(np.median(p1_scores))
        std_p1 = float(np.std(p1_scores, ddof=1)) if len(p1_scores) > 1 else 0.0

        # Bootstrap 95% CI
        if len(p1_scores) >= 3:
            boot_means = [np.mean(np.random.choice(p1_scores, size=len(p1_scores), replace=True)) for _ in range(bootstrap_samples)]
            alpha_tail = (1.0 - ci_level) / 2.0
            ci_low = float(np.percentile(boot_means, 100.0 * alpha_tail))
            ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha_tail)))
        else:
            ci_low, ci_high = mean_p1, mean_p1

        # Part 2 Aggregation (if present)
        p2_vals = [q.part2_score for q in successful if q.part2_score is not None]
        mean_p2, median_p2, ci95_p2 = None, None, None
        if p2_vals:
            p2_arr = np.array(p2_vals)
            mean_p2 = float(np.mean(p2_arr))
            median_p2 = float(np.median(p2_arr))
            if len(p2_arr) >= 3:
                boot2 = [np.mean(np.random.choice(p2_arr, size=len(p2_arr), replace=True)) for _ in range(bootstrap_samples)]
                ci95_p2 = (float(np.percentile(boot2, 2.5)), float(np.percentile(boot2, 97.5)))
            else:
                ci95_p2 = (mean_p2, mean_p2)

        # Context length breakdowns
        ctx_breakdowns = {}
        ctx_groups: Dict[int, List[QueryEvaluationResult]] = {}
        for q in successful:
            ctx_groups.setdefault(q.context_length, []).append(q)

        for ctx, q_list in ctx_groups.items():
            ctx_breakdowns[ctx] = {
                "part1_score": float(np.mean([x.part1_score for x in q_list])),
                "normalized_quality": float(np.mean([x.normalized_quality for x in q_list])),
                "resource_efficiency": float(np.mean([x.resource_efficiency for x in q_list])),
                "effective_bpt": float(np.mean([x.method_effective_bpt for x in q_list])),
                "count": len(q_list),
            }

        # Task breakdowns
        task_breakdowns = {}
        task_groups: Dict[str, List[QueryEvaluationResult]] = {}
        for q in successful:
            task_groups.setdefault(q.task_name, []).append(q)

        for t_name, q_list in task_groups.items():
            task_breakdowns[t_name] = {
                "part1_score": float(np.mean([x.part1_score for x in q_list])),
                "normalized_quality": float(np.mean([x.normalized_quality for x in q_list])),
                "resource_efficiency": float(np.mean([x.resource_efficiency for x in q_list])),
                "effective_bpt": float(np.mean([x.method_effective_bpt for x in q_list])),
                "count": len(q_list),
            }

        return DatasetAggregateResult(
            dataset_name=dataset_name,
            model_name=model_name,
            method_name=method_name,
            total_queries=total_count,
            successful_queries=success_count,
            failed_queries=failed_count,
            mean_part1_score=mean_p1,
            median_part1_score=median_p1,
            std_part1_score=std_p1,
            ci95_part1_score=(ci_low, ci_high),
            mean_normalized_quality=float(np.mean(q_scores)),
            median_normalized_quality=float(np.median(q_scores)),
            mean_resource_efficiency=float(np.mean(r_scores)),
            median_resource_efficiency=float(np.median(r_scores)),
            mean_effective_bpt=float(np.mean(bpt_scores)),
            mean_part2_score=mean_p2,
            median_part2_score=median_p2,
            ci95_part2_score=ci95_p2,
            task_breakdowns=task_breakdowns,
            context_length_breakdowns=ctx_breakdowns,
            query_results=query_results,
            formula_name=formula_name,
            alpha=alpha,
        )
