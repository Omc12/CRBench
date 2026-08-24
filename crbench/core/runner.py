"""
BenchmarkRunner: Main execution pipeline for CRBench experiments.
"""

from __future__ import annotations
import json
import os
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig
from crbench.core.registry import Registry
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.adapter import BaseContextAdapter
from crbench.adapters.dense import DenseAdapter
import crbench.adapters  # Registers all adapters
import crbench.tasks     # Registers all tasks
from crbench.tasks.base import BaseTask, EvaluationSample, TaskResult
from crbench.profiler.memory import MemoryProfiler, MemoryProfileResult
from crbench.profiler.latency import LatencyProfiler, LatencyProfileResult
from crbench.scoring.normalizer import QualityNormalizer
from crbench.scoring.pareto import OperatingPoint
from crbench.scoring.resource_score import CRBenchResourceScorer, CRBenchResourceScoreResult
from crbench.scoring.system_score import CRBenchSystemScorer, CRBenchSystemScoreResult, SystemRuntimeMetrics
from crbench.scoring.utility import (
    CRBENCH_ALPHA,
    CRBENCH_FORMULA_NAME,
    compute_utility,
    compute_query_resource_efficiency,
    compute_query_system_efficiency,
)
from crbench.core.query_result import (
    QueryEvaluationResult,
    QueryEvaluator,
    QueryAggregationEngine,
    DatasetAggregateResult,
)
from crbench.reporting.plots import (
    plot_quality_vs_memory_pareto,
    plot_auqc_vs_context_length,
    plot_isobudget_comparison,
    plot_resource_vs_system_score,
)
from crbench.reporting.report_generator import ReportGenerator


class BenchmarkRunner:
    """
    Coordinates model loading, task generation, adapter evaluation,
    resource profiling, scoring, statistical validation, and reporting.
    Supports atomic Query-Level and Dataset-Level evaluation.
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = self._resolve_device(config.model.device)
        self.tokenizer = None
        self.model = None
        self.normalizer = QualityNormalizer(
            floor_score=config.scoring.floor_quality,
            min_dynamic_range=config.scoring.min_dynamic_range
        )
        self.memory_profiler = MemoryProfiler(device=self.device)
        self.latency_profiler = LatencyProfiler(device=self.device)
        self.resource_scorer = CRBenchResourceScorer(
            log_scale_auqc=config.scoring.auqc_log_scale,
            weighting_scheme=config.scoring.context_weighting,
            standard_budgets_bpt=config.scoring.standard_budgets_bpt
        )
        self.system_scorer = CRBenchSystemScorer(
            alpha=config.scoring.utility_alpha,
            formula=config.scoring.utility_formula
        )
        self.query_evaluator = QueryEvaluator(
            normalizer=self.normalizer,
            alpha=config.scoring.utility_alpha,
            formula=config.scoring.utility_formula,
            enable_part2=config.scoring.enable_part2
        )

    def _resolve_device(self, device_str: str) -> torch.device:
        if device_str == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device_str)

    def load_model(self) -> None:
        """Loads Hugging Face model and tokenizer."""
        model_name = self.config.model.model_name_or_path
        print(f"[*] Loading model: {model_name} on {self.device}...", flush=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config.model.trust_remote_code
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16 if torch.cuda.is_bf16_supported() or self.device.type == "mps" else torch.float32,
                "float32": torch.float32
            }
            torch_dtype = dtype_map.get(self.config.model.dtype, torch.float32)

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=self.config.model.trust_remote_code,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            print(f"[✓] Model loaded successfully.")
        except Exception as e:
            print(f"[!] Warning: Could not load live model from HF ({e}). Initializing mock model for benchmark simulation.")
            self.tokenizer = None
            self.model = None

    def run(self) -> Dict[str, Any]:
        """
        Executes complete benchmark run across all configured tasks, context lengths, and adapters.
        """
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70, flush=True)
        print(f"Starting CRBench Execution: {self.config.benchmark_name}", flush=True)
        print("=" * 70, flush=True)

        # Dictionary tracking operating points: method_name -> context_length -> List[OperatingPoint]
        operating_points: Dict[str, Dict[int, List[OperatingPoint]]] = {}
        runtime_metrics_by_method: Dict[str, List[LatencyProfileResult]] = {}
        dense_scores_by_task_len: Dict[Tuple[str, int], float] = {}

        # 1. Initialize Adapters
        adapters: List[BaseContextAdapter] = []
        for ad_cfg in self.config.adapters:
            ad_cls = Registry.get_adapter(ad_cfg.adapter_type)
            ad_inst = ad_cls(name=ad_cfg.adapter_name, config=ad_cfg.params)
            adapters.append(ad_inst)
            operating_points[ad_inst.name] = {}
            runtime_metrics_by_method[ad_inst.name] = []

        raw_measurements: List[Dict[str, Any]] = []
        query_eval_results: List[QueryEvaluationResult] = []

        # 2. Iterate through Tasks
        for task_cfg in self.config.tasks:
            task_cls = Registry.get_task(task_cfg.task_name)
            task_inst: BaseTask = task_cls(name=task_cfg.task_name, seed=task_cfg.seed, config=task_cfg.task_kwargs)
            print(f"\n[+] Running Task: {task_inst.name}", flush=True)

            for ctx_len in task_cfg.context_lengths:
                print(f"  --> Context Length: {ctx_len:,} tokens (Samples: {task_cfg.num_samples})", flush=True)
                
                # Generate samples
                samples = task_inst.generate_samples(
                    context_length=ctx_len,
                    num_samples=task_cfg.num_samples,
                    tokenizer=self.tokenizer
                )

                # First establish dense reference score on this exact batch of queries
                dense_adapter = DenseAdapter(name="dense_fp16_ref")
                if self.model:
                    dense_adapter.prepare_model(self.model, self.tokenizer)
                
                dense_sample_map: Dict[str, SampleEvaluationResult] = {}
                dense_kv_meta = dense_adapter.get_kv_metadata(ctx_len)

                try:
                    dense_task_score = self._evaluate_adapter_on_task(dense_adapter, task_inst, samples)
                    dense_ref_val = max(1e-2, dense_task_score.mean_score)
                    dense_scores_by_task_len[(task_inst.name, ctx_len)] = dense_ref_val
                    for s_res in dense_task_score.sample_results:
                        dense_sample_map[s_res.sample_id] = s_res
                    print(f"      [Ref] Dense Baseline Raw Score: {dense_task_score.mean_score:.2f}%", flush=True)
                except Exception as e:
                    dense_ref_val = 1e-2
                    dense_scores_by_task_len[(task_inst.name, ctx_len)] = dense_ref_val
                    print(f"      [!] Dense Reference Error: {e}", flush=True)

                # Now evaluate each configured adapter under budget sweeps
                for ad_inst, ad_cfg in zip(adapters, self.config.adapters):
                    if self.model:
                        ad_inst.prepare_model(self.model, self.tokenizer)

                    if ctx_len not in operating_points[ad_inst.name]:
                        operating_points[ad_inst.name][ctx_len] = []

                    budgets = ad_cfg.budgets if ad_cfg.budgets else [16.0]

                    for b_val in budgets:
                        budget_obj = self._parse_budget(b_val, ad_inst.method_type)
                        ad_inst.apply_budget(budget_obj, ctx_len)

                        # Environment validation
                        is_supported, support_reason = ad_inst.validate_environment(self.device)
                        if not is_supported:
                            raw_measurements.append({
                                "task_name": task_inst.name,
                                "context_length": ctx_len,
                                "adapter_name": ad_inst.name,
                                "budget_spec": b_val,
                                "status": "UNSUPPORTED",
                                "error_message": support_reason
                            })
                            print(f"      [{ad_inst.name} | Budget={b_val}] UNSUPPORTED: {support_reason}", flush=True)
                            continue

                        # Evaluate Task Quality & Profile Latency
                        try:
                            task_res, lat_res = self._evaluate_adapter_with_profiling(ad_inst, task_inst, samples, ctx_len)
                            
                            # Normalize quality against dense baseline
                            dense_ref = dense_scores_by_task_len.get((task_inst.name, ctx_len), 1e-2)
                            norm_q = self.normalizer.normalize(
                                raw_score=task_res.mean_score,
                                dense_reference_score=dense_ref,
                                task_floor=task_inst.floor_score
                            )

                            # Memory Footprint
                            kv_meta = ad_inst.get_kv_metadata(ctx_len)
                            mem_cost = kv_meta.effective_bits_per_element
                            method_bytes = float(kv_meta.algorithmic_bytes) if kv_meta.algorithmic_bytes > 0 else float(mem_cost * ctx_len * 2)
                            dense_bytes = float(dense_kv_meta.algorithmic_bytes) if dense_kv_meta.algorithmic_bytes > 0 else float(16.0 * ctx_len * 2)

                            # Atomic Query Evaluation Results for each query
                            for s_idx, sample_obj in enumerate(samples):
                                d_sample_res = dense_sample_map.get(sample_obj.sample_id)
                                d_raw = float(d_sample_res.score) if d_sample_res else (dense_ref / 100.0)
                                m_sample_res = task_res.sample_results[s_idx] if s_idx < len(task_res.sample_results) else None
                                m_raw = float(m_sample_res.score) if m_sample_res else 0.0

                                q_norm = self.normalizer.normalize(m_raw, d_raw, task_floor=task_inst.floor_score)
                                r_eff = compute_query_resource_efficiency(dense_bytes, method_bytes, 16.0, mem_cost)
                                p1_s = compute_utility(q_norm, r_eff, alpha=self.config.scoring.utility_alpha, formula=self.config.scoring.utility_formula)

                                query_eval = QueryEvaluationResult(
                                    query_id=sample_obj.sample_id,
                                    task_name=task_inst.name,
                                    context_length=ctx_len,
                                    model_name=self.config.model.model_name_or_path,
                                    method_name=ad_inst.name,
                                    budget_spec=b_val,
                                    dense_raw_score=d_raw,
                                    method_raw_score=m_raw,
                                    task_floor=task_inst.floor_score,
                                    normalized_quality=float(q_norm),
                                    quality_retained_pct=float(q_norm),
                                    dense_memory_bytes=dense_bytes,
                                    method_memory_bytes=method_bytes,
                                    dense_effective_bpt=16.0,
                                    method_effective_bpt=float(mem_cost),
                                    resource_efficiency=float(r_eff),
                                    part1_score=float(p1_s),
                                    dense_ttft_ms=float(lat_res.ttft_ms),
                                    method_ttft_ms=float(lat_res.ttft_ms),
                                    dense_decode_throughput=float(lat_res.decode_throughput_tok_per_sec),
                                    method_decode_throughput=float(lat_res.decode_throughput_tok_per_sec),
                                    dense_prediction=d_sample_res.prediction if d_sample_res else "",
                                    method_prediction=m_sample_res.prediction if m_sample_res else "",
                                    ground_truths=sample_obj.ground_truths,
                                    formula_name=self.config.scoring.utility_formula,
                                    alpha=self.config.scoring.utility_alpha,
                                )
                                query_eval_results.append(query_eval)

                            op_pt = OperatingPoint(
                                method_name=ad_inst.name,
                                context_length=ctx_len,
                                budget_value=mem_cost,
                                quality_score=norm_q,
                                memory_cost=mem_cost,
                                latency_ms=lat_res.decode_latency_ms_per_token,
                                metadata={"raw_score": task_res.mean_score, "budget_spec": str(b_val)}
                            )
                            operating_points[ad_inst.name][ctx_len].append(op_pt)
                            runtime_metrics_by_method[ad_inst.name].append(lat_res)

                            raw_measurements.append({
                                "task_name": task_inst.name,
                                "context_length": ctx_len,
                                "adapter_name": ad_inst.name,
                                "budget_spec": b_val,
                                "status": "SUCCESS",
                                "raw_score": float(task_res.mean_score),
                                "dense_reference_score": float(dense_ref),
                                "normalized_score": float(norm_q),
                                "effective_bpt": float(mem_cost),
                                "algorithmic_bytes": float(kv_meta.algorithmic_bytes),
                                "metadata_bytes": float(kv_meta.metadata_overhead_bytes),
                                "ttft_ms": float(lat_res.ttft_ms),
                                "decode_throughput_tok_sec": float(lat_res.decode_throughput_tok_per_sec),
                                "decode_latency_ms": float(lat_res.decode_latency_ms_per_token),
                                "predictions": [s.prediction for s in task_res.sample_results]
                            })

                            print(f"      [{ad_inst.name} | Budget={b_val}] Raw: {task_res.mean_score:.1f}% -> Norm: {norm_q:.1f}% | Eff BPT: {mem_cost:.2f} bpt | Latency: {lat_res.decode_latency_ms_per_token:.1f}ms/tok", flush=True)

                        except torch.cuda.OutOfMemoryError as e:
                            raw_measurements.append({
                                "task_name": task_inst.name,
                                "context_length": ctx_len,
                                "adapter_name": ad_inst.name,
                                "budget_spec": b_val,
                                "status": "OOM",
                                "error_message": str(e)
                            })
                            print(f"      [{ad_inst.name} | Budget={b_val}] FAILED: Out of Memory (OOM)", flush=True)
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception as e:
                            raw_measurements.append({
                                "task_name": task_inst.name,
                                "context_length": ctx_len,
                                "adapter_name": ad_inst.name,
                                "budget_spec": b_val,
                                "status": "RUNTIME_ERROR",
                                "error_message": str(e)
                            })
                            print(f"      [{ad_inst.name} | Budget={b_val}] FAILED: Runtime Error ({e})", flush=True)

        # Dataset Aggregations
        dataset_aggregates = []
        if query_eval_results:
            method_groups: Dict[str, List[QueryEvaluationResult]] = {}
            for q in query_eval_results:
                method_groups.setdefault(q.method_name, []).append(q)
            for m_name, q_list in method_groups.items():
                agg = QueryAggregationEngine.aggregate(q_list, dataset_name=self.config.benchmark_name)
                dataset_aggregates.append(agg.to_dict() if hasattr(agg, "to_dict") else asdict(agg))

        # Save Raw Measurements JSON Schema v2.0.0 (Atomic Query Level)
        import platform
        raw_manifest = {
            "schema_version": "2.0.0",
            "benchmark_name": self.config.benchmark_name,
            "timestamp": str(os.popen("date").read().strip()),
            "environment": {
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "device": str(self.device),
                "os": platform.platform()
            },
            "model_name": self.config.model.model_name_or_path,
            "scoring_config": {
                "utility_formula": self.config.scoring.utility_formula,
                "utility_alpha": self.config.scoring.utility_alpha,
                "resource_normalization_max": self.config.scoring.resource_normalization_max,
                "enable_part2": self.config.scoring.enable_part2,
            },
            "query_results": [q.to_dict() for q in query_eval_results],
            "dataset_aggregates": dataset_aggregates,
            "raw_measurements": raw_measurements
        }
        raw_json_path = out_dir / "raw_results_v1.json"
        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_manifest, f, indent=2)
        print(f"[✓] Versioned query-level measurements saved: {raw_json_path}", flush=True)

        # 3. Part 1 Scoring: CRBench Resource Scores
        print("\n" + "=" * 70, flush=True)
        print("Computing Part 1 — CRBench Resource Scores (S_res)...", flush=True)
        print("=" * 70, flush=True)
        
        resource_results: List[CRBenchResourceScoreResult] = []
        for ad_name, ctx_pts in operating_points.items():
            if not any(pts for pts in ctx_pts.values()):
                continue
            res_score = self.resource_scorer.score_method(
                method_name=ad_name,
                operating_points_by_context=ctx_pts,
                min_budget_bound=1.0,
                max_budget_bound=16.0
            )
            resource_results.append(res_score)
            print(f"  * {ad_name:<20}: S_res = {res_score.resource_score:6.2f} (Mean AUQC: {res_score.mean_auqc:6.2f})", flush=True)

        # 4. Part 2 Scoring: CRBench System Scores
        print("\n" + "=" * 70, flush=True)
        print("Computing Part 2 — CRBench System Scores (S_sys)...", flush=True)
        print("=" * 70, flush=True)

        system_results: List[CRBenchSystemScoreResult] = []
        dense_profiles = runtime_metrics_by_method.get("dense_fp16", runtime_metrics_by_method.get("dense", []))
        if dense_profiles:
            dense_mean_ttft = float(sum(p.ttft_ms for p in dense_profiles) / max(1, len(dense_profiles)))
            dense_mean_thru = float(sum(p.decode_throughput_tok_per_sec for p in dense_profiles) / max(1, len(dense_profiles)))
        else:
            dense_mean_ttft = 1000.0
            dense_mean_thru = 30.0

        for res_score in resource_results:
            ad_name = res_score.method_name
            lat_profiles = runtime_metrics_by_method.get(ad_name, [])
            if not lat_profiles:
                continue
            
            mean_ttft = float(sum(p.ttft_ms for p in lat_profiles) / max(1, len(lat_profiles)))
            mean_pref_thru = float(sum(p.prefill_throughput_tok_per_sec for p in lat_profiles) / max(1, len(lat_profiles)))
            mean_dec_lat = float(sum(p.decode_latency_ms_per_token for p in lat_profiles) / max(1, len(lat_profiles)))
            mean_dec_thru = float(sum(p.decode_throughput_tok_per_sec for p in lat_profiles) / max(1, len(lat_profiles)))
            
            ctx_pts = operating_points.get(ad_name, {})
            all_pts = [pt for pts in ctx_pts.values() for pt in pts]
            mean_mem_cost_bpt = float(sum(p.memory_cost for p in all_pts) / max(1, len(all_pts))) if all_pts else 16.0
            vram_mb = 1024.0 * (mean_mem_cost_bpt / 16.0)

            sys_metrics = SystemRuntimeMetrics(
                mean_ttft_ms=mean_ttft,
                mean_prefill_throughput_tok_per_sec=mean_pref_thru,
                mean_decode_latency_ms_per_tok=mean_dec_lat,
                mean_decode_throughput_tok_per_sec=mean_dec_thru,
                peak_vram_mb=vram_mb
            )

            sys_res = self.system_scorer.score_system(
                part1_result=res_score,
                runtime_metrics=sys_metrics,
                reference_ttft_ms=dense_mean_ttft,
                reference_decode_throughput_tok_sec=dense_mean_thru
            )
            system_results.append(sys_res)
            print(f"  * {ad_name:<20}: S_sys = {sys_res.system_score:6.2f} (Util Mult: {sys_res.system_utility_multiplier:.2f}x | TTFT: {mean_ttft:.1f}ms)", flush=True)

        # 5. Visualizations & Reports
        plot_paths: Dict[str, str] = {}
        if self.config.save_plots:
            print("\nGenerating publication figures...", flush=True)
            fig_dir = out_dir / "figures"
            fig_dir.mkdir(exist_ok=True)
            
            # Flatten operating points for Pareto plotting
            all_pts_flat: Dict[str, List[OperatingPoint]] = {}
            for m_name, ctx_dict in operating_points.items():
                all_pts_flat[m_name] = [pt for pts in ctx_dict.values() for pt in pts]

            for task_cfg in self.config.tasks:
                for ctx_len in task_cfg.context_lengths:
                    p_path = str(fig_dir / f"pareto_frontier_{ctx_len}.png")
                    plot_quality_vs_memory_pareto(all_pts_flat, ctx_len, output_path=p_path)
                    plot_paths[f"Pareto Frontier ({ctx_len:,} tokens)"] = p_path

                    iso_path = str(fig_dir / f"isobudget_comparison_{ctx_len}.png")
                    plot_isobudget_comparison(resource_results, ctx_len, output_path=iso_path)
                    plot_paths[f"Iso-Budget Retention ({ctx_len:,} tokens)"] = iso_path

            auqc_path = str(fig_dir / "auqc_context_scaling.png")
            plot_auqc_vs_context_length(resource_results, output_path=auqc_path)
            plot_paths["AUQC Context Scaling"] = auqc_path

            sys_path = str(fig_dir / "resource_vs_system_score.png")
            plot_resource_vs_system_score(system_results, output_path=sys_path)
            plot_paths["Part 1 vs Part 2 System Tradeoff"] = sys_path
            print(f"[✓] Publication figures saved in {fig_dir}")

        # 6. Weighting Sensitivity Analysis
        from crbench.statistics.sensitivity import WeightingSensitivityAnalyzer
        sensitivity_analyzer = WeightingSensitivityAnalyzer(log_scale_auqc=self.config.scoring.auqc_log_scale)
        sensitivity_res = sensitivity_analyzer.analyze(operating_points)

        # 7. Generate Markdown Report
        report_file = str(out_dir / "CRBENCH_REPORT.md")
        ReportGenerator.generate_markdown_report(
            benchmark_name=self.config.benchmark_name,
            model_name=self.config.model.model_name_or_path,
            resource_results=resource_results,
            system_results=system_results,
            sensitivity_result=sensitivity_res,
            plot_paths=plot_paths,
            output_file=report_file
        )
        print(f"[✓] Final Benchmark Report generated: {report_file}")

        return {
            "resource_results": resource_results,
            "system_results": system_results,
            "operating_points": operating_points,
            "report_path": report_file,
        }

    def _parse_budget(self, val: Any, method_type: str) -> ContextBudget:
        if isinstance(val, (int, float)):
            if method_type in ("dense", "quantization"):
                return ContextBudget.from_bits_per_token(float(val))
            elif method_type in ("eviction", "merging", "compressed", "custom"):
                # If val <= 1.0, treat as ratio; if > 1.0, treat as bpt
                if float(val) <= 1.0:
                    return ContextBudget.from_compression_ratio(float(val))
                else:
                    return ContextBudget.from_bits_per_token(float(val))
        elif isinstance(val, dict):
            b_type = BudgetType(val.get("type", "bits_per_token"))
            return ContextBudget(budget_type=b_type, value=float(val.get("value", 16.0)))
        return ContextBudget.from_bits_per_token(16.0)

    def _evaluate_adapter_on_task(
        self,
        adapter: BaseContextAdapter,
        task: BaseTask,
        samples: List[EvaluationSample]
    ) -> TaskResult:
        res, _ = self._evaluate_adapter_with_profiling(adapter, task, samples, samples[0].context_length if samples else 0)
        return res

    def _evaluate_adapter_with_profiling(
        self,
        adapter: BaseContextAdapter,
        task: BaseTask,
        samples: List[EvaluationSample],
        context_length: int
    ) -> Tuple[TaskResult, LatencyProfileResult]:
        predictions: List[str] = []
        prompt_tok_count = context_length
        
        def run_inference() -> None:
            for s in samples:
                if self.model is None or self.tokenizer is None:
                    raise RuntimeError("Model and tokenizer must be loaded for live evaluation.")

                if hasattr(self.tokenizer, "apply_chat_template") and getattr(self.tokenizer, "chat_template", None):
                    prompt_text = self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": s.full_prompt}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    prompt_text = s.full_prompt

                inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=min(context_length, 8192)).to(self.device)
                out_tokens = adapter.forward_or_generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=32
                )
                gen_text = self.tokenizer.decode(out_tokens[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                predictions.append(gen_text)

        lat_profile = self.latency_profiler.benchmark_generation(
            generate_fn=run_inference,
            prompt_tokens=prompt_tok_count * max(1, len(samples)),
            max_new_tokens=16 * max(1, len(samples))
        )

        task_res = task.evaluate_batch(predictions, samples)
        
        # Immediate memory cleanup for Mac / GPU safety
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

        return task_res, lat_profile


def recompute_scores_from_raw_file(
    raw_results_path: str,
    weighting_scheme: str = "logarithmic",
    utility_formula: str = "linear",
    utility_alpha: float = CRBENCH_ALPHA,
) -> Dict[str, Any]:
    """
    Recomputes Part 1 Resource Scores, Part 2 System Scores, and Query-Level Dataset
    Aggregates directly from saved raw measurement JSON data without re-executing any models.
    """
    with open(raw_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    measurements = data.get("raw_measurements", [])
    raw_query_results = data.get("query_results", [])
    benchmark_name = data.get("benchmark_name", "recomputed_benchmark")

    # 1. Recompute Query-Level Results if present
    recomputed_queries: List[QueryEvaluationResult] = []
    if raw_query_results:
        for q_dict in raw_query_results:
            q_res = QueryEvaluationResult.from_dict(q_dict)
            # Recompute utility score with new formula and alpha
            q_res.formula_name = utility_formula
            q_res.alpha = utility_alpha
            q_res.part1_score = compute_utility(
                q_res.normalized_quality,
                q_res.resource_efficiency,
                alpha=utility_alpha,
                formula=utility_formula,
            )
            if q_res.system_runtime_efficiency is not None:
                q_res.part2_score = compute_utility(
                    q_res.normalized_quality,
                    q_res.system_runtime_efficiency,
                    alpha=utility_alpha,
                    formula=utility_formula,
                )
            recomputed_queries.append(q_res)

    # 2. Recompute Dataset Aggregates
    dataset_aggregates = {}
    if recomputed_queries:
        method_groups: Dict[str, List[QueryEvaluationResult]] = {}
        for q in recomputed_queries:
            method_groups.setdefault(q.method_name, []).append(q)
        for m_name, q_list in method_groups.items():
            dataset_aggregates[m_name] = QueryAggregationEngine.aggregate(q_list, dataset_name=benchmark_name)

    # 3. Recompute Operating Points and AUQC
    operating_points: Dict[str, Dict[int, List[OperatingPoint]]] = {}
    runtime_metrics_by_method: Dict[str, List[SystemRuntimeMetrics]] = {}

    for m in measurements:
        if m.get("status") != "SUCCESS":
            continue
        ad_name = m["adapter_name"]
        ctx_len = m["context_length"]
        mem_cost = m["effective_bpt"]
        norm_q = m["normalized_score"]
        lat = m.get("decode_latency_ms", 0.0)

        if ad_name not in operating_points:
            operating_points[ad_name] = {}
        if ctx_len not in operating_points[ad_name]:
            operating_points[ad_name][ctx_len] = []

        op_pt = OperatingPoint(
            method_name=ad_name,
            context_length=ctx_len,
            budget_value=mem_cost,
            quality_score=norm_q,
            memory_cost=mem_cost,
            latency_ms=lat,
            metadata={"raw_score": m.get("raw_score", 0.0)}
        )
        operating_points[ad_name][ctx_len].append(op_pt)

        if ad_name not in runtime_metrics_by_method:
            runtime_metrics_by_method[ad_name] = []
        runtime_metrics_by_method[ad_name].append(SystemRuntimeMetrics(
            mean_ttft_ms=m.get("ttft_ms", 1000.0),
            mean_prefill_throughput_tok_per_sec=1000.0,
            mean_decode_latency_ms_per_tok=lat,
            mean_decode_throughput_tok_per_sec=m.get("decode_throughput_tok_sec", 30.0),
            peak_vram_mb=1024.0 * (mem_cost / 16.0)
        ))

    scorer = CRBenchResourceScorer(weighting_scheme=weighting_scheme)
    resource_results = [
        scorer.score_method(name, pts)
        for name, pts in operating_points.items()
        if any(pts.values())
    ]

    sys_scorer = CRBenchSystemScorer(alpha=utility_alpha, formula=utility_formula)
    dense_profiles = runtime_metrics_by_method.get("dense_fp16", runtime_metrics_by_method.get("dense", []))
    dense_ttft = sum(p.mean_ttft_ms for p in dense_profiles) / max(1, len(dense_profiles)) if dense_profiles else 1000.0
    dense_thru = sum(p.mean_decode_throughput_tok_per_sec for p in dense_profiles) / max(1, len(dense_profiles)) if dense_profiles else 30.0

    system_results = []
    for r in resource_results:
        m_list = runtime_metrics_by_method.get(r.method_name, [])
        if not m_list:
            continue
        avg_metrics = SystemRuntimeMetrics(
            mean_ttft_ms=sum(x.mean_ttft_ms for x in m_list) / len(m_list),
            mean_prefill_throughput_tok_per_sec=sum(x.mean_prefill_throughput_tok_per_sec for x in m_list) / len(m_list),
            mean_decode_latency_ms_per_tok=sum(x.mean_decode_latency_ms_per_tok for x in m_list) / len(m_list),
            mean_decode_throughput_tok_per_sec=sum(x.mean_decode_throughput_tok_per_sec for x in m_list) / len(m_list),
            peak_vram_mb=sum(x.peak_vram_mb for x in m_list) / len(m_list)
        )
        sys_res = sys_scorer.score_system(
            part1_result=r,
            runtime_metrics=avg_metrics,
            reference_ttft_ms=dense_ttft,
            reference_decode_throughput_tok_sec=dense_thru,
            alpha=utility_alpha,
            formula=utility_formula,
        )
        system_results.append(sys_res)

    return {
        "resource_results": resource_results,
        "system_results": system_results,
        "operating_points": operating_points,
        "query_results": recomputed_queries,
        "dataset_aggregates": dataset_aggregates,
    }
