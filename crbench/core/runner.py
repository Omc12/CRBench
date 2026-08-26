"""
BenchmarkRunner: Main execution pipeline for CRBench experiments.
"""

from __future__ import annotations
import json
import os
import math
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig
from crbench.core.registry import Registry
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.adapter import BaseContextAdapter
from crbench.core.inference import GenerationTrace, chunked_prefill_generate
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
from crbench.scoring.coverage import (
    CoverageRecord, coverage_by_cell, dense_anchor_usable, roll_up,
)
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
        # Device allocation attributable to weights alone; every peak-VRAM
        # figure downstream is reported relative to this.
        self._weight_bytes: int = 0
        # The RoPE parameters actually handed to the model, recorded in the
        # manifest so a long-context score can be read in the right regime.
        self._effective_rope: Optional[Dict[str, Any]] = None
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
        """Loads Hugging Face model and tokenizer with support for 4-bit / 8-bit quantization."""
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

            kwargs: Dict[str, Any] = {
                "torch_dtype": torch_dtype,
                "trust_remote_code": self.config.model.trust_remote_code,
                "low_cpu_mem_usage": True,
            }

            if self.config.model.attn_implementation:
                kwargs["attn_implementation"] = self.config.model.attn_implementation

            # RoPE scaling must be an explicit, recorded decision: Qwen2.5's
            # native window is 32768 and anything beyond it without YaRN
            # measures positional extrapolation, not the KV representation.
            if self.config.model.rope_scaling or self.config.model.config_overrides:
                from transformers import AutoConfig
                hf_cfg = AutoConfig.from_pretrained(
                    model_name, trust_remote_code=self.config.model.trust_remote_code
                )
                for key, value in (self.config.model.config_overrides or {}).items():
                    setattr(hf_cfg, key, value)
                    print(f"[*] Config override: {key} = {value!r}", flush=True)
                kwargs["config"] = hf_cfg
                # transformers 5.x keeps RoPE settings in one dict (rope_parameters,
                # aliased as rope_scaling) that also carries rope_theta. Replacing
                # it outright drops the base wavelength and YaRN initialisation
                # fails on `None ** tensor`, so merge into what the model shipped.
            if self.config.model.rope_scaling:
                existing = dict(getattr(hf_cfg, "rope_parameters", None)
                                or getattr(hf_cfg, "rope_scaling", None) or {})
                existing.update(self.config.model.rope_scaling)
                if existing.get("rope_theta") is None and getattr(hf_cfg, "rope_theta", None):
                    existing["rope_theta"] = hf_cfg.rope_theta
                if hasattr(hf_cfg, "rope_parameters"):
                    hf_cfg.rope_parameters = existing
                else:
                    hf_cfg.rope_scaling = existing
                if self.config.model.max_model_len:
                    hf_cfg.max_position_embeddings = int(self.config.model.max_model_len)
                self._effective_rope = existing
                print(f"[*] RoPE scaling enabled: {existing}", flush=True)

            # Quantization support (BitsAndBytes)
            if self.config.model.load_in_4bit or self.config.model.load_in_8bit:
                try:
                    from transformers import BitsAndBytesConfig
                    compute_dtype = dtype_map.get(self.config.model.bnb_4bit_compute_dtype, torch.bfloat16)
                    if self.config.model.load_in_4bit:
                        kwargs["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=compute_dtype,
                            bnb_4bit_quant_type=self.config.model.bnb_4bit_quant_type,
                            bnb_4bit_use_double_quant=self.config.model.bnb_4bit_use_double_quant,
                        )
                    elif self.config.model.load_in_8bit:
                        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                except ImportError:
                    if self.config.model.load_in_4bit:
                        kwargs["load_in_4bit"] = True
                    elif self.config.model.load_in_8bit:
                        kwargs["load_in_8bit"] = True

                device_map = self.config.model.device_map or "auto"
                kwargs["device_map"] = device_map
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            else:
                if self.config.model.device_map:
                    kwargs["device_map"] = self.config.model.device_map
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs).to(self.device)

            self.model.eval()
            import gc, ctypes
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            except Exception:
                pass

            # Everything allocated at this point is weights; every later peak is
            # reported relative to it, so the KV figures are not inflated by the
            # model's own footprint.
            if torch.cuda.is_available() and self.device.type == "cuda":
                self._weight_bytes = int(torch.cuda.memory_allocated(self.device))
                print(f"[OK] Model loaded. Weights resident: "
                      f"{self._weight_bytes / 2**30:.2f} GiB", flush=True)
            else:
                self._weight_bytes = 0
                print("[OK] Model loaded successfully.")
        except Exception as e:
            # A failed load must not silently degrade into a simulation: every
            # number downstream would be fabricated. Fail loudly instead.
            raise RuntimeError(
                f"Could not load model '{model_name}' on {self.device}: {e}"
            ) from e

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

        completed_groups, replay_queries, replay_measurements = self._load_checkpoint()
        if replay_measurements:
            raw_measurements.extend(replay_measurements)
            query_eval_results.extend(
                QueryEvaluationResult.from_dict(q) for q in replay_queries)
        # Measured peak device allocation above the weight baseline, per method
        # and context length; feeds the Part 2 system score in place of the
        # bits-per-token proxy it used to be derived from.
        peak_vram_by_method_ctx: Dict[Tuple[str, int], float] = {}

        # 2. Iterate through Tasks
        for task_cfg in self.config.tasks:
            task_cls = Registry.get_task(task_cfg.task_name)
            task_inst: BaseTask = task_cls(name=task_cfg.task_name, seed=task_cfg.seed, config=task_cfg.task_kwargs)
            print(f"\n[+] Running Task: {task_inst.name}", flush=True)

            for ctx_len in task_cfg.context_lengths:
                if (task_inst.name, ctx_len) in completed_groups:
                    print(f"  --> Context Length: {ctx_len:,} tokens -- already complete, skipping.",
                          flush=True)
                    continue
                print(f"  --> Context Length: {ctx_len:,} tokens (Samples: {task_cfg.num_samples})", flush=True)
                group_queries_start = len(query_eval_results)
                group_measurements_start = len(raw_measurements)
                
                # Generate samples
                # One budget for this task, used by the dense reference and every
                # method on the same query, so the pairing stays valid.
                task_budget = (task_cfg.max_new_tokens
                               if task_cfg.max_new_tokens is not None
                               else self.config.profiler.max_new_tokens)
                samples = task_inst.generate_samples(
                    context_length=ctx_len,
                    num_samples=task_cfg.num_samples,
                    tokenizer=self.tokenizer
                )

                # Establish the dense reference on this exact batch of queries.
                # This is the anchor the whole benchmark is defined against, so
                # it is measured at every context length -- including the long
                # ones -- rather than assumed. If it cannot run, the context
                # length is unusable and every method at it is skipped, because
                # a relative score with no reference is not a score.
                dense_adapter = DenseAdapter(name="dense_fp16_ref")
                dense_adapter.prepare_model(self.model, self.tokenizer)
                dense_kv_meta = dense_adapter.get_kv_metadata(ctx_len)
                observed_geometry: Dict[str, int] = {}

                dense_sample_map: Dict[str, SampleEvaluationResult] = {}
                dense_traces: List[GenerationTrace] = []
                try:
                    dense_task_score, dense_traces = self._evaluate_adapter_on_samples(
                        dense_adapter, task_inst, samples, ctx_len, task_budget
                    )
                    dense_ref_val = dense_task_score.mean_score
                    dense_scores_by_task_len[(task_inst.name, ctx_len)] = dense_ref_val
                    for s_res in dense_task_score.sample_results:
                        dense_sample_map[s_res.sample_id] = s_res
                    if dense_traces and dense_traces[0].cache_geometry:
                        observed_geometry = dense_traces[0].cache_geometry
                        dense_adapter.observed_geometry = observed_geometry
                        dense_kv_meta = dense_adapter.get_kv_metadata(ctx_len)
                    dense_lat = self._latency_from_traces(dense_traces)
                    peak_gib = max((t.peak_total_bytes for t in dense_traces), default=0) / 2 ** 30
                    print(f"      [Ref] Dense baseline raw score: {dense_ref_val:.1f}% | "
                          f"TTFT {dense_lat.ttft_ms / 1000.0:.1f}s | "
                          f"decode {dense_lat.decode_latency_ms_per_token:.0f} ms/tok | "
                          f"peak {peak_gib:.2f} GiB", flush=True)
                except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
                    self._release_memory()
                    raw_measurements.append({
                        "task_name": task_inst.name, "context_length": ctx_len,
                        "adapter_name": "dense_fp16_ref", "budget_spec": 16.0,
                        "status": "OOM",
                        "error_message": f"Dense reference OOM at {ctx_len} tokens: {e}",
                    })
                    print(f"      [Ref] Dense reference OOM at {ctx_len:,} tokens -- "
                          f"skipping this context length entirely.", flush=True)
                    continue
                except Exception as e:
                    self._release_memory()
                    raw_measurements.append({
                        "task_name": task_inst.name, "context_length": ctx_len,
                        "adapter_name": "dense_fp16_ref", "budget_spec": 16.0,
                        "status": "RUNTIME_ERROR", "error_message": str(e),
                    })
                    print(f"      [Ref] Dense reference failed at {ctx_len:,} tokens "
                          f"({type(e).__name__}: {e}) -- skipping this context length.", flush=True)
                    continue

                # Now evaluate each configured adapter under budget sweeps
                for ad_inst, ad_cfg in zip(adapters, self.config.adapters):
                    if self.model:
                        ad_inst.prepare_model(self.model, self.tokenizer)
                    if observed_geometry:
                        ad_inst.observed_geometry = observed_geometry

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
                            self._record_failed_queries(
                                query_eval_results, samples, task_inst, ctx_len,
                                ad_inst, b_val, dense_sample_map, "UNSUPPORTED", support_reason)
                            continue

                        # Evaluate Task Quality & Profile Latency
                        try:
                            if ad_inst.method_type == "dense":
                                # The dense reference above already ran exactly
                                # this configuration on exactly these samples;
                                # re-running it would double the most expensive
                                # pass in the sweep and, worse, let the baseline
                                # disagree with itself through run-to-run noise.
                                task_res, traces = dense_task_score, dense_traces
                            else:
                                task_res, traces = self._evaluate_adapter_on_samples(
                                    ad_inst, task_inst, samples, ctx_len, task_budget
                                )
                            lat_res = self._latency_from_traces(traces)

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

                            # Atomic Query Evaluation Results for each query.
                            # Dense and method runtime figures come from the two
                            # runs of *this* query, so the Part 2 comparison is a
                            # genuine pairing rather than a method's own numbers
                            # entered on both sides.
                            for s_idx, sample_obj in enumerate(samples):
                                d_sample_res = dense_sample_map.get(sample_obj.sample_id)
                                d_raw = float(d_sample_res.score) if d_sample_res else (dense_ref / 100.0)
                                m_sample_res = task_res.sample_results[s_idx] if s_idx < len(task_res.sample_results) else None
                                m_raw = float(m_sample_res.score) if m_sample_res else 0.0

                                d_tr = dense_traces[s_idx] if s_idx < len(dense_traces) else None
                                m_tr = traces[s_idx] if s_idx < len(traces) else None

                                q_norm = self.normalizer.normalize(m_raw, d_raw, task_floor=task_inst.floor_score)
                                r_eff = compute_query_resource_efficiency(dense_bytes, method_bytes, 16.0, mem_cost)
                                p1_s = compute_utility(q_norm, r_eff, alpha=self.config.scoring.utility_alpha, formula=self.config.scoring.utility_formula)

                                q_meta: Dict[str, Any] = {
                                    "max_new_tokens": task_budget,
                                "kv_state_metadata": kv_meta.custom_metrics,
                                    "algorithmic_bytes": float(kv_meta.algorithmic_bytes),
                                    "metadata_overhead_bytes": float(kv_meta.metadata_overhead_bytes),
                                }
                                if m_tr is not None:
                                    q_meta.update({
                                        "method_prefill_seconds": m_tr.prefill_seconds,
                                        "method_compression_seconds": m_tr.compression_seconds,
                                        "method_decode_ms_per_token": m_tr.decode_seconds_per_token * 1000.0,
                                        "method_latency_jitter_ms": m_tr.latency_jitter_ms,
                                        "method_peak_prefill_bytes": m_tr.peak_prefill_bytes,
                                        "method_resident_kv_bytes": m_tr.kv_bytes_after_transform,
                                        "method_kv_tokens_retained": m_tr.kv_tokens_after_transform,
                                        "method_prompt_tokens": m_tr.prompt_tokens,
                                        "transform": m_tr.method_metadata,
                                    })
                                if d_tr is not None:
                                    q_meta.update({
                                        "dense_prefill_seconds": d_tr.prefill_seconds,
                                        "dense_resident_kv_bytes": d_tr.kv_bytes_after_transform,
                                    })

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
                                    dense_ttft_ms=float(d_tr.ttft_seconds * 1000.0) if d_tr else None,
                                    method_ttft_ms=float(m_tr.ttft_seconds * 1000.0) if m_tr else None,
                                    dense_decode_throughput=float(d_tr.decode_throughput_tok_per_sec) if d_tr else None,
                                    method_decode_throughput=float(m_tr.decode_throughput_tok_per_sec) if m_tr else None,
                                    dense_peak_vram_mb=(d_tr.peak_total_bytes - d_tr.weight_baseline_bytes) / 2**20 if d_tr else None,
                                    method_peak_vram_mb=(m_tr.peak_total_bytes - m_tr.weight_baseline_bytes) / 2**20 if m_tr else None,
                                    dense_prediction=d_sample_res.prediction if d_sample_res else "",
                                    method_prediction=m_sample_res.prediction if m_sample_res else "",
                                    ground_truths=sample_obj.ground_truths,
                                    formula_name=self.config.scoring.utility_formula,
                                    alpha=self.config.scoring.utility_alpha,
                                    metadata=q_meta,
                                    dense_success=dense_anchor_usable(d_raw, task_inst.floor_score),
                                    method_success=True,
                                    paired_success=dense_anchor_usable(d_raw, task_inst.floor_score),
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

                            peak_kv_mb = (
                                sum(t.peak_total_bytes - t.weight_baseline_bytes for t in traces)
                                / max(1, len(traces)) / 2 ** 20
                            )
                            resident_kv_mb = (
                                sum(t.kv_bytes_after_transform for t in traces)
                                / max(1, len(traces)) / 2 ** 20
                            )
                            peak_vram_by_method_ctx[(ad_inst.name, ctx_len)] = peak_kv_mb

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
                                "latency_jitter_ms": float(lat_res.latency_jitter_ms),
                                # Measured on the device, not inferred from bits/token.
                                "peak_vram_above_weights_mb": float(peak_kv_mb),
                                "resident_kv_bytes_measured": float(resident_kv_mb * 2 ** 20),
                                "kv_tokens_retained": int(
                                    sum(t.kv_tokens_after_transform for t in traces) / max(1, len(traces))
                                ),
                                "predictions": [s.prediction for s in task_res.sample_results],
                            })

                            print(f"      [{ad_inst.name} | Budget={b_val}] Raw: {task_res.mean_score:.1f}% -> "
                                  f"Norm: {norm_q:.1f}% | {mem_cost:.2f} bpt | "
                                  f"TTFT {lat_res.ttft_ms / 1000.0:.1f}s | "
                                  f"{lat_res.decode_latency_ms_per_token:.0f} ms/tok | "
                                  f"KV {resident_kv_mb / 1024.0:.2f} GiB | "
                                  f"peak+ {peak_kv_mb / 1024.0:.2f} GiB", flush=True)

                        except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as e:
                            raw_measurements.append({
                                "task_name": task_inst.name,
                                "context_length": ctx_len,
                                "adapter_name": ad_inst.name,
                                "budget_spec": b_val,
                                "status": "OOM",
                                "error_message": str(e)
                            })
                            print(f"      [{ad_inst.name} | Budget={b_val}] FAILED: Out of Memory (OOM)", flush=True)
                            self._release_memory()
                            self._record_failed_queries(
                                query_eval_results, samples, task_inst, ctx_len,
                                ad_inst, b_val, dense_sample_map, "OOM", str(e))
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

                # This (task, context length) group is complete, dense anchor and
                self._record_failed_queries(
                    query_eval_results, samples, task_inst, ctx_len,
                    ad_inst, b_val, dense_sample_map, "RUNTIME_ERROR", str(e))
                # all methods together. Flush it to disk before starting the next
                # one so an interruption costs this group and nothing earlier.
                self._append_checkpoint(
                    task_inst.name,
                    ctx_len,
                    query_eval_results[group_queries_start:],
                    raw_measurements[group_measurements_start:],
                )

        # Replayed groups contribute to the scores exactly as freshly measured
        # ones do; their operating points are rebuilt from the saved manifest.
        if replay_measurements:
            self._replay_operating_points(
                replay_measurements, operating_points,
                runtime_metrics_by_method, peak_vram_by_method_ctx)

        # Dataset Aggregations
        dataset_aggregates = []
        if query_eval_results:
            method_groups: Dict[str, List[QueryEvaluationResult]] = {}
            for q in query_eval_results:
                method_groups.setdefault(q.method_name, []).append(q)
            for m_name, q_list in method_groups.items():
                agg = QueryAggregationEngine.aggregate(q_list, dataset_name=self.config.benchmark_name)
                dataset_aggregates.append(agg.to_dict() if hasattr(agg, "to_dict") else asdict(agg))

        # ---- Evaluation coverage (C), reported separately from Q and R_mem ----
        # N_total comes from the task config, i.e. what the grid assigned, so a
        # method that produced no rows for a cell is still charged for them.
        expected_per_cell: Dict[Tuple[str, int], int] = {}
        for t_cfg in self.config.tasks:
            for c_len in t_cfg.context_lengths:
                expected_per_cell[(t_cfg.task_name, int(c_len))] = int(t_cfg.num_samples)

        cell_coverage = coverage_by_cell(query_eval_results, expected_per_cell)
        coverage_manifest = {
            "definition": ("C = paired_success / total_queries, where paired_success "
                           "requires a usable dense anchor AND a valid method result. "
                           "Reported separately from Q, R_mem and S_res; none of those "
                           "are affected by it."),
            "by_task_method_context": [r.to_dict() for r in cell_coverage],
            "by_method_context": [r.to_dict() for r in roll_up(
                cell_coverage, by=("method_name", "budget_spec", "context_length"))],
            "by_method": [r.to_dict() for r in roll_up(
                cell_coverage, by=("method_name", "budget_spec"))],
        }

        # Save Raw Measurements JSON Schema v2.0.0 (Atomic Query Level)
        import platform
        from datetime import datetime
        raw_manifest = {
            "schema_version": "2.0.0",
            "benchmark_name": self.config.benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "environment": {
                "python_version": platform.python_version(),
                "pytorch_version": torch.__version__,
                "transformers_version": __import__("transformers").__version__,
                "device": str(self.device),
                "device_name": (torch.cuda.get_device_name(self.device)
                                if torch.cuda.is_available() and self.device.type == "cuda" else None),
                "device_total_memory_bytes": (
                    torch.cuda.get_device_properties(self.device).total_memory
                    if torch.cuda.is_available() and self.device.type == "cuda" else None),
                "os": platform.platform()
            },
            "model_name": self.config.model.model_name_or_path,
            "model_config": {
                "dtype": self.config.model.dtype,
                "load_in_4bit": self.config.model.load_in_4bit,
                "load_in_8bit": self.config.model.load_in_8bit,
                "bnb_4bit_quant_type": self.config.model.bnb_4bit_quant_type,
                "bnb_4bit_use_double_quant": self.config.model.bnb_4bit_use_double_quant,
                "attn_implementation": self.config.model.attn_implementation,
                "max_model_len": self.config.model.max_model_len,
                # Recorded because a scaled RoPE changes what a long-context
                # score means; a reader must be able to tell the two regimes apart.
                "rope_scaling": self._effective_rope or self.config.model.rope_scaling,
                "chat_template_kwargs": self.config.model.chat_template_kwargs,
                "weight_bytes_resident": self._weight_bytes,
            },
            "execution_config": {
                "prefill_chunk_size": self.config.profiler.prefill_chunk_size,
                "max_new_tokens": self.config.profiler.max_new_tokens,
                "decoding": "greedy",
                "prefill": "chunked, shared preallocated cache",
            },
            # Where each method's implementation came from, straight from the
            # adapters, so a reader can separate an upstream reference from a
            # CRBench-internal baseline without reading the source.
            "method_provenance": {ad.name: ad.provenance() for ad in adapters},
            "scoring_config": {
                "utility_formula": self.config.scoring.utility_formula,
                "utility_alpha": self.config.scoring.utility_alpha,
                "resource_normalization_max": self.config.scoring.resource_normalization_max,
                "enable_part2": self.config.scoring.enable_part2,
            },
            "query_results": [q.to_dict() for q in query_eval_results],
            "dataset_aggregates": dataset_aggregates,
            "coverage": coverage_manifest,
            "raw_measurements": raw_measurements
        }
        raw_json_path = out_dir / "raw_results_v1.json"
        with open(raw_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_manifest, f, indent=2)
        print(f"[OK] Versioned query-level measurements saved: {raw_json_path}", flush=True)

        # 3. Part 1 Scoring: CRBench Resource Scores
        print("\n" + "=" * 70, flush=True)
        print("Computing Part 1 -- CRBench Resource Scores (S_res)...", flush=True)
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
        print("Computing Part 2 -- CRBench System Scores (S_sys)...", flush=True)
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

            # Measured peak device allocation above the weight baseline, averaged
            # over the context lengths this method completed.  Previously this
            # was 1024 MB scaled by bits/token, which is not a measurement of
            # anything -- it restated the Part 1 memory axis as though it were a
            # hardware observation, so Part 2 could never disagree with Part 1.
            method_peaks = [v for (m, _), v in peak_vram_by_method_ctx.items() if m == ad_name]
            vram_mb = float(sum(method_peaks) / len(method_peaks)) if method_peaks else 0.0

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
            print(f"[OK] Publication figures saved in {fig_dir}")

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
        print(f"[OK] Final Benchmark Report generated: {report_file}")

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


    # ------------------------------------------------------------------ #
    # Checkpointing                                                       #
    # ------------------------------------------------------------------ #

    def _checkpoint_path(self) -> Path:
        return Path(self.config.output_dir) / "progress.jsonl"

    def _load_checkpoint(self) -> Tuple[set, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Replay completed (task, context length) groups from a previous attempt.

        A full sweep is many GPU-hours, and anything that interrupts it -- a
        reclaimed GPU, a reboot -- should cost one context length, not the run.
        The unit is the (task, context length) group because that is the unit the
        dense anchor is established over: resuming mid-group would leave the
        methods in it normalised against a reference from a different process.
        """
        path = self._checkpoint_path()
        done: set = set()
        queries: List[Dict[str, Any]] = []
        measurements: List[Dict[str, Any]] = []
        if not path.is_file():
            return done, queries, measurements

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # A run killed mid-write leaves a partial final line; the
                    # group it belonged to simply gets redone.
                    print("[*] Ignoring a truncated checkpoint line "
                          "(the run it came from was interrupted mid-write).", flush=True)
                    continue
                done.add((rec["task_name"], int(rec["context_length"])))
                queries.extend(rec.get("query_results", []))
                measurements.extend(rec.get("raw_measurements", []))

        if done:
            print(f"[*] Resuming: {len(done)} (task, context length) groups already "
                  f"complete, {len(queries)} query results replayed.", flush=True)
        return done, queries, measurements

    def _append_checkpoint(
        self,
        task_name: str,
        ctx_len: int,
        queries: List[QueryEvaluationResult],
        measurements: List[Dict[str, Any]],
    ) -> None:
        path = self._checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "task_name": task_name,
            "context_length": ctx_len,
            "query_results": [q.to_dict() for q in queries],
            "raw_measurements": measurements,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())

    @staticmethod
    def _replay_operating_points(
        measurements: List[Dict[str, Any]],
        operating_points: Dict[str, Dict[int, List[OperatingPoint]]],
        runtime_metrics: Dict[str, List[LatencyProfileResult]],
        peak_vram: Dict[Tuple[str, int], float],
    ) -> None:
        """Rebuild the scoring inputs from replayed measurements."""
        for m in measurements:
            if m.get("status") != "SUCCESS":
                continue
            name, ctx = m["adapter_name"], int(m["context_length"])
            operating_points.setdefault(name, {}).setdefault(ctx, []).append(
                OperatingPoint(
                    method_name=name,
                    context_length=ctx,
                    budget_value=m["effective_bpt"],
                    quality_score=m["normalized_score"],
                    memory_cost=m["effective_bpt"],
                    latency_ms=m.get("decode_latency_ms", 0.0),
                    metadata={"raw_score": m.get("raw_score", 0.0),
                              "budget_spec": str(m.get("budget_spec"))},
                )
            )
            runtime_metrics.setdefault(name, []).append(LatencyProfileResult(
                ttft_ms=m.get("ttft_ms", 0.0),
                prefill_throughput_tok_per_sec=0.0,
                decode_latency_ms_per_token=m.get("decode_latency_ms", 0.0),
                decode_throughput_tok_per_sec=m.get("decode_throughput_tok_sec", 0.0),
                total_time_seconds=0.0,
                prompt_tokens=ctx,
                generated_tokens=0,
                inter_token_latencies_ms=[],
            ))
            if m.get("peak_vram_above_weights_mb") is not None:
                peak_vram[(name, ctx)] = float(m["peak_vram_above_weights_mb"])

    def _record_failed_queries(
        self,
        sink: List[QueryEvaluationResult],
        samples: List[EvaluationSample],
        task: BaseTask,
        ctx_len: int,
        adapter: BaseContextAdapter,
        budget_spec: Any,
        dense_sample_map: Dict[str, Any],
        status: str,
        error_message: str,
    ) -> None:
        """Emit a query row per sample when a method produced no result.

        Without this a failed method contributes no rows at all, and coverage --
        paired successes over the queries the cell was assigned -- would read
        100% for a method that crashed on every one of them. These rows carry
        method_success=False and normalized_quality=0.0 but are never eligible
        for Q, which continues to be computed only over paired successes, so
        quality scoring is unaffected.
        """
        for sample in samples:
            d_res = dense_sample_map.get(sample.sample_id)
            d_raw = float(d_res.score) if d_res is not None else 0.0
            sink.append(QueryEvaluationResult(
                query_id=sample.sample_id,
                task_name=task.name,
                context_length=ctx_len,
                model_name=self.config.model.model_name_or_path,
                method_name=adapter.name,
                budget_spec=budget_spec,
                dense_raw_score=d_raw,
                method_raw_score=0.0,
                task_floor=task.floor_score,
                normalized_quality=0.0,
                quality_retained_pct=0.0,
                status=status,
                error_message=error_message,
                ground_truths=list(sample.ground_truths),
                formula_name=self.config.scoring.utility_formula,
                alpha=self.config.scoring.utility_alpha,
                provenance="method_failure",
                dense_success=dense_anchor_usable(d_raw, task.floor_score),
                method_success=False,
                paired_success=False,
            ))

    def _encode_sample(self, sample: EvaluationSample, context_length: int) -> torch.Tensor:
        """Tokenise one sample's prompt, applying the model's chat template."""
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be loaded for live evaluation.")

        if getattr(self.tokenizer, "chat_template", None):
            prompt_text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": sample.full_prompt}],
                tokenize=False,
                add_generation_prompt=True,
                **(self.config.model.chat_template_kwargs or {}),
            )
        else:
            prompt_text = sample.full_prompt

        max_ctx = self.config.model.max_model_len or max(context_length + 256, 4096)
        enc = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=max_ctx)
        target_dev = getattr(self.model, "device", self.device)
        return enc.input_ids.to(target_dev)

    def _release_memory(self) -> None:
        import gc
        gc.collect()
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def _run_query(
        self,
        adapter: BaseContextAdapter,
        sample: EvaluationSample,
        context_length: int,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, GenerationTrace]:
        """Run one query end to end under one method, and measure it.

        Prefill is chunked so peak activation memory is bounded by the chunk
        size rather than the context length.  An OOM recorded here therefore
        means the *KV representation* did not fit -- a genuine result -- rather
        than that the prompt was fed to the model too greedily, which is not.
        """
        input_ids = self._encode_sample(sample, context_length)
        prof = self.config.profiler

        adapter.begin_query(self.model, input_ids)
        try:
            trace = chunked_prefill_generate(
                self.model,
                input_ids,
                max_new_tokens=(max_new_tokens if max_new_tokens is not None
                                else prof.max_new_tokens),
                chunk_size=prof.prefill_chunk_size,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                on_chunk_end=adapter.on_chunk_stored if adapter.streaming_transform else None,
                on_token_end=adapter.on_token_stored if adapter.streaming_transform else None,
                transform_cache=adapter.transform_cache if adapter.oneshot_transform else None,
                weight_baseline_bytes=self._weight_bytes,
                device=self.device,
                empty_cache_between_chunks=prof.empty_cache_between_chunks,
            )
        finally:
            adapter.end_query()

        # Hand the adapter the geometry the cache actually had, so its memory
        # accounting is not built on a config field that does not describe this
        # architecture (depth recurrence, hybrid or heterogeneous attention).
        if trace.cache_geometry:
            adapter.observed_geometry = trace.cache_geometry

        prompt_len = int(input_ids.shape[-1])
        text = self.tokenizer.decode(
            trace.generated_ids[0][prompt_len:], skip_special_tokens=True
        )
        del input_ids
        self._release_memory()
        return text, trace

    def _evaluate_adapter_on_samples(
        self,
        adapter: BaseContextAdapter,
        task: BaseTask,
        samples: List[EvaluationSample],
        context_length: int,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[TaskResult, List[GenerationTrace]]:
        """Run every sample under one method; returns scores plus per-query traces.

        ``max_new_tokens`` is the task's budget when it declares one. The dense
        reference and every method on a given query always receive the same
        value, so the comparison stays paired.
        """
        predictions: List[str] = []
        traces: List[GenerationTrace] = []
        for sample in samples:
            text, trace = self._run_query(adapter, sample, context_length, max_new_tokens)
            predictions.append(text)
            traces.append(trace)
        return task.evaluate_batch(predictions, samples), traces

    @staticmethod
    def _latency_from_traces(traces: List[GenerationTrace]) -> LatencyProfileResult:
        """Aggregate per-query traces into the Part 2 latency record.

        Every field is a measurement: prefill and decode were timed as separate
        device-synchronised stages, and jitter is the spread of real inter-token
        intervals.  An earlier revision split one wall-clock total between the
        two stages by a fixed ratio and synthesised the intervals from it.
        """
        if not traces:
            return LatencyProfileResult(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, [])

        n = len(traces)
        return LatencyProfileResult(
            ttft_ms=sum(t.ttft_seconds for t in traces) / n * 1000.0,
            prefill_throughput_tok_per_sec=sum(t.prefill_throughput_tok_per_sec for t in traces) / n,
            decode_latency_ms_per_token=sum(t.decode_seconds_per_token for t in traces) / n * 1000.0,
            decode_throughput_tok_per_sec=sum(t.decode_throughput_tok_per_sec for t in traces) / n,
            total_time_seconds=sum(t.ttft_seconds + t.decode_seconds for t in traces),
            prompt_tokens=sum(t.prompt_tokens for t in traces),
            generated_tokens=sum(t.generated_tokens for t in traces),
            inter_token_latencies_ms=[x * 1000.0 for t in traces for x in t.inter_token_seconds],
        )


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
