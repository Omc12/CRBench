"""
CRBench Command-Line Interface (CLI).
Provides commands for running benchmarks, evaluating individual queries,
evaluating datasets, recomputing scores, and generating reports.
"""

from __future__ import annotations
import sys
import os
import json
from pathlib import Path
import click
import yaml
import torch

from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig, ModelConfig, ScoringConfig
from crbench.core.runner import BenchmarkRunner, recompute_scores_from_raw_file
from crbench.core.registry import Registry
from crbench.tasks.base import EvaluationSample, BaseTask
from crbench.core.query_result import QueryEvaluator, QueryAggregationEngine, QueryEvaluationResult
from crbench.adapters.dense import DenseAdapter
from crbench.scoring.utility import CRBENCH_ALPHA, CRBENCH_FORMULA_NAME, compute_utility
from crbench.statistics.hypothesis import HypothesisEngine
from crbench.statistics.bootstrap import BootstrapEngine


@click.group()
@click.version_option(version="0.2.0", prog_name="crbench")
def cli():
    """CRBench — Context Resource Benchmark for Long-Context LLMs."""
    pass


@cli.command("evaluate")
@click.option("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name or HuggingFace path.")
@click.option("--query", "-q", required=True, help="Query string or path to query text file.")
@click.option("--context", "-c", default="", help="Context string or path to context text file.")
@click.option("--ground-truth", "-g", multiple=True, help="Expected ground truth answer(s).")
@click.option("--method", "-a", required=True, help="Compression adapter type (e.g. kv_quant_int8, snapkv, custom_dkv, dense).")
@click.option("--budget", "-b", default=4.0, type=float, help="Resource budget value (e.g. bits/token or ratio).")
@click.option("--dense/--no-dense", default=True, help="Run uncompressed Dense FP16 reference on the identical query.")
@click.option("--alpha", default=CRBENCH_ALPHA, type=float, help="Quality weight alpha in utility formulation.")
@click.option("--formula", default="linear", help="Scoring utility formula (linear, cobb_douglas, harmonic, power_mean_2).")
@click.option("--part2/--part1", default=False, help="Enable Part 2 System runtime profiling (TTFT, throughput).")
def evaluate_cmd(model, query, context, ground_truth, method, budget, dense, alpha, formula, part2):
    """
    Atomic Query Evaluation:
    Evaluates ONE MODEL + ONE QUERY/CONTEXT + DENSE REFERENCE + USER METHOD.
    """
    click.echo("=" * 72)
    click.echo(f"CRBench Atomic Query Evaluation")
    click.echo("=" * 72)

    # 1. Resolve context and query
    if os.path.exists(query):
        query_text = Path(query).read_text(encoding="utf-8")
    else:
        query_text = query

    if context and os.path.exists(context):
        context_text = Path(context).read_text(encoding="utf-8")
    else:
        context_text = context or ""

    gt_list = list(ground_truth) if ground_truth else [""]

    # 2. Build benchmark configuration for this single query
    sample = EvaluationSample(
        sample_id="cli_query_001",
        context=context_text,
        query=query_text,
        ground_truths=gt_list,
        context_length=max(512, len((context_text + " " + query_text).split())),
    )

    # Custom single query task wrapper
    task_cls = Registry.get_task("single_niah")
    task_inst = task_cls(name="cli_evaluation_task")

    # Load Adapter
    ad_cls = Registry.get_adapter(method if method in Registry.list_adapters() else "custom")
    ad_inst = ad_cls(name=method, config={"budget": budget})

    cfg = BenchmarkConfig(
        benchmark_name="cli_single_query_eval",
        model=ModelConfig(model_name_or_path=model),
        scoring=ScoringConfig(utility_alpha=alpha, utility_formula=formula, enable_part2=part2),
    )
    runner = BenchmarkRunner(cfg)
    runner.load_model()

    dense_ad = DenseAdapter(name="dense_fp16") if dense else None

    click.echo(f"[*] Evaluating query on model: {model} ...")
    query_evaluator = QueryEvaluator(
        normalizer=runner.normalizer,
        alpha=alpha,
        formula=formula,
        enable_part2=part2
    )

    res = query_evaluator.evaluate_query(
        model=runner.model,
        tokenizer=runner.tokenizer,
        sample=sample,
        method_adapter=ad_inst,
        task=task_inst,
        dense_adapter=dense_ad,
        device=runner.device
    )

    click.echo("\n" + "=" * 72)
    click.echo(res.format_summary())
    click.echo("=" * 72)


@cli.command("evaluate-dataset")
@click.option("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name or HuggingFace path.")
@click.option("--dataset", "-d", default="single_niah", help="Dataset / Task name (e.g. single_niah, multi_niah, ruler_kv).")
@click.option("--method", "-a", required=True, help="Compression adapter type (e.g. kv_quant_int8, snapkv, custom_dkv).")
@click.option("--budget", "-b", default=4.0, type=float, help="Resource budget value.")
@click.option("--context-lengths", "-l", multiple=True, type=int, default=[2048, 4096], help="Context lengths (tokens).")
@click.option("--samples", "-s", default=3, type=int, help="Number of query samples per context length.")
@click.option("--alpha", default=CRBENCH_ALPHA, type=float, help="Quality weight alpha in utility formulation.")
@click.option("--formula", default="linear", help="Scoring utility formula.")
@click.option("--output-dir", "-o", default="results/cli_dataset_eval", help="Directory to save experimental results.")
def evaluate_dataset_cmd(model, dataset, method, budget, context_lengths, samples, alpha, formula, output_dir):
    """
    Dataset Evaluation:
    Repeatedly evaluates the atomic query-level primitive across a dataset and aggregates results.
    """
    click.echo("=" * 72)
    click.echo(f"CRBench Dataset Evaluation: {dataset} ({method})")
    click.echo("=" * 72)

    task_cfgs = [
        TaskConfig(task_name=dataset, context_lengths=list(context_lengths), num_samples=samples)
    ]
    adapter_cfgs = [
        AdapterConfig(adapter_name="dense_fp16", adapter_type="dense", budgets=[16.0]),
        AdapterConfig(adapter_name=method, adapter_type=method if method in Registry.list_adapters() else "custom", budgets=[budget]),
    ]
    cfg = BenchmarkConfig(
        benchmark_name=f"dataset_eval_{dataset}",
        model=ModelConfig(model_name_or_path=model),
        tasks=task_cfgs,
        adapters=adapter_cfgs,
        scoring=ScoringConfig(utility_alpha=alpha, utility_formula=formula),
        output_dir=output_dir,
        save_plots=True,
    )

    runner = BenchmarkRunner(cfg)
    runner.load_model()
    results = runner.run()

    # Print summary table
    raw_json_path = Path(output_dir) / "raw_results_v1.json"
    if raw_json_path.exists():
        with open(raw_json_path) as f:
            data = json.load(f)
        aggs = data.get("dataset_aggregates", [])
        for agg_dict in aggs:
            if agg_dict.get("method_name") == method:
                agg_obj = QueryAggregationEngine.aggregate(
                    [QueryEvaluationResult.from_dict(q) for q in data.get("query_results", []) if q.get("method_name") == method],
                    dataset_name=dataset
                )
                click.echo("\n" + agg_obj.format_table())

    click.echo(f"\n[OK] Results saved in: {output_dir}")


@cli.command("recompute")
@click.option("--raw-file", "-f", required=True, type=click.Path(exists=True), help="Path to raw_results_v1.json file.")
@click.option("--alpha", default=CRBENCH_ALPHA, type=float, help="Quality weight alpha.")
@click.option("--formula", default="linear", help="Utility formula.")
@click.option("--weighting", default="logarithmic", help="Context weighting scheme.")
def recompute_cmd(raw_file, alpha, formula, weighting):
    """Recompute all CRBench scores from an existing raw_results_v1.json without re-running models."""
    click.echo(f"[*] Recomputing scores from: {raw_file} (Formula: {formula}, α = {alpha:.2f})")
    recomputed = recompute_scores_from_raw_file(
        raw_results_path=raw_file,
        weighting_scheme=weighting,
        utility_formula=formula,
        utility_alpha=alpha,
    )
    click.echo("\n=== Recomputed Part 1 Resource Scores ===")
    for r in recomputed["resource_results"]:
        click.echo(f"  * {r.method_name:<20}: S_res = {r.resource_score:6.2f} (Mean AUQC: {r.mean_auqc:6.2f})")

    if recomputed.get("dataset_aggregates"):
        click.echo("\n=== Recomputed Dataset Aggregates ===")
        for m_name, agg in recomputed["dataset_aggregates"].items():
            click.echo(f"  * {m_name:<20}: Mean S = {agg.mean_part1_score:6.2f} | 95% CI: [{agg.ci95_part1_score[0]:.2f}, {agg.ci95_part1_score[1]:.2f}]")


@cli.command("run")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to benchmark YAML configuration file.")
@click.option("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name or path.")
@click.option("--tasks", "-t", multiple=True, default=["single_niah", "ruler_kv", "multihop_qa"], help="Tasks to evaluate.")
@click.option("--context-lengths", "-l", multiple=True, type=int, default=[2048, 4096], help="Context lengths (tokens).")
@click.option("--samples", "-s", default=3, type=int, help="Number of evaluation samples per context length.")
@click.option("--output-dir", "-o", default="results", help="Directory to save experimental results.")
def run_cmd(config, model, tasks, context_lengths, samples, output_dir):
    """Execute CRBench benchmark across models, tasks, and context lengths."""
    if config:
        cfg = BenchmarkConfig.from_yaml(config)
    else:
        task_cfgs = [
            TaskConfig(task_name=t, context_lengths=list(context_lengths), num_samples=samples)
            for t in tasks
        ]
        adapter_cfgs = [
            AdapterConfig(adapter_name="dense_fp16", adapter_type="dense", budgets=[16.0]),
            AdapterConfig(adapter_name="kv_quant_int8", adapter_type="quantized", budgets=[8.0]),
            AdapterConfig(adapter_name="kv_quant_int4", adapter_type="quantized", budgets=[4.0]),
            AdapterConfig(adapter_name="snapkv", adapter_type="eviction", budgets=[4.0], params={"strategy": "snapkv"}),
            AdapterConfig(adapter_name="custom_dkv", adapter_type="custom", budgets=[4.0], params={"subspace_dim_ratio": 0.5}),
        ]
        cfg = BenchmarkConfig(
            benchmark_name="crbench_experiment",
            model=ModelConfig(model_name_or_path=model),
            tasks=task_cfgs,
            adapters=adapter_cfgs,
            output_dir=output_dir,
            save_plots=True
        )

    runner = BenchmarkRunner(cfg)
    runner.load_model()
    results = runner.run()
    click.echo(f"\n[OK] CRBench run finished successfully. Results saved in: {output_dir}")


@cli.command("report")
@click.option("--results-dir", "-d", default="results", type=click.Path(exists=True), help="Directory containing experiment results.")
def report_cmd(results_dir):
    """Generate Markdown reports from benchmark results."""
    click.echo(f"[*] Reading results from {results_dir}...")
    report_path = Path(results_dir) / "CRBENCH_REPORT.md"
    if report_path.exists():
        click.echo(report_path.read_text(encoding="utf-8"))
    else:
        click.echo(f"[!] No CRBENCH_REPORT.md found in {results_dir}. Run `crbench run` first.")


@cli.command("compare")
@click.argument("method_a")
@click.argument("method_b")
@click.option("--scores-a", "-a", multiple=True, type=float, required=True, help="Quality scores for Method A.")
@click.option("--scores-b", "-b", multiple=True, type=float, required=True, help="Quality scores for Method B.")
def compare_cmd(method_a, method_b, scores_a, scores_b):
    """Statistically compare two methods using paired permutation tests."""
    engine = HypothesisEngine()
    res = engine.paired_permutation_test(list(scores_a), list(scores_b), method_a_name=method_a, method_b_name=method_b)
    
    click.echo(f"=== Statistical Comparison: {method_a} vs. {method_b} ===")
    click.echo(f"Mean Difference ({method_a} - {method_b}): {res.mean_diff:+.3f}")
    click.echo(f"Cohen's d Effect Size: {res.cohens_d:.3f}")
    click.echo(f"Permutation p-value: {res.p_value:.4f}")
    click.echo(f"Statistically Significant (alpha=0.05): {'YES [OK]' if res.is_statistically_significant else 'NO'}")


def main():
    cli()


if __name__ == "__main__":
    main()
