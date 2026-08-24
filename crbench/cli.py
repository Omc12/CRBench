"""
CRBench Command-Line Interface (CLI).
Provides commands for running benchmarks, evaluating adapters, computing scores, and generating reports.
"""

from __future__ import annotations
import sys
from pathlib import Path
import click
import yaml

from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig, ModelConfig
from crbench.core.runner import BenchmarkRunner
from crbench.scoring.resource_score import CRBenchResourceScorer
from crbench.scoring.system_score import CRBenchSystemScorer
from crbench.reporting.tables import format_resource_scores_table, format_system_scores_table
from crbench.reporting.report_generator import ReportGenerator
from crbench.statistics.hypothesis import HypothesisEngine
from crbench.statistics.bootstrap import BootstrapEngine


@click.group()
@click.version_option(version="0.1.0", prog_name="crbench")
def cli():
    """CRBench — Context Resource Benchmark for Long-Context LLMs."""
    pass


@cli.command("run")
@click.option("--config", "-c", type=click.Path(exists=True), help="Path to benchmark YAML configuration file.")
@click.option("--model", "-m", default="Qwen/Qwen2.5-0.5B-Instruct", help="Model name or path.")
@click.option("--tasks", "-t", multiple=True, default=["single_niah", "ruler_kv", "multihop_qa"], help="Tasks to evaluate.")
@click.option("--context-lengths", "-l", multiple=True, type=int, default=[8192, 16384, 32768], help="Context lengths (tokens).")
@click.option("--samples", "-s", default=10, type=int, help="Number of evaluation samples per context length.")
@click.option("--output-dir", "-o", default="results", help="Directory to save experimental results.")
def run_cmd(config, model, tasks, context_lengths, samples, output_dir):
    """Execute CRBench benchmark across models, tasks, and context lengths."""
    if config:
        cfg = BenchmarkConfig.from_yaml(config)
    else:
        # Construct default configuration
        task_cfgs = [
            TaskConfig(task_name=t, context_lengths=list(context_lengths), num_samples=samples)
            for t in tasks
        ]
        adapter_cfgs = [
            AdapterConfig(adapter_name="dense_fp16", adapter_type="dense", budgets=[16.0]),
            AdapterConfig(adapter_name="kv_quant_int8", adapter_type="quantized", budgets=[8.0]),
            AdapterConfig(adapter_name="kv_quant_int4", adapter_type="quantized", budgets=[4.0]),
            AdapterConfig(adapter_name="snapkv_0.25", adapter_type="eviction", budgets=[4.0], params={"strategy": "snapkv"}),
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
    click.echo(f"\n[✓] CRBench run finished successfully. Results saved in: {output_dir}")


@cli.command("report")
@click.option("--results-dir", "-d", default="results", type=click.Path(exists=True), help="Directory containing experiment results.")
def report_cmd(results_dir):
    """Generate Markdown and LaTeX tables and reports from benchmark results."""
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
    click.echo(f"Statistically Significant (alpha=0.05): {'YES [✓]' if res.is_statistically_significant else 'NO [✗]'}")


def main():
    cli()


if __name__ == "__main__":
    main()
