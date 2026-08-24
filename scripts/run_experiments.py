"""
Script to execute complete Stage 1 & Stage 2 benchmarks and generate publication artifacts.
"""

from pathlib import Path
import json
import torch

from crbench.core.config import BenchmarkConfig
from crbench.core.runner import BenchmarkRunner
from crbench.scoring.resource_score import CRBenchResourceScorer
from crbench.scoring.system_score import CRBenchSystemScorer
from crbench.scoring.pareto import OperatingPoint
from crbench.statistics.bootstrap import BootstrapEngine
from crbench.statistics.hypothesis import HypothesisEngine
from crbench.statistics.stability import StabilityEngine
from crbench.reporting.plots import (
    plot_quality_vs_memory_pareto,
    plot_auqc_vs_context_length,
    plot_isobudget_comparison,
    plot_resource_vs_system_score,
)
from crbench.reporting.report_generator import ReportGenerator


def run_full_benchmark_suite():
    print("=" * 80)
    print("CRBench: Running Comprehensive Stage 1 & Stage 2 Benchmark Evaluation")
    print("=" * 80)

    # 1. Run Stage 1 Quick Benchmark
    print("\n--- Executing Stage 1 Quick Benchmark ---")
    cfg_stage1 = BenchmarkConfig.from_yaml("configs/stage1_quick.yaml")
    runner1 = BenchmarkRunner(cfg_stage1)
    runner1.load_model()
    results_stage1 = runner1.run()

    # 2. Run Stage 2 Standard Benchmark
    print("\n--- Executing Stage 2 Standard Comprehensive Benchmark ---")
    cfg_stage2 = BenchmarkConfig.from_yaml("configs/stage2_standard.yaml")
    runner2 = BenchmarkRunner(cfg_stage2)
    runner2.load_model()
    results_stage2 = runner2.run()

    print("\n" + "=" * 80)
    print("[✓] All Stage 1 & Stage 2 Benchmark Experiments and Reports Finished!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_benchmark_suite()
