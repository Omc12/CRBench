"""
Script to execute complete Quickstart & Standard benchmarks and generate publication artifacts.
"""

from pathlib import Path
import json
import torch

from crbench.core.config import BenchmarkConfig
from crbench.core.runner import BenchmarkRunner


def run_full_benchmark_suite():
    print("=" * 80)
    print("CRBench: Running Comprehensive Quickstart & Standard Benchmark Evaluation")
    print("=" * 80)

    # 1. Run Quickstart Benchmark
    print("\n--- Executing Quickstart Benchmark ---")
    cfg_quick = BenchmarkConfig.from_yaml("configs/quickstart.yaml")
    runner1 = BenchmarkRunner(cfg_quick)
    runner1.load_model()
    results_quick = runner1.run()

    # 2. Run Standard Benchmark
    print("\n--- Executing Standard Comprehensive Benchmark ---")
    cfg_standard = BenchmarkConfig.from_yaml("configs/standard_benchmark.yaml")
    runner2 = BenchmarkRunner(cfg_standard)
    runner2.load_model()
    results_standard = runner2.run()

    print("\n" + "=" * 80)
    print("[✓] All Quickstart & Standard Benchmark Experiments and Reports Finished!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_benchmark_suite()
