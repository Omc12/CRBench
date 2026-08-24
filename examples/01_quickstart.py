"""
Example 01: Running CRBench Quickstart.
"""

from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig
from crbench.core.runner import BenchmarkRunner


def main():
    config = BenchmarkConfig(
        benchmark_name="quickstart_demo",
        tasks=[
            TaskConfig(task_name="single_niah", context_lengths=[8192, 16384], num_samples=5),
            TaskConfig(task_name="ruler_kv", context_lengths=[8192, 16384], num_samples=5),
        ],
        adapters=[
            AdapterConfig(adapter_name="dense_fp16", adapter_type="dense", budgets=[16.0]),
            AdapterConfig(adapter_name="kv_quant_int8", adapter_type="quantized", budgets=[8.0]),
            AdapterConfig(adapter_name="kv_quant_int4", adapter_type="quantized", budgets=[4.0]),
            AdapterConfig(adapter_name="snapkv_0.25", adapter_type="eviction", budgets=[4.0]),
        ],
        output_dir="results/quickstart_demo",
        save_plots=True
    )

    runner = BenchmarkRunner(config)
    runner.load_model()
    results = runner.run()

    print("\n[✓] Quickstart completed! Results available in results/quickstart_demo/")


if __name__ == "__main__":
    main()
