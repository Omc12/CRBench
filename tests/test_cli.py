"""
Unit tests for CRBench CLI.
"""

from click.testing import CliRunner
from crbench.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "CRBench — Context Resource Benchmark" in result.output


def test_cli_evaluate_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate", "--help"])
    assert result.exit_code == 0
    assert "Atomic Query Evaluation" in result.output
    assert "--model" in result.output
    assert "--query" in result.output
    assert "--method" in result.output
    assert "--alpha" in result.output


def test_cli_evaluate_dataset_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["evaluate-dataset", "--help"])
    assert result.exit_code == 0
    assert "Dataset Evaluation" in result.output
    assert "--dataset" in result.output
    assert "--method" in result.output


def test_cli_recompute(tmp_path):
    import json
    manifest = {
        "schema_version": "2.0.0",
        "benchmark_name": "test_cli_recompute",
        "raw_measurements": [
            {
                "task_name": "single_niah",
                "context_length": 2048,
                "adapter_name": "dense_fp16",
                "budget_spec": 16.0,
                "status": "SUCCESS",
                "raw_score": 100.0,
                "dense_reference_score": 100.0,
                "normalized_score": 100.0,
                "effective_bpt": 16.0,
                "algorithmic_bytes": 16000.0,
                "metadata_bytes": 0.0,
                "ttft_ms": 100.0,
                "decode_throughput_tok_sec": 50.0,
                "decode_latency_ms": 20.0,
            }
        ]
    }
    raw_file = tmp_path / "raw.json"
    with open(raw_file, "w") as f:
        json.dump(manifest, f)

    runner = CliRunner()
    result = runner.invoke(cli, ["recompute", "--raw-file", str(raw_file), "--alpha", "0.70", "--formula", "linear"])
    assert result.exit_code == 0
    assert "Recomputed Part 1 Resource Scores" in result.output
    assert "dense_fp16" in result.output
