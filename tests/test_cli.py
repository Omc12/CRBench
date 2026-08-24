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


def test_cli_compare():
    runner = CliRunner()
    result = runner.invoke(cli, [
        "compare", "MethodA", "MethodB",
        "-a", "90.0", "-a", "92.0", "-a", "88.0",
        "-b", "70.0", "-b", "72.0", "-b", "68.0"
    ])
    assert result.exit_code == 0
    assert "Statistical Comparison: MethodA vs. MethodB" in result.output
    assert "Cohen's d Effect Size" in result.output
