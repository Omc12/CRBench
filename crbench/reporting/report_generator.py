"""
Automated markdown experiment report generator for CRBench.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
from crbench.scoring.resource_score import CRBenchResourceScoreResult
from crbench.scoring.system_score import CRBenchSystemScoreResult
from crbench.statistics.sensitivity import WeightingSensitivityResult
from crbench.reporting.tables import (
    format_resource_scores_table,
    format_system_scores_table,
    format_provenance_audit_table
)


class ReportGenerator:
    """
    Generates structured experimental reports containing results, tables, Pareto frontiers,
    and methodological notes ready for inclusion in research manuscripts.
    """

    @classmethod
    def generate_markdown_report(
        cls,
        benchmark_name: str,
        model_name: str,
        resource_results: List[CRBenchResourceScoreResult],
        system_results: Optional[List[CRBenchSystemScoreResult]] = None,
        sensitivity_result: Optional[WeightingSensitivityResult] = None,
        plot_paths: Optional[Dict[str, str]] = None,
        output_file: Optional[str] = None
    ) -> str:
        lines = [
            f"# CRBench Benchmark Evaluation Report: {benchmark_name}",
            "",
            f"**Model Evaluated:** `{model_name}`  ",
            f"**Evaluation Tracks:** Part 1 (Resource Score) & Part 2 (System Score)  ",
            "",
            "---",
            "",
            "## 1. Executive Summary",
            "",
            "CRBench evaluates long-context representation efficiency by characterizing the quality–resource tradeoff curve under strict resource constraints.",
            "",
            "### Part 1: CRBench Resource Scores (S_res)",
            "",
            format_resource_scores_table(resource_results, format_type="markdown"),
            "",
        ]

        if system_results:
            lines.extend([
                "### Part 2: CRBench System Scores (S_sys)",
                "",
                format_system_scores_table(system_results, format_type="markdown"),
                "",
            ])

        if sensitivity_result:
            lines.extend([
                "### Context-Length Weighting Sensitivity Analysis",
                "",
                sensitivity_result.summary_text,
                "",
            ])

        if plot_paths:
            lines.extend([
                "## 2. Benchmark Visualizations",
                "",
            ])
            for title, path in plot_paths.items():
                lines.append(f"### {title}")
                lines.append(f"![{title}]({path})")
                lines.append("")

        lines.extend([
            "## 3. Scientific Metric Provenance Audit",
            "",
            format_provenance_audit_table(format_type="markdown"),
            "",
            "## 4. Methodological Notes & Axioms",
            "",
            "1. **Model-Relative Normalization**: Quality measures the exact fraction of the base model's uncompressed Dense FP16 capability retained (q_rel = (Q_m - Q_floor) / max(0.05, Q_dense - Q_floor)).",
            "2. **Primary Resource Axis**: Effective Bits-Per-Token (b_eff in [2.0, 16.0] bpt), unifying quantization, eviction, merging, and low-rank state sizes.",
            "3. **AUQC Integration**: Integrated in the logarithmic domain ln(b_eff) giving equal weight to successive 2x compression factors.",
            "4. **Logarithmic Context Weighting**: Context lengths are weighted proportionally to 1 + log2(L / L_min) to reflect sequence scaling difficulty.",
            "5. **No Arbitrary Weights**: Part 1 evaluates pure representation fidelity without runtime metrics; Part 2 models system utility via a multiplicative constrained utility function.",
            "",
            "---",
            "*Report generated automatically by CRBench v0.1.0.*"
        ])

        report_str = "\n".join(lines)

        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(report_str)

        return report_str
