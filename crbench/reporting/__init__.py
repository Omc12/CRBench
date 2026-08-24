"""
Reporting and visualization module for CRBench.
"""

from crbench.reporting.plots import (
    plot_quality_vs_memory_pareto,
    plot_auqc_vs_context_length,
    plot_isobudget_comparison,
    plot_resource_vs_system_score,
)
from crbench.reporting.tables import (
    format_resource_scores_table,
    format_system_scores_table,
)
from crbench.reporting.report_generator import ReportGenerator

__all__ = [
    "plot_quality_vs_memory_pareto",
    "plot_auqc_vs_context_length",
    "plot_isobudget_comparison",
    "plot_resource_vs_system_score",
    "format_resource_scores_table",
    "format_system_scores_table",
    "ReportGenerator",
]
