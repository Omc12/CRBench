"""
CRBench — Context Resource Benchmark
A method-agnostic benchmark characterizing the quality-resource tradeoff of long-context LLMs.
"""

__version__ = "0.2.0"
__author__ = "CRBench Authors"

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata, ExecutionStatus
from crbench.core.budget import ContextBudget
from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig, ModelConfig, ScoringConfig
from crbench.core.runner import BenchmarkRunner, recompute_scores_from_raw_file
from crbench.core.registry import Registry
from crbench.core.query_result import (
    QueryEvaluationResult,
    QueryEvaluator,
    QueryAggregationEngine,
    DatasetAggregateResult,
)
from crbench.scoring.utility import (
    CRBENCH_ALPHA,
    CRBENCH_FORMULA_NAME,
    CRBENCH_FORMULA_DESCRIPTION,
    compute_utility,
    compute_query_resource_efficiency,
    compute_query_system_efficiency,
)

# Import tasks and adapters to register all components in Registry
import crbench.tasks
import crbench.adapters

__all__ = [
    "BaseContextAdapter",
    "KVStateMetadata",
    "ExecutionStatus",
    "ContextBudget",
    "BenchmarkConfig",
    "TaskConfig",
    "AdapterConfig",
    "ModelConfig",
    "ScoringConfig",
    "BenchmarkRunner",
    "recompute_scores_from_raw_file",
    "Registry",
    "QueryEvaluationResult",
    "QueryEvaluator",
    "QueryAggregationEngine",
    "DatasetAggregateResult",
    "CRBENCH_ALPHA",
    "CRBENCH_FORMULA_NAME",
    "CRBENCH_FORMULA_DESCRIPTION",
    "compute_utility",
    "compute_query_resource_efficiency",
    "compute_query_system_efficiency",
]
