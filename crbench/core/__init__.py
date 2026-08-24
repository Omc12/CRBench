"""
Core primitives and base classes for CRBench.
"""

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig, ModelConfig
from crbench.core.registry import Registry
from crbench.core.runner import BenchmarkRunner

__all__ = [
    "BaseContextAdapter",
    "KVStateMetadata",
    "ContextBudget",
    "BudgetType",
    "BenchmarkConfig",
    "TaskConfig",
    "AdapterConfig",
    "ModelConfig",
    "Registry",
    "BenchmarkRunner",
]
