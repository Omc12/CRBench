"""
CRBench — Context Resource Benchmark
A method-agnostic benchmark characterizing the quality-resource tradeoff of long-context LLMs.
"""

__version__ = "0.1.0"
__author__ = "CRBench Authors"

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget
from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig
from crbench.core.runner import BenchmarkRunner
from crbench.core.registry import Registry

# Import tasks and adapters to register all components in Registry
import crbench.tasks
import crbench.adapters

__all__ = [
    "BaseContextAdapter",
    "KVStateMetadata",
    "ContextBudget",
    "BenchmarkConfig",
    "TaskConfig",
    "AdapterConfig",
    "BenchmarkRunner",
    "Registry",
]
