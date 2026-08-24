"""
Tasks module for CRBench.
"""

from crbench.tasks.base import (
    BaseTask,
    EvaluationSample,
    SampleEvaluationResult,
    TaskResult,
    compute_exact_match,
    compute_token_f1,
)
from crbench.tasks.niah import SingleNeedleTask, MultiNeedleTask
from crbench.tasks.ruler import RulerKVTask, RulerVariableTrackingTask
from crbench.tasks.multihop import MultiHopQATask
from crbench.tasks.longbench import LongBenchQATask

__all__ = [
    "BaseTask",
    "EvaluationSample",
    "SampleEvaluationResult",
    "TaskResult",
    "compute_exact_match",
    "compute_token_f1",
    "SingleNeedleTask",
    "MultiNeedleTask",
    "RulerKVTask",
    "RulerVariableTrackingTask",
    "MultiHopQATask",
    "LongBenchQATask",
]
