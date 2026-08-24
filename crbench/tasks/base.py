"""
BaseTask, EvaluationSample, and metric computation utilities for CRBench.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import re
import string


@dataclass
class EvaluationSample:
    """A single evaluation sample in a context task."""
    sample_id: str
    context: str
    query: str
    ground_truths: List[str]
    context_length: int  # Target or approximate token length
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def full_prompt(self) -> str:
        return f"{self.context}\n\n{self.query}"


@dataclass
class SampleEvaluationResult:
    """Result of evaluating a single sample."""
    sample_id: str
    context_length: int
    prediction: str
    ground_truths: List[str]
    score: float  # Raw task score [0.0, 1.0]
    metrics: Dict[str, float] = field(default_factory=dict)
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Aggregate result for a task at a specific context length."""
    task_name: str
    context_length: int
    num_samples: int
    mean_score: float  # [0.0, 100.0] scale
    std_score: float
    sample_results: List[SampleEvaluationResult] = field(default_factory=list)
    submetrics: Dict[str, float] = field(default_factory=dict)
    floor_score: float = 0.0


def normalize_text(s: str) -> str:
    """Standard lowercasing and punctuation removal for evaluation."""
    s = s.lower()
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    # Normalize whitespace
    s = " ".join(s.split())
    return s


def compute_exact_match(prediction: str, ground_truths: List[str]) -> float:
    """Computes exact match score after normalization."""
    norm_pred = normalize_text(prediction)
    for gt in ground_truths:
        norm_gt = normalize_text(gt)
        if norm_pred == norm_gt or norm_gt in norm_pred:
            return 1.0
    return 0.0


def compute_token_f1(prediction: str, ground_truths: List[str]) -> float:
    """Computes token-level F1 score against ground truth list."""
    norm_pred = normalize_text(prediction).split()
    if not norm_pred:
        return 0.0

    best_f1 = 0.0
    for gt in ground_truths:
        norm_gt = normalize_text(gt).split()
        if not norm_gt:
            continue
        common = set(norm_pred) & set(norm_gt)
        if not common:
            continue
        num_same = sum(min(norm_pred.count(w), norm_gt.count(w)) for w in common)
        if num_same == 0:
            continue
        precision = 1.0 * num_same / len(norm_pred)
        recall = 1.0 * num_same / len(norm_gt)
        f1 = (2 * precision * recall) / (precision + recall)
        if f1 > best_f1:
            best_f1 = f1
    return best_f1


class BaseTask(ABC):
    """
    Abstract base class for all long-context evaluation tasks in CRBench.
    """

    def __init__(self, name: str, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        self._name = name
        self.seed = seed
        self.config = config or {}

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def floor_score(self) -> float:
        """Expected random guessing or floor performance for this task (0.0 to 100.0)."""
        pass

    @abstractmethod
    def generate_samples(
        self,
        context_length: int,
        num_samples: int,
        tokenizer: Any,
        **kwargs: Any
    ) -> List[EvaluationSample]:
        """Generates or retrieves evaluation samples matching the target context length."""
        pass

    @abstractmethod
    def evaluate_prediction(
        self,
        prediction: str,
        sample: EvaluationSample
    ) -> SampleEvaluationResult:
        """Scores a model prediction against the evaluation sample."""
        pass

    def evaluate_batch(
        self,
        predictions: List[str],
        samples: List[EvaluationSample]
    ) -> TaskResult:
        """Evaluates a batch of predictions and aggregates task scores."""
        sample_results = []
        scores = []
        for pred, sample in zip(predictions, samples):
            res = self.evaluate_prediction(pred, sample)
            sample_results.append(res)
            scores.append(res.score * 100.0)  # Scale to 0-100

        if not scores:
            return TaskResult(
                task_name=self.name,
                context_length=0,
                num_samples=0,
                mean_score=0.0,
                std_score=0.0,
                sample_results=[],
                floor_score=self.floor_score
            )

        mean_s = float(sum(scores) / len(scores))
        variance = sum((s - mean_s) ** 2 for s in scores) / max(1, len(scores) - 1)
        std_s = float(variance ** 0.5)

        ctx_len = samples[0].context_length if samples else 0

        return TaskResult(
            task_name=self.name,
            context_length=ctx_len,
            num_samples=len(samples),
            mean_score=mean_s,
            std_score=std_s,
            sample_results=sample_results,
            floor_score=self.floor_score
        )
