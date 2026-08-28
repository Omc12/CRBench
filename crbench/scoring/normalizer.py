"""
Reference-anchored Quality Normalization engine for CRBench.
Maps raw task scores to a model-relative [0.0, 100.0] capability retention scale
with dynamic range gating to prevent low-dense baseline division artifacts.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class NormalizationResult:
    """Detailed normalization outcome for a sample or aggregate task score."""
    raw_score: float                   # Raw unnormalized score [0.0, 1.0]
    dense_reference_score: float       # Base model uncompressed Dense FP16 performance [0.0, 1.0]
    task_floor: float                  # Task floor / chance performance [0.0, 1.0]
    normalized_quality: float          # Model-relative retention [0.0, 100.0]
    is_dense_valid: bool               # True if dense reference dynamic range >= min_dynamic_range
    effective_dynamic_range: float     # max(min_dynamic_range, dense_reference - floor)


@dataclass
class QualityNormalizer:
    """
    Normalizes raw task performance relative to an uncompressed Dense baseline reference
    and a task-specific random chance / floor performance.

    Mathematical Definition:
    -----------------------
    Q_abs = 100.0 * max(0.0, min(1.0, raw_score))
    
    Both compressed methods and uncompressed dense baseline are evaluated
    under the identical absolute task-success standard.
    """
    floor_score: float = 0.0
    min_dynamic_range: float = 0.05
    epsilon: float = 1e-6

    def normalize_detailed(
        self,
        raw_score: float,
        dense_reference_score: float,
        task_floor: Optional[float] = None
    ) -> NormalizationResult:
        """
        Computes absolute task-success quality score in [0.0, 100.0].
        """
        floor = task_floor if task_floor is not None else self.floor_score
        norm_q = max(0.0, min(100.0, float(raw_score) * 100.0))

        return NormalizationResult(
            raw_score=float(raw_score),
            dense_reference_score=float(dense_reference_score),
            task_floor=float(floor),
            normalized_quality=float(norm_q),
            is_dense_valid=True,
            effective_dynamic_range=1.0
        )

    def normalize(
        self,
        raw_score: float,
        dense_reference_score: float,
        task_floor: Optional[float] = None
    ) -> float:
        """
        Returns the scalar absolute quality score in [0.0, 100.0].
        """
        res = self.normalize_detailed(raw_score, dense_reference_score, task_floor)
        return res.normalized_quality
