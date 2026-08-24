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
    q_relative = (Q_method - Q_floor) / max(Delta_min, Q_dense - Q_floor)

    If (Q_dense - Q_floor) < Delta_min:
        The task at this context length exceeds the base model's uncompressed capability envelope.
        Unless Q_method outperforms Q_dense + Delta_min, normalized retention is 0.0.
    """
    floor_score: float = 0.0
    min_dynamic_range: float = 0.05   # Delta_min = 5% minimum dynamic range threshold
    epsilon: float = 1e-6

    def normalize_detailed(
        self,
        raw_score: float,
        dense_reference_score: float,
        task_floor: Optional[float] = None
    ) -> NormalizationResult:
        """
        Computes detailed model-relative normalization with dynamic range gating.
        """
        floor = task_floor if task_floor is not None else self.floor_score
        
        dense_gain = dense_reference_score - floor
        method_gain = raw_score - floor

        is_dense_valid = dense_gain >= self.min_dynamic_range
        effective_dr = max(self.min_dynamic_range, dense_gain)

        if not is_dense_valid:
            # Dense baseline is at or near floor (model cannot do the task even with FP16 KV cache)
            if raw_score <= dense_reference_score + self.epsilon:
                norm_q = 0.0
            else:
                # Compression method somehow outperforms dense (e.g. slight denoising effect)
                norm_q = max(0.0, min(100.0, (method_gain / self.min_dynamic_range) * 100.0))
        else:
            # Standard model-relative capability retention
            fraction = method_gain / effective_dr
            norm_q = max(0.0, min(100.0, fraction * 100.0))

        return NormalizationResult(
            raw_score=float(raw_score),
            dense_reference_score=float(dense_reference_score),
            task_floor=float(floor),
            normalized_quality=float(norm_q),
            is_dense_valid=bool(is_dense_valid),
            effective_dynamic_range=float(effective_dr)
        )

    def normalize(
        self,
        raw_score: float,
        dense_reference_score: float,
        task_floor: Optional[float] = None
    ) -> float:
        """
        Returns the scalar normalized quality score in [0.0, 100.0].
        """
        res = self.normalize_detailed(raw_score, dense_reference_score, task_floor)
        return res.normalized_quality
