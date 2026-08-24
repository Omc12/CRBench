"""
Memory profiling engine for CRBench.
Distinguishes between Algorithmic Representation Cost and Physical Hardware Memory Footprint.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os
import psutil
import torch


@dataclass
class MemoryProfileResult:
    """Detailed memory measurement result."""
    algorithmic_bytes: float         # Analytical theoretical state size in bytes
    metadata_overhead_bytes: float   # Index/scale/cluster metadata overhead
    total_representation_bytes: float # algorithmic + metadata
    effective_bits_per_token: float  # Total representation bits / context_length
    compression_ratio: float         # total_representation_bytes / dense_fp16_bytes
    physical_peak_allocated_bytes: float  # Actual device peak allocated memory
    physical_rss_delta_bytes: float       # Host process RSS memory delta
    device: str
    breakdown: Dict[str, Any] = field(default_factory=dict)

    @property
    def algorithmic_mb(self) -> float:
        return self.total_representation_bytes / (1024.0 ** 2)

    @property
    def physical_peak_mb(self) -> float:
        return self.physical_peak_allocated_bytes / (1024.0 ** 2)


class MemoryProfiler:
    """
    Measures both theoretical algorithmic memory and real physical memory consumption.
    """

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
        self._process = psutil.Process(os.getpid())
        self._start_rss: int = 0

    def start_tracking(self) -> None:
        """Starts physical memory tracking window."""
        self._start_rss = self._process.memory_info().rss
        if torch.cuda.is_available() and self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def stop_tracking(
        self,
        algorithmic_bytes: float,
        metadata_overhead_bytes: float,
        context_length: int,
        dense_fp16_bytes: float,
        custom_breakdown: Optional[Dict[str, Any]] = None
    ) -> MemoryProfileResult:
        """Stops tracking and computes memory profile metrics."""
        end_rss = self._process.memory_info().rss
        rss_delta = max(0, end_rss - self._start_rss)

        peak_allocated = 0.0
        if torch.cuda.is_available() and self.device.type == "cuda":
            peak_allocated = float(torch.cuda.max_memory_allocated(self.device))
        elif torch.backends.mps.is_available() and self.device.type == "mps":
            try:
                peak_allocated = float(torch.mps.current_allocated_memory())
            except Exception:
                peak_allocated = float(rss_delta)
        else:
            peak_allocated = float(rss_delta)

        total_rep_bytes = algorithmic_bytes + metadata_overhead_bytes
        total_rep_bits = total_rep_bytes * 8.0
        effective_bpt = total_rep_bits / max(1, context_length)
        comp_ratio = total_rep_bytes / max(1.0, dense_fp16_bytes)

        return MemoryProfileResult(
            algorithmic_bytes=algorithmic_bytes,
            metadata_overhead_bytes=metadata_overhead_bytes,
            total_representation_bytes=total_rep_bytes,
            effective_bits_per_token=effective_bpt,
            compression_ratio=comp_ratio,
            physical_peak_allocated_bytes=peak_allocated,
            physical_rss_delta_bytes=float(rss_delta),
            device=str(self.device),
            breakdown=custom_breakdown or {}
        )
