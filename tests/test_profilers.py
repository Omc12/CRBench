"""
Unit tests for Memory and Latency Profilers.
"""

import time
import pytest
from crbench.profiler.memory import MemoryProfiler
from crbench.profiler.latency import LatencyProfiler


def test_memory_profiler():
    profiler = MemoryProfiler()
    profiler.start_tracking()
    
    # Simulate work
    res = profiler.stop_tracking(
        algorithmic_bytes=1024 * 1024 * 10,  # 10 MB
        metadata_overhead_bytes=1024 * 512,  # 0.5 MB
        context_length=8192,
        dense_fp16_bytes=1024 * 1024 * 40    # 40 MB
    )
    
    assert res.algorithmic_bytes == 1024 * 1024 * 10
    assert abs(res.compression_ratio - (10.5 / 40.0)) < 1e-3
    assert res.effective_bits_per_token > 0.0


def test_latency_profiler():
    profiler = LatencyProfiler()
    
    def dummy_gen():
        time.sleep(0.01)
        return "output"

    lat_res = profiler.benchmark_generation(
        generate_fn=dummy_gen,
        prompt_tokens=4096,
        max_new_tokens=16
    )

    assert lat_res.total_time_seconds > 0.005
    assert lat_res.ttft_ms > 0.0
    assert lat_res.decode_throughput_tok_per_sec > 0.0
