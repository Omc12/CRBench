"""
Adapters module for CRBench.
"""

from crbench.adapters.dense import DenseAdapter
from crbench.adapters.quantized import QuantizedKVAdapter
from crbench.adapters.eviction import EvictionKVAdapter
from crbench.adapters.merging import MergingKVAdapter
from crbench.adapters.compressed import LowRankCompressedKVAdapter
from crbench.adapters.custom_example import DKVContextAdapter

__all__ = [
    "DenseAdapter",
    "QuantizedKVAdapter",
    "EvictionKVAdapter",
    "MergingKVAdapter",
    "LowRankCompressedKVAdapter",
    "DKVContextAdapter",
]
