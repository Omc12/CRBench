"""
Unit tests for Context Adapters.
"""

import pytest
from crbench.core.registry import Registry
from crbench.core.budget import ContextBudget
from crbench.adapters.dense import DenseAdapter
from crbench.adapters.quantized import QuantizedKVAdapter
from crbench.adapters.eviction import EvictionKVAdapter
from crbench.adapters.merging import MergingKVAdapter
from crbench.adapters.compressed import LowRankCompressedKVAdapter
from crbench.adapters.custom_example import DKVContextAdapter


def test_adapter_registry():
    assert "dense" in Registry.list_adapters()
    assert "quantized" in Registry.list_adapters()
    assert "eviction" in Registry.list_adapters()
    assert "merging" in Registry.list_adapters()
    assert "compressed" in Registry.list_adapters()
    assert "dkv" in Registry.list_adapters()


def test_dense_adapter_metadata():
    adapter = DenseAdapter()
    meta = adapter.get_kv_metadata(context_length=4096)
    assert meta.effective_bits_per_element == 16.0
    assert meta.total_tokens_stored == 4096
    assert meta.compression_ratio == 1.0


def test_quantized_adapter_metadata():
    adapter = QuantizedKVAdapter(bits=4, group_size=64)
    meta = adapter.get_kv_metadata(context_length=4096)
    # Effective bits is 4 + scale overhead (16 / 64 = 0.25) = 4.25
    assert abs(meta.effective_bits_per_element - 4.25) < 1e-3
    assert meta.compression_ratio < 0.3


def test_eviction_adapter_metadata():
    adapter = EvictionKVAdapter(retention_ratio=0.25)
    meta = adapter.get_kv_metadata(context_length=4096)
    assert meta.total_tokens_stored <= 1024 + 160
    assert meta.compression_ratio < 0.35


def test_merging_adapter_metadata():
    adapter = MergingKVAdapter(merge_ratio=0.5)
    meta = adapter.get_kv_metadata(context_length=4096)
    assert meta.total_tokens_stored == 2048
    assert abs(meta.compression_ratio - 0.5) < 0.05


def test_compressed_adapter_metadata():
    adapter = LowRankCompressedKVAdapter(rank_ratio=0.25)
    meta = adapter.get_kv_metadata(context_length=4096)
    assert meta.compression_ratio < 0.3


def test_dkv_adapter_metadata():
    adapter = DKVContextAdapter(subspace_dim_ratio=0.5, token_sparsity=0.5)
    meta = adapter.get_kv_metadata(context_length=4096)
    assert meta.compression_ratio < 0.6
