"""
Unit tests for ContextBudget.
"""

import pytest
from crbench.core.budget import ContextBudget, BudgetType


def test_budget_conversions():
    # 32 layers, 32 kv heads, 128 head dim, 8192 context length
    num_layers = 32
    num_kv_heads = 32
    head_dim = 128
    ctx_len = 8192
    
    # 1. From bits per token
    b_bpt = ContextBudget.from_bits_per_token(4.0)
    assert b_bpt.budget_type == BudgetType.BITS_PER_TOKEN
    assert b_bpt.to_bits_per_token(num_layers, num_kv_heads, head_dim, ctx_len) == 4.0
    
    # 2. From compression ratio (0.25 of FP16 -> 4.0 bpt)
    b_ratio = ContextBudget.from_compression_ratio(0.25)
    assert b_ratio.budget_type == BudgetType.COMPRESSION_RATIO
    assert b_ratio.to_bits_per_token(num_layers, num_kv_heads, head_dim, ctx_len) == 4.0

    # 3. From token capacity (2048 / 8192 = 0.25 -> 4.0 bpt)
    b_tok = ContextBudget.from_token_capacity(2048)
    assert b_tok.budget_type == BudgetType.TOKEN_CAPACITY
    assert b_tok.to_bits_per_token(num_layers, num_kv_heads, head_dim, ctx_len) == 4.0

    # 4. Total bytes round-trip
    bytes_val = b_bpt.to_bytes(num_layers, num_kv_heads, head_dim, ctx_len)
    b_bytes = ContextBudget.from_bytes(bytes_val)
    recovered_bpt = b_bytes.to_bits_per_token(num_layers, num_kv_heads, head_dim, ctx_len)
    assert abs(recovered_bpt - 4.0) < 1e-4
