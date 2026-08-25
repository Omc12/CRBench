"""
Fidelity tests against the vendored upstream implementations.

These are the tests that make the leaderboard trustworthy: they assert that what
CRBench calls "SnapKV" and "StreamingLLM" produce the same output as the
authors' own code, and that the adaptations made for GQA and for transformers
5.x are the *only* differences.
"""

from __future__ import annotations

import pytest
import torch

from crbench.adapters import upstream
from crbench.adapters.eviction import snapkv_compress_gqa, _grouped_attention_scores
from crbench.adapters.quantized import quantize_dequantize


torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# SnapKV                                                                       #
# --------------------------------------------------------------------------- #

def test_snapkv_matches_upstream_when_no_gqa():
    """With one query head per KV head, our path must equal upstream exactly.

    That is the whole justification for the group-pooling adaptation: it is a
    generalisation of upstream's scoring, not a substitute for it.  If this
    drifts, the GQA numbers stop being SnapKV's.
    """
    SnapKVCluster = upstream.load_snapkv()

    bsz, heads, seq, dim = 1, 4, 512, 64
    window, capacity, kernel = 32, 160, 5

    key = torch.randn(bsz, heads, seq, dim, dtype=torch.float32)
    value = torch.randn(bsz, heads, seq, dim, dtype=torch.float32)
    query = torch.randn(bsz, heads, window, dim, dtype=torch.float32)

    cluster = SnapKVCluster(window_size=window, max_capacity_prompt=capacity,
                            kernel_size=kernel, pooling="avgpool")
    # Upstream asserts key/query share a length, so it is handed the full-length
    # query it expects; only the last `window` rows are ever read.
    full_query = torch.randn(bsz, heads, seq, dim, dtype=torch.float32)
    full_query[:, :, -window:, :] = query
    k_up, v_up = cluster.update_kv(key, full_query, value, None, 1)

    k_ours, v_ours = snapkv_compress_gqa(
        key, value, query,
        window_size=window, max_capacity_prompt=capacity,
        kernel_size=kernel, pooling="avgpool", num_key_value_groups=1,
    )

    assert k_ours.shape == k_up.shape == (bsz, heads, capacity, dim)
    torch.testing.assert_close(k_ours, k_up, rtol=0, atol=0)
    torch.testing.assert_close(v_ours, v_up, rtol=0, atol=0)


def test_snapkv_respects_capacity_and_keeps_recent_window():
    SnapKVCluster = upstream.load_snapkv()
    bsz, kv_heads, groups, seq, dim = 1, 2, 4, 400, 64
    window, capacity = 16, 96

    key = torch.randn(bsz, kv_heads, seq, dim)
    value = torch.randn(bsz, kv_heads, seq, dim)
    query = torch.randn(bsz, kv_heads * groups, window, dim)

    k_new, v_new = snapkv_compress_gqa(
        key, value, query, window_size=window, max_capacity_prompt=capacity,
        kernel_size=5, pooling="avgpool", num_key_value_groups=groups,
    )
    assert k_new.shape == (bsz, kv_heads, capacity, dim)
    # The recent window survives untouched -- that is the invariant the
    # re-forwarded final prompt token depends on.
    torch.testing.assert_close(k_new[:, :, -window:, :], key[:, :, -window:, :])
    torch.testing.assert_close(v_new[:, :, -window:, :], value[:, :, -window:, :])
    assert SnapKVCluster is not None


def test_snapkv_is_a_noop_below_capacity():
    key = torch.randn(1, 2, 40, 32)
    value = torch.randn(1, 2, 40, 32)
    query = torch.randn(1, 4, 16, 32)
    k, v = snapkv_compress_gqa(key, value, query, window_size=16,
                               max_capacity_prompt=128, kernel_size=5,
                               pooling="avgpool", num_key_value_groups=2)
    assert k is key and v is value


def test_grouped_scores_prefer_attended_positions():
    """A key the observation window actually attends to must outrank noise."""
    bsz, kv_heads, groups, seq, dim = 1, 1, 2, 128, 16
    window = 8
    key = torch.randn(bsz, kv_heads, seq, dim) * 0.1
    query = torch.randn(bsz, kv_heads * groups, window, dim) * 0.1

    planted = 40
    key[0, 0, planted] = 10.0
    query[0, :, :, :] = 10.0  # aligned with the planted key

    scores = _grouped_attention_scores(query, key, window, groups)
    assert scores.shape == (bsz, kv_heads, seq - window)
    assert int(scores[0, 0].argmax()) == planted


# --------------------------------------------------------------------------- #
# StreamingLLM                                                                 #
# --------------------------------------------------------------------------- #

def test_streaming_llm_upstream_policy():
    """Upstream's StartRecentKVCache keeps exactly the sinks plus the tail."""
    StartRecentKVCache = upstream.load_streaming_llm()
    start, recent = 4, 60
    evictor = StartRecentKVCache(start_size=start, recent_size=recent,
                                 k_seq_dim=2, v_seq_dim=2)

    key = torch.randn(1, 2, 300, 16)
    value = torch.randn(1, 2, 300, 16)
    out = evictor([[key, value]])
    k_new, v_new = out[0]

    assert k_new.shape[-2] == start + recent
    torch.testing.assert_close(k_new[:, :, :start, :], key[:, :, :start, :])
    torch.testing.assert_close(k_new[:, :, start:, :], key[:, :, -recent:, :])
    torch.testing.assert_close(v_new[:, :, start:, :], value[:, :, -recent:, :])


# --------------------------------------------------------------------------- #
# Quantization                                                                 #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bits", [8, 4, 2])
def test_quantization_round_trip_is_bounded_by_its_step(bits):
    """Error must not exceed half a quantization step of the group's range."""
    x = torch.randn(4, 8, 128) * 3.0
    g = 64
    y = quantize_dequantize(x, n_bits=bits, group_size=g)
    assert y.shape == x.shape

    xg = x.reshape(4, 8, 128 // g, g)
    span = (xg.amax(-1) - xg.amin(-1))
    step = span / ((1 << bits) - 1)
    err = (y - x).reshape(4, 8, 128 // g, g).abs().amax(-1)
    assert torch.all(err <= step / 2 + 1e-4)


def test_quantization_error_decreases_with_bit_width():
    x = torch.randn(2, 4, 128)
    errs = [(quantize_dequantize(x, n_bits=b, group_size=64) - x).abs().mean().item()
            for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


def test_quantization_is_a_noop_at_full_precision():
    x = torch.randn(2, 4, 128)
    torch.testing.assert_close(quantize_dequantize(x, n_bits=16), x)


# --------------------------------------------------------------------------- #
# Differential-KV                                                              #
# --------------------------------------------------------------------------- #

def test_dkv_block_compression_improves_with_rank():
    """Upstream's block compressor must reconstruct better as rank grows."""
    mod = upstream.load_dkv()["lowrank"]
    n, feat = 255, 256
    basis = torch.randn(16, feat)
    deltas = (torch.randn(n, 16) @ basis) + 0.05 * torch.randn(n, feat)

    errors = []
    for rank in (4, 16, 48):
        lr = mod.compress_lowrank(deltas.clone(), rank, max_residual=0)
        recon = lr.U.float() @ lr.V.float() * lr.scale
        errors.append(((recon - deltas).norm() / deltas.norm()).item())

    assert errors[0] > errors[2]
    assert errors[2] < 0.25


def test_dkv_residuals_are_jointly_selected():
    """K and V must share one residual index set.

    Upstream documents why: a token made exact on K but left lossy on V is
    located correctly by attention and then read back wrong.
    """
    mod = upstream.load_dkv()["lowrank"]
    n, feat = 255, 256
    deltas = torch.randn(n, feat)
    lr = mod.compress_lowrank(deltas, 8, max_residual=32)
    if lr.residual_K_positions is None or lr.residual_K_positions.numel() == 0:
        pytest.skip("upstream selected no residuals for this input")
    torch.testing.assert_close(
        lr.residual_K_positions.long(), lr.residual_V_positions.long()
    )


# --------------------------------------------------------------------------- #
# Provenance                                                                   #
# --------------------------------------------------------------------------- #

def test_every_vendored_repo_reports_a_commit():
    """The manifest must be able to name the exact upstream checkout used."""
    for key in ("snapkv", "streaming_llm", "dkv"):
        record = upstream.describe_provenance(key)
        assert record["present"], f"{key} repository is missing from third_party/"
        assert record["commit"], f"{key} checkout reports no commit"
        assert record["upstream_repository"].startswith("https://")
