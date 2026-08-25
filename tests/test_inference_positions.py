"""
Positional consistency across a cache transform.

A method that evicts or merges leaves the surviving keys carrying the rotary
phase they were written with -- their true positions in the prompt. transformers
derives ``position_ids`` from ``past_key_values.get_seq_length()`` when they are
not supplied, so a compressed cache would rotate the next query at the
*compressed* length: thousands of positions behind the keys it must match, with
RoPE's relative offsets running negative.

This is silent. The model keeps generating fluent text, and the only symptom is
that retrieval quality collapses -- which reads exactly like the compression
method being bad. On Qwen2.5-7B at 16384 tokens it drove SnapKV to 0.0% retention
at every budget, below StreamingLLM, which is not a plausible ordering for those
two methods on a needle task.
"""

from __future__ import annotations

import pytest
import torch

from crbench.core.inference import (chunked_prefill_generate, kv_tensors,
                                    rebuild_cache, cache_seq_length)


class _RecordingModel(torch.nn.Module):
    """Minimal stand-in that records the position_ids it is handed.

    Avoids downloading a model: the property under test is what the caller
    passes, not what any particular architecture does with it.
    """

    class _Config:
        num_hidden_layers = 2
        num_attention_heads = 2
        num_key_value_heads = 2
        hidden_size = 8
        head_dim = 4
        vocab_size = 32

        def get_text_config(self, decoder=False):
            return self

    def __init__(self) -> None:
        super().__init__()
        self.config = self._Config()
        self.seen_positions: list[int] = []

    def forward(self, input_ids=None, past_key_values=None, position_ids=None,
                use_cache=True, logits_to_keep=1, **kw):
        b, s = input_ids.shape
        if position_ids is not None:
            self.seen_positions.extend(position_ids.flatten().tolist())
        else:
            past = past_key_values.get_seq_length() if past_key_values is not None else 0
            self.seen_positions.extend(range(past, past + s))

        for layer_idx in range(self.config.num_hidden_layers):
            k = torch.zeros(b, self.config.num_key_value_heads, s, self.config.head_dim)
            past_key_values.update(k, k.clone(), layer_idx)

        logits = torch.zeros(b, 1, self.config.vocab_size)
        logits[..., 7] = 1.0
        return type("Out", (), {"logits": logits})()


def _keep_last(n: int):
    def transform(cache, input_ids, valid):
        pairs = [(k[..., -n:, :].clone(), v[..., -n:, :].clone())
                 for k, v in kv_tensors(cache, valid_length=valid)]
        return rebuild_cache(pairs), {"kept": n}
    return transform


def test_positions_continue_from_original_length_after_eviction():
    """Generated tokens must be placed after the *uncompressed* prompt."""
    model = _RecordingModel()
    prompt_len, keep, new_tokens = 64, 16, 4
    ids = torch.randint(0, 32, (1, prompt_len))

    trace = chunked_prefill_generate(
        model, ids, max_new_tokens=new_tokens, chunk_size=32,
        transform_cache=_keep_last(keep), empty_cache_between_chunks=False,
    )

    # Prefill covers 0..prompt_len-1; the re-forwarded final prompt token is at
    # prompt_len - 1; generated tokens follow at prompt_len, prompt_len + 1, ...
    tail = model.seen_positions[prompt_len:]
    assert tail[0] == prompt_len - 1, (
        f"re-forwarded prompt token placed at {tail[0]}, expected {prompt_len - 1}")
    assert tail[1:] == list(range(prompt_len, prompt_len + new_tokens - 1)), (
        f"generated tokens placed at {tail[1:]}, expected "
        f"{list(range(prompt_len, prompt_len + new_tokens - 1))}")

    # And emphatically not restarted from the compressed cache length.
    assert keep not in tail[:2], "positions restarted from the compressed length"
    assert trace.kv_tokens_after_transform == keep


def test_positions_unchanged_when_no_transform_runs():
    """Without a transform the fix must be a no-op."""
    model = _RecordingModel()
    prompt_len, new_tokens = 48, 4
    ids = torch.randint(0, 32, (1, prompt_len))

    chunked_prefill_generate(model, ids, max_new_tokens=new_tokens, chunk_size=16,
                             empty_cache_between_chunks=False)

    assert model.seen_positions[:prompt_len] == list(range(prompt_len))
    assert model.seen_positions[prompt_len:] == list(
        range(prompt_len, prompt_len + new_tokens - 1))


@pytest.mark.parametrize("keep", [8, 24])
def test_transformed_cache_drops_the_reforwarded_token(keep):
    """The final prompt token is re-forwarded, so it must not be duplicated."""
    model = _RecordingModel()
    prompt_len = 64
    ids = torch.randint(0, 32, (1, prompt_len))
    trace = chunked_prefill_generate(
        model, ids, max_new_tokens=3, chunk_size=32,
        transform_cache=_keep_last(keep), empty_cache_between_chunks=False,
    )
    # keep tokens, minus the one cropped, plus the re-forward and 2 decode steps.
    assert trace.kv_tokens_after_transform == keep
