"""
KV cache quantization adapter (INT8 / INT4 / INT2) for CRBench.

Quantization is applied to what is *stored*, not to what is computed: after each
prefill chunk lands in the cache, the newly written keys and values are
round-tripped through the target bit-width, so every later chunk -- and every
decoded token -- attends to the quantized history.  This is the property that
distinguishes a KV-cache quantizer from activation quantization, and hooking
``k_proj``/``v_proj`` outputs, as an earlier revision did, does not have it:
that also perturbs the current chunk's own attention computation, which a cache
quantizer never touches.

Grouping axis
-------------
Keys are quantized **per channel** (grouped along the token axis) and values
**per token** (grouped along the head dimension).  This asymmetry is the central
empirical finding of KIVI (Liu et al., 2024): key channels carry
channel-specific outliers, so grouping a key's 128 head-dimension entries into
one scale lets a few large channels set the range for all of them and destroys
the small ones.  Values have no such structure and quantize fine per token.

The distinction is not cosmetic.  Measured here on Qwen2.5-7B NF4 at 2048
tokens, single-needle retrieval: per-token keys at 4 bits scored 0%, while
8 bits scored 100% -- the collapse was the grouping axis, not the bit-width.
Reporting the per-token variant as "INT4 KV quantization" would have put a
strawman in the leaderboard.

Because per-channel key groups span ``group_size`` tokens, tokens that do not
fill a group stay in full precision.  KIVI does the same, calling it the
residual window; ``get_kv_metadata`` charges those tokens at 16 bits.

The stored tensors remain bf16 after the round trip (simulated quantization).
Quality is therefore exact for the scheme; the memory figure is analytical, and
is labelled as such in the provenance audit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.inference import cache_layers, growing_layer_indices
from crbench.core.registry import Registry


def _quantize_groups(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Asymmetric min-max round trip over the last axis of ``x``."""
    lo = x.amin(dim=-1, keepdim=True)
    hi = x.amax(dim=-1, keepdim=True)
    levels = (1 << n_bits) - 1
    scale = ((hi - lo) / levels).clamp(min=1e-8)
    q = torch.round((x - lo) / scale).clamp_(0, levels)
    return q * scale + lo


def quantize_dequantize(
    tensor: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 64,
) -> torch.Tensor:
    """Per-token quantization: group along the head dimension (the last axis).

    Used for values, and for keys only when explicitly requested.
    """
    if n_bits >= 16:
        return tensor
    orig_shape = tensor.shape
    d = orig_shape[-1]
    g = group_size if (d % group_size == 0 and d >= group_size) else d
    x = tensor.reshape(*orig_shape[:-1], d // g, g).to(torch.float32)
    return _quantize_groups(x, n_bits).reshape(orig_shape).to(tensor.dtype)


def quantize_dequantize_per_channel(
    tensor: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 64,
) -> Tuple[torch.Tensor, int]:
    """Per-channel quantization: group along the token axis.

    Args:
        tensor: (B, H, S, D).

    Returns:
        The round-tripped tensor and the number of leading tokens that were
        quantized.  Tokens beyond that did not fill a group and are returned
        untouched, in full precision -- KIVI's residual window.
    """
    if n_bits >= 16:
        return tensor, tensor.shape[-2]

    b, h, s, d = tensor.shape
    n_full = (s // group_size) * group_size
    if n_full == 0:
        return tensor, 0

    out = tensor.clone()
    # (B, H, S, D) -> (B, H, D, S) so groups run along tokens for each channel.
    x = tensor[..., :n_full, :].transpose(-1, -2).to(torch.float32)
    x = x.reshape(b, h, d, n_full // group_size, group_size)
    deq = _quantize_groups(x, n_bits)
    deq = deq.reshape(b, h, d, n_full).transpose(-1, -2)
    out[..., :n_full, :] = deq.to(tensor.dtype)
    return out, n_full


@Registry.register_adapter("quantized")
@Registry.register_adapter("kv_quant")
@Registry.register_adapter("int8")
@Registry.register_adapter("int4")
@Registry.register_adapter("int2")
class QuantizedKVAdapter(BaseContextAdapter):
    """KIVI-style quantized KV cache at INT8 / INT4 / INT2."""

    streaming_transform = True

    def __init__(
        self,
        name: str = "kv_quant_int4",
        bits: int = 4,
        group_size: int = 64,
        key_per_channel: bool = True,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.bits = int(self.config.get("bits", bits))
        self.group_size = int(self.config.get("group_size", group_size))
        self.key_per_channel = bool(self.config.get("key_per_channel", key_per_channel))
        self.scale_bits = 16   # fp16 scale
        self.zero_bits = 16    # fp16 zero-point
        # Tokens left in full precision because they did not fill a key group.
        self._residual_tokens = 0

    @property
    def method_type(self) -> str:
        return "quantization"

    def provenance(self) -> Dict[str, Any]:
        return {
            "implementation": "crbench_internal",
            "scheme": (
                "asymmetric min-max grouped quantization; keys per-channel "
                "(grouped along tokens), values per-token (grouped along head_dim)"
                if self.key_per_channel else
                "asymmetric min-max grouped quantization; keys and values per-token"
            ),
            "follows": "KIVI (Liu et al., 2024) grouping asymmetry",
            "applied_to": "stored KV cache (streaming, after each prefill chunk and each decoded token)",
            "simulated": True,
            "note": (
                "Tensors are round-tripped through the target bit-width and kept in bf16, "
                "so quality is exact for the scheme while the memory figure is analytical. "
                "Tokens that do not fill a per-channel key group stay in full precision "
                "and are charged at 16 bits."
            ),
        }

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        if budget.budget_type == BudgetType.BITS_PER_TOKEN:
            self.bits = max(2, min(16, int(round(budget.value))))
        elif budget.budget_type == BudgetType.COMPRESSION_RATIO:
            self.bits = max(2, min(16, int(round(16.0 * budget.value))))

    # ------------------------------------------------------------------ #
    # Streaming transform                                                 #
    # ------------------------------------------------------------------ #

    def begin_query(self, model: Any, input_ids: torch.Tensor) -> None:
        self._residual_tokens = 0

    def _quantize_span(self, cache: Any, start: int, end: int) -> None:
        if self.bits >= 16 or end <= start:
            return
        residual = 0
        layers = cache_layers(cache)
        # Sliding-window and linear-attention layers hold bounded state that does
        # not grow with context, so quantizing them saves nothing and their
        # length does not match the span being written.
        for idx in growing_layer_indices(cache):
            layer = layers[idx]
            k, v = layer.keys, layer.values
            if k is None or v is None:
                continue

            v[..., start:end, :] = quantize_dequantize(
                v[..., start:end, :], self.bits, self.group_size
            )

            if self.key_per_channel:
                kq, n_full = quantize_dequantize_per_channel(
                    k[..., start:end, :], self.bits, self.group_size
                )
                k[..., start:end, :] = kq
                residual = (end - start) - n_full
            else:
                k[..., start:end, :] = quantize_dequantize(
                    k[..., start:end, :], self.bits, self.group_size
                )
        self._residual_tokens += residual

    def on_chunk_stored(self, cache: Any, start: int, end: int, valid_length: int) -> None:
        self._quantize_span(cache, start, end)

    def on_token_stored(self, cache: Any, position: int) -> None:
        # A single decoded token cannot fill a per-channel key group, so its key
        # joins the residual window; its value still quantizes per token.
        if self.bits >= 16:
            return
        if not self.key_per_channel:
            self._quantize_span(cache, position, position + 1)
            return
        layers = cache_layers(cache)
        for idx in growing_layer_indices(cache):
            v = layers[idx].values
            if v is None:
                continue
            v[..., position:position + 1, :] = quantize_dequantize(
                v[..., position:position + 1, :], self.bits, self.group_size
            )
        self._residual_tokens += 1

    # ------------------------------------------------------------------ #
    # Resource accounting                                                 #
    # ------------------------------------------------------------------ #

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers, num_kv_heads, head_dim = self.model_kv_geometry()
        total_elements = self.dense_element_count(context_length)

        # Keys in the residual window stay at full precision.
        residual = min(context_length, max(0, self._residual_tokens))
        if self.key_per_channel and residual == 0:
            residual = context_length % self.group_size
        key_elements = total_elements // 2
        residual_frac = residual / max(1, context_length)
        residual_key_elements = key_elements * residual_frac

        quantized_elements = total_elements - residual_key_elements
        algorithmic_bytes = (
            quantized_elements * self.bits + residual_key_elements * 16.0
        ) / 8.0

        g = self.group_size if (head_dim % self.group_size == 0 and head_dim >= self.group_size) else head_dim
        num_groups = quantized_elements / max(1, g)
        metadata_bytes = num_groups * (self.scale_bits + self.zero_bits) / 8.0

        effective_bpe = (algorithmic_bytes + metadata_bytes) * 8.0 / max(1, total_elements)

        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=effective_bpe,
            total_tokens_stored=context_length,
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            algorithmic_bytes=algorithmic_bytes,
            metadata_overhead_bytes=metadata_bytes,
            custom_metrics={
                "bits": self.bits,
                "group_size": g,
                "key_grouping": "per_channel" if self.key_per_channel else "per_token",
                "value_grouping": "per_token",
                "full_precision_residual_tokens": residual,
                "scale_zero_bits_per_element": (self.scale_bits + self.zero_bits) / g,
            },
        )
