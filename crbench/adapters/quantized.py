"""
KV cache quantization adapter (INT8 / INT4 / INT2) for CRBench.

Quantization is applied to what is *stored*, not to what is computed: after each
prefill chunk lands in the cache, the newly written keys and values are
round-tripped through the target bit-width, so every later chunk -- and every
decoded token -- attends to the quantized history.  This is the property that
distinguishes a KV-cache quantizer from activation quantization, and the
previous implementation, which hooked ``k_proj``/``v_proj`` outputs, did not
have it: hooking the projection also perturbs the current chunk's own attention
computation, which a cache quantizer never touches.

Scheme: asymmetric (min-max) grouped quantization along the head dimension, the
layout KIVI and the mainstream KV-quantization baselines use.  Each group of
``group_size`` elements carries one fp16 scale and one fp16 zero-point, and both
are charged to the method in ``get_kv_metadata`` -- at INT2 with group 64 that
metadata is 0.5 bits/element, a quarter of the payload, which is exactly the
kind of overhead a resource-aware benchmark exists to surface.

The stored tensors remain bf16 after the round trip (simulated quantization).
Quality is therefore exact; the memory figure is analytical, and is labelled as
such in the provenance audit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.inference import cache_layers
from crbench.core.registry import Registry


def quantize_dequantize(
    tensor: torch.Tensor,
    n_bits: int = 4,
    group_size: int = 64,
) -> torch.Tensor:
    """Asymmetric grouped quantization round trip along the last dimension.

    Args:
        tensor: (..., D) with D divisible by ``group_size`` (falls back to a
            single per-row group when it is not).
    """
    if n_bits >= 16:
        return tensor

    orig_shape = tensor.shape
    orig_dtype = tensor.dtype
    d = orig_shape[-1]
    g = group_size if (d % group_size == 0 and d >= group_size) else d

    t = tensor.reshape(*orig_shape[:-1], d // g, g).to(torch.float32)
    lo = t.amin(dim=-1, keepdim=True)
    hi = t.amax(dim=-1, keepdim=True)

    levels = (1 << n_bits) - 1
    scale = ((hi - lo) / levels).clamp(min=1e-8)
    q = torch.round((t - lo) / scale).clamp_(0, levels)
    deq = q * scale + lo

    return deq.reshape(orig_shape).to(orig_dtype)


@Registry.register_adapter("quantized")
@Registry.register_adapter("kv_quant")
@Registry.register_adapter("int8")
@Registry.register_adapter("int4")
@Registry.register_adapter("int2")
class QuantizedKVAdapter(BaseContextAdapter):
    """Quantized KV cache at INT8 / INT4 / INT2."""

    streaming_transform = True

    def __init__(
        self,
        name: str = "kv_quant_int4",
        bits: int = 4,
        group_size: int = 64,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.bits = int(self.config.get("bits", bits))
        self.group_size = int(self.config.get("group_size", group_size))
        self.scale_bits = 16   # fp16 scale
        self.zero_bits = 16    # fp16 zero-point

    @property
    def method_type(self) -> str:
        return "quantization"

    def provenance(self) -> Dict[str, Any]:
        return {
            "implementation": "crbench_internal",
            "scheme": "asymmetric min-max grouped quantization along head_dim",
            "applied_to": "stored KV cache (streaming, after each prefill chunk and each decoded token)",
            "simulated": True,
            "note": (
                "Tensors are round-tripped through the target bit-width and kept in bf16, "
                "so quality is exact for the scheme while the memory figure is analytical."
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

    def _quantize_span(self, cache: Any, start: int, end: int) -> None:
        if self.bits >= 16 or end <= start:
            return
        for layer in cache_layers(cache):
            k, v = layer.keys, layer.values
            if k is None or v is None:
                continue
            k[..., start:end, :] = quantize_dequantize(k[..., start:end, :], self.bits, self.group_size)
            v[..., start:end, :] = quantize_dequantize(v[..., start:end, :], self.bits, self.group_size)

    def on_chunk_stored(self, cache: Any, start: int, end: int, valid_length: int) -> None:
        self._quantize_span(cache, start, end)

    def on_token_stored(self, cache: Any, position: int) -> None:
        self._quantize_span(cache, position, position + 1)

    # ------------------------------------------------------------------ #
    # Resource accounting                                                 #
    # ------------------------------------------------------------------ #

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers, num_kv_heads, head_dim = self.model_kv_geometry()
        total_elements = self.dense_element_count(context_length)

        algorithmic_bytes = total_elements * self.bits / 8.0

        g = self.group_size if (head_dim % self.group_size == 0 and head_dim >= self.group_size) else head_dim
        num_groups = total_elements / max(1, g)
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
                "scale_zero_bits_per_element": (self.scale_bits + self.zero_bits) / g,
            },
        )
