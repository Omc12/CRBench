"""
KV cache merging / pooling adapter for CRBench.

Consecutive prompt tokens are merged into centroid tokens by averaging their
keys and values within a fixed temporal window, keeping the attention sinks and
a recent window intact.  This is the token-merging family (temporal pooling,
the simplest member of the cluster-and-merge group), and it is the natural
counterpart to eviction: eviction discards the tokens it does not keep, merging
folds them into a survivor.

Merging happens on the resident prompt cache.  The previous implementation
instead subsampled the *input token ids* with a stride before the forward pass,
which deletes text rather than merging states -- a strictly different and much
more destructive intervention, and one whose cost model does not match the
merged-centroid representation it was accounted as.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.inference import kv_tensors, rebuild_cache
from crbench.core.registry import Registry


@Registry.register_adapter("merging")
@Registry.register_adapter("kv_merging")
@Registry.register_adapter("token_merge")
class MergingKVAdapter(BaseContextAdapter):
    """Temporal mean-pooling of prompt KV states under a bits-per-token budget."""

    oneshot_transform = True

    def __init__(
        self,
        name: str = "kv_merging",
        merge_ratio: float = 0.25,
        sink_tokens: int = 4,
        recent_window: int = 32,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.merge_ratio = float(self.config.get("merge_ratio", merge_ratio))
        self.sink_tokens = int(self.config.get("sink_tokens", sink_tokens))
        self.recent_window = int(self.config.get("recent_window", recent_window))

    @property
    def method_type(self) -> str:
        return "merging"

    def provenance(self) -> Dict[str, Any]:
        return {
            "implementation": "crbench_internal",
            "scheme": "temporal mean-pooling of consecutive KV states, sinks and recent window preserved",
            "applied_to": "resident prompt cache, one-shot after prefill",
        }

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        if budget.budget_type == BudgetType.COMPRESSION_RATIO:
            self.merge_ratio = float(budget.value)
        elif budget.budget_type == BudgetType.BITS_PER_TOKEN:
            # Merged centroids are stored dense, so a budget of b bits/element
            # buys b/16 as many stored tokens.
            self.merge_ratio = float(budget.value) / 16.0

    def _plan(self, valid_length: int) -> Tuple[int, int, int, int]:
        """(sink, recent, merge_factor, merged_token_count)."""
        sink = min(self.sink_tokens, max(0, valid_length // 4))
        recent = min(self.recent_window, max(0, valid_length - sink - 1))
        middle = max(0, valid_length - sink - recent)
        factor = max(1, int(round(1.0 / max(1e-6, self.merge_ratio))))
        merged = (middle + factor - 1) // factor if middle else 0
        return sink, recent, factor, merged

    def transform_cache(
        self,
        cache: Any,
        input_ids: torch.Tensor,
        valid_length: int,
    ) -> Tuple[Any, Dict[str, Any]]:
        sink, recent, factor, merged = self._plan(valid_length)
        if factor <= 1 or merged == 0:
            return cache, {"applied": False, "merge_factor": factor}

        middle = valid_length - sink - recent
        pad = (factor - (middle % factor)) % factor

        new_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for k, v in kv_tensors(cache, valid_length=valid_length):
            parts_k, parts_v = [], []
            if sink:
                parts_k.append(k[..., :sink, :])
                parts_v.append(v[..., :sink, :])

            k_mid = k[..., sink:sink + middle, :]
            v_mid = v[..., sink:sink + middle, :]
            if pad:
                # Repeat the final state so the last group is a true mean of the
                # tokens it covers rather than a mean over zeros.
                k_mid = torch.cat([k_mid, k_mid[..., -1:, :].expand(-1, -1, pad, -1)], dim=-2)
                v_mid = torch.cat([v_mid, v_mid[..., -1:, :].expand(-1, -1, pad, -1)], dim=-2)
            b, h, n, d = k_mid.shape
            parts_k.append(k_mid.view(b, h, n // factor, factor, d).mean(dim=-2))
            parts_v.append(v_mid.view(b, h, n // factor, factor, d).mean(dim=-2))

            if recent:
                parts_k.append(k[..., -recent:, :])
                parts_v.append(v[..., -recent:, :])

            new_pairs.append((torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)))

        retained = int(new_pairs[0][0].shape[-2])
        new_cache = rebuild_cache(new_pairs, cache=cache)
        del new_pairs
        return new_cache, {
            "applied": True,
            "merge_factor": factor,
            "retained_tokens": retained,
            "sink_tokens": sink,
            "recent_window_exact": recent,
        }

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers, num_kv_heads, head_dim = self.model_kv_geometry()
        sink, recent, factor, merged = self._plan(context_length)
        stored = sink + merged + recent

        algorithmic_bytes = 2.0 * num_layers * num_kv_heads * head_dim * stored * 2.0
        # Each stored token records the span it represents (int32 start + count)
        # so decoding can place it at a RoPE-consistent position.
        metadata_bytes = stored * 8.0 * num_layers

        dense_elems = self.dense_element_count(context_length)
        effective_bpe = (algorithmic_bytes + metadata_bytes) * 8.0 / max(1, dense_elems)

        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=effective_bpe,
            total_tokens_stored=stored,
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            algorithmic_bytes=algorithmic_bytes,
            metadata_overhead_bytes=metadata_bytes,
            custom_metrics={
                "merge_factor": factor,
                "merged_tokens": merged,
                "stored_tokens": stored,
                "sink_tokens": sink,
                "recent_window_exact": recent,
            },
        )
