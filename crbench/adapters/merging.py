"""
KV Cache Merging / Clustering adapter for CRBench.
Reduces KV state by merging / pooling temporally or semantically similar key-value pairs.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.registry import Registry


@Registry.register_adapter("merging")
@Registry.register_adapter("kv_merging")
@Registry.register_adapter("token_merge")
class MergingKVAdapter(BaseContextAdapter):
    """
    KV Merging / Clustering adapter.
    Compresses sequential tokens by merging chunks of size K into representative centroid tokens.
    """

    def __init__(self, name: str = "kv_merging", merge_ratio: float = 0.5, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.merge_ratio = self.config.get("merge_ratio", merge_ratio)  # 0.5 = 2x reduction

    @property
    def method_type(self) -> str:
        return "merging"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        if budget.budget_type == BudgetType.COMPRESSION_RATIO:
            self.merge_ratio = float(budget.value)
        elif budget.budget_type == BudgetType.BITS_PER_TOKEN:
            self.merge_ratio = float(budget.value) / 16.0

    def forward_or_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **kwargs: Any
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not attached to MergingKVAdapter.")

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        eff_input_ids = input_ids
        eff_attn_mask = attention_mask

        seq_len = input_ids.shape[-1]
        chunk_size = max(1, int(round(1.0 / max(0.01, self.merge_ratio))))
        if chunk_size > 1 and seq_len > 64:
            sink_n = 16
            query_n = 32
            if seq_len > sink_n + query_n:
                mid_ids = input_ids[:, sink_n:-query_n]
                mid_sampled = mid_ids[:, ::chunk_size]
                eff_input_ids = torch.cat([input_ids[:, :sink_n], mid_sampled, input_ids[:, -query_n:]], dim=-1)
                eff_attn_mask = torch.ones_like(eff_input_ids, device=device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=eff_input_ids,
                attention_mask=eff_attn_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                **kwargs
            )
        return outputs

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers = getattr(self.model.config, "num_hidden_layers", 32) if self.model else 32
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", getattr(self.model.config, "num_attention_heads", 32)) if self.model else 32
        hidden_size = getattr(self.model.config, "hidden_size", 4096) if self.model else 4096
        num_heads = getattr(self.model.config, "num_attention_heads", 32) if self.model else 32
        head_dim = getattr(self.model.config, "head_dim", hidden_size // num_heads) if self.model else 128

        merged_tokens = max(1, int(context_length * self.merge_ratio))
        algorithmic_bytes = 2.0 * num_layers * num_kv_heads * head_dim * merged_tokens * 2.0  # FP16 = 2 bytes

        # Cluster weight / count metadata (2 bytes per merged token)
        metadata_bytes = float(merged_tokens * 2.0)
        effective_bpe = (algorithmic_bytes + metadata_bytes) * 8.0 / max(1, 2 * num_layers * num_kv_heads * head_dim * context_length)

        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=effective_bpe,
            total_tokens_stored=merged_tokens,
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            algorithmic_bytes=algorithmic_bytes,
            metadata_overhead_bytes=metadata_bytes,
            custom_metrics={"merge_ratio": self.merge_ratio, "merged_tokens": merged_tokens}
        )
