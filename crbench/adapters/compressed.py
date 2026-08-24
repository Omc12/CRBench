"""
Low-Rank and Linear Compressed KV adapter for CRBench.
Compresses the hidden head dimension D_head -> D_rank via low-rank projection.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.registry import Registry


@Registry.register_adapter("compressed")
@Registry.register_adapter("low_rank_kv")
@Registry.register_adapter("linear_kv")
class LowRankCompressedKVAdapter(BaseContextAdapter):
    """
    Low-Rank / Linear Projection Compressed KV Adapter.
    Reduces the key/value hidden dimension per head from D to r (rank ratio r/D).
    """

    def __init__(self, name: str = "low_rank_kv", rank_ratio: float = 0.25, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.rank_ratio = self.config.get("rank_ratio", rank_ratio)

    @property
    def method_type(self) -> str:
        return "compressed"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        if budget.budget_type == BudgetType.COMPRESSION_RATIO:
            self.rank_ratio = float(budget.value)
        elif budget.budget_type == BudgetType.BITS_PER_TOKEN:
            self.rank_ratio = float(budget.value) / 16.0

    def forward_or_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **kwargs: Any
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not attached to LowRankCompressedKVAdapter.")

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        def low_rank_project(tensor: torch.Tensor, ratio: float) -> torch.Tensor:
            if ratio >= 1.0:
                return tensor
            orig_dtype = tensor.dtype
            d = tensor.shape[-1]
            t_float = tensor.float()
            freq = torch.fft.rfft(t_float, dim=-1)
            keep_freq = max(1, int(freq.shape[-1] * ratio))
            freq[..., keep_freq:] = 0.0
            reconstructed = torch.fft.irfft(freq, n=d, dim=-1)
            return reconstructed.to(orig_dtype)

        hooks = []
        if self.rank_ratio < 1.0:
            layers = getattr(self.model, "model", self.model)
            layer_list = getattr(layers, "layers", getattr(layers, "decoder_layers", getattr(layers, "h", [])))
            for layer in layer_list:
                attn = getattr(layer, "self_attn", getattr(layer, "attn", getattr(layer, "attention", None)))
                if attn is not None:
                    k_proj = getattr(attn, "k_proj", getattr(attn, "key", None))
                    v_proj = getattr(attn, "v_proj", getattr(attn, "value", None))
                    if k_proj is not None:
                        hooks.append(k_proj.register_forward_hook(
                            lambda m, inp, out, r=self.rank_ratio: low_rank_project(out, ratio=r)
                        ))
                    if v_proj is not None:
                        hooks.append(v_proj.register_forward_hook(
                            lambda m, inp, out, r=self.rank_ratio: low_rank_project(out, ratio=r)
                        ))

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                    eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                    **kwargs
                )
            return outputs
        finally:
            for h in hooks:
                h.remove()

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers = getattr(self.model.config, "num_hidden_layers", 32) if self.model else 32
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", getattr(self.model.config, "num_attention_heads", 32)) if self.model else 32
        hidden_size = getattr(self.model.config, "hidden_size", 4096) if self.model else 4096
        num_heads = getattr(self.model.config, "num_attention_heads", 32) if self.model else 32
        head_dim = getattr(self.model.config, "head_dim", hidden_size // num_heads) if self.model else 128

        effective_dim = max(4, int(head_dim * self.rank_ratio))
        # 2 tensors (Key, Value) * num_layers * num_kv_heads * effective_dim * context_length * 2 bytes (FP16)
        algorithmic_bytes = 2.0 * num_layers * num_kv_heads * effective_dim * context_length * 2.0

        effective_bpe = (algorithmic_bytes * 8.0) / max(1, 2 * num_layers * num_kv_heads * head_dim * context_length)

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
            metadata_overhead_bytes=0.0,
            custom_metrics={"rank_ratio": self.rank_ratio, "effective_head_dim": effective_dim}
        )
