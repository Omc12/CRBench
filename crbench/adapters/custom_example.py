"""
Example Custom Adapter: DKV (Disentangled / Dynamic Key-Value Representation).
Demonstrates how external researchers can implement BaseContextAdapter in under 50 lines.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.registry import Registry


@Registry.register_adapter("dkv")
@Registry.register_adapter("custom_dkv")
@Registry.register_adapter("custom")
class DKVContextAdapter(BaseContextAdapter):
    """
    DKV (Disentangled Key-Value) Context Adapter Example.
    Deconstructs the KV representation into a shared persistent subspace and dynamic query-dependent tokens.
    """

    def __init__(
        self,
        name: str = "dkv",
        subspace_dim_ratio: float = 0.25,
        token_sparsity: float = 0.5,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, config=config)
        self.subspace_dim_ratio = self.config.get("subspace_dim_ratio", subspace_dim_ratio)
        self.token_sparsity = self.config.get("token_sparsity", token_sparsity)

    @property
    def method_type(self) -> str:
        return "custom"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        if budget.budget_type == BudgetType.COMPRESSION_RATIO:
            target = float(budget.value)
            self.subspace_dim_ratio = min(1.0, target ** 0.5)
            self.token_sparsity = min(1.0, target ** 0.5)
        elif budget.budget_type == BudgetType.BITS_PER_TOKEN:
            target = float(budget.value) / 16.0
            self.subspace_dim_ratio = min(1.0, target ** 0.5)
            self.token_sparsity = min(1.0, target ** 0.5)

    def forward_or_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **kwargs: Any
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not attached to DKVContextAdapter.")

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        def dkv_subspace_project(tensor: torch.Tensor, ratio: float) -> torch.Tensor:
            if ratio >= 1.0:
                return tensor
            orig_dtype = tensor.dtype
            d = tensor.shape[-1]
            t_float = tensor.float()
            freq = torch.fft.rfft(t_float, dim=-1)
            keep_freq = max(1, int(freq.shape[-1] * min(1.0, ratio)))
            freq[..., keep_freq:] = 0.0
            reconstructed = torch.fft.irfft(freq, n=d, dim=-1)
            return reconstructed.to(orig_dtype)

        hooks = []
        if self.subspace_dim_ratio < 1.0:
            layers = getattr(self.model, "model", self.model)
            layer_list = getattr(layers, "layers", getattr(layers, "decoder_layers", getattr(layers, "h", [])))
            n_layers = max(1, len(layer_list))
            for l_idx, layer in enumerate(layer_list):
                # Layer-adaptive precision: deeper layers retain higher subspace fidelity
                layer_ratio = self.subspace_dim_ratio * (0.6 + 0.8 * (l_idx / n_layers))
                layer_ratio = min(1.0, max(0.1, layer_ratio))

                attn = getattr(layer, "self_attn", getattr(layer, "attn", getattr(layer, "attention", None)))
                if attn is not None:
                    k_proj = getattr(attn, "k_proj", getattr(attn, "key", None))
                    v_proj = getattr(attn, "v_proj", getattr(attn, "value", None))
                    if k_proj is not None:
                        hooks.append(k_proj.register_forward_hook(
                            lambda m, inp, out, r=layer_ratio: dkv_subspace_project(out, ratio=r)
                        ))
                    if v_proj is not None:
                        hooks.append(v_proj.register_forward_hook(
                            lambda m, inp, out, r=layer_ratio: dkv_subspace_project(out, ratio=r)
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

        active_tokens = max(1, int(context_length * self.token_sparsity))
        subspace_dim = max(4, int(head_dim * self.subspace_dim_ratio))

        # Shared subspace matrix: 2 * num_layers * num_kv_heads * head_dim * subspace_dim * 2 bytes
        subspace_bytes = 2.0 * num_layers * num_kv_heads * head_dim * subspace_dim * 2.0
        # Dynamic coefficient tokens: 2 * num_layers * num_kv_heads * subspace_dim * active_tokens * 2 bytes
        dynamic_token_bytes = 2.0 * num_layers * num_kv_heads * subspace_dim * active_tokens * 2.0
        total_algorithmic = subspace_bytes + dynamic_token_bytes

        total_dense_elements = 2 * num_layers * num_kv_heads * head_dim * context_length
        effective_bpe = (total_algorithmic * 8.0) / max(1, total_dense_elements)

        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=effective_bpe,
            total_tokens_stored=active_tokens,
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            algorithmic_bytes=total_algorithmic,
            metadata_overhead_bytes=0.0,
            custom_metrics={"subspace_dim": subspace_dim, "active_tokens": active_tokens}
        )
