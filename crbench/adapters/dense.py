"""
Dense FP16 / BF16 baseline context adapter for CRBench.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget
from crbench.core.registry import Registry


@Registry.register_adapter("dense")
@Registry.register_adapter("dense_fp16")
@Registry.register_adapter("dense_bf16")
class DenseAdapter(BaseContextAdapter):
    """
    Standard uncompressed Dense baseline (FP16 / BF16 / FP32).
    Represents the full-capability theoretical ceiling for the model.
    """

    def __init__(self, name: str = "dense_fp16", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.dtype = self.config.get("dtype", "float16")
        self.bits_per_element = 16.0 if self.dtype in ("float16", "bfloat16") else 32.0

    @property
    def method_type(self) -> str:
        return "dense"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)

    def forward_or_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **kwargs: Any
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not attached to DenseAdapter.")

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

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

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers = getattr(self.model.config, "num_hidden_layers", 32) if self.model else 32
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", getattr(self.model.config, "num_attention_heads", 32)) if self.model else 32
        hidden_size = getattr(self.model.config, "hidden_size", 4096) if self.model else 4096
        num_heads = getattr(self.model.config, "num_attention_heads", 32) if self.model else 32
        head_dim = getattr(self.model.config, "head_dim", hidden_size // num_heads) if self.model else 128

        bytes_per_elem = self.bits_per_element / 8.0
        # 2 tensors (Key, Value) * num_layers * num_kv_heads * head_dim * context_length * bytes_per_elem
        algorithmic_bytes = 2.0 * num_layers * num_kv_heads * head_dim * context_length * bytes_per_elem

        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=self.bits_per_element,
            total_tokens_stored=context_length,
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            algorithmic_bytes=algorithmic_bytes,
            metadata_overhead_bytes=0.0,
            custom_metrics={"dtype": self.dtype}
        )
