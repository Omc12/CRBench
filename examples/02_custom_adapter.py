"""
Example 02: Implementing and Benchmarking a Custom Context Representation in CRBench.

This file demonstrates how an external researcher can implement a new KV cache
compression / representation method in under 50 lines and benchmark it using CRBench.
"""

from typing import Any, Dict, Optional
import torch
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget
from crbench.core.registry import Registry


@Registry.register_adapter("my_custom_kv_method")
class MyCustomKVMethod(BaseContextAdapter):
    """
    Interface skeleton for custom KV representation and compression methods.
    """

    def __init__(self, name: str = "my_custom_kv_method", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.compression_ratio: float = 0.5  # e.g., 2x memory reduction

    @property
    def method_type(self) -> str:
        return "custom"  # e.g., 'quantized', 'eviction', 'merging', 'custom'

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        """Apply resource budget target (e.g. bits-per-token or retention ratio)."""
        super().apply_budget(budget, context_length)
        if budget.is_bits_per_token:
            self.compression_ratio = min(1.0, budget.value / 16.0)
        else:
            self.compression_ratio = min(1.0, budget.value)

    def forward_or_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **kwargs: Any
    ) -> torch.Tensor:
        """Execute autoregressive generation under custom KV cache state."""
        if self.model is None:
            raise RuntimeError("Model reference not loaded into adapter.")

        # Attach custom KV kernel, attention mechanism, or forward hooks, then generate:
        with torch.no_grad():
            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None,
                eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                **kwargs
            )

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        """Report theoretical KV tensor storage bytes and metadata overheads."""
        # Use actual model configuration dimensions:
        num_layers = getattr(self.model.config, "num_hidden_layers", 24) if self.model else 24
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", 2) if self.model else 2
        head_dim = getattr(self.model.config, "head_dim", 64) if self.model else 64

        total_elements = 2 * num_layers * num_kv_heads * head_dim * context_length
        algorithmic_bytes = total_elements * 2.0 * self.compression_ratio
        metadata_overhead_bytes = (total_elements / 64.0) * 2.0  # Scales, codebooks, or index overhead
        effective_bpe = (algorithmic_bytes + metadata_overhead_bytes) * 8.0 / max(1, total_elements)

        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=effective_bpe,
            total_tokens_stored=int(context_length * self.compression_ratio),
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            algorithmic_bytes=algorithmic_bytes,
            metadata_overhead_bytes=metadata_overhead_bytes
        )


def main():
    print("[*] Testing custom adapter interface skeleton...")
    adapter_cls = Registry.get_adapter("my_custom_kv_method")
    print(f"[OK] Retrieved registered adapter class: {adapter_cls.__name__}")

    inst = adapter_cls(name="my_custom_kv_method")
    inst.apply_budget(ContextBudget.from_bits_per_token(4.0), context_length=4096)
    meta = inst.get_kv_metadata(4096)
    print(f"[OK] Algorithmic Bytes: {meta.algorithmic_bytes:,.0f} B | Metadata: {meta.metadata_overhead_bytes:,.0f} B | Eff BPT: {meta.effective_bits_per_element:.2f} bpt")


if __name__ == "__main__":
    main()
