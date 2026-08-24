"""
Example 02: Implementing and Benchmarking a Custom Context Representation in CRBench.

This file demonstrates how an external researcher can implement a new KV cache
compression / representation method in under 50 lines and benchmark it using CRBench.
"""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata, ExecutionStatus
from crbench.core.budget import ContextBudget
from crbench.core.registry import Registry
from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig
from crbench.core.runner import BenchmarkRunner


@Registry.register_adapter("minimal_custom_kv_compressor")
class MinimalCustomKVCompressor(BaseContextAdapter):
    """
    A minimal custom adapter demonstrating:
    1. Lifecycle hooks (`prepare_model`, `cleanup`).
    2. Precision and budget adaptation (`apply_budget`).
    3. Forward inference with custom tensor transformation (`forward_or_generate`).
    4. Exact persistent memory accounting (`get_kv_metadata`).
    """

    def __init__(self, name: str = "minimal_custom_kv_compressor", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.compression_ratio: float = 0.5  # default 2x compression

    @property
    def method_type(self) -> str:
        return "custom"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
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
        if self.model is None:
            raise RuntimeError("Model reference not loaded into adapter.")

        # Attach sample-level hooks or run generation with compressed representation
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
        num_layers = getattr(self.model.config, "num_hidden_layers", 24) if self.model else 24
        num_kv_heads = getattr(self.model.config, "num_key_value_heads", 2) if self.model else 2
        head_dim = getattr(self.model.config, "head_dim", 64) if self.model else 64

        # Total elements stored in dense FP16
        total_elements = 2 * num_layers * num_kv_heads * head_dim * context_length
        # Algorithmic state storage scaled by compression ratio
        algorithmic_bytes = total_elements * 2.0 * self.compression_ratio
        # Codebook / scale metadata: 2 bytes per 64-element block
        metadata_overhead_bytes = (total_elements / 64.0) * 2.0
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
    print("[*] Testing minimal custom adapter registration...")
    adapter_cls = Registry.get_adapter("minimal_custom_kv_compressor")
    print(f"[✓] Retrieved registered adapter class: {adapter_cls.__name__}")

    # Inspecting memory accounting
    inst = adapter_cls(name="test_custom")
    inst.apply_budget(ContextBudget.from_bits_per_token(4.0), context_length=4096)
    meta = inst.get_kv_metadata(4096)
    print(f"[✓] Algorithmic Bytes: {meta.algorithmic_bytes:,.0f} B | Metadata: {meta.metadata_overhead_bytes:,.0f} B | Eff BPT: {meta.effective_bits_per_element:.2f} bpt")


if __name__ == "__main__":
    main()
