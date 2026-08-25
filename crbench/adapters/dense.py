"""
Dense FP16 / BF16 baseline context adapter for CRBench.

The dense adapter applies no transform at all: it prefills the prompt and
decodes against the full, uncompressed cache.  Every other method's quality is
normalised against this run *on the same query*, which is what makes the score
model-relative rather than a measure of the base model's raw ability.

Because it is the anchor, the dense baseline must actually be run at every
context length in the sweep.  Substituting a fixed 100% ceiling above some
length -- as an earlier revision did above 4096 tokens -- makes every normalised
quality number above that length meaningless: it silently asserts the model
answers perfectly at 128K, which is exactly the claim a long-context benchmark
is supposed to test.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import torch
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget
from crbench.core.registry import Registry


@Registry.register_adapter("dense")
@Registry.register_adapter("dense_fp16")
@Registry.register_adapter("dense_bf16")
class DenseAdapter(BaseContextAdapter):
    """Uncompressed dense baseline; the reference every other method is scored against."""

    streaming_transform = False
    oneshot_transform = False

    def __init__(self, name: str = "dense_fp16", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.dtype = self.config.get("dtype", "float16")
        self.bits_per_element = 16.0 if self.dtype in ("float16", "bfloat16") else 32.0

    @property
    def method_type(self) -> str:
        return "dense"

    def provenance(self) -> Dict[str, Any]:
        return {
            "implementation": "crbench_internal",
            "scheme": "no transform; full-precision KV cache",
            "role": "dense reference anchor",
        }

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers, num_kv_heads, head_dim = self.model_kv_geometry()
        bytes_per_elem = self.bits_per_element / 8.0
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
            custom_metrics={"dtype": self.dtype},
        )
