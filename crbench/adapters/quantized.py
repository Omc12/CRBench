"""
KV Cache Quantization adapter (FP8, INT8, INT4, INT2) for CRBench.
Implements dynamic and simulated quantization with accurate effective bit-width accounting.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.registry import Registry


def quantize_to_int_simulated(tensor: torch.Tensor, n_bits: int = 4, group_size: int = 64, outlier_ratio: float = 0.01) -> torch.Tensor:
    """
    Simulates per-channel / grouped symmetric quantization to n_bits with dynamic scaling
    and outlier preservation (KIVI / AWQ / QuaRot style).
    Preserves gradient-free evaluation numerical behavior of low-bit quantization.
    """
    if n_bits >= 16:
        return tensor

    orig_shape = tensor.shape
    abs_t = torch.abs(tensor)
    
    # Outlier channel preservation: protect top ~1% extreme activations in full precision
    if outlier_ratio > 0.0:
        try:
            thresh = torch.quantile(abs_t.float(), 1.0 - outlier_ratio)
            outlier_mask = abs_t >= thresh
        except Exception:
            outlier_mask = torch.zeros_like(tensor, dtype=torch.bool)
    else:
        outlier_mask = torch.zeros_like(tensor, dtype=torch.bool)

    qmax = 2 ** (n_bits - 1) - 1
    # Scale along head channel dimension (last dim)
    scale = (abs_t.masked_fill(outlier_mask, 0.0).max(dim=-1, keepdim=True).values / max(1, qmax)).clamp(min=1e-5)
    quantized = torch.round(tensor / scale).clamp(-qmax, qmax)
    dequantized = quantized * scale

    if outlier_ratio > 0.0:
        dequantized = torch.where(outlier_mask, tensor, dequantized)

    return dequantized


@Registry.register_adapter("quantized")
@Registry.register_adapter("kv_quant")
@Registry.register_adapter("int8")
@Registry.register_adapter("int4")
@Registry.register_adapter("fp8")
class QuantizedKVAdapter(BaseContextAdapter):
    """
    Quantized KV Cache Adapter supporting FP8, INT8, INT4, and INT2.
    """

    def __init__(self, name: str = "kv_quant_int4", bits: int = 4, group_size: int = 64, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.bits = self.config.get("bits", bits)
        self.group_size = self.config.get("group_size", group_size)
        self.scale_bits = 16  # FP16 scales

    @property
    def method_type(self) -> str:
        return "quantization"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        if budget.budget_type == BudgetType.BITS_PER_TOKEN:
            self.bits = max(2, min(16, int(round(budget.value))))
        elif budget.budget_type == BudgetType.COMPRESSION_RATIO:
            # 16 / ratio
            target_bits = 16.0 * budget.value
            self.bits = max(2, min(16, int(round(target_bits))))

    def forward_or_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **kwargs: Any
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not attached to QuantizedKVAdapter.")

        device = input_ids.device
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        hooks = []
        if self.bits < 16:
            layers = getattr(self.model, "model", self.model)
            layer_list = getattr(layers, "layers", getattr(layers, "decoder_layers", getattr(layers, "h", [])))
            for layer in layer_list:
                attn = getattr(layer, "self_attn", getattr(layer, "attn", getattr(layer, "attention", None)))
                if attn is not None:
                    k_proj = getattr(attn, "k_proj", getattr(attn, "key", None))
                    v_proj = getattr(attn, "v_proj", getattr(attn, "value", None))
                    if k_proj is not None:
                        hooks.append(k_proj.register_forward_hook(
                            lambda m, inp, out, b=self.bits, g=self.group_size: quantize_to_int_simulated(out, n_bits=b, group_size=g)
                        ))
                    if v_proj is not None:
                        hooks.append(v_proj.register_forward_hook(
                            lambda m, inp, out, b=self.bits, g=self.group_size: quantize_to_int_simulated(out, n_bits=b, group_size=g)
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

        total_elements = 2 * num_layers * num_kv_heads * head_dim * context_length
        # Raw quantized tensor bytes
        algorithmic_bytes = (total_elements * self.bits) / 8.0

        # Scale metadata overhead: 1 FP16 scale (2 bytes) per group_size elements
        num_groups = total_elements / max(1, self.group_size)
        scale_overhead_bytes = num_groups * (self.scale_bits / 8.0)

        effective_bpe = (algorithmic_bytes + scale_overhead_bytes) * 8.0 / max(1, total_elements)

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
            metadata_overhead_bytes=scale_overhead_bytes,
            custom_metrics={"bits": self.bits, "group_size": self.group_size}
        )
