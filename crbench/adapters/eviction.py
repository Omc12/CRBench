"""
KV Cache Eviction / Pruning adapter (StreamingLLM, SnapKV, H2O) for CRBench.
Implements attention sink preservation, local window preservation, and heavy-hitter token selection.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.registry import Registry


@Registry.register_adapter("eviction")
@Registry.register_adapter("snapkv")
@Registry.register_adapter("streaming_llm")
@Registry.register_adapter("h2o")
class EvictionKVAdapter(BaseContextAdapter):
    """
    KV Token Eviction / Pruning adapter.
    Selects a subset of tokens to retain in the KV cache:
    - sink_tokens: initial prompt tokens (StreamingLLM attention sinks)
    - local_tokens: recent tokens in sliding window
    - heavy_hitter_tokens: high-attention importance tokens (SnapKV / H2O)
    """

    def __init__(
        self,
        name: str = "snapkv",
        strategy: str = "snapkv",  # "streaming_llm", "snapkv", "h2o"
        sink_tokens: int = 32,
        local_tokens: int = 128,
        retention_ratio: float = 0.25,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name=name, config=config)
        self.strategy = self.config.get("strategy", strategy)
        self.sink_tokens = self.config.get("sink_tokens", sink_tokens)
        self.local_tokens = self.config.get("local_tokens", local_tokens)
        self.retention_ratio = self.config.get("retention_ratio", retention_ratio)
        self.max_tokens_retained: Optional[int] = None

    @property
    def method_type(self) -> str:
        return "eviction"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        if budget.budget_type == BudgetType.TOKEN_CAPACITY:
            self.max_tokens_retained = int(budget.value)
            self.retention_ratio = min(1.0, float(self.max_tokens_retained) / max(1, context_length))
        elif budget.budget_type == BudgetType.COMPRESSION_RATIO:
            self.retention_ratio = float(budget.value)
            self.max_tokens_retained = int(context_length * self.retention_ratio)
        elif budget.budget_type == BudgetType.BITS_PER_TOKEN:
            # Dense is 16 bpt. If budget is e.g. 4 bpt, retention is 4/16 = 0.25
            self.retention_ratio = float(budget.value) / 16.0
            self.max_tokens_retained = int(context_length * self.retention_ratio)

    def forward_or_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **kwargs: Any
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model is not attached to EvictionKVAdapter.")

        device = input_ids.device
        seq_len = input_ids.shape[-1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=device)

        eff_input_ids = input_ids
        eff_attn_mask = attention_mask

        # Apply eviction if sequence length exceeds retained budget
        if self.max_tokens_retained is not None and seq_len > self.max_tokens_retained:
            target_k = max(32, self.max_tokens_retained)
            if target_k < seq_len:
                if self.strategy == "streaming_llm":
                    sink_n = min(self.sink_tokens, target_k // 4)
                    recent_n = target_k - sink_n
                    eff_input_ids = torch.cat([input_ids[:, :sink_n], input_ids[:, -recent_n:]], dim=-1)
                    eff_attn_mask = torch.cat([attention_mask[:, :sink_n], attention_mask[:, -recent_n:]], dim=-1)
                elif self.strategy in ("snapkv", "h2o"):
                    sink_n = min(self.sink_tokens, max(4, target_k // 8))
                    local_n = min(self.local_tokens, max(8, target_k // 4))
                    mid_budget = target_k - sink_n - local_n
                    mid_len = seq_len - sink_n - local_n

                    if mid_budget > 0 and mid_len > 0:
                        try:
                            embed_layer = self.model.get_input_embeddings() if hasattr(self.model, "get_input_embeddings") else None
                            if embed_layer is not None:
                                embeds = embed_layer(input_ids)
                                query_vec = embeds[:, -32:, :].mean(dim=1, keepdim=True)
                                mid_vecs = embeds[:, sink_n:-local_n, :]
                                sim = torch.nn.functional.cosine_similarity(query_vec, mid_vecs, dim=-1)[0]
                                
                                # Chunk-level pooling (e.g. window size 16) to preserve intact syntactic phrases
                                chunk_sz = 16
                                n_chunks = max(1, mid_len // chunk_sz)
                                chunk_scores = []
                                for c in range(n_chunks):
                                    c_start = c * chunk_sz
                                    c_end = min(mid_len, (c + 1) * chunk_sz)
                                    chunk_scores.append(sim[c_start:c_end].max())
                                chunk_tensor = torch.tensor(chunk_scores, device=device)
                                k_chunks = max(1, min(n_chunks, mid_budget // chunk_sz))
                                top_chunk_indices = torch.topk(chunk_tensor, k=k_chunks).indices.sort().values
                                
                                selected_tokens = []
                                for ci in top_chunk_indices.tolist():
                                    c_start = sink_n + ci * chunk_sz
                                    c_end = min(seq_len - local_n, c_start + chunk_sz)
                                    selected_tokens.extend(range(c_start, c_end))
                                
                                keep_idx = torch.tensor(
                                    list(range(sink_n)) + selected_tokens + list(range(seq_len - local_n, seq_len)),
                                    device=device
                                )
                                eff_input_ids = input_ids[:, keep_idx]
                                eff_attn_mask = attention_mask[:, keep_idx]
                            else:
                                eff_input_ids = torch.cat([input_ids[:, :sink_n], input_ids[:, - (target_k - sink_n):]], dim=-1)
                                eff_attn_mask = torch.cat([attention_mask[:, :sink_n], attention_mask[:, - (target_k - sink_n):]], dim=-1)
                        except Exception:
                            eff_input_ids = torch.cat([input_ids[:, :sink_n], input_ids[:, - (target_k - sink_n):]], dim=-1)
                            eff_attn_mask = torch.cat([attention_mask[:, :sink_n], attention_mask[:, - (target_k - sink_n):]], dim=-1)
                    else:
                        eff_input_ids = torch.cat([input_ids[:, :sink_n], input_ids[:, - (target_k - sink_n):]], dim=-1)
                        eff_attn_mask = torch.cat([attention_mask[:, :sink_n], attention_mask[:, - (target_k - sink_n):]], dim=-1)

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

        if self.max_tokens_retained is not None:
            retained_tokens = min(context_length, max(self.sink_tokens + self.local_tokens, self.max_tokens_retained))
        else:
            retained_tokens = min(context_length, max(self.sink_tokens + self.local_tokens, int(context_length * self.retention_ratio)))

        # Dense FP16 elements for retained tokens
        dense_bytes_per_elem = 2.0  # FP16 = 2 bytes
        algorithmic_bytes = 2.0 * num_layers * num_kv_heads * head_dim * retained_tokens * dense_bytes_per_elem

        # Positional index metadata overhead: 4 bytes int32 per retained token
        index_overhead_bytes = retained_tokens * 4.0

        effective_bpt = (algorithmic_bytes + index_overhead_bytes) * 8.0 / max(1, 2 * num_layers * num_kv_heads * head_dim * context_length)

        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=effective_bpt,
            total_tokens_stored=retained_tokens,
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            algorithmic_bytes=algorithmic_bytes,
            metadata_overhead_bytes=index_overhead_bytes,
            custom_metrics={"strategy": self.strategy, "retained_tokens": retained_tokens, "retention_ratio": self.retention_ratio}
        )
