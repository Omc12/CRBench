"""
Low-rank subspace KV adapter for CRBench.

Each attention head's key and value matrices are projected onto a rank-``r``
subspace: for a head's ``K`` of shape ``(L, D)`` the representation stores
``U (L x r)``, the singular values ``S (r)`` and ``V (r x D)`` instead of the
full matrix, so the cost per token falls from ``D`` to ``r`` elements plus a
per-head basis amortised over the whole sequence.  This is the family of
Eigen-Attention / LoRC-style KV compressors.

The transform runs once on the resident prompt cache and writes back the rank-r
reconstruction, so decoding attends to exactly what the representation can
express.  ``torch.svd_lowrank`` (randomized SVD with two power iterations) is
used rather than a full SVD: on a 65536 x 128 matrix a full decomposition is
pure waste when only the leading 32 directions are kept, and the randomized
estimate of those directions is accurate to well under the reconstruction error
the method already accepts.

The previous implementation projected only the region between a 16-token prefix
and a 32-token suffix and used a full ``torch.linalg.svd`` on the *batch*
dimension of the ``k_proj`` output, i.e. across all heads jointly rather than
per head -- a different, and much stronger, transform than the method it was
named after.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.inference import kv_tensors
from crbench.core.registry import Registry


def low_rank_reconstruct(x: torch.Tensor, rank: int, niter: int = 2) -> torch.Tensor:
    """Rank-``r`` reconstruction of a batch of matrices.

    Args:
        x: (H, L, D) -- one matrix per head.
        rank: target rank, clamped to ``min(L, D)``.
    """
    h, seq_len, dim = x.shape
    r = max(1, min(rank, seq_len, dim))
    if r >= min(seq_len, dim):
        return x

    xf = x.float()
    # Centre before projecting: the leading direction of an uncentred KV matrix
    # is dominated by the per-channel mean, which costs a whole rank slot to
    # represent something a D-element vector stores exactly.
    mean = xf.mean(dim=1, keepdim=True)
    u, s, v = torch.svd_lowrank(xf - mean, q=min(r + 4, min(seq_len, dim)), niter=niter)
    recon = (u[..., :r] * s[..., :r].unsqueeze(-2)) @ v[..., :r].transpose(-1, -2)
    return (recon + mean).to(x.dtype)


@Registry.register_adapter("compressed")
@Registry.register_adapter("low_rank_kv")
@Registry.register_adapter("linear_kv")
class LowRankCompressedKVAdapter(BaseContextAdapter):
    """Per-head truncated-SVD subspace compression of the prompt KV cache."""

    oneshot_transform = True

    def __init__(
        self,
        name: str = "low_rank_kv",
        rank_ratio: float = 0.25,
        recent_window: int = 32,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.rank_ratio = float(self.config.get("rank_ratio", rank_ratio))
        # A short exact tail: the most recent tokens are what the query attends
        # to most sharply, and every method in this suite keeps such a window.
        self.recent_window = int(self.config.get("recent_window", recent_window))

    @property
    def method_type(self) -> str:
        return "compressed"

    def provenance(self) -> Dict[str, Any]:
        return {
            "implementation": "crbench_internal",
            "scheme": "per-head mean-centred truncated SVD (randomized, niter=2) of the prompt K and V",
            "applied_to": "resident prompt cache, one-shot after prefill",
            "simulated": True,
            "note": "Reconstruction is written back in bf16; the memory figure is the U/S/V factor size.",
        }

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        if budget.budget_type == BudgetType.COMPRESSION_RATIO:
            self.rank_ratio = float(budget.value)
        elif budget.budget_type == BudgetType.BITS_PER_TOKEN:
            self.rank_ratio = float(budget.value) / 16.0

    def _rank(self) -> int:
        _, _, head_dim = self.model_kv_geometry()
        return max(1, int(round(head_dim * self.rank_ratio)))

    def transform_cache(
        self,
        cache: Any,
        input_ids: torch.Tensor,
        valid_length: int,
    ) -> Tuple[Any, Dict[str, Any]]:
        rank = self._rank()
        _, _, head_dim = self.model_kv_geometry()
        if rank >= head_dim:
            return cache, {"rank": rank, "applied": False}

        window = min(self.recent_window, max(0, valid_length - 1))
        split = valid_length - window

        # Written in place, one layer at a time. The tensors from kv_tensors are
        # views onto the live cache, so this rewrites the cache without ever
        # holding a second copy of it -- at 131072 tokens a cloned cache would
        # be another 4.5 GiB, which does not fit beside the original.
        for k, v in kv_tensors(cache, valid_length=valid_length):
            if split > rank:
                k[0, :, :split, :] = low_rank_reconstruct(k[0, :, :split, :], rank)
                v[0, :, :split, :] = low_rank_reconstruct(v[0, :, :split, :], rank)

        return cache, {"rank": rank, "head_dim": head_dim,
                       "recent_window_exact": window, "applied": True}

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers, num_kv_heads, head_dim = self.model_kv_geometry()
        rank = self._rank()
        window = min(self.recent_window, max(0, context_length - 1))
        projected = max(0, context_length - window)

        per_head_tensor = (
            projected * rank          # U: one r-vector per projected token
            + rank                    # S: singular values
            + rank * head_dim         # V: the basis, amortised over the sequence
            + head_dim                # the per-channel mean that was factored out
            + window * head_dim       # the exact recent window
        )
        # 2 tensors (K and V) x layers x kv heads, fp16
        algorithmic_bytes = 2.0 * num_layers * num_kv_heads * per_head_tensor * 2.0

        dense_elems = self.dense_element_count(context_length)
        effective_bpe = algorithmic_bytes * 8.0 / max(1, dense_elems)

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
            custom_metrics={
                "rank": rank,
                "rank_ratio": rank / max(1, head_dim),
                "recent_window_exact": window,
            },
        )
