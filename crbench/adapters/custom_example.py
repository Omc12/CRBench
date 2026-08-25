"""
Differential-KV (DKV) adapter for CRBench.

DKV is driven from its author's vendored repository
(``third_party/Differential-KV``); the block compressor called here is
``native_core.compression.lowrank.compress_lowrank``, the same function the
runtime's own compression paths use.

The representation, as the repository documents it, splits the cache into fixed
micro-blocks and stores four things per block:

1. an **anchor token**, the block's first, kept exact;
2. a **joint K|V low-rank SVD delta** of the remaining tokens relative to that
   anchor -- one decomposition over the concatenated key and value features, at
   a layer-adaptive rank;
3. a budget of **exact residual tokens**, the rows whose low-rank
   reconstruction error is largest, stored verbatim;
4. a **dense recency window** at the end of the sequence, uncompressed.

Reconstruction follows the repository's own residual semantics.  With
``DKV_RESIDUAL_EXACT_KEYS`` on -- the default on every device -- a residual
holds the *anchor-relative exact* value and substitutes for the low-rank row;
with it off it holds a correction and is added.  ``_exact_keys_enabled`` is
queried rather than assumed, so this adapter follows whichever the environment
selects.

Memory is not estimated for this method: each block reports the true size of the
factors it produced, so ``get_kv_metadata`` returns bytes that were counted, not
modelled.  The dynamic-rank selection inside ``compress_lowrank`` means a block's
rank is data-dependent, and an analytical formula would miss that entirely.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.inference import kv_tensors, rebuild_cache
from crbench.core.registry import Registry
from crbench.adapters import upstream


#: Layer-adaptive rank multipliers, from the repository's README: early layers
#: 0.75r, mid layers 1.5r, late layers 0.5r.
def _layer_rank(base_rank: int, layer_idx: int, num_layers: int) -> int:
    frac = layer_idx / max(1, num_layers - 1)
    if frac < 1.0 / 3.0:
        mult = 0.75
    elif frac < 2.0 / 3.0:
        mult = 1.5
    else:
        mult = 0.5
    return max(4, int(round(base_rank * mult)))


@Registry.register_adapter("dkv")
@Registry.register_adapter("custom_dkv")
@Registry.register_adapter("custom")
@Registry.register_adapter("dkv_high")
@Registry.register_adapter("dkv_mid")
class DKVContextAdapter(BaseContextAdapter):
    """Anchor + joint low-rank delta + exact residuals, from the DKV repository."""

    oneshot_transform = True

    def __init__(
        self,
        name: str = "dkv",
        block_size: int = 256,
        base_rank: int = 32,
        residual_budget: int = 128,
        recent_window: int = 256,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.block_size = int(self.config.get("block_size", block_size))
        self.base_rank = int(self.config.get("base_rank", base_rank))
        self.residual_budget = int(self.config.get("residual_budget", residual_budget))
        self.recent_window = int(self.config.get("recent_window", recent_window))
        # Filled in by transform_cache; get_kv_metadata reports what was counted.
        self._measured: Dict[str, float] = {}

    @property
    def method_type(self) -> str:
        return "custom"

    def provenance(self) -> Dict[str, Any]:
        record = dict(upstream.describe_provenance("dkv"))
        record["implementation"] = "upstream_reference"
        record["entry_point"] = "native_core.compression.lowrank.compress_lowrank"
        record["adaptation"] = (
            "The repository's serving runtime owns its own paged block pool and a Triton "
            "decode kernel that is unavailable on Windows. The block compressor is therefore "
            "driven directly over the resident prompt cache and its output reconstructed "
            "in place, which measures the representation's fidelity and size but not the "
            "runtime's sparse-decode speedup."
        )
        record["measured_memory"] = True
        return record

    def validate_environment(self, device: torch.device) -> Tuple[bool, str]:
        try:
            upstream.load_dkv()
        except upstream.UpstreamUnavailable as exc:
            return False, str(exc)
        return True, "Supported"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        _, num_kv_heads, head_dim = self.model_kv_geometry()
        feat_dim = 2 * num_kv_heads * head_dim

        if budget.budget_type == BudgetType.BITS_PER_TOKEN:
            target_ratio = float(budget.value) / 16.0
        elif budget.budget_type == BudgetType.COMPRESSION_RATIO:
            target_ratio = float(budget.value)
        else:
            return

        # Per compressed token the representation costs one r-vector (U row);
        # the block's V factor (r x feat) and anchor (feat) amortise over the
        # block. Solving ratio ~= (r + (r*feat + feat)/block) / feat for r:
        block = max(2, self.block_size)
        r = (target_ratio * feat_dim - feat_dim / block) / (1.0 + feat_dim / block)
        self.base_rank = max(4, min(feat_dim // 2, int(round(r))))

    # ------------------------------------------------------------------ #
    # The transform                                                       #
    # ------------------------------------------------------------------ #

    def transform_cache(
        self,
        cache: Any,
        input_ids: torch.Tensor,
        valid_length: int,
    ) -> Tuple[Any, Dict[str, Any]]:
        mod = upstream.load_dkv()["lowrank"]
        compress = mod.compress_lowrank
        exact_keys = bool(mod._exact_keys_enabled(None))

        num_layers, num_kv_heads, head_dim = self.model_kv_geometry()
        feat_dim = 2 * num_kv_heads * head_dim
        half = feat_dim // 2

        window = min(self.recent_window, max(0, valid_length - 1))
        compressible = valid_length - window

        total_bytes = 0.0
        anchor_bytes = 0.0
        factor_bytes = 0.0
        residual_bytes = 0.0
        n_blocks = 0
        n_residuals = 0
        rank_sum = 0

        new_pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx, (k, v) in enumerate(kv_tensors(cache, valid_length=valid_length)):
            rank = _layer_rank(self.base_rank, layer_idx, num_layers)
            k_new = k.clone()
            v_new = v.clone()

            if compressible > self.block_size:
                # (1, H, L, D) -> (L, H*D) for K and V, concatenated as [K | V]
                feats = torch.cat(
                    [
                        k[0, :, :compressible, :].permute(1, 0, 2).reshape(compressible, half),
                        v[0, :, :compressible, :].permute(1, 0, 2).reshape(compressible, half),
                    ],
                    dim=1,
                ).float()

                for start in range(0, compressible, self.block_size):
                    end = min(start + self.block_size, compressible)
                    if end - start < 2:
                        continue
                    anchor = feats[start]
                    deltas = feats[start + 1:end] - anchor
                    lr = compress(deltas, rank, max_residual=self.residual_budget)

                    recon = (lr.U.float() @ lr.V.float() * lr.scale).to(deltas.device)
                    self._apply_residuals(lr, recon, half, exact_keys)
                    feats[start + 1:end] = anchor + recon

                    n_blocks += 1
                    rank_sum += int(lr.U.shape[1])
                    anchor_bytes += feat_dim * 2
                    factor_bytes += lr.U.numel() * 2 + lr.V.numel() * 2
                    if lr.residual_K_positions is not None and lr.residual_K_positions.numel():
                        npos = int(lr.residual_K_positions.numel())
                        n_residuals += npos
                        # values (fp16, both halves) + positions (int16)
                        residual_bytes += npos * feat_dim * 2 + npos * 2

                    del lr, recon, deltas

                k_flat, v_flat = feats[:, :half], feats[:, half:]
                k_new[0, :, :compressible, :] = (
                    k_flat.reshape(compressible, num_kv_heads, head_dim).permute(1, 0, 2).to(k.dtype)
                )
                v_new[0, :, :compressible, :] = (
                    v_flat.reshape(compressible, num_kv_heads, head_dim).permute(1, 0, 2).to(v.dtype)
                )
                del feats, k_flat, v_flat

            # The dense recency window is stored exactly, at fp16.
            total_bytes += window * feat_dim * 2
            new_pairs.append((k_new, v_new))

        total_bytes += anchor_bytes + factor_bytes + residual_bytes
        self._measured = {
            "total_bytes": total_bytes,
            "anchor_bytes": anchor_bytes,
            "factor_bytes": factor_bytes,
            "residual_bytes": residual_bytes,
            "recency_bytes": window * feat_dim * 2 * num_layers,
            "context_length": valid_length,
            "blocks": n_blocks,
            "mean_dynamic_rank": rank_sum / max(1, n_blocks),
            "residual_tokens": n_residuals,
        }

        new_cache = rebuild_cache(new_pairs)
        del new_pairs
        return new_cache, {
            "applied": n_blocks > 0,
            "blocks": n_blocks,
            "base_rank": self.base_rank,
            "mean_dynamic_rank": self._measured["mean_dynamic_rank"],
            "residual_tokens": n_residuals,
            "exact_keys": exact_keys,
            "measured_state_bytes": total_bytes,
            "recent_window_exact": window,
        }

    @staticmethod
    def _apply_residuals(lr: Any, recon: torch.Tensor, half: int, exact_keys: bool) -> None:
        """Fold DKV's exact residual rows back into the reconstruction.

        ``residual_{K,V}_positions`` index rows of the delta matrix; the K and V
        halves share one index set (upstream selects them jointly, so a token is
        never made exact on one half and left lossy on the other).
        """
        posk = lr.residual_K_positions
        if posk is None or posk.numel() == 0:
            return
        idx = posk.long().to(recon.device)
        if lr.residual_K_values is not None:
            vals = lr.residual_K_values.float().to(recon.device)
            if exact_keys:
                recon[idx, :half] = vals
            else:
                recon[idx, :half] += vals
        posv = lr.residual_V_positions
        if posv is not None and lr.residual_V_values is not None and posv.numel():
            idxv = posv.long().to(recon.device)
            valsv = lr.residual_V_values.float().to(recon.device)
            if exact_keys:
                recon[idxv, half:] = valsv
            else:
                recon[idxv, half:] += valsv

    # ------------------------------------------------------------------ #
    # Resource accounting                                                 #
    # ------------------------------------------------------------------ #

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers, num_kv_heads, head_dim = self.model_kv_geometry()
        feat_dim = 2 * num_kv_heads * head_dim
        dense_elems = self.dense_element_count(context_length)

        m = self._measured
        if m and m.get("context_length") == context_length and m.get("total_bytes", 0) > 0:
            algorithmic_bytes = float(m["total_bytes"] - m["residual_bytes"])
            metadata_bytes = float(m["residual_bytes"])
            custom = {
                "source": "measured",
                "blocks": m["blocks"],
                "mean_dynamic_rank": m["mean_dynamic_rank"],
                "residual_tokens": m["residual_tokens"],
                "anchor_bytes": m["anchor_bytes"],
                "factor_bytes": m["factor_bytes"],
                "recency_bytes": m["recency_bytes"],
            }
        else:
            # Pre-run estimate, used only for planning; replaced once a query runs.
            window = min(self.recent_window, max(0, context_length - 1))
            compressible = max(0, context_length - window)
            blocks = max(1, compressible // max(1, self.block_size))
            per_layer = (
                blocks * feat_dim * 2                                   # anchors
                + max(0, compressible - blocks) * self.base_rank * 2    # U rows
                + blocks * self.base_rank * feat_dim * 2                # V factors
                + window * feat_dim * 2                                 # recency window
            )
            algorithmic_bytes = float(per_layer * num_layers)
            metadata_bytes = 0.0
            custom = {"source": "estimated", "base_rank": self.base_rank}

        effective_bpe = (algorithmic_bytes + metadata_bytes) * 8.0 / max(1, dense_elems)

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
            metadata_overhead_bytes=metadata_bytes,
            custom_metrics={
                **custom,
                "block_size": self.block_size,
                "residual_budget": self.residual_budget,
                "recent_window": self.recent_window,
            },
        )
