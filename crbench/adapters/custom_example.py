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

import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.inference import kv_tensors
from crbench.core.registry import Registry
from crbench.adapters import upstream


#: The repository's own presets, mirrored from
#: ``third_party/Differential-KV/ACTIVE_RUNTIME/native_core/config.py``.
#:
#: ``svd_energy`` is the load-bearing dial, not ``rank``. The repository is
#: explicit about this: "`rank` is only a CEILING: the compressor keeps the
#: smallest k whose singular values carry this fraction of the total energy",
#: and it records that asking for rank 32 against rank 128 moved the realised
#: median rank only 24 -> 34. An earlier version of this adapter set a rank and
#: never touched the energy target, which left every preset compressing to the
#: same realised rank.
#:
#: ``max_active_dense_tokens`` is the uncompressed recency window, and
#: ``kv_quant`` is the precision that window is held at -- both are charged in
#: ``get_kv_metadata``.
DKV_PRESETS: Dict[str, Dict[str, Any]] = {
    "low":  {"rank": 32,  "svd_energy": 0.999,   "max_residual_tokens": 40,
             "max_active_dense_tokens": 1024, "kv_quant": "q4_0"},
    "mid":  {"rank": 64,  "svd_energy": 0.9999,  "max_residual_tokens": 40,
             "max_active_dense_tokens": 2048, "kv_quant": "q8_0"},
    "high": {"rank": 128, "svd_energy": 0.99999, "max_residual_tokens": 128,
             "max_active_dense_tokens": 4096, "kv_quant": "f16"},
}

#: Effective bits per element of the quantized formats the presets name for the
#: dense recency window. q4_0 / q8_0 carry one fp16 scale per 32-value block.
_KV_QUANT_BITS: Dict[str, float] = {
    "f16": 16.0,
    "q8_0": 8.0 + 16.0 / 32.0,
    "q4_0": 4.0 + 16.0 / 32.0,
}

#: The repository's default micro-block: one anchor plus 256 active tokens.
DKV_MICRO_BLOCK_SIZE = 256


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
        preset: str = "mid",
        config: Optional[Dict[str, Any]] = None,
        **_legacy: Any,
    ):
        super().__init__(name=name, config=config)

        # The preset is the operating point. DKV's knobs interact -- energy sets
        # the realised rank, residuals cost a full-width token each, and the
        # dense window's precision is part of the deal -- so picking them
        # individually to hit a bits-per-token target produces a configuration
        # the authors never characterised. CRBench takes the preset as given and
        # reports where on the quality/size plane it lands.
        # An explicit `preset:` in the config wins; otherwise the adapter name
        # selects it ("dkv_high" -> high), and only then does the default apply.
        # Reading the default first would shadow the name, which silently ran
        # every adapter as `mid`.
        explicit = self.config.get("preset")
        if explicit and str(explicit).lower() in DKV_PRESETS:
            chosen = str(explicit).lower()
        else:
            chosen = next((c for c in DKV_PRESETS if c in name.lower()), preset.lower())
            if chosen not in DKV_PRESETS:
                chosen = "mid"
        self.preset = chosen
        p = DKV_PRESETS[self.preset]

        self.block_size = int(self.config.get("block_size", DKV_MICRO_BLOCK_SIZE))
        self.base_rank = int(self.config.get("base_rank", p["rank"]))
        self.svd_energy = float(self.config.get("svd_energy", p["svd_energy"]))
        self.residual_budget = int(self.config.get("residual_budget", p["max_residual_tokens"]))
        self.recent_window = int(self.config.get("recent_window", p["max_active_dense_tokens"]))
        self.kv_quant = str(self.config.get("kv_quant", p["kv_quant"]))
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
        record["preset"] = self.preset
        record["preset_source"] = (
            "third_party/Differential-KV/ACTIVE_RUNTIME/native_core/config.py")
        record["preset_values"] = {
            "rank_ceiling": self.base_rank,
            "svd_energy": self.svd_energy,
            "max_residual_tokens": self.residual_budget,
            "max_active_dense_tokens": self.recent_window,
            "kv_quant": self.kv_quant,
            "micro_block_size": self.block_size,
        }
        return record

    def validate_environment(self, device: torch.device) -> Tuple[bool, str]:
        try:
            upstream.load_dkv()
        except upstream.UpstreamUnavailable as exc:
            return False, str(exc)
        return True, "Supported"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        """Record the budget without overriding the preset.

        Every other adapter here is budget-driven, because quantization,
        eviction, merging and low-rank all expose one monotone dial. DKV does
        not: its rank follows a spectral-energy target rather than the rank
        ceiling, an exact residual token costs as much as an uncompressed one,
        and the dense window has its own precision. Solving those jointly for a
        bits-per-token target invents a configuration the authors never
        characterised, so the preset stands and CRBench reports the b_eff it
        actually produces.
        """
        super().apply_budget(budget, context_length)

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

        # compress_lowrank reads its energy target from DKV_SVD_ENERGY on every
        # call (deliberately, so tests can toggle it per process), which is the
        # only way to hand it the preset's value.
        prev_energy = os.environ.get("DKV_SVD_ENERGY")
        os.environ["DKV_SVD_ENERGY"] = repr(self.svd_energy)

        # Read the joint K|V width off the tensors rather than from the config.
        # The observed geometry is not available on a query's first transform --
        # it is measured after prefill -- and the config fallback is meaningless
        # on a heterogeneous model: Gemma 4's growing layers are 2 heads x 512,
        # while its global attributes describe something else entirely, which
        # produced "shape '[6026, 640]' is invalid for input of size 6170624".
        pairs_preview = kv_tensors(cache, valid_length=valid_length)
        if not pairs_preview:
            return cache, {"applied": False, "reason": "no growing KV layers"}
        k0 = pairs_preview[0][0]
        num_layers = len(pairs_preview)
        num_kv_heads, head_dim = int(k0.shape[1]), int(k0.shape[-1])
        feat_dim = 2 * num_kv_heads * head_dim
        half = feat_dim // 2

        window = min(self.recent_window, max(0, valid_length - 1))
        compressible = valid_length - window
        window_bits = _KV_QUANT_BITS.get(self.kv_quant, 16.0)

        total_bytes = 0.0
        anchor_bytes = 0.0
        factor_bytes = 0.0
        residual_bytes = 0.0
        n_blocks = 0
        n_residuals = 0
        rank_sum = 0

        # Written in place, one layer at a time: the tensors from kv_tensors are
        # views onto the live cache, so the compressed state replaces the
        # original without a second full cache ever existing. At 131072 tokens a
        # cloned cache would be another 4.5 GiB and would not fit beside it.
        for layer_idx, (k, v) in enumerate(kv_tensors(cache, valid_length=valid_length)):
            rank = _layer_rank(self.base_rank, layer_idx, num_layers)

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
                k[0, :, :compressible, :] = (
                    k_flat.reshape(compressible, num_kv_heads, head_dim).permute(1, 0, 2).to(k.dtype)
                )
                v[0, :, :compressible, :] = (
                    v_flat.reshape(compressible, num_kv_heads, head_dim).permute(1, 0, 2).to(v.dtype)
                )
                del feats, k_flat, v_flat

            # The dense recency window, at the precision the preset names for
            # it (`kv_quant`): `mid` holds it at q8_0, `high` at f16. Charging it
            # at f16 for every preset would overstate `mid` by half the window.
            total_bytes += window * feat_dim * window_bits / 8.0

        total_bytes += anchor_bytes + factor_bytes + residual_bytes
        if prev_energy is None:
            os.environ.pop("DKV_SVD_ENERGY", None)
        else:
            os.environ["DKV_SVD_ENERGY"] = prev_energy

        self._measured = {
            "total_bytes": total_bytes,
            "anchor_bytes": anchor_bytes,
            "factor_bytes": factor_bytes,
            "residual_bytes": residual_bytes,
            "recency_bytes": window * feat_dim * window_bits / 8.0 * num_layers,
            "context_length": valid_length,
            "blocks": n_blocks,
            "mean_dynamic_rank": rank_sum / max(1, n_blocks),
            "residual_tokens": n_residuals,
        }

        return cache, {
            "applied": n_blocks > 0,
            "preset": self.preset,
            "svd_energy": self.svd_energy,
            "residual_budget": self.residual_budget,
            "dense_window_tokens": window,
            "dense_window_bits": window_bits,
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
        # The transform records the *tokenised* prompt length, which is a little
        # under the nominal grid value (the task trims the context to leave room
        # for the question). Requiring exact equality here meant the measured
        # size was silently discarded on every single query and the estimate
        # below was reported instead -- an estimate that omitted residual tokens,
        # the largest component. Accept anything within one block.
        if m and m.get("total_bytes", 0) > 0 and \
                abs(int(m.get("context_length", -1)) - context_length) <= self.block_size:
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
            # Pre-run estimate, used only for planning; replaced once a query
            # runs. It must include the exact residual tokens: at a residual
            # budget of 128 against a 256-token block they were 7.95 of the
            # 11.63 bits/element actually stored, so leaving them out does not
            # produce a rough estimate, it produces a wrong one.
            window = min(self.recent_window, max(0, context_length - 1))
            compressible = max(0, context_length - window)
            blocks = max(1, compressible // max(1, self.block_size))
            residuals = blocks * min(self.residual_budget, self.block_size - 1)
            per_layer = (
                blocks * feat_dim * 2                                   # anchors
                + max(0, compressible - blocks) * self.base_rank * 2    # U rows
                + blocks * self.base_rank * feat_dim * 2                # V factors
                + residuals * (feat_dim * 2 + 2)                        # exact residual tokens
                + window * feat_dim * _KV_QUANT_BITS.get(self.kv_quant, 16.0) / 8.0
            )
            algorithmic_bytes = float(per_layer * num_layers)
            metadata_bytes = 0.0
            custom = {"source": "estimated", "base_rank": self.base_rank,
                      "residual_budget": self.residual_budget}

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
                "preset": self.preset,
                "svd_energy": self.svd_energy,
                "block_size": self.block_size,
                "residual_budget": self.residual_budget,
                "recent_window": self.recent_window,
                "dense_window_quant": self.kv_quant,
            },
        )
