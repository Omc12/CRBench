"""
KV cache eviction adapters (SnapKV, StreamingLLM) for CRBench.

Both methods are driven from their authors' vendored repositories rather than
reimplemented:

* **SnapKV** -- ``SnapKVCluster.update_kv`` from ``third_party/SnapKV``.  The
  algorithm scores every prompt key by the attention it receives from the last
  ``window_size`` prompt queries, smooths those scores with ``avg_pool1d`` over
  ``kernel_size`` neighbours, keeps the top ``max_capacity_prompt -
  window_size`` of them, and appends the untouched recent window.
* **StreamingLLM** -- ``StartRecentKVCache`` from ``third_party/streaming-llm``.
  Keep the first ``start_size`` attention-sink tokens and the most recent
  ``recent_size``; drop the middle.

Both are *prompt* compressors: they run once, on the resident prompt cache, and
what survives is what the generated tokens attend to.  The previous
implementation instead deleted input tokens before the forward pass, which is a
different intervention -- it changes the positions the model sees, changes
prefill cost, and gives the method no access to the attention signal SnapKV is
defined in terms of.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget, BudgetType
from crbench.core.inference import (kv_tensors, rebuild_cache, cache_layers,
                                    growing_layer_indices)
from crbench.core.registry import Registry
from crbench.adapters import upstream


def decoder_layers(model: nn.Module) -> List[nn.Module]:
    """The decoder layer list, wherever this architecture keeps it.

    ``model.model.layers`` covers plain causal LMs, but multimodal wrappers nest
    the text stack further down -- Gemma 4 loads as
    ``Gemma4ForConditionalGeneration`` -- and returning an empty list there made
    SnapKV index into nothing and raise "list index out of range".
    """
    for path in (("model", "layers"),
                 ("model", "language_model", "layers"),
                 ("language_model", "model", "layers"),
                 ("model", "text_model", "layers"),
                 ("layers",)):
        node: Any = model
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node is not None and len(node) > 0:
            return list(node)
    return []


def _grouped_attention_scores(
    query_states: torch.Tensor,   # (B, H_q, w, D) post-RoPE observation window
    key_states: torch.Tensor,     # (B, H_kv, L, D) post-RoPE prompt keys
    window_size: int,
    num_key_value_groups: int,
) -> torch.Tensor:
    """SnapKV's ``attn_weights_sum``, pooled over each GQA group's query heads.

    This reproduces upstream ``SnapKVCluster.update_kv``'s scoring exactly --
    scaled dot product, causal mask over the observation window, softmax in
    float32, sum across the window's query axis -- and then sums the result over
    the ``num_key_value_groups`` query heads that share one KV head.

    The pooling is required because upstream stores its compressed cache with KV
    repeated to the full query-head count (``repeat_kv`` is moved ahead of the
    cache write in every one of their hijacks).  On a model like Qwen2.5-7B,
    with 28 query heads over 4 KV heads, that would multiply the retained cache
    by seven and measure an implementation artifact rather than the algorithm.
    Selecting per KV head keeps the comparison on the representation the model
    actually needs.  With ``num_key_value_groups == 1`` this is identical to
    upstream, which ``tests/test_upstream_fidelity.py`` asserts numerically.

    Returns: (B, H_kv, L - window_size) score tensor.
    """
    bsz, num_heads, w, head_dim = query_states.shape
    num_kv_heads = key_states.shape[1]

    # Score the observation window against every prompt key, one KV group at a
    # time: the full (B, H_q, w, L) tensor is 235 MB at L=64K on a 28-head model
    # and there is no need to hold all of it at once.
    scores = torch.zeros(bsz, num_kv_heads, key_states.shape[-2],
                         device=key_states.device, dtype=torch.float32)

    # Upstream's causal mask over the observation window's own positions.
    mask = torch.full((w, w), torch.finfo(query_states.dtype).min, device=query_states.device)
    mask_cond = torch.arange(mask.size(-1), device=query_states.device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask[None, None, :, :]

    for g in range(num_kv_heads):
        q_g = query_states[:, g * num_key_value_groups:(g + 1) * num_key_value_groups]  # (B, rep, w, D)
        k_g = key_states[:, g:g + 1]                                                    # (B, 1, L, D)
        attn = torch.matmul(q_g, k_g.transpose(2, 3)) / math.sqrt(head_dim)             # (B, rep, w, L)
        attn[:, :, -w:, -w:] = attn[:, :, -w:, -w:] + mask
        attn = nn.functional.softmax(attn, dim=-1, dtype=torch.float32)
        scores[:, g] = attn.sum(dim=-2).sum(dim=1)   # sum over window queries, then over the group
        del attn, q_g

    return scores[..., :-window_size]


def snapkv_compress_gqa(
    key_states: torch.Tensor,     # (B, H_kv, L, D)
    value_states: torch.Tensor,   # (B, H_kv, L, D)
    query_states: torch.Tensor,   # (B, H_q,  w, D)
    *,
    window_size: int,
    max_capacity_prompt: int,
    kernel_size: int,
    pooling: str,
    num_key_value_groups: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """SnapKV prompt compression with per-KV-head selection.

    Everything after the scoring step is upstream's, unchanged: ``avg_pool1d``
    (or ``max_pool1d``) smoothing with ``kernel_size`` and ``padding =
    kernel_size // 2``, a top-k of ``max_capacity_prompt - window_size``, a
    gather over the pre-window keys and values, and the untouched recent window
    concatenated back on.

    With ``num_key_value_groups == 1`` this is numerically identical to
    ``SnapKVCluster.update_kv``; ``tests/test_upstream_fidelity.py`` asserts it.
    """
    if key_states.shape[-2] < max_capacity_prompt:
        return key_states, value_states

    scores = _grouped_attention_scores(
        query_states, key_states, window_size, num_key_value_groups
    )

    if pooling == "avgpool":
        pooled = F.avg_pool1d(scores, kernel_size=kernel_size,
                              padding=kernel_size // 2, stride=1)
    elif pooling == "maxpool":
        pooled = F.max_pool1d(scores, kernel_size=kernel_size,
                              padding=kernel_size // 2, stride=1)
    else:
        raise ValueError("Pooling method not supported")

    indices = pooled.topk(max_capacity_prompt - window_size, dim=-1).indices
    indices = indices.unsqueeze(-1).expand(-1, -1, -1, key_states.shape[-1])
    k_past = key_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    v_past = value_states[:, :, :-window_size, :].gather(dim=2, index=indices)
    return (
        torch.cat([k_past, key_states[:, :, -window_size:, :]], dim=2),
        torch.cat([v_past, value_states[:, :, -window_size:, :]], dim=2),
    )


@Registry.register_adapter("eviction")
@Registry.register_adapter("snapkv")
@Registry.register_adapter("streaming_llm")
class EvictionKVAdapter(BaseContextAdapter):
    """Prompt-cache eviction under a bits-per-token budget.

    A budget of ``b`` bits/element on a dense-fp16 (16 bit) baseline means the
    representation may keep ``b/16`` of the tokens; the retained cache is stored
    at full precision, so the saving is entirely in token count.
    """

    oneshot_transform = True

    #: Upper bound on retained captures per attention module. Only the final
    #: chunk's are used, and no architecture here runs more passes than this.
    _MAX_CAPTURES_PER_MODULE = 8

    def __init__(
        self,
        name: str = "snapkv",
        strategy: str = "snapkv",
        sink_tokens: int = 4,
        window_size: int = 32,
        kernel_size: int = 5,
        pooling: str = "avgpool",
        retention_ratio: float = 0.25,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name, config=config)
        self.strategy = self.config.get("strategy", strategy)
        # StreamingLLM's paper and repo default to 4 attention-sink tokens.
        self.sink_tokens = int(self.config.get("sink_tokens", sink_tokens))
        # SnapKV's observation window and pooling kernel, at the repo defaults.
        self.window_size = int(self.config.get("window_size", window_size))
        self.kernel_size = int(self.config.get("kernel_size", kernel_size))
        self.pooling = self.config.get("pooling", pooling)
        self.retention_ratio = float(self.config.get("retention_ratio", retention_ratio))
        self.max_tokens_retained: Optional[int] = None
        self._observation: Dict[int, List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = {}
        self._num_modules: int = 0
        self._passes: int = 1

    @property
    def method_type(self) -> str:
        return "eviction"

    def provenance(self) -> Dict[str, Any]:
        key = "streaming_llm" if self.strategy == "streaming_llm" else "snapkv"
        record = dict(upstream.describe_provenance(key))
        record["implementation"] = "upstream_reference"
        if key == "snapkv":
            record["entry_point"] = "snapkv.monkeypatch.snapkv_utils.SnapKVCluster.update_kv"
            record["adaptation"] = (
                "Upstream's monkeypatch targets transformers==4.37 and Llama/Mistral/Mixtral; "
                "this runs transformers 5.x on Qwen2, so SnapKVCluster is driven directly. "
                "Attention scores are pooled across each GQA group's query heads so selection "
                "is per KV head; upstream instead stores KV repeated to the query-head count."
            )
        else:
            record["entry_point"] = "streaming_llm.kv_cache.StartRecentKVCache"
            record["adaptation"] = "Applied to the resident prompt cache; policy is upstream's, unmodified."
        return record

    def validate_environment(self, device: torch.device) -> Tuple[bool, str]:
        try:
            if self.strategy == "streaming_llm":
                upstream.load_streaming_llm()
            else:
                upstream.load_snapkv()
        except upstream.UpstreamUnavailable as exc:
            return False, str(exc)
        return True, "Supported"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        if budget.budget_type == BudgetType.TOKEN_CAPACITY:
            self.max_tokens_retained = int(budget.value)
            self.retention_ratio = min(1.0, self.max_tokens_retained / max(1, context_length))
        elif budget.budget_type == BudgetType.COMPRESSION_RATIO:
            self.retention_ratio = float(budget.value)
            self.max_tokens_retained = int(context_length * self.retention_ratio)
        elif budget.budget_type == BudgetType.BITS_PER_TOKEN:
            # Retained tokens are stored dense (16 bits/element), so a budget of
            # b bits/element buys b/16 of the tokens.
            self.retention_ratio = float(budget.value) / 16.0
            self.max_tokens_retained = int(context_length * self.retention_ratio)

    # ------------------------------------------------------------------ #
    # SnapKV needs the observation window's post-RoPE queries              #
    # ------------------------------------------------------------------ #

    def _record(self, module_idx: int, hidden: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> None:
        """Append one observation per call of this attention module.

        A depth-recurrent model reuses each attention module once per loop, and
        every pass writes a *different* cache layer: Nanbeige4.2 drives 44 cache
        layers from 22 modules. Keeping only the latest capture would score the
        first loop's cache entries with the second loop's queries.

        Every prefill chunk also calls each module, so the list holds
        ``chunks x passes`` entries by the end. Only the final chunk's window is
        the observation window, so the reader takes the last ``passes`` entries;
        the pass count is derived from how many cache layers the model actually
        produced rather than assumed. Older entries are dropped as they go stale
        to keep this bounded.
        """
        captures = self._observation.setdefault(module_idx, [])
        captures.append((hidden, cos, sin))
        if len(captures) > self._MAX_CAPTURES_PER_MODULE:
            del captures[:-self._MAX_CAPTURES_PER_MODULE]

    def begin_query(self, model: nn.Module, input_ids: torch.Tensor) -> None:
        self._observation = {}
        self._num_modules = len(decoder_layers(model))
        if self.strategy == "streaming_llm":
            return

        layers = decoder_layers(model)
        for idx, layer in enumerate(layers):
            attn = getattr(layer, "self_attn", None)
            if attn is None:
                continue

            def capture(module, args, kwargs, _idx=idx, _mod=attn):
                # Keep only the tail of each chunk: after the final prefill chunk
                # this holds the last window_size prompt positions, which is the
                # observation window SnapKV scores with.
                hidden = kwargs.get("hidden_states", args[0] if args else None)
                if hidden is None:
                    return None
                w = min(self.window_size, hidden.shape[1])

                pos_emb = kwargs.get("position_embeddings")
                if pos_emb is not None:
                    cos, sin = pos_emb
                    self._record(_idx, hidden[:, -w:, :].detach(),
                                 cos[:, -w:, :].detach(), sin[:, -w:, :].detach())
                    return None

                # Some architectures hand the attention module `position_ids` and
                # build cos/sin inside it (Nanbeige4.2 does). Rebuild them from
                # the module's own rotary embedding rather than giving up: the
                # alternative used to be a silent fallback to a bare recency
                # window, which is not SnapKV and scores like a broken method.
                pos_ids = kwargs.get("position_ids")
                rotary = getattr(_mod, "rotary_emb", None) or getattr(
                    getattr(self.model, "model", self.model), "rotary_emb", None)
                if pos_ids is None or rotary is None:
                    return None
                try:
                    cos, sin = rotary(hidden, pos_ids[:, -w:])
                except Exception:
                    return None
                self._record(_idx, hidden[:, -w:, :].detach(), cos.detach(), sin.detach())
                return None

            self.hooks.append(attn.register_forward_pre_hook(capture, with_kwargs=True))

    def _observation_queries(self, attn_module: nn.Module, layer_idx: int) -> Optional[torch.Tensor]:
        """Recompute the observation window's post-RoPE queries.

        Uses the model's own ``q_proj`` weights, the model's own rotary
        embeddings for those exact positions, and the model's own
        ``apply_rotary_pos_emb``; nothing about the attention maths is restated
        here.  Only the last ``window_size`` positions are recomputed, so the
        extra work is negligible against a multi-thousand-token prefill.
        """
        n_modules = getattr(self, "_num_modules", 0) or 1
        module_idx, pass_idx = layer_idx % n_modules, layer_idx // n_modules
        captures = self._observation.get(module_idx)
        if not captures:
            return None
        # The last `passes` entries are the final chunk's, one per loop pass.
        recent = captures[-self._passes:] if self._passes > 1 else captures[-1:]
        hidden, cos, sin = recent[pass_idx] if pass_idx < len(recent) else recent[-1]

        module_ns = type(attn_module).__module__
        apply_rope = getattr(__import__(module_ns, fromlist=["apply_rotary_pos_emb"]),
                             "apply_rotary_pos_emb", None)
        if apply_rope is None:
            return None

        head_dim = attn_module.head_dim
        q = attn_module.q_proj(hidden)
        q = q.view(*hidden.shape[:-1], -1, head_dim).transpose(1, 2)
        if hasattr(attn_module, "q_norm") and attn_module.q_norm is not None:
            q = attn_module.q_norm(q)
        # Architectures disagree on this function's shape. Llama/Qwen take
        # (q, k, cos, sin, position_ids=None, unsqueeze_dim=1) and return a pair;
        # Gemma 4 takes (x, cos, sin, unsqueeze_dim=1) and returns one tensor.
        # Calling the pair form on Gemma passes `sin` where `unsqueeze_dim` is
        # expected and raises "unsqueeze(): argument 'dim' must be int, not
        # Tensor", so the signature decides the call.
        import inspect as _inspect
        try:
            params = list(_inspect.signature(apply_rope).parameters)
        except (TypeError, ValueError):
            params = []
        if len(params) >= 4 and params[1] in ("k", "key", "key_states"):
            q, _ = apply_rope(q, q, cos, sin)
        else:
            q = apply_rope(q, cos, sin)
        return q

    # ------------------------------------------------------------------ #
    # The transform                                                       #
    # ------------------------------------------------------------------ #

    def transform_cache(
        self,
        cache: Any,
        input_ids: torch.Tensor,
        valid_length: int,
    ) -> Tuple[Any, Dict[str, Any]]:
        target = self.max_tokens_retained or valid_length
        target = max(self.sink_tokens + self.window_size + 1, min(valid_length, target))
        if target >= valid_length:
            return cache, {"retained_tokens": valid_length, "evicted": 0, "applied": False}

        pairs = kv_tensors(cache, valid_length=valid_length)
        if self.strategy == "streaming_llm":
            new_pairs = self._evict_streaming_llm(pairs, target)
        else:
            new_pairs = self._evict_snapkv(pairs, target, growing_layer_indices(cache))

        retained = int(new_pairs[0][0].shape[-2])
        new_cache = rebuild_cache(new_pairs, cache=cache)
        del pairs, new_pairs
        return new_cache, {
            "retained_tokens": retained,
            "evicted": valid_length - retained,
            "strategy": self.strategy,
            "applied": True,
        }

    def _evict_streaming_llm(
        self, pairs: List[Tuple[torch.Tensor, torch.Tensor]], target: int
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        StartRecentKVCache = upstream.load_streaming_llm()
        start_size = min(self.sink_tokens, max(1, target // 4))
        evictor = StartRecentKVCache(
            start_size=start_size,
            recent_size=target - start_size,
            k_seq_dim=2,
            v_seq_dim=2,
        )
        out = evictor([list(p) for p in pairs])
        return [(k, v) for k, v in out]

    def _evict_snapkv(
        self,
        pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        target: int,
        cache_indices: Optional[List[int]] = None,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        SnapKVCluster = upstream.load_snapkv()
        layers = decoder_layers(self.model)
        cluster = SnapKVCluster(
            window_size=self.window_size,
            max_capacity_prompt=target,
            kernel_size=self.kernel_size,
            pooling=self.pooling,
        )

        out: List[Tuple[torch.Tensor, torch.Tensor]] = []
        n_modules = len(layers) or 1
        # How many times the model ran each module: cache layers / modules.
        # Measured, not assumed from config, so a model that shares KV across
        # loops (one cache layer per module) resolves to a single pass.
        self._num_modules = n_modules
        self._passes = max(1, len(pairs) // n_modules)
        # `cache_indices` are the layer's real positions in the cache. On a
        # hybrid model the growing layers are scattered (Gemma 4: 3, 7, 11, ...),
        # so a position within the compacted list is not the module index.
        indices = cache_indices if cache_indices is not None else list(range(len(pairs)))
        for pos, (k, v) in enumerate(pairs):
            idx = indices[pos]
            # Cache layer -> attention module, folding depth-recurrent passes.
            attn_module = layers[idx % n_modules].self_attn
            q_obs = self._observation_queries(attn_module, idx)
            groups = int(getattr(attn_module, "num_key_value_groups", 1))

            if q_obs is None:
                # Refuse to substitute. Without the observation window this is
                # not SnapKV -- it degenerates to a bare recency window, which
                # scores like a badly broken method and would be published under
                # SnapKV's name. Fail loudly so the cause gets fixed instead.
                raise RuntimeError(
                    f"SnapKV could not recover the observation-window queries for layer "
                    f"{idx} of {type(attn_module).__name__}. The capture hook needs either "
                    f"`position_embeddings` or `position_ids` plus a rotary embedding on "
                    f"the attention module or the base model. Without them the method "
                    f"cannot be evaluated on this architecture."
                )

            if groups == 1:
                # Head counts already match: upstream's own function, verbatim.
                nk, nv = cluster.update_kv(k, q_obs, v, None, groups)
                out.append((nk, nv))
                continue

            nk, nv = snapkv_compress_gqa(
                k, v, q_obs,
                window_size=self.window_size,
                max_capacity_prompt=target,
                kernel_size=self.kernel_size,
                pooling=self.pooling,
                num_key_value_groups=groups,
            )
            out.append((nk, nv))

        return out

    def end_query(self) -> None:
        self._observation = {}
        super().end_query()

    # ------------------------------------------------------------------ #
    # Resource accounting                                                 #
    # ------------------------------------------------------------------ #

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers, num_kv_heads, head_dim = self.model_kv_geometry()

        floor = self.sink_tokens + self.window_size + 1
        if self.max_tokens_retained is not None:
            retained = min(context_length, max(floor, self.max_tokens_retained))
        else:
            retained = min(context_length, max(floor, int(context_length * self.retention_ratio)))

        # Retained tokens stay in fp16; the saving is purely in token count.
        algorithmic_bytes = 2.0 * num_layers * num_kv_heads * head_dim * retained * 2.0
        # Each surviving token needs its original position recorded (int32) so
        # RoPE-consistent decoding can address it.
        index_overhead_bytes = retained * 4.0 * num_layers

        dense_elems = self.dense_element_count(context_length)
        effective_bpe = (algorithmic_bytes + index_overhead_bytes) * 8.0 / max(1, dense_elems)

        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=effective_bpe,
            total_tokens_stored=retained,
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            algorithmic_bytes=algorithmic_bytes,
            metadata_overhead_bytes=index_overhead_bytes,
            custom_metrics={
                "strategy": self.strategy,
                "retained_tokens": retained,
                "retention_ratio": retained / max(1, context_length),
                "window_size": self.window_size,
                "sink_tokens": self.sink_tokens,
            },
        )
