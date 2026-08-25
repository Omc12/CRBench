"""
Chunked-prefill inference core for CRBench.

Long-context evaluation on a consumer GPU fails for two *different* reasons that
are easy to conflate:

1. The KV cache itself does not fit.  That is a real, reportable property of the
   representation under test and CRBench must measure it.
2. Prefill *activations* do not fit.  Feeding 128K tokens through the model in
   one forward materialises MLP intermediates of shape (L, intermediate_size);
   for Qwen2.5-7B at 131072 tokens that is 4.8 GiB of transient memory that has
   nothing to do with the KV representation being benchmarked.

Reason 2 is an artifact of how the prompt is fed, not a property of the method,
so this module removes it: the prompt is prefilled in fixed-size chunks sharing
one preallocated cache.  Peak activation memory then tracks the chunk size
instead of the context length, and the only thing scaling with L is the KV cache
-- which is exactly what we set out to measure.

Why the cache is preallocated
-----------------------------
``DynamicCache`` implements ``update`` as ``torch.cat([self.keys, new], dim=-2)``,
so every prefill chunk and every decode step reallocates and copies the whole
cache.  Measured on this project's 12 GiB card with Qwen2.5-7B NF4 at 65536
tokens: 8.69 GiB of live tensors, but 13.00 GiB *reserved* by the caching
allocator, because each freed intermediate leaves a segment no differently-sized
request can reuse.  That overflows the device and spills into host memory --
prefill fell to 132 tok/s and decode to 48 s/token.  ``expandable_segments``,
the usual mitigation, is not supported on Windows ("expandable_segments not
supported on this platform").  A ``StaticCache`` sized once up front removes both
the fragmentation and the per-token O(L) copy.
"""

from __future__ import annotations

import functools
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import DynamicCache
from transformers.cache_utils import Cache, DynamicLayer


DEFAULT_PREFILL_CHUNK = 4096


class PreallocatedLayer(DynamicLayer):
    """A cache layer that is allocated once and written in place.

    ``DynamicLayer.update`` is ``torch.cat([self.keys, new], dim=-2)``: a fresh
    allocation and a full copy of the cache on *every* prefill chunk and *every*
    decoded token.  Measured with Qwen2.5-7B NF4 at 65536 tokens on a 12 GiB
    card, that cost 10.0 s per decoded token -- the repeated 3.5 GiB
    allocate/copy/free cycle fragments the caching allocator until it spills
    into host memory.

    ``StaticCache`` avoids the reallocation but reports its full padded length,
    so prefill attention runs against a padded 4-D mask; that made a 32768-token
    prefill more than ten times slower than the dynamic path.

    This layer takes the third option: preallocate to the known maximum, write
    each update in place, and return a **view of the valid prefix**.  The model
    therefore sees exactly the same key/value length it would see from a
    ``DynamicCache`` -- so the mask is built identically and prefill keeps its
    speed -- while nothing is ever reallocated or copied.
    """

    is_sliding = False

    def __init__(self, max_cache_len: int) -> None:
        super().__init__()
        self.max_cache_len = int(max_cache_len)
        self.pos = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        b, h, _, d = key_states.shape
        self.dtype, self.device = key_states.dtype, key_states.device
        self.keys = torch.empty(b, h, self.max_cache_len, d, dtype=self.dtype, device=self.device)
        self.values = torch.empty(b, h, self.max_cache_len, d, dtype=self.dtype, device=self.device)
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        n = key_states.shape[-2]
        if self.pos + n > self.max_cache_len:
            raise RuntimeError(
                f"PreallocatedLayer overflow: {self.pos} + {n} > {self.max_cache_len}"
            )
        self.keys[..., self.pos:self.pos + n, :] = key_states
        self.values[..., self.pos:self.pos + n, :] = value_states
        self.pos += n
        return self.keys[..., :self.pos, :], self.values[..., :self.pos, :]

    def get_seq_length(self) -> int:
        return self.pos

    # get_max_length / get_max_cache_shape deliberately inherit DynamicLayer's
    # "-1 == unbounded".  transformers picks its attention-mask strategy from
    # them: reporting a finite maximum selects the padded static-cache mask,
    # which builds a (q_len x max_cache_len) bias instead of a
    # (q_len x valid_len) one.  Measured with Qwen2.5-7B NF4 at 32768 tokens,
    # reporting the maximum here cost 573 s of prefill against 16 s -- a 36x
    # regression -- and 1.5 GiB of extra peak allocation.  The preallocation is
    # an allocator detail; the model must not see it.

    def crop(self, tokens_to_remove: int) -> None:
        if tokens_to_remove > 0:  # legacy absolute-size form
            self.pos = min(self.pos, tokens_to_remove)
        else:
            self.pos = max(0, self.pos - abs(tokens_to_remove))

    def reset(self) -> None:
        self.pos = 0


class PreallocatedCache(Cache):
    """A ``Cache`` whose layers are :class:`PreallocatedLayer`."""

    def __init__(self, num_layers: int, max_cache_len: int) -> None:
        super().__init__(layers=[PreallocatedLayer(max_cache_len) for _ in range(num_layers)])

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.layers[layer_idx].get_seq_length()

    def crop(self, max_length: int) -> None:
        for layer in self.layers:
            layer.crop(max_length)


@functools.lru_cache(maxsize=8)
def _accepts_logits_to_keep(model_cls: type) -> bool:
    """Does this model's forward accept `logits_to_keep`?

    Prefill only needs the last position's logits; without this argument the
    model materialises (chunk_size x vocab_size) every chunk, which on a 150K
    vocabulary is hundreds of MiB of pure waste. Vendored `trust_remote_code`
    modelling written before the argument existed -- Nanbeige4.2 -- raises
    TypeError on it, so it is passed only where it is supported.
    """
    try:
        params = inspect.signature(model_cls.forward).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return "logits_to_keep" in params or "num_logits_to_keep" in params
    return "logits_to_keep" in params


def make_cache(model: Any) -> Any:
    """Build the cache this architecture actually indexes into.

    There is no single right answer, and each wrong answer fails differently:

    * **Hybrid attention** (Qwen3.5's linear-attention layers, Gemma 4's
      sliding-window layers) needs *typed* layers. A plain ``DynamicCache``
      gives them ordinary KV slots and the model raises inside
      ``update_conv_state``. These need ``DynamicCache(config=...)``.
    * **Depth recurrence** (Nanbeige4.2: ``num_loops = 2``,
      ``loop_share_kv = False``) indexes 44 slots from 22 decoder modules. A
      config-built cache allocates one slot per module and the model raises
      ``IndexError`` on the second loop. These need the grow-on-demand
      ``DynamicCache()``, which appends a layer per new index -- the same object
      the model's own code constructs for itself.

    Passing ``None`` and letting the model build its own would cover both, but
    Nanbeige's vendored modeling then routes through
    ``DynamicCache.from_legacy_cache``, removed in transformers 5.

    The rule: typed layers only when the model declares heterogeneous ones.
    """
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", cfg)
    layer_types = list(getattr(text_cfg, "layer_types", []) or [])
    if layer_types and any(t != "full_attention" for t in layer_types):
        return DynamicCache(config=cfg)
    return DynamicCache()


@dataclass
class GenerationTrace:
    """Everything measured during one prompt -> generation cycle."""

    generated_ids: torch.Tensor          # (1, prompt_len + generated_tokens)
    prompt_tokens: int
    generated_tokens: int

    # --- timing (seconds, device-synchronised at every boundary) ---
    prefill_seconds: float               # whole prompt through the model
    compression_seconds: float           # the method's cache transform
    ttft_seconds: float                  # prefill + compression + first token
    decode_seconds: float                # tokens 2..N
    inter_token_seconds: List[float] = field(default_factory=list)

    # --- memory (bytes) ---
    peak_prefill_bytes: int = 0          # torch peak allocation during prefill
    peak_total_bytes: int = 0            # torch peak allocation over the cycle
    weight_baseline_bytes: int = 0       # allocation attributable to weights
    kv_bytes_before_transform: int = 0   # resident cache storage after prefill
    kv_bytes_after_transform: int = 0    # resident cache storage post-transform
    kv_tokens_after_transform: int = 0   # cache length post-transform

    # KV geometry read off the live cache, not inferred from the config.
    cache_geometry: Dict[str, int] = field(default_factory=dict)
    method_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def decode_seconds_per_token(self) -> float:
        n = len(self.inter_token_seconds)
        if n == 0 or self.decode_seconds <= 0.0:
            return self.ttft_seconds
        return self.decode_seconds / n

    @property
    def decode_throughput_tok_per_sec(self) -> float:
        spt = self.decode_seconds_per_token
        return 1.0 / spt if spt > 0 else 0.0

    @property
    def prefill_throughput_tok_per_sec(self) -> float:
        return self.prompt_tokens / self.prefill_seconds if self.prefill_seconds > 0 else 0.0

    @property
    def latency_jitter_ms(self) -> float:
        xs = self.inter_token_seconds
        if len(xs) <= 1:
            return 0.0
        mean = sum(xs) / len(xs)
        var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
        return float(var ** 0.5) * 1000.0


def sync(device: Optional[torch.device] = None) -> None:
    """Device-synchronise so wall-clock boundaries mean what they say."""
    if torch.cuda.is_available() and (device is None or device.type == "cuda"):
        torch.cuda.synchronize(device)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        try:
            torch.mps.synchronize()
        except Exception:
            pass


class _LegacyLayerView:
    """Gives a transformers<5 cache the 5.x ``.keys`` / ``.values`` shape."""

    def __init__(self, cache: Any, idx: int) -> None:
        self._cache = cache
        self._idx = idx

    @property
    def keys(self) -> torch.Tensor:
        return self._cache.key_cache[self._idx]

    @keys.setter
    def keys(self, t: torch.Tensor) -> None:
        self._cache.key_cache[self._idx] = t

    @property
    def values(self) -> torch.Tensor:
        return self._cache.value_cache[self._idx]

    @values.setter
    def values(self, t: torch.Tensor) -> None:
        self._cache.value_cache[self._idx] = t


def cache_layers(cache: Any) -> List[Any]:
    """Per-layer entries of a transformers Cache, normalised across versions."""
    layers = getattr(cache, "layers", None)
    if layers is not None:
        return list(layers)
    if getattr(cache, "key_cache", None) is not None:
        return [_LegacyLayerView(cache, i) for i in range(len(cache.key_cache))]
    raise TypeError(f"Unsupported cache type for CRBench: {type(cache)!r}")


def measure_cache_bytes(cache: Any, valid_length: Optional[int] = None) -> int:
    """Resident storage of the cache in bytes.

    A ``StaticCache`` is preallocated to its maximum length, so its raw storage
    reflects the allocation rather than the representation.  ``valid_length``
    rescales it to the tokens actually written, which is the quantity Part 1
    scores.  Storages are keyed by data pointer so a key/value pair that shares
    one allocation is not double-counted.
    """
    seen: Dict[int, int] = {}
    for layer in cache_layers(cache):
        for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if t is None:
                continue
            if valid_length is not None and t.dim() >= 2:
                allocated_len = t.shape[-2]
                if allocated_len > 0:
                    n = t.numel() // allocated_len * min(valid_length, allocated_len)
                    seen[id(t)] = n * t.element_size()
                    continue
            storage = t.untyped_storage()
            seen[storage.data_ptr()] = storage.nbytes()
    return int(sum(seen.values()))


def observe_cache_geometry(cache: Any) -> Dict[str, int]:
    """Read the KV geometry off a live cache instead of inferring it from config.

    Config-derived geometry is wrong for a growing share of architectures, and
    wrong in different directions:

    * **Depth-recurrent models.** Nanbeige4.2 sets ``num_loops = 2`` with
      ``loop_share_kv = False``, so its 22 decoder layers occupy 44 cache slots.
      Reading ``num_hidden_layers`` halves its true footprint.
    * **Hybrid attention.** Qwen3.5 interleaves 24 linear-attention layers with
      8 full-attention ones, and Gemma 4 interleaves 35 sliding-window layers
      with 7 full ones. Only the full-attention layers hold a cache that grows
      with context; counting every layer overstates the footprint four- to
      six-fold.
    * **Heterogeneous layers.** Gemma 4's per-layer configs differ in
      ``head_dim`` between its sliding and full layers, so there is no single
      global value to read.

    Counting what the model actually allocated sidesteps all of it. Returns the
    layers that hold a real, growing KV tensor, plus their head geometry.
    """
    layers = cache_layers(cache)
    kv_layers = 0
    heads = 0
    head_dim = 0
    elements = 0
    for layer in layers:
        k = getattr(layer, "keys", None)
        if not torch.is_tensor(k) or k.numel() == 0 or k.dim() < 4:
            continue
        kv_layers += 1
        heads = max(heads, int(k.shape[1]))
        head_dim = max(head_dim, int(k.shape[-1]))
        # Keys and values together, per token of this layer's cache.
        elements += 2 * int(k.shape[1]) * int(k.shape[-1])
    return {
        "kv_layers": kv_layers,
        "num_kv_heads": heads,
        "head_dim": head_dim,
        "total_layers": len(layers),
        # Authoritative for sizing: sums each layer's own geometry, so
        # heterogeneous layers do not need one global head count.
        "kv_elements_per_token": elements,
    }


def cache_seq_length(cache: Any) -> int:
    """Number of tokens currently held, as a plain int."""
    try:
        n = cache.get_seq_length()
        return int(n.item()) if torch.is_tensor(n) else int(n)
    except Exception:
        layers = cache_layers(cache)
        if layers and getattr(layers[0], "keys", None) is not None:
            return int(layers[0].keys.shape[-2])
        return 0


def kv_tensors(cache: Any, valid_length: Optional[int] = None) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """(key, value) per layer, sliced to the written region for static caches."""
    out: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for layer in cache_layers(cache):
        k, v = layer.keys, layer.values
        if valid_length is not None:
            k, v = k[..., :valid_length, :], v[..., :valid_length, :]
        out.append((k, v))
    return out


#: Decode headroom reserved when a method rebuilds the cache.  Must exceed the
#: benchmark's ``max_new_tokens`` plus the one re-forwarded prompt token.
#:
#: Sized for reasoning models, which answer only after closing a chain-of-thought
#: block and so need a far larger generation budget than the ~32 tokens an
#: extractive task otherwise wants.  The cost is one padded allocation per
#: rebuilt cache: at Nanbeige4.2's 176 KiB/token that is 88 MiB, which is cheap
#: against the several GiB such a cache already occupies.
REBUILD_RESERVE = 512


def rebuild_cache(
    kv_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    reserve: int = REBUILD_RESERVE,
) -> PreallocatedCache:
    """Pack per-layer (k, v) tensors into a fresh preallocated cache.

    Used after a method shortens or rewrites the prompt cache.  ``reserve``
    leaves room for the tokens still to be decoded, so the compressed cache is
    never reallocated mid-generation either.  Source tensors are released layer
    by layer as they are copied: the caller is typically holding a cache several
    GiB in size and materialising a second full copy would defeat the point.
    """
    length = int(kv_pairs[0][0].shape[-2])
    cache = PreallocatedCache(len(kv_pairs), length + reserve)
    for idx in range(len(kv_pairs)):
        k, v = kv_pairs[idx]
        cache.layers[idx].update(k, v)
        kv_pairs[idx] = (None, None)  # drop our reference so the source can free
    return cache


def to_preallocated(cache: Any, reserve: int = REBUILD_RESERVE) -> Any:
    """Move a grown cache into preallocated storage, releasing the source as it goes.

    Called between prefill and decode.  The two stages want opposite layouts:

    * Prefill wants **contiguous** key/value tensors.  A prefix view of a
      preallocated buffer has a head-dimension stride that does not match its
      logical shape, and SDPA falls back to its math backend for such inputs --
      which materialises the full (heads x q_len x kv_len) attention matrix.
      Measured with Qwen2.5-7B NF4 at 32768 tokens: peak allocation 13.03 GiB
      against 9.75, and prefill 435 s against 16 s.
    * Decode wants **no reallocation**.  ``DynamicCache`` concatenates the whole
      cache on every step; at 65536 tokens that reallocate/copy/free cycle
      fragmented the allocator into host memory at 10.0 s per token.

    With ``q_len == 1`` the strided view costs nothing -- the attention matrix is
    a single row -- so decoding from preallocated storage is free of both
    problems.  The copy is done layer by layer with the source dropped
    immediately, so the transient cost is one layer, not a second full cache.
    """
    if isinstance(cache, PreallocatedCache):
        return cache

    layers = cache_layers(cache)
    # Only take over a cache whose every layer is an ordinary growing KV layer.
    # Hybrid architectures mix in linear-attention or sliding-window layers that
    # carry their own state and update rules; replacing those with plain
    # preallocated storage silently breaks the model rather than speeding it up.
    if not all(type(l).__name__ == "DynamicLayer" for l in layers):
        return cache
    length = int(layers[0].keys.shape[-2])
    new_cache = PreallocatedCache(len(layers), length + reserve)
    for idx, layer in enumerate(layers):
        new_cache.layers[idx].update(layer.keys, layer.values)
        layer.keys = None
        layer.values = None
    return new_cache


@torch.no_grad()
def chunked_prefill_generate(
    model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    chunk_size: int = DEFAULT_PREFILL_CHUNK,
    eos_token_id: Optional[int] = None,
    on_chunk_end: Optional[Callable[[Any, int, int, int], None]] = None,
    on_token_end: Optional[Callable[[Any, int], None]] = None,
    transform_cache: Optional[Callable[[Any, torch.Tensor, int], Tuple[Any, Dict[str, Any]]]] = None,
    weight_baseline_bytes: int = 0,
    device: Optional[torch.device] = None,
    empty_cache_between_chunks: bool = True,
) -> GenerationTrace:
    """Prefill ``input_ids`` in chunks, apply the method's cache transform, decode.

    Args:
        on_chunk_end: ``(cache, chunk_start, chunk_end, valid_length)`` after each
            prefill chunk lands in the cache.  Streaming methods -- KV
            quantization compresses what is *stored*, so later chunks must attend
            to the compressed history -- mutate the cache in place here.
        transform_cache: ``(cache, input_ids, valid_length) -> (cache, metadata)``
            once the whole prompt is resident.  One-shot prompt compressors
            (SnapKV, StreamingLLM eviction, low-rank, merging, DKV) run here and
            may return a *different*, shorter cache.

    When a transform runs, the last prompt position is re-forwarded against the
    transformed cache before the first token is emitted.  The logits produced
    during prefill attended to the *uncompressed* history; scoring them would
    credit the method with information its representation no longer holds.

    Decoding is greedy: CRBench compares a method to the dense baseline on the
    identical query, and sampling noise is indistinguishable from a quality
    difference.
    """
    if device is None:
        device = input_ids.device
    if max_new_tokens + 2 > REBUILD_RESERVE:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} exceeds the decode headroom a method's "
            f"rebuilt cache reserves (REBUILD_RESERVE={REBUILD_RESERVE})."
        )

    prompt_len = int(input_ids.shape[-1])
    # Let the model build its own cache on the first chunk rather than guessing a
    # class. Which cache is correct is architecture-specific and getting it wrong
    # fails in different ways: Qwen3.5's linear-attention layers need
    # LinearAttentionLayer slots and raise on a plain DynamicCache, Gemma 4 needs
    # sliding-window layers, and Nanbeige4.2 runs its 22 decoder layers twice and
    # needs 44 slots -- more than a config-built cache allocates. Passing None
    # lets each model construct exactly what it indexes into.
    cache: Any = make_cache(model)
    keep_last: Dict[str, Any] = (
        {"logits_to_keep": 1} if _accepts_logits_to_keep(type(model)) else {})

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ---- Stage 1: chunked prefill -------------------------------------------
    sync(device)
    t_prefill_start = time.perf_counter()

    logits = None
    for start in range(0, prompt_len, chunk_size):
        end = min(start + chunk_size, prompt_len)
        out = model(
            input_ids=input_ids[:, start:end],
            past_key_values=cache,
            use_cache=True,
            # Only the final position is needed to pick the first token; keeping
            # every chunk's logits would allocate (chunk, vocab) for nothing.
            **keep_last,
        )
        logits = out.logits
        del out
        if on_chunk_end is not None:
            on_chunk_end(cache, start, end, end)
        if empty_cache_between_chunks and torch.cuda.is_available() and device.type == "cuda":
            # Windows has no expandable_segments, so the allocator cannot reuse
            # a freed segment for a differently sized request.  Without this,
            # Qwen2.5-7B at 65536 tokens reserved 13.00 GiB for 8.69 GiB of live
            # tensors and spilled into host memory, dropping prefill from 811 to
            # 132 tok/s.  Releasing between chunks costs nothing measurable.
            torch.cuda.empty_cache()

    sync(device)
    t_prefill_end = time.perf_counter()

    peak_prefill = (
        int(torch.cuda.max_memory_allocated(device))
        if torch.cuda.is_available() and device.type == "cuda"
        else 0
    )
    kv_before = measure_cache_bytes(cache, valid_length=prompt_len)
    geometry = observe_cache_geometry(cache)

    # ---- Stage 2: the method's cache transform -------------------------------
    method_metadata: Dict[str, Any] = {}
    transformed = False
    t_compress_start = time.perf_counter()
    if transform_cache is not None:
        cache, meta = transform_cache(cache, input_ids, prompt_len)
        transformed = True
        if isinstance(meta, dict):
            method_metadata = meta
    sync(device)
    t_compress_end = time.perf_counter()

    kv_tokens_after = cache_seq_length(cache)
    # Always scale to the tokens actually held: a rebuilt cache is preallocated
    # with decode headroom, and charging the method for that padding would
    # overstate its footprint.
    kv_after = measure_cache_bytes(cache, valid_length=kv_tokens_after)

    # Hand the cache over to preallocated storage for the decode stage.  A
    # transform that rebuilt the cache has already produced one.
    cache = to_preallocated(cache, reserve=max_new_tokens + 2)
    if empty_cache_between_chunks and torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.empty_cache()

    # Positions continue from the ORIGINAL prompt length, never from the
    # compressed cache length.
    #
    # A method that evicts or merges leaves the surviving keys carrying the
    # rotary phase they were written with -- their true positions in the prompt.
    # transformers derives position_ids from `past_key_values.get_seq_length()`
    # when they are not supplied, so after eviction the next query would be
    # rotated at the *compressed* length, thousands of positions behind the keys
    # it has to match, and RoPE's relative offsets would go negative.
    #
    # Upstream SnapKV handles this by tracking the uncompressed `kv_seq_len` and
    # feeding it to `prepare_inputs_for_generation`; the same convention is used
    # here. Measured on Qwen2.5-0.5B with a 1229-token prompt, keeping the last
    # 60% (with the needle inside the retained window): the passkey came back as
    # "366" instead of "361659" without this, and correctly with it.
    #
    # When no transform ran, the cache length equals the prompt length and these
    # position_ids are exactly what transformers would have derived anyway.
    def _positions(index: int) -> torch.Tensor:
        return torch.tensor([[index]], device=input_ids.device, dtype=torch.long)

    # ---- Stage 3: first generated token (completes TTFT) ---------------------
    if transformed:
        # Re-forward the final prompt token so it attends to the transformed
        # history.  It is already the last entry of the cache (every method here
        # preserves the recent window), so drop it first to avoid duplication.
        cache.crop(kv_tokens_after - 1)
        out = model(input_ids=input_ids[:, -1:], past_key_values=cache,
                    position_ids=_positions(prompt_len - 1),
                    use_cache=True, **keep_last)
        logits = out.logits
        del out

    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    sync(device)
    t_first_token = time.perf_counter()
    generated: List[int] = [int(next_token.item())]

    # ---- Stage 4: remaining decode steps -------------------------------------
    inter_token: List[float] = []
    t_decode_start = time.perf_counter()
    for step in range(max_new_tokens - 1):
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break
        t0 = time.perf_counter()
        # Generated token `step` occupies original position prompt_len + step.
        out = model(input_ids=next_token, past_key_values=cache,
                    position_ids=_positions(prompt_len + step),
                    use_cache=True, **keep_last)
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        del out
        if on_token_end is not None:
            on_token_end(cache, cache_seq_length(cache) - 1)
        sync(device)
        inter_token.append(time.perf_counter() - t0)
        generated.append(int(next_token.item()))
    sync(device)
    t_decode_end = time.perf_counter()

    peak_total = (
        int(torch.cuda.max_memory_allocated(device))
        if torch.cuda.is_available() and device.type == "cuda"
        else 0
    )

    gen_tensor = torch.tensor([generated], device=input_ids.device, dtype=input_ids.dtype)
    trace = GenerationTrace(
        generated_ids=torch.cat([input_ids, gen_tensor], dim=-1),
        prompt_tokens=prompt_len,
        generated_tokens=len(generated),
        prefill_seconds=t_prefill_end - t_prefill_start,
        compression_seconds=t_compress_end - t_compress_start,
        ttft_seconds=t_first_token - t_prefill_start,
        decode_seconds=t_decode_end - t_decode_start,
        inter_token_seconds=inter_token,
        peak_prefill_bytes=peak_prefill,
        peak_total_bytes=peak_total,
        weight_baseline_bytes=weight_baseline_bytes,
        kv_bytes_before_transform=kv_before,
        kv_bytes_after_transform=kv_after,
        kv_tokens_after_transform=kv_tokens_after,
        cache_geometry=geometry,
        method_metadata=method_metadata,
    )

    del cache, logits
    return trace
