"""
BaseContextAdapter and state metadata specifications for CRBench.
Provides the standard, method-agnostic interface for evaluating any KV/context representation.

A context-compression method is modelled as a **transform on the KV cache**,
because that is what these methods physically are.  Two hooks cover the
published families:

* ``on_chunk_stored`` -- a *streaming* transform.  KV quantization compresses
  what is written to the cache, so every later prefill chunk must attend to the
  compressed history, not the original.
* ``transform_cache`` -- a *one-shot prompt* transform.  SnapKV, StreamingLLM
  eviction, token merging, low-rank projection and DKV all run once, after the
  prompt is resident, and may return a shorter cache.

Adapters that need per-query state (SnapKV must capture the observation-window
queries during prefill) set it up in ``begin_query`` and release it in
``end_query``.  An adapter that implements neither hook is the dense baseline.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from crbench.core.budget import ContextBudget


class ExecutionStatus(str, Enum):
    """Execution status for benchmark sample evaluation."""
    SUCCESS = "SUCCESS"
    OOM = "OOM"
    UNSUPPORTED = "UNSUPPORTED"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    INVALID_CONFIG = "INVALID_CONFIG"


@dataclass
class KVStateMetadata:
    """
    Metadata describing the state and resource characteristics of a context representation.
    Explicitly separates algorithmic state payload, metadata overheads, and alignment padding.
    """
    adapter_name: str
    method_type: str  # "dense", "quantization", "eviction", "merging", "compressed", "custom"
    effective_bits_per_element: float
    total_tokens_stored: int
    context_length: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    algorithmic_bytes: float              # Raw compressed state tensor storage
    metadata_overhead_bytes: float = 0.0  # Scales, codebooks, indices, cluster centroids
    alignment_overhead_bytes: float = 0.0 # Byte alignment / padding (e.g. 64-byte multiples)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_state_bytes(self) -> float:
        """Total persistent KV representation storage in bytes."""
        return self.algorithmic_bytes + self.metadata_overhead_bytes + self.alignment_overhead_bytes

    @property
    def effective_bits_per_token(self) -> float:
        """Total persistent bits divided by sequence length."""
        return (self.total_state_bytes * 8.0) / max(1, self.context_length)

    @property
    def compression_ratio(self) -> float:
        """Ratio of compressed persistent bytes to baseline dense FP16 storage."""
        dense_bytes = (
            2 * self.num_layers * self.num_kv_heads * self.head_dim * self.context_length * 2.0  # FP16 = 2 bytes
        )
        if dense_bytes == 0:
            return 1.0
        return self.total_state_bytes / dense_bytes


class BaseContextAdapter(ABC):
    """
    Abstract Base Class for all context representations and compression methods in CRBench.

    To benchmark a new KV cache representation or memory method:
    1. Subclass `BaseContextAdapter` and register via `@Registry.register_adapter("your_name")`.
    2. Implement `method_type`, `apply_budget()`, `get_kv_metadata()`.
    3. Implement `transform_cache()` (one-shot prompt compression) and/or
       `on_chunk_stored()` (streaming compression of what gets written).
    """

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self._name = name
        self.config = config or {}
        self.current_budget: Optional[ContextBudget] = None
        self.model: Optional[nn.Module] = None
        self.tokenizer: Optional[Any] = None
        self.hooks: List[Any] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    @abstractmethod
    def method_type(self) -> str:
        """Type tag: 'dense', 'quantization', 'eviction', 'merging', 'compressed', 'custom'."""
        pass

    # ------------------------------------------------------------------ #
    # Provenance                                                          #
    # ------------------------------------------------------------------ #

    def provenance(self) -> Dict[str, Any]:
        """Where this method's implementation comes from.

        Written verbatim into the results manifest so a reader can tell an
        upstream reference implementation from a CRBench-internal one.
        """
        return {"implementation": "crbench_internal"}

    # ------------------------------------------------------------------ #
    # Model / budget lifecycle                                            #
    # ------------------------------------------------------------------ #

    def prepare_model(self, model: nn.Module, tokenizer: Optional[Any] = None) -> None:
        """Stores model and tokenizer references; may install persistent patches."""
        self.model = model
        self.tokenizer = tokenizer

    def validate_environment(self, device: torch.device) -> Tuple[bool, str]:
        """
        Validates whether the execution environment and device support this adapter.
        Returns (is_supported, reason_if_not).
        """
        return True, "Supported"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        """
        Configures the adapter parameters (quantization bitwidth, eviction cache budget, etc.)
        to satisfy the requested ContextBudget at the target context length.
        """
        self.current_budget = budget

    # ------------------------------------------------------------------ #
    # Cache-transform protocol                                            #
    # ------------------------------------------------------------------ #

    #: True when the method compresses what is *written* to the cache, so that
    #: later prefill chunks and decode steps attend to the compressed history.
    streaming_transform: bool = False

    #: True when the method rewrites the prompt cache once, after prefill.
    oneshot_transform: bool = False

    def begin_query(self, model: nn.Module, input_ids: torch.Tensor) -> None:
        """Per-query setup, before prefill starts (e.g. install capture hooks)."""
        return None

    def on_chunk_stored(self, cache: Any, start: int, end: int, valid_length: int) -> None:
        """Streaming transform, called after each prefill chunk lands in the cache."""
        return None

    def on_token_stored(self, cache: Any, position: int) -> None:
        """Streaming transform for a single decoded token's KV."""
        return None

    def transform_cache(
        self,
        cache: Any,
        input_ids: torch.Tensor,
        valid_length: int,
    ) -> Tuple[Any, Dict[str, Any]]:
        """One-shot prompt-cache transform.

        Returns the (possibly new and shorter) cache plus a metadata dict
        describing what the transform actually did -- retained token counts,
        ranks, block statistics -- which is recorded per query.
        """
        return cache, {}

    def end_query(self) -> None:
        """Per-query teardown; always called, including after an exception."""
        self.cleanup()

    # ------------------------------------------------------------------ #
    # Resource accounting                                                 #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        """
        Returns accurate metadata about the KV state, analytical memory footprint,
        and effective bits per element.
        """
        pass

    #: KV geometry read off a live cache by the inference core. Set by the runner
    #: before scoring; empty until the first query has run.
    observed_geometry: Dict[str, int] = {}

    def model_kv_geometry(self) -> Tuple[int, int, int]:
        """(cache_layers, num_kv_heads, head_dim), preferring what was observed.

        The config is not a reliable source. ``num_hidden_layers`` counts decoder
        modules, and for a growing set of architectures that is not the number of
        cache slots: a depth-recurrent model like Nanbeige4.2 (``num_loops = 2``,
        ``loop_share_kv = False``) runs its 22 layers twice and occupies 44,
        while hybrid models like Qwen3.5 and Gemma 4 give only 8 of 32 and 7 of
        42 layers a cache that grows with context. Reading the config would
        halve the first and quadruple the others.
        """
        obs = self.observed_geometry
        if obs and obs.get("kv_layers"):
            return int(obs["kv_layers"]), int(obs["num_kv_heads"]), int(obs["head_dim"])

        cfg = getattr(self.model, "config", None)
        if cfg is None:
            return 32, 32, 128
        cfg = getattr(cfg, "text_config", cfg)
        num_layers = getattr(cfg, "num_hidden_layers", 32)
        # Depth recurrence multiplies cache slots unless the loops share KV.
        loops = int(getattr(cfg, "num_loops", 1) or 1)
        if loops > 1 and not getattr(cfg, "loop_share_kv", False):
            num_layers *= loops
        num_heads = getattr(cfg, "num_attention_heads", 32)
        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
        hidden_size = getattr(cfg, "hidden_size", 4096)
        try:
            head_dim = getattr(cfg, "head_dim", None) or (hidden_size // max(1, num_heads))
        except Exception:
            # Heterogeneous per-layer configs (Gemma 4) refuse a global read.
            head_dim = hidden_size // max(1, num_heads)
        return int(num_layers), int(num_kv_heads), int(head_dim)

    def dense_element_count(self, context_length: int) -> int:
        """Number of KV elements an uncompressed cache would hold at this length."""
        obs = self.observed_geometry
        if obs and obs.get("kv_elements_per_token"):
            # Sums each layer's own head geometry, so heterogeneous layers are
            # priced individually rather than through one global head count.
            return int(obs["kv_elements_per_token"]) * context_length
        num_layers, num_kv_heads, head_dim = self.model_kv_geometry()
        return 2 * num_layers * num_kv_heads * head_dim * context_length

    def compute_algorithmic_memory(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        seq_len: int,
        dense_bits: int = 16
    ) -> float:
        """
        Calculates theoretical KV memory requirement in bytes.
        """
        metadata = self.get_kv_metadata(seq_len)
        return metadata.total_state_bytes

    def reset(self) -> None:
        """Resets any internal cache, hooks, or accumulated states between samples."""
        pass

    def cleanup(self) -> None:
        """Removes any PyTorch hooks or monkey-patches from the model."""
        for hook in self.hooks:
            try:
                hook.remove()
            except Exception:
                pass
        self.hooks.clear()
