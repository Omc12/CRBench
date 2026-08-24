"""
BaseContextAdapter and state metadata specifications for CRBench.
Provides the standard, method-agnostic interface for evaluating any KV/context representation.
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
    2. Implement `method_type`, `apply_budget()`, `forward_or_generate()`, and `get_kv_metadata()`.
    3. Use `self.hooks` or module replacement in `prepare_model()` / `cleanup()`.
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

    def prepare_model(self, model: nn.Module, tokenizer: Optional[Any] = None) -> None:
        """
        Attaches hooks, replaces attention/KV cache modules, or prepares model for inference.
        Default implementation stores model and tokenizer references.
        """
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

    @abstractmethod
    def forward_or_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **kwargs: Any
    ) -> torch.Tensor:
        """
        Executes prefill over the context and generates tokens under the configured representation.
        Returns generated token IDs tensor of shape (batch, seq_len + max_new_tokens).
        """
        pass

    @abstractmethod
    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        """
        Returns accurate metadata about the KV state, analytical memory footprint,
        and effective bits per element.
        """
        pass

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
