"""
Resource budget definitions and conversion utilities for CRBench.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BudgetType(str, Enum):
    """Type of resource constraint."""
    BITS_PER_TOKEN = "bits_per_token"  # Effective bits per token for KV state
    TOTAL_BYTES = "total_bytes"        # Absolute KV memory budget in bytes
    TOKEN_CAPACITY = "token_capacity"  # Token budget (for pruning / eviction)
    COMPRESSION_RATIO = "compression_ratio"  # Ratio relative to Dense FP16 (e.g., 0.25 = 4x compression)


@dataclass
class ContextBudget:
    """
    Encapsulates a resource budget constraint.
    """
    budget_type: BudgetType
    value: float
    description: Optional[str] = None

    @classmethod
    def from_bits_per_token(cls, bpt: float) -> ContextBudget:
        return cls(budget_type=BudgetType.BITS_PER_TOKEN, value=float(bpt), description=f"{bpt:.1f} bpt")

    @classmethod
    def from_bytes(cls, num_bytes: float) -> ContextBudget:
        return cls(budget_type=BudgetType.TOTAL_BYTES, value=float(num_bytes), description=f"{num_bytes / (1024**2):.1f} MB")

    @classmethod
    def from_token_capacity(cls, tokens: int) -> ContextBudget:
        return cls(budget_type=BudgetType.TOKEN_CAPACITY, value=float(tokens), description=f"{tokens} tokens")

    @classmethod
    def from_compression_ratio(cls, ratio: float) -> ContextBudget:
        return cls(budget_type=BudgetType.COMPRESSION_RATIO, value=float(ratio), description=f"{ratio:.2f}x of dense")

    @property
    def is_bits_per_token(self) -> bool:
        return self.budget_type == BudgetType.BITS_PER_TOKEN

    @property
    def is_compression_ratio(self) -> bool:
        return self.budget_type == BudgetType.COMPRESSION_RATIO

    def to_bits_per_token(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        context_length: int,
        dense_bits: int = 16
    ) -> float:
        """
        Converts any budget format to effective bits per token.
        Analytical dense KV elements per token = 2 (Key + Value) * num_layers * num_kv_heads * head_dim.
        Dense bits per token = dense_elements_per_token * dense_bits.
        """
        dense_elements_per_token = 2 * num_layers * num_kv_heads * head_dim
        
        if self.budget_type == BudgetType.BITS_PER_TOKEN:
            return float(self.value)
        elif self.budget_type == BudgetType.COMPRESSION_RATIO:
            return float(self.value) * dense_bits
        elif self.budget_type == BudgetType.TOKEN_CAPACITY:
            # If we retain only `tokens` out of `context_length`:
            retention_fraction = min(1.0, float(self.value) / max(1, context_length))
            return retention_fraction * dense_bits
        elif self.budget_type == BudgetType.TOTAL_BYTES:
            dense_bytes_total = dense_elements_per_token * (dense_bits / 8.0) * max(1, context_length)
            fraction = min(1.0, self.value / max(1e-6, dense_bytes_total))
            return fraction * dense_bits
        else:
            raise ValueError(f"Unknown budget type {self.budget_type}")

    def to_bytes(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        context_length: int,
        dense_bits: int = 16
    ) -> float:
        """
        Converts budget to absolute total KV bytes.
        """
        dense_bytes_per_token = 2 * num_layers * num_kv_heads * head_dim * (dense_bits / 8.0)
        
        if self.budget_type == BudgetType.TOTAL_BYTES:
            return float(self.value)
        elif self.budget_type == BudgetType.BITS_PER_TOKEN:
            fraction = self.value / dense_bits
            return fraction * dense_bytes_per_token * context_length
        elif self.budget_type == BudgetType.COMPRESSION_RATIO:
            return self.value * dense_bytes_per_token * context_length
        elif self.budget_type == BudgetType.TOKEN_CAPACITY:
            retained_tokens = min(float(context_length), self.value)
            return retained_tokens * dense_bytes_per_token
        else:
            raise ValueError(f"Unknown budget type {self.budget_type}")
