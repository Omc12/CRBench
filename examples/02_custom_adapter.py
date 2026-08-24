"""
Example 02: Implementing and Benchmarking a Custom Context Representation in CRBench.
"""

from typing import Any, Dict, Optional
import torch
from crbench.core.adapter import BaseContextAdapter, KVStateMetadata
from crbench.core.budget import ContextBudget
from crbench.core.registry import Registry
from crbench.core.config import BenchmarkConfig, TaskConfig, AdapterConfig
from crbench.core.runner import BenchmarkRunner


@Registry.register_adapter("my_novel_kv_compressor")
class NovelKVCompressorAdapter(BaseContextAdapter):
    """
    Example of a novel KV compressor that halves the key dimension and quantizes values.
    """

    def __init__(self, name: str = "my_novel_kv_compressor", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, config=config)
        self.effective_bpe = 4.0  # 4 bits per element target

    @property
    def method_type(self) -> str:
        return "custom"

    def apply_budget(self, budget: ContextBudget, context_length: int) -> None:
        super().apply_budget(budget, context_length)
        # Configure compressor parameters to fit the budget
        self.effective_bpe = budget.to_bits_per_token(32, 32, 128, context_length)

    def forward_or_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        **kwargs: Any
    ) -> torch.Tensor:
        # Plug in custom PyTorch/Triton forward pass or model.generate call
        if self.model is not None:
            return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)
        return input_ids

    def get_kv_metadata(self, context_length: int) -> KVStateMetadata:
        num_layers = 32
        num_kv_heads = 32
        head_dim = 128
        total_elements = 2 * num_layers * num_kv_heads * head_dim * context_length
        algorithmic_bytes = (total_elements * self.effective_bpe) / 8.0

        return KVStateMetadata(
            adapter_name=self.name,
            method_type=self.method_type,
            effective_bits_per_element=self.effective_bpe,
            total_tokens_stored=context_length,
            context_length=context_length,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            algorithmic_bytes=algorithmic_bytes
        )


def main():
    print("[*] Testing custom adapter registration...")
    adapter_cls = Registry.get_adapter("my_novel_kv_compressor")
    print(f"[✓] Retrieved registered adapter class: {adapter_cls.__name__}")

    config = BenchmarkConfig(
        benchmark_name="custom_adapter_evaluation",
        tasks=[TaskConfig(task_name="single_niah", context_lengths=[8192], num_samples=3)],
        adapters=[
            AdapterConfig(adapter_name="dense_fp16", adapter_type="dense", budgets=[16.0]),
            AdapterConfig(adapter_name="my_novel_kv_compressor", adapter_type="my_novel_kv_compressor", budgets=[4.0]),
        ],
        output_dir="results/custom_adapter_demo"
    )

    runner = BenchmarkRunner(config)
    runner.load_model()
    results = runner.run()
    print("[✓] Custom adapter evaluated successfully!")


if __name__ == "__main__":
    main()
