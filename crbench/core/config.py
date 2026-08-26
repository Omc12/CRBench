"""
Configuration dataclasses and YAML loading/saving for CRBench.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import yaml
import json


@dataclass
class ModelConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"  # "float16", "bfloat16", "float32"
    device: str = "auto"     # "cuda", "mps", "cpu", "auto"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = None  # "sdpa", "flash_attention_2", "eager"
    max_model_len: Optional[int] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    device_map: Optional[str] = None
    # RoPE scaling override, applied to the model config at load time.
    # Qwen2.5 ships max_position_embeddings=32768 and reaches 131072 only via
    # YaRN, e.g. {"rope_type": "yarn", "factor": 4.0,
    # "original_max_position_embeddings": 32768}.  Running past the native window
    # without it measures positional extrapolation failure, not KV compression,
    # so this must be set deliberately and is recorded in the results manifest.
    rope_scaling: Optional[Dict[str, Any]] = None
    # Attributes forced onto the HF config before the model is built. Needed for
    # checkpoints whose vendored `trust_remote_code` modeling was written against
    # an older transformers: Nanbeige4.2 reads `config.rope_scaling["type"]`, a
    # key transformers 5 renamed to `rope_type`, so it raises KeyError on a model
    # that uses no scaling at all. Setting `rope_scaling: null` selects its
    # unscaled branch, which is what its own config asks for.
    config_overrides: Dict[str, Any] = field(default_factory=dict)
    # Extra arguments for `tokenizer.apply_chat_template`. Reasoning models open
    # a chain-of-thought block in their generation prompt and only answer after
    # closing it, so with a short `max_new_tokens` the benchmark would score the
    # first 32 tokens of deliberation instead of an answer. Nanbeige4.2's
    # template ends in an open `<think>` tag and accepts `enable_thinking: false`,
    # which emits a closed, empty block so the model answers directly.
    chat_template_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskConfig:
    task_name: str
    context_lengths: List[int] = field(default_factory=lambda: [4096, 8192, 16384])
    num_samples: int = 10
    seed: int = 42
    task_kwargs: Dict[str, Any] = field(default_factory=dict)
    # Generation budget for this task, overriding ProfilerConfig.max_new_tokens.
    #
    # One global value cannot serve every task. A passkey lookup answers in a
    # few tokens; multi-step variable tracking reasons before it answers. At the
    # global 32, Qwen2.5-7B's dense output on ruler_variable_tracking was cut off
    # mid-sentence at every one of 15 queries -- "we need to follow the sequence
    # of assignments step by step and keep track of the value of `var_r" -- so
    # the task scored 0 for the dense reference and every method alike, and was
    # recorded as "the model cannot do this". It was never allowed to finish.
    #
    # None means "use the profiler value", which keeps existing configs and their
    # results exactly as they were.
    max_new_tokens: Optional[int] = None


@dataclass
class AdapterConfig:
    adapter_name: str
    adapter_type: str  # "dense", "quantized", "eviction", "merging", "compressed", "custom"
    budgets: List[Union[float, int, Dict[str, Any]]] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProfilerConfig:
    track_physical_memory: bool = True
    track_latency: bool = True
    warmup_steps: int = 1
    device_synchronize: bool = True
    # Prompt tokens fed per forward during prefill.  Bounds peak activation
    # memory independently of context length, so an OOM in the results means the
    # KV representation did not fit -- not that the prompt was fed too greedily.
    prefill_chunk_size: int = 4096
    # Generated tokens per query.  The tasks here answer in a short span; the
    # dense baseline and every method use the same value on the same query.
    max_new_tokens: int = 32
    # Release cached allocator segments between prefill chunks.  Windows has no
    # expandable_segments, and without this the allocator reserved 13.00 GiB for
    # 8.69 GiB of live tensors at 65536 tokens and spilled into host memory.
    empty_cache_between_chunks: bool = True


@dataclass
class ScoringConfig:
    standard_budgets_bpt: List[float] = field(default_factory=lambda: [2.0, 4.0, 8.0, 16.0])
    standard_budgets_gb: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 4.0, 8.0])
    auqc_log_scale: bool = True
    context_weighting: str = "logarithmic"   # "logarithmic", "uniform", "linear"
    min_dynamic_range: float = 0.05          # Delta_min = 5% minimum dynamic range for relative scoring
    reference_dense_name: str = "dense_fp16"
    floor_quality: float = 0.0
    bootstrap_samples: int = 1000
    ci_level: float = 0.95
    utility_formula: str = "linear"          # "linear", "cobb_douglas", "harmonic", "power_mean_2", "logarithmic", "gated_linear"
    utility_alpha: float = 0.70              # Quality weight α ∈ [0.10, 0.90]
    resource_normalization_max: float = 100.0
    enable_part2: bool = False               # False = Part 1 only (Quality + Memory); True = Part 2 (Quality + Memory + Runtime)


@dataclass
class BenchmarkConfig:
    benchmark_name: str = "crbench_experiment"
    model: ModelConfig = field(default_factory=ModelConfig)
    tasks: List[TaskConfig] = field(default_factory=list)
    adapters: List[AdapterConfig] = field(default_factory=list)
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    output_dir: str = "results"
    save_plots: bool = True
    save_raw_json: bool = True

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> BenchmarkConfig:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BenchmarkConfig:
        model = ModelConfig(**data.get("model", {})) if "model" in data else ModelConfig()
        tasks = [TaskConfig(**t) for t in data.get("tasks", [])]
        adapters = [AdapterConfig(**a) for a in data.get("adapters", [])]
        profiler = ProfilerConfig(**data.get("profiler", {})) if "profiler" in data else ProfilerConfig()
        scoring = ScoringConfig(**data.get("scoring", {})) if "scoring" in data else ScoringConfig()
        
        return cls(
            benchmark_name=data.get("benchmark_name", "crbench_experiment"),
            model=model,
            tasks=tasks,
            adapters=adapters,
            profiler=profiler,
            scoring=scoring,
            output_dir=data.get("output_dir", "results"),
            save_plots=data.get("save_plots", True),
            save_raw_json=data.get("save_raw_json", True),
        )

    def to_yaml(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
