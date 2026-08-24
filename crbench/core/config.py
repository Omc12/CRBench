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


@dataclass
class TaskConfig:
    task_name: str
    context_lengths: List[int] = field(default_factory=lambda: [4096, 8192, 16384])
    num_samples: int = 10
    seed: int = 42
    task_kwargs: Dict[str, Any] = field(default_factory=dict)


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
