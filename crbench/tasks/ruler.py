"""
RULER-style long-context benchmark tasks: Key-Value Retrieval, Variable Tracing, and Aggregation.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import random
import uuid
from crbench.tasks.base import BaseTask, EvaluationSample, SampleEvaluationResult, compute_exact_match, compute_token_f1
from crbench.core.registry import Registry


@Registry.register_task("ruler_kv")
@Registry.register_task("ruler_retrieval")
class RulerKVTask(BaseTask):
    """
    RULER Key-Value Retrieval Task.
    Synthesizes hundreds/thousands of UUID key-value mappings in JSON/text format,
    and asks for the value associated with a specific key.
    """

    def __init__(self, name: str = "ruler_kv", seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, seed=seed, config=config)

    @property
    def floor_score(self) -> float:
        return 0.0

    def generate_samples(
        self,
        context_length: int,
        num_samples: int,
        tokenizer: Any,
        **kwargs: Any
    ) -> List[EvaluationSample]:
        rng = random.Random(self.seed + context_length)
        samples: List[EvaluationSample] = []

        for s_idx in range(num_samples):
            # Calculate number of KV pairs needed (approx 15 tokens per pair)
            num_pairs = max(20, int(context_length / 15))
            
            keys = [f"key_{rng.randint(100000, 999999)}_{i}" for i in range(num_pairs)]
            values = [f"val_{rng.randint(100000, 999999)}" for _ in range(num_pairs)]

            target_idx = rng.randint(0, num_pairs - 1)
            target_key = keys[target_idx]
            target_value = values[target_idx]

            lines = ["Here is a database of key-value records:\n"]
            for k, v in zip(keys, values):
                lines.append(f'"{k}": "{v}",')
            
            context = "\n".join(lines)
            if tokenizer is not None:
                target_len = max(100, context_length - 120)
                tokens = tokenizer.encode(context, add_special_tokens=False)
                if len(tokens) > target_len:
                    # Make sure the target key is kept in context
                    tokens = tokens[:target_len]
                    context = tokenizer.decode(tokens, skip_special_tokens=True)
                    if target_key not in context:
                        context = f'"{target_key}": "{target_value}",\n' + context

            query = f'What is the value associated with "{target_key}"? Answer with only the value string.'

            sample = EvaluationSample(
                sample_id=f"ruler_kv_{context_length}_{s_idx}",
                context=context,
                query=query,
                ground_truths=[target_value],
                context_length=context_length,
                metadata={"target_key": target_key, "target_value": target_value, "num_pairs": num_pairs}
            )
            samples.append(sample)

        return samples

    def evaluate_prediction(
        self,
        prediction: str,
        sample: EvaluationSample
    ) -> SampleEvaluationResult:
        target_value = sample.metadata.get("target_value", "")
        score = 1.0 if target_value.lower() in prediction.lower() else 0.0

        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            context_length=sample.context_length,
            prediction=prediction,
            ground_truths=sample.ground_truths,
            score=score,
            metrics={"exact_match": score},
            metadata=sample.metadata
        )


@Registry.register_task("ruler_variable_tracking")
@Registry.register_task("ruler_tracing")
class RulerVariableTrackingTask(BaseTask):
    """
    RULER Variable Tracing Task.
    Tracks variable reassignments across long contexts (e.g. X=1; ... Y=X; ... Z=Y; ... What is Z?).
    """

    def __init__(self, name: str = "ruler_variable_tracking", hops: int = 4, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, seed=seed, config=config)
        self.hops = hops

    @property
    def floor_score(self) -> float:
        return 0.0

    def generate_samples(
        self,
        context_length: int,
        num_samples: int,
        tokenizer: Any,
        **kwargs: Any
    ) -> List[EvaluationSample]:
        rng = random.Random(self.seed + context_length)
        samples: List[EvaluationSample] = []

        var_names = [f"var_{chr(ord('a') + i)}" for i in range(26)]

        for s_idx in range(num_samples):
            # Select chain of variables
            chain_vars = rng.sample(var_names, self.hops + 1)
            initial_value = rng.randint(100, 9999)

            chain_statements = []
            chain_statements.append(f"{chain_vars[0]} = {initial_value}")
            for i in range(self.hops):
                chain_statements.append(f"{chain_vars[i+1]} = {chain_vars[i]}")

            # Generate distractor statements
            num_distractors = max(30, int(context_length / 12))
            distractors = []
            for _ in range(num_distractors):
                v1, v2 = rng.sample(var_names, 2)
                if rng.random() < 0.5:
                    distractors.append(f"{v1} = {rng.randint(10, 9999)}")
                else:
                    distractors.append(f"{v1} = {v2}")

            # Merge chain into distractors at spaced intervals
            all_lines = []
            total_elements = len(distractors) + len(chain_statements)
            chain_positions = [int(total_elements * (i + 1) / (len(chain_statements) + 1)) for i in range(len(chain_statements))]

            d_idx = 0
            c_idx = 0
            for pos in range(total_elements):
                if c_idx < len(chain_positions) and pos == chain_positions[c_idx]:
                    all_lines.append(chain_statements[c_idx])
                    c_idx += 1
                elif d_idx < len(distractors):
                    all_lines.append(distractors[d_idx])
                    d_idx += 1

            context = "Consider the following sequential variable assignments:\n" + "\n".join(all_lines)
            final_var = chain_vars[-1]
            query = f"Following the sequential assignments above from top to bottom, what is the final integer value of {final_var}?"

            sample = EvaluationSample(
                sample_id=f"ruler_trace_{context_length}_{s_idx}",
                context=context,
                query=query,
                ground_truths=[str(initial_value)],
                context_length=context_length,
                metadata={"final_var": final_var, "initial_value": initial_value, "chain": chain_vars}
            )
            samples.append(sample)

        return samples

    def evaluate_prediction(
        self,
        prediction: str,
        sample: EvaluationSample
    ) -> SampleEvaluationResult:
        init_val = str(sample.metadata.get("initial_value", ""))
        score = 1.0 if init_val in prediction else 0.0

        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            context_length=sample.context_length,
            prediction=prediction,
            ground_truths=sample.ground_truths,
            score=score,
            metrics={"exact_match": score},
            metadata=sample.metadata
        )
