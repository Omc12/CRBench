"""
Needle In A Haystack (NIAH) tasks: Single-Needle and Multi-Needle evaluations for CRBench.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import random
from crbench.tasks.base import BaseTask, EvaluationSample, SampleEvaluationResult, compute_exact_match, compute_token_f1
from crbench.core.registry import Registry


HAYSTACK_BASE_TEXT = (
    "The atmospheric conditions on Mars have been a subject of intense scientific inquiry for decades. "
    "Spectroscopic measurements from orbiting probes indicate that the Martian atmosphere is composed primarily "
    "of carbon dioxide, with minor traces of molecular nitrogen and argon. Surface temperatures vary widely "
    "between equatorial midday and polar winter nights, often dropping below 140 Kelvin in shaded craters. "
    "Recent robotic exploration missions have uncovered sedimentary structures suggesting ancient fluvial activity, "
    "though current surface water is almost entirely locked in cryospheric permafrost or polar ice caps. "
    "Furthermore, seasonal variations in methane concentrations continue to spark vigorous debate regarding potential "
    "subsurface geochemical or even biological processes. Planetary scientists continue to analyze data from rover "
    "instruments to better understand the geological history and climate evolution of the Red Planet. "
)


@Registry.register_task("single_niah")
@Registry.register_task("niah")
class SingleNeedleTask(BaseTask):
    """
    Standard Single-Needle In A Haystack Retrieval Task.
    Inserts a secret key/passkey at a controlled depth in a long context.
    """

    def __init__(self, name: str = "single_niah", seed: int = 42, config: Optional[Dict[str, Any]] = None):
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

        # Standard needle templates
        cities = ["Zurich", "Tokyo", "Reykjavik", "Kyoto", "Valparaíso", "Auckland", "Singapore", "Oslo", "Munich", "Calgary"]
        activities = ["eating gelato", "drinking hot chocolate", "studying quantum physics", "observing migrating whales", "programming neural networks", "designing benchmarks"]

        for i in range(num_samples):
            depth = (i + 0.5) / max(1, num_samples)  # Evenly spread across 0% to 100% depth
            city = cities[i % len(cities)]
            activity = activities[i % len(activities)]
            passkey = f"{rng.randint(100000, 999999)}"

            needle = f"Special secret information: The best thing to do in {city} is {activity} with passkey {passkey}."
            query = f"Based on the text above, what is the best thing to do in {city} and what is the passkey?"
            ground_truths = [f"{activity} with passkey {passkey}", f"{activity}", passkey]

            # Construct background haystack
            haystack_chunk = HAYSTACK_BASE_TEXT
            if tokenizer is not None:
                chunk_tokens = len(tokenizer.encode(haystack_chunk, add_special_tokens=False))
                needed_chunks = max(2, int(context_length / max(1, chunk_tokens)))
            else:
                chunk_words = len(haystack_chunk.split())
                needed_chunks = max(2, int(context_length / max(1, chunk_words)))

            haystack = [haystack_chunk for _ in range(needed_chunks)]
            insert_idx = int((len(haystack) - 1) * depth)
            insert_idx = max(0, min(insert_idx, len(haystack) - 1))
            haystack[insert_idx] = haystack[insert_idx] + f"\n{needle}\n"

            full_context = "\n".join(haystack)

            # Ensure prompt fits context length while guaranteeing needle and query are included
            if tokenizer is not None:
                target_len = max(100, context_length - 150)
                tokens = tokenizer.encode(full_context, add_special_tokens=False)
                if len(tokens) > target_len:
                    tokens = tokens[:target_len]
                    full_context = tokenizer.decode(tokens, skip_special_tokens=True)
                    if passkey not in full_context:
                        full_context = full_context[:len(full_context)//2] + f"\n{needle}\n" + full_context[len(full_context)//2:]

            sample = EvaluationSample(
                sample_id=f"niah_{context_length}_{i}",
                context=full_context,
                query=query,
                ground_truths=ground_truths,
                context_length=context_length,
                metadata={"depth": depth, "city": city, "passkey": passkey, "activity": activity}
            )
            samples.append(sample)

        return samples

    def evaluate_prediction(
        self,
        prediction: str,
        sample: EvaluationSample
    ) -> SampleEvaluationResult:
        passkey = sample.metadata.get("passkey", "")
        activity = sample.metadata.get("activity", "")
        
        pred_norm = prediction.lower()
        passkey_found = passkey in prediction
        activity_found = activity.lower() in pred_norm

        if passkey_found and activity_found:
            score = 1.0
        elif passkey_found or activity_found:
            score = 0.5
        else:
            # Fallback to token F1 / EM
            em = compute_exact_match(prediction, sample.ground_truths)
            f1 = compute_token_f1(prediction, sample.ground_truths)
            score = max(em, f1)

        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            context_length=sample.context_length,
            prediction=prediction,
            ground_truths=sample.ground_truths,
            score=score,
            metrics={"exact_match": 1.0 if score == 1.0 else 0.0, "passkey_found": 1.0 if passkey_found else 0.0},
            metadata=sample.metadata
        )


@Registry.register_task("multi_niah")
class MultiNeedleTask(BaseTask):
    """
    Multi-Needle Retrieval Task.
    Inserts multiple distinct needles across the context and asks for cross-needle retrieval.
    """

    def __init__(self, name: str = "multi_niah", num_needles: int = 3, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, seed=seed, config=config)
        self.num_needles = num_needles

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

        items = [
            ("alpha", "Golden Compass"),
            ("beta", "Silver Key"),
            ("gamma", "Emerald Tablet"),
            ("delta", "Obsidian Mirror"),
            ("epsilon", "Sapphire Chalice"),
            ("zeta", "Ruby Pendant"),
        ]

        for s_idx in range(num_samples):
            chosen = rng.sample(items, min(self.num_needles, len(items)))
            needles = []
            queries_parts = []
            ground_truths_parts = []

            for code, artifact in chosen:
                needles.append(f"Security Log: Code designation '{code}' is linked to artifact '{artifact}'.")
                queries_parts.append(code)
                ground_truths_parts.append(artifact)

            haystack_chunk = HAYSTACK_BASE_TEXT
            needed_chunks = max(4, int(context_length / 60))
            haystack = [haystack_chunk for _ in range(needed_chunks)]

            # Scatter needles evenly
            for n_idx, needle_str in enumerate(needles):
                pos = int(len(haystack) * (n_idx + 1) / (len(needles) + 1))
                haystack.insert(pos, f"\n{needle_str}\n")

            full_context = "\n".join(haystack)
            if tokenizer is not None:
                target_len = max(100, context_length - 150)
                tokens = tokenizer.encode(full_context, add_special_tokens=False)
                if len(tokens) > target_len:
                    tokens = tokens[:target_len]
                    full_context = tokenizer.decode(tokens, skip_special_tokens=True)

            query = f"Identify the artifacts associated with each of the following code designations: {', '.join(queries_parts)}."
            gt_combined = ", ".join(f"{c}: {a}" for c, a in chosen)

            sample = EvaluationSample(
                sample_id=f"multi_niah_{context_length}_{s_idx}",
                context=full_context,
                query=query,
                ground_truths=[gt_combined] + ground_truths_parts,
                context_length=context_length,
                metadata={"chosen": chosen}
            )
            samples.append(sample)

        return samples

    def evaluate_prediction(
        self,
        prediction: str,
        sample: EvaluationSample
    ) -> SampleEvaluationResult:
        chosen = sample.metadata.get("chosen", [])
        correct = 0
        pred_norm = prediction.lower()
        for code, artifact in chosen:
            if artifact.lower() in pred_norm:
                correct += 1
        
        score = float(correct / max(1, len(chosen)))
        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            context_length=sample.context_length,
            prediction=prediction,
            ground_truths=sample.ground_truths,
            score=score,
            metrics={"retrieved_fraction": score},
            metadata=sample.metadata
        )
