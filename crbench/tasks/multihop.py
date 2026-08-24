"""
Multi-Hop Question Answering tasks over long contexts for CRBench.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import random
from crbench.tasks.base import BaseTask, EvaluationSample, SampleEvaluationResult, compute_token_f1, compute_exact_match
from crbench.core.registry import Registry


@Registry.register_task("multihop_qa")
@Registry.register_task("multihop")
class MultiHopQATask(BaseTask):
    """
    Multi-Hop Question Answering task.
    Requires synthesizing information from 2 or more distinct paragraphs across the context.
    """

    def __init__(self, name: str = "multihop_qa", seed: int = 42, config: Optional[Dict[str, Any]] = None):
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

        # Synthetic multi-hop relation database
        entities = [
            {"person": "Dr. Elena Vance", "field": "Astrobiology", "institution": "Orion Institute", "city": "Geneva", "founded": "1984"},
            {"person": "Prof. Marcus Thorne", "field": "Quantum Cryptography", "institution": "Helios Laboratory", "city": "Stockholm", "founded": "1972"},
            {"person": "Dr. Amara Patel", "field": "Neural Dynamics", "institution": "Apex Research Facility", "city": "Toronto", "founded": "1999"},
            {"person": "Prof. Jin Woo", "field": "Topological Matter", "institution": "Solstice Center", "city": "Seoul", "founded": "2008"},
            {"person": "Dr. Nadia Rossi", "field": "Atmospheric Chemistry", "institution": "Terra Observatory", "city": "Florence", "founded": "1965"},
        ]

        for s_idx in range(num_samples):
            e = entities[s_idx % len(entities)]
            
            p1 = f"Paragraph A: {e['person']} is a world-renowned pioneer in {e['field']}. In their early career, they were appointed as the lead director of {e['institution']}."
            p2 = f"Paragraph B: The {e['institution']} is located in {e['city']}. The institution was officially established in {e['founded']} to advance groundbreaking scientific discoveries."

            distractor_topics = [
                "Deep-sea hydrothermal vents harbor diverse chemotrophic ecosystems that thrive independently of solar radiation.",
                "High-temperature superconductivity in cuprate materials continues to pose profound theoretical challenges for condensed matter physics.",
                "Autonomous navigation systems rely on tightly integrated sensor fusion involving LiDAR, radar, and visual odometry.",
                "Bioluminescent fungi in temperate rainforests utilize luciferin-luciferase enzymatic pathways to emit steady green light.",
                "Gravitational wave interferometers detect minute spacetime distortions produced by binary black hole inspirals."
            ]

            num_distractors = max(10, int(context_length / 40))
            distractor_paragraphs = []
            for d_idx in range(num_distractors):
                topic = distractor_topics[d_idx % len(distractor_topics)]
                distractor_paragraphs.append(f"Document {d_idx + 1}: {topic} Researchers note that empirical validation requires extensive calibration across multi-spectral sensors.")

            # Place p1 and p2 at distinct positions (e.g. 20% and 75% depth)
            all_paras = list(distractor_paragraphs)
            pos1 = int(len(all_paras) * 0.2)
            pos2 = int(len(all_paras) * 0.75)
            all_paras.insert(pos1, p1)
            all_paras.insert(pos2, p2)

            context = "\n\n".join(all_paras)
            query = f"In which city is the institution directed by {e['person']} located?"
            ground_truth = e["city"]

            sample = EvaluationSample(
                sample_id=f"multihop_{context_length}_{s_idx}",
                context=context,
                query=query,
                ground_truths=[ground_truth],
                context_length=context_length,
                metadata={"entity": e, "target_city": ground_truth}
            )
            samples.append(sample)

        return samples

    def evaluate_prediction(
        self,
        prediction: str,
        sample: EvaluationSample
    ) -> SampleEvaluationResult:
        target_city = sample.metadata.get("target_city", "")
        f1 = compute_token_f1(prediction, sample.ground_truths)
        em = compute_exact_match(prediction, sample.ground_truths)
        score = 1.0 if target_city.lower() in prediction.lower() else max(em, f1)

        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            context_length=sample.context_length,
            prediction=prediction,
            ground_truths=sample.ground_truths,
            score=score,
            metrics={"f1": f1, "exact_match": em},
            metadata=sample.metadata
        )
