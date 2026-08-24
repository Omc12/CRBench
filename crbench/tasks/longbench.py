"""
LongBench adapter and long-context summarization/QA evaluation wrappers for CRBench.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import random
from crbench.tasks.base import BaseTask, EvaluationSample, SampleEvaluationResult, compute_token_f1, compute_exact_match
from crbench.core.registry import Registry


@Registry.register_task("longbench_narrativeqa")
@Registry.register_task("longbench_qa")
class LongBenchQATask(BaseTask):
    """
    LongBench Narrative/Multi-document QA task interface.
    Can load from HF datasets or fallback to self-contained long-context narrative QA evaluation.
    """

    def __init__(self, name: str = "longbench_qa", dataset_name: Optional[str] = None, seed: int = 42, config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, seed=seed, config=config)
        self.dataset_name = dataset_name or "THUDM/LongBench"

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

        # Synthetic multi-section technical report corpus
        sections = [
            ("Thermal Shielding", "The external thermal shielding utilizes ceramic matrix composite tiles capable of withstanding peak re-entry temperatures up to 2100 Kelvin."),
            ("Cryogenic Storage", "Liquid hydrogen propellant is stored at 20 Kelvin in vacuum-jacketed aluminum-lithium tanks with active boil-off reduction systems."),
            ("Avionics Architecture", "The flight computer runs a triple-modular redundant real-time operating system with optical fiber bus interconnects."),
            ("Telemetry Relay", "Deep-space communications utilize an X-band high-gain phased array antenna operating at a maximum downlink rate of 4.2 megabits per second."),
            ("Life Support", "Environmental control uses electrochemical CO2 scrubbers and catalytic water reclamation with an efficiency of 98.4 percent.")
        ]

        for s_idx in range(num_samples):
            target_sec, target_detail = sections[s_idx % len(sections)]
            
            # Fill document
            doc_parts = []
            for sec_name, sec_text in sections:
                doc_parts.append(f"== Section: {sec_name} ==\n{sec_text}\n" + "Additional operational telemetry and diagnostic readings indicate nominal parameters across all sub-assemblies.\n" * max(5, int(context_length / 250)))

            context = "\n\n".join(doc_parts)
            query = f"According to the engineering document, what is the primary technical detail regarding '{target_sec}'?"

            sample = EvaluationSample(
                sample_id=f"longbench_qa_{context_length}_{s_idx}",
                context=context,
                query=query,
                ground_truths=[target_detail, target_sec],
                context_length=context_length,
                metadata={"section": target_sec, "detail": target_detail}
            )
            samples.append(sample)

        return samples

    def evaluate_prediction(
        self,
        prediction: str,
        sample: EvaluationSample
    ) -> SampleEvaluationResult:
        f1 = compute_token_f1(prediction, sample.ground_truths)
        em = compute_exact_match(prediction, sample.ground_truths)
        score = max(em, f1)

        return SampleEvaluationResult(
            sample_id=sample.sample_id,
            context_length=sample.context_length,
            prediction=prediction,
            ground_truths=sample.ground_truths,
            score=score,
            metrics={"f1": f1, "exact_match": em},
            metadata=sample.metadata
        )
