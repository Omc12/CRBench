"""
Unit tests for CRBench Tasks.
"""

import pytest
from crbench.core.registry import Registry
from crbench.tasks.niah import SingleNeedleTask, MultiNeedleTask
from crbench.tasks.ruler import RulerKVTask, RulerVariableTrackingTask
from crbench.tasks.multihop import MultiHopQATask
from crbench.tasks.longbench import LongBenchQATask


def test_task_registry():
    assert "single_niah" in Registry.list_tasks()
    assert "multi_niah" in Registry.list_tasks()
    assert "ruler_kv" in Registry.list_tasks()
    assert "ruler_variable_tracking" in Registry.list_tasks()
    assert "multihop_qa" in Registry.list_tasks()
    assert "longbench_qa" in Registry.list_tasks()


def test_single_niah_generation_and_eval():
    task = SingleNeedleTask()
    samples = task.generate_samples(context_length=1000, num_samples=2, tokenizer=None)
    assert len(samples) == 2
    
    # Correct prediction test
    s = samples[0]
    gt = s.ground_truths[0]
    eval_res = task.evaluate_prediction(prediction=gt, sample=s)
    assert eval_res.score == 1.0

    # Wrong prediction test
    eval_res_wrong = task.evaluate_prediction(prediction="completely unrelated text", sample=s)
    assert eval_res_wrong.score == 0.0


def test_ruler_kv_generation_and_eval():
    task = RulerKVTask()
    samples = task.generate_samples(context_length=1000, num_samples=2, tokenizer=None)
    assert len(samples) == 2
    
    s = samples[0]
    eval_res = task.evaluate_prediction(prediction=s.ground_truths[0], sample=s)
    assert eval_res.score == 1.0


def test_multihop_qa_eval():
    task = MultiHopQATask()
    samples = task.generate_samples(context_length=1000, num_samples=2, tokenizer=None)
    s = samples[0]
    eval_res = task.evaluate_prediction(prediction=s.ground_truths[0], sample=s)
    assert eval_res.score == 1.0
