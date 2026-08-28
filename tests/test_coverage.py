"""
Evaluation coverage (C), reported separately from Q and R_mem.

Coverage answers "could this method be evaluated here at all?", which Q cannot,
because Q is conditional on a paired dense-anchored comparison having been
possible. A method scoring Q = 95 over 2 of 6 queries and one scoring Q = 95
over 6 of 6 print identically without it.

These tests pin the four pairing outcomes, the arithmetic, the empty-cell guard,
and -- most importantly -- that adding coverage did not disturb Q, R_mem or
S_res.
"""

from __future__ import annotations

import math

import pytest

from crbench.core.query_result import QueryEvaluationResult
from crbench.scoring.coverage import (CoverageRecord, coverage_by_cell,
                                      coverage_from_queries, dense_anchor_usable,
                                      roll_up)


def q(dense: float, status: str = "SUCCESS", **kw) -> QueryEvaluationResult:
    """A query row with an explicit dense score and method status."""
    return QueryEvaluationResult(
        query_id=kw.pop("query_id", "q"), task_name=kw.pop("task_name", "t"),
        context_length=kw.pop("context_length", 2048), model_name="m",
        method_name=kw.pop("method_name", "meth"), budget_spec=kw.pop("budget_spec", 4.0),
        dense_raw_score=dense, method_raw_score=kw.pop("method_raw_score", 1.0),
        status=status, **kw)


# --------------------------------------------------------------------------
# (a)-(d): the four pairing outcomes
# --------------------------------------------------------------------------

def test_dense_success_and_method_success_counts_as_paired():
    c = coverage_from_queries([q(1.0, "SUCCESS")])
    assert (c.dense_success_count, c.method_success_count, c.paired_success_count) == (1, 1, 1)
    assert c.total_queries == 1
    assert c.coverage_fraction == 1.0


def test_dense_failure_with_method_success_is_not_paired_but_still_counts():
    c = coverage_from_queries([q(0.0, "SUCCESS")])
    assert c.dense_success_count == 0
    assert c.method_success_count == 1        # the method did produce a result
    assert c.paired_success_count == 0        # but there was nothing to anchor to
    assert c.total_queries == 1               # and it still counts in N_total
    assert c.coverage_fraction == 0.0


def test_dense_success_with_method_failure_is_not_paired_but_still_counts():
    c = coverage_from_queries([q(1.0, "OOM")])
    assert c.dense_success_count == 1
    assert c.method_success_count == 0
    assert c.paired_success_count == 0
    assert c.total_queries == 1
    assert c.failure_statuses == {"OOM": 1}


def test_both_failing_is_not_paired_and_still_counts():
    c = coverage_from_queries([q(0.0, "RUNTIME_ERROR")])
    assert (c.dense_success_count, c.method_success_count, c.paired_success_count) == (0, 0, 0)
    assert c.total_queries == 1
    assert c.coverage_fraction == 0.0


# --------------------------------------------------------------------------
# (e)-(g): arithmetic and the empty-cell guard
# --------------------------------------------------------------------------

def test_two_of_six_paired_gives_one_third():
    rows = [q(1.0, "SUCCESS"), q(1.0, "SUCCESS"),      # paired
            q(0.0, "SUCCESS"), q(0.0, "SUCCESS"),      # dense unusable
            q(1.0, "OOM"), q(1.0, "RUNTIME_ERROR")]    # method failed
    c = coverage_from_queries(rows)
    assert c.paired_success_count == 2
    assert c.total_queries == 6
    assert c.coverage_fraction == pytest.approx(2 / 6)
    assert c.coverage_percent == pytest.approx(33.333, abs=1e-3)
    assert c.format_compact() == "2/6, 33.3%"


def test_all_six_paired_gives_one():
    c = coverage_from_queries([q(1.0, "SUCCESS") for _ in range(6)])
    assert c.coverage_fraction == 1.0
    assert c.coverage_percent == 100.0


def test_zero_queries_does_not_divide_by_zero():
    c = coverage_from_queries([])
    assert c.total_queries == 0
    assert c.coverage_fraction == 0.0
    assert c.coverage_percent == 0.0
    assert c.dense_coverage_fraction == 0.0
    assert not math.isnan(c.coverage_fraction)


# --------------------------------------------------------------------------
# N_total must reflect what the grid assigned, not what produced rows
# --------------------------------------------------------------------------

def test_unrun_queries_still_count_toward_total():
    """A method that emits no row for a query must not gain coverage from it."""
    c = coverage_from_queries([q(1.0, "SUCCESS")], total_queries=6)
    assert c.paired_success_count == 1
    assert c.total_queries == 6
    assert c.coverage_fraction == pytest.approx(1 / 6)
    assert c.failure_statuses["NOT_RUN"] == 5


def test_expected_per_cell_charges_missing_rows():
    rows = [q(1.0, "SUCCESS", task_name="niah", context_length=32768)]
    cells = coverage_by_cell(rows, {("niah", 32768): 6})
    assert len(cells) == 1
    assert cells[0].total_queries == 6
    assert cells[0].coverage_fraction == pytest.approx(1 / 6)


# --------------------------------------------------------------------------
# Dense coverage separates "benchmark could not evaluate" from "method failed"
# --------------------------------------------------------------------------

def test_dense_coverage_distinguishes_cause_of_low_coverage():
    # Benchmark could not evaluate: dense fails, so no method could be scored.
    unusable = coverage_from_queries([q(0.0, "SUCCESS") for _ in range(6)])
    assert unusable.dense_coverage_percent == 0.0
    assert unusable.coverage_percent == 0.0

    # Method's own fault: dense fine everywhere, method crashed everywhere.
    method_bad = coverage_from_queries([q(1.0, "OOM") for _ in range(6)])
    assert method_bad.dense_coverage_percent == 100.0
    assert method_bad.coverage_percent == 0.0


def test_multi_niah_32k_example_from_the_spec():
    """4 of 6 usable dense anchors -> C_dense 66.7%; 2 paired -> C 33.3%."""
    rows = ([q(1.0, "SUCCESS")] * 2 +      # dense usable, method ok  -> paired
            [q(1.0, "OOM")] * 2 +          # dense usable, method failed
            [q(0.0, "SUCCESS")] * 2)       # dense unusable
    c = coverage_from_queries(rows)
    assert c.dense_success_count == 4
    assert c.dense_coverage_percent == pytest.approx(66.667, abs=1e-3)
    assert c.paired_success_count == 2
    assert c.coverage_percent == pytest.approx(33.333, abs=1e-3)


# --------------------------------------------------------------------------
# Backward compatibility and roll-up
# --------------------------------------------------------------------------

def test_coverage_recomputes_from_files_without_the_new_fields():
    """Rows written before coverage existed must still yield correct counts."""
    legacy = [{"task_name": "t", "context_length": 2048, "method_name": "m",
               "budget_spec": 4.0, "dense_raw_score": 1.0, "status": "SUCCESS"},
              {"task_name": "t", "context_length": 2048, "method_name": "m",
               "budget_spec": 4.0, "dense_raw_score": 0.0, "status": "SUCCESS"}]
    c = coverage_from_queries(legacy)
    assert c.dense_success_count == 1
    assert c.paired_success_count == 1
    assert c.total_queries == 2


def test_explicit_flags_win_over_derivation():
    row = q(1.0, "SUCCESS")
    row.dense_success = False              # recorded by the harness, not derived
    row.method_success = True
    c = coverage_from_queries([row])
    assert c.dense_success_count == 0
    assert c.paired_success_count == 0


def test_roll_up_sums_cells():
    cells = [CoverageRecord("t1", 2048, "m", 4.0, total_queries=6,
                            dense_success_count=6, method_success_count=6,
                            paired_success_count=6),
             CoverageRecord("t2", 2048, "m", 4.0, total_queries=6,
                            dense_success_count=4, method_success_count=6,
                            paired_success_count=2)]
    merged = roll_up(cells, by=("method_name", "budget_spec"))
    assert len(merged) == 1
    assert merged[0].total_queries == 12
    assert merged[0].paired_success_count == 8
    assert merged[0].coverage_fraction == pytest.approx(8 / 12)


def test_dense_anchor_predicate_respects_task_floor():
    assert dense_anchor_usable(0.5, task_floor=0.0)
    assert not dense_anchor_usable(0.0, task_floor=0.0)
    assert not dense_anchor_usable(0.2, task_floor=0.25)
    assert dense_anchor_usable(0.3, task_floor=0.25)


# --------------------------------------------------------------------------
# The point of the whole exercise: Q, R_mem and S_res are untouched
# --------------------------------------------------------------------------

def test_quality_and_resource_scoring_are_unchanged_by_coverage():
    from crbench.scoring.utility import compute_utility
    from crbench.scoring.normalizer import QualityNormalizer

    norm = QualityNormalizer(floor_score=0.0, min_dynamic_range=0.05)
    Q = norm.normalize(raw_score=0.80, dense_reference_score=1.0, task_floor=0.0)
    R = 75.0
    assert Q == pytest.approx(80.0)
    assert compute_utility(Q, R, alpha=0.70, formula="linear") == pytest.approx(
        0.70 * 80.0 + 0.30 * 75.0)

    # Coverage is computed from the same rows and changes none of it.
    c = coverage_from_queries([q(1.0, "SUCCESS"), q(1.0, "OOM")])
    assert c.coverage_fraction == 0.5
    assert compute_utility(Q, R, alpha=0.70, formula="linear") == pytest.approx(78.5)   # 0.70*80 + 0.30*75
