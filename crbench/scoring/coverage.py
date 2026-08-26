"""
Evaluation coverage (C) — CRBench Part 1, reported separately from Q and R_mem.

Why coverage is a separate dimension
------------------------------------
Q is *conditional*. It answers "of the contextual capability the dense reference
demonstrated on this query, how much did the method retain?" -- which presupposes
that a dense-anchored comparison was possible at all. When the dense baseline
cannot serve as an anchor, or the method cannot produce a result, the query drops
out of Q's denominator entirely.

That makes two very different situations print identically:

    Q = 95.0 over 2 of 6 queries
    Q = 95.0 over 6 of 6 queries

The first says "on the third of the grid where a comparison was possible, the
method retained 95%". The second says the same thing about the whole grid. Only
the second supports a claim about the method at that context length.

Coverage measures *whether the method could participate in a valid
dense-anchored evaluation*. Quality measures *how much capability it retained
given that it could*. Memory measures *what it cost*. They are three independent
axes and this module keeps them that way: nothing here alters Q, R_mem or S_res.

Dense coverage is reported alongside method coverage, because it separates two
causes of a low number. If dense coverage is also low, the benchmark could not
evaluate that context at all -- the model failed the task, and no method could
have been scored there. If dense coverage is high and method coverage is low,
the method failed where dense succeeded, which is a property of the method.

Definitions
-----------
    N_total          queries assigned to the (task, context length) cell
    dense_success    the dense reference produced a usable anchor
    method_success   the method produced a valid result (ran to completion)
    paired_success   dense_success AND method_success
    C                paired_success_count / N_total
    C_dense          dense_success_count / N_total

A query where the method ran but scored zero is a *success*: it produced a valid
result, and that result was bad. A query where the method hit OOM produced no
result at all. Conflating those would let a method improve its coverage by
crashing instead of answering badly.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

#: Statuses that mean the method produced a usable result.
SUCCESS_STATUSES = frozenset({"SUCCESS"})


def dense_anchor_usable(dense_raw_score: float, task_floor: float = 0.0) -> bool:
    """Could the dense reference serve as an anchor for this query?

    A dense score at or below the task floor carries no signal to retain: the
    normaliser divides by ``max(min_dynamic_range, dense - floor)``, so anything
    at the floor makes the denominator the guard value and turns any non-zero
    method score into an arbitrary retention figure. Those queries are excluded
    from Q, and this is the predicate that excludes them.
    """
    return float(dense_raw_score) > float(task_floor)


@dataclass
class CoverageRecord:
    """Coverage for one (task, context length, method, budget) cell."""

    task_name: str
    context_length: int
    method_name: str
    budget_spec: Any = None

    total_queries: int = 0
    dense_success_count: int = 0
    method_success_count: int = 0
    paired_success_count: int = 0

    #: Reasons the method did not produce a result, e.g. {"OOM": 2}.
    failure_statuses: Dict[str, int] = field(default_factory=dict)

    @property
    def coverage_fraction(self) -> float:
        """C = paired successes / total queries. 0.0 when the cell is empty."""
        return (self.paired_success_count / self.total_queries) if self.total_queries else 0.0

    @property
    def coverage_percent(self) -> float:
        return 100.0 * self.coverage_fraction

    @property
    def dense_coverage_fraction(self) -> float:
        """C_dense = usable dense anchors / total queries."""
        return (self.dense_success_count / self.total_queries) if self.total_queries else 0.0

    @property
    def dense_coverage_percent(self) -> float:
        return 100.0 * self.dense_coverage_fraction

    def format_compact(self) -> str:
        """``2/6, 33.3%`` — the form used inside table cells."""
        return f"{self.paired_success_count}/{self.total_queries}, {self.coverage_percent:.1f}%"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["coverage_fraction"] = self.coverage_fraction
        d["coverage_percent"] = self.coverage_percent
        d["dense_coverage_fraction"] = self.dense_coverage_fraction
        d["dense_coverage_percent"] = self.dense_coverage_percent
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CoverageRecord":
        allowed = {f for f in cls.__dataclass_fields__}          # noqa: SLF001
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def merged_with(self, other: "CoverageRecord") -> "CoverageRecord":
        """Combine two cells into a coarser aggregate (e.g. over tasks)."""
        failures = dict(self.failure_statuses)
        for k, v in other.failure_statuses.items():
            failures[k] = failures.get(k, 0) + v
        return CoverageRecord(
            task_name=self.task_name if self.task_name == other.task_name else "*",
            context_length=(self.context_length
                            if self.context_length == other.context_length else -1),
            method_name=self.method_name if self.method_name == other.method_name else "*",
            budget_spec=self.budget_spec if self.budget_spec == other.budget_spec else "*",
            total_queries=self.total_queries + other.total_queries,
            dense_success_count=self.dense_success_count + other.dense_success_count,
            method_success_count=self.method_success_count + other.method_success_count,
            paired_success_count=self.paired_success_count + other.paired_success_count,
            failure_statuses=failures,
        )


def _query_flags(q: Any) -> tuple:
    """(dense_success, method_success, status) for a query row.

    Accepts a QueryEvaluationResult or a plain dict from a results file. Explicit
    fields win; otherwise they are derived, so coverage can be recomputed from
    result files written before this metric existed.
    """
    get = (lambda k, d=None: q.get(k, d)) if isinstance(q, dict) else (lambda k, d=None: getattr(q, k, d))

    status = get("status", "SUCCESS") or "SUCCESS"

    dense = get("dense_success")
    if dense is None:
        dense = dense_anchor_usable(get("dense_raw_score", 0.0) or 0.0,
                                    get("task_floor", 0.0) or 0.0)

    method = get("method_success")
    if method is None:
        method = status in SUCCESS_STATUSES

    return bool(dense), bool(method), status


def coverage_from_queries(
    queries: Iterable[Any],
    *,
    task_name: str = "*",
    context_length: int = -1,
    method_name: str = "*",
    budget_spec: Any = None,
    total_queries: Optional[int] = None,
) -> CoverageRecord:
    """Build a CoverageRecord from query rows.

    ``total_queries`` defaults to the number of rows supplied. Pass it explicitly
    when the harness knows the cell was assigned more queries than produced rows
    -- a method that fails before scoring may emit no row at all, and those
    queries must still count toward N_total or coverage would read 100%.
    """
    rows = list(queries)
    dense_ok = method_ok = paired = 0
    failures: Dict[str, int] = {}
    for q in rows:
        d, m, status = _query_flags(q)
        dense_ok += int(d)
        method_ok += int(m)
        paired += int(d and m)
        if not m:
            failures[status] = failures.get(status, 0) + 1

    n_total = len(rows) if total_queries is None else int(total_queries)
    if n_total > len(rows):
        # Queries assigned but never scored are method failures of unknown kind.
        failures["NOT_RUN"] = failures.get("NOT_RUN", 0) + (n_total - len(rows))

    return CoverageRecord(
        task_name=task_name, context_length=context_length,
        method_name=method_name, budget_spec=budget_spec,
        total_queries=n_total,
        dense_success_count=dense_ok,
        method_success_count=method_ok,
        paired_success_count=paired,
        failure_statuses=failures,
    )


def coverage_by_cell(
    queries: Iterable[Any],
    expected_per_cell: Optional[Dict[tuple, int]] = None,
) -> List[CoverageRecord]:
    """One CoverageRecord per (task, context length, method, budget).

    ``expected_per_cell`` maps (task_name, context_length) to the number of
    queries the grid assigned there, so cells whose method emitted no rows are
    still charged for them.
    """
    buckets: Dict[tuple, List[Any]] = {}
    for q in queries:
        get = (lambda k, d=None, _q=q: _q.get(k, d)) if isinstance(q, dict) \
            else (lambda k, d=None, _q=q: getattr(_q, k, d))
        key = (get("task_name", "*"), int(get("context_length", -1) or -1),
               get("method_name", "*"), get("budget_spec"))
        buckets.setdefault(key, []).append(q)

    out: List[CoverageRecord] = []
    for (task, ctx, method, budget), rows in sorted(buckets.items(), key=lambda kv: str(kv[0])):
        expected = (expected_per_cell or {}).get((task, ctx))
        out.append(coverage_from_queries(
            rows, task_name=task, context_length=ctx, method_name=method,
            budget_spec=budget, total_queries=expected))
    return out


def roll_up(records: Sequence[CoverageRecord], *, by: Sequence[str]) -> List[CoverageRecord]:
    """Combine cells, keying on the named fields (e.g. ``("method_name",)``)."""
    merged: Dict[tuple, CoverageRecord] = {}
    for r in records:
        key = tuple(getattr(r, f) for f in by)
        merged[key] = merged[key].merged_with(r) if key in merged else r
    return list(merged.values())
