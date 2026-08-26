"""The complete per-task, per-context, per-method CRBench matrix.

Every cell carries its query count. That is not decoration: queries the dense
baseline itself failed are excluded, so a cell's n says how much of the grid
survived to be comparable there, and several cells fall to n=1. A cell of n=1
moves 33 points on a single answer.

Tasks the model cannot perform at all are reported explicitly rather than
vanishing. Qwen2.5-7B scores zero on ruler_variable_tracking at every context
length, so every one of its queries is excluded and the task contributes nothing
-- which is a fact about the model, not an absence of data, and belongs in the
results either way.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

from crbench.scoring.coverage import coverage_from_queries, dense_anchor_usable

ALPHA = 0.70


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def collect(sweeps, runtimes):
    """cells[task][method][ctx] = list of (Q, R); dense_all[task][ctx] = raw scores."""
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dense_all = defaultdict(lambda: defaultdict(dict))     # query_id -> score
    # Every row, including ones excluded from Q, so coverage has a denominator.
    rows_all = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for path in sweeps:
        for q in json.load(open(path, encoding="utf-8"))["query_results"]:
            if q["method_name"] == "dkv":
                continue
            if q["method_name"] == "dense_fp16":
                # Keyed by query_id: the same dense query appears once per merged
                # result file, and counting rows would inflate N_total by that factor.
                dense_all[q["task_name"]][q["context_length"]][q["query_id"]] = q["dense_raw_score"]
            rows_all[q["task_name"]][f'{q["method_name"]}@{q["budget_spec"]}'][
                q["context_length"]][q["query_id"]] = q
            if q["dense_raw_score"] <= 0.0:
                continue
            key = f'{q["method_name"]}@{q["budget_spec"]}'
            cells[q["task_name"]][key][q["context_length"]].append(
                (q["quality_retained_pct"], q["resource_efficiency"]))
    for path in runtimes:
        for r in json.load(open(path, encoding="utf-8"))["records"]:
            if r["status"] != "SUCCESS" or r["dense_raw_score"] <= 0.0:
                continue
            cells[r["task_name"]][r["method_name"]][r["context_length"]].append(
                (100.0 * min(1.0, r["method_raw_score"] / r["dense_raw_score"]),
                 100.0 * max(0.0, 1.0 - r["method_effective_bpt"] / 16.0)))
    for path in runtimes:
        for r in json.load(open(path, encoding="utf-8"))["records"]:
            rows_all[r["task_name"]][r["method_name"]][r["context_length"]][r["query_id"]] = {
                "dense_raw_score": r["dense_raw_score"], "status": r["status"]}
    return cells, dense_all, rows_all


def main():
    args = sys.argv[1:]
    runtimes = [a for a in args if "runtime" in a] or ["results/dkv_runtime_7b.json"]
    sweeps = [a for a in args if a not in runtimes] or [
        "results/bench_7b_native/raw_results_v1.json",
        "results/bench_7b_dkv/raw_results_v1.json"]

    cells, dense_all, rows_all = collect(sweeps, runtimes)
    all_tasks = sorted(dense_all)
    ctxs = sorted({c for t in dense_all.values() for c in t})
    methods = sorted({m for t in cells.values() for m in t})

    print("CRBench Part 1 = 0.70*Q + 0.30*R, per task / method / context length.")
    print("Each cell:  Part1 [paired_success/N_total, coverage%]")
    print()
    print("  C = paired successes / queries assigned to that cell. A paired")
    print("  success needs BOTH a usable dense anchor AND a valid method result.")
    print("  C sits beside Part 1 and is never folded into it: Part 1 is")
    print("  conditional on a comparison being possible, C says how often it was.")
    print("  92.5 [2/3, 66.7%] and 92.5 [3/3, 100.0%] are not the same claim.")
    print("  C_dense is the same ratio for the dense reference: a low C with a")
    print("  low C_dense means the benchmark could not evaluate there at all;")
    print("  a low C with a high C_dense means the method failed where dense did not.")
    print()

    for task in all_tasks:
        usable = {c: sum(1 for x in dense_all[task][c].values() if x > 0) for c in ctxs}
        total = {c: len(dense_all[task][c]) for c in ctxs}
        if not any(usable.values()):
            print(f"=== {task} ===")
            print(f"    EXCLUDED ENTIRELY: the dense baseline scored 0 at every context")
            print(f"    length ({'  '.join(f'{c//1024}K: 0/{total[c]}' for c in ctxs)}).")
            print(f"    The model cannot perform this task, so no method can be scored")
            print(f"    against it. Reported, not silently dropped.\n")
            continue

        print(f"=== {task} ===")
        print(f"{'method':<22}" + "".join(f"{c // 1024:>21}K" for c in ctxs))
        print("-" * (22 + 22 * len(ctxs)))
        for m in methods:
            if m not in cells[task]:
                continue
            row = ""
            for c in ctxs:
                v = cells[task][m].get(c)
                rows = list(rows_all[task][m].get(c, {}).values())
                n_total = total[c]
                cov = coverage_from_queries(rows, total_queries=n_total) if rows else None
                if not v:
                    row += f"{'--':>22}"
                    continue
                s = ALPHA * mean([x[0] for x in v]) + (1 - ALPHA) * mean([x[1] for x in v])
                tag = cov.format_compact() if cov else f"{len(v)}/{n_total}"
                row += f"{s:>8.1f} [{tag:>11}]"
            print(f"{m:<22}{row}")
        print(f"{'dense anchor C_dense':<22}" +
              "".join(f"{str(usable[c]) + '/' + str(total[c]) + ', ' + format(100.0 * usable[c] / total[c] if total[c] else 0.0, '.1f') + '%':>22}"
                      for c in ctxs))
        print()


if __name__ == "__main__":
    main()
