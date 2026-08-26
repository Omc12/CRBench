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

ALPHA = 0.70


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def collect(sweeps, runtimes):
    """cells[task][method][ctx] = list of (Q, R); dense_all[task][ctx] = raw scores."""
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    dense_all = defaultdict(lambda: defaultdict(list))
    for path in sweeps:
        for q in json.load(open(path, encoding="utf-8"))["query_results"]:
            if q["method_name"] == "dkv":
                continue
            if q["method_name"] == "dense_fp16":
                dense_all[q["task_name"]][q["context_length"]].append(q["dense_raw_score"])
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
    return cells, dense_all


def main():
    args = sys.argv[1:]
    runtimes = [a for a in args if "runtime" in a] or ["results/dkv_runtime_7b.json"]
    sweeps = [a for a in args if a not in runtimes] or [
        "results/bench_7b_native/raw_results_v1.json",
        "results/bench_7b_dkv/raw_results_v1.json"]

    cells, dense_all = collect(sweeps, runtimes)
    all_tasks = sorted(dense_all)
    ctxs = sorted({c for t in dense_all.values() for c in t})
    methods = sorted({m for t in cells.values() for m in t})

    print("CRBench Part 1 = 0.70*Q + 0.30*R, per task / method / context length.")
    print("[n] is the number of queries the dense baseline could pose there.\n")

    for task in all_tasks:
        usable = {c: sum(1 for x in dense_all[task][c] if x > 0) for c in ctxs}
        total = {c: len(dense_all[task][c]) for c in ctxs}
        if not any(usable.values()):
            print(f"=== {task} ===")
            print(f"    EXCLUDED ENTIRELY: the dense baseline scored 0 at every context")
            print(f"    length ({'  '.join(f'{c//1024}K: 0/{total[c]}' for c in ctxs)}).")
            print(f"    The model cannot perform this task, so no method can be scored")
            print(f"    against it. Reported, not silently dropped.\n")
            continue

        print(f"=== {task} ===")
        print(f"{'method':<22}" + "".join(f"{c // 1024:>12}K" for c in ctxs))
        print("-" * (22 + 13 * len(ctxs)))
        for m in methods:
            if m not in cells[task]:
                continue
            row = ""
            for c in ctxs:
                v = cells[task][m].get(c)
                if not v:
                    row += f"{'--':>13}"
                    continue
                s = ALPHA * mean([x[0] for x in v]) + (1 - ALPHA) * mean([x[1] for x in v])
                row += f"{s:>9.1f}[{len(v)}]"
            print(f"{m:<22}{row}")
        print(f"{'dense anchor usable':<22}" +
              "".join(f"{str(usable[c]) + '/' + str(total[c]):>13}" for c in ctxs))
        print()


if __name__ == "__main__":
    main()
