"""Per-context-length CRBench tables.

A single mean per method hides what a long-context benchmark exists to measure.
These tables report Part 1, quality and achieved b_eff at each context length.

They also report ``n`` per cell, which is not decoration. Queries the dense
baseline itself failed are excluded -- there is no retained capability to measure
where the model never had the capability -- and dense fails more as context
grows. On Qwen2.5-7B the usable query count falls from 12 of 15 at 8192 tokens
to 6 of 15 at 32768. So a column at 32768 is scored over the subset dense still
answered, which is the easier subset, and a method can appear to *improve* with
context purely because the set it is graded on got easier.

A fixed cohort would fix that, but there is not enough data for one: only 4 of 12
(task, sample) slots survive at every length on this grid. The workable answer is
to report n, refuse to read a trend across columns whose n differs, and collect
more samples per cell -- 3 quantises every task score to {0, 33, 67, 100}, which
is coarse enough that a single query flips a cell by 33 points.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

ALPHA = 0.70


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def load(sweeps, runtimes):
    per = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sweeps:
        for q in json.load(open(path, encoding="utf-8"))["query_results"]:
            if q["method_name"] == "dkv" or q["dense_raw_score"] <= 0.0:
                continue
            k = f'{q["method_name"]}@{q["budget_spec"]}'
            cell = per[k][q["context_length"]]
            cell["Q"].append(q["quality_retained_pct"])
            cell["R"].append(q["resource_efficiency"])
            cell["b"].append(q["method_effective_bpt"])
    for path in runtimes:
        for r in json.load(open(path, encoding="utf-8"))["records"]:
            if r["status"] != "SUCCESS" or r["dense_raw_score"] <= 0.0:
                continue
            cell = per[r["method_name"]][r["context_length"]]
            cell["Q"].append(100.0 * min(1.0, r["method_raw_score"] / r["dense_raw_score"]))
            cell["R"].append(100.0 * max(0.0, 1.0 - r["method_effective_bpt"] / 16.0))
            cell["b"].append(r["method_effective_bpt"])
    return per


def main():
    args = sys.argv[1:]
    runtimes = [a for a in args if "runtime" in a] or ["results/dkv_runtime_7b.json"]
    sweeps = [a for a in args if a not in runtimes] or [
        "results/bench_7b_native/raw_results_v1.json",
        "results/bench_7b_dkv/raw_results_v1.json"]

    per = load(sweeps, runtimes)
    ctxs = sorted({c for v in per.values() for c in v})
    part1 = lambda cell: ALPHA * mean(cell["Q"]) + (1 - ALPHA) * mean(cell["R"])
    order = sorted(per, key=lambda k: -mean([part1(per[k][c]) for c in ctxs if c in per[k]]))

    for title, field, fmt in (("CRBench Part 1 by context length", None, "{:>8.1f}"),
                              ("Quality retained Q% by context length", "Q", "{:>8.1f}"),
                              ("Achieved b_eff by context length", "b", "{:>8.2f}")):
        print(f"\n=== {title} ===")
        print(f"{'method':<22}" + "".join(f"{c // 1024:>8}K" for c in ctxs) + f"{'mean':>9}")
        print("-" * (22 + 8 * len(ctxs) + 9))
        for k in order:
            cells, vals = "", []
            for c in ctxs:
                cell = per[k].get(c)
                if not cell:
                    cells += f"{'-':>8}"
                    continue
                x = part1(cell) if field is None else mean(cell[field])
                vals.append(x)
                cells += fmt.format(x)
            print(f"{k:<22}{cells}" + fmt.format(mean(vals)).replace(">8", ">9"))

    print(f"\n=== queries per cell (n) ===")
    print(f"{'method':<22}" + "".join(f"{c // 1024:>8}K" for c in ctxs))
    print("-" * (22 + 8 * len(ctxs)))
    for k in order:
        print(f"{k:<22}" + "".join(f"{len(per[k][c]['Q']) if c in per[k] else 0:>8}" for c in ctxs))
    print("\nn falls with context because the dense anchor fails more, and queries")
    print("dense failed are excluded. Columns with different n are scored over")
    print("different -- and at long context, easier -- query sets. Do not read a")
    print("trend across them without saying so.")


if __name__ == "__main__":
    main()
