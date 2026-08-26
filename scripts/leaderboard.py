"""CRBench leaderboard, scored by the formula the paper documents.

`score_method` reports a context-weighted mean of AUQC, and AUQC with fewer than
two budget points returns the quality itself -- so for a method measured at a
single budget its S_res carries no resource term at all. The per-query
`part1_score` is the documented formula, alpha*Q + (1-alpha)*R, and that is what
this table uses.

Two exclusions, both stated in the output rather than applied silently:

* Queries the dense baseline itself failed. There is no retained capability to
  measure where the model never had the capability, and the normaliser's
  dynamic-range floor turns those into an arbitrary number.
* Context lengths where a method did not engage. DKV bypasses to dense below
  DKV_ENGAGE_THRESHOLD (4096), so those rows are dense results wearing a
  method's name. The dense baseline itself is of course not "bypassing" and is
  always reported.
"""
import json
import sys
from collections import defaultdict

ALPHA = 0.70
DENSE = ("dense_fp16", "dense")


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def load(paths, runtime_paths):
    per = defaultdict(lambda: defaultdict(list))
    for path in paths:
        for q in json.load(open(path, encoding="utf-8"))["query_results"]:
            if q["method_name"] == "dkv":          # superseded pre-preset rows
                continue
            if q["dense_raw_score"] <= 0.0:
                continue
            k = (q["method_name"], q["budget_spec"], q["context_length"])
            per[k]["Q"].append(q["quality_retained_pct"])
            per[k]["R"].append(q["resource_efficiency"])
            per[k]["b"].append(q["method_effective_bpt"])
    for path in runtime_paths:
        for r in json.load(open(path, encoding="utf-8"))["records"]:
            if r["status"] != "SUCCESS" or r["dense_raw_score"] <= 0.0:
                continue
            k = (r["method_name"], "preset", r["context_length"])
            per[k]["Q"].append(100.0 * min(1.0, r["method_raw_score"] / r["dense_raw_score"]))
            per[k]["R"].append(100.0 * max(0.0, 1.0 - r["method_effective_bpt"] / 16.0))
            per[k]["b"].append(r["method_effective_bpt"])
    return per


def main():
    args = sys.argv[1:]
    runtime = [a for a in args if "runtime" in a]
    sweeps = [a for a in args if a not in runtime]
    if not sweeps:
        sweeps = ["results/bench_7b_native/raw_results_v1.json",
                  "results/bench_7b_dkv/raw_results_v1.json"]
        runtime = runtime or ["results/dkv_runtime_7b.json"]

    per = load(sweeps, runtime)
    agg = defaultdict(lambda: defaultdict(list))
    bypassed = defaultdict(list)
    for (m, b, c), v in per.items():
        if m not in DENSE and mean(v["b"]) >= 15.99:
            bypassed[(m, b)].append(c)
            continue
        for f in ("Q", "R", "b"):
            agg[(m, b)][f].extend(v[f])

    rows = []
    for (m, b), v in agg.items():
        Q, R = mean(v["Q"]), mean(v["R"])
        rows.append((ALPHA * Q + (1 - ALPHA) * R, m, b, mean(v["b"]), Q, R, len(v["Q"])))
    rows.sort(reverse=True)

    print("Part 1 = 0.70*Q + 0.30*R.  Dense-failed queries excluded.")
    print("Method rows aggregate only context lengths where the method engaged.")
    print()
    print(f"{'method':<20}{'budget':>8}{'b_eff':>8}{'Q %':>7}{'R %':>7}{'Part1':>8}{'n':>6}")
    print("-" * 64)
    for p1, m, b, be, Q, R, n in rows:
        mark = "  <- baseline" if m in DENSE else ""
        print(f"{m:<20}{str(b):>8}{be:>8.2f}{Q:>7.1f}{R:>7.1f}{p1:>8.1f}{n:>6}{mark}")
    for (m, b), cs in sorted(bypassed.items()):
        print()
        print(f"  {m} @{b}: did not engage at {sorted(cs)} (DKV_ENGAGE_THRESHOLD=4096);")
        print(f"      those are dense results and are excluded, not averaged in.")


if __name__ == "__main__":
    main()
