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


def per_method_spread(per):
    """Min and max achieved b_eff per method across context lengths.

    A method that lands on the same b_eff at every context length is honouring
    the budget it was given regardless of how much context it is handed. One
    whose b_eff moves is letting context length decide its operating point,
    which matters for anyone sizing a deployment from a single measurement.
    """
    out = {}
    for (m, b, c), v in per.items():
        cur = out.setdefault((m, b), [float("inf"), float("-inf")])
        x = mean(v["b"])
        cur[0], cur[1] = min(cur[0], x), max(cur[1], x)
    return out


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

    # Full-grid aggregate: every context length, including any where a method
    # declined to compress. Reported alongside the engaged-only figure rather
    # than instead of it -- dropping those rows hides that the method chose not
    # to act, and keeping them unlabelled credits it with dense quality at a
    # dense price. Both are the same measurements, aggregated over different sets.
    full = defaultdict(lambda: defaultdict(list))
    for (m, b, c), v in per.items():
        for f in ("Q", "R", "b"):
            full[(m, b)][f].extend(v[f])

    rows = []
    for (m, b), v in agg.items():
        Q, R = mean(v["Q"]), mean(v["R"])
        fv = full[(m, b)]
        fQ, fR, fb = mean(fv["Q"]), mean(fv["R"]), mean(fv["b"])
        rows.append((ALPHA * Q + (1 - ALPHA) * R, m, b, mean(v["b"]), Q, R, len(v["Q"]),
                     fb, ALPHA * fQ + (1 - ALPHA) * fR, len(fv["Q"])))
    rows.sort(reverse=True)

    print("Part 1 = 0.70*Q + 0.30*R.  Dense-failed queries excluded.")
    print("Two aggregates per method: context lengths where it engaged, and the full grid.")
    print()
    print(f"{'':<20}{'':>8}{'--- engaged lengths ---':>32}{'--- full grid ---':>22}")
    print(f"{'method':<20}{'budget':>8}{'b_eff':>8}{'Q %':>7}{'R %':>7}{'Part1':>8}{'n':>6}"
          f"{'b_eff':>9}{'Part1':>8}{'n':>5}")
    print("-" * 96)
    for p1, m, b, be, Q, R, n, fb, fp1, fn in rows:
        mark = "  <- baseline" if m in DENSE else ("  *" if n != fn else "")
        print(f"{m:<20}{str(b):>8}{be:>8.2f}{Q:>7.1f}{R:>7.1f}{p1:>8.1f}{n:>6}"
              f"{fb:>9.2f}{fp1:>8.1f}{fn:>5}{mark}")
    if bypassed:
        print()
        print("* methods whose two columns differ declined to compress at some lengths:")
        for (m, b), cs in sorted(bypassed.items()):
            print(f"    {m} @{b}: passed through at {sorted(cs)} "
                  f"(DKV_ENGAGE_THRESHOLD=4096)")
        print("  The engaged columns exclude those lengths; the full-grid columns")
        print("  include them, where the method scores dense quality at dense cost.")
    print()
    print("Budget adherence -- b_eff spread across the grid, per method:")
    for (m, b), v in sorted(per_method_spread(per).items()):
        lo, hi = v
        flag = "  <- constant" if hi - lo < 0.05 else ""
        print(f"    {m:<20}{str(b):>8}   {lo:5.2f} .. {hi:5.2f}{flag}")


if __name__ == "__main__":
    main()
