"""Build the Qwen2.5-7B leaderboard from the documented Part 1 formula.

`score_method` reports a context-weighted mean of AUQC, and AUQC with fewer than
two budget points returns the quality itself -- so for any method measured at a
single budget, its S_res carries no resource term at all. That is how dkv_high
ranked first while storing 15.5 of 16 bits per element.

The per-query `part1_score` is the formula the paper documents,
alpha*Q + (1-alpha)*R, and it is what this table uses.
"""
import json
from collections import defaultdict

import sys
RUNS = sys.argv[1:] or ["results/bench_7b_native/raw_results_v1.json",
                        "results/bench_7b_dkv/raw_results_v1.json"]

ALPHA = 0.70
per = defaultdict(lambda: defaultdict(list))
dense_failed = set()

for path in RUNS:
    d = json.load(open(path, encoding="utf-8"))
    for q in d["query_results"]:
        key = (q["method_name"], q["budget_spec"])
        # The bare "dkv" rows come from the run made before DKV's preset and
        # byte-accounting fixes; dkv_mid / dkv_high supersede them.
        if q["method_name"] == "dkv":
            continue
        # A query the dense reference itself got wrong carries no information
        # about compression: there is no retained capability to measure, and the
        # normaliser's dynamic-range floor turns it into an arbitrary number.
        if q["dense_raw_score"] <= 0.0:
            dense_failed.add((q["task_name"], q["context_length"], q["query_id"]))
            continue
        per[key]["Q"].append(q["quality_retained_pct"])
        per[key]["R"].append(q["resource_efficiency"])
        per[key]["bpt"].append(q["method_effective_bpt"])
        per[key]["ttft"].append(q.get("method_ttft_ms") or 0.0)
        per[key]["dec"].append(q.get("method_decode_throughput") or 0.0)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


rows = []
for (m, b), v in per.items():
    Q, R = mean(v["Q"]), mean(v["R"])
    rows.append((ALPHA * Q + (1 - ALPHA) * R, m, b, mean(v["bpt"]), Q, R,
                 mean(v["ttft"]) / 1000.0, mean(v["dec"]), len(v["Q"])))
rows.sort(reverse=True)

print("Qwen2.5-7B-Instruct NF4, 2K-32K, 5 tasks x 3 samples, RTX 4070 SUPER")
print("Part 1 = 0.70*Q + 0.30*R, per the documented formula.")
print("Queries the dense baseline itself failed are excluded, not scored zero.\n")
print(f"{'method':<16}{'budget':>7}{'b_eff':>8}{'Q %':>7}{'R %':>7}{'Part1':>8}"
      f"{'TTFT s':>8}{'tok/s':>8}{'n':>6}")
print("-" * 76)
for p1, m, b, bpt, Q, R, ttft, dec, n in rows:
    print(f"{m:<16}{b:>7}{bpt:>8.2f}{Q:>7.1f}{R:>7.1f}{p1:>8.1f}{ttft:>8.1f}{dec:>8.1f}{n:>6}")

print(f"\nExcluded: {len(dense_failed)} queries where the dense baseline scored 0.")
tasks = defaultdict(int)
for t, c, _ in dense_failed:
    tasks[t] += 1
for t, n in sorted(tasks.items(), key=lambda kv: -kv[1]):
    print(f"    {t:<28} {n}")
