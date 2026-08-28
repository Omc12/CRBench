"""
CRBench Task and Context Table Generator.
Generates task-by-task and context-by-context evaluation breakdown tables.
"""

import os
import sys
import json
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CRBench Task and Context Breakdown Tables.")
    parser.add_argument("input_file", nargs="?", default="results/gemma4_e2b/progress.jsonl", help="Path to progress.jsonl file.")
    parser.add_argument("--format", choices=["table", "markdown"], default="table", help="Output display format.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input_file):
        print(f"[!] Path not found: {args.input_file}", file=sys.stderr)
        return

    lines = [json.loads(l) for l in open(args.input_file, encoding="utf-8") if l.strip()]
    queries = [q for l in lines for q in l["query_results"]]

    store = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_methods = set()
    all_contexts = set()
    all_tasks = set()

    for q in queries:
        t = q["task_name"]
        m = q["method_name"]
        c = q["context_length"]
        store[t][m][c].append(q)
        all_methods.add(m)
        all_contexts.add(c)
        all_tasks.add(t)

    sorted_contexts = sorted(all_contexts)
    tasks_order = ["single_niah", "multi_niah", "ruler_kv", "ruler_variable_tracking", "multihop_qa"]
    sorted_tasks = [t for t in tasks_order if t in all_tasks] + [t for t in sorted(all_tasks) if t not in tasks_order]

    for t in sorted_tasks:
        print("\n" + "=" * 105)
        print(f"TASK BREAKDOWN: {t.upper()}")
        print("=" * 105)
        
        if args.format == "markdown":
            header = "| Method | " + " | ".join(f"{c//1024}K" if c >= 1024 else str(c) for c in sorted_contexts) + " | Mean Acc | b_eff | R_mem |"
            sep = "| :--- | " + " | ".join(":---:" for _ in sorted_contexts) + " | :---: | :---: | :---: |"
            print(header)
            print(sep)
        else:
            header = f"{'Method':<22}" + "".join(f"{c//1024:>9}K" if c >= 1024 else f"{c:>10}" for c in sorted_contexts) + f"{'Mean Acc':>12} {'b_eff':>10} {'R_mem':>8}"
            print(header)
            print("-" * 105)

        t_rows = []
        for m in sorted(all_methods):
            m_qs = [x for c in sorted_contexts for x in store[t][m].get(c, [])]
            if not m_qs:
                continue
            avg_q = sum(x.get("method_raw_score", 0.0) for x in m_qs) / len(m_qs) * 100.0
            tot_d = sum(float(x.get("dense_memory_bytes", 0.0) or (x.get("context_length", 2048)*100000)) for x in m_qs)
            tot_m = sum(float(x.get("method_memory_bytes", 0.0) or (float(x.get("dense_memory_bytes", 0.0) or (x.get("context_length", 2048)*100000)) * x.get("method_effective_bpt", 16.0)/16.0)) for x in m_qs)
            r_mem = 100.0 * max(0.0, (tot_d - tot_m) / tot_d) if tot_d > 0 else 0.0
            b_eff = 16.0 * (1.0 - r_mem / 100.0)

            if args.format == "markdown":
                row_str = f"| **`{m}`** | "
                for c in sorted_contexts:
                    qs = store[t][m].get(c, [])
                    if not qs:
                        row_str += "-- | "
                    else:
                        c_acc = sum(x.get("method_raw_score", 0.0) for x in qs) / len(qs) * 100.0
                        row_str += f"{c_acc:.1f}% | "
                row_str += f"**{avg_q:.1f}%** | {b_eff:.2f} bpt | {r_mem:.1f}% |"
            else:
                row_str = f"{m:<22}"
                for c in sorted_contexts:
                    qs = store[t][m].get(c, [])
                    if not qs:
                        row_str += f"{'--':>10}"
                    else:
                        c_acc = sum(x.get("method_raw_score", 0.0) for x in qs) / len(qs) * 100.0
                        row_str += f"{c_acc:>9.1f}%"
                row_str += f"{avg_q:>11.1f}% {b_eff:>9.2f} {r_mem:>7.1f}%"

            t_rows.append((avg_q, b_eff, row_str))

        t_rows.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        for _, _, r_str in t_rows:
            print(r_str)


if __name__ == "__main__":
    main()
