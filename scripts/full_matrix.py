"""
CRBench Full Evaluation Matrix Generator.
Prints the complete multi-task x multi-context accuracy matrix for all methods.
"""

import os
import sys
import json
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CRBench Full Evaluation Matrix.")
    parser.add_argument("input_file", nargs="?", default="results/gemma4_e2b/progress.jsonl", help="Path to progress.jsonl file.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.exists(args.input_file):
        print(f"[!] Path not found: {args.input_file}", file=sys.stderr)
        return

    lines = [json.loads(l) for l in open(args.input_file, encoding="utf-8") if l.strip()]
    queries = [q for l in lines for q in l["query_results"]]

    by_method = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    tasks = sorted(list(set(q["task_name"] for q in queries)))
    contexts = sorted(list(set(q["context_length"] for q in queries)))
    methods = sorted(list(set(q["method_name"] for q in queries)))

    for q in queries:
        by_method[q["method_name"]][q["task_name"]][q["context_length"]].append(q)

    for m in methods:
        print("\n" + "=" * 90)
        print(f"METHOD MATRIX: {m.upper()}")
        print("=" * 90)
        header = f"{'Task':<26}" + "".join(f"{c//1024:>9}K" if c >= 1024 else f"{c:>10}" for c in contexts) + f"{'Mean Acc':>12}"
        print(header)
        print("-" * 90)

        all_task_accs = []
        for t in tasks:
            row_str = f"{t:<26}"
            t_accs = []
            for c in contexts:
                qs = by_method[m][t].get(c, [])
                if not qs:
                    row_str += f"{'--':>10}"
                else:
                    acc = sum(x.get("method_raw_score", 0.0) for x in qs) / len(qs) * 100.0
                    t_accs.append(acc)
                    all_task_accs.append(acc)
                    row_str += f"{acc:>9.1f}%"
            mean_t = sum(t_accs)/len(t_accs) if t_accs else 0.0
            row_str += f"{mean_t:>11.1f}%"
            print(row_str)

        total_mean = sum(all_task_accs)/len(all_task_accs) if all_task_accs else 0.0
        print("-" * 90)
        print(f"{'OVERALL AVERAGE':<26}" + " " * (10 * len(contexts)) + f"{total_mean:>11.1f}%")


if __name__ == "__main__":
    main()
