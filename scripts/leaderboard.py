"""
CRBench Canonical Leaderboard Generator.
Evaluates algorithmic resource fidelity (Part 1, S_res) and hardware serving utility (Part 2, S_sys)
under Option B Physical Memory Aggregation.
"""

import os
import sys
import json
import argparse
from collections import defaultdict

ALPHA = 0.70


def parse_args():
    parser = argparse.ArgumentParser(description="Generate CRBench Leaderboard from evaluation progress files.")
    parser.add_argument("input_files", nargs="*", default=["results/gemma4_e2b/progress.jsonl"], help="Path(s) to progress.jsonl or raw results json file(s).")
    parser.add_argument("--alpha", type=float, default=0.70, help="Weighting factor alpha for quality vs memory (default: 0.70)")
    parser.add_argument("--format", choices=["table", "markdown", "csv"], default="table", help="Output display format.")
    return parser.parse_args()


def load_queries(paths):
    queries = []
    for p in paths:
        if not os.path.exists(p):
            print(f"[!] Warning: Path not found: {p}", file=sys.stderr)
            continue
        if p.endswith(".jsonl"):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    data = json.loads(line)
                    if "query_results" in data:
                        queries.extend(data["query_results"])
                    elif "method_name" in data:
                        queries.append(data)
        elif p.endswith(".json"):
            data = json.load(open(p, encoding="utf-8"))
            if "query_results" in data:
                queries.extend(data["query_results"])
            elif "records" in data:
                queries.extend(data["records"])
    return queries


def main():
    args = parse_args()
    queries = load_queries(args.input_files)
    if not queries:
        print("[!] No evaluation queries found in provided paths.", file=sys.stderr)
        return

    methods_data = defaultdict(list)
    for q in queries:
        methods_data[q["method_name"]].append(q)

    dense_qs = methods_data.get("dense_fp16", []) or methods_data.get("dense", [])
    ref_ttft = sum(q.get("method_ttft_ms", 1000.0) for q in dense_qs) / len(dense_qs) if dense_qs else 1000.0
    ref_thru = sum(q.get("method_decode_throughput", 10.0) for q in dense_qs) / len(dense_qs) if dense_qs else 10.0
    ref_vram = sum(q.get("method_peak_vram_mb", 1000.0) for q in dense_qs) / len(dense_qs) if dense_qs else 1000.0

    rows = []
    for m, qs in methods_data.items():
        avg_q = sum(x.get("method_raw_score", 0.0) for x in qs) / len(qs) * 100.0
        tot_d = sum(float(x.get("dense_memory_bytes", 0.0) or (x.get("context_length", 2048)*100000)) for x in qs)
        tot_m = sum(float(x.get("method_memory_bytes", 0.0) or (float(x.get("dense_memory_bytes", 0.0) or (x.get("context_length", 2048)*100000)) * x.get("method_effective_bpt", 16.0)/16.0)) for x in qs)
        r_mem = 100.0 * max(0.0, (tot_d - tot_m) / tot_d) if tot_d > 0 else 0.0
        b_eff = 16.0 * (1.0 - r_mem / 100.0)
        s_res = args.alpha * avg_q + (1.0 - args.alpha) * r_mem

        m_ttft = sum(q.get("method_ttft_ms", 1000.0) for q in qs) / len(qs)
        m_thru = sum(q.get("method_decode_throughput", 10.0) for q in qs) / len(qs)
        m_vram = sum(q.get("method_peak_vram_mb", 1000.0) for q in qs) / len(qs)

        phi_ttft = max(0.0, min(1.0, ref_ttft / max(0.01, m_ttft)))
        phi_thru = max(0.0, min(1.0, m_thru / max(0.01, ref_thru)))
        phi_vram = max(0.0, min(1.0, ref_vram / max(0.01, m_vram)))

        mult = (phi_ttft ** 0.35) * (phi_thru ** 0.35) * (phi_vram ** 0.30)
        r_sys = 100.0 * mult
        s_sys = args.alpha * s_res + (1.0 - args.alpha) * r_sys

        rows.append((s_res, s_sys, avg_q, b_eff, r_mem, m_ttft, m_thru, m_vram, mult, m, len(qs)))

    rows.sort(key=lambda x: x[0], reverse=True)

    if args.format == "markdown":
        print("| Rank | Method | Queries | Accuracy | b_eff | R_mem | Part 1 (S_res) | TTFT | Throughput | Part 2 (S_sys) |")
        print("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for idx, (s_res, s_sys, avg_q, b_eff, r_mem, m_ttft, m_thru, m_vram, mult, m, nq) in enumerate(rows, 1):
            print(f"| **{idx}** | **`{m}`** | {nq} | {avg_q:.1f}% | {b_eff:.2f} bpt | {r_mem:.1f}% | **{s_res:.1f}** | {m_ttft/1000.0:.2f}s | {m_thru:.1f} tok/s | **{s_sys:.1f}** |")
    elif args.format == "csv":
        print("Rank,Method,Queries,Accuracy,b_eff,R_mem,S_res,TTFT_ms,Throughput,S_sys")
        for idx, (s_res, s_sys, avg_q, b_eff, r_mem, m_ttft, m_thru, m_vram, mult, m, nq) in enumerate(rows, 1):
            print(f"{idx},{m},{nq},{avg_q:.2f},{b_eff:.2f},{r_mem:.2f},{s_res:.2f},{m_ttft:.1f},{m_thru:.2f},{s_sys:.2f}")
    else:
        print("=" * 115)
        print(f"CRBENCH CANONICAL LEADERBOARD (Option B Aggregation, Alpha={args.alpha:.2f})")
        print("=" * 115)
        print(f"{'Rank':<5} {'Method':<22} {'Queries':<8} {'Accuracy':<10} {'b_eff':<8} {'R_mem':<8} {'Part 1 (S_res)':<16} {'TTFT':<10} {'Throughput':<12} {'Part 2 (S_sys)':<15}")
        print("-" * 115)
        for idx, (s_res, s_sys, avg_q, b_eff, r_mem, m_ttft, m_thru, m_vram, mult, m, nq) in enumerate(rows, 1):
            print(f"{idx:<5} {m:<22} {nq:<8} {avg_q:>8.1f}% {b_eff:>7.2f} {r_mem:>6.1f}% {s_res:>14.1f} {m_ttft/1000.0:>8.2f}s {m_thru:>9.1f} tok/s {s_sys:>14.1f}")


if __name__ == "__main__":
    main()
