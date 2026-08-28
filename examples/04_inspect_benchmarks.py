"""
Example 04: Inspecting and reporting canonical CRBench benchmark results.
Reads unified progress files from results/ and prints Part 1 and Part 2 leaderboards.
"""

import os
import json
from collections import defaultdict


def analyze_model(model_name: str, progress_file: str, alpha: float = 0.70):
    if not os.path.exists(progress_file):
        print(f"[!] File not found: {progress_file}")
        return

    lines = [json.loads(l) for l in open(progress_file, encoding="utf-8") if l.strip()]
    queries = [q for l in lines for q in l["query_results"]]

    methods_data = defaultdict(list)
    for q in queries:
        methods_data[q["method_name"]].append(q)

    dense_qs = methods_data.get("dense_fp16", [])
    ref_ttft = sum(q["method_ttft_ms"] for q in dense_qs) / len(dense_qs) if dense_qs else 1000.0
    ref_thru = sum(q["method_decode_throughput"] for q in dense_qs) / len(dense_qs) if dense_qs else 10.0
    ref_vram = sum(q["method_peak_vram_mb"] for q in dense_qs) / len(dense_qs) if dense_qs else 1000.0

    rows = []
    for m, qs in methods_data.items():
        avg_q = sum(x["method_raw_score"] for x in qs) / len(qs) * 100.0
        tot_d = sum(float(x.get("dense_memory_bytes", 0.0) or (x.get("context_length", 2048)*100000)) for x in qs)
        tot_m = sum(float(x.get("method_memory_bytes", 0.0) or (float(x.get("dense_memory_bytes", 0.0) or (x.get("context_length", 2048)*100000)) * x.get("method_effective_bpt", 16.0)/16.0)) for x in qs)
        r_mem = 100.0 * max(0.0, (tot_d - tot_m) / tot_d) if tot_d > 0 else 0.0
        b_eff = 16.0 * (1.0 - r_mem / 100.0)
        s_res = alpha * avg_q + (1.0 - alpha) * r_mem

        m_ttft = sum(q["method_ttft_ms"] for q in qs) / len(qs)
        m_thru = sum(q["method_decode_throughput"] for q in qs) / len(qs)
        m_vram = sum(q["method_peak_vram_mb"] for q in qs) / len(qs)

        phi_ttft = max(0.0, min(1.0, ref_ttft / max(0.01, m_ttft)))
        phi_thru = max(0.0, min(1.0, m_thru / max(0.01, ref_thru)))
        phi_vram = max(0.0, min(1.0, ref_vram / max(0.01, m_vram)))

        mult = (phi_ttft ** 0.35) * (phi_thru ** 0.35) * (phi_vram ** 0.30)
        r_sys = 100.0 * mult
        s_sys = alpha * s_res + (1.0 - alpha) * r_sys

        rows.append((s_res, s_sys, avg_q, b_eff, r_mem, m_ttft, m_thru, m_vram, mult, m))

    rows.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 115)
    print(f"CRBENCH CANONICAL LEADERBOARD: {model_name.upper()}")
    print("=" * 115)
    print(f"{'Rank':<5} {'Method':<22} {'Accuracy':<12} {'b_eff':<8} {'R_mem':<8} {'Part 1 (S_res)':<16} {'TTFT':<10} {'Throughput':<14} {'Part 2 (S_sys)':<16}")
    print("-" * 115)
    for idx, (s_res, s_sys, avg_q, b_eff, r_mem, m_ttft, m_thru, m_vram, mult, m) in enumerate(rows, 1):
        print(f"{idx:<5} {m:<22} {avg_q:>10.1f}% {b_eff:>7.2f} {r_mem:>6.1f}% {s_res:>14.1f} {m_ttft/1000.0:>8.2f}s {m_thru:>11.1f} tok/s {s_sys:>14.1f}")


def main():
    analyze_model("Gemma 4 E2B (2K - 128K)", "results/gemma4_e2b/progress.jsonl")
    analyze_model("Gemma 4 E4B (2K - 32K)", "results/gemma4_e4b/progress.jsonl")
    analyze_model("Qwen2.5 7B (2K - 32K)", "results/qwen2.5_7b/progress.jsonl")


if __name__ == "__main__":
    main()
