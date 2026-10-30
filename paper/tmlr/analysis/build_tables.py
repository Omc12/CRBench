#!/usr/bin/env python3
"""Single source of truth for every number in the CRBench manuscript.

Reads the paired query records in ``results/<model>/progress.jsonl`` and emits
(a) a JSON blob of every quantity the paper quotes and (b) the LaTeX table
bodies included by the manuscript.

Two corrections relative to the first submission, both recomputations of the
already-collected 3,150 paired query executions rather than new runs:

1. ``b_eff`` now includes representation metadata, as Definition 3 has always
   stated.  The previous aggregate counted the compressed payload only, which
   understated DKV by 23-28% and KIVI by 7.6%.
2. Results are reported at each nominal budget (4 and 8 bits/element) instead of
   pooling the two runs into a single row.

The Part 2 hardware multiplier is computed per query against that query's own
paired dense execution and then averaged, which is what the manuscript
describes; the first submission computed it from run-level means.

Usage:  python paper/tmlr/analysis/build_tables.py
"""
from __future__ import annotations

import collections
import json
import os
import statistics

_HERE = os.path.dirname(os.path.abspath(__file__))          # paper/tmlr/analysis
REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
OUT = os.path.join(REPO, "paper", "tmlr", "analysis")

MODELS = [
    ("Gemma 4 E2B", "gemma4_e2b", "google/gemma-4-e2b-it", "2K--128K"),
    ("Gemma 4 E4B", "gemma4_e4b", "google/gemma-4-e4b-it", "2K--32K"),
    ("Qwen2.5 7B", "qwen2.5_7b", "Qwen/Qwen2.5-7B-Instruct", "2K--32K"),
]

# raw method_name -> display name.  The 7B run recorded the quantizer as
# "kv_quant"; it is the same adapter as "kivi_style_kv_quant".
DISPLAY = {
    "kivi_style_kv_quant": "KIVI (INT4/INT8)",
    "kv_quant": "KIVI (INT4/INT8)",
    "snapkv": "SnapKV",
    "streaming_llm": "StreamingLLM",
    "kv_merging": "KV-Merging",
    "low_rank_kv": "Low-Rank KV",
    "dkv_mid": "DKV-Mid",
    "dkv_high": "DKV-High",
    "dense_fp16": "Dense FP16",
}

ALPHA = 0.70
W_TTFT, W_THRU, W_VRAM = 0.35, 0.35, 0.30

TASK_DISPLAY = {
    "single_niah": "Single NIAH",
    "multi_niah": "Multi NIAH",
    "ruler_kv": "RULER KV",
    "ruler_variable_tracking": "Variable Tracking",
    "multihop_qa": "Multi-Hop QA",
}


def load(slug):
    path = os.path.join(REPO, "results", slug, "progress.jsonl")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.extend(json.loads(line)["query_results"])
    return rows


def bytes_dense(q):
    return float(q.get("dense_memory_bytes") or 0.0)


def bytes_payload(q):
    return float(q.get("method_memory_bytes") or 0.0)


def bytes_meta(q):
    md = q.get("metadata") or {}
    return float(md.get("metadata_overhead_bytes") or 0.0)


def clamp01(x):
    return max(0.0, min(1.0, x))


def summarise(qs):
    """Aggregate one (method, budget) cell."""
    n = len(qs)
    ok = [q for q in qs if q.get("status") == "SUCCESS"]

    # Q_abs: absolute task success over every attempted query; a failed run
    # scores zero rather than being dropped.
    q_abs = 100.0 * sum(float(q.get("method_raw_score") or 0.0) for q in qs) / n

    tot_d = sum(bytes_dense(q) for q in ok)
    tot_pay = sum(bytes_payload(q) for q in ok)
    tot_meta = sum(bytes_meta(q) for q in ok)
    # failed queries fall back to the dense footprint (no saving credited)
    fail_d = sum(bytes_dense(q) for q in qs if q.get("status") != "SUCCESS")
    tot_d_all = tot_d + fail_d
    tot_all = tot_pay + tot_meta + fail_d

    b_pay = 16.0 * (tot_pay + fail_d) / tot_d_all if tot_d_all else 16.0
    b_eff = 16.0 * tot_all / tot_d_all if tot_d_all else 16.0
    meta_share = 100.0 * tot_meta / (tot_pay + tot_meta) if (tot_pay + tot_meta) else 0.0
    r_mem = 100.0 * max(0.0, (tot_d_all - tot_all) / tot_d_all) if tot_d_all else 0.0
    r_mem_pay = 100.0 * max(0.0, 1.0 - (tot_pay + fail_d) / tot_d_all) if tot_d_all else 0.0

    s_res = ALPHA * q_abs + (1.0 - ALPHA) * r_mem
    s_res_pay = ALPHA * q_abs + (1.0 - ALPHA) * r_mem_pay

    # descriptive system means
    ttft = statistics.fmean(float(q.get("method_ttft_ms") or 0.0) for q in ok) / 1000.0
    thru = statistics.fmean(float(q.get("method_decode_throughput") or 0.0) for q in ok)
    vram = statistics.fmean(float(q.get("method_peak_vram_mb") or 0.0) for q in ok)
    d_ttft = statistics.fmean(float(q.get("dense_ttft_ms") or 0.0) for q in ok) / 1000.0
    d_thru = statistics.fmean(float(q.get("dense_decode_throughput") or 0.0) for q in ok)
    d_vram = statistics.fmean(float(q.get("dense_peak_vram_mb") or 0.0) for q in ok)

    # phi computed per query against that query's own paired dense execution
    phis = []
    for q in ok:
        mt, dt = float(q.get("method_ttft_ms") or 0.0), float(q.get("dense_ttft_ms") or 0.0)
        mh, dh = float(q.get("method_decode_throughput") or 0.0), float(q.get("dense_decode_throughput") or 0.0)
        mv, dv = float(q.get("method_peak_vram_mb") or 0.0), float(q.get("dense_peak_vram_mb") or 0.0)
        if mt <= 0 or dh <= 0 or mv <= 0:
            continue
        p_t = clamp01(dt / mt)
        p_h = clamp01(mh / dh)
        p_v = clamp01(dv / mv)
        phis.append((p_t ** W_TTFT) * (p_h ** W_THRU) * (p_v ** W_VRAM))
    phi = statistics.fmean(phis) if phis else 0.0
    r_sys = 100.0 * phi
    s_sys = ALPHA * s_res + (1.0 - ALPHA) * r_sys

    return dict(
        n=n, n_ok=len(ok), coverage=len(ok) / n,
        q_abs=q_abs, b_pay=b_pay, b_eff=b_eff, meta_share=meta_share,
        r_mem=r_mem, r_mem_pay=r_mem_pay, s_res=s_res, s_res_payload_only=s_res_pay,
        ttft=ttft, thru=thru, vram=vram,
        dense_ttft=d_ttft, dense_thru=d_thru, dense_vram=d_vram,
        ttft_x_dense=ttft / d_ttft if d_ttft else float("nan"),
        vram_x_dense=vram / d_vram if d_vram else float("nan"),
        phi=phi, r_sys=r_sys, s_sys=s_sys,
    )


def build():
    out = {"alpha": ALPHA, "models": {}}
    for label, slug, hf, ladder in MODELS:
        rows = load(slug)
        cells = collections.defaultdict(list)
        for q in rows:
            name = DISPLAY[q["method_name"]]
            budget = q.get("budget_spec")
            if name in ("DKV-Mid", "DKV-High", "Dense FP16"):
                budget = None  # adaptive / reference: a single operating point
            cells[(name, budget)].append(q)

        model = {"hf_id": hf, "ladder": ladder, "cells": {},
                 "n_records": len(rows),
                 "n_unique_queries": len({q["query_id"] for q in rows}),
                 "contexts": sorted({q["context_length"] for q in rows}),
                 "tasks": sorted({q["task_name"] for q in rows})}
        for (name, budget), qs in cells.items():
            key = f"{name}@{budget:.0f}" if budget is not None else name
            model["cells"][key] = dict(method=name, budget=budget, **summarise(qs))

        # per-task accuracy, pooled over budgets, for the heatmap
        tb = collections.defaultdict(lambda: collections.defaultdict(list))
        for q in rows:
            tb[DISPLAY[q["method_name"]]][q["task_name"]].append(
                float(q.get("method_raw_score") or 0.0))
        model["task_accuracy"] = {
            m: {TASK_DISPLAY[t]: 100.0 * statistics.fmean(v) for t, v in d.items()}
            for m, d in tb.items()}

        # per-context accuracy, pooled over budgets, for the scaling curve
        cb = collections.defaultdict(lambda: collections.defaultdict(list))
        for q in rows:
            cb[DISPLAY[q["method_name"]]][q["context_length"]].append(
                float(q.get("method_raw_score") or 0.0))
        model["context_accuracy"] = {
            m: {str(c): 100.0 * statistics.fmean(v) for c, v in sorted(d.items())}
            for m, d in cb.items()}

        out["models"][label] = model
    return out


# --------------------------------------------------------------------------
# LaTeX emission
# --------------------------------------------------------------------------
ORDER_FIXED = ["KIVI (INT4/INT8)", "SnapKV", "Low-Rank KV", "StreamingLLM", "KV-Merging"]
ORDER_ADAPTIVE = ["DKV-Mid", "DKV-High"]


def write_table(colspec, header, lines, path):
    r"""Emit a complete ``tabular`` block.

    The whole environment is generated rather than just the body rows: LaTeX's
    ``\input`` is not expandable in the mouth, so an inputted file that starts a
    row with ``\midrule`` or ``\multicolumn`` raises "Misplaced \noalign" /
    "Misplaced \omit".  Inputting a self-contained ``tabular`` avoids the
    alignment boundary entirely.  The manuscript supplies caption and label.
    """
    if lines and lines[0].strip() == r"\midrule":
        lines = lines[1:]
    out = ([r"\begin{tabular}{" + colspec + "}", r"\toprule", header, r"\midrule"]
           + lines + [r"\bottomrule", r"\end{tabular}"])
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def table_part1(out, path):
    """Main Part 1 table: every operating point, all three models."""
    lines = []
    for label, _, _, ladder in MODELS:
        cells = out["models"][label]["cells"]
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\textit{" + label + r" (" + ladder + r")}} \\")
        lines.append(r"\midrule")
        rows = []
        for m in ORDER_FIXED:
            for b in (4, 8):
                k = f"{m}@{b}"
                if k in cells:
                    rows.append((f"{m}", f"{b}", cells[k]))
        for m in ORDER_ADAPTIVE:
            if m in cells:
                rows.append((m, "adaptive", cells[m]))
        if "Dense FP16" in cells:
            rows.append(("Dense FP16", "--", cells["Dense FP16"]))
        best = max(r[2]["s_res"] for r in rows)
        for name, budget, c in rows:
            s = fmt(c["s_res"])
            s = r"\textbf{" + s + "}" if abs(c["s_res"] - best) < 1e-9 else s
            lines.append(
                f"{name} & {budget} & {fmt(c['q_abs'])}\\% & {fmt(c['b_pay'], 2)} & "
                f"{fmt(c['b_eff'], 2)} & {fmt(c['meta_share'])}\\% & {s} \\\\")
    header = (r"\textbf{Method} & \textbf{Budget} & \textbf{Accuracy} & "
              r"$b_{\text{pay}}$ & $b_{\text{eff}}$ & \textbf{meta} & "
              r"$\mathcal{S}_{\text{res}}$ \\")
    write_table("ll ccc c c", header, lines, path)


def table_part2(out, path):
    lines = []
    for label, _, _, _ in MODELS:
        cells = out["models"][label]["cells"]
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\textit{" + label + r"}} \\")
        lines.append(r"\midrule")
        rows = []
        for m in ORDER_FIXED:
            for b in (4, 8):
                k = f"{m}@{b}"
                if k in cells:
                    rows.append((f"{m} ({b} bpt)", cells[k]))
        for m in ORDER_ADAPTIVE:
            if m in cells:
                rows.append((m, cells[m]))
        if "Dense FP16" in cells:
            rows.append(("Dense FP16", cells["Dense FP16"]))
        rows.sort(key=lambda r: -r[1]["s_sys"])
        best = rows[0][1]["s_sys"]
        for name, c in rows:
            s = fmt(c["s_sys"])
            s = r"\textbf{" + s + "}" if abs(c["s_sys"] - best) < 1e-9 else s
            lines.append(
                f"{name} & {fmt(c['ttft'], 2)}\\,s & {fmt(c['ttft_x_dense'], 2)}$\\times$ & "
                f"{fmt(c['thru'])} & {fmt(c['vram'], 0)} & {fmt(c['phi'], 2)} & {s} \\\\")
    header = (r"\textbf{Method} & \textbf{TTFT} & \textbf{$\times$dense} & "
              r"\textbf{Decode} & \textbf{Peak VRAM} & $\phi$ & "
              r"$\mathcal{S}_{\text{sys}}$ \\" + "\n"
              r" & (s) & & (tok/s) & (MB) & & \\")
    write_table("l cc cc cc", header, lines, path)


def table_matched(out, path):
    """The headline table: matched nominal bitrate, unmatched everything else."""
    lines = []
    for budget in (4, 8):
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{l}{\textit{Nominal budget: " + str(budget)
                     + r"~bits/element}} \\")
        lines.append(r"\midrule")
        for m in ORDER_FIXED:
            cs = []
            for label, _, _, _ in MODELS:
                cs.append(out["models"][label]["cells"].get(f"{m}@{budget}"))
            if not all(cs):
                continue
            acc = " & ".join(fmt(c["q_abs"]) + r"\%" for c in cs)
            beff = statistics.fmean(c["b_eff"] for c in cs)
            ttft = statistics.fmean(c["ttft_x_dense"] for c in cs)
            phi = statistics.fmean(c["phi"] for c in cs)
            lines.append(f"{m} & {acc} & {fmt(beff, 2)} & {fmt(ttft, 2)}$\\times$ & {fmt(phi, 2)} \\\\")
    header = (r"\textbf{Method} & \textbf{E2B} & \textbf{E4B} & \textbf{7B} & "
              r"$b_{\text{eff}}$ & \textbf{TTFT $\times$dense} & $\phi$ \\")
    write_table("l ccc ccc", header, lines, path)


def spread_stats(out):
    """Accuracy spread among methods sharing a nominal bitrate."""
    res = {}
    for label, _, _, _ in MODELS:
        cells = out["models"][label]["cells"]
        for budget in (4, 8):
            pts = [(m, cells[f"{m}@{budget}"]) for m in ORDER_FIXED
                   if f"{m}@{budget}" in cells]
            if len(pts) < 2:
                continue
            accs = sorted(pts, key=lambda p: p[1]["q_abs"])
            lo, hi = accs[0], accs[-1]
            beffs = [p[1]["b_pay"] for p in pts]
            effs = [p[1]["b_eff"] for p in pts]
            res[f"{label}@{budget}"] = dict(
                n=len(pts), b_min=round(min(beffs), 2), b_max=round(max(beffs), 2),
                b_eff_min=round(min(effs), 2), b_eff_max=round(max(effs), 2),
                b_eff_spread=round(max(effs) - min(effs), 2),
                acc_lo=round(lo[1]["q_abs"], 1), method_lo=lo[0],
                acc_hi=round(hi[1]["q_abs"], 1), method_hi=hi[0],
                span=round(hi[1]["q_abs"] - lo[1]["q_abs"], 1),
                ratio=round(hi[1]["q_abs"] / lo[1]["q_abs"], 1) if lo[1]["q_abs"] else None)
    return res


def main():
    out = build()
    out["spread"] = spread_stats(out)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "paper_numbers.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    tex = os.path.join(REPO, "paper", "tmlr", "tables")
    os.makedirs(tex, exist_ok=True)
    table_part1(out, os.path.join(tex, "part1_body.tex"))
    table_part2(out, os.path.join(tex, "part2_body.tex"))
    table_matched(out, os.path.join(tex, "matched_body.tex"))

    # console digest
    for label, _, _, _ in MODELS:
        m = out["models"][label]
        print("=" * 90)
        print(f"{label}  ({m['n_records']} paired executions over "
              f"{m['n_unique_queries']} distinct queries)")
        print(f"{'cell':<26}{'Q':>7}{'b_pay':>8}{'b_eff':>8}{'meta%':>8}"
              f"{'R_mem':>8}{'S_res':>8}{'TTFT':>9}{'xDense':>8}{'phi':>7}{'S_sys':>8}")
        for k, c in sorted(m["cells"].items(), key=lambda kv: -kv[1]["s_res"]):
            print(f"{k:<26}{c['q_abs']:>6.1f}%{c['b_pay']:>8.2f}{c['b_eff']:>8.2f}"
                  f"{c['meta_share']:>7.1f}%{c['r_mem']:>7.1f}%{c['s_res']:>8.1f}"
                  f"{c['ttft']:>8.2f}s{c['ttft_x_dense']:>7.2f}x{c['phi']:>7.2f}"
                  f"{c['s_sys']:>8.1f}")
        print()

    print("=" * 90)
    print("Matched-bitrate accuracy spread")
    for k, v in out["spread"].items():
        print(f"  {k:<22} n={v['n']} b_eff in [{v['b_min']},{v['b_max']}] -> "
              f"{v['acc_lo']}% ({v['method_lo']}) .. {v['acc_hi']}% ({v['method_hi']}), "
              f"span {v['span']} pts, ratio {v['ratio']}x")

    print()
    print("wrote", os.path.join(OUT, "paper_numbers.json"))
    print("wrote", tex)


if __name__ == "__main__":
    main()
