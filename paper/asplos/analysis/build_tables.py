#!/usr/bin/env python3
"""Emit the LaTeX table bodies used by the ASPLOS manuscript.

Reads the same ``paper_numbers.json`` the TMLR version used, so no number is
recomputed here -- this script only chooses which columns the two-column
ASPLOS layout can afford and how they are grouped.

    python paper/asplos/analysis/build_tables.py
"""
from __future__ import annotations

import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ASPLOS = os.path.dirname(HERE)
OUT = os.path.join(ASPLOS, "tables")
DATADIR = os.path.join(os.path.dirname(ASPLOS), "tmlr", "analysis")

MODELS = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen2.5 7B"]
LADDER = {"Gemma 4 E2B": "2K--128K", "Gemma 4 E4B": "2K--32K", "Qwen2.5 7B": "2K--32K"}
FIXED = ["KIVI (INT4/INT8)", "SnapKV", "Low-Rank KV", "StreamingLLM", "KV-Merging"]
ADAPTIVE = ["DKV-Mid", "DKV-High"]

# Short labels: the two-column table cannot afford "KIVI (INT4/INT8)".
LABEL = {
    "KIVI (INT4/INT8)": "KIVI",
    "SnapKV": "SnapKV",
    "Low-Rank KV": "Low-Rank",
    "StreamingLLM": "StreamLLM",
    "KV-Merging": "KV-Merge",
    "DKV-Mid": "DKV-Mid",
    "DKV-High": "DKV-High",
    "Dense FP16": "Dense FP16",
}


def load():
    with open(os.path.join(DATADIR, "paper_numbers.json"), encoding="utf-8") as fh:
        return json.load(fh)


def rows_for(cells):
    """Canonical operating-point order, identical in every table and figure."""
    out = []
    for m in FIXED:
        for b in (4, 8):
            k = f"{m}@{b}"
            if k in cells:
                out.append((LABEL[m], str(b), cells[k]))
    for m in ADAPTIVE:
        if m in cells:
            out.append((LABEL[m], "adapt.", cells[m]))
    out.append((LABEL["Dense FP16"], "--", cells["Dense FP16"]))
    return out


def f(x, nd=1):
    return f"{x:.{nd}f}"


def write(path, lines):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, path), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  wrote tables/{path}  ({len(lines)} lines)")


# ---------------------------------------------------------------------------
# The paper's central data artifact: all three axes, every operating point.
# ---------------------------------------------------------------------------
def table_three_axis(D):
    lines = [
        r"\begin{tabular}{@{}ll rrr rr rrrr rr@{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{\textbf{Axis 1: representation}} "
        r"& \multicolumn{2}{c}{\textbf{Axis 2: task}} "
        r"& \multicolumn{4}{c}{\textbf{Axis 3: measured serving}} & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-11}",
        r"\textbf{Method} & \textbf{Bits} & $b_{\text{pay}}$ & $\beff$ & meta "
        r"& Acc. & $\Sres$ & TTFT & $\times$dense & Dec. & VRAM & $\phi$ & $\Ssys$ \\",
        r" & & & & (\%) & (\%) & & (s) & & (tok/s) & (MB) & & \\",
        r"\midrule",
    ]
    for mi, model in enumerate(MODELS):
        cells = D["models"][model]["cells"]
        rows = rows_for(cells)
        best_res = max(r[2]["s_res"] for r in rows)
        best_phi = max(r[2]["phi"] for r in rows if r[0] != "Dense FP16")
        if mi:
            lines.append(r"\midrule")
        lines.append(
            r"\multicolumn{13}{@{}l}{\textit{" + model + r"} \ (" + LADDER[model]
            + r")} \\[1pt]")
        for name, budget, c in rows:
            sres = f(c["s_res"])
            if abs(c["s_res"] - best_res) < 1e-9:
                sres = r"\textbf{" + sres + "}"
            phi = f(c["phi"], 2)
            if name != "Dense FP16" and abs(c["phi"] - best_phi) < 1e-9:
                phi = r"\textbf{" + phi + "}"
            lines.append(
                f"{name} & {budget} & {f(c['b_pay'], 2)} & {f(c['b_eff'], 2)} & "
                f"{f(c['meta_share'])} & {f(c['q_abs'])} & {sres} & "
                f"{f(c['ttft'], 2)} & {f(c['ttft_x_dense'], 2)}$\\times$ & "
                f"{f(c['thru'])} & {c['vram']:.0f} & {phi} & {f(c['s_sys'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("three_axis_body.tex", lines)


# ---------------------------------------------------------------------------
# The controlled comparison: identical nominal budget, three models.
# ---------------------------------------------------------------------------
def table_matched(D):
    lines = [
        r"\begin{tabular}{@{}l ccc c cc@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{\textbf{Task accuracy (\%)}} & & "
        r"\multicolumn{2}{c}{\textbf{Serving}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){6-7}",
        r"\textbf{Method} & E2B & E4B & 7B & $\beff$ & TTFT & $\phi$ \\",
        r"\midrule",
    ]
    for bi, budget in enumerate((4, 8)):
        if bi:
            lines.append(r"\midrule")
        lines.append(r"\multicolumn{7}{@{}l}{\textit{Identical nominal budget: "
                     + str(budget) + r"~bits/element}} \\[1pt]")
        for m in FIXED:
            cs = [D["models"][mo]["cells"][f"{m}@{budget}"] for mo in MODELS]
            acc = " & ".join(f(c["q_abs"]) for c in cs)
            beff = statistics.fmean(c["b_eff"] for c in cs)
            ttft = statistics.fmean(c["ttft_x_dense"] for c in cs)
            phi = statistics.fmean(c["phi"] for c in cs)
            lines.append(f"{LABEL[m]} & {acc} & {f(beff, 2)} & "
                         f"{f(ttft, 2)}$\\times$ & {f(phi, 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("matched_body.tex", lines)


def main():
    D = load()
    print("writing ASPLOS table bodies ->", OUT)
    table_three_axis(D)
    table_matched(D)


if __name__ == "__main__":
    main()
