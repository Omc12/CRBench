#!/usr/bin/env python3
"""Assert that the numbers written into the manuscript match the computed data.

Every claim below is stated twice: once as a fact derived from
``paper_numbers.json`` / ``decoupling_stats.json``, and once as the literal
string the manuscript must contain.  A mismatch fails loudly rather than
shipping a paper whose prose has drifted from its tables.

Usage:  python paper/tmlr/analysis/verify_claims.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(os.path.dirname(HERE), "crbench_tmlr_v2.tex")

D = json.load(open(os.path.join(HERE, "paper_numbers.json"), encoding="utf-8"))
S = json.load(open(os.path.join(HERE, "decoupling_stats.json"), encoding="utf-8"))
RAW = open(TEX, encoding="utf-8").read()


def normalise(s):
    """Strip the LaTeX markup that separates a number from its prose form."""
    for a, b in (("$", ""), ("{,}", ","), ("{-}", "-"), ("\\,", " "),
                 ("\\%", "%"), ("\\times", "x"), ("~", " "), ("{", ""), ("}", "")):
        s = s.replace(a, b)
    return " ".join(s.split())


TEXT = normalise(RAW)

M = D["models"]
E2B, E4B, Q7B = M["Gemma 4 E2B"], M["Gemma 4 E4B"], M["Qwen2.5 7B"]

checks = []


def claim(label, computed, literal):
    """`computed` is what the data says; `literal` must appear in the .tex."""
    checks.append((label, computed, literal, normalise(literal) in TEXT))


# --- matched-bitrate accuracy spread (Section 6.1, Figure 1a, abstract) ------
for key, mdl in (("Gemma 4 E2B@4", "E2B"), ("Gemma 4 E4B@4", "E4B"),
                 ("Qwen2.5 7B@4", "7B")):
    sp = S["matched_bitrate_spread"][key]
    claim(f"{mdl} 4-bit accuracy low", sp["acc_lo"], f"{sp['acc_lo']}\\%")
    claim(f"{mdl} 4-bit accuracy high", sp["acc_hi"], f"{sp['acc_hi']}\\%")
    claim(f"{mdl} 4-bit spread ratio", sp["ratio"], f"{sp['ratio']}\\times")

for key, mdl in (("Gemma 4 E2B@8", "E2B"), ("Gemma 4 E4B@8", "E4B"),
                 ("Qwen2.5 7B@8", "7B")):
    sp = S["matched_bitrate_spread"][key]
    claim(f"{mdl} 8-bit spread (pts)", sp["span"], f"{sp['span']}")

# --- the matched pair (Section 6.3, Figure 1b) ------------------------------
k4 = E4B["cells"]["KIVI (INT4/INT8)@4"]
s4 = E4B["cells"]["SnapKV@4"]
claim("KIVI@4 E4B b_eff", k4["b_eff"], f"{k4['b_eff']:.2f}")
claim("SnapKV@4 E4B b_eff", s4["b_eff"], f"{s4['b_eff']:.2f}")
claim("KIVI@4 E4B accuracy", k4["q_abs"], f"{k4['q_abs']:.1f}\\%")
claim("SnapKV@4 E4B accuracy", s4["q_abs"], f"{s4['q_abs']:.1f}\\%")
claim("KIVI@4 E4B TTFT", k4["ttft"], f"{k4['ttft']:.2f}$\\,$s" if False else f"{k4['ttft']:.2f}")
claim("SnapKV@4 E4B TTFT", s4["ttft"], f"{s4['ttft']:.2f}")
claim("TTFT ratio", s4["ttft"] / k4["ttft"], f"{s4['ttft'] / k4['ttft']:.1f}\\times")
claim("VRAM ratio", s4["vram"] / k4["vram"], f"{s4['vram'] / k4['vram']:.2f}\\times")
claim("KIVI@4 E4B phi", k4["phi"], f"{k4['phi']:.2f}")
claim("SnapKV@4 E4B phi", s4["phi"], f"{s4['phi']:.2f}")
claim("KIVI@4 E4B S_sys", k4["s_sys"], f"{k4['s_sys']:.1f}")
claim("SnapKV@4 E4B S_sys", s4["s_sys"], f"{s4['s_sys']:.1f}")

# --- pooled rank correlations (abstract, Sections 6.1 and 6.3) --------------
P = S["pooled"]
claim("rho(R_mem, accuracy)", P["rho_r_mem_q_abs"], f"{P['rho_r_mem_q_abs']:.2f}")
claim("rho(R_mem, phi)", P["rho_r_mem_phi"], f"{P['rho_r_mem_phi']:.2f}")
claim("rho(b_eff, TTFT)", P["rho_b_eff_ttft_x_dense"], f"{P['rho_b_eff_ttft_x_dense']:.2f}")
claim("rho(b_eff, VRAM)", P["rho_b_eff_vram_x_dense"], f"{abs(P['rho_b_eff_vram_x_dense']):.2f}")

# --- metadata accounting (Section 6.2, abstract) ----------------------------
dkv = E4B["cells"]["DKV-Mid"]
claim("DKV-Mid E4B payload", dkv["b_pay"], f"{dkv['b_pay']:.2f}")
claim("DKV-Mid E4B effective", dkv["b_eff"], f"{dkv['b_eff']:.2f}")
dkvh = E4B["cells"]["DKV-High"]
claim("DKV-High E4B payload", dkvh["b_pay"], f"{dkvh['b_pay']:.2f}")
claim("DKV-High E4B effective", dkvh["b_eff"], f"{dkvh['b_eff']:.2f}")
claim("KIVI@4 E4B payload", k4["b_pay"], f"{k4['b_pay']:.2f}")

# --- the model-dependent SnapKV penalty (Section 6.3) ----------------------
s4q = Q7B["cells"]["SnapKV@4"]
s4e = E2B["cells"]["SnapKV@4"]
claim("SnapKV 7B TTFT x dense", s4q["ttft_x_dense"], f"{s4q['ttft_x_dense']:.1f}\\times")
claim("SnapKV E2B TTFT x dense", s4e["ttft_x_dense"], f"{s4e['ttft_x_dense']:.2f}\\times")
claim("SnapKV E2B VRAM x dense", s4e["vram_x_dense"], f"{s4e['vram_x_dense']:.2f}\\times")
claim("SnapKV E2B phi", s4e["phi"], f"{s4e['phi']:.2f}")

# --- rank displacement (Section 6.3) ---------------------------------------
for lbl, short in (("Gemma 4 E2B", "E2B"), ("Gemma 4 E4B", "E4B"),
                   ("Qwen2.5 7B", "7B")):
    pm = S["per_model"][lbl]
    claim(f"{short} mean rank shift", pm["mean_rank_shift"], f"{pm['mean_rank_shift']:.1f}")
    claim(f"{short} rho(S_res, phi)", pm["rho_sres_phi"], f"{abs(round(pm['rho_sres_phi'], 2)):.2f}")

# --- design accounting (Section 5.1) ---------------------------------------
per_model = E2B["n_records"]
total = sum(m["n_records"] for m in M.values())
claim("paired executions per model", per_model, f"{per_model:,}".replace(",", "{,}"))
claim("total paired executions", total, f"{total:,}".replace(",", "{,}"))
claim("distinct queries per model", E2B["n_unique_queries"], str(E2B["n_unique_queries"]))
claim("configurations per query", per_model // E2B["n_unique_queries"],
      f"${per_model // E2B['n_unique_queries']}$ configurations")

# --- StreamingLLM budget sensitivity (Section 6.1) -------------------------
sl4, sl8 = E2B["cells"]["StreamingLLM@4"], E2B["cells"]["StreamingLLM@8"]
claim("StreamingLLM E2B gain", sl8["q_abs"] - sl4["q_abs"], f"${sl8['q_abs'] - sl4['q_abs']:.1f}$ points")
ki4, ki8 = E2B["cells"]["KIVI (INT4/INT8)@4"], E2B["cells"]["KIVI (INT4/INT8)@8"]
claim("KIVI E2B 4-bit", ki4["q_abs"], f"{ki4['q_abs']:.1f}\\%")
claim("KIVI E2B 8-bit", ki8["q_abs"], f"{ki8['q_abs']:.1f}\\%")

failed = [c for c in checks if not c[3]]
for label, computed, literal, ok in checks:
    print(f"  {'ok ' if ok else 'MISS'}  {label:<34} data={computed!s:<10} tex needs {literal!r}")
print()
print(f"{len(checks) - len(failed)}/{len(checks)} claims verified against the data")
if failed:
    print("\nThe manuscript does not contain these strings:")
    for label, computed, literal, _ in failed:
        print(f"  {label}: expected {literal!r} (data says {computed})")
    sys.exit(1)
