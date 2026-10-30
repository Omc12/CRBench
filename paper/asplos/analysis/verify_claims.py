#!/usr/bin/env python3
"""Assert that the numbers in the ASPLOS manuscript match the measured data,
and that the manuscript is anonymous.

Every claim below is stated twice: once as a fact derived from
``paper_numbers.json`` / ``decoupling_stats.json``, and once as the literal
string the manuscript must contain.  A mismatch fails loudly rather than
shipping a paper whose prose has drifted from its measurements.

Two claims checked by the TMLR version of this script are deliberately absent
here: the paper no longer quotes S_sys values in prose, because S_sys is now
presented as a convenience summary rather than as evidence (see the paper's
"phi is the serving axis" paragraph).

    python paper/asplos/analysis/verify_claims.py
"""
from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ASPLOS = os.path.dirname(HERE)
TEX = os.path.join(ASPLOS, "asplos.tex")
DATADIR = os.path.join(os.path.dirname(ASPLOS), "tmlr", "analysis")

D = json.load(open(os.path.join(DATADIR, "paper_numbers.json"), encoding="utf-8"))
S = json.load(open(os.path.join(DATADIR, "decoupling_stats.json"), encoding="utf-8"))
RAW = open(TEX, encoding="utf-8").read()


def normalise(s):
    """Strip the LaTeX markup that separates a number from its prose form."""
    for a, b in (("$", ""), ("{,}", ","), ("{-}", "-"), ("{+}", "+"),
                 ("\\,", " "), ("\\%", "%"), ("\\times", "x"), ("~", " "),
                 ("{", ""), ("}", "")):
        s = s.replace(a, b)
    return " ".join(s.split())


TEXT = normalise(RAW)

M = D["models"]
E2B, E4B, Q7B = M["Gemma 4 E2B"], M["Gemma 4 E4B"], M["Qwen2.5 7B"]

checks = []


def claim(label, computed, literal):
    """`computed` is what the data says; `literal` must appear in the .tex."""
    checks.append((label, computed, literal, normalise(literal) in TEXT))


# --- matched-bitrate accuracy spread (abstract, Fig. 1a, Sec. 5.1) ----------
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

# --- the matched pair (abstract, Fig. 1b, Sec. 5.3) ------------------------
k4 = E4B["cells"]["KIVI (INT4/INT8)@4"]
s4 = E4B["cells"]["SnapKV@4"]
claim("KIVI@4 E4B b_eff", k4["b_eff"], f"{k4['b_eff']:.2f}")
claim("SnapKV@4 E4B b_eff", s4["b_eff"], f"{s4['b_eff']:.2f}")
claim("KIVI@4 E4B accuracy", k4["q_abs"], f"{k4['q_abs']:.1f}\\%")
claim("SnapKV@4 E4B accuracy", s4["q_abs"], f"{s4['q_abs']:.1f}\\%")
claim("KIVI@4 E4B TTFT", k4["ttft"], f"{k4['ttft']:.2f}")
claim("SnapKV@4 E4B TTFT", s4["ttft"], f"{s4['ttft']:.2f}")
claim("TTFT ratio", s4["ttft"] / k4["ttft"], f"{s4['ttft'] / k4['ttft']:.1f}\\times")
claim("VRAM ratio", s4["vram"] / k4["vram"], f"{s4['vram'] / k4['vram']:.2f}\\times")
claim("KIVI@4 E4B phi", k4["phi"], f"{k4['phi']:.2f}")
claim("SnapKV@4 E4B phi", s4["phi"], f"{s4['phi']:.2f}")

# The KIVI-vs-SnapKV pair ratio on Qwen is 11.7x; SnapKV's inflation against
# its *own* dense run is 11.8x.  The paper quotes both, in the right places.
k4q, s4q_ = Q7B["cells"]["KIVI (INT4/INT8)@4"], Q7B["cells"]["SnapKV@4"]
claim("KIVI/SnapKV TTFT ratio, 7B", s4q_["ttft"] / k4q["ttft"],
      f"{s4q_['ttft'] / k4q['ttft']:.1f}\\times")

# --- pooled rank correlations (abstract, Secs. 5.1 and 5.3) ----------------
P = S["pooled"]
claim("rho(R_mem, accuracy)", P["rho_r_mem_q_abs"], f"{P['rho_r_mem_q_abs']:.2f}")
claim("rho(R_mem, phi)", P["rho_r_mem_phi"], f"{P['rho_r_mem_phi']:.2f}")
claim("rho(b_eff, TTFT)", P["rho_b_eff_ttft_x_dense"], f"{P['rho_b_eff_ttft_x_dense']:.2f}")
claim("rho(b_eff, VRAM)", P["rho_b_eff_vram_x_dense"], f"{abs(P['rho_b_eff_vram_x_dense']):.2f}")

# --- metadata accounting (abstract, Sec. 5.2) ------------------------------
dkv, dkvh = E4B["cells"]["DKV-Mid"], E4B["cells"]["DKV-High"]
claim("DKV-Mid E4B payload", dkv["b_pay"], f"{dkv['b_pay']:.2f}")
claim("DKV-Mid E4B effective", dkv["b_eff"], f"{dkv['b_eff']:.2f}")
claim("DKV-High E4B payload", dkvh["b_pay"], f"{dkvh['b_pay']:.2f}")
claim("DKV-High E4B effective", dkvh["b_eff"], f"{dkvh['b_eff']:.2f}")
claim("KIVI@4 E4B payload", k4["b_pay"], f"{k4['b_pay']:.2f}")

# --- the model-dependent serving penalty (intro, Sec. 5.3) -----------------
s4q, s4e = Q7B["cells"]["SnapKV@4"], E2B["cells"]["SnapKV@4"]
claim("SnapKV 7B TTFT x dense", s4q["ttft_x_dense"], f"{s4q['ttft_x_dense']:.1f}\\times")
claim("SnapKV E2B TTFT x dense", s4e["ttft_x_dense"], f"{s4e['ttft_x_dense']:.2f}\\times")
claim("SnapKV E2B VRAM x dense", s4e["vram_x_dense"], f"{s4e['vram_x_dense']:.2f}\\times")
claim("SnapKV E2B phi", s4e["phi"], f"{s4e['phi']:.2f}")

# --- rank displacement and per-model correlations (Sec. 5.3) ---------------
for lbl, short in (("Gemma 4 E2B", "E2B"), ("Gemma 4 E4B", "E4B"),
                   ("Qwen2.5 7B", "7B")):
    pm = S["per_model"][lbl]
    claim(f"{short} mean rank shift", pm["mean_rank_shift"], f"{pm['mean_rank_shift']:.1f}")
    claim(f"{short} rho(S_res, phi)", pm["rho_sres_phi"], f"{abs(round(pm['rho_sres_phi'], 2)):.2f}")

# --- design accounting (Sec. 4) --------------------------------------------
per_model = E2B["n_records"]
total = sum(m["n_records"] for m in M.values())
claim("paired executions per model", per_model, f"{per_model:,}".replace(",", "{,}"))
claim("total paired executions", total, f"{total:,}".replace(",", "{,}"))
claim("distinct queries per model", E2B["n_unique_queries"], str(E2B["n_unique_queries"]))
claim("configurations per query", per_model // E2B["n_unique_queries"],
      f"${per_model // E2B['n_unique_queries']}$ configurations")

# --- budget sensitivity (Sec. 5.1) -----------------------------------------
sl4, sl8 = E2B["cells"]["StreamingLLM@4"], E2B["cells"]["StreamingLLM@8"]
claim("StreamingLLM E2B gain", sl8["q_abs"] - sl4["q_abs"],
      f"${sl8['q_abs'] - sl4['q_abs']:.1f}$ points")
ki4, ki8 = E2B["cells"]["KIVI (INT4/INT8)@4"], E2B["cells"]["KIVI (INT4/INT8)@8"]
claim("KIVI E2B 4-bit", ki4["q_abs"], f"{ki4['q_abs']:.1f}\\%")
claim("KIVI E2B 8-bit", ki8["q_abs"], f"{ki8['q_abs']:.1f}\\%")

# --- metadata ranges (abstract, intro, Secs. 5.2 and 6, conclusion) --------
# These were wrong in an earlier draft and unchecked, so they are checked now.
# "meta_share" is the metadata's share of the realised footprint, which is what
# the "meta" column of the three-axis table reports.
def shares(pred):
    return [c["meta_share"] for m in M.values() for k, c in m["cells"].items()
            if c["method"] != "Dense FP16" and pred(k, c)]


carry = shares(lambda k, c: c["meta_share"] > 1.0)          # families with metadata
claim("metadata share, low end", min(carry), f"${round(min(carry))}$--")
claim("metadata share, high end", max(carry), f"--${round(max(carry))}\\%$")

quant = shares(lambda k, c: k.startswith("KIVI"))
claim("quantisation share range", (min(quant), max(quant)),
      f"${round(min(quant))}$--${round(max(quant))}\\%$ for grouped quantisation")

fac_gemma = shares(lambda k, c: k.startswith("DKV"))        # all three models
claim("factorised share range", (min(fac_gemma), max(fac_gemma)),
      f"${round(min(fac_gemma))}$--${round(max(fac_gemma))}\\%$ for factorised state")

# --- accuracy orderings quoted in Sec. 5.1 ---------------------------------
FIXED = ["KIVI (INT4/INT8)", "SnapKV", "Low-Rank KV", "StreamingLLM", "KV-Merging"]


def rank_of(model, method, budget):
    cells = M[model]["cells"]
    order = sorted(FIXED, key=lambda k: -cells[f"{k}@{budget}"]["q_abs"])
    return order.index(method) + 1


for lbl, model, method, budget, expect in (
        ("SnapKV rank, E2B@4", "Gemma 4 E2B", "SnapKV", 4, 1),
        ("SnapKV rank, E4B@4", "Gemma 4 E4B", "SnapKV", 4, 2),
        ("SnapKV rank, 7B@4", "Qwen2.5 7B", "SnapKV", 4, 2),
        ("Low-Rank rank, E2B@4", "Gemma 4 E2B", "Low-Rank KV", 4, 3),
        ("Low-Rank rank, E4B@4", "Gemma 4 E4B", "Low-Rank KV", 4, 3),
        ("Low-Rank rank, 7B@4", "Qwen2.5 7B", "Low-Rank KV", 4, 4),
        ("StreamLLM rank, E4B@8", "Gemma 4 E4B", "StreamingLLM", 8, 5),
        ("StreamLLM rank, 7B@8", "Qwen2.5 7B", "StreamingLLM", 8, 3)):
    got = rank_of(model, method, budget)
    checks.append((lbl, f"rank {got}", f"prose says rank {expect}", got == expect))

sl8e4b = M["Gemma 4 E4B"]["cells"]["StreamingLLM@8"]["q_abs"]
sl8q = M["Qwen2.5 7B"]["cells"]["StreamingLLM@8"]["q_abs"]
claim("StreamLLM E4B 8-bit", sl8e4b, f"{sl8e4b:.1f}\\%")
claim("StreamLLM 7B 8-bit", sl8q, f"{sl8q:.1f}\\%")

# --- operating-point count (abstract, Sec. 5.3) ----------------------------
n_ops = sum(1 for m in M.values() for c in m["cells"].values()
            if c["method"] != "Dense FP16")
claim("compressed operating points", n_ops, f"{n_ops} operating points")

failed = [c for c in checks if not c[3]]
for label, computed, literal, ok in checks:
    print(f"  {'ok ' if ok else 'MISS'}  {label:<34} data={computed!s:<10} tex needs {literal!r}")
print()
print(f"{len(checks) - len(failed)}/{len(checks)} numeric claims verified against the data")

# --- anonymity ------------------------------------------------------------
# The submission is double-blind.  These are the identifying strings that the
# public versions of this work carry; none may appear in the manuscript.
# "Chimurkar" is the sole exception: the CFP requires that one's own public
# prior work be cited "as written by a third party", so the DKV citation in
# refs.bib carries its real author.  It must not appear in asplos.tex.
FORBIDDEN = [
    # Personal and institutional identifiers.
    "Chimurkar", "Rishihood", "Newton School", "omchimurkar",
    # Repositories and archives that resolve to a page naming the author.
    "Omc12", "CRBench", "Differential-KV", "zenodo", "10.5281", "huggingface",
    # Redaction markers the ASPLOS CFP forbids.
    "reference removed", "removed for blind", "withheld for",
    "anonymous authors",
]
# A third-party repository URL is not a leak; only our own identifiers are.
# gkamradt's Needle-in-a-Haystack repo is cited legitimately, so "github.com"
# is deliberately not in the list above -- "Omc12" catches the one that would
# matter.
# Both of these are uploaded to the submission site and are reviewer-visible,
# so both must be anonymous.
DOCS = {"asplos.tex": RAW}
for extra in ("refs.bib", "asplos.bbl", "changes_note.tex"):
    path = os.path.join(ASPLOS, extra)
    if os.path.exists(path):
        DOCS[extra] = open(path, encoding="utf-8", errors="replace").read()

print()
leaks = {}
for name, body in DOCS.items():
    hits = [w for w in FORBIDDEN if re.search(re.escape(w), body, re.IGNORECASE)]
    if hits:
        leaks[name] = hits
        print(f"ANONYMITY: {len(hits)} identifying string(s) in {name}: {hits}")
    else:
        print(f"ANONYMITY: {name} clean")

if failed:
    print("\nThe manuscript does not contain these strings:")
    for label, computed, literal, _ in failed:
        print(f"  {label}: expected {literal!r} (data says {computed})")
if failed or leaks:
    sys.exit(1)
