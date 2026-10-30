#!/usr/bin/env python3
"""Decoupling analysis: the evidence behind the paper's central claim.

Consumes ``paper_numbers.json`` (written by ``build_tables.py``) and quantifies
how weakly the three axes predict one another:

    representation efficiency  ->  task utility  ->  serving efficiency

No new model executions are performed; this is a re-analysis of the 3,150
paired query executions already collected.

Run ``build_tables.py`` first, then this script.
"""
from __future__ import annotations

import itertools
import json
import os

from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
NUMBERS = os.path.join(HERE, "paper_numbers.json")

# operating points that share a controlled nominal budget
FIXED = ["KIVI (INT4/INT8)", "SnapKV", "Low-Rank KV", "StreamingLLM", "KV-Merging"]


def load():
    with open(NUMBERS, encoding="utf-8") as fh:
        return json.load(fh)


def points(model, include_dense=True):
    """All operating points for one model as a flat list."""
    out = []
    for key, c in model["cells"].items():
        if not include_dense and c["method"] == "Dense FP16":
            continue
        out.append((key, c))
    return out


def rank(vals, higher_is_better=True):
    order = sorted(range(len(vals)), key=lambda i: -vals[i] if higher_is_better else vals[i])
    rk = [0] * len(vals)
    for pos, i in enumerate(order, 1):
        rk[i] = pos
    return rk


def main():
    data = load()
    res = {"note": "re-analysis of already-collected runs; no new executions",
           "per_model": {}, "pooled": {}}

    print("=" * 84)
    print("A. Matched nominal bitrate, unmatched task utility")
    print("=" * 84)
    for k, v in data["spread"].items():
        print(f"  {k:<22} {v['n']} methods within [{v['b_min']}, {v['b_max']}] bits/elem "
              f"-> {v['acc_lo']}% ({v['method_lo']}) .. {v['acc_hi']}% ({v['method_hi']});"
              f" span {v['span']} pts, ratio {v['ratio']}x")
    res["matched_bitrate_spread"] = data["spread"]

    print()
    print("=" * 84)
    print("B. Does memory saving predict accuracy?  (rank correlation over")
    print("   operating points; dense reference excluded)")
    print("=" * 84)
    for label, model in data["models"].items():
        pts = points(model, include_dense=False)
        rmem = [c["r_mem"] for _, c in pts]
        acc = [c["q_abs"] for _, c in pts]
        rho, p = spearmanr(rmem, acc)
        print(f"  {label:<14} rho(R_mem, Q_abs) = {rho:+.3f}  (p = {p:.3f}, n = {len(pts)})")
        res["per_model"].setdefault(label, {}).update(
            rho_rmem_acc=round(float(rho), 3), p_rmem_acc=round(float(p), 4),
            n_points=len(pts))

    print()
    print("=" * 84)
    print("C. Does algorithmic utility predict serving efficiency?")
    print("   rho is taken against phi, the pure hardware term.  S_sys is NOT used")
    print("   here: S_sys = a*S_res + (1-a)*100*phi embeds S_res by construction, so")
    print("   rho(S_res, S_sys) is circular and says nothing.  The S_res -> S_sys rank")
    print("   shift is still reported, as a LOWER bound on the disagreement.")
    print("=" * 84)
    for label, model in data["models"].items():
        pts = points(model)
        names = [k for k, _ in pts]
        sres = [c["s_res"] for _, c in pts]
        ssys = [c["s_sys"] for _, c in pts]
        phi = [c["phi"] for _, c in pts]
        rho, p = spearmanr(sres, phi)
        r1, r2 = rank(sres), rank(ssys)
        shift = [abs(a - b) for a, b in zip(r1, r2)]
        worst = max(range(len(names)), key=lambda i: shift[i])
        print(f"  {label:<14} rho(S_res, phi) = {rho:+.3f} (p = {p:.3f}) | "
              f"S_res->S_sys mean |rank shift| = {sum(shift) / len(shift):.2f}, "
              f"max = {shift[worst]} ({names[worst]}: #{r1[worst]} -> #{r2[worst]})")
        res["per_model"][label].update(
            rho_sres_phi=round(float(rho), 3), p_sres_phi=round(float(p), 4),
            mean_rank_shift=round(sum(shift) / len(shift), 2),
            max_rank_shift=int(shift[worst]), max_rank_shift_point=names[worst])

    print()
    print("=" * 84)
    print("D. Does the compressed footprint predict prefill cost?")
    print("=" * 84)
    for label, model in data["models"].items():
        pts = points(model, include_dense=False)
        beff = [c["b_eff"] for _, c in pts]
        ttft = [c["ttft_x_dense"] for _, c in pts]
        rho, p = spearmanr(beff, ttft)
        print(f"  {label:<14} rho(b_eff, TTFT/dense) = {rho:+.3f} (p = {p:.3f})")
        res["per_model"][label].update(
            rho_beff_ttft=round(float(rho), 3), p_beff_ttft=round(float(p), 4))

    print()
    print("=" * 84)
    print("E. Pairs matched on BOTH footprint and accuracy, split on serving cost")
    print("=" * 84)
    res["matched_pairs"] = {}
    for label, model in data["models"].items():
        pts = points(model, include_dense=False)
        for (ka, ca), (kb, cb) in itertools.combinations(pts, 2):
            if abs(ca["b_eff"] - cb["b_eff"]) > 0.6:
                continue
            if abs(ca["q_abs"] - cb["q_abs"]) > 4.0:
                continue
            t = max(ca["ttft"], cb["ttft"]) / min(ca["ttft"], cb["ttft"])
            v = max(ca["vram"], cb["vram"]) / min(ca["vram"], cb["vram"])
            if t < 1.5 and v < 1.5:
                continue
            slower = ka if ca["ttft"] > cb["ttft"] else kb
            print(f"  {label:<14} {ka} vs {kb}: "
                  f"d_b_eff = {abs(ca['b_eff'] - cb['b_eff']):.2f} bits, "
                  f"d_acc = {abs(ca['q_abs'] - cb['q_abs']):.1f} pts "
                  f"-> TTFT {t:.1f}x, VRAM {v:.1f}x, "
                  f"d_S_sys = {abs(ca['s_sys'] - cb['s_sys']):.1f} (slower: {slower})")
            res["matched_pairs"].setdefault(label, []).append(dict(
                a=ka, b=kb, d_beff=round(abs(ca["b_eff"] - cb["b_eff"]), 2),
                d_acc=round(abs(ca["q_abs"] - cb["q_abs"]), 1),
                ttft_ratio=round(t, 1), vram_ratio=round(v, 1),
                d_ssys=round(abs(ca["s_sys"] - cb["s_sys"]), 1), slower=slower))

    print()
    print("=" * 84)
    print("F. Pooled across all three models")
    print("=" * 84)
    allpts = [c for m in data["models"].values() for _, c in points(m, include_dense=False)]
    for x, y, lab in (("r_mem", "q_abs", "memory saving vs accuracy"),
                      ("r_mem", "phi", "memory saving vs hardware multiplier"),
                      ("q_abs", "phi", "accuracy vs hardware multiplier"),
                      ("s_res", "phi", "S_res vs hardware multiplier"),
                      ("b_eff", "ttft_x_dense", "b_eff vs TTFT inflation"),
                      ("b_eff", "vram_x_dense", "b_eff vs peak-VRAM inflation")):
        rho, p = spearmanr([c[x] for c in allpts], [c[y] for c in allpts])
        print(f"  rho({lab:38s}) = {rho:+.3f}  (p = {p:.4f}, n = {len(allpts)})")
        res["pooled"][f"rho_{x}_{y}"] = round(float(rho), 3)
        res["pooled"][f"p_{x}_{y}"] = round(float(p), 4)

    print()
    print("=" * 84)
    print("G. Metadata share of the realised footprint")
    print("=" * 84)
    res["metadata"] = {}
    for label, model in data["models"].items():
        for k, c in sorted(model["cells"].items(), key=lambda kv: -kv[1]["meta_share"]):
            if c["meta_share"] < 0.5:
                continue
            print(f"  {label:<14} {k:<24} payload {c['b_pay']:5.2f} -> effective "
                  f"{c['b_eff']:5.2f} bits/elem ({c['meta_share']:.1f}% metadata)")
            res["metadata"].setdefault(label, {})[k] = dict(
                b_pay=round(c["b_pay"], 2), b_eff=round(c["b_eff"], 2),
                meta_share=round(c["meta_share"], 1))

    print()
    print("=" * 84)
    print("H. Does charging for metadata reorder the Part 1 table?")
    print("=" * 84)
    res["metadata_reranking"] = {}
    for label, model in data["models"].items():
        cells = model["cells"]
        pay = {k: i + 1 for i, k in enumerate(
            sorted(cells, key=lambda k: -cells[k]["s_res_payload_only"]))}
        met = {k: i + 1 for i, k in enumerate(
            sorted(cells, key=lambda k: -cells[k]["s_res"]))}
        moves = [(k, pay[k], met[k]) for k in cells if pay[k] != met[k]]
        dense = cells["Dense FP16"]["s_res"]
        for k, a, b in sorted(moves, key=lambda t: -abs(t[1] - t[2])):
            print(f"  {label:<14} {k:<24} rank {a:2d} -> {b:2d}   "
                  f"S_res {cells[k]['s_res_payload_only']:5.1f} -> {cells[k]['s_res']:5.1f}")
            res["metadata_reranking"].setdefault(label, []).append(dict(
                point=k, rank_payload=a, rank_effective=b,
                s_res_payload=round(cells[k]["s_res_payload_only"], 1),
                s_res_effective=round(cells[k]["s_res"], 1)))
        below = [k for k in cells if cells[k]["s_res"] < dense
                 and cells[k]["s_res_payload_only"] >= dense]
        print(f"  {label:<14} configurations pushed below dense by the correction: "
              f"{below or 'none'}")

    dest = os.path.join(HERE, "decoupling_stats.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print()
    print("wrote", dest)


if __name__ == "__main__":
    main()
