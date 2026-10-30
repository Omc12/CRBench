#!/usr/bin/env python3
"""Render every ASPLOS-submission figure from the study's analysis outputs.

    python paper/asplos/analysis/make_figures.py

Nothing here measures anything.  Both JSON inputs are produced by the
existing pipeline (build_tables.py, decoupling_analysis.py) from the
released per-query records; this script only lays those numbers out at the
sizes and point sizes the ASPLOS format requires.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from figstyle import (COLWIDTH, FIGDIR, GRID, INK, METHOD_COLOR, MODEL_COLOR,
                      MODEL_MARKER, MUTED, SHORT, TEXTWIDTH, numbers,
                      panel_label, save, stats, tidy, use_style)

MODELS = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen2.5 7B"]
FIXED = ["KIVI (INT4/INT8)", "SnapKV", "Low-Rank KV", "StreamingLLM", "KV-Merging"]
ADAPTIVE = ["DKV-Mid", "DKV-High"]

# One canonical row order, shared by every panel of every multi-model figure,
# so that a reader compares like with like across models by eye.
def canonical_rows(include_dense=False):
    rows = []
    for m in FIXED:
        for b in (4, 8):
            rows.append((f"{SHORT[m]} {b}", f"{m}@{b}", m))
    for m in ADAPTIVE:
        rows.append((SHORT[m], m, m))
    if include_dense:
        rows.append((SHORT["Dense FP16"], "Dense FP16", "Dense FP16"))
    return rows


def compressed_points(D):
    """(model, key, cell) for every compressed operating point."""
    out = []
    for m in MODELS:
        for k, c in D["models"][m]["cells"].items():
            if c["method"] != "Dense FP16":
                out.append((m, k, c))
    return out


# ---------------------------------------------------------------------------
# Figure 1 -- the headline, and the only figure the rapid round will see
# ---------------------------------------------------------------------------
def fig_decoupling(D, S):
    fig = plt.figure(figsize=(TEXTWIDTH, 3.25))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.12, 0.94, 0.94], wspace=0.62,
                          left=0.132, right=0.972, top=0.795, bottom=0.30)

    # --- (a) same budget, very different task utility -----------------------
    ax = fig.add_subplot(gs[0, 0])
    tidy(ax, grid_axis="x")
    for yi, model in enumerate(MODELS):
        y = len(MODELS) - 1 - yi
        cells = D["models"][model]["cells"]
        pts = [(m, cells[f"{m}@4"]["q_abs"]) for m in FIXED if f"{m}@4" in cells]
        xs = [p[1] for p in pts]
        ax.plot([min(xs), max(xs)], [y, y], color=GRID, lw=3.4,
                solid_capstyle="round", zorder=1)
        for m, x in pts:
            ax.scatter([x], [y], s=38, color=METHOD_COLOR[m], zorder=3,
                       edgecolor="white", linewidth=0.7)
        lo = min(pts, key=lambda p: p[1])
        hi = max(pts, key=lambda p: p[1])
        pct = lambda v: f"{v:.1f}%" if v < 10 else f"{v:.0f}%"
        ax.annotate(pct(hi[1]), (hi[1], y), textcoords="offset points",
                    xytext=(7, 0), va="center", fontsize=8.0, color=INK)
        ax.annotate(pct(lo[1]), (lo[1], y), textcoords="offset points",
                    xytext=(-7, 0), va="center", ha="right", fontsize=8.0, color=INK)
        span = hi[1] / lo[1] if lo[1] else float("nan")
        ax.annotate(rf"${span:.1f}\times$", (np.mean(xs), y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.0, color=MUTED)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(list(reversed(MODELS)), fontsize=8.0, color=INK)
    ax.set_ylim(-0.5, 2.62)
    ax.set_xlim(-23, 82)
    ax.set_xlabel("Task accuracy (%)")
    ax.set_title("(a) Identical 4-bit budget,\n$18.7\\times$ spread in task utility", pad=7)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=5.0,
                      markerfacecolor=METHOD_COLOR[m], markeredgecolor="white",
                      markeredgewidth=0.6, label=SHORT[m]) for m in FIXED]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.46, -0.30),
              ncol=2, handletextpad=0.25, columnspacing=0.9, borderpad=0.0,
              labelspacing=0.30)

    # --- (b) same budget AND same accuracy, split serving cost --------------
    ax = fig.add_subplot(gs[0, 1])
    tidy(ax, grid_axis="x")
    c_k = D["models"]["Gemma 4 E4B"]["cells"]["KIVI (INT4/INT8)@4"]
    c_s = D["models"]["Gemma 4 E4B"]["cells"]["SnapKV@4"]
    rows = [
        ("Bits/elem", c_s["b_eff"] / c_k["b_eff"]),
        ("Accuracy", c_s["q_abs"] / c_k["q_abs"]),
        ("Peak VRAM", c_s["vram"] / c_k["vram"]),
        ("TTFT", c_s["ttft"] / c_k["ttft"]),
    ]
    ys = np.arange(len(rows))
    for y, (_, v) in zip(ys, rows):
        serving = y >= 2
        ax.barh(y, v, height=0.58, color="#D55E00" if serving else "#B8B8B8",
                zorder=2, alpha=0.95 if serving else 0.85)
        ax.annotate(rf"${v:.2f}\times$" if v < 10 else rf"${v:.1f}\times$",
                    (v, y), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=8.0, color=INK)
    ax.axvline(1.0, color=MUTED, lw=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 320)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["$1\\times$", "$10\\times$", "$100\\times$"])
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.0, color=INK)
    ax.set_xlabel("SnapKV $\\div$ KIVI, Gemma 4 E4B")
    ax.set_title("(b) Matched on both reported\nnumbers, $30\\times$ the prefill", pad=7)
    ax.annotate("grey: what a paper reports\norange: what serving costs",
                xy=(0.02, -0.40), xycoords="axes fraction", va="top", ha="left",
                fontsize=8.0, color=INK)

    # --- (c) footprint carries no information about prefill cost ------------
    ax = fig.add_subplot(gs[0, 2])
    tidy(ax, grid_axis="both")
    for model, key, c in compressed_points(D):
        ax.scatter([c["b_eff"]], [c["ttft_x_dense"]], s=30,
                   marker=MODEL_MARKER[model], color=METHOD_COLOR[c["method"]],
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("Effective bits / element")
    ax.set_ylabel("TTFT $\\div$ paired dense")
    rho = S["pooled"]["rho_b_eff_ttft_x_dense"]
    p = S["pooled"]["p_b_eff_ttft_x_dense"]
    ax.set_title("(c) Footprint does not\ntrack prefill cost", pad=7)
    ax.annotate(rf"$\rho = {rho:+.2f}$, $p = {p:.2f}$", xy=(0.97, 0.86),
                xycoords="axes fraction", ha="right", va="top", fontsize=8.0,
                color=INK,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9))
    ax.set_xlim(2.5, 15)
    mh = [Line2D([], [], marker=MODEL_MARKER[m], linestyle="", markersize=5.0,
                 markerfacecolor=MUTED, markeredgecolor="white", markeredgewidth=0.5,
                 label=m) for m in MODELS]
    ax.legend(handles=mh, loc="upper center", bbox_to_anchor=(0.44, -0.30),
              ncol=1, handletextpad=0.25, borderpad=0.0, labelspacing=0.30)

    save(fig, "fig1_decoupling")


# ---------------------------------------------------------------------------
# Figure: the paired-execution primitive
# ---------------------------------------------------------------------------
def fig_framework():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.30))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.99, bottom=0.01)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    # Point sizes here are absolute (the axes is a fixed 100x40 canvas at a
    # fixed figure width), so 8.0 really is 8pt on paper.
    def box(x, y, w, h, title, body, edge, face="white", tsize=8.0, bsize=8.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.0,rounding_size=1.1",
                                    linewidth=0.9, edgecolor=edge, facecolor=face,
                                    zorder=2))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top",
                fontsize=tsize, fontweight="bold", color=INK, zorder=3)
        if body:
            ax.text(x + w / 2, y + h - 7.4, body, ha="center", va="top",
                    fontsize=bsize, color=INK, zorder=3, linespacing=1.40)

    def arrow(x0, y0, x1, y1, color=MUTED):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.9, color=color,
                                     shrinkA=0, shrinkB=0, zorder=1))

    blue, orange, green = "#0072B2", "#D55E00", "#009E73"

    box(0.0, 12.0, 14.0, 16.0, "One query",
        "$(x_i,\\ y_i)$\ntask $\\tau_i$\nlength $L_i$", MUTED, "#F4F4F4")

    box(18.5, 22.0, 22.5, 16.0, "Dense reference",
        "$\\mathcal{M}(x_i \\mid \\mathcal{R}_{\\mathrm{dense}})$\n16 bits/element", MUTED)
    box(18.5, 2.0, 22.5, 16.0, "Candidate",
        "$\\mathcal{M}(x_i \\mid \\mathcal{A}(\\mathcal{B}))$\nsame prompt,\nsame session", blue)

    arrow(14.0, 23.0, 18.5, 29.0)
    arrow(14.0, 17.0, 18.5, 11.0)

    box(46.5, 26.5, 25.5, 12.0, "Axis 1 $\\cdot$ Representation",
        "$b_{\\mathrm{eff}}$ = payload + metadata", green)
    box(46.5, 14.0, 25.5, 12.0, "Axis 2 $\\cdot$ Task",
        "$Q_{\\mathrm{abs}}$, paired $\\Delta_i$", green)
    box(46.5, 1.5, 25.5, 12.0, "Axis 3 $\\cdot$ Serving",
        "TTFT, decode, peak VRAM", orange)

    arrow(41.0, 30.0, 46.5, 32.5)
    arrow(41.0, 10.0, 46.5, 20.0)
    arrow(41.0, 10.0, 46.5, 7.5)

    box(78.0, 20.5, 21.5, 15.0, "Part 1",
        "$\\mathcal{S}_{\\mathrm{res}}$, coverage $C$", green)
    box(78.0, 1.5, 21.5, 15.0, "Part 2",
        "$\\phi$ and its\ncomponents", orange)

    arrow(72.0, 32.5, 78.0, 29.0)
    arrow(72.0, 20.0, 78.0, 25.0)
    arrow(72.0, 20.0, 78.0, 11.0)
    arrow(72.0, 7.5, 78.0, 8.5)

    save(fig, "fig2_framework")


# ---------------------------------------------------------------------------
# Figure: nominal payload vs effective bits (why metadata must be counted)
# ---------------------------------------------------------------------------
def fig_effective_bits(D):
    rows = canonical_rows()
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.95), sharey=True)
    fig.subplots_adjust(left=0.128, right=0.995, top=0.885, bottom=0.135, wspace=0.10)
    ys = np.arange(len(rows))
    for ax, model in zip(axes, MODELS):
        tidy(ax, grid_axis="x")
        cells = D["models"][model]["cells"]
        for y, (_, key, meth) in zip(ys, rows):
            c = cells[key]
            col = METHOD_COLOR[meth]
            ax.barh(y, c["b_pay"], height=0.66, color=col, zorder=2, linewidth=0)
            ax.barh(y, c["b_eff"] - c["b_pay"], height=0.66, left=c["b_pay"],
                    color=col, alpha=0.42, zorder=2, linewidth=0,
                    hatch="////", edgecolor="white")
            if c["meta_share"] >= 5:
                ax.annotate(f"+{c['meta_share']:.0f}%", (c["b_eff"], y),
                            textcoords="offset points", xytext=(3, 0),
                            va="center", fontsize=8.0, color=INK)
        ax.set_xlim(0, 17.8)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_title(model, pad=5)
        ax.set_xlabel("Bits / element")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=8.0, color=INK)
    axes[0].annotate("solid: payload\nhatched: mandatory metadata",
                     xy=(8.6, 2.30), va="top", ha="left",
                     fontsize=8.0, color=MUTED)
    save(fig, "fig3_effective_bits")


# ---------------------------------------------------------------------------
# Figure: the pure hardware term, and where it is lost
# ---------------------------------------------------------------------------
def fig_hardware(D):
    rows = canonical_rows(include_dense=True)
    fig = plt.figure(figsize=(TEXTWIDTH, 3.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.06, 0.94], wspace=0.30,
                          left=0.152, right=0.985, top=0.865, bottom=0.155)

    # --- (a) phi per operating point, one marker per model ------------------
    ax = fig.add_subplot(gs[0, 0])
    tidy(ax, grid_axis="x")
    ys = np.arange(len(rows))
    for y, (_, key, _) in zip(ys, rows):
        vals = [D["models"][m]["cells"][key]["phi"] for m in MODELS]
        ax.plot([min(vals), max(vals)], [y, y], color=GRID, lw=3.0,
                solid_capstyle="round", zorder=1)
        for m, v in zip(MODELS, vals):
            ax.scatter([v], [y], s=30, marker=MODEL_MARKER[m],
                       color=MODEL_COLOR[m], edgecolor="white", linewidth=0.6,
                       zorder=3)
    ax.axvline(1.0, color=MUTED, lw=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.0, color=INK)
    ax.set_ylim(len(rows) - 0.4, -0.8)
    ax.set_xlim(0.35, 1.10)
    ax.set_xlabel("Hardware multiplier $\\phi$  (1.0 = paired dense)")
    ax.set_title("(a) The same representation,\nthree different serving outcomes", pad=6)
    mh = [Line2D([], [], marker=MODEL_MARKER[m], linestyle="", markersize=5.0,
                 markerfacecolor=MODEL_COLOR[m], markeredgecolor="white",
                 markeredgewidth=0.5, label=m) for m in MODELS]
    ax.legend(handles=mh, loc="lower left", ncol=1, handletextpad=0.25,
              borderpad=0.0, labelspacing=0.26, bbox_to_anchor=(0.005, 0.005))

    # --- (b) which axis the loss comes from ---------------------------------
    ax = fig.add_subplot(gs[0, 1])
    tidy(ax, grid_axis="both")
    for model, key, c in compressed_points(D):
        ax.scatter([c["ttft_x_dense"]], [c["vram_x_dense"]], s=30,
                   marker=MODEL_MARKER[model], color=METHOD_COLOR[c["method"]],
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.scatter([1.0], [1.0], s=52, marker="*", color=INK, zorder=4)
    ax.annotate("paired dense", (1.0, 1.0), textcoords="offset points",
                xytext=(6, -11), fontsize=8.0, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("TTFT $\\div$ paired dense")
    ax.set_ylabel("Peak VRAM $\\div$ paired dense")
    ax.set_title("(b) Prefill-bound losses (right),\nmemory-bound losses (up)", pad=6)
    ax.annotate("SnapKV", (30.3, 3.46), textcoords="offset points",
                xytext=(-6, 6), ha="right", fontsize=8.0, color="#D55E00")
    ax.set_xlim(0.75, 90)
    ax.set_ylim(0.55, 6.2)
    ax.set_yticks([1, 2, 4])
    ax.set_yticklabels(["$1\\times$", "$2\\times$", "$4\\times$"])
    ax.set_xticks([1, 10])
    ax.set_xticklabels(["$1\\times$", "$10\\times$"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    save(fig, "fig4_hardware")


# ---------------------------------------------------------------------------
# Figure: task heatmap (single column)
# ---------------------------------------------------------------------------
def fig_task_heatmap(D):
    tasks = ["Single\nNIAH", "Multi\nNIAH", "RULER\nKV", "Multi-hop\nQA", "Var.\ntrack."]
    keys = ["Single NIAH", "Multi NIAH", "RULER KV", "Multi-Hop QA", "Variable Tracking"]
    methods = FIXED + ADAPTIVE + ["Dense FP16"]
    M = np.zeros((len(methods), len(keys)))
    for i, m in enumerate(methods):
        for j, t in enumerate(keys):
            vals = [D["models"][mo]["task_accuracy"].get(m, {}).get(t) for mo in MODELS]
            vals = [v for v in vals if v is not None]
            M[i, j] = float(np.mean(vals)) if vals else np.nan

    fig, ax = plt.subplots(figsize=(COLWIDTH, 2.75))
    fig.subplots_adjust(left=0.245, right=0.985, top=0.845, bottom=0.145)
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, fontsize=8.0)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([SHORT[m] for m in methods], fontsize=8.0)
    for i in range(len(methods)):
        for j in range(len(keys)):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.0,
                    color="white" if v > 55 else INK)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("Accuracy (%), mean of the three models", fontsize=8.5)
    save(fig, "fig5_task_heatmap")


# ---------------------------------------------------------------------------
# Figure: context scaling on the 2K-128K ladder (single column)
# ---------------------------------------------------------------------------
def fig_context(D):
    fig, ax = plt.subplots(figsize=(COLWIDTH, 3.20))
    fig.subplots_adjust(left=0.155, right=0.985, top=0.985, bottom=0.340)
    tidy(ax, grid_axis="both")
    ca = D["models"]["Gemma 4 E2B"]["context_accuracy"]
    for m in FIXED + ADAPTIVE + ["Dense FP16"]:
        if m not in ca:
            continue
        xs = sorted(int(k) for k in ca[m])
        ys = [ca[m][str(x)] for x in xs]
        dense = m == "Dense FP16"
        ax.plot(xs, ys, marker="o", markersize=3.2, lw=2.0 if dense else 1.2,
                color=METHOD_COLOR[m], linestyle="--" if dense else "-",
                zorder=4 if dense else 3, label=SHORT[m])
    ax.set_xscale("log", base=2)
    ax.set_xticks([2048, 8192, 32768, 65536, 131072])
    ax.set_xticklabels(["2K", "8K", "32K", "64K", "128K"])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Task accuracy (%)")
    ax.set_ylim(-3, 92)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.46, -0.20),
              handletextpad=0.35, columnspacing=0.85, labelspacing=0.28,
              handlelength=1.6, borderpad=0.0)
    save(fig, "fig6_context_scaling")


# ---------------------------------------------------------------------------
# Appendix figure: Part 1 utility at every operating point
# ---------------------------------------------------------------------------
def fig_part1(D):
    rows = canonical_rows(include_dense=True)
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 3.05), sharey=True)
    fig.subplots_adjust(left=0.125, right=0.995, top=0.90, bottom=0.13, wspace=0.10)
    ys = np.arange(len(rows))
    for ax, model in zip(axes, MODELS):
        tidy(ax, grid_axis="x")
        cells = D["models"][model]["cells"]
        for y, (_, key, meth) in zip(ys, rows):
            c = cells[key]
            ax.barh(y, c["s_res"], height=0.66, color=METHOD_COLOR[meth],
                    zorder=2, linewidth=0)
            ax.annotate(f"{c['s_res']:.1f}", (c["s_res"], y), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=8.0,
                        color=INK)
        dense = cells["Dense FP16"]["s_res"]
        ax.axvline(dense, color=INK, lw=0.8, linestyle=(0, (3, 2)), zorder=3)
        ax.set_ylim(len(rows) - 0.4, -0.6)
        ax.set_xlim(0, 88)
        ax.set_xticks([0, 25, 50, 75])
        ax.set_title(model, pad=5)
        ax.set_xlabel("$\\mathcal{S}_{\\mathrm{res}}$")
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=8.0, color=INK)
    save(fig, "fig7_part1")


# ---------------------------------------------------------------------------
# Appendix figure: quality-memory frontier, computed within each model
# ---------------------------------------------------------------------------
def fig_pareto(D):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.95), sharey=True)
    fig.subplots_adjust(left=0.072, right=0.962, top=0.90, bottom=0.30, wspace=0.10)
    for ax, model in zip(axes, MODELS):
        tidy(ax, grid_axis="both")
        cells = D["models"][model]["cells"]
        pts = [(k, c) for k, c in cells.items()]
        # non-dominated: nothing has both fewer bits and higher accuracy
        front = []
        for k, c in pts:
            dominated = any(o["b_eff"] <= c["b_eff"] and o["q_abs"] >= c["q_abs"]
                            and (o["b_eff"] < c["b_eff"] or o["q_abs"] > c["q_abs"])
                            for _, o in pts)
            if not dominated:
                front.append((k, c))
        for k, c in pts:
            on = any(k == fk for fk, _ in front)
            ax.scatter([c["b_eff"]], [c["q_abs"]], s=46 if on else 28,
                       color=METHOD_COLOR[c["method"]],
                       edgecolor=INK if on else "white",
                       linewidth=1.0 if on else 0.6, zorder=4 if on else 3)
        front.sort(key=lambda t: t[1]["b_eff"])
        ax.plot([c["b_eff"] for _, c in front], [c["q_abs"] for _, c in front],
                color=MUTED, lw=0.8, linestyle=(0, (3, 2)), zorder=2)
        ax.set_xlim(2.5, 17.5)
        ax.set_ylim(-4, 78)
        ax.set_title(model, pad=5)
        ax.set_xlabel("Effective bits / element")
    axes[0].set_ylabel("Task accuracy (%)")
    ch = [Line2D([], [], marker="o", linestyle="", markersize=5.0,
                 markerfacecolor=METHOD_COLOR[m], markeredgecolor="white",
                 markeredgewidth=0.6, label=SHORT[m])
          for m in FIXED + ADAPTIVE + ["Dense FP16"]]
    fig.legend(handles=ch, loc="lower center", ncol=4, fontsize=8.0,
               handletextpad=0.3, columnspacing=1.2, labelspacing=0.30,
               bbox_to_anchor=(0.53, 0.005))
    save(fig, "fig8_pareto")


# ---------------------------------------------------------------------------
# Appendix figure: sensitivity to alpha
# ---------------------------------------------------------------------------
def fig_alpha(D):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.75))
    fig.subplots_adjust(left=0.072, right=0.995, top=0.90, bottom=0.165, wspace=0.22)
    alphas = np.linspace(0.2, 1.0, 33)

    ax = axes[0]
    tidy(ax, grid_axis="both")
    cells = D["models"]["Gemma 4 E4B"]["cells"]
    for key, c in cells.items():
        ys = [a * c["q_abs"] + (1 - a) * c["r_mem"] for a in alphas]
        ax.plot(alphas, ys, lw=1.2, color=METHOD_COLOR[c["method"]],
                linestyle="--" if c["method"] == "Dense FP16" else "-")
    ax.axvline(0.70, color=INK, lw=0.8, linestyle=(0, (3, 2)))
    ax.annotate("default $\\alpha = 0.70$", (0.70, 4), textcoords="offset points",
                xytext=(-5, 0), ha="right", fontsize=8.0, color=INK)
    ax.set_xlabel("$\\alpha$  (weight on task accuracy)")
    ax.set_ylabel("$\\mathcal{S}_{\\mathrm{res}}$")
    ax.set_title("Score trajectories (Gemma 4 E4B)", pad=5)

    ax = axes[1]
    tidy(ax, grid_axis="both")
    for model in MODELS:
        cells = D["models"][model]["cells"]
        base = sorted(cells, key=lambda k: -(0.70 * cells[k]["q_abs"]
                                             + 0.30 * cells[k]["r_mem"]))
        shifts = []
        for a in alphas:
            order = sorted(cells, key=lambda k: -(a * cells[k]["q_abs"]
                                                  + (1 - a) * cells[k]["r_mem"]))
            shifts.append(float(np.mean([abs(order.index(k) - base.index(k))
                                         for k in cells])))
        ax.plot(alphas, shifts, lw=1.4, color=MODEL_COLOR[model], label=model)
    ax.axvline(0.70, color=INK, lw=0.8, linestyle=(0, (3, 2)))
    ax.set_xlabel("$\\alpha$  (weight on task accuracy)")
    ax.set_ylabel("Mean rank shift vs. $\\alpha = 0.70$")
    ax.set_title("Ranking stability", pad=5)
    ax.legend(loc="upper center", labelspacing=0.28, handlelength=1.6)

    save(fig, "fig9_alpha_sensitivity")


def main():
    use_style()
    D, S = numbers(), stats()
    print("rendering ASPLOS figures ->", FIGDIR)
    fig_decoupling(D, S)
    fig_framework()
    fig_effective_bits(D)
    fig_hardware(D)
    fig_task_heatmap(D)
    fig_context(D)
    fig_part1(D)
    fig_pareto(D)
    fig_alpha(D)


if __name__ == "__main__":
    main()
