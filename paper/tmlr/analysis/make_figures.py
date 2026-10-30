#!/usr/bin/env python3
"""Generate every figure in the CRBench manuscript from paper_numbers.json.

Run after build_tables.py and decoupling_analysis.py:

    python paper/tmlr/analysis/build_tables.py
    python paper/tmlr/analysis/decoupling_analysis.py
    python paper/tmlr/analysis/make_figures.py
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from figstyle import (FIGDIR, GRID, INK, METHOD_COLOR, MODEL_COLOR, MODEL_MARKER,
                      MUTED, SHORT, TEXTWIDTH, numbers, panel_label, save,
                      stats, tidy, use_style)

MODELS = ["Gemma 4 E2B", "Gemma 4 E4B", "Qwen2.5 7B"]
FIXED = ["KIVI (INT4/INT8)", "SnapKV", "Low-Rank KV", "StreamingLLM", "KV-Merging"]
ADAPTIVE = ["DKV-Mid", "DKV-High"]


def compressed_points(D):
    """(model, key, cell) for every compressed operating point."""
    out = []
    for m in MODELS:
        for k, c in D["models"][m]["cells"].items():
            if c["method"] != "Dense FP16":
                out.append((m, k, c))
    return out


# ---------------------------------------------------------------------------
# Figure 1 -- the headline: three axes that do not substitute
# ---------------------------------------------------------------------------
def fig_decoupling(D, S):
    fig = plt.figure(figsize=(TEXTWIDTH, 2.62))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.92, 0.93], wspace=0.42,
                          left=0.055, right=0.995, top=0.80, bottom=0.16)

    # --- (a) matched budget, unmatched task utility -------------------------
    ax = fig.add_subplot(gs[0, 0])
    tidy(ax, grid_axis="x")
    for yi, model in enumerate(MODELS):
        y = len(MODELS) - 1 - yi
        cells = D["models"][model]["cells"]
        pts = [(m, cells[f"{m}@4"]["q_abs"]) for m in FIXED if f"{m}@4" in cells]
        xs = [p[1] for p in pts]
        ax.plot([min(xs), max(xs)], [y, y], color=GRID, lw=3.0,
                solid_capstyle="round", zorder=1)
        for m, x in pts:
            ax.scatter([x], [y], s=34, color=METHOD_COLOR[m], zorder=3,
                       edgecolor="white", linewidth=0.7)
        lo = min(pts, key=lambda p: p[1])
        hi = max(pts, key=lambda p: p[1])
        pct = lambda v: f"{v:.1f}%" if v < 10 else f"{v:.0f}%"
        ax.annotate(pct(hi[1]), (hi[1], y), textcoords="offset points",
                    xytext=(6, 0), va="center", fontsize=6.8, color=INK)
        ax.annotate(pct(lo[1]), (lo[1], y), textcoords="offset points",
                    xytext=(-6, 0), va="center", ha="right", fontsize=6.8, color=INK)
        span = hi[1] / lo[1] if lo[1] else float("nan")
        ax.annotate(rf"${span:.1f}\times$", (np.mean(xs), y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=7.0, color=MUTED)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels(list(reversed(MODELS)), fontsize=7.2, color=INK)
    ax.set_ylim(-0.45, 2.52)
    ax.set_xlim(-14, 78)
    ax.set_xlabel("Task accuracy (%)")
    ax.set_title("Same memory budget,\nvery different task utility", fontsize=8.0, pad=6)
    panel_label(ax, "(a)", dx=-0.30)

    handles = [Line2D([], [], marker="o", linestyle="", markersize=4.6,
                      markerfacecolor=METHOD_COLOR[m], markeredgecolor="white",
                      markeredgewidth=0.6, label=SHORT[m]) for m in FIXED]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.48, -0.53),
              ncol=3, handletextpad=0.25, columnspacing=0.9, borderpad=0.0)

    # --- (b) matched budget AND accuracy, split serving cost -----------------
    ax = fig.add_subplot(gs[0, 1])
    tidy(ax, grid_axis="x")
    c_k = D["models"]["Gemma 4 E4B"]["cells"]["KIVI (INT4/INT8)@4"]
    c_s = D["models"]["Gemma 4 E4B"]["cells"]["SnapKV@4"]
    rows = [
        ("Effective bits", c_s["b_eff"] / c_k["b_eff"]),
        ("Task accuracy", c_s["q_abs"] / c_k["q_abs"]),
        ("Peak VRAM", c_s["vram"] / c_k["vram"]),
        ("TTFT", c_s["ttft"] / c_k["ttft"]),
    ]
    ys = np.arange(len(rows))
    for y, (_, v) in zip(ys, rows):
        serving = y >= 2
        ax.barh(y, v, height=0.56, color="#D55E00" if serving else "#B8B8B8",
                zorder=2, alpha=0.95 if serving else 0.85)
        ax.annotate(rf"${v:.2f}\times$" if v < 10 else rf"${v:.1f}\times$",
                    (v, y), textcoords="offset points", xytext=(4, 0),
                    va="center", fontsize=6.9, color=INK)
    ax.axvline(1.0, color=MUTED, lw=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.55, 130)
    ax.set_xticks([1, 10, 100])
    ax.set_xticklabels(["$1\\times$", "$10\\times$", "$100\\times$"])
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7.2, color=INK)
    ax.set_xlabel("SnapKV relative to KIVI\n(Gemma 4 E4B, 4 bpt budget)")
    ax.set_title("Same budget, same accuracy,\n$30\\times$ the prefill cost",
                 fontsize=8.0, pad=6)
    panel_label(ax, "(b)", dx=-0.34)

    # --- (c) footprint carries no information about prefill cost ------------
    ax = fig.add_subplot(gs[0, 2])
    tidy(ax, grid_axis="both")
    for model, key, c in compressed_points(D):
        ax.scatter([c["b_eff"]], [c["ttft_x_dense"]], s=26,
                   marker=MODEL_MARKER[model], color=METHOD_COLOR[c["method"]],
                   edgecolor="white", linewidth=0.6, zorder=3)
    ax.axhline(1.0, color=MUTED, lw=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("Effective bits / element")
    ax.set_ylabel("TTFT $\\div$ paired dense")
    rho = S["pooled"]["rho_b_eff_ttft_x_dense"]
    p = S["pooled"]["p_b_eff_ttft_x_dense"]
    ax.set_title("Footprint predicts nothing\nabout prefill cost", fontsize=8.0, pad=6)
    ax.annotate(rf"$\rho = {rho:+.3f}$   $p = {p:.2f}$", xy=(0.97, 0.80),
                xycoords="axes fraction", ha="right", va="top", fontsize=7.0,
                color=INK,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9))
    ax.set_xlim(2.5, 15)
    panel_label(ax, "(c)", dx=-0.30)
    mh = [Line2D([], [], marker=MODEL_MARKER[m], linestyle="", markersize=4.2,
                 markerfacecolor=MUTED, markeredgecolor="white", markeredgewidth=0.5,
                 label=m) for m in MODELS]
    ax.legend(handles=mh, loc="lower center", bbox_to_anchor=(0.46, -0.53),
              ncol=1, handletextpad=0.25, borderpad=0.0, labelspacing=0.25)

    save(fig, "fig1_decoupling")


# ---------------------------------------------------------------------------
# Figure: nominal payload vs effective bits (why metadata must be counted)
# ---------------------------------------------------------------------------
def fig_effective_bits(D):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.15), sharey=True)
    for ax, model in zip(axes, MODELS):
        tidy(ax, grid_axis="y")
        cells = D["models"][model]["cells"]
        order = []
        for m in FIXED:
            for b in (4, 8):
                if f"{m}@{b}" in cells:
                    order.append((f"{SHORT[m]} {b}", cells[f"{m}@{b}"], m))
        for m in ADAPTIVE:
            if m in cells:
                order.append((SHORT[m], cells[m], m))
        order.sort(key=lambda t: t[1]["b_eff"])
        xs = np.arange(len(order))
        pay = [c["b_pay"] for _, c, _ in order]
        meta = [c["b_eff"] - c["b_pay"] for _, c, _ in order]
        cols = [METHOD_COLOR[m] for _, _, m in order]
        ax.bar(xs, pay, width=0.72, color=cols, zorder=2, linewidth=0)
        ax.bar(xs, meta, width=0.72, bottom=pay, color=cols, alpha=0.42,
               zorder=2, linewidth=0, hatch="////", edgecolor="white")
        for x, (_, c, _) in zip(xs, order):
            if c["meta_share"] >= 5:
                ax.annotate(f"+{c['meta_share']:.0f}%", (x, c["b_eff"]),
                            textcoords="offset points", xytext=(0, 2.5),
                            ha="center", fontsize=6.0, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([o[0] for o in order], rotation=52, ha="right", fontsize=6.2)
        ax.set_title(model, fontsize=8.0, pad=4)
        ax.set_ylim(0, 15.6)
    axes[0].set_ylabel("Bits / element")
    axes[0].annotate("solid: compressed payload\nhatched: mandatory metadata",
                     xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                     fontsize=6.4, color=MUTED)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.30, wspace=0.09)
    save(fig, "fig4_effective_bits")


# ---------------------------------------------------------------------------
# Figure: Part 1 algorithmic utility across operating points
# ---------------------------------------------------------------------------
def _ordered_cells(cells):
    order = []
    for m in FIXED:
        for b in (4, 8):
            if f"{m}@{b}" in cells:
                order.append((f"{SHORT[m]} {b}", cells[f"{m}@{b}"], m))
    for m in ADAPTIVE:
        if m in cells:
            order.append((SHORT[m], cells[m], m))
    if "Dense FP16" in cells:
        order.append((SHORT["Dense FP16"], cells["Dense FP16"], "Dense FP16"))
    return order


def fig_part1(D):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.55), sharex=True)
    for ax, model in zip(axes, MODELS):
        tidy(ax, grid_axis="x")
        order = sorted(_ordered_cells(D["models"][model]["cells"]),
                       key=lambda t: t[1]["s_res"])
        ys = np.arange(len(order))
        ax.barh(ys, [c["s_res"] for _, c, _ in order], height=0.68,
                color=[METHOD_COLOR[m] for _, _, m in order], zorder=2, linewidth=0)
        dense = D["models"][model]["cells"]["Dense FP16"]["s_res"]
        ax.axvline(dense, color=INK, lw=0.8, linestyle=(0, (3, 2)), zorder=3)
        for y, (_, c, _) in zip(ys, order):
            ax.annotate(f"{c['s_res']:.1f}", (c["s_res"], y), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=6.2,
                        color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([o[0] for o in order], fontsize=6.4)
        ax.set_title(model, fontsize=8.0, pad=4)
        ax.set_xlim(0, 78)
        ax.set_xlabel("$\\mathcal{S}_{\\mathrm{res}}$")
    fig.subplots_adjust(left=0.105, right=0.995, top=0.91, bottom=0.15, wspace=0.55)
    save(fig, "fig5_part1")


# ---------------------------------------------------------------------------
# Figure: where the hardware multiplier is lost
# ---------------------------------------------------------------------------
def fig_hardware(D):
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.55), sharex=True)
    for ax, model in zip(axes, MODELS):
        tidy(ax, grid_axis="x")
        order = sorted(_ordered_cells(D["models"][model]["cells"]),
                       key=lambda t: t[1]["phi"])
        ys = np.arange(len(order))
        ax.barh(ys, [c["phi"] for _, c, _ in order], height=0.68,
                color=[METHOD_COLOR[m] for _, _, m in order], zorder=2, linewidth=0)
        for y, (_, c, _) in zip(ys, order):
            tag, why = f"{c['phi']:.2f}", []
            if c["ttft_x_dense"] > 1.5:
                why.append(f"TTFT {c['ttft_x_dense']:.0f}$\\times$")
            if c["vram_x_dense"] > 1.4:
                why.append(f"VRAM {c['vram_x_dense']:.1f}$\\times$")
            if why:
                tag += "  (" + ", ".join(why) + ")"
            ax.annotate(tag, (c["phi"], y), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=6.2, color=INK)
        ax.set_yticks(ys)
        ax.set_yticklabels([o[0] for o in order], fontsize=6.4)
        ax.set_title(model, fontsize=8.0, pad=4)
        ax.set_xlim(0, 1.95)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xlabel("Hardware multiplier $\\phi$")
    fig.subplots_adjust(left=0.105, right=0.995, top=0.91, bottom=0.15, wspace=0.55)
    save(fig, "fig6_hardware")


# ---------------------------------------------------------------------------
# Figure: task heatmap
# ---------------------------------------------------------------------------
def fig_task_heatmap(D):
    tasks = ["Single NIAH", "Multi NIAH", "RULER KV", "Multi-Hop QA", "Variable Tracking"]
    methods = FIXED + ADAPTIVE + ["Dense FP16"]
    M = np.zeros((len(methods), len(tasks)))
    for i, m in enumerate(methods):
        for j, t in enumerate(tasks):
            vals = [D["models"][mo]["task_accuracy"].get(m, {}).get(t)
                    for mo in MODELS]
            vals = [v for v in vals if v is not None]
            M[i, j] = float(np.mean(vals)) if vals else np.nan

    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.74, 2.55))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=26, ha="right", fontsize=7.0)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([SHORT[m] for m in methods], fontsize=7.0)
    for i in range(len(methods)):
        for j in range(len(tasks)):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.6,
                    color="white" if v > 55 else INK)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.02)
    cb.set_label("Accuracy (%), mean of 3 models", fontsize=7.0)
    cb.ax.tick_params(labelsize=6.6, length=2)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.98, bottom=0.26)
    save(fig, "fig7_task_heatmap")


# ---------------------------------------------------------------------------
# Figure: context scaling on the 2K-128K ladder
# ---------------------------------------------------------------------------
def fig_context(D):
    fig, ax = plt.subplots(figsize=(TEXTWIDTH * 0.68, 2.35))
    tidy(ax, grid_axis="both")
    ca = D["models"]["Gemma 4 E2B"]["context_accuracy"]
    for m in FIXED + ADAPTIVE + ["Dense FP16"]:
        if m not in ca:
            continue
        xs = sorted(int(k) for k in ca[m])
        ys = [ca[m][str(x)] for x in xs]
        dense = m == "Dense FP16"
        ax.plot(xs, ys, marker="o", markersize=3.2, lw=1.9 if dense else 1.2,
                color=METHOD_COLOR[m], linestyle="--" if dense else "-",
                zorder=4 if dense else 3, label=SHORT[m])
    ax.set_xscale("log", base=2)
    ax.set_xticks([2048, 8192, 32768, 65536, 131072])
    ax.set_xticklabels(["2K", "8K", "32K", "64K", "128K"])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Task accuracy (%)")
    ax.set_ylim(-3, 88)
    ax.legend(ncol=2, loc="upper right", handletextpad=0.4, columnspacing=1.0,
              labelspacing=0.28, fontsize=6.6)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.97, bottom=0.185)
    save(fig, "fig8_context_scaling")


# ---------------------------------------------------------------------------
# Figure: quality-memory Pareto frontier
# ---------------------------------------------------------------------------
def fig_pareto(D):
    """Quality-memory frontier, computed within each model.

    A frontier across models would be meaningless: accuracy differences between
    Gemma 4 E2B and Qwen2.5 7B reflect base-model capability, not the
    representation.
    """
    fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH, 2.35), sharey=True)
    for ax, model in zip(axes, MODELS):
        tidy(ax, grid_axis="both")
        cells = list(D["models"][model]["cells"].values())
        front = [c for c in cells
                 if not any(o["b_eff"] <= c["b_eff"] and o["q_abs"] > c["q_abs"]
                            for o in cells)]
        front.sort(key=lambda c: c["b_eff"])
        ax.plot([c["b_eff"] for c in front], [c["q_abs"] for c in front],
                color=MUTED, lw=0.9, linestyle=(0, (4, 2)), zorder=2)
        for c in cells:
            on = c in front
            ax.scatter([c["b_eff"]], [c["q_abs"]], s=34 if on else 24,
                       marker="o", color=METHOD_COLOR[c["method"]],
                       edgecolor=INK if on else "white",
                       linewidth=0.8 if on else 0.5, zorder=4 if on else 3)
        ax.set_xlabel("Effective bits / element")
        ax.set_title(model, fontsize=8.0, pad=4)
        ax.set_xlim(2.5, 17.5)
        ax.set_xticks([4, 8, 12, 16])
    axes[0].set_ylabel("Task accuracy (%)")
    axes[0].set_ylim(-4, 74)
    ch = [Line2D([], [], marker="o", linestyle="", markersize=4.2,
                 markerfacecolor=METHOD_COLOR[m], markeredgecolor="white",
                 markeredgewidth=0.5, label=SHORT[m])
          for m in FIXED + ADAPTIVE + ["Dense FP16"]]
    fig.legend(handles=ch, loc="lower center", ncol=8, fontsize=6.6,
               handletextpad=0.3, columnspacing=1.1, bbox_to_anchor=(0.5, -0.015))
    axes[0].annotate("outlined: non-dominated", xy=(0.96, 0.06),
                     xycoords="axes fraction", ha="right", fontsize=6.2, color=MUTED)
    fig.subplots_adjust(left=0.075, right=0.995, top=0.90, bottom=0.28, wspace=0.09)
    save(fig, "fig9_pareto")


# ---------------------------------------------------------------------------
# Figure: alpha sensitivity of the Part 1 ranking
# ---------------------------------------------------------------------------
def fig_alpha(D):
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.25))
    alphas = np.linspace(0.1, 0.9, 81)

    ax = axes[0]
    tidy(ax, grid_axis="both")
    cells = D["models"]["Gemma 4 E4B"]["cells"]
    order = _ordered_cells(cells)
    for name, c, m in order:
        ys = [a * c["q_abs"] + (1 - a) * c["r_mem"] for a in alphas]
        ax.plot(alphas, ys, lw=1.1, color=METHOD_COLOR[m],
                linestyle="--" if m == "Dense FP16" else "-")
        ax.annotate(name, (0.9, ys[-1]), xytext=(3, 0), textcoords="offset points",
                    va="center", fontsize=5.8, color=METHOD_COLOR[m])
    ax.axvline(0.70, color=MUTED, lw=0.7, linestyle=(0, (3, 2)))
    ax.annotate("default $\\alpha=0.70$", (0.70, 4), xytext=(-4, 0),
                textcoords="offset points", rotation=90, ha="right", va="bottom",
                fontsize=6.3, color=MUTED)
    ax.set_xlabel("Quality preference $\\alpha$")
    ax.set_ylabel("$\\mathcal{S}_{\\mathrm{res}}$")
    ax.set_xlim(0.1, 1.16)
    ax.set_ylim(0, 82)
    ax.set_title("Score trajectories (Gemma 4 E4B)", fontsize=8.0, pad=4)
    panel_label(ax, "(a)", dx=-0.18)

    ax = axes[1]
    tidy(ax, grid_axis="both")
    for model in MODELS:
        cells = D["models"][model]["cells"]
        keys = list(cells)
        frac = []
        for a in alphas:
            sc = {k: a * cells[k]["q_abs"] + (1 - a) * cells[k]["r_mem"] for k in keys}
            frac.append({k: i for i, k in enumerate(sorted(keys, key=lambda k: -sc[k]))})
        ref = {k: i for i, k in enumerate(sorted(
            keys, key=lambda k: -(0.70 * cells[k]["q_abs"] + 0.30 * cells[k]["r_mem"])))}
        agree = [np.mean([abs(r[k] - ref[k]) for k in keys]) for r in frac]
        ax.plot(alphas, agree, lw=1.3, color=MODEL_COLOR[model], label=model)
    ax.axvline(0.70, color=MUTED, lw=0.7, linestyle=(0, (3, 2)))
    ax.set_xlabel("Quality preference $\\alpha$")
    ax.set_ylabel("Mean $|$rank shift$|$ vs. $\\alpha=0.70$")
    ax.set_xlim(0.1, 0.9)
    ax.legend(fontsize=6.6, loc="upper center", labelspacing=0.25)
    ax.set_title("Ranking stability", fontsize=8.0, pad=4)
    panel_label(ax, "(b)", dx=-0.18)

    fig.subplots_adjust(left=0.085, right=0.995, top=0.88, bottom=0.19, wspace=0.36)
    save(fig, "fig10_alpha_sensitivity")


# ---------------------------------------------------------------------------
# Figure: the measurement flow
# ---------------------------------------------------------------------------
def fig_framework():
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 2.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis("off")

    def box(x, y, w, h, title, body, edge, face="white", tsize=7.0, bsize=6.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.0,rounding_size=1.1",
                                    linewidth=0.9, edgecolor=edge, facecolor=face,
                                    zorder=2))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top",
                fontsize=tsize, fontweight="bold", color=INK, zorder=3)
        if body:
            ax.text(x + w / 2, y + h - 7.0, body, ha="center", va="top",
                    fontsize=bsize, color=INK, zorder=3, linespacing=1.45)

    def arrow(x0, y0, x1, y1, color=MUTED):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=0.9, color=color,
                                     shrinkA=0, shrinkB=0, zorder=1))

    blue, orange, green = "#0072B2", "#D55E00", "#009E73"

    box(0.0, 13.0, 15.5, 16.0, "One query",
        "$(x_i,\\ y_i)$\ntask $\\tau_i$\ncontext $L_i$", MUTED, "#F4F4F4")

    box(20.5, 23.5, 24.0, 15.0, "Dense reference",
        "$\\mathcal{M}(x_i \\mid \\mathcal{R}_{\\mathrm{dense}})$\n16 bits / element", MUTED)
    box(20.5, 3.5, 24.0, 15.0, "Candidate",
        "$\\mathcal{M}(x_i \\mid \\mathcal{A}(\\mathcal{B}))$\nquantise $\\cdot$ evict\npool $\\cdot$ factorise",
        blue)

    arrow(15.5, 24.5, 20.5, 30.0)
    arrow(15.5, 17.5, 20.5, 12.0)

    box(50.0, 27.0, 27.0, 12.0, "Axis 1 $\\cdot$ Representation",
        "$b_{\\mathrm{eff}}$ = payload + metadata", green)
    box(50.0, 14.5, 27.0, 12.0, "Axis 2 $\\cdot$ Task",
        "$Q_{\\mathrm{abs}}$, paired $\\Delta_i$", green)
    box(50.0, 2.0, 27.0, 12.0, "Axis 3 $\\cdot$ Serving",
        "TTFT $\\cdot$ decode $\\cdot$ peak VRAM", orange)

    arrow(44.5, 31.0, 50.0, 33.0)
    arrow(44.5, 11.0, 50.0, 20.5)
    arrow(44.5, 11.0, 50.0, 8.0)

    box(82.0, 21.0, 18.0, 15.0, "Part 1",
        "$\\mathcal{S}_{\\mathrm{res}}$\ncoverage $C$", green)
    box(82.0, 2.5, 18.0, 15.0, "Part 2",
        "$\\mathcal{S}_{\\mathrm{sys}}$\n$\\phi$ components", orange)

    arrow(77.0, 33.0, 82.0, 30.0)
    arrow(77.0, 20.5, 82.0, 26.0)
    arrow(77.0, 20.5, 82.0, 12.0)
    arrow(77.0, 8.0, 82.0, 9.0)

    ax.text(91.0, 0.2, "reported separately, never merged", ha="center", va="bottom",
            fontsize=6.0, color=MUTED, style="italic")
    save(fig, "fig2_framework")


def main():
    use_style()
    D, S = numbers(), stats()
    print("rendering figures ->", FIGDIR)
    fig_framework()
    fig_decoupling(D, S)
    fig_effective_bits(D)
    fig_part1(D)
    fig_hardware(D)
    fig_task_heatmap(D)
    fig_context(D)
    fig_pareto(D)
    fig_alpha(D)


if __name__ == "__main__":
    main()
