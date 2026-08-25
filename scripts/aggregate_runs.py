#!/usr/bin/env python
"""
Combine several CRBench runs into one report.

A single sweep cannot cover 2K-128K on a 12 GiB card: the model that fits at
131072 tokens is not the one you want at 2048, and Qwen2.5's native 32768-token
window means the long lengths need YaRN while the short ones must not have it.
The benchmark is therefore split across runs, and this script joins them without
blurring the seams -- every row carries the model and the RoPE regime it was
measured under, because a quality number at 128K under scaled positions is not
comparable to one at 8K under native positions.

Usage:
    python scripts/aggregate_runs.py results/bench_7b_native results/bench_3b_native \
        results/bench_3b_yarn_long -o results/COMBINED_REPORT.md
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tabulate import tabulate


def load_run(path: Path) -> Dict[str, Any]:
    raw = path / "raw_results_v1.json"
    if not raw.is_file():
        raise SystemExit(f"{path}: no raw_results_v1.json (did the run finish?)")
    with open(raw, "r", encoding="utf-8") as f:
        return json.load(f)


def rope_regime(run: Dict[str, Any]) -> str:
    rope = (run.get("model_config") or {}).get("rope_scaling")
    if not rope:
        return "native"
    rtype = rope.get("rope_type") or rope.get("type") or "scaled"
    factor = rope.get("factor")
    return f"{rtype}x{factor:g}" if factor else str(rtype)


def short_model(name: str) -> str:
    return name.split("/")[-1].replace("-Instruct", "")


def collect(runs: List[Tuple[Path, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """One row per (run, method, budget, context length), averaged over queries."""
    buckets: Dict[Tuple, Dict[str, Any]] = defaultdict(
        lambda: {"q": [], "bpt": [], "ttft": [], "dec": [], "vram": [], "kv": [], "n": 0}
    )
    for path, run in runs:
        model = short_model(run.get("model_name", "?"))
        regime = rope_regime(run)
        for q in run.get("query_results", []):
            key = (model, regime, q["method_name"], q["budget_spec"], q["context_length"])
            b = buckets[key]
            b["q"].append(q["quality_retained_pct"])
            b["bpt"].append(q["method_effective_bpt"])
            if q.get("method_ttft_ms") is not None:
                b["ttft"].append(q["method_ttft_ms"])
            if q.get("method_decode_throughput") is not None:
                b["dec"].append(q["method_decode_throughput"])
            if q.get("method_peak_vram_mb") is not None:
                b["vram"].append(q["method_peak_vram_mb"])
            kv = (q.get("metadata") or {}).get("method_resident_kv_bytes")
            if kv is not None:
                b["kv"].append(kv)
            b["n"] += 1

    def mean(xs, default=float("nan")):
        return sum(xs) / len(xs) if xs else default

    rows = []
    for (model, regime, method, budget, ctx), b in sorted(buckets.items()):
        rows.append({
            "model": model, "regime": regime, "method": method, "budget": budget,
            "context": ctx, "queries": b["n"],
            "quality": mean(b["q"]), "bpt": mean(b["bpt"]),
            "ttft_ms": mean(b["ttft"]), "decode_tok_s": mean(b["dec"]),
            "peak_vram_mb": mean(b["vram"]), "kv_bytes": mean(b["kv"]),
        })
    return rows


def failures(runs: List[Tuple[Path, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Non-SUCCESS measurements: these are results, not omissions."""
    out = []
    for path, run in runs:
        model = short_model(run.get("model_name", "?"))
        for m in run.get("raw_measurements", []):
            if m.get("status") == "SUCCESS":
                continue
            out.append({
                "model": model, "context": m.get("context_length"),
                "method": m.get("adapter_name"), "budget": m.get("budget_spec"),
                "status": m.get("status"),
                "detail": (m.get("error_message") or "")[:110],
            })
    return out


def quality_matrix(rows: List[Dict[str, Any]]) -> str:
    """Quality retained (%) by method and context length, per model."""
    out = []
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        contexts = sorted({r["context"] for r in sub})
        methods = sorted({(r["method"], r["budget"]) for r in sub},
                         key=lambda t: (t[0], t[1]))
        headers = ["Method", "Budget", "b_eff"] + [f"{c // 1024}K" for c in contexts]
        table = []
        for method, budget in methods:
            cells = []
            bpts = []
            for c in contexts:
                match = [r for r in sub if r["method"] == method
                         and r["budget"] == budget and r["context"] == c]
                if match:
                    cells.append(f"{match[0]['quality']:.1f}")
                    bpts.append(match[0]["bpt"])
                else:
                    cells.append("--")
            beff = f"{sum(bpts) / len(bpts):.2f}" if bpts else "--"
            table.append([method, budget, beff] + cells)
        regimes = sorted({r["regime"] for r in sub})
        out.append(f"### {model}  (RoPE: {', '.join(regimes)})\n")
        out.append(tabulate(table, headers=headers, tablefmt="github"))
        out.append("")
    return "\n".join(out)


def system_table(rows: List[Dict[str, Any]]) -> str:
    """Measured runtime and memory at each model's longest completed context."""
    out = []
    for model in sorted({r["model"] for r in rows}):
        sub = [r for r in rows if r["model"] == model]
        longest = max(r["context"] for r in sub)
        sel = [r for r in sub if r["context"] == longest]
        table = []
        for r in sorted(sel, key=lambda r: (r["method"], r["budget"])):
            table.append([
                r["method"], r["budget"], f"{r['bpt']:.2f}", f"{r['quality']:.1f}",
                f"{r['ttft_ms'] / 1000.0:.1f}" if r["ttft_ms"] == r["ttft_ms"] else "--",
                f"{r['decode_tok_s']:.1f}" if r["decode_tok_s"] == r["decode_tok_s"] else "--",
                f"{r['kv_bytes'] / 2 ** 30:.2f}" if r["kv_bytes"] == r["kv_bytes"] else "--",
                f"{r['peak_vram_mb'] / 1024.0:.2f}" if r["peak_vram_mb"] == r["peak_vram_mb"] else "--",
            ])
        out.append(f"### {model} @ {longest:,} tokens\n")
        out.append(tabulate(
            table,
            headers=["Method", "Budget", "b_eff", "Quality %", "TTFT (s)",
                     "Decode (tok/s)", "Resident KV (GiB)", "Peak above weights (GiB)"],
            tablefmt="github"))
        out.append("")
    return "\n".join(out)


def provenance_table(runs: List[Tuple[Path, Dict[str, Any]]]) -> str:
    seen: Dict[str, Dict[str, Any]] = {}
    for _, run in runs:
        for name, rec in (run.get("method_provenance") or {}).items():
            seen.setdefault(name, rec)
    table = []
    for name, rec in sorted(seen.items()):
        table.append([
            name,
            rec.get("implementation", "?"),
            rec.get("commit") or "--",
            (rec.get("entry_point") or rec.get("scheme") or "")[:78],
        ])
    return tabulate(table, headers=["Method", "Source", "Upstream commit", "Entry point / scheme"],
                    tablefmt="github")


def environment_block(runs: List[Tuple[Path, Dict[str, Any]]]) -> str:
    lines = []
    for path, run in runs:
        env = run.get("environment", {})
        mc = run.get("model_config", {})
        ex = run.get("execution_config", {})
        total = env.get("device_total_memory_bytes")
        lines.append(
            f"- **{run.get('benchmark_name')}** (`{path.name}`): "
            f"{run.get('model_name')}, "
            f"{'NF4 4-bit' if mc.get('load_in_4bit') else mc.get('dtype')}, "
            f"weights {mc.get('weight_bytes_resident', 0) / 2 ** 30:.2f} GiB, "
            f"RoPE {rope_regime(run)}, "
            f"prefill chunk {ex.get('prefill_chunk_size')}, "
            f"{ex.get('max_new_tokens')} new tokens, {ex.get('decoding')} decoding"
        )
        lines.append(
            f"  - {env.get('device_name')} "
            f"({total / 2 ** 30:.2f} GiB), torch {env.get('pytorch_version')}, "
            f"transformers {env.get('transformers_version')}, {env.get('os')}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path, help="Run output directories")
    ap.add_argument("-o", "--output", type=Path, default=Path("results/COMBINED_REPORT.md"))
    args = ap.parse_args()

    loaded = [(p, load_run(p)) for p in args.runs]
    rows = collect(loaded)
    fails = failures(loaded)

    doc = [
        "# CRBench -- Combined Results (2K to 128K, quantized models)",
        "",
        "Quality is **retention relative to the same model's own uncompressed run on the "
        "same query**, not absolute task accuracy, so 100% means a method matched its own "
        "dense baseline and says nothing about how strong that baseline was.",
        "",
        "## Runs",
        "",
        environment_block(loaded),
        "",
        "## Quality retained vs. dense baseline (%)",
        "",
        quality_matrix(rows),
        "## Measured system behaviour at the longest context",
        "",
        system_table(rows),
        "## Method provenance",
        "",
        provenance_table(loaded),
        "",
    ]

    if fails:
        doc += [
            "## Recorded failures",
            "",
            "These are measurements, not gaps: a method that could not run at a context "
            "length is a property of the method on this hardware.",
            "",
            tabulate([[f["model"], f["context"], f["method"], f["budget"], f["status"], f["detail"]]
                      for f in fails],
                     headers=["Model", "Context", "Method", "Budget", "Status", "Detail"],
                     tablefmt="github"),
            "",
        ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(doc), encoding="utf-8")
    print(f"Wrote {args.output}  ({len(rows)} rows from {len(loaded)} runs, {len(fails)} failures)")


if __name__ == "__main__":
    main()
