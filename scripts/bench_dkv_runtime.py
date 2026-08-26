"""Benchmark Differential-KV through its own serving runtime.

Why this exists
---------------
The ``dkv`` adapter in ``crbench/adapters/custom_example.py`` calls
``native_core.compression.lowrank.compress_lowrank`` directly, one level below
the interface the repository is driven through. That measures DKV's standalone
block compressor and misses most of what DKV is:

* the paged block pool and its own memory accounting,
* the Triton sparse-decode kernels,
* the residual selection reached only via ``compress_layer_blocks_gpu``,
* and the presets as the CLI applies them.

It also got the micro-block wrong. The adapter used 256; the runtime resolves
1024, and the CLI records why: "MLX: 256 -> 1024 took linkbench 9/24 to 24/24 =
dense, and the pool from 0.95x of the KV it replaces to 0.28x".

This script instead constructs ``PyTorchDKVHFWrapper`` exactly as
``ACTIVE_RUNTIME/serving/cli.py`` does -- model id, preset, quantization,
serving mode -- and measures the result.

Structure
---------
The wrapper owns its own model, so it cannot share the process with CRBench's
dense pass on a 12 GiB card. The dense anchor is therefore read from an existing
CRBench result file. That is sound because the samples are deterministic: the
same task name, seed and context length regenerate byte-identical prompts, which
this script asserts per query rather than assumes.

Fairness
--------
DKV is given the prompt and nothing else -- no pinned query, no ground truth --
the same information every other method in CRBench receives.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path(__file__).resolve().parents[1]
DKV_RUNTIME = REPO / "third_party" / "Differential-KV" / "ACTIVE_RUNTIME"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(DKV_RUNTIME))

# Must precede the torch import inside the wrapper: it bounds allocator
# fragmentation, which the repository measured as the difference between 16.2
# and 8.9 GiB reserved at 32k on a 12 GiB card.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Set before ANY wrapper is constructed, not per preset.
#
# hf_dkv_wrapper.py:617 does os.environ.setdefault("DKV_CAD_ALPHA", "0.5") when
# the preset is high/quality/max, and setdefault mutates the process environment
# permanently: building `high` before `mid` in one process leaves alpha at 0.5
# for the `mid` run too. That is order-dependent, so it would present as noise
# rather than as a bug.
#
# Context-Aware Decoding also gates on bool(query_text) at :1819, which is False
# under this benchmark's equal-information rule, so CAD cannot engage either way.
# Pinning the value to 0 makes that explicit instead of incidental, and setdefault
# yields to an existing value. `high` is therefore measured on its compression
# settings alone -- rank 128, svd_energy 0.99999, 128 residuals, 4096-token f16
# window -- and must be labelled that way rather than as full `high`.
os.environ.setdefault("DKV_CAD_ALPHA", "0")

import torch  # noqa: E402

from crbench.core.config import BenchmarkConfig  # noqa: E402
from crbench.core.registry import Registry  # noqa: E402
import crbench.tasks  # noqa: E402,F401


def build_prompt(tokenizer, sample, max_len: int) -> str:
    """Reproduce BenchmarkRunner._encode_sample's prompt text exactly."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": sample.full_prompt}],
            tokenize=False, add_generation_prompt=True,
        )
    return sample.full_prompt


def decoder_layer_probe(wrapper):
    """A weight module whose class reveals whether quantization took effect."""
    m = wrapper.model
    for path in (("model", "layers"), ("model", "language_model", "layers"),
                 ("language_model", "model", "layers"), ("layers",)):
        node = m
        for attr in path:
            node = getattr(node, attr, None)
            if node is None:
                break
        if node:
            layer = node[0]
            for name in ("mlp", "feed_forward"):
                mlp = getattr(layer, name, None)
                if mlp is not None:
                    for proj in ("gate_proj", "up_proj", "w1"):
                        q = getattr(mlp, proj, None)
                        if q is not None:
                            return q
            return getattr(getattr(layer, "self_attn", layer), "q_proj", layer)
    raise RuntimeError("could not locate a decoder weight module")


def extract_completion_by_tokens(tokenizer, raw, prompt: str) -> str:
    """Strip the prompt by TOKEN COUNT, which is the only reliable way here.

    generate() builds its output as `generated = prompt_ids.copy()` and decodes
    the whole thing with skip_special_tokens=True, then passes it through
    _normalize_references(), which rewrites citation lines. Both destroy an exact
    string prefix, so text.startswith(prompt) silently fails -- and on a needle
    task the prompt contains the needle, so the fallback scores a perfect answer
    for a method that produced nothing.
    """
    ids = getattr(raw, "generated_ids", None)
    if ids is not None:
        n_prompt = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        return tokenizer.decode(list(ids)[n_prompt:], skip_special_tokens=True)
    return None


def extract_completion(text: str, prompt: str, sample) -> str:
    """Return only what the model generated.

    The wrapper returns the whole decoded sequence, and it does not always start
    with the prompt verbatim -- it re-renders the chat template, so a prefix
    comparison silently fails and the *prompt* gets scored. On a needle task the
    prompt contains the needle, so that reads as a perfect score for a method
    that answered nothing. Anchor on the question instead, which the task places
    at the very end of the prompt.
    """
    if text.startswith(prompt):
        return text[len(prompt):]
    anchor = sample.query.strip()
    idx = text.rfind(anchor)
    if idx >= 0:
        return text[idx + len(anchor):]
    tail = prompt[-200:].strip()
    idx = text.rfind(tail)
    if idx >= 0:
        return text[idx + len(tail):]
    # Nothing matched: refuse to score rather than score the prompt.
    raise RuntimeError("could not separate completion from prompt")


def pool_bytes(manager, session: str, feat_dim_bytes: int, rank: int) -> Dict[str, float]:
    """Stored KV bytes, from the runtime's own per-block state.

    `total_compressions` and `vram_saved_bytes` stay at zero on the GPU-batched
    path -- they are incremented by the single-block helper -- so the block state
    published in `manager.sessions` is the authoritative source. Each layer
    reports how many blocks it holds, the dense span each covers, and the exact
    residual count kept per block.
    """
    try:
        st = dict(manager.sessions).get(session) or {}
    except Exception:
        return {}
    num_blocks = st.get("num_blocks") or []
    dense_lens = st.get("dense_lens") or []
    comp_res_n = st.get("comp_res_n") or []
    if not num_blocks:
        return {}

    total = 0.0
    compressed_blocks = dense_blocks = residual_tokens = 0
    for layer_idx, n_blk in enumerate(num_blocks):
        span = dense_lens[layer_idx] if layer_idx < len(dense_lens) else 0
        res_row = comp_res_n[layer_idx] if layer_idx < len(comp_res_n) else []
        for b in range(int(n_blk)):
            res = int(res_row[b]) if b < len(res_row) else 0
            if res > 0:
                compressed_blocks += 1
                residual_tokens += res
                # anchor + U (span x rank) + V (rank x feat) + exact residuals
                total += feat_dim_bytes
                total += span * rank * 2 + rank * feat_dim_bytes
                total += res * feat_dim_bytes
            else:
                dense_blocks += 1
                total += span * feat_dim_bytes
    return {"stored_bytes": total, "compressed_blocks": compressed_blocks,
            "dense_blocks": dense_blocks, "residual_tokens": residual_tokens}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="CRBench config supplying model, tasks and grid")
    ap.add_argument("--dense-from", required=True, help="raw_results_v1.json holding the dense anchor")
    ap.add_argument("--presets", default="mid,high")
    ap.add_argument("--output", required=True)
    ap.add_argument("--serving-mode", default="balanced")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    cfg = BenchmarkConfig.from_yaml(args.config)
    model_id = cfg.model.model_name_or_path

    dense_doc = json.load(open(args.dense_from, encoding="utf-8"))
    dense: Dict[str, Dict[str, Any]] = {}
    for q in dense_doc["query_results"]:
        if q["method_name"] in ("dense_fp16", "dense"):
            dense[f'{q["task_name"]}|{q["context_length"]}|{q["query_id"]}'] = q
    if not dense:
        print(f"[!] No dense results in {args.dense_from}", flush=True)
        return 2
    print(f"[*] Dense anchor: {len(dense)} queries from {args.dense_from}", flush=True)

    from serving.hf_dkv_wrapper import PyTorchDKVHFWrapper

    quant = "int4" if cfg.model.load_in_4bit else ("int8" if cfg.model.load_in_8bit else None)
    out_records: List[Dict[str, Any]] = []

    for preset in [p.strip() for p in args.presets.split(",") if p.strip()]:
        print(f"\n{'=' * 70}\n[*] DKV runtime, preset={preset}, model={model_id}\n{'=' * 70}", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        t0 = time.perf_counter()
        # Build the BitsAndBytesConfig and pass it as the separate
        # `quantization_config=` argument, exactly as cli.py:869-919 does.
        #
        # The config dict's "quantization" key alone is not sufficient and does
        # not fail loudly: this script previously passed only the dict, the
        # wrapper's auto-load branch matched nothing, and the model loaded in
        # fp16 with no message. Qwen2.5-7B is 7.62B parameters, so fp16 weights
        # are 15.24 GB -- which is what the 15.49 GiB peak in the compile A/B
        # actually was, and why it spilled on a 12 GiB card.
        bnb = None
        if quant:
            from transformers import BitsAndBytesConfig
            bnb = (BitsAndBytesConfig(load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16,
                                      bnb_4bit_quant_type="nf4",
                                      bnb_4bit_use_double_quant=True)
                   if quant == "int4" else BitsAndBytesConfig(load_in_8bit=True))
        wrapper = PyTorchDKVHFWrapper(
            model_id,
            config={"serving_mode": args.serving_mode, "mode": "fp16",
                    "quantization": quant, "preset": preset},
            device="cuda",
            quantization_config=bnb,
        )
        # Verify rather than trust: a silent fp16 fallback is exactly the failure
        # this script already shipped once.
        try:
            first = decoder_layer_probe(wrapper)
            kind = type(first).__name__
            print(f"[*] weight module: {kind} "
                  f"({'QUANTIZED' if 'bit' in kind.lower() else 'NOT QUANTIZED'})", flush=True)
            if quant and "bit" not in kind.lower():
                raise SystemExit(f"[!] Requested {quant} but weights are {kind}; refusing to "
                                 f"report these as quantized results.")
        except SystemExit:
            raise
        except Exception as exc:                                  # noqa: BLE001
            print(f"[!] Could not verify quantization: {exc}", flush=True)
        tok = wrapper.tokenizer
        weights = torch.cuda.memory_allocated()
        print(f"[OK] Wrapper up in {time.perf_counter() - t0:.1f}s | weights {weights / 2**30:.2f} GiB | "
              f"micro_block={wrapper.micro_block_size} rank={wrapper.rank} "
              f"layers={wrapper.num_layers} kv_heads={wrapper.kv_heads} head_dim={wrapper.head_dim}",
              flush=True)

        kv_per_token = 2 * wrapper.num_layers * wrapper.kv_heads * wrapper.head_dim * 2

        for task_cfg in cfg.tasks:
            task = Registry.get_task(task_cfg.task_name)(
                name=task_cfg.task_name, seed=task_cfg.seed, config=task_cfg.task_kwargs)
            for ctx_len in task_cfg.context_lengths:
                samples = task.generate_samples(context_length=ctx_len,
                                                num_samples=task_cfg.num_samples, tokenizer=tok)
                for sample in samples:
                    key = f"{task.name}|{ctx_len}|{sample.sample_id}"
                    ref = dense.get(key)
                    if ref is None:
                        print(f"    [skip] no dense anchor for {key}", flush=True)
                        continue

                    prompt = build_prompt(tok, sample, ctx_len)
                    n_prompt = len(tok(prompt).input_ids)

                    try:
                        wrapper.clear_session()
                    except Exception:
                        pass
                    mgr = wrapper.manager
                    # runtime_summary() is the runtime's own accounting surface;
                    # the bare vram_saved_bytes counter read zero because the
                    # compression it tracks is published elsewhere on this path.
                    def _summary():
                        try:
                            return dict(mgr.runtime_summary())
                        except Exception:
                            return {}
                    sum_before = _summary()
                    saved_before = int(getattr(mgr, "vram_saved_bytes", 0) or 0)
                    gc.collect(); torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()

                    t_start = time.perf_counter()
                    try:
                        # Parity with every other method in CRBench, which decodes
                        # greedily. The wrapper defaults to temperature=0.7,
                        # top_p=0.9, repetition_penalty=1.15 -- sampling, which
                        # would make DKV the only stochastic entry in the table.
                        # _sample_logits takes argmax at temperature <= 0.01.
                        # query_text is left unset: pinning the question is the
                        # privileged-information channel the fairness rule excludes.
                        raw = wrapper.generate(prompt, max_new_tokens=args.max_new_tokens,
                                               temperature=0.0, top_p=1.0,
                                               repetition_penalty=1.0)
                        elapsed = time.perf_counter() - t_start
                        text = str(raw)
                        completion = (extract_completion_by_tokens(tok, raw, prompt)
                                      or extract_completion(text, prompt, sample))
                        score = task.evaluate_prediction(completion, sample).score
                        status, err = "SUCCESS", None
                    except Exception as exc:                      # noqa: BLE001
                        elapsed = time.perf_counter() - t_start
                        completion, score = "", 0.0
                        status, err = type(exc).__name__, str(exc)[:300]
                        print(f"    [!] {key}: {status}: {err}", flush=True)

                    sum_after = _summary()
                    feat_bytes = wrapper.kv_heads * wrapper.head_dim * 2 * 2  # K and V, bf16
                    pool = pool_bytes(mgr, "default", feat_bytes, int(wrapper.rank) or 1)
                    saved = int(getattr(mgr, "vram_saved_bytes", 0) or 0) - saved_before
                    n_comp = int(sum_after.get("total_compressions", 0)) - int(sum_before.get("total_compressions", 0))
                    saved_mb = float(sum_after.get("vram_saved_mb", 0.0)) - float(sum_before.get("vram_saved_mb", 0.0))
                    if saved <= 0 and saved_mb > 0:
                        saved = int(saved_mb * 1e6)
                    dense_bytes = kv_per_token * n_prompt
                    if pool.get("stored_bytes"):
                        stored = float(pool["stored_bytes"])
                    else:
                        stored = max(0, dense_bytes - saved)
                    b_eff = (stored * 8.0 / max(1, 2 * wrapper.num_layers * wrapper.kv_heads
                                                * wrapper.head_dim * n_prompt)) if n_prompt else 16.0

                    out_records.append({
                        "task_name": task.name, "context_length": ctx_len,
                        "query_id": sample.sample_id, "method_name": f"dkv_runtime_{preset}",
                        "preset": preset, "status": status, "error_message": err,
                        "prompt_tokens": n_prompt,
                        "dense_raw_score": ref["dense_raw_score"],
                        "method_raw_score": score,
                        "dense_kv_bytes": dense_bytes,
                        "method_kv_bytes": stored,
                        "vram_saved_bytes": saved,
                        "compressions": n_comp,
                        "pool": pool,
                        "runtime_summary_after": {k: v for k, v in sum_after.items()
                                                  if k in ("total_compressions", "vram_saved_mb",
                                                           "avg_cosine_sim", "rank_histogram")},
                        "method_effective_bpt": b_eff,
                        "seconds": elapsed,
                        "peak_bytes": int(torch.cuda.max_memory_allocated()),
                        "prediction": completion[:400],
                        "ground_truths": sample.ground_truths,
                    })
                    print(f"    {task.name:<24} {ctx_len:>7} {sample.sample_id:<22} "
                          f"score={score:.2f} dense={ref['dense_raw_score']:.2f} "
                          f"b_eff={b_eff:5.2f} blocks={pool.get('compressed_blocks',0)}c/{pool.get('dense_blocks',0)}d {elapsed:5.1f}s", flush=True)

                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                json.dump({"model": model_id, "records": out_records},
                          open(args.output, "w", encoding="utf-8"), indent=1)

        try:
            wrapper.close()
        except Exception:
            pass
        del wrapper
        gc.collect(); torch.cuda.empty_cache()

    json.dump({"model": model_id, "records": out_records},
              open(args.output, "w", encoding="utf-8"), indent=1)
    print(f"\n[OK] {len(out_records)} records -> {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
