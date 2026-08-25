"""
Bridge to the vendored upstream reference implementations.

CRBench compares published KV-compression methods, so the algorithms must come
from the authors' own code rather than from a paraphrase of their papers.  The
repositories live under ``third_party/`` and are imported from source here:

===================  ==========================================================
``snapkv``           https://github.com/FasterDecoding/SnapKV
``streaming_llm``    https://github.com/mit-han-lab/streaming-llm
``dkv``              https://github.com/Omc12/Differential-KV
===================  ==========================================================

Nothing here reimplements an algorithm.  Where an upstream entry point cannot be
called as-is -- SnapKV's ``monkeypatch`` targets ``transformers==4.37`` and only
Llama/Mistral/Mixtral, while this project runs transformers 5.x on Qwen2 -- the
bridge calls the upstream *algorithm* object directly and says so in the
adapter's provenance record.  ``describe_provenance`` returns exactly which
upstream commit each method was taken from, and the runner writes it into the
results manifest.
"""

from __future__ import annotations

import functools
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party"

_REPOS: Dict[str, Path] = {
    "snapkv": THIRD_PARTY / "SnapKV",
    "streaming_llm": THIRD_PARTY / "streaming-llm",
    "dkv": THIRD_PARTY / "Differential-KV",
}

_UPSTREAM_URLS: Dict[str, str] = {
    "snapkv": "https://github.com/FasterDecoding/SnapKV",
    "streaming_llm": "https://github.com/mit-han-lab/streaming-llm",
    "dkv": "https://github.com/Omc12/Differential-KV",
}


class UpstreamUnavailable(RuntimeError):
    """Raised when a vendored repository is missing or cannot be imported.

    Deliberately fatal.  Silently falling back to an approximation would publish
    a number under a method's name that the method did not produce.
    """


def _require_repo(key: str) -> Path:
    path = _REPOS[key]
    if not path.is_dir():
        raise UpstreamUnavailable(
            f"{key}: expected the upstream repository at {path}. "
            f"Clone it with:  git clone {_UPSTREAM_URLS[key]} {path}"
        )
    return path


def _add_to_path(path: Path) -> None:
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)


def repo_commit(key: str) -> Optional[str]:
    """Short SHA of the vendored checkout, for the provenance record."""
    try:
        path = _REPOS[key]
        if not path.is_dir():
            return None
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def load_snapkv():
    """Return upstream SnapKV's ``SnapKVCluster``.

    ``SnapKVCluster.update_kv`` is the whole algorithm: attention of the last
    ``window_size`` prompt queries over the prompt keys, softmax, sum across the
    observation window, ``avg_pool1d`` smoothing over ``kernel_size``, top-k
    selection, then a gather that keeps the selected past plus the intact recent
    window.  It is pure tensor code with no transformers dependency, so it runs
    unmodified under transformers 5.x.

    Only the surrounding ``monkeypatch`` module is version-locked (it rewrites
    ``LlamaAttention.forward`` as of 4.37), which is why the adapter drives this
    object itself instead of calling ``replace_llama()``.
    """
    _add_to_path(_require_repo("snapkv"))
    try:
        from snapkv.monkeypatch.snapkv_utils import SnapKVCluster  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the checkout
        raise UpstreamUnavailable(f"snapkv: import failed ({exc})") from exc
    return SnapKVCluster


@functools.lru_cache(maxsize=1)
def load_streaming_llm():
    """Return upstream StreamingLLM's ``StartRecentKVCache``.

    The published eviction policy: keep the first ``start_size`` tokens (the
    attention sinks) and the most recent ``recent_size``, drop everything
    between.  ``__call__`` applies it to a list of per-layer (key, value) pairs.
    """
    _add_to_path(_require_repo("streaming_llm"))
    try:
        from streaming_llm.kv_cache import StartRecentKVCache  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the checkout
        raise UpstreamUnavailable(f"streaming_llm: import failed ({exc})") from exc
    return StartRecentKVCache


@functools.lru_cache(maxsize=1)
def load_dkv() -> Dict[str, Any]:
    """Return Differential-KV's block compressor and its configuration surface.

    DKV partitions the cache into fixed micro-blocks and stores, per block, an
    exact anchor token, a joint K|V truncated-SVD delta, and a budget of exact
    residual tokens, with a dense recency window left uncompressed.  The runtime
    ships a Triton decode kernel that is unavailable on Windows; the repository
    guards it behind ``HAS_TRITON`` and keeps a PyTorch path, which is what this
    bridge uses.
    """
    repo = _require_repo("dkv")
    _add_to_path(repo / "ACTIVE_RUNTIME")
    try:
        from native_core.compression import lowrank  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on the checkout
        raise UpstreamUnavailable(f"dkv: import failed ({exc})") from exc
    return {"lowrank": lowrank}


def describe_provenance(key: str) -> Dict[str, Any]:
    """Provenance record for the results manifest."""
    path = _REPOS.get(key)
    return {
        "upstream_repository": _UPSTREAM_URLS.get(key),
        "vendored_path": str(path.relative_to(THIRD_PARTY.parent)) if path else None,
        "commit": repo_commit(key),
        "present": bool(path and path.is_dir()),
    }
