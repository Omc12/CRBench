"""Master Sequential Queue Runner for All 3 Models on Refreshed Upstream DKV (mid and high).

Runs strictly one model at a time on GPU:
1. Gemma 4 E2B (2K-128K)
2. Gemma 4 E4B (2K-32K)
3. Qwen2.5-7B (2K-32K)
"""

import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXE = REPO_ROOT / ".venv" / "Scripts" / "python.exe"

LOG_PATH = REPO_ROOT / "benchmarks" / "run_all_models_dkv_refreshed.log"

JOBS = [
    {
        "model_tag": "Gemma 4 E2B",
        "config": REPO_ROOT / "benchmarks" / "bench_gemma4_e2b_dkv_refreshed.yaml",
        "out_dir": REPO_ROOT / "benchmarks" / "bench_gemma4_e2b_dkv_refreshed",
    },
    {
        "model_tag": "Gemma 4 E4B",
        "config": REPO_ROOT / "benchmarks" / "bench_gemma4_e4b_dkv_refreshed.yaml",
        "out_dir": REPO_ROOT / "benchmarks" / "bench_gemma4_e4b_dkv_refreshed",
    },
    {
        "model_tag": "Qwen2.5-7B",
        "config": REPO_ROOT / "benchmarks" / "bench_7b_dkv_refreshed.yaml",
        "out_dir": REPO_ROOT / "benchmarks" / "bench_7b_dkv_refreshed",
    },
]

def main():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    with open(LOG_PATH, "w", encoding="utf-8", buffering=1) as log_file:
        def log(msg: str):
            ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
            line = f"{ts} {msg}\n"
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()

        log("=" * 80)
        log("MASTER SEQUENTIAL DKV QUEUE RUNNER: E2B -> E4B -> 7B")
        log("Testing normal standard presets: mid and high (plus dense baseline)")
        log("Using refreshed upstream Differential-KV commit 46737636")
        log("=" * 80)

        total_jobs = len(JOBS)
        for idx, job in enumerate(JOBS, 1):
            tag = job["model_tag"]
            cfg_path = job["config"]
            log(f"\n[{idx}/{total_jobs}] STARTING BENCHMARK: {tag}")
            log(f"      Config: {cfg_path}")
            log(f"      Output: {job['out_dir']}")

            cmd = [
                str(PYTHON_EXE),
                "-u",
                "-m",
                "crbench.cli",
                "run",
                "--config",
                str(cfg_path),
            ]

            t_start = time.time()
            proc = subprocess.Popen(
                cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                cwd=str(REPO_ROOT),
                env=env,
            )
            log(f"      Subprocess PID: {proc.pid} active on GPU. Awaiting completion...")
            ret = proc.wait()
            elapsed = time.time() - t_start

            if ret != 0:
                log(f"[ERROR] Benchmark for {tag} failed with exit code {ret} after {elapsed/60:.1f} mins!")
                sys.exit(ret)
            else:
                log(f"[SUCCESS] Benchmark for {tag} completed successfully in {elapsed/60:.1f} mins!")

            # Allow GPU memory to settle between runs
            time.sleep(5)

        log("\n" + "=" * 80)
        log("ALL 3 MODEL BENCHMARKS COMPLETED SUCCESSFULLY!")
        log("=" * 80)

if __name__ == "__main__":
    main()
