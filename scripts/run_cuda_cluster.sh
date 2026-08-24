#!/usr/bin/env bash
set -e

echo "================================================================================"
echo "CRBench — Full Scale CUDA Cluster Benchmark Runner"
echo "================================================================================"

# 1. Environment Setup
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv .venv
fi

source .venv/bin/activate
echo "[*] Installing/verifying dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# 2. CUDA & Hardware Check
python3 -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device Name:', torch.cuda.get_device_name(0))
    print('Device Count:', torch.cuda.device_count())
    print('Device Capability:', torch.cuda.get_device_capability(0))
    print('Total VRAM (GB):', round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2))
"

# 3. Run Unit Verification Test Suite
echo ""
echo "[*] Running CRBench unit test suite..."
pytest tests/ -v

# 4. Execute Full Scale CUDA Benchmark Sweep
echo ""
echo "[*] Executing CRBench CUDA Full Sweep (8K to 128K context)..."
crbench run --config configs/stage3_cuda_full.yaml

# 5. Output Summary
echo ""
echo "================================================================================"
echo "[✓] CRBench CUDA Full Sweep Completed Successfully!"
echo "    Results directory : results/cuda_full_sweep/"
echo "    Report path       : results/cuda_full_sweep/CRBENCH_REPORT.md"
echo "    Figures directory : results/cuda_full_sweep/figures/"
echo "================================================================================"
