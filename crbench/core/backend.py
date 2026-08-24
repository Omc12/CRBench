"""
Backend capability detection and hardware management for CRBench.
CUDA is the primary high-performance target; MPS and CPU provide graceful fallback
with explicit capability flags and unsupported warnings.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import torch


@dataclass
class DeviceCapabilities:
    """Detailed hardware capability metadata."""
    device_type: str                   # "cuda", "mps", "cpu"
    device_name: str                   # e.g. "NVIDIA A100-SXM4-80GB", "Apple M-series", "Intel CPU"
    total_memory_bytes: int            # Total physical device memory in bytes
    available_memory_bytes: int        # Available memory before allocation
    supports_float16: bool
    supports_bfloat16: bool
    supports_flash_attn: bool
    supports_mps_sync: bool
    compute_capability: Optional[Tuple[int, int]] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_bytes / (1024.0 ** 3)


class BackendManager:
    """
    Manages device selection, backend capability queries, and runtime fallbacks.
    """

    @staticmethod
    def get_preferred_device(override_device: Optional[str] = None) -> torch.device:
        """
        Determines the optimal device:
        1. User override if provided.
        2. CUDA if available.
        3. Apple MPS if available on macOS.
        4. CPU fallback.
        """
        if override_device is not None:
            return torch.device(override_device)

        if torch.cuda.is_available():
            return torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def get_capabilities(device: Optional[torch.device] = None) -> DeviceCapabilities:
        """
        Queries and returns the hardware capability descriptor for the given device.
        """
        dev = device or BackendManager.get_preferred_device()
        dev_type = dev.type

        if dev_type == "cuda" and torch.cuda.is_available():
            dev_idx = dev.index or 0
            props = torch.cuda.get_device_properties(dev_idx)
            total_mem = props.total_memory
            try:
                free_mem, _ = torch.cuda.mem_get_info(dev_idx)
            except Exception:
                free_mem = total_mem
            major, minor = props.major, props.minor
            supports_bf16 = major >= 8 or (major == 7 and minor >= 5)
            supports_fa = major >= 8

            return DeviceCapabilities(
                device_type="cuda",
                device_name=props.name,
                total_memory_bytes=total_mem,
                available_memory_bytes=free_mem,
                supports_float16=True,
                supports_bfloat16=supports_bf16,
                supports_flash_attn=supports_fa,
                supports_mps_sync=False,
                compute_capability=(major, minor),
                properties={"multi_processor_count": getattr(props, "multi_processor_count", None)}
            )

        elif dev_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            import psutil
            total_ram = psutil.virtual_memory().total
            avail_ram = psutil.virtual_memory().available
            return DeviceCapabilities(
                device_type="mps",
                device_name="Apple Silicon (MPS)",
                total_memory_bytes=total_ram,
                available_memory_bytes=avail_ram,
                supports_float16=True,
                supports_bfloat16=True,
                supports_flash_attn=False,
                supports_mps_sync=True,
                compute_capability=None,
                properties={"os": "macOS", "unified_memory": True}
            )

        else:
            import psutil
            total_ram = psutil.virtual_memory().total
            avail_ram = psutil.virtual_memory().available
            return DeviceCapabilities(
                device_type="cpu",
                device_name="Host CPU",
                total_memory_bytes=total_ram,
                available_memory_bytes=avail_ram,
                supports_float16=False,
                supports_bfloat16=True,
                supports_flash_attn=False,
                supports_mps_sync=False,
                compute_capability=None,
                properties={"cores": psutil.cpu_count(logical=False), "threads": psutil.cpu_count(logical=True)}
            )

    @staticmethod
    def validate_method_support(method_type: str, device: torch.device) -> Tuple[bool, str]:
        """
        Validates if the requested compression method is executable on the given device.
        """
        caps = BackendManager.get_capabilities(device)

        if method_type in ("flash_attn", "flash_attention_2"):
            if not caps.supports_flash_attn:
                return False, f"FlashAttention requires CUDA compute capability >= 8.0 (found {caps.device_type})"

        if method_type in ("fp8", "e4m3", "e5m2"):
            if caps.device_type != "cuda" or (caps.compute_capability and caps.compute_capability[0] < 8):
                return False, f"FP8 hardware acceleration requires NVIDIA Ada/Hopper architecture (found {caps.device_type})"

        return True, "Supported"
