"""Device selection and the hardware string recorded with every run."""

from __future__ import annotations

import platform
import subprocess

import psutil
import torch
from torch import nn


def select_device() -> torch.device:
    """CUDA if available, else Apple Silicon MPS, else CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    """Wait for queued work, so a timer measures the device rather than the dispatch queue."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters, reported for every model in a comparison."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def reset_peak_memory(device: torch.device) -> None:
    """Start a fresh peak-memory measurement, where the device can report one."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory(device: torch.device) -> int:
    """Bytes at the high-water mark of device memory, for the compute-cost table.

    CUDA tracks a true peak. MPS exposes no peak counter, so this reports the driver pool,
    which only grows and so is a peak over the process rather than since the last reset. CPU
    has no device pool and reports the resident set size instead.
    """
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if device.type == "mps":
        return int(torch.mps.driver_allocated_memory())
    return int(psutil.Process().memory_info().rss)


def describe_hardware() -> str:
    """Accelerator, host processor, and memory; times only compare across runs on one machine."""
    device = select_device()
    host = f"{_processor()}, {psutil.cpu_count(logical=False)} cores, {_gigabytes(psutil.virtual_memory().total)} RAM"
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(0)
        return f"cuda: {properties.name}, {_gigabytes(properties.total_memory)} VRAM; host {host}"
    if device.type == "mps":
        return f"mps: {host}"
    return f"cpu: {host}"


def _gigabytes(size: int) -> str:
    return f"{size / 1024 ** 3:.0f} GB"


def _processor() -> str:
    """Chip name, which `platform.processor()` reports only as the architecture on macOS."""
    if platform.system() == "Darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return platform.processor() or platform.machine()
