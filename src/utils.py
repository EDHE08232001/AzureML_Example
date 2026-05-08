"""
General utility helpers: reproducibility, device selection, and logging.
"""

import os
import random
from typing import Dict

import numpy as np
import torch


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Device selection ───────────────────────────────────────────────────────────

def get_device(preferred: str = "auto") -> torch.device:
    """Return the best available compute device.

    preferred:
        "auto"  — try CUDA, then MPS, then CPU (default).
        "cuda"  — use CUDA if available, else fall back to CPU.
        "mps"   — use Apple MPS if available, else fall back to CPU.
        "cpu"   — always use CPU.
    """
    if preferred in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred in ("auto", "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Directory helper ───────────────────────────────────────────────────────────

def ensure_dir(path: str) -> str:
    """Create a directory (and parents) if it does not exist; return absolute path."""
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


# ── Logging helper ─────────────────────────────────────────────────────────────

def format_metrics(metrics: Dict[str, float]) -> str:
    """Format a metrics dict as a compact single-line string."""
    return "  ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
