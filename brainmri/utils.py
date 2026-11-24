"""Seeding, device selection, code version, and the shared run registry."""
from __future__ import annotations

import csv
import fcntl
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(min_gpu_gb: float = 0.0) -> torch.device:
    """CUDA, then Apple MPS, then CPU. With min_gpu_gb > 0, refuse a CUDA card smaller than that."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gb = props.total_memory / 1024**3
        if gb < min_gpu_gb:
            raise RuntimeError(
                f"Got GPU '{props.name}' with {gb:.1f} GB (< {min_gpu_gb} GB). "
                "Select a larger GPU (e.g. via CUDA_VISIBLE_DEVICES or your scheduler) or lower --min_gpu_gb."
            )
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return device.type


def code_version() -> str:
    """Git commit (+'-dirty'). On the cluster the code is rsynced without .git, so read CODE_VERSION instead."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        f = REPO_ROOT / "CODE_VERSION"
        return f.read_text().strip() if f.exists() else "unknown"


def append_registry(path: Path, row: dict):
    """Append one line per finished run to a shared CSV. File lock makes parallel jobs safe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        existing = f.read()
        header = existing.splitlines()[0].split(",") if existing else list(row)
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if not existing:
            w.writeheader()
        w.writerow(row)
        fcntl.flock(f, fcntl.LOCK_UN)


def slurm_job_id() -> str:
    return os.environ.get("SLURM_JOB_ID", "local")
