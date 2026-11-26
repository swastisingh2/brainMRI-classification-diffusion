"""Small helpers shared by all scripts: seeding, device selection, code version, and the run registry."""
from __future__ import annotations

import csv
try:
    import fcntl
except ImportError:
    fcntl = None
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int):
    """Seed every random number generator we use, so a run with the same seed starts identically.

    Note: GPU kernels are not fully deterministic, so two runs with the same seed can still differ slightly.
    That is why we report several seeds rather than relying on one.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(min_gpu_gb: float = 0.0) -> torch.device:
    """CUDA, then Apple MPS, then CPU. With min_gpu_gb > 0, refuse a CUDA card smaller than that.

    The size check exists for machines that also have a small display GPU: if the job ends up on it,
    failing immediately is better than running out of memory (or running very slowly) halfway through.
    Default 0 = no check, so the code works on everyone's hardware.
    """
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)  # device 0 = the first GPU this process can see
        gb = props.total_memory / 1024**3
        if gb < min_gpu_gb:
            raise RuntimeError(
                f"Got GPU '{props.name}' with {gb:.1f} GB (< {min_gpu_gb} GB). "
                "Select a larger GPU (e.g. via CUDA_VISIBLE_DEVICES or your scheduler) or lower --min_gpu_gb."
            )
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple-silicon GPU
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    """Readable device name for the run record: the CUDA GPU's model name, or 'mps' / 'cpu'."""
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return device.type


def code_version() -> str:
    """Which version of the code produced a result: the git commit, plus '-dirty' if files were edited.

    Saved into every run's config.json so any number can be traced back to the exact code.
    If the code was copied to a machine without its .git folder, the copy script writes a CODE_VERSION
    file instead, and that is read here.
    """
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                                      stderr=subprocess.DEVNULL, text=True).strip()
        # Only tracked files count as "dirty"; untracked files (notes, PDFs) do not change the code.
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT,
                                        stderr=subprocess.DEVNULL, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        f = REPO_ROOT / "CODE_VERSION"
        return f.read_text().strip() if f.exists() else "unknown"


def append_registry(path: Path, row: dict):
    """Append one line per finished run to a shared CSV. File lock makes parallel jobs safe.

    If a new run has columns the file does not have yet (e.g. 'n_synthetic' added in Phase 2), the header
    is extended and older rows get blanks, so the registry never silently drops information.
    The whole file is rewritten while holding an exclusive lock, so two jobs finishing at the same moment
    cannot interleave their writes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+", newline="") as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)  # wait until no other process is writing
            f.seek(0)
            old = list(csv.DictReader(f))
            header = list(old[0]) if old else []
            header += [k for k in row if k not in header]  # new columns are added, old rows get blanks
            f.seek(0)
            f.truncate()
            w = csv.DictWriter(f, fieldnames=header, restval="")
            w.writeheader()
            w.writerows(old + [row])
            f.flush()
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_UN)


def slurm_job_id() -> str:
    """The Slurm job ID if running under a Slurm scheduler, else 'local'. Recorded for traceability."""
    return os.environ.get("SLURM_JOB_ID", "local")
