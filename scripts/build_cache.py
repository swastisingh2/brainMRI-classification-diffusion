"""Preprocess all 3064 Figshare slices into one .npz cache.

Run once per machine (about 5 minutes). It checks every file's label and patient ID against the split
CSVs, normalises intensities per image, resizes, and stores images + tumor masks + split info together.
Use --size 256 for the classifiers (default) and --size 512 for the diffusion model.
The cache contains the MRI images: keep it out of git.

Example:
    python scripts/build_cache.py /path/to/figshare_data /path/to/cache/cache_256.npz
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainmri.data import build_cache  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("data_root", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--size", type=int, default=256)
a = p.parse_args()
meta = build_cache(a.data_root, a.out, a.size)
print(json.dumps(meta, indent=2))
print(f"Wrote {a.out} ({a.out.stat().st_size / 1e6:.0f} MB)")
