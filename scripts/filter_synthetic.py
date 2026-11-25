"""Drop synthetic images whose tumor faded away during generation.

Why: img2img keeps the layout of the source slice, so the source's tumor mask still marks where the tumor
should be in the synthetic image. If the tumor no longer stands out from the tissue around it, the image
still carries its tumor label but no longer shows that tumor: label noise that can hurt the classifier.

Tumor contrast = |mean intensity inside the mask - mean intensity in a ring just outside it|.
An image is kept if its contrast is at least --min_ratio times the source image's contrast.

Writes <out>.npz (same keys as generate_synthetic.py, so train_classifier.py --synthetic accepts it)
and <out>.json with keep rates per class and the contrast ratio of every image.

Example:
    python scripts/filter_synthetic.py --synthetic syn/E2.npz --cache cache/cache_256.npz --out syn/E2_filt
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainmri import CLASS_NAMES  # noqa: E402
from brainmri.data import Cache  # noqa: E402


def tumor_contrast(img: np.ndarray, mask: np.ndarray, ring_px: int) -> float:
    """|mean inside mask - mean of a ring_px-wide band around it| on a 0-1 intensity scale."""
    mask = mask.astype(bool)
    ring = ndimage.binary_dilation(mask, iterations=ring_px) & ~mask
    if not mask.any() or not ring.any():
        return float("nan")
    x = img.astype(np.float32) / 255.0
    return float(abs(x[mask].mean() - x[ring].mean()))


p = argparse.ArgumentParser()
p.add_argument("--synthetic", type=Path, required=True, help="npz from generate_synthetic.py")
p.add_argument("--cache", type=Path, required=True, help="real-image cache with masks, same size as the synthetic images")
p.add_argument("--out", type=Path, required=True, help="output path without extension")
p.add_argument("--min_ratio", type=float, default=0.5, help="keep if synthetic contrast >= this x source contrast")
p.add_argument("--ring_px", type=int, default=8)
a = p.parse_args()

z = np.load(a.synthetic)
cache = Cache(a.cache)
syn = z["images"]
if syn.shape[1:] != cache.images.shape[1:]:
    sys.exit(f"Size mismatch: synthetic {syn.shape[1:]} vs cache {cache.images.shape[1:]}; use the matching cache.")
path_to_idx = {str(pth): i for i, pth in enumerate(cache.paths)}
train = set(cache.indices("train").tolist())

ratios = np.empty(len(syn), dtype=np.float32)
for k, sp in enumerate(z["source_paths"].astype(str)):
    i = path_to_idx[sp]
    assert i in train, f"source {sp} is not a training image"  # same leakage guard as the rest of the pipeline
    c_src = tumor_contrast(cache.images[i], cache.masks[i], a.ring_px)
    c_syn = tumor_contrast(syn[k], cache.masks[i], a.ring_px)
    ratios[k] = c_syn / c_src if c_src > 1e-3 else np.nan

keep = np.nan_to_num(ratios, nan=0.0) >= a.min_ratio
labels = z["labels"]
a.out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(a.out.with_suffix(".npz"), images=syn[keep], labels=labels[keep],
                    source_paths=z["source_paths"][keep], rows=z["rows"][keep])
report = {
    "synthetic": str(a.synthetic), "min_ratio": a.min_ratio, "ring_px": a.ring_px,
    "n_in": int(len(syn)), "n_kept": int(keep.sum()),
    "per_class": {c: {"in": int((labels == j).sum()), "kept": int(((labels == j) & keep).sum()),
                      "median_ratio": float(np.nanmedian(ratios[labels == j])) if (labels == j).any() else None}
                  for j, c in enumerate(CLASS_NAMES)},
    "ratios": [None if np.isnan(r) else round(float(r), 4) for r in ratios],
}
a.out.with_suffix(".json").write_text(json.dumps(report, indent=1))
print(f"kept {keep.sum()}/{len(syn)}  " + "  ".join(
    f"{c}: {v['kept']}/{v['in']} (median ratio {v['median_ratio']:.2f})"
    for c, v in report["per_class"].items() if v["in"]))
