"""Loading the Figshare .mat files, preprocessing, the on-disk cache, and the PyTorch dataset.

Data flow in this project:

    Figshare .mat files (3,064 slices, mostly 512x512, int16 intensities)
        |  build_cache(): read each file, normalise intensity per image, resize, store as uint8
        v
    one cache .npz (images, tumor masks, labels, split, patient group, path for every slice)
        |  Cache(): load the whole .npz into memory once
        v
    MRIDataset(cache, indices, transform): what the PyTorch DataLoader iterates over

Why a cache: reading 3,064 HDF5 files every epoch is slow; the cache is ~125 MB at 256 px and loads in
seconds. Preprocessing uses only each image's own pixels, so no information flows between splits.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import v2

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "splits"  # the agreed split CSVs (train.csv / val.csv / test.csv)

# ImageNet statistics: the classifiers start from ImageNet-pretrained weights, which expect inputs
# normalised this way. These are fixed constants, not computed from our data.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------- raw files
def load_mat(path: Path) -> dict:
    """Read one Figshare .mat file and return its image, tumor mask, label (1-3) and patient ID.

    The files are MATLAB v7.3, which is HDF5 underneath, so h5py reads them (scipy.io cannot).
    MATLAB stores arrays column-major, so h5py returns them transposed; `.T` restores the orientation
    you would see in MATLAB (and in the Figshare documentation).
    """
    with h5py.File(path, "r") as f:
        c = f["cjdata"]  # every file has one struct called cjdata with the fields below
        image = np.array(c["image"], dtype=np.float32).T
        mask = np.array(c["tumorMask"], dtype=np.uint8).T  # 1 inside the tumor, 0 elsewhere
        label = int(np.array(c["label"]).item())  # 1 meningioma, 2 glioma, 3 pituitary
        # The patient ID is stored as an array of character codes; join them back into a string.
        pid = "".join(chr(int(x)) for x in np.array(c["PID"]).ravel())
    return {"image": image, "mask": mask, "label": label, "pid": pid}


def normalize_intensity(image: np.ndarray, lo_pct: float = 0.5, hi_pct: float = 99.5) -> np.ndarray:
    """Per-image percentile clip, then scale to [0, 1].

    MRI intensities have no fixed scale (they differ between scanners and scans), so each image is
    normalised by its own 0.5th and 99.5th percentiles. Clipping at percentiles rather than min/max
    stops a few very bright pixels from squashing the rest of the image into dark grey.
    Uses only the image itself (no dataset statistics), so it cannot leak information between splits.
    """
    lo, hi = np.percentile(image, [lo_pct, hi_pct])
    if hi <= lo:  # a flat image (never seen in this dataset, but avoids dividing by zero)
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def resize_pair(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Resize an image and its tumor mask to size x size.

    The image uses bilinear interpolation with antialiasing (smooth, no aliasing when shrinking 512 -> 256).
    The mask uses nearest-neighbour so it stays strictly 0/1 instead of getting blurry 0.5 edges.
    """
    img_t = torch.from_numpy(image)[None, None]  # add batch and channel dims: (1, 1, H, W)
    msk_t = torch.from_numpy(mask.astype(np.float32))[None, None]
    img_r = F.interpolate(img_t, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
    msk_r = F.interpolate(msk_t, size=(size, size), mode="nearest")
    return img_r[0, 0].clamp(0, 1).numpy(), msk_r[0, 0].numpy().astype(np.uint8)


def read_split_rows(splits_dir: Path = SPLITS_DIR) -> list[dict]:
    """All rows of train.csv, val.csv and test.csv, in that order (each row knows its own split)."""
    rows = []
    for split in ("train", "val", "test"):
        with open(splits_dir / f"{split}.csv", newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def file_sha256(path: Path) -> str:
    """Fingerprint of a file, recorded in the cache metadata so we know which split files built it."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_cache(data_root: Path, out_path: Path, size: int = 256, splits_dir: Path = SPLITS_DIR) -> dict:
    """Preprocess all 3064 slices once into a single .npz (uint8 images + masks + metadata).

    Every slice listed in the split CSVs is read, checked, normalised, resized and stored. The row order
    of the cache is the CSV order, and each row carries its split name, so the training code selects
    train / val / test by that column (Cache.indices) rather than by position.
    Also writes <out>.json describing how the cache was made.
    """
    rows = read_split_rows(splits_dir)
    n = len(rows)
    images = np.zeros((n, size, size), dtype=np.uint8)  # uint8 (0-255) keeps the cache 4x smaller than float
    masks = np.zeros((n, size, size), dtype=np.uint8)
    labels = np.zeros(n, dtype=np.int64)
    orig_shapes = []
    for i, r in enumerate(rows):
        d = load_mat(Path(data_root) / r["relative_path"])
        # Safety check: the label and patient ID inside the file must match the split CSV. A mismatch means
        # a different dataset release or a mislabelled file, and every downstream number would be wrong.
        if d["label"] != int(r["label"]) or d["pid"] != r["patient_id"]:
            raise ValueError(f"Label/patient mismatch for {r['relative_path']}: file says {d['label']}/{d['pid']}")
        img, msk = resize_pair(normalize_intensity(d["image"]), d["mask"], size)
        images[i] = np.round(img * 255).astype(np.uint8)
        masks[i] = msk
        labels[i] = d["label"] - 1  # model labels are 0-based: 0 meningioma, 1 glioma, 2 pituitary
        orig_shapes.append(d["image"].shape)
        if (i + 1) % 500 == 0:
            print(f"  cached {i + 1}/{n}", flush=True)
    # Human-readable record of how this cache was made (saved next to it as .json).
    meta = {
        "size": size,
        "intensity": "per-image clip at 0.5/99.5 percentiles, scaled to [0,1], stored as uint8",
        "resize": "bilinear+antialias for images, nearest for masks",
        "n_images": n,
        "n_orig_256": sum(1 for s in orig_shapes if s == (256, 256)),  # 15 slices are 256x256 in the source
        "split_files_sha256": {s: file_sha256(splits_dir / f"{s}.csv") for s in ("train", "val", "test")},
        "data_root": str(data_root),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        images=images,
        masks=masks,
        labels=labels,
        split=np.array([r["split"] for r in rows]),  # "train" / "val" / "test"
        groups=np.array([r["split_group"] for r in rows]),  # patient group used for splitting and CV
        patient_ids=np.array([r["patient_id"] for r in rows]),
        paths=np.array([r["relative_path"] for r in rows]),  # stable ID of each slice across all files
    )
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return meta


class Cache:
    """The preprocessed cache, loaded fully into memory (about 125 MB at 256 px, 450 MB at 512 px).

    All arrays are aligned: index j refers to the same slice in images, masks, labels, split, groups,
    patient_ids and paths. Training code passes around these integer indices, not the images themselves.
    """

    def __init__(self, path: Path):
        z = np.load(path, allow_pickle=False)  # allow_pickle=False: never execute code from a data file
        self.images = z["images"]
        self.masks = z["masks"]
        self.labels = z["labels"]
        self.split = z["split"].astype(str)
        self.groups = z["groups"].astype(str)
        self.patient_ids = z["patient_ids"].astype(str)
        self.paths = z["paths"].astype(str)
        meta_path = Path(path).with_suffix(".json")
        self.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def indices(self, split: str) -> np.ndarray:
        """Cache indices of every slice in one split ("train", "val", "test" or "synthetic")."""
        return np.flatnonzero(self.split == split)

    def append_synthetic(self, path: Path) -> np.ndarray:
        """Append a generated set (scripts/generate_synthetic.py) as split 'synthetic'; returns its indices.

        Refuses sets whose source images are not training images, so val/test never leak into training.
        The caller adds the returned indices to the TRAINING indices only. The cache on disk is not changed;
        this only extends the in-memory arrays for the current run.
        """
        z = np.load(path, allow_pickle=False)
        imgs, labels = z["images"], z["labels"].astype(np.int64)
        if imgs.shape[1:] != self.images.shape[1:]:
            raise ValueError(f"{path}: image size {imgs.shape[1:]} != cache {self.images.shape[1:]}")
        # Leakage guard: every img2img source must be a training slice. (A txt2img set has no real source, but
        # the plan still records a training image per row, so the check applies to every set.)
        train_paths = set(self.paths[self.split == "train"])
        bad = set(z["source_paths"].astype(str)) - train_paths
        if bad:
            raise RuntimeError(f"{path}: {len(bad)} source images are not in the training split, e.g. {sorted(bad)[:3]}")
        start, n, stem = len(self.labels), len(labels), Path(path).stem
        self.images = np.concatenate([self.images, imgs])
        self.masks = np.concatenate([self.masks, np.zeros_like(imgs)])  # synthetic images have no masks
        self.labels = np.concatenate([self.labels, labels])
        self.split = np.concatenate([self.split, np.full(n, "synthetic")])
        self.groups = np.concatenate([self.groups, np.array([f"syn:{stem}"] * n)])
        self.patient_ids = np.concatenate([self.patient_ids, np.array([""] * n)])
        self.paths = np.concatenate([self.paths, np.array([f"synthetic/{stem}/{i}" for i in range(n)])])
        return np.arange(start, start + n)


# ---------------------------------------------------------------- transforms
def make_transform(aug: str, img_size: int = 224):
    """Build the image transform. Transforms operate on a 1xHxW float tensor in [0, 1].

    aug="none"     (B0, and always for validation/test): only resize to img_size.
    aug="standard" (B1): random flip / rotation / shift / scale / brightness-contrast, then resize.
    Horizontal flips are reasonable for brain MRI (left-right roughly symmetric); vertical flips are not
    used because an upside-down brain never occurs in real scans.
    """
    if aug == "none":
        ops = []
    elif aug == "standard":
        ops = [
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.9, 1.1)),  # +-15 deg, +-5% shift
            v2.ColorJitter(brightness=0.2, contrast=0.2),  # mimics scanner-to-scanner intensity differences
        ]
    else:
        raise ValueError(f"unknown aug '{aug}'")
    ops.append(v2.Resize((img_size, img_size), antialias=True))  # 256 px cache -> 224 px network input
    return v2.Compose(ops)


class MRIDataset(Dataset):
    """PyTorch dataset over a subset of the cache.

    idx = which cache rows belong to this dataset (e.g. Cache.indices("train")).
    Each item is (image tensor 3x224x224, label 0-2, cache index). The cache index travels with the batch
    so predictions can be written back next to the right file path and patient.
    """

    def __init__(self, cache: Cache, idx: np.ndarray, transform):
        self.cache = cache
        self.idx = np.asarray(idx)
        self.transform = transform
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])  # position in the cache
        x = torch.from_numpy(self.cache.images[j]).float().div_(255.0).unsqueeze(0)  # uint8 -> [0,1], 1xHxW
        x = self.transform(x).clamp_(0, 1)  # augmentation can push values slightly outside [0, 1]
        x = (x.expand(3, -1, -1) - self.mean) / self.std  # grayscale -> 3 channels for ImageNet weights
        return x, int(self.cache.labels[j]), j
