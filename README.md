# Brain tumor dataset handoff

Version: splits_v1_provisional. Seed: 42.

## What is included
Portable split CSVs and scripts. MRI data, masks, model weights and training code are not included. The assignments are identical to the original seed-42 splits; only the absolute Windows paths were replaced.

## Dataset setup
Use the Cheng/Figshare dataset discussed by the team:
https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
The supplied inventory has 3,064 slices and 233 original patient IDs. The precise downloaded release version was not recorded; use the same extracted dataset copy as the sender when possible. No per-image content checksums are available in this package.

**Verified release (2026-11-24):** Figshare article 1512427 **version 8** (last modified 2024-12-21) matches these splits. The four zip files matched Figshare's published MD5s (`brainTumorDataPublic_1-766.zip` 74b949ad33f042e6e103523091cd1428, `_767-1532.zip` 7e8a875500d2c8a346f270538e29890e, `_1533-2298.zip` 8227bf6080cb71f15a88be8d25c79ae7, `_2299-3064.zip` b378a80d6174e5317d59eb28430c6652). Each zip holds flat `.mat` files; extract each into a folder named after the zip. `check_data.py` passes, and the label and patient ID stored inside all 3,064 `.mat` files match `manifest_with_splits.csv`. 15 images are 256×256 (the rest 512×512) and every image has a non-empty tumor mask.

Set DATA_ROOT to the folder containing brainTumorDataPublic_1-766 and the other extracted image folders. Keep those subfolder names unchanged. Dataset access and reuse follow the source terms.

## Check your local dataset
From this package folder, run:

```powershell
python scripts/check_data.py "D:\path	o\dataset"
```

This checks file existence and split consistency, not image contents or near-duplicates. It requires only Python's standard library.

## Load an image path

```python
import csv
from pathlib import Path
DATA_ROOT = Path("/your/path/to/dataset")
with open("splits/train.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
image_path = DATA_ROOT / rows[0]["relative_path"]
label = int(rows[0]["label"]) - 1
```

Labels in the CSV: 1 meningioma, 2 glioma, 3 pituitary. For zero-based model labels subtract 1. The masks remain inside the original .mat files. All original patient IDs and grouping notes are preserved.

## Agreed split
| Split | Images | Groups |
|---|---:|---:|
| Train | 2193 | 147 |
| Validation | 363 | 31 |
| Test | 508 | 31 |

Approximately 70/15/15 of split groups, stratified separately within each class. Images were not randomly split. A stable SHA256(seed:group) order is used within each class; each holdout receives rounded 15% of the class's groups. Different group sizes explain differing image percentages.

Shared MR numeric prefixes were conservatively grouped, resulting in 209 split groups. These are not 209 verified patients. MR024780 received visual review; the other shared prefixes were grouped automatically by team decision. No images were removed by this grouping.

## Team rules
Use the supplied CSV assignments for every model. Do not split the images again or change seeds based on model performance. Fit learned preprocessing on training only. Use validation for settings and checkpoint selection. Keep test for final evaluation. Generate augmented data only from the allowed training groups; low-data experiments must also restrict the generator's training data.

## Audit status
Verified: all records assigned once, all classes represented in each split, and no split-group, original-patient-ID or path overlap.
Pending: review remaining near-duplicate candidates crossing splits. This is a provisional handoff, not a claim that the dataset is duplicate-free. Complete that audit before finalizing the experiment splits.

## Baseline classifier (Phase 1)

Code: `brainmri/` (data, model, metrics, plots) and `scripts/`. Install with `pip install -r requirements.txt`.

1. Build the preprocessed cache once (per-image 0.5/99.5 percentile clip, 256px, images + masks):
   `python scripts/build_cache.py <DATA_ROOT> <cache_dir>/cache_256.npz`
2. Train one run. Checkpoints are selected on validation macro-F1; the test split is never used here:
   `python scripts/train_classifier.py --cache <cache.npz> --out_root runs --results_root results --exp B1 --aug standard --seed 0`
   Add `--fold k` (k = 0-4) for a group-stratified cross-validation fold of train, which also saves out-of-fold predictions. Add `--smoke` for a 2-epoch pipeline test.
3. Summarise: `python scripts/summarize_runs.py results` writes tables and figures to `results/summary/`, including the pooled out-of-fold confusion matrix used to pick which tumor classes to target.
4. Final test numbers only, after every setting is fixed: `python scripts/evaluate.py --ckpt <run>/best.pt --cache <cache.npz> --split test`.

Experiments: **B0** no augmentation, **B1** standard augmentation (flip, ±15° rotation, small shift/scale, brightness/contrast), each with seeds 0-4; **CV** B1 settings over 5 folds for confusion analysis.

### Where results are kept
| What | Where |
|---|---|
| Per-run metrics, predictions, curves, confusion matrices, log, config | `<results_root>/<run_name>/` |
| One line per finished run | `<results_root>/registry.csv` |
| Aggregated tables and figures | `<results_root>/summary/` |
| Checkpoint (`best.pt`) | `<out_root>/<run_name>/`, plus `<ckpt_store>/<run_name>/` if given |

`results/`, `runs/`, `checkpoints/` and `logs/` are git-ignored: run configs record machine paths, user names and job IDs, so they stay with whoever ran them. Share numbers deliberately (for example a copy of `summary/summary_val.csv`).

Run names are `<exp>_<aug>_<model>_seed<N>[_fold<k>]_<timestamp>`. Each `config.json` records the code version, GPU and cache fingerprint.

Each member runs on their own hardware; keep machine-specific job scripts (Slurm files, paths, hostnames) outside the repo. The device is picked automatically (CUDA, then Apple MPS, then CPU). `--min_gpu_gb N` makes a run refuse a GPU smaller than N GB, which helps on machines that also have a small display GPU.

## Reproduce (optional)
The supplied CSVs are ready to read; regeneration is not required.

```powershell
python scripts/create_splits.py splits/input_manifest.csv --out reproduced_splits --seed 42
```

Keep the split version and Git commit with experiment results. split_metadata.json records the input fingerprints; CHECKSUMS.sha256 covers package files, not MRI contents.
