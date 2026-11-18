# Brain tumor dataset handoff

Version: splits_v1_provisional. Seed: 42.

## What is included
Portable split CSVs and scripts. MRI data, masks, model weights and training code are not included. The assignments are identical to the original seed-42 splits; only the absolute Windows paths were replaced.

## Dataset setup
Use the Cheng/Figshare dataset discussed by the team:
https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
The supplied inventory has 3,064 slices and 233 original patient IDs. The precise downloaded release version was not recorded; use the same extracted dataset copy as the sender when possible. No per-image content checksums are available in this package.

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

## Reproduce (optional)
The supplied CSVs are ready to read; regeneration is not required.

```powershell
python scripts/create_splits.py splits/input_manifest.csv --out reproduced_splits --seed 42
```

Keep the split version and Git commit with experiment results. split_metadata.json records the input fingerprints; CHECKSUMS.sha256 covers package files, not MRI contents.
