"""Decide which synthetic images to generate: one row per image = (class, source training image).

Strategies, all with the same total --budget so comparisons are fair:
  untargeted  equal share per class; sources drawn uniformly from that class's training images
  ca_budget   class shares proportional to out-of-fold error rate (1 - recall); sources uniform
  ca_hard     same class shares as ca_budget; sources weighted toward images the classifier got wrong or
              was unsure about (weight = 1 - p(true class) + --hard_floor)

Error rates and hardness come from pooled out-of-fold (CV) predictions on the TRAINING split only,
e.g. results/summary/cv_CV_seed0_oof_predictions.csv. Validation and test are never used.

Usage:
  python scripts/make_synthetic_plan.py --oof_csv <oof.csv> --strategy ca_hard --budget 1100 --out plan.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainmri import CLASS_NAMES  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--oof_csv", type=Path, required=True)
p.add_argument("--strategy", choices=["untargeted", "ca_budget", "ca_hard"], required=True)
p.add_argument("--budget", type=int, required=True, help="total number of synthetic images")
p.add_argument("--min_share", type=float, default=0.05, help="ca_*: minimum share of the budget per class")
p.add_argument("--hard_floor", type=float, default=0.02, help="ca_hard: base weight so every image can be picked")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", type=Path, required=True)
a = p.parse_args()

# ---- read the pooled out-of-fold predictions: one row per training image, with its true label, the CV
# model's prediction, and the probability that model gave to each class.
rows = list(csv.DictReader(open(a.oof_csv)))
paths = np.array([r["relative_path"] for r in rows])
y = np.array([int(r["label"]) for r in rows])
pred = np.array([int(r["pred"]) for r in rows])
p_true = np.array([float(r[f"prob_{CLASS_NAMES[c]}"]) for r, c in zip(rows, y)])
assert len(set(paths)) == len(paths), "duplicate images in OOF file"

# ---- how many synthetic images per class
# err[c] = out-of-fold error rate of class c (1 - recall). With our baseline: meningioma 0.162,
# glioma 0.038, pituitary 0.031, so confusion-aware plans give meningioma ~70% of the budget.
k = len(CLASS_NAMES)
err = np.array([1 - (pred[y == c] == c).mean() for c in range(k)])
if a.strategy == "untargeted":
    share = np.full(k, 1 / k)
else:
    share = err / err.sum()
    share = np.maximum(share, a.min_share)  # never starve a class completely (default floor 5%)
    share /= share.sum()
counts = np.floor(share * a.budget).astype(int)
counts[np.argmax(share)] += a.budget - counts.sum()  # rounding remainder to the largest class

# ---- which training image each synthetic image starts from (its img2img "source")
rng = np.random.default_rng(a.seed)
plan = []
for c in range(k):
    members = np.flatnonzero(y == c)  # this class's training images
    if a.strategy == "ca_hard":
        # Hardness weight: 1 - p(true class). A confidently correct image (p = 0.99) gets weight ~0.03;
        # a misclassified one (p = 0.2) gets 0.82, i.e. ~27x more likely to be picked as a source.
        w = (1 - p_true[members]) + a.hard_floor
    else:
        w = np.ones(len(members))
    # Sample sources with replacement in proportion to w; repeated sources get different noise seeds.
    pick = rng.choice(members, size=counts[c], replace=True, p=w / w.sum())
    for i in pick:
        plan.append({"label": c, "class_name": CLASS_NAMES[c], "source_path": paths[i],
                     "source_pred": int(pred[i]), "source_p_true": round(float(p_true[i]), 4)})
rng.shuffle(plan)  # mix classes so a partial run (--limit) still covers all of them
for n, r in enumerate(plan):
    r["row"] = n  # row number = the image's ID; generation seeds its noise with seed + row

a.out.parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["row", "label", "class_name", "source_path", "source_pred", "source_p_true"])
    w.writeheader()
    w.writerows(plan)
# ---- a short summary (<out>.json) to compare plans at a glance: class counts, how many distinct source
# images were used, and how "hard" the chosen sources are on average.
src = np.array([r["source_path"] for r in plan])
lab = np.array([r["label"] for r in plan])
summary = {
    "strategy": a.strategy, "budget": a.budget, "seed": a.seed, "oof_csv": a.oof_csv.name,
    "oof_error_rate": dict(zip(CLASS_NAMES, np.round(err, 4).tolist())),
    "per_class": dict(zip(CLASS_NAMES, counts.tolist())),
    "unique_sources_per_class": {CLASS_NAMES[c]: int(len(set(src[lab == c]))) for c in range(k)},
    "share_from_misclassified_sources": round(float(np.mean([r["source_pred"] != r["label"] for r in plan])), 4),
    "mean_source_p_true": round(float(np.mean([r["source_p_true"] for r in plan])), 4),
}
a.out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
