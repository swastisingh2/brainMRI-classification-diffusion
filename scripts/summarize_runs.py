"""Aggregate finished runs in a results folder into tables and figures.

Writes to <results>/summary/:
    summary_val.csv               mean / std / min / max per experiment (normal runs, smoke excluded)
    summary_val_macro_f1.png      bar = mean, error bar = std, dots = individual seeds
    summary_val_accuracy.png
    cv_<exp>_seed<s>_oof.json     pooled out-of-fold metrics over all CV folds (which classes get confused)
    cv_<exp>_seed<s>_oof_confusion.png
    cv_<exp>_seed<s>_oof_predictions.csv

Usage:  python scripts/summarize_runs.py results
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainmri import CLASS_NAMES  # noqa: E402
from brainmri.metrics import compute_metrics  # noqa: E402
from brainmri.plots import plot_confusion, plot_summary  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("results", type=Path)
a = p.parse_args()
out = a.results / "summary"
out.mkdir(parents=True, exist_ok=True)

# ---- collect finished runs: every results/<run>/ folder with a metrics.json. Smoke tests and unfinished
# runs are skipped so they never enter the tables.
runs = []
for mf in sorted(a.results.glob("*/metrics.json")):
    cfg = json.loads((mf.parent / "config.json").read_text())
    if cfg.get("smoke") or cfg.get("status") != "done":
        continue
    runs.append((cfg, json.loads(mf.read_text()), mf.parent))
print(f"{len(runs)} finished runs found")

# ---- normal runs: per-experiment mean/std on validation
by_exp = defaultdict(list)
for cfg, m, _ in runs:
    if cfg["fold"] < 0:
        by_exp[cfg["exp"]].append((cfg, m))

metrics = ["accuracy", "macro_f1"] + [f"recall_{c}" for c in CLASS_NAMES]


def get(m, k):
    # One validation metric from a run's metrics.json: "accuracy", "macro_f1" or "recall_<class>".
    return m["val"]["per_class"][k[7:]]["recall"] if k.startswith("recall_") else m["val"][k]


rows = []
for exp in sorted(by_exp):
    items = by_exp[exp]
    r = {"exp": exp, "n_seeds": len(items), "seeds": " ".join(str(c["seed"]) for c, _ in items)}
    for k in metrics:
        v = np.array([get(m, k) for _, m in items])
        # std uses ddof=1 (sample standard deviation), the usual choice for a handful of seeds.
        r.update({f"{k}_mean": round(v.mean(), 4), f"{k}_std": round(v.std(ddof=1) if len(v) > 1 else 0.0, 4),
                  f"{k}_min": round(v.min(), 4), f"{k}_max": round(v.max(), 4)})
    rows.append(r)
    print(f"  {exp}: n={len(items)}  val macro-F1 {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.4f}  "
          f"acc {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.4f}")
if rows:
    with open(out / "summary_val.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for k, label in (("macro_f1", "Validation macro-F1"), ("accuracy", "Validation accuracy")):
        plot_summary([{"exp": e, "mean": float(np.mean(vals := [get(m, k) for _, m in by_exp[e]])),
                       "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, "values": vals}
                      for e in sorted(by_exp)], label, out / f"summary_val_{k}.png",
                     f"{label}: mean ± std over seeds")

# ---- CV runs: pool out-of-fold predictions across folds
# Each fold run predicted ~1/5 of the training images; together the 5 folds cover all 2,193 exactly once.
# The pooled file (cv_<exp>_seed<s>_oof_predictions.csv) is the input to make_synthetic_plan.py.
by_cv = defaultdict(dict)
for cfg, m, d in runs:
    if cfg["fold"] >= 0 and (d / "predictions_oof.csv").exists():
        by_cv[(cfg["exp"], cfg["seed"])][cfg["fold"]] = d
for (exp, seed), folds in sorted(by_cv.items()):
    preds = []
    for k in sorted(folds):
        with open(folds[k] / "predictions_oof.csv") as f:
            for r in csv.DictReader(f):
                r["fold"] = k
                preds.append(r)
    y = np.array([int(r["label"]) for r in preds])
    yhat = np.array([int(r["pred"]) for r in preds])
    groups = np.array([r["split_group"] for r in preds])
    stem = f"cv_{exp}_seed{seed}"
    m = compute_metrics(y, yhat, groups)
    m["folds_present"] = sorted(folds)
    (out / f"{stem}_oof.json").write_text(json.dumps(m, indent=2))
    plot_confusion(m["confusion_matrix"], out / f"{stem}_oof_confusion.png",
                   f"{exp} seed {seed} - pooled out-of-fold ({len(folds)} folds, n={len(y)})")
    with open(out / f"{stem}_oof_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(preds[0]))
        w.writeheader()
        w.writerows(preds)
    worst = max(m["pair_confusion"].items(), key=lambda kv: kv[1]["rate"])
    print(f"  CV {exp} seed {seed}: folds {sorted(folds)}  OOF macro-F1 {m['macro_f1']:.4f}  "
          f"most-confused pair {worst[0]} ({worst[1]['errors']} errors, {worst[1]['rate']:.1%})")
print(f"Wrote summary to {out}")
