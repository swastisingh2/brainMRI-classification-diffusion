"""Classification metrics, including patient-group bootstrap confidence intervals.

Why these metrics:
- accuracy alone hides class imbalance (glioma is ~46% of the data), so we also report
- macro-F1: the F1 score of each class averaged with equal weight, our main model-selection metric;
- per-class precision and recall, and the confusion matrix, to see WHICH tumors are mistaken for which;
- pair confusion: errors between two classes in both directions, used to choose augmentation targets;
- 95% confidence intervals by resampling PATIENT GROUPS, because slices from one patient are strongly
  correlated: resampling individual images would pretend we have more independent evidence than we do.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from . import CLASS_NAMES

LABELS = list(range(len(CLASS_NAMES)))  # [0, 1, 2]; passed explicitly so a missing class still gets a row


def compute_metrics(y_true, y_pred, groups=None, n_boot: int = 1000, seed: int = 0) -> dict:
    """All metrics for one set of predictions, as a JSON-ready dict.

    y_true, y_pred: 0-based class labels per image. groups: patient group per image (enables the
    bootstrap CIs). n_boot: number of bootstrap resamples (0 = skip).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "per_class": {
            c: {"precision": float(p[k]), "recall": float(r[k]), "f1": float(f[k]), "support": int(s[k])}
            for k, c in enumerate(CLASS_NAMES)
        },
        "confusion_matrix": cm.tolist(),  # rows = true class, cols = predicted class
        "pair_confusion": {},
    }
    # Share of the two classes' images that were mistaken for each other (both directions).
    # Example: meningioma<->glioma = (meningiomas called glioma + gliomas called meningioma)
    #          divided by (all meningiomas + all gliomas).
    for i in LABELS:
        for j in LABELS:
            if i < j:
                denom = cm[i].sum() + cm[j].sum()
                out["pair_confusion"][f"{CLASS_NAMES[i]}<->{CLASS_NAMES[j]}"] = {
                    "errors": int(cm[i, j] + cm[j, i]),
                    "rate": float((cm[i, j] + cm[j, i]) / denom) if denom else 0.0,
                }
    if groups is not None and n_boot > 0:
        out["bootstrap_95ci"] = group_bootstrap(y_true, y_pred, np.asarray(groups), n_boot, seed)
    return out


def group_bootstrap(y_true, y_pred, groups, n_boot=1000, seed=0) -> dict:
    """Resample patient groups (not images) with replacement, since slices from one patient are correlated.

    Each resample draws as many groups as there are, with replacement, keeps ALL slices of each drawn
    group, and recomputes accuracy and macro-F1. The 2.5th and 97.5th percentiles over the resamples form
    the 95% interval. A wide interval means the result could easily change with a different set of patients.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in uniq}  # group -> indices of its slices
    accs, f1s = [], []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([members[g] for g in pick])
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
        f1s.append(f1_score(y_true[idx], y_pred[idx], labels=LABELS, average="macro", zero_division=0))
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"accuracy": q(accs), "macro_f1": q(f1s), "n_boot": n_boot, "n_groups": int(len(uniq))}
