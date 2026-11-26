"""Static PNG figures for run folders and summaries (matplotlib, headless).

plot_curves     -> <run>/curves.png        loss, accuracy, macro-F1 per epoch (train vs validation)
plot_confusion  -> <run>/confusion_*.png   confusion matrix, colour = row share, text = count + share
plot_summary    -> summary/summary_*.png   mean +/- std per experiment, each seed as a dot

"Agg" is a non-interactive backend so plots can be saved on machines without a display (cluster nodes).
The colours are a colour-blind-checked palette; train is always blue and validation always orange.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from . import CLASS_NAMES  # noqa: E402

# Validated reference palette (dataviz skill): categorical slots in fixed order, blue sequential ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
BLUES = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)


def _style(ax):
    # Shared look: light background, no top/right box lines, faint horizontal grid behind the data.
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(history: list[dict], best_epoch: int, path, title: str):
    """Small multiples: loss, accuracy, macro-F1 - one y-axis each, train vs val."""
    ep = [h["epoch"] for h in history]
    panels = [
        ("Loss", "train_loss", "val_loss"),
        ("Accuracy", "train_acc", "val_acc"),
        ("Macro-F1", "train_macro_f1", "val_macro_f1"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), facecolor=SURFACE)
    for ax, (name, tr, va) in zip(axes, panels):
        _style(ax)
        for key, label, color in ((tr, "train", SERIES[0]), (va, "val", SERIES[1])):
            ys = [h[key] for h in history]
            ax.plot(ep, ys, color=color, linewidth=2, label=label)
            ax.annotate(label, (ep[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        color=INK2, fontsize=9, va="center")
        # Dashed line = the epoch whose checkpoint was kept (best validation macro-F1).
        ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle="--")
        ax.set_title(name, color=INK, fontsize=11, loc="left")
        ax.set_xlabel("epoch", color=INK2, fontsize=9)
    for ax in axes:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK2)
    axes[2].annotate(f"best epoch {best_epoch}", (best_epoch, axes[2].get_ylim()[0]), xytext=(4, 6),
                     textcoords="offset points", color=MUTED, fontsize=8)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(cm, path, title: str):
    """Row-normalised colour (recall per true class) with raw counts printed in each cell."""
    # Colour by the share of each TRUE class (row), so a small class's errors are as visible as a large
    # class's: 45 errors out of 537 meningiomas shows as 8%, not as a faint cell next to 972 gliomas.
    cm = np.asarray(cm)
    row = cm.sum(axis=1, keepdims=True)
    frac = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), facecolor=SURFACE)
    ax.imshow(frac, cmap=BLUES, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ink = "#ffffff" if frac[i, j] > 0.55 else INK
            ax.text(j, i, f"{cm[i, j]}\n{frac[i, j]:.0%}", ha="center", va="center", color=ink, fontsize=10)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, color=INK2, fontsize=9)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, color=INK2, fontsize=9)
    ax.set_xlabel("predicted", color=INK2)
    ax.set_ylabel("true", color=INK2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_summary(rows: list[dict], metric: str, path, title: str):
    """Mean +/- std across seeds per experiment, with each seed shown as a dot."""
    fig, ax = plt.subplots(figsize=(max(5.0, 1.6 + 1.3 * len(rows)), 3.8), facecolor=SURFACE)
    _style(ax)
    for k, r in enumerate(rows):
        color = SERIES[k % len(SERIES)]
        ax.bar(k, r["mean"], width=0.55, color=color, edgecolor=SURFACE, linewidth=2)
        ax.errorbar(k, r["mean"], yerr=r["std"], color=INK, capsize=4, linewidth=1)
        ax.scatter(np.full(len(r["values"]), k), r["values"], s=14, color=INK, zorder=3)
        ax.annotate(f"{r['mean']:.3f}", (k, r["mean"] + r["std"]), xytext=(0, 4), textcoords="offset points",
                    ha="center", color=INK, fontsize=9)
    ax.set_xticks(range(len(rows)), [f"{r['exp']}\n(n={len(r['values'])})" for r in rows], color=INK2)
    lo = min(min(r["values"]) for r in rows)
    ax.set_ylim(max(0.0, lo - 0.08), 1.0)
    ax.set_ylabel(metric, color=INK2)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
