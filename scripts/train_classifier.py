"""Train one classifier run (baseline B0/B1, or one cross-validation fold).

Model selection uses the official validation split only. The test split is never touched here;
use scripts/evaluate.py --split test once, for final numbers.

Each run writes a folder <out_root>/<run_name>/ containing:
    config.json         all settings, code version, GPU, cache fingerprint
    train.log           full training log
    history.csv         per-epoch loss / accuracy / macro-F1 (train and val)
    best.pt             checkpoint with the best validation macro-F1
    predictions_val.csv per-image predictions and class probabilities
    metrics.json        validation metrics at the best epoch (+ group-bootstrap 95% CIs)
    curves.png          training curves
    confusion_val.png   validation confusion matrix
    (fold runs also write predictions_oof.csv, confusion_oof.png and an "oof" block in metrics.json)
Everything except *.pt is copied to --results_root/<run_name>/ and a row is appended to
--results_root/registry.csv. best.pt is copied to --ckpt_store/<run_name>/ when given.
Run folders record local paths and job IDs, so results/ is git-ignored; share summaries deliberately.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from brainmri import CLASS_NAMES, SPLIT_VERSION  # noqa: E402
from brainmri.data import Cache, MRIDataset, make_transform  # noqa: E402
from brainmri.metrics import compute_metrics  # noqa: E402
from brainmri.models import build_model  # noqa: E402
from brainmri.plots import plot_confusion, plot_curves  # noqa: E402
from brainmri.utils import (append_registry, code_version, device_name, pick_device,  # noqa: E402
                            seed_everything, slurm_job_id)

CV_SEED = 42  # fixed fold assignment, independent of the training seed (same folds for everyone)

# Typical commands:
#   B0:   --exp B0 --aug none     --seed 0..4
#   B1:   --exp B1 --aug standard --seed 0..4
#   CV:   --exp CV --aug standard --seed 0 --fold 0..4
#   E1-3: --exp E1 --aug standard --seed 0..4 --synthetic <set>.npz


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--out_root", type=Path, required=True, help="where run folders (incl. checkpoints) go")
    p.add_argument("--results_root", type=Path, default=None, help="copy light artifacts + registry here")
    p.add_argument("--ckpt_store", type=Path, default=None, help="also copy best.pt here (durable backup)")
    p.add_argument("--exp", required=True, help="experiment id, e.g. B0, B1, CV")
    p.add_argument("--aug", choices=["none", "standard"], required=True)
    p.add_argument("--model", default="resnet18", choices=["resnet18", "resnet50", "efficientnet_b0"])
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fold", type=int, default=-1, help="-1 = normal run; k = hold out CV fold k of train")
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=10, help="stop after this many epochs without val improvement")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=1)
    p.add_argument("--class_weights", choices=["none", "balanced"], default="none")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--num_workers", type=int, default=-1, help="-1 = auto")
    p.add_argument("--min_gpu_gb", type=float, default=0.0, help="refuse a CUDA GPU smaller than this (0 = off)")
    p.add_argument("--n_boot", type=int, default=1000)
    p.add_argument("--synthetic", type=Path, nargs="*", default=[],
                   help="generated .npz set(s) to add to the TRAINING data (never to val/test)")
    p.add_argument("--tag", default="", help="optional extra text for the run name")
    p.add_argument("--smoke", action="store_true", help="tiny quick run to test the pipeline")
    return p.parse_args()


def setup_logging(run_dir: Path) -> logging.Logger:
    """Log to the terminal and to <run_dir>/train.log at the same time."""
    log = logging.getLogger("train")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(run_dir / "train.log"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


@torch.no_grad()  # no gradients needed for evaluation: faster and uses less memory
def predict(model, loader, device, criterion, amp_dtype):
    """Run the model over a loader; return mean loss, class probabilities, predictions, labels, cache indices.

    Also used by scripts/evaluate.py, so validation during training and the final test use identical code.
    """
    model.eval()  # evaluation mode: batch-norm uses its stored statistics, dropout is off
    losses, probs, labels, idxs = [], [], [], []
    for x, y_cpu, j in loader:
        # Labels are kept from the CPU batch: reading a non_blocking device copy back can return garbage.
        x, y = x.to(device, non_blocking=True), y_cpu.to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        losses.append(criterion(logits.float(), y).item() * len(y_cpu))
        probs.append(torch.softmax(logits.float(), 1).cpu())
        labels.append(y_cpu)
        idxs.append(j)
    probs = torch.cat(probs).numpy()
    labels = torch.cat(labels).numpy()
    return {
        "loss": sum(losses) / len(labels),
        "probs": probs,
        "pred": probs.argmax(1),
        "labels": labels,
        "idx": torch.cat(idxs).numpy(),
    }


def write_predictions(path: Path, res: dict, cache: Cache):
    """One CSV row per image: file, patient, true label, prediction, and the probability of each class.

    These files are what later analysis builds on (e.g. the CV file picks the hard images for Phase 2).
    The first check makes sure every prediction is written next to the right image.
    """
    if not np.array_equal(res["labels"], cache.labels[res["idx"]]):
        raise RuntimeError("Collected labels do not match the cache labels - predictions would be misattributed.")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "patient_id", "split_group", "label", "pred", "correct"]
                   + [f"prob_{c}" for c in CLASS_NAMES])
        for k, j in enumerate(res["idx"]):
            w.writerow([cache.paths[j], cache.patient_ids[j], cache.groups[j], int(res["labels"][k]),
                        int(res["pred"][k]), int(res["labels"][k] == res["pred"][k])]
                       + [f"{p:.5f}" for p in res["probs"][k]])


def main():
    a = parse_args()
    if a.smoke:  # a 2-epoch run on a small subset, just to check the pipeline works end to end
        a.epochs, a.patience, a.n_boot = 2, 2, 50
    # Run name = what the run is + when it started, e.g. B1_standard_resnet18_seed0_20261124-160310.
    # The timestamp makes every folder unique, so re-running never overwrites an earlier result.
    name = f"{a.exp}_{a.aug}_{a.model}_seed{a.seed}" + (f"_fold{a.fold}" if a.fold >= 0 else "")
    name += (f"_{a.tag}" if a.tag else "") + ("_smoke" if a.smoke else "")
    name += "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = a.out_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    log = setup_logging(run_dir)

    seed_everything(a.seed)
    device = pick_device(a.min_gpu_gb)
    # Mixed precision: on NVIDIA GPUs most maths runs in bfloat16, roughly 2x faster with no accuracy loss
    # for this task. Apple MPS / CPU run in full float32.
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    # DataLoader worker processes prepare (augment) the next batches while the GPU trains on the current one.
    # On Mac/CPU we use 0 workers (main process only), which avoids copying the cache into each worker.
    workers = a.num_workers if a.num_workers >= 0 else (8 if device.type == "cuda" else 0)

    # ---- data
    # Everything below works with integer indices into the cache (see brainmri/data.py).
    cache = Cache(a.cache)
    train_idx, val_idx, oof_idx = cache.indices("train"), cache.indices("val"), None
    if a.fold >= 0:
        # Cross-validation mode: split the TRAINING set into 5 folds by patient group and class.
        # This run trains on 4 folds and predicts the 5th ("out-of-fold", OOF). Doing all 5 folds gives an
        # honest prediction for every training image from a model that never saw it, which is how we find
        # the confused classes and hard images without touching validation or test.
        # random_state is fixed (CV_SEED) so every fold run, and every teammate, gets the same folds.
        sgkf = StratifiedGroupKFold(n_splits=a.n_folds, shuffle=True, random_state=CV_SEED)
        folds = list(sgkf.split(train_idx, cache.labels[train_idx], cache.groups[train_idx]))
        tr, ho = folds[a.fold]
        oof_idx, train_idx = train_idx[ho], train_idx[tr]
        # Leakage check: no patient may appear in both the training part and the held-out fold.
        assert not set(cache.groups[oof_idx]) & set(cache.groups[train_idx]), "group leak between CV folds"
    if a.smoke:  # tiny random subsets so the smoke test finishes in seconds
        rng = np.random.default_rng(0)
        train_idx = rng.choice(train_idx, 128, replace=False)
        val_idx = rng.choice(val_idx, 64, replace=False)
        if oof_idx is not None:
            oof_idx = rng.choice(oof_idx, min(64, len(oof_idx)), replace=False)
    n_real_train = len(train_idx)
    # Phase 2: add generated images to the TRAINING indices only. Validation is never changed, so B1 and
    # the E1/E2/E3 runs are scored on exactly the same 363 real validation images.
    syn_counts = {}
    if a.synthetic:
        if a.fold >= 0:
            raise ValueError("--synthetic with --fold would leak: synthetic sources can come from the held-out fold.")
        for sp in a.synthetic:
            sidx = cache.append_synthetic(sp)
            if a.smoke:
                sidx = sidx[:64]
            train_idx = np.concatenate([train_idx, sidx])
            syn_counts[sp.stem] = int(len(sidx))

    # Seeded generator: the order in which training images are shuffled depends only on --seed.
    g = torch.Generator().manual_seed(a.seed)
    loader_kw = dict(batch_size=a.batch_size, num_workers=workers, pin_memory=device.type == "cuda",
                     persistent_workers=workers > 0)
    train_dl = DataLoader(MRIDataset(cache, train_idx, make_transform(a.aug, a.img_size)),
                          shuffle=True, generator=g, drop_last=False, **loader_kw)
    eval_tf = make_transform("none", a.img_size)  # evaluation never uses random augmentation
    val_dl = DataLoader(MRIDataset(cache, val_idx, eval_tf), shuffle=False, **loader_kw)

    counts = np.bincount(cache.labels[train_idx], minlength=len(CLASS_NAMES))
    weights = None
    if a.class_weights == "balanced":
        # Optional: weight the loss so each class counts equally (rarer classes get larger weights).
        # weight_c = total / (n_classes * count_c). Off by default; B0/B1 used no weights.
        weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32, device=device)

    # config.json: every setting of this run plus where/how it ran, so any result can be reproduced or
    # explained later. Written now (status "running") and again at the end (status "done").
    config = {
        "run_name": name,
        **{k: (str(v) if isinstance(v, Path) else [str(x) for x in v] if isinstance(v, list) else v)
           for k, v in vars(a).items()},
        "status": "running", "started": datetime.now().isoformat(timespec="seconds"),
        "device": device_name(device), "torch": torch.__version__, "code_version": code_version(),
        "slurm_job_id": slurm_job_id(), "split_version": SPLIT_VERSION, "cache_meta": cache.meta,
        "n_train": int(len(train_idx)), "n_real_train": int(n_real_train), "synthetic_sets": syn_counts,
        "n_val": int(len(val_idx)),
        "n_oof": int(len(oof_idx)) if oof_idx is not None else 0,
        "train_class_counts": dict(zip(CLASS_NAMES, counts.tolist())),
        "class_weight_values": weights.tolist() if weights is not None else None,
        "cv_seed": CV_SEED if a.fold >= 0 else None, "num_workers": workers,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    log.info(f"Run {name} on {config['device']} | train {len(train_idx)} val {len(val_idx)}"
             + (f" oof {len(oof_idx)}" if oof_idx is not None else "") + f" | counts {config['train_class_counts']}")

    # ---- model / optimiser
    model = build_model(a.model, pretrained=not a.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)  # standard loss for single-label classification
    # AdamW: Adam with decoupled weight decay; all layers are fine-tuned with the same learning rate.
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    # Learning-rate schedule as a multiplier of --lr per epoch:
    #   warm-up: start small so the new random output layer does not wreck the pretrained weights,
    #   then cosine decay: smoothly lower the rate to ~0 by the last epoch for a stable final model.
    def lr_factor(epoch):  # linear warmup, then cosine decay (stepped per epoch)
        if epoch < a.warmup_epochs:
            return (epoch + 1) / (a.warmup_epochs + 1)
        t = (epoch - a.warmup_epochs) / max(1, a.epochs - a.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)

    # ---- train
    # best = (best val macro-F1, its val loss); stale = epochs since the last improvement (for early stop)
    history, best, best_epoch, stale = [], (-1.0, float("inf")), 0, 0
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()  # training mode: dropout/batch-norm behave as in training
        te = time.time()
        tot_loss, preds, labels = 0.0, [], []
        for x, y_cpu, _ in train_dl:
            x, y = x.to(device, non_blocking=True), y_cpu.to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
                loss = criterion(logits.float(), y)
            # Standard update: clear old gradients, backpropagate this batch's loss, take one optimiser step.
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot_loss += loss.item() * len(y)
            preds.append(logits.argmax(1).cpu())
            labels.append(y_cpu)
        sched.step()  # next epoch's learning rate
        # Train metrics are measured on augmented images while the weights change, so they are only a rough
        # guide; the validation metrics below are the ones used for decisions.
        preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
        v = predict(model, val_dl, device, criterion, amp_dtype)
        row = {
            "epoch": epoch, "lr": opt.param_groups[0]["lr"],
            "train_loss": tot_loss / len(labels), "train_acc": float((preds == labels).mean()),
            "train_macro_f1": float(f1_score(labels, preds, average="macro")),
            "val_loss": v["loss"], "val_acc": float((v["pred"] == v["labels"]).mean()),
            "val_macro_f1": float(f1_score(v["labels"], v["pred"], average="macro")),
            "seconds": round(time.time() - te, 1),
        }
        history.append(row)
        # Model selection: keep the epoch with the highest validation macro-F1; on a tie, the lower val loss.
        # (Tuple comparison checks the first element, then the second.)
        improved = (row["val_macro_f1"], -row["val_loss"]) > (best[0], -best[1])
        if improved:
            best, best_epoch, stale = (row["val_macro_f1"], row["val_loss"]), epoch, 0
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch,
                        "val_macro_f1": row["val_macro_f1"]}, run_dir / "best.pt")
        else:
            stale += 1
        log.info(f"ep {epoch:3d} | loss {row['train_loss']:.4f}/{row['val_loss']:.4f} "
                 f"| acc {row['train_acc']:.3f}/{row['val_acc']:.3f} | val F1 {row['val_macro_f1']:.3f}"
                 f"{' *' if improved else ''} | {row['seconds']}s")
        # history.csv is rewritten every epoch, so progress is visible (and kept) even if the job dies.
        with open(run_dir / "history.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
            w.writerows(history)
        if stale >= a.patience:  # early stopping: further epochs are unlikely to help
            log.info(f"Early stop: no val improvement for {a.patience} epochs")
            break

    # ---- final evaluation with the best checkpoint
    # The model in memory is from the LAST epoch; reload the saved best epoch before writing results.
    model.load_state_dict(torch.load(run_dir / "best.pt", map_location=device, weights_only=False)["model"])
    v = predict(model, val_dl, device, criterion, amp_dtype)
    # Self-check: the reloaded checkpoint should score what was logged for that epoch. A large gap would mean
    # a saving/loading bug or strong GPU nondeterminism, and the run's numbers should not be trusted.
    logged = history[best_epoch - 1]["val_macro_f1"]
    reloaded = float(f1_score(v["labels"], v["pred"], average="macro"))
    if abs(reloaded - logged) > 0.01:
        log.info(f"WARNING: reloaded best checkpoint gives val macro-F1 {reloaded:.4f}, "
                 f"but {logged:.4f} was logged at epoch {best_epoch}. Check for nondeterminism.")
    write_predictions(run_dir / "predictions_val.csv", v, cache)
    metrics = {"run_name": name, "best_epoch": best_epoch,
               "val": compute_metrics(v["labels"], v["pred"], cache.groups[v["idx"]], a.n_boot, a.seed)}
    plot_confusion(metrics["val"]["confusion_matrix"], run_dir / "confusion_val.png",
                   f"{a.exp} seed {a.seed} - validation (n={len(v['labels'])})")
    if oof_idx is not None:
        # CV mode: also predict the held-out fold. summarize_runs.py pools the 5 folds' files into one
        # out-of-fold prediction for every training image.
        oof_dl = DataLoader(MRIDataset(cache, oof_idx, eval_tf), shuffle=False, **loader_kw)
        o = predict(model, oof_dl, device, criterion, amp_dtype)
        write_predictions(run_dir / "predictions_oof.csv", o, cache)
        metrics["oof"] = compute_metrics(o["labels"], o["pred"], cache.groups[o["idx"]], a.n_boot, a.seed)
        plot_confusion(metrics["oof"]["confusion_matrix"], run_dir / "confusion_oof.png",
                       f"{a.exp} fold {a.fold} - held-out train fold (n={len(o['labels'])})")
    plot_curves(history, best_epoch, run_dir / "curves.png", f"{name}")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    config.update(status="done", finished=datetime.now().isoformat(timespec="seconds"),
                  minutes=round((time.time() - t0) / 60, 1), best_epoch=best_epoch, epochs_run=len(history))
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    m = metrics["val"]
    log.info(f"Best epoch {best_epoch} | val acc {m['accuracy']:.4f} macro-F1 {m['macro_f1']:.4f} "
             f"| recall " + " ".join(f"{c}={m['per_class'][c]['recall']:.3f}" for c in CLASS_NAMES))
    log.info(f"Run folder: {run_dir}")

    # ---- durable copies + registry
    # out_root may be fast temporary storage; these copies go somewhere safer:
    #   --ckpt_store    a second copy of best.pt
    #   --results_root  everything except checkpoints (small: logs, metrics, predictions, plots)
    #                   plus one summary line in registry.csv
    if a.ckpt_store:
        dst = a.ckpt_store / name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(run_dir / "best.pt", dst / "best.pt")
    if a.results_root:
        shutil.copytree(run_dir, a.results_root / name, ignore=shutil.ignore_patterns("*.pt"))
        reg = {
            "run_name": name, "exp": a.exp, "aug": a.aug, "model": a.model, "seed": a.seed, "fold": a.fold,
            "class_weights": a.class_weights, "smoke": int(a.smoke),
            "synthetic": "+".join(syn_counts) or "", "n_synthetic": sum(syn_counts.values()), "best_epoch": best_epoch,
            "epochs_run": len(history), "val_acc": round(m["accuracy"], 4), "val_macro_f1": round(m["macro_f1"], 4),
            **{f"val_recall_{c}": round(m["per_class"][c]["recall"], 4) for c in CLASS_NAMES},
            "oof_macro_f1": round(metrics["oof"]["macro_f1"], 4) if "oof" in metrics else "",
            "minutes": config["minutes"], "device": config["device"], "code_version": config["code_version"],
            "slurm_job_id": config["slurm_job_id"], "finished": config["finished"],
        }
        append_registry(a.results_root / "registry.csv", reg)
        log.info(f"Copied results to {a.results_root / name} and appended registry.csv")


if __name__ == "__main__":
    main()
